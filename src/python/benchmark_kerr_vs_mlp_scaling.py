"""
Scaling Benchmark: Kerr Singularity Geodesic Neuron vs. Scaled MLPs
Tests:
1. Scaling Dataset to N=5,000 with Noise: Precision, Recall, and Parameter Efficiency
2. Out-of-Distribution (OOD) Extrapolation: Train on 50% sector [0, pi], Test on unseen 50% [pi, 2*pi]
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Dataset Generation: Large Scale 3D Torus
# =====================================================================
N_total = 5000
R_major, r_tube = 2.0, 0.55

pts_x = np.random.uniform(-4.0, 4.0, N_total)
pts_y = np.random.uniform(-4.0, 4.0, N_total)
pts_z = np.random.uniform(-2.0, 2.0, N_total)

# Add small sensor measurement noise
pts_x += np.random.randn(N_total) * 0.03
pts_y += np.random.randn(N_total) * 0.03
pts_z += np.random.randn(N_total) * 0.03

r_xy = np.sqrt(pts_x**2 + pts_y**2)
d_torus = np.sqrt((r_xy - R_major)**2 + pts_z**2)
y_all = (d_torus <= r_tube).astype(np.float32)

# Compute azimuthal angle theta in xy-plane: [-pi, pi]
angles = np.arctan2(pts_y, pts_x)

# Split 1: Standard Random Train/Test Split (80/20)
indices = np.random.permutation(N_total)
train_idx, test_idx = indices[:4000], indices[4000:]

X_train = torch.tensor(np.column_stack([pts_x[train_idx], pts_y[train_idx], pts_z[train_idx]]), dtype=torch.float32)
Y_train = torch.tensor(y_all[train_idx], dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(np.column_stack([pts_x[test_idx], pts_y[test_idx], pts_z[test_idx]]), dtype=torch.float32)
Y_test = torch.tensor(y_all[test_idx], dtype=torch.float32).unsqueeze(-1)

# Split 2: Out-of-Distribution (OOD) Split
# Train strictly on theta in [0, pi] (half the donut)
# Test strictly on unseen theta in [-pi, 0)
ood_train_mask = (angles >= 0)
ood_test_mask = (angles < 0)

X_ood_train = torch.tensor(np.column_stack([pts_x[ood_train_mask], pts_y[ood_train_mask], pts_z[ood_train_mask]]), dtype=torch.float32)
Y_ood_train = torch.tensor(y_all[ood_train_mask], dtype=torch.float32).unsqueeze(-1)
X_ood_test = torch.tensor(np.column_stack([pts_x[ood_test_mask], pts_y[ood_test_mask], pts_z[ood_test_mask]]), dtype=torch.float32)
Y_ood_test = torch.tensor(y_all[ood_test_mask], dtype=torch.float32).unsqueeze(-1)


# =====================================================================
# 2. Models
# =====================================================================
class KerrRingNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.center = nn.Parameter(torch.tensor([0.1, -0.1, 0.05]))
        self.normal_raw = nn.Parameter(torch.tensor([0.05, 0.05, 1.0]))
        self.R = nn.Parameter(torch.tensor([1.5]))
        self.alpha = nn.Parameter(torch.tensor([3.0]))
        self.threshold = nn.Parameter(torch.tensor([0.3]))
        
    def forward(self, x):
        normal = self.normal_raw / (torch.norm(self.normal_raw) + 1e-6)
        diff = x - self.center.unsqueeze(0)
        h = torch.sum(diff * normal.unsqueeze(0), dim=-1, keepdim=True)
        r_perp = diff - h * normal.unsqueeze(0)
        rho = torch.norm(r_perp, dim=-1, keepdim=True)
        d_ring_sq = (rho - torch.abs(self.R))**2 + h**2
        return torch.sigmoid(self.alpha * (self.threshold - d_ring_sq))

def make_mlp(hidden_dims):
    layers = []
    in_dim = 3
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


# =====================================================================
# 3. Training & Evaluation Pipeline
# =====================================================================
crit = nn.BCELoss()

def run_model(model, name, X_tr, Y_tr, X_te, Y_te, lr=0.08, epochs=250):
    params = sum(p.numel() for p in model.parameters())
    opt = optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X_tr)
        loss = crit(out, Y_tr)
        loss.backward()
        opt.step()
    elapsed = (time.time() - t0) * 1000.0 # ms
    
    with torch.no_grad():
        preds = (model(X_te) > 0.5).float()
        acc = (preds == Y_te).float().mean().item() * 100.0
        pos_mask = (Y_te == 1.0)
        pos_recall = (preds[pos_mask] == 1.0).float().mean().item() * 100.0 if pos_mask.sum() > 0 else 0.0
        neg_mask = (Y_te == 0.0)
        neg_recall = (preds[neg_mask] == 0.0).float().mean().item() * 100.0 if neg_mask.sum() > 0 else 0.0
    return {
        'name': name,
        'params': params,
        'acc': acc,
        'pos_recall': pos_recall,
        'neg_recall': neg_recall,
        'loss': loss.item(),
        'time_ms': elapsed
    }

# =====================================================================
# 4. Execute Scaling Benchmark
# =====================================================================
print("=" * 80)
print(f"BENCHMARK 1: SCALED DATASET (N = 5,000 SAMPLES WITH NOISE) [RANDOM 80/20 SPLIT]")
print("=" * 80)

models_scaled = [
    (KerrRingNeuron(), "Kerr Ring Singularity Neuron", 0.08, 200),
    (make_mlp([8]), "MLP-Tiny (1 hidden, 8 units)", 0.05, 250),
    (make_mlp([8, 8]), "MLP-Small (2 hidden, 8-8 units)", 0.05, 250),
    (make_mlp([16, 16]), "MLP-Medium (2 hidden, 16-16 units)", 0.04, 250),
    (make_mlp([32, 32]), "MLP-Large (2 hidden, 32-32 units)", 0.03, 250),
    (make_mlp([64, 64, 32]), "MLP-Deep (3 hidden, 64-64-32 units)", 0.02, 300),
]

results_1 = []
print(f"{'Model Architecture':<38} | {'Params':>6} | {'Test Acc':>8} | {'Torus Recall':>12} | {'Time (ms)':>9}")
print("-" * 82)
for m, name, lr, ep in models_scaled:
    r = run_model(m, name, X_train, Y_train, X_test, Y_test, lr=lr, epochs=ep)
    results_1.append(r)
    print(f"{r['name']:<38} | {r['params']:>6d} | {r['acc']:>7.2f}% | {r['pos_recall']:>11.2f}% | {r['time_ms']:>8.1f}ms")


print("\n" + "=" * 80)
print("BENCHMARK 2: OUT-OF-DISTRIBUTION (OOD) EXTRAPOLATION TEST")
print("Train strictly on theta in [0, pi] (half the donut) -> Test on unseen [pi, 2*pi]!")
print("=" * 80)

models_ood = [
    (KerrRingNeuron(), "Kerr Ring Singularity Neuron", 0.08, 200),
    (make_mlp([16, 16]), "MLP-Medium (16-16 units)", 0.04, 250),
    (make_mlp([32, 32]), "MLP-Large (32-32 units)", 0.03, 250),
    (make_mlp([64, 64, 32]), "MLP-Deep (64-64-32 units)", 0.02, 300),
]

print(f"{'Model Architecture':<38} | {'Params':>6} | {'OOD Acc':>8} | {'OOD Recall':>12} | {'OOD Rejection':>13}")
print("-" * 85)
for m, name, lr, ep in models_ood:
    r = run_model(m, name, X_ood_train, Y_ood_train, X_ood_test, Y_ood_test, lr=lr, epochs=ep)
    print(f"{r['name']:<38} | {r['params']:>6d} | {r['acc']:>7.2f}% | {r['pos_recall']:>11.2f}% | {r['neg_recall']:>12.2f}%")
