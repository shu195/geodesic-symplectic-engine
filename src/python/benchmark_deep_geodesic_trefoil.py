"""
The Ultimate Topological Stress Test: 3D Trefoil Knot Manifold
Comparing:
1. Deep 4-Layer Geodesic Spacetime Network (Curvature Flow -> Diffeomorphism -> Curvature Flow -> Singularity Core)
2. Standard Deep MLP (4 layers, matched and scaled)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Dataset Generation: 3D Trefoil Knot Manifold (Braid Group B_3)
# =====================================================================
print("Generating 3D Trefoil Knot Dataset (N = 3,000 points)...")
N = 3000
pts_x = np.random.uniform(-4.0, 4.0, N)
pts_y = np.random.uniform(-4.0, 4.0, N)
pts_z = np.random.uniform(-2.5, 2.5, N)
X_np = np.column_stack([pts_x, pts_y, pts_z])

# Compute distance from each point to the continuous Trefoil curve
# Discretize trefoil curve with 300 points
t_fine = np.linspace(0, 2 * np.pi, 300, endpoint=False)
kx = np.sin(t_fine) + 2.0 * np.sin(2.0 * t_fine)
ky = np.cos(t_fine) - 2.0 * np.cos(2.0 * t_fine)
kz = -np.sin(3.0 * t_fine)
knot_curve = np.column_stack([kx, ky, kz]) # [300, 3]

# Compute Euclidean distance from each sample point to closest knot segment
diffs = X_np[:, None, :] - knot_curve[None, :, :] # [N, 300, 3]
min_dist = np.min(np.sqrt(np.sum(diffs**2, axis=-1)), axis=-1)

r_tube = 0.55
y_np = (min_dist <= r_tube).astype(np.float32)

X = torch.tensor(X_np, dtype=torch.float32)
Y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)

print(f"Dataset ready. Knot positive ratio: {y_np.mean()*100:.2f}% ({int(y_np.sum())}/{N} points)")

# =====================================================================
# 2. Deep 4-Layer Geodesic Spacetime Network
# =====================================================================
class GeodesicFlowBlock(nn.Module):
    """A continuous geodesic warping layer: x' = x - eta * grad_x(Phi(x))"""
    def __init__(self, num_masses=4):
        super().__init__()
        self.mass_centers = nn.Parameter(torch.randn(num_masses, 3) * 1.5)
        self.mass_scales = nn.Parameter(torch.ones(num_masses) * 0.8)
        self.mass_weights = nn.Parameter(torch.randn(num_masses))
        self.step_size = nn.Parameter(torch.tensor(0.25))
        
    def forward(self, x):
        diff = x.unsqueeze(1) - self.mass_centers.unsqueeze(0) # [B, K, 3]
        dist_sq = torch.sum(diff**2, dim=-1) # [B, K]
        t = torch.tanh(dist_sq * self.mass_scales.unsqueeze(0))
        sech2 = 1.0 - t**2
        coeffs = (2.0 * self.mass_scales.unsqueeze(0) * self.mass_weights.unsqueeze(0) * sech2).unsqueeze(-1)
        grad = torch.sum(coeffs * diff, dim=1)
        return x - self.step_size * grad

class DeepGeodesicSpacetimeNet(nn.Module):
    """
    4-Layer Deep Geodesic Spacetime Network:
    Layer 1: Geodesic Flow 1 (Non-linear gravitational bending)
    Layer 2: Diffeomorphism Layer (Affine coordinate untangling)
    Layer 3: Geodesic Flow 2 (Secondary curvature focusing)
    Layer 4: Topological Energy Detection Head
    """
    def __init__(self):
        super().__init__()
        # Layer 1: Curvature Flow 1
        self.flow1 = GeodesicFlowBlock(num_masses=5)
        # Layer 2: Diffeomorphism
        self.diffeo = nn.Linear(3, 3)
        self.diffeo.weight.data = torch.eye(3) + torch.randn(3, 3) * 0.05
        # Layer 3: Curvature Flow 2
        self.flow2 = GeodesicFlowBlock(num_masses=5)
        # Layer 4: Topological Attractor Head (Radial quadric distance to core)
        self.attractor_center = nn.Parameter(torch.zeros(3))
        self.gain = nn.Parameter(torch.tensor(2.0))
        self.threshold = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        # Layer 1: Geodesic flow
        x = self.flow1(x)
        # Layer 2: Diffeomorphism warp
        x = self.diffeo(x)
        # Layer 3: Geodesic flow
        x = self.flow2(x)
        # Layer 4: Energy detection
        dist_sq = torch.sum((x - self.attractor_center.unsqueeze(0))**2, dim=-1, keepdim=True)
        return torch.sigmoid(self.gain * (self.threshold - dist_sq))

# =====================================================================
# 3. Standard Deep MLP Baseline (Matched Depth)
# =====================================================================
class DeepMLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# =====================================================================
# 4. Train & Benchmark
# =====================================================================
crit = nn.BCELoss()

def train_model(model, name, lr=0.05, epochs=300):
    params = sum(p.numel() for p in model.parameters())
    opt = optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        preds = model(X)
        loss = crit(preds, Y)
        loss.backward()
        opt.step()
    elapsed = (time.time() - t0) * 1000.0
    
    with torch.no_grad():
        p = (model(X) > 0.5).float()
        acc = (p == Y).float().mean().item() * 100.0
        pos_mask = (Y == 1.0)
        pos_recall = (p[pos_mask] == 1.0).float().mean().item() * 100.0 if pos_mask.sum() > 0 else 0.0
        neg_mask = (Y == 0.0)
        neg_recall = (p[neg_mask] == 0.0).float().mean().item() * 100.0
    return {
        'name': name,
        'params': params,
        'acc': acc,
        'recall': pos_recall,
        'neg_recall': neg_recall,
        'loss': loss.item(),
        'time_ms': elapsed
    }

print("=" * 80)
print("BENCHMARK: 3D TREFOIL KNOT MANIFOLD (DEEP 4-LAYER GEODESIC VS DEEP MLP)")
print("=" * 80)

geo_model = DeepGeodesicSpacetimeNet()
mlp_model = DeepMLP(hidden=32)

r_geo = train_model(geo_model, "Deep Geodesic Spacetime Net (4 Layers)", lr=0.05, epochs=300)
r_mlp = train_model(mlp_model, "Standard Deep MLP (4 Layers, 32 units)", lr=0.03, epochs=300)

print(f"{'Model Architecture':<40} | {'Params':>6} | {'Accuracy':>8} | {'Knot Recall':>11} | {'Loss':>7} | {'Time (ms)':>9}")
print("-" * 88)
print(f"{r_geo['name']:<40} | {r_geo['params']:>6d} | {r_geo['acc']:>7.2f}% | {r_geo['recall']:>10.2f}% | {r_geo['loss']:>7.4f} | {r_geo['time_ms']:>8.1f}ms")
print(f"{r_mlp['name']:<40} | {r_mlp['params']:>6d} | {r_mlp['acc']:>7.2f}% | {r_mlp['recall']:>10.2f}% | {r_mlp['loss']:>7.4f} | {r_mlp['time_ms']:>8.1f}ms")
