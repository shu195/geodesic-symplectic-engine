"""
Benchmark: Hysteretic Geodesic Spacetime Network
Combining:
1. 3D Geodesic Metric Flow (Curvature in Space)
2. Hysteretic Self-Loop Latch (w_self > 1.0 Non-linear Memory)

Task: 3D Spatiotemporal Vortex Tracking (Clockwise vs. Counter-Clockwise 3D Spirals)
- Single static points are identical in 3D space (occupy the same torus/cylinder).
- Classification requires fusing 3D Spatial Curvature with Temporal Direction Memory.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Dataset Generation: 3D Spatiotemporal Vortices
# =====================================================================
N_sequences = 1200
T_steps = 15
radius = 2.0
pitch = 0.15 # z-drift per step

# Generate vortex trajectories:
# Class 0: Clockwise spiral in (x, y) ascending in z
# Class 1: Counter-clockwise spiral in (x, y) ascending in z
X_list, Y_list = [], []

for i in range(N_sequences):
    direction = 1 if i % 2 == 1 else -1 # 1: CCW, -1: CW
    phase_offset = np.random.uniform(0, 2 * np.pi)
    z_start = np.random.uniform(-1.5, 0.0)
    noise_jitter = np.random.randn(T_steps, 3) * 0.04
    
    t = np.arange(T_steps) * 0.35 * direction + phase_offset
    xs = radius * np.cos(t) + noise_jitter[:, 0]
    ys = radius * np.sin(t) + noise_jitter[:, 1]
    zs = z_start + np.arange(T_steps) * pitch + noise_jitter[:, 2]
    
    traj = np.column_stack([xs, ys, zs]) # [T_steps, 3]
    X_list.append(traj)
    Y_list.append(1.0 if direction == 1 else 0.0)

X_all = torch.tensor(np.array(X_list), dtype=torch.float32) # [N, T_steps, 3]
Y_all = torch.tensor(np.array(Y_list), dtype=torch.float32).unsqueeze(-1) # [N, 1]

# Split 80/20
X_train, X_test = X_all[:960], X_all[960:]
Y_train, Y_test = Y_all[:960], Y_all[960:]

print(f"Dataset generated: {N_sequences} trajectories of length T={T_steps}.")
print(f"At any single step t, spatial distribution is identical for CW and CCW.")

# =====================================================================
# 2. Models
# =====================================================================

# Model 1: Static Geodesic Net (No temporal memory, evaluates final step)
class StaticGeodesicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.masses = nn.Parameter(torch.randn(6, 3))
        self.scales = nn.Parameter(torch.ones(6) * 0.5)
        self.weights = nn.Parameter(torch.randn(6))
        self.head = nn.Linear(3, 1)
        
    def forward(self, x_seq):
        # Only looks at static end-point x_seq[:, -1, :]
        x = x_seq[:, -1, :]
        diff = x.unsqueeze(1) - self.masses.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=-1)
        pot = torch.tanh(dist_sq * self.scales.unsqueeze(0))
        h = torch.sum(pot * self.weights.unsqueeze(0), dim=-1, keepdim=True)
        return torch.sigmoid(self.head(x) + h)

# Model 2: Standard Recurrent Unit (Vanilla RNN on 3D points)
class StandardRNN(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        
    def forward(self, x_seq):
        out, _ = self.rnn(x_seq)
        return torch.sigmoid(self.head(out[:, -1, :]))

# Model 3: Hysteretic Geodesic Spacetime Net
# Fuses 3D Geodesic Metric Flow with Non-linear Self-Loop Hysteresis (w_self > 1.0)
class HystereticGeodesicSpacetimeNet(nn.Module):
    def __init__(self, num_nodes=4):
        super().__init__()
        # Mass centers in 3D space
        self.centers = nn.Parameter(torch.randn(num_nodes, 3) * 1.5)
        # Hysteretic Self-Loop weights (initialized in bistable regime w_self > 1.0)
        self.w_self = nn.Parameter(torch.ones(num_nodes) * 1.8)
        # Spatial-to-temporal coupling weights
        self.w_cross = nn.Parameter(torch.randn(num_nodes) * 0.5)
        self.scales = nn.Parameter(torch.ones(num_nodes) * 0.8)
        self.head = nn.Linear(num_nodes, 1)
        
    def forward(self, x_seq):
        B, T, _ = x_seq.shape
        # Internal hysteretic memory state for each spatial node
        h = torch.zeros(B, len(self.centers))
        
        for t in range(T):
            x_t = x_seq[:, t, :] # [B, 3]
            # 1. 3D Geodesic metric potential to each node
            diff = x_t.unsqueeze(1) - self.centers.unsqueeze(0) # [B, K, 3]
            dist_sq = torch.sum(diff**2, dim=-1) # [B, K]
            # Spatial metric excitation
            spatial_flux = torch.tanh(dist_sq * self.scales.unsqueeze(0))
            
            # 2. Hysteretic Self-Loop update:
            # h_{t+1} = tanh( w_self * h_t + w_cross * spatial_flux )
            h = torch.tanh(self.w_self.unsqueeze(0) * h + self.w_cross.unsqueeze(0) * spatial_flux)
            
        return torch.sigmoid(self.head(h))

# =====================================================================
# 3. Train & Benchmark
# =====================================================================
crit = nn.BCELoss()

def train_eval(model, name, lr=0.05, epochs=200):
    params = sum(p.numel() for p in model.parameters())
    opt = optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X_train)
        loss = crit(out, Y_train)
        loss.backward()
        opt.step()
    elapsed = (time.time() - t0) * 1000.0
    
    with torch.no_grad():
        preds = (model(X_test) > 0.5).float()
        acc = (preds == Y_test).float().mean().item() * 100.0
    return {
        'name': name,
        'params': params,
        'acc': acc,
        'loss': loss.item(),
        'time_ms': elapsed
    }

print("=" * 85)
print("BENCHMARK: 3D SPATIOTEMPORAL VORTEX TRACKING (CW VS CCW SPIRALS)")
print("=" * 85)

m_static = StaticGeodesicNet()
m_rnn = StandardRNN(hidden=16)
m_hyst_geo = HystereticGeodesicSpacetimeNet(num_nodes=4)

r_static = train_eval(m_static, "Static Geodesic Net (No Memory)", lr=0.05, epochs=200)
r_rnn = train_eval(m_rnn, "Standard Recurrent Net (Vanilla RNN)", lr=0.03, epochs=200)
r_hyst_geo = train_eval(m_hyst_geo, "Hysteretic Geodesic Net (Ours)", lr=0.06, epochs=200)

print(f"{'Model Architecture':<38} | {'Params':>6} | {'Test Acc':>8} | {'Loss':>7} | {'Time (ms)':>9}")
print("-" * 85)
for r in [r_static, r_rnn, r_hyst_geo]:
    print(f"{r['name']:<38} | {r['params']:>6d} | {r['acc']:>7.2f}% | {r['loss']:>7.4f} | {r['time_ms']:>8.1f}ms")

# Check learned w_self values
print("\nLearned Internal Self-Loop Parameters (w_self):")
print(f"w_self: {m_hyst_geo.w_self.detach().numpy().round(3)}")
print(f"Are all nodes in bistable hysteretic regime (w_self > 1.0)? -> {bool(np.all(m_hyst_geo.w_self.detach().numpy() > 1.0))}")
