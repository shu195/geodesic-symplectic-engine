"""
Head-to-Head Benchmark: Standard PINN (Physics-Informed Neural Net) vs. Physical Phase-Space Neuron
Problem: Non-Linear Duffing Oscillator
Equation: d^2x/dt^2 + gamma * dx/dt - alpha * x + beta * x^3 = 0
Data: Sparse, noisy observations on t in [0, 5.0] s (50 collocation points)
Goal:
1. Discover physical governing parameters (gamma, alpha, beta)
2. Fit the training curve (t in [0, 5.0])
3. Extrapolate into unseen future (t in [5.0, 10.0] s)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Ground Truth Simulation of Duffing System
# =====================================================================
gamma_true = 0.20
alpha_true = 1.00 # linear negative stiffness (double well)
beta_true = 0.50  # cubic restoring force

dt = 0.02
t_full = np.arange(0, 10.0, dt)
x_sim = np.zeros(len(t_full))
v_sim = np.zeros(len(t_full))
x_sim[0] = 0.8
v_sim[0] = 0.0

# Symplectic simulation
for i in range(len(t_full) - 1):
    accel = -gamma_true * v_sim[i] + alpha_true * x_sim[i] - beta_true * (x_sim[i]**3)
    v_sim[i+1] = v_sim[i] + dt * accel
    x_sim[i+1] = x_sim[i] + dt * v_sim[i+1]

# Split: Train horizon [0, 5.0] s (250 steps), Test horizon [5.0, 10.0] s (250 steps)
N_train_steps = int(5.0 / dt)
t_train_np = t_full[:N_train_steps]
x_train_np = x_sim[:N_train_steps]
t_test_np = t_full[N_train_steps:]
x_test_np = x_sim[N_train_steps:]

# Subsample 50 sparse noisy observations for training
obs_indices = np.linspace(0, N_train_steps - 1, 50, dtype=int)
t_obs = torch.tensor(t_train_np[obs_indices], dtype=torch.float32).unsqueeze(-1).requires_grad_(True)
x_obs = torch.tensor(x_train_np[obs_indices] + np.random.randn(50) * 0.02, dtype=torch.float32).unsqueeze(-1)

# Dense collocation points for PINN physics residual evaluation on [0, 5.0]
t_colloc = torch.tensor(t_train_np, dtype=torch.float32).unsqueeze(-1).requires_grad_(True)

# Full tensors for evaluation
t_test_t = torch.tensor(t_test_np, dtype=torch.float32).unsqueeze(-1)
x_test_t = torch.tensor(x_test_np, dtype=torch.float32).unsqueeze(-1)

print("=" * 85)
print(f"BENCHMARK: STANDARD PINN VS. PHYSICAL PHASE-SPACE NEURON (DUFFING SYSTEM)")
print(f"Ground Truth Parameters: gamma={gamma_true}, alpha={alpha_true}, beta={beta_true}")
print(f"Training on 50 sparse noisy points (t <= 5.0s), Testing extrapolation on t in [5.0, 10.0]s")
print("=" * 85)

# =====================================================================
# 2. Standard Physics-Informed Neural Network (PINN)
# =====================================================================
class StandardPINN(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        # Learnable physical parameters
        self.gamma = nn.Parameter(torch.tensor([0.5]))
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.beta = nn.Parameter(torch.tensor([0.1]))
        
    def forward(self, t):
        return self.net(t)
        
    def compute_residual(self, t):
        # 1st order derivative dx/dt via autograd
        x = self.net(t)
        dx_dt = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        # 2nd order derivative d^2x/dt^2 via autograd
        d2x_dt2 = torch.autograd.grad(dx_dt, t, grad_outputs=torch.ones_like(dx_dt), create_graph=True)[0]
        # Physics residual: d^2x/dt^2 + gamma * dx/dt - alpha * x + beta * x^3
        residual = d2x_dt2 + self.gamma * dx_dt - self.alpha * x + self.beta * (x**3)
        return x, residual

# =====================================================================
# 3. Physical Phase-Space Neuron (Ours)
# =====================================================================
class PhysicalPhaseSpaceNeuron(nn.Module):
    """
    Directly parameterizes the symplectic vector field:
    dx/dt = v
    dv/dt = -gamma * v + alpha * x - beta * x^3
    """
    def __init__(self, dt_val=0.02):
        super().__init__()
        self.dt = dt_val
        self.gamma = nn.Parameter(torch.tensor([0.5]))
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.beta = nn.Parameter(torch.tensor([0.1]))
        self.x0 = nn.Parameter(torch.tensor([0.7]))
        self.v0 = nn.Parameter(torch.tensor([0.0]))
        
    def forward(self, steps):
        x_traj = [self.x0]
        v = self.v0
        x = self.x0
        for _ in range(steps - 1):
            accel = -self.gamma * v + self.alpha * x - self.beta * (x**3)
            v = v + self.dt * accel
            x = x + self.dt * v
            x_traj.append(x)
        return torch.stack(x_traj, dim=0)

# =====================================================================
# 4. Train & Benchmark Both
# =====================================================================

# 1. Train PINN
print("\n--- Training Standard PINN (3 hidden layers, ~2,200 params + autograd) ---")
pinn = StandardPINN(hidden=32)
pinn_params = sum(p.numel() for p in pinn.parameters())
pinn_opt = optim.Adam(pinn.parameters(), lr=0.005)

t0 = time.time()
for epoch in range(1200):
    pinn_opt.zero_grad()
    # Data loss on sparse points
    x_pred_obs = pinn(t_obs)
    l_data = torch.mean((x_pred_obs - x_obs)**2)
    # Physics residual loss on collocation points
    _, residual = pinn.compute_residual(t_colloc)
    l_phys = torch.mean(residual**2)
    
    loss_pinn = l_data + 0.1 * l_phys
    loss_pinn.backward()
    pinn_opt.step()
pinn_time = (time.time() - t0) * 1000.0

with torch.no_grad():
    pinn_train_fit = torch.mean((pinn(torch.tensor(t_train_np, dtype=torch.float32).unsqueeze(-1)) - torch.tensor(x_train_np, dtype=torch.float32).unsqueeze(-1))**2).item()
    pinn_test_pred = pinn(t_test_t)
    pinn_extrap_nrmse = np.sqrt(torch.mean((pinn_test_pred - x_test_t)**2).item()) / np.std(x_test_np)

# 2. Train Physical Phase-Space Neuron
print("--- Training Physical Phase-Space Neuron (Ours: 5 params) ---")
phys_neuron = PhysicalPhaseSpaceNeuron(dt_val=dt)
phys_params = sum(p.numel() for p in phys_neuron.parameters())
phys_opt = optim.Adam(phys_neuron.parameters(), lr=0.02)

t0 = time.time()
for epoch in range(1200):
    phys_opt.zero_grad()
    # Forward unroll across train horizon
    traj = phys_neuron(steps=N_train_steps)
    # Loss strictly on sparse observations
    l_data = torch.mean((traj[obs_indices] - x_obs)**2)
    l_data.backward()
    phys_opt.step()
phys_time = (time.time() - t0) * 1000.0

with torch.no_grad():
    full_traj = phys_neuron(steps=len(t_full))
    phys_train_fit = torch.mean((full_traj[:N_train_steps] - torch.tensor(x_train_np, dtype=torch.float32).unsqueeze(-1))**2).item()
    phys_extrap_pred = full_traj[N_train_steps:]
    phys_extrap_nrmse = np.sqrt(torch.mean((phys_extrap_pred - x_test_t)**2).item()) / np.std(x_test_np)

print("\n" + "=" * 90)
print(f"{'Metric':<35} | {'Standard PINN (Raissi)':>25} | {'Physical Phase-Space Neuron (Ours)':>30}")
print("-" * 96)
print(f"{'Total Trainable Parameters':<35} | {pinn_params:>25d} | {phys_params:>30d}")
print(f"{'Training Time (1200 epochs)':<35} | {pinn_time:>23.1f}ms | {phys_time:>28.1f}ms")
print(f"{'Training Interpolation MSE':<35} | {pinn_train_fit:>25.6f} | {phys_train_fit:>30.6f}")
print(f"{'Future Extrapolation NRMSE':<35} | {pinn_extrap_nrmse:>25.5f} | {phys_extrap_nrmse:>30.5f}")
print(f"{'Discovered gamma (Truth: 0.200)':<35} | {pinn.gamma.item():>25.4f} | {phys_neuron.gamma.item():>30.4f}")
print(f"{'Discovered alpha (Truth: 1.000)':<35} | {pinn.alpha.item():>25.4f} | {phys_neuron.alpha.item():>30.4f}")
print(f"{'Discovered beta  (Truth: 0.500)':<35} | {pinn.beta.item():>25.4f} | {phys_neuron.beta.item():>30.4f}")
print("=" * 90)
