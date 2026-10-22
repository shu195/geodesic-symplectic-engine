"""
Real Physical Sensor Telemetry Benchmark: The Duffing Bistable Beam & Van Der Pol Pacemaker
Comparing:
1. Classical Linear AR Baseline (Linear)
2. Standard Deep MLP (1,200 params)
3. Standard Recurrent Neural Network (LSTM / RNN)
4. Physical Hysteretic Phase-Space Neuron (Ours - 4 params!)

Task:
- Train on first 60% of telemetry (t in [0.0, 7.2] s)
- Test on UNSEEN future 40% (t in [7.2, 12.0] s)
- Evaluate: 1-step NRMSE, and Multi-step Autoregressive Rollout (Autonomous Forecasting)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Load Real Telemetry
# =====================================================================
duffing_path = r'C:\Users\Admin\.gemini\antigravity\scratch\code-neuron-discovery\benchmarks\telemetry\02_duffing_bistable.csv'
df = pd.read_csv(duffing_path, comment='#')
time_vals = df['t'].values
x_raw = df['x'].values
dt = float(time_vals[1] - time_vals[0])

# Normalize slightly to zero mean, unit variance for standard models
x_mean = np.mean(x_raw)
x_std = np.std(x_raw)
x_norm = (x_raw - x_mean) / x_std

# Windowing: window size W = 5
W = 5
X_windows, Y_targets = [], []
for i in range(len(x_norm) - W):
    X_windows.append(x_norm[i:i+W])
    Y_targets.append(x_norm[i+W])

X_windows = np.array(X_windows, dtype=np.float32)
Y_targets = np.array(Y_targets, dtype=np.float32)[:, None]

# Train / Test split by time horizon: 60% Train, 40% Future Extrapolation
split_idx = int(len(X_windows) * 0.60)
X_train, X_test = torch.tensor(X_windows[:split_idx]), torch.tensor(X_windows[split_idx:])
Y_train, Y_test = torch.tensor(Y_targets[:split_idx]), torch.tensor(Y_targets[split_idx:])

print(f"Duffing Telemetry loaded: {len(x_raw)} steps, dt = {dt:.3f}s.")
print(f"Train horizon: {split_idx} steps (t <= {time_vals[split_idx]:.2f}s)")
print(f"Test future horizon: {len(X_test)} steps (t in [{time_vals[split_idx]:.2f}, {time_vals[-1]:.2f}]s)")

# =====================================================================
# 2. Model Definitions
# =====================================================================

# Model 1: Linear Autoregressive Baseline (AR-5)
class LinearAR(nn.Module):
    def __init__(self, in_features=5):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
    def forward(self, x):
        return self.linear(x)

# Model 2: Standard Deep MLP (2 hidden layers, 32-32)
class DeepMLP(nn.Module):
    def __init__(self, in_features=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

# Model 3: Standard Recurrent Net (LSTM)
class LSTMModel(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        # x: [B, W] -> [B, W, 1]
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1, :])

# Model 4: Physical Hysteretic Phase-Space Neuron (Ours)
# Uses physical phase space [x, v]:
# v = (x_t - x_{t-1}) / dt
# Next state generated via non-linear double-well hysteretic acceleration:
# a = -gamma * v + w_self * tanh(x) - beta * x^3
class PhysicalHystereticNeuron(nn.Module):
    def __init__(self, dt_val=0.01):
        super().__init__()
        self.dt = dt_val
        # Physical parameters: damping gamma, bistable self-loop w_self, cubic non-linearity beta
        self.gamma = nn.Parameter(torch.tensor([0.15]))
        self.w_self = nn.Parameter(torch.tensor([1.8])) # bistable potential well
        self.beta = nn.Parameter(torch.tensor([0.4]))
        self.bias = nn.Parameter(torch.tensor([0.0]))
        
    def forward(self, x_window):
        # x_window: [B, W]. We extract the current state x_t and velocity v_t from the last two observations
        x_curr = x_window[:, -1:]
        x_prev = x_window[:, -2:-1]
        v_curr = (x_curr - x_prev) / self.dt
        
        # Physical acceleration from non-linear hysteretic potential:
        accel = -self.gamma * v_curr + self.w_self * torch.tanh(x_curr) - self.beta * (x_curr**3) + self.bias
        # Symplectic Verlet / Euler-Cromer integration step:
        v_next = v_curr + self.dt * accel
        x_next = x_curr + self.dt * v_next
        return x_next

# =====================================================================
# 3. Training & Evaluation Pipeline
# =====================================================================
crit = nn.MSELoss()

def train_and_eval(model, name, lr=0.01, epochs=250):
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
        test_preds = model(X_test)
        test_mse = crit(test_preds, Y_test).item()
        # Normalized Root Mean Squared Error (NRMSE = RMSE / std)
        nrmse = np.sqrt(test_mse) / np.std(Y_test.numpy())
        
        # Multi-Step Autoregressive Rollout (Autonomous Forecasting for 50 future steps)
        # Starting from the first test window, we feed the model's OWN prediction back into itself
        curr_window = X_test[0:1].clone()
        rollout_preds = []
        for step in range(50):
            next_pt = model(curr_window)
            rollout_preds.append(next_pt.item())
            # Shift window: drop oldest, append prediction
            curr_window = torch.cat([curr_window[:, 1:], next_pt], dim=1)
            
        ground_truth_50 = Y_test[:50].numpy().flatten()
        rollout_nrmse = np.sqrt(np.mean((np.array(rollout_preds) - ground_truth_50)**2)) / np.std(ground_truth_50)
        
    return {
        'name': name,
        'params': params,
        'test_nrmse': nrmse,
        'rollout_nrmse': rollout_nrmse,
        'time_ms': elapsed,
        'model': model
    }

print("=" * 85)
print("BENCHMARK: DUFFING BISTABLE BEAM PHYSICAL TELEMETRY")
print("=" * 85)

models = [
    (LinearAR(in_features=W), "Classical Linear AR (AR-5)", 0.02, 200),
    (DeepMLP(in_features=W, hidden=32), "Standard Deep MLP (32-32)", 0.01, 250),
    (LSTMModel(hidden=16), "Standard LSTM Recurrent Net", 0.01, 250),
    (PhysicalHystereticNeuron(dt_val=dt), "Hysteretic Physical Neuron (Ours)", 0.02, 250),
]

results = []
print(f"{'Model Architecture':<36} | {'Params':>6} | {'1-Step NRMSE':>12} | {'50-Step Rollout NRMSE':>21} | {'Time (ms)':>9}")
print("-" * 92)
for m, name, lr, ep in models:
    r = train_and_eval(m, name, lr=lr, epochs=ep)
    results.append(r)
    print(f"{r['name']:<36} | {r['params']:>6d} | {r['test_nrmse']:>12.5f} | {r['rollout_nrmse']:>21.5f} | {r['time_ms']:>8.1f}ms")

# Physical parameter discovery
h_model = results[-1]['model']
print("\nLearned Physical Parameters from Real Sensor Telemetry:")
print(f"gamma (Damping): {h_model.gamma.item():.4f}")
print(f"w_self (Bistable Hysteresis): {h_model.w_self.item():.4f} (Bistable > 1.0: {h_model.w_self.item() > 1.0})")
print(f"beta (Cubic Non-Linearity): {h_model.beta.item():.4f}")
