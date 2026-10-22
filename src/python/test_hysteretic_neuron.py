"""
Benchmark: Hysteretic Self-Loop Neuron vs. Standard Vanilla RNN
Task: Long-Range Bit Latch over T=50 and T=100 Delay Steps with Distractor Noise
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Models
# =====================================================================
class VanillaRNNCell(nn.Module):
    """Standard vanilla RNN with 1 recurrent unit."""
    def __init__(self):
        super().__init__()
        self.w_in = nn.Parameter(torch.randn(1) * 0.5)
        self.w_rec = nn.Parameter(torch.randn(1) * 0.5)
        self.b = nn.Parameter(torch.zeros(1))
        
    def forward(self, x_seq):
        # x_seq: [batch, T]
        batch, T = x_seq.shape
        h = torch.zeros(batch, 1)
        for t in range(T):
            x_t = x_seq[:, t:t+1]
            h = torch.tanh(self.w_in * x_t + self.w_rec * h + self.b)
        return h

class HystereticNeuron(nn.Module):
    """
    Self-Looping Hysteretic Neuron.
    Initializes or bounds w_self to explore the bistable regime (w_self > 1.0).
    """
    def __init__(self, init_w_self=1.8):
        super().__init__()
        self.w_in = nn.Parameter(torch.tensor([1.5]))
        self.w_self = nn.Parameter(torch.tensor([float(init_w_self)]))
        self.b = nn.Parameter(torch.zeros(1))
        
    def forward(self, x_seq):
        batch, T = x_seq.shape
        # Initialize near 0
        y = torch.zeros(batch, 1)
        trajectory = []
        for t in range(T):
            x_t = x_seq[:, t:t+1]
            # Fixed point iteration or single-step update:
            # y_t = tanh(w_in * x_t + w_self * y_{t-1} + b)
            y = torch.tanh(self.w_in * x_t + self.w_self * y + self.b)
            trajectory.append(y.squeeze(-1))
        return y, torch.stack(trajectory, dim=1)

# =====================================================================
# 2. Dataset Generator: Long-Range Bit Latch Task
# =====================================================================
def generate_bit_latch_data(batch_size=128, T=50, noise_std=0.08):
    # Target bit at t=0: randomly -1.0 or +1.0
    bits = (torch.randint(0, 2, (batch_size,)) * 2 - 1).float()
    
    # Input sequence:
    # At t=0: the bit pulse
    # At t=1..T-1: distractor Gaussian noise with mean 0
    x_seq = torch.randn(batch_size, T) * noise_std
    x_seq[:, 0] = bits
    
    # Target is the original bit at final step
    targets = (bits > 0).float().unsqueeze(-1) # 0 or 1 binary target
    return x_seq, targets, bits

# =====================================================================
# 3. Experiment Execution
# =====================================================================
def run_benchmark(T=50):
    print("=" * 65)
    print(f"BENCHMARK: LONG-RANGE BIT LATCH (DELAY T = {T} STEPS)")
    print("Can a SINGLE neuron hold a bit across 50-100 steps of noise?")
    print("=" * 65)
    
    train_x, train_y, _ = generate_bit_latch_data(batch_size=512, T=T)
    test_x, test_y, raw_bits = generate_bit_latch_data(batch_size=200, T=T)
    
    # 1. Evaluate Vanilla RNN
    print("\n--- Training Standard Vanilla RNN (1 unit) ---")
    vanilla = VanillaRNNCell()
    opt_v = optim.Adam(vanilla.parameters(), lr=0.05)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(300):
        opt_v.zero_grad()
        out = vanilla(train_x)
        loss = criterion(out, train_y)
        loss.backward()
        opt_v.step()
        
    with torch.no_grad():
        v_preds = (torch.sigmoid(vanilla(test_x)) > 0.5).float()
        v_acc = (v_preds == test_y).float().mean().item() * 100.0
        print(f"Vanilla RNN Accuracy at T={T}: {v_acc:.1f}%")
        print(f"Vanilla Learned Weights: w_in={vanilla.w_in.item():.3f}, w_rec={vanilla.w_rec.item():.3f}")
        
    # 2. Evaluate Hysteretic Self-Loop Neuron
    print("\n--- Training Hysteretic Self-Loop Neuron (1 unit) ---")
    hysteretic = HystereticNeuron(init_w_self=1.8)
    opt_h = optim.Adam(hysteretic.parameters(), lr=0.05)
    
    for epoch in range(300):
        opt_h.zero_grad()
        out, _ = hysteretic(train_x)
        loss = criterion(out, train_y)
        loss.backward()
        opt_h.step()
        
    with torch.no_grad():
        h_out, traj = hysteretic(test_x)
        h_preds = (torch.sigmoid(h_out) > 0.5).float()
        h_acc = (h_preds == test_y).float().mean().item() * 100.0
        print(f"Hysteretic Neuron Accuracy at T={T}: {h_acc:.1f}%")
        print(f"Hysteretic Learned Weights: w_in={hysteretic.w_in.item():.3f}, w_self={hysteretic.w_self.item():.3f}")
        print(f"Bistable Regime Check: w_self > 1.0 -> {hysteretic.w_self.item() > 1.0}")
        
        # Check trajectory latch stability across first 5 test samples
        print("\nTrajectory Snapshot across time steps t=[0, 1, 10, 25, 49]:")
        sample_indices = [0, 1, 10, min(25, T-1), T-1]
        for i in range(min(4, len(test_x))):
            t_slice = traj[i, sample_indices].numpy().round(3)
            print(f"Sample {i} (Target Bit {int(raw_bits[i].item()):+d}): y(t)={t_slice.tolist()} -> Final Latch State: {h_out[i].item():+.3f}")

if __name__ == '__main__':
    run_benchmark(T=50)
    print("\n")
    run_benchmark(T=100)
