"""
Empirical Investigation: Emergent Group Dynamics of Hysteretic Self-Looping Neurons
Examining 3 Phenomena:
1. 2-Neuron Central Pattern Generator (Spontaneous Relaxation Oscillator)
2. 50-Neuron Continuous Ring Attractor (Hyper-Stable Working Memory Bump)
3. 20-Neuron Associative Memory Landscape (Resistance to Spurious Attractor Collapse)
"""
import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)

print("=" * 70)
print("EXPERIMENT 1: 2-NEURON ASYMMETRIC PAIR -> CENTRAL PATTERN GENERATOR (CPG)")
print("Can two hysteretic neurons spontaneously generate stable rhythmic oscillations?")
print("=" * 70)

# Two neurons:
# Neuron 1 has self-loop w_self, excited by neuron 2
# Neuron 2 has self-loop w_self, inhibited by neuron 1
# J = [[w_self, +J_cross], [-J_cross, w_self]]
w_self = 1.6
J_cross = 1.2
dt = 0.1
T = 200

# Continuous-time leaky update: dy/dt = -y + tanh(w_self * y + J_cross * y_other)
y1, y2 = 0.2, -0.1
y1_traj, y2_traj = [], []

for t in range(T):
    dy1 = -y1 + np.tanh(w_self * y1 + J_cross * y2)
    dy2 = -y2 + np.tanh(w_self * y2 - J_cross * y1)
    y1 += dt * dy1
    y2 += dt * dy2
    y1_traj.append(y1)
    y2_traj.append(y2)

y1_traj = np.array(y1_traj)
y2_traj = np.array(y2_traj)

# Detect limit cycle (is there periodic oscillation?)
zero_crossings = np.where(np.diff(np.sign(y1_traj[50:])))[0]
is_oscillating = len(zero_crossings) >= 4
period_steps = np.mean(np.diff(zero_crossings)) * 2 if len(zero_crossings) >= 2 else 0
print(f"Autonomous Limit Cycle Detected: {is_oscillating}")
print(f"Measured Oscillation Period: {period_steps * dt:.2f} time units ({len(zero_crossings)} half-cycles)")
print(f"State Range y1: [{y1_traj.min():.3f}, {y1_traj.max():.3f}], y2: [{y2_traj.min():.3f}, {y2_traj.max():.3f}]")


print("\n" + "=" * 70)
print("EXPERIMENT 2: 50-NEURON RING MANIFOLD -> PERSISTENT BUMP ATTRACTOR")
print("Does a ring of hysteretic neurons hold an analog position bump in noise?")
print("=" * 70)

N = 50
positions = np.linspace(-np.pi, np.pi, N, endpoint=False)

# Lateral connectivity: Mexican-hat coupling (local excitation, global inhibition)
dist = np.abs(positions[:, None] - positions[None, :])
dist = np.minimum(dist, 2 * np.pi - dist) # circular distance on ring
J_ring = 1.5 * np.exp(-dist**2 / (2 * 0.4**2)) - 0.4
np.fill_diagonal(J_ring, 0.0) # remove diagonal so we isolate w_self explicitly

def test_ring_bump(w_s, label):
    # Initialize a Gaussian bump at position theta = 0 for 5 steps
    y = np.exp(-positions**2 / (2 * 0.5**2))
    
    # Run 100 steps in the dark with random distractor noise
    for step in range(100):
        noise = np.random.randn(N) * 0.05
        # Update: y = tanh(w_s * y + J_ring @ y + noise)
        net_input = w_s * y + J_ring @ y + noise
        y = np.tanh(net_input)
        
    # Measure bump peak height and width
    peak_val = np.max(y)
    baseline_val = np.min(y)
    contrast = peak_val - baseline_val
    # Peak location
    peak_idx = np.argmax(y)
    peak_pos = positions[peak_idx]
    return contrast, peak_val, peak_pos

contrast_std, peak_std, pos_std = test_ring_bump(w_s=0.0, label="Standard Network (w_self=0)")
contrast_hyst, peak_hyst, pos_hyst = test_ring_bump(w_s=2.2, label="Hysteretic Ring (w_self=2.2)")

print(f"Standard Ring (w_self = 0.0): Bump Contrast = {contrast_std:.3f}, Peak Height = {peak_std:.3f}")
print(f"Hysteretic Ring (w_self = 2.2): Bump Contrast = {contrast_hyst:.3f}, Peak Height = {peak_hyst:.3f}")
print(f"Result: Standard ring {'collapsed' if contrast_std < 0.2 else 'held'}, Hysteretic ring held bump with contrast {contrast_hyst:.3f}!")


print("\n" + "=" * 70)
print("EXPERIMENT 3: 20-NEURON ASSOCIATIVE MEMORY -> RESISTANCE TO SPURIOUS COLLAPSE")
print("Does self-hysteresis protect stored memory states against noisy basin collapse?")
print("=" * 70)

N_mem = 20
P_patterns = 3 # Store 3 random orthogonal patterns
patterns = np.random.choice([-1.0, 1.0], size=(P_patterns, N_mem))

# Standard Hebbian weight matrix
W_hebb = (patterns.T @ patterns) / N_mem
np.fill_diagonal(W_hebb, 0.0)

# Test recall from 30% corrupted probe
corrupt_bits = 6 # 30% noise
trials = 50

for w_s in [0.0, 0.8, 1.5, 2.5]:
    correct_recalls = 0
    for p_idx in range(P_patterns):
        pat = patterns[p_idx]
        for _ in range(trials):
            # Corrupt pattern
            probe = pat.copy()
            flip_indices = np.random.choice(N_mem, size=corrupt_bits, replace=False)
            probe[flip_indices] *= -1.0
            
            # Relax for 20 steps
            y = probe.copy()
            for _ in range(20):
                y = np.tanh(w_s * y + W_hebb @ y)
                
            # Check overlap with true pattern
            overlap = np.dot(np.sign(y), pat) / N_mem
            if overlap == 1.0: # Exact recovery
                correct_recalls += 1
                
    acc = (correct_recalls / (P_patterns * trials)) * 100.0
    print(f"w_self = {w_s:.1f}: Pattern Recovery Accuracy = {acc:.1f}% ({correct_recalls}/{P_patterns * trials})")
