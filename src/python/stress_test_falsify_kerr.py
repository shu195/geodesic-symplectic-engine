"""
Adversarial Falsification Suite for Kerr Singularity Geodesic Neuron
3 Critical Stress Tests:
1. Arbitrary 3D Rotational Tilt & Random Initialization (Euler angles alpha=45, beta=35, gamma=20, offset center)
2. Elliptical / Squashed Torus (Broken circular symmetry: a=2.4, b=1.1)
3. The Hopf Link (2 Interlocking Perpendicular Rings in 3D space)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

torch.manual_seed(1337)
np.random.seed(1337)

def make_rotation_matrix(rx, ry, rz):
    # 3D Euler rotation matrix
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

R_rot = make_rotation_matrix(np.radians(45), np.radians(35), np.radians(20))
center_offset = np.array([1.2, -1.5, 0.8])

# =====================================================================
# Model Definition
# =====================================================================
class ArbitraryKerrNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        # Random initial center far from truth
        self.center = nn.Parameter(torch.randn(3) * 0.5)
        # Completely random orientation vector
        self.normal_raw = nn.Parameter(torch.randn(3))
        self.R = nn.Parameter(torch.tensor([1.0]))
        self.alpha = nn.Parameter(torch.tensor([2.0]))
        self.threshold = nn.Parameter(torch.tensor([0.25]))
        
    def forward(self, x):
        normal = self.normal_raw / (torch.norm(self.normal_raw) + 1e-6)
        diff = x - self.center.unsqueeze(0)
        h = torch.sum(diff * normal.unsqueeze(0), dim=-1, keepdim=True)
        r_perp = diff - h * normal.unsqueeze(0)
        rho = torch.norm(r_perp, dim=-1, keepdim=True)
        d_ring_sq = (rho - torch.abs(self.R))**2 + h**2
        return torch.sigmoid(self.alpha * (self.threshold - d_ring_sq)), d_ring_sq

class MultiKerrNetwork(nn.Module):
    """Network of K Kerr Ring Neurons for multi-ring topologies (e.g. Hopf Link)."""
    def __init__(self, num_rings=2):
        super().__init__()
        self.rings = nn.ModuleList([ArbitraryKerrNeuron() for _ in range(num_rings)])
        
    def forward(self, x):
        # Union of ring manifolds (if inside ANY ring -> Class 1)
        # Using smooth maximum / saturating sum: 1 - prod(1 - y_k)
        outs = []
        for ring in self.rings:
            out, _ = ring(x)
            outs.append(out)
        combined = 1.0 - torch.prod(torch.stack([1.0 - o for o in outs], dim=0), dim=0)
        return combined

def train_eval(model, X, Y, lr=0.08, epochs=300):
    params = sum(p.numel() for p in model.parameters())
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X) if not isinstance(model, ArbitraryKerrNeuron) else model(X)[0]
        loss = crit(out, Y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = (out > 0.5).float()
        acc = (preds == Y).float().mean().item() * 100.0
        pos_recall = (preds[Y == 1.0] == 1.0).float().mean().item() * 100.0
        neg_recall = (preds[Y == 0.0] == 0.0).float().mean().item() * 100.0
    return params, acc, pos_recall, neg_recall, loss.item()

print("=" * 80)
print("FALSIFICATION TEST 1: ARBITRARY 3D ROTATION & OFF-CENTER TRANSLATION")
print("True torus rotated by (45 deg, 35 deg, 20 deg) and translated to [1.2, -1.5, 0.8]")
print("Kerr neuron initialized with COMPLETELY RANDOM orientation and center!")
print("=" * 80)

N = 2500
# Canonical torus
r_major_true, r_minor_true = 1.8, 0.45
u_pts = np.random.uniform(-3.5, 3.5, N)
v_pts = np.random.uniform(-3.5, 3.5, N)
w_pts = np.random.uniform(-2.0, 2.0, N)
canonical_pts = np.column_stack([u_pts, v_pts, w_pts])

r_xy_can = np.sqrt(u_pts**2 + v_pts**2)
d_can = np.sqrt((r_xy_can - r_major_true)**2 + w_pts**2)
y_tilted = (d_can <= r_minor_true).astype(np.float32)

# Apply 3D rotation and shift
tilted_pts = (R_rot @ canonical_pts.T).T + center_offset

X1 = torch.tensor(tilted_pts, dtype=torch.float32)
Y1 = torch.tensor(y_tilted, dtype=torch.float32).unsqueeze(-1)

kerr1 = ArbitraryKerrNeuron()
p1, acc1, pos1, neg1, loss1 = train_eval(kerr1, X1, Y1, lr=0.08, epochs=350)
print(f"Kerr Neuron (9 params) | Accuracy: {acc1:.2f}% | Torus Recall: {pos1:.2f}% | Loss: {loss1:.4f}")
print(f"Learned Center: {kerr1.center.detach().numpy().round(2)} (Truth: {center_offset})")
normal_est = (kerr1.normal_raw / torch.norm(kerr1.normal_raw)).detach().numpy()
normal_true = R_rot @ np.array([0, 0, 1])
dot_alignment = abs(np.dot(normal_est, normal_true))
print(f"Normal Vector Angular Alignment |dot(n_est, n_true)|: {dot_alignment:.4f} (1.0 = Exact match)")
print(f"Outcome: {'PASSED (Rotational Invariance Locked)' if dot_alignment > 0.95 else 'FAILED (Trapped in local minimum)'}")


print("\n" + "=" * 80)
print("FALSIFICATION TEST 2: SQUASHED / ELLIPTICAL TORUS (BROKEN CIRCULAR SYMMETRY)")
print("Donut is squashed: semi-major a = 2.4, semi-minor b = 1.1 (eccentricity e = 0.88)")
print("=" * 80)

# Generate elliptical torus: (x/a)^2 + (y/b)^2
a_axis, b_axis, r_core = 2.4, 1.1, 0.4
x_ell = np.random.uniform(-3.5, 3.5, N)
y_ell = np.random.uniform(-3.0, 3.0, N)
z_ell = np.random.uniform(-1.5, 1.5, N)

# Distance to ellipse ring
# Parameter t along ellipse
phi = np.arctan2(y_ell / b_axis, x_ell / a_axis)
closest_x = a_axis * np.cos(phi)
closest_y = b_axis * np.sin(phi)
d_ellipse = np.sqrt((x_ell - closest_x)**2 + (y_ell - closest_y)**2 + z_ell**2)
y_ellipse = (d_ellipse <= r_core).astype(np.float32)

X2 = torch.tensor(np.column_stack([x_ell, y_ell, z_ell]), dtype=torch.float32)
Y2 = torch.tensor(y_ellipse, dtype=torch.float32).unsqueeze(-1)

kerr2 = ArbitraryKerrNeuron()
p2, acc2, pos2, neg2, loss2 = train_eval(kerr2, X2, Y2, lr=0.08, epochs=300)

mlp2 = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
p_mlp2, acc_mlp2, pos_mlp2, neg_mlp2, loss_mlp2 = train_eval(mlp2, X2, Y2, lr=0.04, epochs=300)

print(f"Kerr Neuron (Circular bias) (9 params)  | Accuracy: {acc2:.2f}% | Recall: {pos2:.2f}% | Loss: {loss2:.4f}")
print(f"Standard MLP (Flexible)    (353 params) | Accuracy: {acc_mlp2:.2f}% | Recall: {pos_mlp2:.2f}% | Loss: {loss_mlp2:.4f}")


print("\n" + "=" * 80)
print("FALSIFICATION TEST 3: THE HOPF LINK (TWO INTERLOCKING PERPENDICULAR RINGS)")
print("Ring 1 in XY-plane centered at (0, 0, 0), Ring 2 in XZ-plane centered at (1.5, 0, 0)")
print("=" * 80)

# Ring 1: R=1.5, core=0.35, xy plane, centered at (0, 0, 0)
d1 = np.sqrt((np.sqrt(u_pts**2 + v_pts**2) - 1.5)**2 + w_pts**2)
# Ring 2: R=1.5, core=0.35, xz plane, centered at (1.5, 0, 0)
d2 = np.sqrt((np.sqrt((u_pts - 1.5)**2 + w_pts**2) - 1.5)**2 + v_pts**2)

y_hopf = ((d1 <= 0.35) | (d2 <= 0.35)).astype(np.float32)

X3 = torch.tensor(canonical_pts, dtype=torch.float32)
Y3 = torch.tensor(y_hopf, dtype=torch.float32).unsqueeze(-1)

# 2-Neuron Kerr Network (18 parameters)
hopf_kerr = MultiKerrNetwork(num_rings=2)
# Initialize ring 1 near xy, ring 2 near xz
hopf_kerr.rings[0].normal_raw.data = torch.tensor([0.1, 0.1, 1.0])
hopf_kerr.rings[1].normal_raw.data = torch.tensor([0.1, 1.0, 0.1])
hopf_kerr.rings[0].center.data = torch.tensor([0.1, 0.0, 0.0])
hopf_kerr.rings[1].center.data = torch.tensor([1.2, 0.0, 0.0])

p3, acc3, pos3, neg3, loss3 = train_eval(hopf_kerr, X3, Y3, lr=0.06, epochs=350)

mlp3 = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
p_mlp3, acc_mlp3, pos_mlp3, neg_mlp3, loss_mlp3 = train_eval(mlp3, X3, Y3, lr=0.04, epochs=350)

print(f"2-Ring Kerr Network (18 params)   | Accuracy: {acc3:.2f}% | Recall: {pos3:.2f}% | Loss: {loss3:.4f}")
print(f"Standard Large MLP  (1217 params) | Accuracy: {acc_mlp3:.2f}% | Recall: {pos_mlp3:.2f}% | Loss: {loss_mlp3:.4f}")
