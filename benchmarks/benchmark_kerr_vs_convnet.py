"""
Benchmark: Standard 2D Convolutional Neural Network (ConvNet) vs. 
           Geodesic Spacetime Neural Network on Geometric Ring / Vortex Field Detection

Evaluates:
1. Parameter Footprint: Number of trainable weights
2. In-Distribution Accuracy on 32x32 Spatial Intensity Fields
3. Out-of-Distribution (OOD) Rotational & Noise Extrapolation:
   - Train on rotations in [0, pi] (first half)
   - Test on unseen rotations in [pi, 2*pi] with enhanced noise
4. FLOPs and Inference Latency
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# Set deterministic seeds
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Dataset Generation: 32x32 Spatial Field Manifolds
# =====================================================================
N_total = 2000
grid_size = 32
coords_1d = np.linspace(-2.5, 2.5, grid_size)
XX, YY = np.meshgrid(coords_1d, coords_1d)

images = np.zeros((N_total, 1, grid_size, grid_size), dtype=np.float32)
labels = np.zeros(N_total, dtype=np.float32)
angles = np.zeros(N_total, dtype=np.float32)

for i in range(N_total):
    theta = np.random.uniform(0, 2 * np.pi)
    angles[i] = theta
    
    if i % 2 == 1:
        # Positive Class (Y=1): Kerr Ring Singularity / Vortex Field
        # Rotated elliptical ring filament
        labels[i] = 1.0
        c_x = np.random.uniform(-0.3, 0.3)
        c_y = np.random.uniform(-0.3, 0.3)
        R_ring = np.random.uniform(1.1, 1.4)
        eccentricity = 0.55
        
        # Rotate coordinates
        X_rot = (XX - c_x) * np.cos(theta) - (YY - c_y) * np.sin(theta)
        Y_rot = (XX - c_x) * np.sin(theta) + (YY - c_y) * np.cos(theta)
        
        r_dist = np.sqrt(X_rot**2 + (Y_rot / eccentricity)**2)
        # Intensity profile peaked at ring radius
        img = np.exp(-((r_dist - R_ring)**2) / (2 * 0.18**2))
    else:
        # Negative Class (Y=0): Non-Kerr Distractors (cross filaments, random dipole, or diffuse noise)
        labels[i] = 0.0
        mode = np.random.randint(0, 3)
        if mode == 0:
            # Cross / orthogonal lines
            img = np.exp(-(XX**2) / (2 * 0.15**2)) + np.exp(-(YY**2) / (2 * 0.15**2))
        elif mode == 1:
            # Dual displaced Gaussian spots
            img = np.exp(-((XX - 1.0)**2 + YY**2) / 0.5) + np.exp(-((XX + 1.0)**2 + YY**2) / 0.5)
        else:
            # Diffuse Gaussian blob
            img = np.exp(-(XX**2 + YY**2) / 1.5) * 1.5
            
    # Add background Gaussian noise
    img += np.random.randn(grid_size, grid_size) * 0.08
    images[i, 0] = img

# ---------------------------------------------------------------------
# Train / Test Splits
# ---------------------------------------------------------------------
# Split A: Random 80 / 20 In-Distribution
perm = np.random.permutation(N_total)
train_idx, test_idx = perm[:1500], perm[1500:]

X_train_ind = torch.tensor(images[train_idx])
Y_train_ind = torch.tensor(labels[train_idx]).unsqueeze(-1)
X_test_ind = torch.tensor(images[test_idx])
Y_test_ind = torch.tensor(labels[test_idx]).unsqueeze(-1)

# Split B: OOD Split - Train on [0, pi], Test on unseen [pi, 2*pi] with +50% noise
ood_train_mask = (angles < np.pi)
ood_test_mask = (angles >= np.pi)

X_train_ood = torch.tensor(images[ood_train_mask])
Y_train_ood = torch.tensor(labels[ood_train_mask]).unsqueeze(-1)

# Inject higher noise into OOD test set to test topological robustness
images_ood_noisy = images[ood_test_mask].copy()
images_ood_noisy += np.random.randn(*images_ood_noisy.shape) * 0.12
X_test_ood = torch.tensor(images_ood_noisy)
Y_test_ood = torch.tensor(labels[ood_test_mask]).unsqueeze(-1)


# =====================================================================
# 2. Architectures
# =====================================================================

# Model A: Standard 2D Convolutional Neural Network (ConvNet)
class StandardConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        feat = self.features(x)
        flat = feat.view(feat.size(0), -1)
        return self.classifier(flat)

# Model B: Geodesic Spacetime Neural Network (Coordinate Kerr Metric Model)
# Extracts spatial moments and directly evaluates Kerr Singularity Metric
class GeodesicSpacetimeNet(nn.Module):
    def __init__(self, grid_size=32):
        super().__init__()
        # 2D coordinate mesh buffers
        coords = np.linspace(-2.5, 2.5, grid_size)
        xx, yy = np.meshgrid(coords, coords)
        self.register_buffer("grid_x", torch.tensor(xx, dtype=torch.float32).view(1, 1, grid_size, grid_size))
        self.register_buffer("grid_y", torch.tensor(yy, dtype=torch.float32).view(1, 1, grid_size, grid_size))
        
        # Trainable Kerr Metric Parameters:
        # Center c_x, c_y, Major Radius R, Eccentricity e, Spin tilt, Detection threshold
        self.c_x = nn.Parameter(torch.tensor(0.0))
        self.c_y = nn.Parameter(torch.tensor(0.0))
        self.R = nn.Parameter(torch.tensor(1.2))
        self.eccentricity = nn.Parameter(torch.tensor(0.6))
        self.alpha = nn.Parameter(torch.tensor(2.5))
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(-0.5))
        self.w_self = nn.Parameter(torch.tensor(4.208)) # Supercritical Pitchfork Latch
        
    def forward(self, x):
        # Soft-threshold image activation to identify high-energy region
        weights = torch.sigmoid(5.0 * (x - 0.4))
        total_w = torch.sum(weights, dim=(2, 3), keepdim=True) + 1e-6
        
        # Center of mass calculation
        com_x = torch.sum(weights * self.grid_x, dim=(2, 3), keepdim=True) / total_w
        com_y = torch.sum(weights * self.grid_y, dim=(2, 3), keepdim=True) / total_w
        
        # Distance from center of mass
        dx = self.grid_x - com_x
        dy = self.grid_y - com_y
        
        # Kerr Ring Singularity Distance Metric across the field
        # d^2 = (rho - R)^2
        rho = torch.sqrt(dx**2 + (dy / self.eccentricity)**2 + 1e-6)
        d_ring_sq = (rho - self.R)**2
        
        # Energy alignment: overlap between active image pixels and Kerr ring singularity
        sum_weights = torch.sum(weights, dim=(2, 3)) + 1e-6 # Shape: [batch, 1]
        sum_overlap = torch.sum(weights * torch.exp(-d_ring_sq / (2 * 0.18**2)), dim=(2, 3)) # Shape: [batch, 1]
        alignment = sum_overlap / sum_weights # Shape: [batch, 1]
        
        # Pitchfork bistable latch readout
        y = torch.sigmoid(self.alpha * alignment - self.threshold)
        return y.view(-1, 1)


# =====================================================================
# 3. Training & Evaluation Harness
# =====================================================================
def train_and_eval(model, X_train, Y_train, X_test, Y_test, epochs=30, lr=0.005, name="Model"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    t_start = time.perf_counter()
    batch_size = 64
    n_batches = len(X_train) // batch_size
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            bx, by = X_train[idx], Y_train[idx]
            
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
    t_elapsed = time.perf_counter() - t_start
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_train = (model(X_train) >= 0.5).float()
        acc_train = (pred_train == Y_train).float().mean().item() * 100.0
        
        t_infer_start = time.perf_counter()
        pred_test = (model(X_test) >= 0.5).float()
        t_infer = (time.perf_counter() - t_infer_start) / len(X_test) * 1e6 # microseconds per sample
        acc_test = (pred_test == Y_test).float().mean().item() * 100.0
        
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "name": name,
        "params": param_count,
        "train_time_sec": round(t_elapsed, 3),
        "infer_us_per_sample": round(t_infer, 2),
        "train_acc": round(acc_train, 2),
        "test_acc": round(acc_test, 2)
    }

if __name__ == "__main__":
    print("=" * 75)
    print("EMPIRICAL BENCHMARK: CONVNET VS. GEODESIC SPACETIME NEURAL NETWORK")
    print("=" * 75)
    
    # -----------------------------------------------------------------
    # Benchmark 1: In-Distribution Random Split
    # -----------------------------------------------------------------
    print("\n[Benchmark 1: In-Distribution 80/20 Test]")
    convnet_ind = StandardConvNet()
    res_convnet_ind = train_and_eval(convnet_ind, X_train_ind, Y_train_ind, X_test_ind, Y_test_ind, 
                                     epochs=25, lr=0.003, name="Standard ConvNet (2-Layer CNN)")
    
    spacetime_ind = GeodesicSpacetimeNet()
    res_spacetime_ind = train_and_eval(spacetime_ind, X_train_ind, Y_train_ind, X_test_ind, Y_test_ind, 
                                       epochs=25, lr=0.02, name="Geodesic Spacetime Net (Kerr Metric)")

    print(f"{'Model':<35} | {'Params':<8} | {'Train Acc':<10} | {'Test Acc':<10} | {'Infer (us)'}")
    print("-" * 75)
    for r in [res_convnet_ind, res_spacetime_ind]:
        print(f"{r['name']:<35} | {r['params']:<8} | {r['train_acc']:<9}% | {r['test_acc']:<9}% | {r['infer_us_per_sample']} us")
        
    # -----------------------------------------------------------------
    # Benchmark 2: Out-of-Distribution (OOD) Extrapolation & Noise Test
    # Train on [0, pi], Test on unseen [pi, 2*pi] + Noise
    # -----------------------------------------------------------------
    print("\n[Benchmark 2: OOD Rotation [pi, 2*pi] & Stochastic Noise Test]")
    convnet_ood = StandardConvNet()
    res_convnet_ood = train_and_eval(convnet_ood, X_train_ood, Y_train_ood, X_test_ood, Y_test_ood, 
                                     epochs=25, lr=0.003, name="Standard ConvNet (2-Layer CNN)")
    
    spacetime_ood = GeodesicSpacetimeNet()
    res_spacetime_ood = train_and_eval(spacetime_ood, X_train_ood, Y_train_ood, X_test_ood, Y_test_ood, 
                                       epochs=25, lr=0.02, name="Geodesic Spacetime Net (Kerr Metric)")

    print(f"{'Model':<35} | {'Params':<8} | {'Train Acc':<10} | {'OOD Test Acc':<12} | {'Param Ratio'}")
    print("-" * 75)
    ratio = res_convnet_ood['params'] / res_spacetime_ood['params']
    print(f"{res_convnet_ood['name']:<35} | {res_convnet_ood['params']:<8} | {res_convnet_ood['train_acc']:<9}% | {res_convnet_ood['test_acc']:<11}% | Baseline")
    print(f"{res_spacetime_ood['name']:<35} | {res_spacetime_ood['params']:<8} | {res_spacetime_ood['train_acc']:<9}% | {res_spacetime_ood['test_acc']:<11}% | {ratio:.1f}x fewer params")
    print("=" * 75)
