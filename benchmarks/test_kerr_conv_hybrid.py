import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

torch.manual_seed(42)
np.random.seed(42)

# Load data generator from benchmark
from benchmark_kerr_vs_convnet import X_train_ind, Y_train_ind, X_test_ind, Y_test_ind, X_train_ood, Y_train_ood, X_test_ood, Y_test_ood, train_and_eval

# Hybrid Architecture: Conv Feature Extractor + Kerr Geodesic Spacetime Readout Head
class KerrConvHybridNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Compact Conv Front-End (Extracts spatial phase features)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 8x8
        )
        # Coordinate grid for 8x8 feature map
        coords = np.linspace(-2.0, 2.0, 8)
        xx, yy = np.meshgrid(coords, coords)
        self.register_buffer("grid_x", torch.tensor(xx, dtype=torch.float32).view(1, 1, 8, 8))
        self.register_buffer("grid_y", torch.tensor(yy, dtype=torch.float32).view(1, 1, 8, 8))
        
        # Kerr Geodesic Spacetime Head (Replaces 65,000-parameter dense FC layer)
        self.R = nn.Parameter(torch.tensor(1.2))
        self.eccentricity = nn.Parameter(torch.tensor(0.6))
        self.alpha = nn.Parameter(torch.tensor(3.0))
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        feat = self.features(x) # [batch, 16, 8, 8]
        # Collapse channel intensity to spatial curvature map
        curv_map = torch.mean(feat, dim=1, keepdim=True) # [batch, 1, 8, 8]
        weights = torch.sigmoid(4.0 * curv_map)
        
        sum_weights = torch.sum(weights, dim=(2, 3), keepdim=True) + 1e-6 # [batch, 1, 1, 1]
        com_x = torch.sum(weights * self.grid_x, dim=(2, 3), keepdim=True) / sum_weights # [batch, 1, 1, 1]
        com_y = torch.sum(weights * self.grid_y, dim=(2, 3), keepdim=True) / sum_weights # [batch, 1, 1, 1]
        
        dx = self.grid_x - com_x # [batch, 1, 8, 8]
        dy = self.grid_y - com_y # [batch, 1, 8, 8]
        rho = torch.sqrt(dx**2 + (dy / self.eccentricity)**2 + 1e-6)
        d_ring_sq = (rho - self.R)**2
        
        sum_overlap = torch.sum(weights * torch.exp(-d_ring_sq / 0.35), dim=(2, 3), keepdim=True)
        alignment = (sum_overlap / sum_weights).view(-1, 1) # [batch, 1]
        
        y = torch.sigmoid(self.alpha * alignment - self.threshold)
        return y.view(-1, 1)

if __name__ == "__main__":
    hybrid_ind = KerrConvHybridNet()
    res_hybrid_ind = train_and_eval(hybrid_ind, X_train_ind, Y_train_ind, X_test_ind, Y_test_ind, epochs=25, lr=0.005, name="Kerr-Conv Hybrid (Geodesic Head)")
    
    hybrid_ood = KerrConvHybridNet()
    res_hybrid_ood = train_and_eval(hybrid_ood, X_train_ood, Y_train_ood, X_test_ood, Y_test_ood, epochs=25, lr=0.005, name="Kerr-Conv Hybrid (Geodesic Head)")
    
    print("\n[Hybrid Kerr-Conv Benchmark Results]")
    print(f"Parameters: {res_hybrid_ind['params']} (vs 70,497 ConvNet -> {70497/res_hybrid_ind['params']:.1f}x reduction)")
    print(f"In-Distribution Test Accuracy: {res_hybrid_ind['test_acc']}%")
    print(f"OOD Rotational Test Accuracy:  {res_hybrid_ood['test_acc']}%")
    print(f"Inference Latency: {res_hybrid_ind['infer_us_per_sample']} us (vs 510.47 us ConvNet)")
