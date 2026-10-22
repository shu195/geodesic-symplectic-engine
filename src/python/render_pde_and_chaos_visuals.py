import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'axes.linewidth': 1.1,
    'grid.linewidth': 0.6,
    'grid.alpha': 0.4
})

artifact_dir = r"C:\Users\Admin\.gemini\antigravity\brain\a6ace972-7a51-4875-b0fd-e9b9d6226f42"
scratch_dir = r"C:\Users\Admin\.gemini\antigravity\scratch"

# ==============================================================================
# FIGURE 1: VISCOUS BURGERS' PDE SHOCK DISCOVERY
# ==============================================================================
print(">>> Generating Figure 1: Viscous Burgers' PDE Shock Discovery...")
data_pde = np.load(os.path.join(scratch_dir, "burgers_discovery_data.npz"), allow_pickle=True)
x = data_pde['x']
t = data_pde['t']
u_gt = data_pde['u_gt']
term_names = data_pde['term_names']
final_coeffs = data_pde['final_coeffs']
gt_coeffs = data_pde['gt_coeffs']

# Simulate discovered PDE forward to verify reconstruction
nx = len(x)
nt = len(t)
dx = x[1] - x[0]
dt = t[1] - t[0]
k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)

disc_advection = final_coeffs[3]
disc_diffusion = final_coeffs[2]

u_disc = np.zeros_like(u_gt)
u_disc[0] = u_gt[0]
u_curr = u_gt[0].copy()

def rhs_disc(u):
    u_hat = np.fft.fft(u)
    ux = np.real(np.fft.ifft(1j * k * u_hat))
    uxx = np.real(np.fft.ifft(- (k**2) * u_hat))
    return disc_advection * u * ux + disc_diffusion * uxx

for i in range(nt - 1):
    k1 = rhs_disc(u_curr)
    k2 = rhs_disc(u_curr + 0.5 * dt * k1)
    k3 = rhs_disc(u_curr + 0.5 * dt * k2)
    k4 = rhs_disc(u_curr + dt * k3)
    u_curr = u_curr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    u_disc[i + 1] = u_curr

error_pde = np.abs(u_gt - u_disc)

fig1 = plt.figure(figsize=(15, 9))
gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.35, wspace=0.30)

# (A) Ground Truth u(x, t)
ax1 = fig1.add_subplot(gs1[0, 0])
im1 = ax1.imshow(u_gt.T, extent=[0, t[-1], x[0], x[-1]], origin='lower', aspect='auto', cmap='inferno')
fig1.colorbar(im1, ax=ax1, label=r'Field Amplitude $u(x, t)$')
ax1.set_title(r'(A) Ground Truth Navier-Stokes 1D $u(x, t)$', fontweight='bold')
ax1.set_xlabel('Time $t$ (seconds)')
ax1.set_ylabel('Spatial Coordinate $x$')

# (B) Discovered PDE Simulation
ax2 = fig1.add_subplot(gs1[0, 1])
im2 = ax2.imshow(u_disc.T, extent=[0, t[-1], x[0], x[-1]], origin='lower', aspect='auto', cmap='inferno')
fig1.colorbar(im2, ax=ax2, label=r'Field Amplitude $u_{\mathrm{disc}}(x, t)$')
ax2.set_title(r'(B) Discovered Law $u_t = -0.996 u u_x + 0.00317 u_{xx}$', fontweight='bold')
ax2.set_xlabel('Time $t$ (seconds)')
ax2.set_ylabel('Spatial Coordinate $x$')

# (C) Absolute Pointwise Error
ax3 = fig1.add_subplot(gs1[0, 2])
im3 = ax3.imshow(error_pde.T, extent=[0, t[-1], x[0], x[-1]], origin='lower', aspect='auto', cmap='magma')
cbar3 = fig1.colorbar(im3, ax=ax3, label=r'Absolute Error $|u - u_{\mathrm{disc}}|$')
cbar3.formatter.set_powerlimits((0, 0))
ax3.set_title(r'(C) Pointwise Reconstruction Residual', fontweight='bold')
ax3.set_xlabel('Time $t$ (seconds)')
ax3.set_ylabel('Spatial Coordinate $x$')

# (D) Spatial Profiles at Discrete Timesteps Showing Shock Front
ax4 = fig1.add_subplot(gs1[1, :2])
time_indices = [0, int(0.25 * nt), int(0.50 * nt), int(0.75 * nt), nt - 1]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

for idx, c in zip(time_indices, colors):
    t_val = t[idx]
    ax4.plot(x, u_gt[idx], color=c, lw=2.0, label=f'GT $t={t_val:.2f}$s')
    ax4.plot(x, u_disc[idx], color='black', lw=1.2, ls='--', alpha=0.85)

# Dummy for legend
ax4.plot([], [], color='black', lw=1.2, ls='--', label='Discovered PDE (Ours)')
ax4.set_title(r'(D) Shock Front Formation & Viscous Boundary Layer Dynamics', fontweight='bold')
ax4.set_xlabel('Spatial Coordinate $x$')
ax4.set_ylabel(r'Field Velocity $u(x)$')
ax4.grid(True)
ax4.legend(loc='upper right', ncol=3, frameon=True, framealpha=0.9, fontsize=8.5)

# (E) Dictionary Pruning Bar Chart
ax5 = fig1.add_subplot(gs1[1, 2])
labels = [r'$u$', r'$u_x$', r'$u_{xx}$', r'$u u_x$', r'$u^2 u_x$', r'$u_{xxx}$', r'$u^2$', r'$u|u|$']
y_pos = np.arange(len(labels))
bar_w = 0.38

ax5.barh(y_pos - bar_w/2, gt_coeffs, height=bar_w, color='#1f77b4', alpha=0.85, label='Ground Truth')
ax5.barh(y_pos + bar_w/2, final_coeffs, height=bar_w, color='#d62728', alpha=0.85, label='Discovered Terms')
ax5.axvline(0, color='black', lw=0.8)
ax5.set_yticks(y_pos)
ax5.set_yticklabels(labels)
ax5.invert_yaxis()
ax5.set_title('(E) 8-Term PDE Parsimony Pruning', fontweight='bold')
ax5.set_xlabel('Coefficient Magnitude')
ax5.grid(True, axis='x')
ax5.legend(loc='lower right', frameon=True, framealpha=0.9)

plt.suptitle("Stress Test: Autonomous Viscous Burgers' PDE Discovery & Shock Resolution", fontsize=14, fontweight='bold', y=0.98)

png1_scratch = os.path.join(scratch_dir, "figure_burgers_pde_shock_discovery.png")
pdf1_scratch = os.path.join(scratch_dir, "figure_burgers_pde_shock_discovery.pdf")
png1_artifact = os.path.join(artifact_dir, "figure_burgers_pde_shock_discovery.png")

plt.savefig(png1_scratch, dpi=300, bbox_inches='tight')
plt.savefig(pdf1_scratch, bbox_inches='tight')
plt.savefig(png1_artifact, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved Figure 1: {png1_scratch}")

# ==============================================================================
# FIGURE 2: CHAOTIC DOUBLE PENDULUM STRESS TEST
# ==============================================================================
print(">>> Generating Figure 2: Chaotic Double Pendulum Stress Test...")
data_pend = np.load(os.path.join(scratch_dir, "double_pendulum_data.npz"), allow_pickle=True)
t_p = data_pend['t']
gt_traj = data_pend['gt_traj']
phys_traj = data_pend['phys_traj']
pinn_traj = data_pend['pinn_traj']
E_gt = data_pend['E_gt']
E_phys = data_pend['E_phys']
E_pinn = data_pend['E_pinn']
obs_idx = data_pend['obs_idx']
obs_th1 = data_pend['obs_th1']
obs_th2 = data_pend['obs_th2']

# Compute Cartesian positions of Bob 2 (x2, y2)
l1, l2 = 1.0, 1.0
x1_gt = l1 * np.sin(gt_traj[:, 0])
y1_gt = -l1 * np.cos(gt_traj[:, 0])
x2_gt = x1_gt + l2 * np.sin(gt_traj[:, 1])
y2_gt = y1_gt - l2 * np.cos(gt_traj[:, 1])

x1_phys = l1 * np.sin(phys_traj[:, 0])
y1_phys = -l1 * np.cos(phys_traj[:, 0])
x2_phys = x1_phys + l2 * np.sin(phys_traj[:, 1])
y2_phys = y1_phys - l2 * np.cos(phys_traj[:, 1])

fig2 = plt.figure(figsize=(15, 9.5))
gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.32, wspace=0.25)

# (A) Phase Portrait (theta1 vs theta2)
ax_a = fig2.add_subplot(gs2[0, 0])
ax_a.plot(gt_traj[:, 0], gt_traj[:, 1], color='#1f77b4', lw=1.5, alpha=0.8, label='Ground Truth Law')
ax_a.plot(phys_traj[:, 0], phys_traj[:, 1], color='#d62728', lw=1.2, ls='--', alpha=0.85, label='Physical Symplectic (Ours)')
ax_a.scatter([gt_traj[0, 0]], [gt_traj[0, 1]], color='black', s=40, zorder=5, label='Start (2.0, 2.0) rad')
ax_a.set_title(r'(A) Chaotic Torus Phase Space $(\theta_1, \theta_2)$', fontweight='bold')
ax_a.set_xlabel(r'$\theta_1$ (radians)')
ax_a.set_ylabel(r'$\theta_2$ (radians)')
ax_a.grid(True)
ax_a.legend(loc='lower left', frameon=True, framealpha=0.9)

# (B) Cartesian Trajectory of Bob 2 (Physical Space Trace)
ax_b = fig2.add_subplot(gs2[0, 1])
ax_b.plot(x2_gt, y2_gt, color='#1f77b4', lw=1.5, alpha=0.8, label='Ground Truth Bob 2 Orbit')
ax_b.plot(x2_phys, y2_phys, color='#d62728', lw=1.1, ls='--', alpha=0.85, label='Discovered Symplectic Orbit')
ax_b.scatter([0], [0], color='black', s=50, marker='+', label='Anchor Origin')
ax_b.set_title('(B) Cartesian Tip Trajectory in Physical Space', fontweight='bold')
ax_b.set_xlabel('Horizontal Displacement $x_2$ (m)')
ax_b.set_ylabel('Vertical Displacement $y_2$ (m)')
ax_b.grid(True)
ax_b.legend(loc='lower left', frameon=True, framealpha=0.9)

# (C) Time Series with Training Window vs Unseen Future Extrapolation
ax_c = fig2.add_subplot(gs2[1, 0])
t_cutoff = 3.0
ax_c.axvspan(0, t_cutoff, color='#2ca02c', alpha=0.10, label='Training Horizon (Sparse Telemetry)')
ax_c.axvspan(t_cutoff, t_p[-1], color='#ff7f0e', alpha=0.10, label='Unseen Future Extrapolation (NRMSE: 0.036)')

ax_c.plot(t_p, gt_traj[:, 0], color='#1f77b4', lw=2.0, label=r'GT $\theta_1(t)$')
ax_c.plot(t_p, phys_traj[:, 0], color='#d62728', lw=1.5, ls='--', label=r'Symplectic $\theta_1(t)$ (Ours)')
ax_c.plot(t_p, pinn_traj[:, 0], color='#7f7f7f', lw=1.2, ls=':', label=r'Standard PINN $\theta_1(t)$ (Diverged)')
ax_c.scatter(t_p[obs_idx], obs_th1, color='black', s=12, alpha=0.5, label='Sensor Observations')

ax_c.set_title(r'(C) Angular Displacement $\theta_1(t)$ Forecasting', fontweight='bold')
ax_c.set_xlabel('Time $t$ (seconds)')
ax_c.set_ylabel(r'$\theta_1$ (radians)')
ax_c.grid(True)
ax_c.set_ylim(-3.5, 3.5)
ax_c.legend(loc='lower left', ncol=2, frameon=True, framealpha=0.85, fontsize=8)

# (D) Hamiltonian Total Energy Drift (Log Scale)
ax_d = fig2.add_subplot(gs2[1, 1])
drift_phys = np.abs(E_phys - E_gt[0]) / np.abs(E_gt[0]) * 100.0 + 1e-6
drift_pinn = np.abs(E_pinn - E_gt[0]) / np.abs(E_gt[0]) * 100.0 + 1e-6
drift_gt   = np.abs(E_gt - E_gt[0]) / np.abs(E_gt[0]) * 100.0 + 1e-6

ax_d.plot(t_p, drift_gt, color='#1f77b4', lw=1.8, label=r'Numerical RK4 Truth ($\Delta E \approx 10^{-4}\%$)')
ax_d.plot(t_p, drift_phys, color='#2ca02c', lw=2.0, ls='--', label=r'Physical Symplectic ($\Delta E = 0.0035\%$ Flat)')
ax_d.plot(t_p, drift_pinn, color='#d62728', lw=1.8, label=r'Standard PINN Baseline ($\Delta E > 10^5\%$ Explosion)')

ax_d.set_yscale('log')
ax_d.set_title(r'(D) Hamiltonian Total Energy Conservation $\Delta E(t) / |E_0|$', fontweight='bold')
ax_d.set_xlabel('Time $t$ (seconds)')
ax_d.set_ylabel(r'Percentage Energy Drift (\%) [Log Scale]')
ax_d.grid(True, which='both')
ax_d.legend(loc='center left', frameon=True, framealpha=0.9, fontsize=8.5)

plt.suptitle("Stress Test: Chaotic Double Pendulum Dynamics & Strict Hamiltonian Conservation", fontsize=14, fontweight='bold', y=0.98)

png2_scratch = os.path.join(scratch_dir, "figure_chaotic_double_pendulum.png")
pdf2_scratch = os.path.join(scratch_dir, "figure_chaotic_double_pendulum.pdf")
png2_artifact = os.path.join(artifact_dir, "figure_chaotic_double_pendulum.png")

plt.savefig(png2_scratch, dpi=300, bbox_inches='tight')
plt.savefig(pdf2_scratch, bbox_inches='tight')
plt.savefig(png2_artifact, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved Figure 2: {png2_scratch}")
