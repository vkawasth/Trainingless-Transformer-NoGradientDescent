import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Set dark background plotting style
plt.style.use('dark_background')

def generate_and_plot_witness_complex_pristine():
    print("Generating Clean Chronological MaxMin Bergman Witness Complex Plot...")
    
    # 1. Simulate Zone 3 Trajectory (25 steps, D=2000 parameter space projected to k=128)
    torch.manual_seed(42)
    steps = 25
    D = 2000
    k_subspace = 128

    # Create a sweeping trajectory arc through parameter space
    t = torch.linspace(0, 1, steps)
    base_path = torch.stack([
        torch.sin(1.2 * math.pi * t) * 15.0,
        torch.cos(1.2 * math.pi * t) * 12.0 - 5.0
    ], dim=1) # [25, 2]
    
    # Lift to full parameter dimension and sketch down to k=128 subspace
    proj_D = torch.randn(2, D)
    raw_trajectory = torch.matmul(base_path, proj_D) + torch.randn(steps, D) * 0.02
    
    sketch_proj = torch.randn(D, k_subspace) / math.sqrt(k_subspace)
    X = torch.matmul(raw_trajectory, sketch_proj) # [25, 128]

    # 2. MaxMin Sampling for Well-Spaced Landmarks
    landmarks_idx = [0]
    dmat = torch.cdist(X, X, p=2)
    for _ in range(1, 6):
        min_dists = dmat[landmarks_idx].min(dim=0).values
        next_idx = torch.argmax(min_dists).item()
        landmarks_idx.append(next_idx)

    # Sort landmark indices chronologically along trajectory step order
    landmarks_idx = torch.tensor(sorted(landmarks_idx)) # [6]
    landmarks = X[landmarks_idx]

    # 3. Metric & Simplex Filtration Setup
    sigma = 4.6390
    r_cutoff = 1.0923

    # Project Subspace Trajectory and Landmarks to 2D via PCA (SVD) for Plotting
    X_centered = X - X.mean(dim=0)
    _, _, V = torch.pca_lowrank(X_centered, q=2)
    X_2d = torch.matmul(X_centered, V).numpy()
    landmarks_2d = X_2d[landmarks_idx.numpy()]

    # Construct chronological local edges (1-simplices)
    edges = [(i, i + 1) for i in range(len(landmarks_idx) - 1)]

    # Log-Stabilized hbar_eff Calculation (Prevents inf Overflow)
    cov_matrix = torch.cov(X.T)
    scaled_trace = torch.trace(cov_matrix).item() / k_subspace
    hbar_eff = math.exp(min(scaled_trace / 1000.0, 50.0))

    # =====================================================================
    # MATPLOTLIB RENDERING
    # =====================================================================
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=120)

    # Render 2-Simplices (Localized along trajectory arc, no interior chords)
    triangle_count = 0
    for i in range(len(landmarks_idx) - 2):
        idx_start = landmarks_idx[i].item()
        idx_mid = landmarks_idx[i+1].item()
        idx_end = landmarks_idx[i+2].item()
        
        # Pull localized trajectory points along the curve segment
        seg_pts = X_2d[idx_start:idx_end+1]
        poly = Polygon(seg_pts, closed=True, facecolor='#a855f7', edgecolor='none', alpha=0.35, zorder=1)
        ax.add_patch(poly)
        triangle_count += 1

    # Render Subspace Trajectory Path & Checkpoints
    ax.plot(X_2d[:, 0], X_2d[:, 1], color='#38bdf8', linestyle='-', linewidth=1.5, alpha=0.6, label='Subspace Trajectory')
    ax.scatter(X_2d[:, 0], X_2d[:, 1], color='#0284c7', s=30, zorder=3, label='Witness Checkpoints (Steps)')

    # Render 1-Simplices (Edges tracing adjacent landmarks)
    edge_drawn = False
    for edge in edges:
        p1, p2 = landmarks_2d[edge[0]], landmarks_2d[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#22c55e', linestyle='--', linewidth=2.0, zorder=2, 
                label='Active 1-Simplex (Edge)' if not edge_drawn else "")
        edge_drawn = True

    # Render 0-Simplices (MaxMin Morse Landmarks)
    ax.scatter(landmarks_2d[:, 0], landmarks_2d[:, 1], color='#f59e0b', s=150, edgecolors='#ffffff', 
               linewidth=1.5, zorder=4, label='MaxMin Landmarks (0-Simplices)')

    # Label text offset coordinates tuned to avoid all shapes and boxes
    offsets = [
        (25, -15),   # L0
        (-115, -15), # L1
        (15, 20),    # L2
        (15, -25),   # L3
        (20, 15),    # L4
        (20, -15)    # L5
    ]

    for idx, (lx, ly) in enumerate(landmarks_2d):
        dx, dy = offsets[idx]
        ax.text(lx + dx, ly + dy, f"L{idx} (Step {landmarks_idx[idx].item()})", 
                color='#fbbf24', fontsize=9.5, weight='bold', zorder=5)

    # Set explicit axis boundaries to give top overlay box plenty of clearance
    y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
    ax.set_ylim(y_min - 100, y_max + 320)

    # Floating Info Overlay (Top-Left placement with clear margin above L1/L2)
    info_text = (
        f"--- Zone 3 MaxMin Bergman Audit ---\n"
        f"Adaptive Kernel Scale (σ) : {sigma:.4f}\n"
        f"Distance Cutoff (r_cutoff): {r_cutoff:.4f}\n"
        f"0-Simplices (Nodes)       : {len(landmarks_idx)}\n"
        f"1-Simplices (Edges)       : {len(edges)}\n"
        f"2-Simplices (Triangles)   : {triangle_count}\n"
        f"Subspaced hbar_eff        : {hbar_eff:.6e}"
    )
    ax.text(0.03, 0.96, info_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#0f172a', edgecolor='#334155', alpha=0.9),
            fontfamily='monospace', fontsize=9.5, color='#e2e8f0')

    # Chart Formatting
    ax.set_title("Zone 3 Topological Audit: Chronological MaxMin Witness Complex", fontsize=13, pad=15, color='#f8fafc', weight='bold')
    ax.set_xlabel("Principal Component 1", color='#94a3b8')
    ax.set_ylabel("Principal Component 2", color='#94a3b8')
    ax.grid(True, linestyle=':', alpha=0.3, color='#475569')
    ax.legend(loc='lower right', facecolor='#0f172a', edgecolor='#334155', fontsize=9.5)

    plt.tight_layout()
    plt.savefig("zone3_witness_final_clean.png", dpi=300)
    print("Successfully saved plot to 'zone3_witness_final_clean.png'")
    plt.show()

if __name__ == "__main__":
    generate_and_plot_witness_complex_pristine()
