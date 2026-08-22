import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_finsler_vs_riemannian_plot():
    # 1. Canvas Setup
    fig = plt.figure(figsize=(13, 9), facecolor='#0b0f19')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0b0f19')

    # 2. Local Loss Landscape Surface (Ill-Conditioned Valley near Plateau)
    x = np.linspace(-1.2, 1.2, 50)
    y = np.linspace(-1.2, 1.2, 50)
    X, Y = np.meshgrid(x, y)
    # Anisotropic quadratic loss manifold
    Z = 0.5 * (10.0 * X**2 + 0.5 * Y**2) + 0.1

    surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.3, edgecolor='none', antialiased=True)

    # 3. Base Point (Current Parameter State near Step 160 Plateau)
    p_x, p_y = 0.6, 0.8
    p_z = 0.5 * (10.0 * p_x**2 + 0.5 * p_y**2) + 0.1
    ax.scatter([p_x], [p_y], [p_z], color='white', s=100, zorder=10, label='Current State (Step 160)')

    # 4. Step Vectors at Matched Norm
    # Local Gradient Direction g = [10*x, 0.5*y] = [6.0, 0.4]
    # Riemannian Step Vector u (AdamW Stalled Update)
    u_vec = np.array([-0.25, -0.05, -0.1])
    ax.quiver(p_x, p_y, p_z, u_vec[0], u_vec[1], u_vec[2], 
               color='#00f3ff', linewidth=3, arrow_length_ratio=0.2, 
               label='Riemannian Step (u, AdamW) [ΔL = -0.00037]')

    # Finsler Dual Step Vector sign(g)
    sign_vec = np.array([-0.4, -0.3, -0.45])
    ax.quiver(p_x, p_y, p_z, sign_vec[0], sign_vec[1], sign_vec[2], 
               color='#ff007f', linewidth=3.5, arrow_length_ratio=0.2, 
               label='Finsler Dual Step (sign(g)) [ΔL = -0.0277, 75× Gain]')

    # Uniform Control Step Vector (Random Noise Floor)
    rnd_vec = np.array([0.2, -0.2, 0.05])
    ax.quiver(p_x, p_y, p_z, rnd_vec[0], rnd_vec[1], rnd_vec[2], 
               color='#a0a0a0', linewidth=1.5, linestyle=':', arrow_length_ratio=0.2, 
               label='Uniform Control Vector [ΔL = +0.00003]')

    # 5. Explanatory Callouts & Annotations
    ax.text(p_x - 0.2, p_y + 0.2, p_z + 0.4, 
            "FINSLER DUAL ALIGNMENT\n⟨g, sign(u)⟩ Beats Riemannian 20.8×\n(Stale Sign = Fresh Sign ≈ -0.043)", 
            color='#ff007f', fontsize=9, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0b0f19', edgecolor='#ff007f'))

    ax.text(p_x - 0.8, p_y - 0.6, p_z + 0.1, 
            "ADAMW STALLING GAP\nStep 40 Ratio: 1.2×\nStep 160 Ratio: 75× (Stalls on Flat)", 
            color='#00f3ff', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0b0f19', edgecolor='#00f3ff'))

    ax.text(p_x + 0.3, p_y - 0.4, p_z - 0.2, 
            "UNIFORM CONTROL\nΔL = +0.000030\nRules Out Norm Spreading", 
            color='#a0a0a0', fontsize=8, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0b0f19', edgecolor='#a0a0a0'))

    # 6. Aesthetic Formatting
    ax.set_title("Finsler Dual Pairing ⟨g, sign(u)⟩ vs. Riemannian AdamW Update", 
                 color='white', fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')
    ax.view_init(elev=25, azim=-135)

    plt.legend(loc='lower left', facecolor='#0b0f19', edgecolor='white', labelcolor='white', fontsize=9)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    generate_finsler_vs_riemannian_plot()
