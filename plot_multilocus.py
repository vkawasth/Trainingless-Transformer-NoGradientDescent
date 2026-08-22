import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_multilocus_diagram():
    # 1. Setup Figure and 3D Axis
    fig = plt.figure(figsize=(12, 9), facecolor='#0b0f19')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0b0f19')

    # 2. Generate Grassmannian Surface Mesh (Saddle/Level-Set Geometry)
    u = np.linspace(-1.5, 1.5, 40)
    v = np.linspace(-1.5, 1.5, 40)
    U, V = np.meshgrid(u, v)
    Z = 0.3 * (U**2 - V**2)  # Hyperbolic paraboloid representing level-set curvature

    # Plot Surface
    surf = ax.plot_surface(U, V, Z, cmap='coolwarm', alpha=0.35, edgecolor='none', antialiased=True)

    # 3. Plot Cut Locus Boundary Line (theta_4 ≈ pi/2 ≈ 1.55 rad)
    cut_x = np.linspace(-1.4, 1.4, 100)
    cut_y = np.zeros_like(cut_x)
    cut_z = 0.3 * (cut_x**2)
    ax.plot(cut_x, cut_y, cut_z, color='#00f3ff', linewidth=3.5, label='Cut Locus Boundary (θ₄ ≈ π/2)')

    # 4. Trajectory Approaching Cut Locus (True Path)
    t_in = np.linspace(-1.2, 0, 50)
    traj_x = t_in
    traj_y = 0.8 * (t_in + 1.2)**2 - 0.8
    traj_z = 0.3 * (traj_x**2 - traj_y**2)
    ax.plot(traj_x, traj_y, traj_z, color='#ffd700', linewidth=3, label='True Update Trajectory')

    # 5. Multivalued Logarithm Branches (Geodesic Extrapolation Failure)
    # Branch 1 (Primary Log Branch)
    t_out1 = np.linspace(0, 1.2, 50)
    b1_x = t_out1
    b1_y = 0.5 * t_out1
    b1_z = 0.3 * (b1_x**2 - b1_y**2)
    ax.plot(b1_x, b1_y, b1_z, color='#ff007f', linestyle='--', linewidth=2.5, label='Log Branch 1 (Exp Map)')

    # Branch 2 (Secondary Log Branch due to pi/2 phase flip)
    b2_x = t_out1
    b2_y = -0.7 * t_out1
    b2_z = 0.3 * (b2_x**2 - b2_y**2)
    ax.plot(b2_x, b2_y, b2_z, color='#ff0055', linestyle=':', linewidth=2.5, label='Log Branch 2 (Multi-Valued)')

    # 6. Persistent Principal Axis Vector (theta_1 = 0.28 rad) vs Rotating Frame
    # Origin at trajectory step
    ox, oy, oz = traj_x[25], traj_y[25], traj_z[25]
    
    # Persistent Axis (θ_1)
    ax.quiver(ox, oy, oz, 0.2, 0.4, 0.3, color='#ffd700', linewidth=3, arrow_length_ratio=0.2)
    
    # Non-persistent Rotating Axes (θ_2, θ_3, θ_4 ≈ 1.54 rad)
    ax.quiver(ox, oy, oz, -0.4, 0.2, 0.1, color='#00f3ff', linewidth=1.5, linestyle='--', arrow_length_ratio=0.2)
    ax.quiver(ox, oy, oz, 0.1, -0.5, 0.2, color='#00f3ff', linewidth=1.5, linestyle='--', arrow_length_ratio=0.2)

    # 7. Add Explanatory Callouts & Annotations
    ax.text(0, 0, 0.2, "CUT LOCUS INTERSECTION\n(θ₄ = 1.552 rad ≈ π/2)\nInjectivity Radius Reached", 
            color='#00f3ff', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#0b0f19', edgecolor='#00f3ff'))

    ax.text(0.6, 0.4, 0.4, "MULTIVALUED LOGARITHM\nBranching Explodes k² Error\n(Geodesic Integration Fails)", 
            color='#ff007f', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#0b0f19', edgecolor='#ff007f'))

    ax.text(ox - 0.5, oy - 0.3, oz + 0.3, "PERSISTENT AXIS (θ₁ = 0.279 rad)\n3 Free Rotating Axes (θ₂₋₄ ≈ 1.54 rad)", 
            color='#ffd700', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#0b0f19', edgecolor='#ffd700'))

    # 8. Formatting & Aesthetics
    ax.set_title("Grassmannian Cut Locus Saturation & Trajectory Multi-Valuedness", color='white', fontsize=14, pad=20)
    ax.axis('off')  # Clean minimalist backdrop
    ax.view_init(elev=28, azim=-125)
    
    plt.legend(loc='lower left', facecolor='#0b0f19', edgecolor='white', labelcolor='white', fontsize=9)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_multilocus_diagram()
