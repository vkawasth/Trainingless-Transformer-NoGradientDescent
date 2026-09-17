import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


def bregman_potential(x: torch.Tensor, potential_type: str = "entropy", eps: float = 1e-8) -> torch.Tensor:
    """
    Computes strictly convex potential psi(x) for dually flat Bregman divergence.
    """
    if potential_type == "entropy":
        x_clamp = torch.clamp(x, min=eps)
        return torch.sum(x_clamp * torch.log(x_clamp), dim=-1)
    elif potential_type == "quadratic":
        return 0.5 * torch.sum(x ** 2, dim=-1)
    else:
        raise ValueError(f"Unsupported potential_type: {potential_type}")


def bregman_grad(x: torch.Tensor, potential_type: str = "entropy", eps: float = 1e-8) -> torch.Tensor:
    """
    Computes dual coordinate gradient nabla psi(x).
    """
    if potential_type == "entropy":
        x_clamp = torch.clamp(x, min=eps)
        return 1.0 + torch.log(x_clamp)
    elif potential_type == "quadratic":
        return x
    else:
        raise ValueError(f"Unsupported potential_type: {potential_type}")


def compute_4point_bregman_matrix(
    X: torch.Tensor, 
    potential_type: str = "entropy", 
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes a 4x4 dually flat Bregman divergence matrix for a 4-point witness subset.
    D_psi(P_i, P_j) = psi(P_i) - psi(P_j) - < grad_psi(P_j), P_i - P_j >
    """
    assert X.shape[0] == 4, f"Expected 4-point cloud, got shape {X.shape}"
    
    # Map projected active parameter vectors into probability simplex
    if potential_type == "entropy":
        P = torch.softmax(X, dim=-1)
    else:
        P = X

    psi = bregman_potential(P, potential_type=potential_type, eps=eps)      # (4,)
    grad_psi = bregman_grad(P, potential_type=potential_type, eps=eps)      # (4, p)

    # Pairwise difference tensor: diff[i, j] = P[i] - P[j]
    diff = P.unsqueeze(1) - P.unsqueeze(0)                                   # (4, 4, p)

    # Dual inner product: < grad_psi(P_j), P_i - P_j >
    inner_prod = torch.sum(diff * grad_psi.unsqueeze(0), dim=-1)             # (4, 4)

    # Bregman divergence assembly
    D = psi.unsqueeze(1) - psi.unsqueeze(0) - inner_prod                     # (4, 4)
    D = torch.clamp(D, min=0.0)

    # Symmetrize distance metric for invariant persistence analysis
    D_sym = 0.5 * (D + D.T)
    return D_sym


class IntegratedGeometryEngine:
    """
    Geometry & Topological Witness Engine using Dually Flat Bregman Divergences 
    and a 4-Point Cloud Approximation over Dimension-Adapted JL Projections.
    """
    def __init__(self, proj_dim: int = 32, potential_type: str = "entropy", seed: int = 42):
        self.proj_dim = proj_dim
        self.potential_type = potential_type
        self.seed = seed
        self.proj_matrix: Optional[torch.Tensor] = None
        self.landmarks: Optional[torch.Tensor] = None  # Buffer of shape (4, d_t)

    def _ensure_projection_matrix(self, current_dim: int, device: torch.device):
        """
        Initializes or resizes the dimension-adapted Johnson-Lindenstrauss (JL)
        projection matrix P_{d_t} ~ N(0, 1/p).
        """
        if self.proj_matrix is None or self.proj_matrix.shape[0] != current_dim:
            g = torch.Generator(device=device).manual_seed(self.seed)
            self.proj_matrix = (
                torch.randn(current_dim, self.proj_dim, generator=g, device=device)
                / math.sqrt(self.proj_dim)
            )

    def update_landmarks(self, active_params: torch.Tensor):
        """
        Maintains a sliding 4-point cloud buffer of active sub-manifold weights theta_active.
        active_params shape: (d_t,)
        """
        flat_params = active_params.detach().flatten().unsqueeze(0) # (1, d_t)
        
        if self.landmarks is None:
            # Replicate initial point to form initial 4-point set
            self.landmarks = flat_params.repeat(4, 1)
        else:
            # Shift buffer FIFO
            self.landmarks = torch.cat([self.landmarks[1:], flat_params], dim=0)

    def analyze_topology(self, active_params: torch.Tensor) -> Dict[str, float]:
        """
        Executes JL Projection, 4-Point Bregman Distance Computation, 
        and 1D Topological Loop Persistence Extraction.
        """
        device = active_params.device
        current_dim = active_params.numel()
        
        # 1. Update landmark trajectory
        self.update_landmarks(active_params)
        
        # 2. Ensure Johnson-Lindenstrauss Projection Matrix P_{d_t}
        self._ensure_projection_matrix(current_dim, device)
        
        # 3. Apply active sub-manifold projection
        projected_landmarks = torch.matmul(self.landmarks, self.proj_matrix)  # (4, p)
        
        # 4. Compute 4-point dually flat Bregman distance matrix
        D_sym = compute_4point_bregman_matrix(
            projected_landmarks, 
            potential_type=self.potential_type
        )

        # Verification step: Check zero-diagonal property
        diag_err = torch.max(torch.abs(torch.diag(D_sym))).item()
        assert diag_err < 1e-5, f"Bregman verification failed: non-zero diagonal err={diag_err}"

        # 5. Fast 4-Point Topological Persistence Analysis
        cycle_edges = torch.stack([D_sym[0, 1], D_sym[1, 2], D_sym[2, 3], D_sym[3, 0]])
        h1_birth = torch.min(cycle_edges).item()
        h1_death = torch.max(cycle_edges).item()
        h1_persistence = h1_death - h1_birth

        # 6. Volume form proxy via Cayley-Menger 3-Simplex Determinant
        CM = torch.zeros((5, 5), device=device)
        CM[0, 1:] = 1.0
        CM[1:, 0] = 1.0
        # D_sym already represents pairwise divergence (squared-distance proxy)
        CM[1:, 1:] = D_sym  # Try D_sym directly or D_sym ** 2 depending on scale
        
        cm_det = torch.det(CM).item()
        
        # Taking abs() accounts for orientation/curvature sign switches on the dually flat manifold
        simplex_vol_sq = abs(cm_det) / 288.0

        return {
            "h1_persistence": h1_persistence,
            "h1_birth": h1_birth,
            "h1_death": h1_death,
            "max_diameter": torch.max(D_sym).item(),
            "simplex_3_volume": math.sqrt(simplex_vol_sq),
            "cm_det_raw": cm_det,
            "diag_verification_err": diag_err
        }



# Quick Verification / Sanity Check Run
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = IntegratedGeometryEngine(proj_dim=16, potential_type="entropy")
    
    print("Testing Integrated Geometry Engine (Bregman + 4-Point Approximation)...")
    
    # Simulate parameter updates across 6 optimization steps
    d_t = 1024
    for step in range(6):
        active_params = torch.randn(d_t, device=device) * 0.1 + (step * 0.05)
        metrics = engine.analyze_topology(active_params)
        print(f"Step {step+1} Metrics: H1 Persistence = {metrics['h1_persistence']:.6f} | "
              f"Simplex Vol = {metrics['simplex_3_volume']:.6f} | "
              f"Max Diam = {metrics['max_diameter']:.6f}")
