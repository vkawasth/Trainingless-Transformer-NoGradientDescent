import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
import itertools

class BergmanMorseSmaleWitness:
    """
    Constructs a Morse-Smale Witness Complex using a local Bergman Metric 
    over parameter trajectories in Zone 3 (Basin Settle).
    """
    def __init__(self, degree=2, sigma=1.0, alpha=1e-3):
        self.degree = degree
        self.sigma = sigma
        self.alpha = alpha

    def compute_bergman_kernel(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        """
        Computes Bergman Kernel matrix K(X, Y) via standard Gaussian-polynomial 
        reproducing kernel Hilbert space (RKHS) approximation.
        X: [N, D], Y: [M, D]
        """
        if Y is None:
            Y = X
            
        # Distances scaled by variance hyperparameter
        sq_dist = torch.cdist(X, Y, p=2) ** 2
        K_base = torch.exp(-sq_dist / (2 * self.sigma ** 2))
        
        # Polynomial kernel factor to model local Kähler potential geometry
        K_poly = (1 + torch.mm(X, Y.T)) ** self.degree
        
        return K_base * K_poly

    def compute_bergman_distance_matrix(self, X: torch.Tensor, Landmark: torch.Tensor) -> torch.Tensor:
        """
        Calculates Bergman geodesic distance proxy d_K(x, w) = K(x,x) + K(w,w) - 2 Re(K(x,w)).
        X (Witnesses): [N_w, D]
        Landmark (Critical Points): [N_l, D]
        """
        K_XX = torch.diagonal(self.compute_bergman_kernel(X, X))
        K_LL = torch.diagonal(self.compute_bergman_kernel(Landmark, Landmark))
        K_XL = self.compute_bergman_kernel(X, Landmark)

        # Distance proxy under potential function
        D_squared = K_XX.unsqueeze(1) + K_LL.unsqueeze(0) - 2 * K_XL
        D_squared = torch.clamp(D_squared, min=0.0)
        return torch.sqrt(D_squared)

    def extract_morse_landmarks(self, trajectory: torch.Tensor, losses: torch.Tensor, top_k: int = 8) -> torch.Tensor:
        """
        Extracts local critical points (minima/saddles) from Zone 3 trajectory 
        acting as Morse-Smale landmark nodes.
        """
        grads = torch.gradient(losses)[0]
        grad_norms = torch.abs(grads)
        
        # Select local minima/saddle points where gradient norm is minimal
        _, indices = torch.topk(grad_norms, k=min(top_k, len(losses)), largest=False)
        return trajectory[indices]

    def build_witness_complex(self, landmarks: torch.Tensor, witnesses: torch.Tensor, max_dim: int = 2, r_cutoff: float = 1.5):
        """
        Constructs the Morse-Smale Witness Complex up to dimension `max_dim`.
        Returns active simplices and their topological persistence scales.
        """
        N_l = landmarks.size(0)
        N_w = witnesses.size(0)
        
        # Compute Bergman distance matrix between witnesses and landmarks
        D_KW = self.compute_bergman_distance_matrix(witnesses, landmarks) # [N_w, N_l]
        
        # Find nearest landmark indices for each witness
        sorted_dists, sorted_indices = torch.sort(D_KW, dim=1)
        
        simplices = {d: set() for d in range(max_dim + 1)}
        
        # 0-simplices (Vertices)
        for i in range(N_l):
            simplices[0].add((i,))

        # Higher-dimensional simplices driven by witness points
        for w_idx in range(N_w):
            dists = sorted_dists[w_idx]
            indices = sorted_indices[w_idx]
            
            for dim in range(1, max_dim + 1):
                if dim + 1 > N_l:
                    continue
                # Check distance gap for witness acceptance
                if dists[dim] - dists[0] <= r_cutoff:
                    simplex = tuple(sorted(indices[:dim + 1].tolist()))
                    simplices[dim].add(simplex)
                    
        return simplices

# --- Zone 3 Pipeline Integration ---

def run_zone3_witness_analysis(zone3_parameter_history, zone3_loss_history):
    """
    Integrates Bergman Morse-Smale Witness complex directly onto Zone 3 trajectory.
    """
    print("[Zone 3 Topological Audit] Constructing Bergman Morse-Smale Witness Complex...")
    
    # Flatten parameter checkpoints to 2D tensors [Steps, D]
    flat_params = torch.stack([torch.cat([p.flatten() for p in step]) for step in zone3_parameter_history])
    losses = torch.tensor(zone3_loss_history)

    # Initialize Bergman Witness Builder
    bms = BergmanMorseSmaleWitness(degree=2, sigma=2.0)
    
    # 1. Extract Morse Critical Point Landmarks
    landmarks = bms.extract_morse_landmarks(flat_params, losses, top_k=6)
    
    # 2. Extract Witnesses (Trajectory samples within Zone 3)
    witnesses = flat_params
    
    # 3. Compute Complex Topology
    simplices = bms.build_witness_complex(landmarks, witnesses, max_dim=2, r_cutoff=0.8)
    
    print(f"  └─ Extracted Critical Landmarks: {landmarks.size(0)}")
    print(f"  └─ Active 0-Simplices (Nodes): {len(simplices[0])}")
    print(f"  └─ Active 1-Simplices (Edges): {len(simplices[1])}")
    print(f"  └─ Active 2-Simplices (Triangles): {len(simplices[2])}")
    
    return simplices
