import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# REAL BLOCK-LOCAL LANCZOS + RITZ PROJECTION OPTIMIZER
# =====================================================================
class VerifiedBlockKrylovNewton:
    def __init__(
        self, 
        model, 
        lr_kn=0.40, 
        damping=0.10, 
        k_per_block=4, 
        trust_radius=0.50,
        mode="projection", # "projection" (zero out <=0) or "clamp" (floor to min_eig)
        min_eig=1e-3
    ):
        self.model = model
        self.lr_kn = lr_kn
        self.damping = damping
        self.k_per_block = k_per_block
        self.trust_radius = trust_radius
        self.mode = mode
        self.min_eig = min_eig

        # Partition parameters into layer blocks
        self.block_params = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            block_key = name.split('.')[0] if '.' in name else 'head'
            if block_key not in self.block_params:
                self.block_params[block_key] = []
            self.block_params[block_key].append((name, param))

    def compute_block_hvp(self, loss, block_p_list, v_flat):
        """Pearlmutter exact HVP for a specific block."""
        grads = torch.autograd.grad(loss, block_p_list, create_graph=True, retain_graph=True)
        g_flat = torch.cat([g.flatten() for g in grads])
        
        grad_v = torch.dot(g_flat, v_flat)
        hvp_grads = torch.autograd.grad(grad_v, block_p_list, retain_graph=True)
        return torch.cat([h.flatten() for h in hvp_grads]).detach()

    def build_real_block_krylov(self, loss, block_p_list):
        """Builds REAL Ritz tridiagonal T_b and basis Q_b via block Lanczos."""
        grads = torch.autograd.grad(loss, block_p_list, retain_graph=True)
        g_flat = torch.cat([g.flatten() for g in grads]).detach()
        
        norm_g = torch.linalg.norm(g_flat)
        if norm_g < 1e-12:
            P_b = g_flat.shape[0]
            return torch.zeros((P_b, 1), device=g_flat.device), torch.zeros((1, 1), device=g_flat.device), g_flat

        v = g_flat / norm_g
        V_cols = [v]
        alphas, betas = [], []

        v_prev = torch.zeros_like(v)
        beta_prev = 0.0

        for j in range(self.k_per_block):
            w = self.compute_block_hvp(loss, block_p_list, V_cols[-1])
            alpha = torch.dot(V_cols[-1], w)
            alphas.append(float(alpha))

            w_proj = w - alpha * V_cols[-1] - beta_prev * v_prev
            beta = torch.linalg.norm(w_proj)
            betas.append(float(beta))

            if beta < 1e-8 or j == self.k_per_block - 1:
                break

            v_prev = V_cols[-1]
            beta_prev = beta
            V_cols.append(w_proj / beta)

        Q_b = torch.stack(V_cols, dim=1)
        m = len(alphas)
        T_b = torch.zeros((m, m), device=g_flat.device)
        for i in range(m):
            T_b[i, i] = alphas[i]
            if i > 0:
                T_b[i, i - 1] = betas[i - 1]
                T_b[i - 1, i] = betas[i - 1]

        return Q_b, T_b, g_flat

    def invert_operator(self, T_b):
        """Inverts real tridiagonal T_b using either Truncated Projection or Hard Clamping."""
        T_damped = T_b + self.damping * torch.eye(T_b.shape[0], device=T_b.device)
        eigvals, eigvecs = torch.linalg.eigh(T_damped)

        if self.mode == "projection":
            # ZERO OUT non-positive/indefinite curvature modes (Pseudoinverse)
            inv_eig = torch.where(
                eigvals > self.min_eig, 
                1.0 / eigvals, 
                torch.zeros_like(eigvals)
            )
        else: # "clamp"
            # HARD FLOOR CLAMPING (Pathology under indefinite spectrum)
            clamped_eig = torch.clamp(eigvals, min=self.min_eig)
            inv_eig = 1.0 / clamped_eig

        return eigvecs @ torch.diag(inv_eig) @ eigvecs.T, eigvals

    def step(self, loss):
        total_tau_num = 0.0
        total_tau_den = 0.0

        for block_key, param_tuples in self.block_params.items():
            block_p_list = [p for _, p in param_tuples]
            
            # 1. Compute REAL Lanczos Subspace & Tridiagonal Hessian Operator
            Q_b, T_b, g_b = self.build_real_block_krylov(loss, block_p_list)

            # 2. Invert using selected strategy
            T_b_inv, raw_eigvals = self.invert_operator(T_b)

            # 3. Compute Newton Update
            rhs = Q_b.T @ g_b
            step_parallel = Q_b @ (T_b_inv @ rhs)

            # 4. Measure Frame Torsion τ_L = ||(I - Q Q^T) Δθ|| / ||Q Q^T Δθ||
            q_proj = Q_b @ (Q_b.T @ step_parallel)
            q_ortho = step_parallel - q_proj
            
            total_tau_num += torch.linalg.norm(q_ortho).item() ** 2
            total_tau_den += torch.linalg.norm(q_proj).item() ** 2

            # 5. Trust Region Safety Clip
            norm_step = torch.linalg.norm(step_parallel)
            norm_g = torch.linalg.norm(g_b) + 1e-12
            if norm_step / norm_g > self.trust_radius:
                step_parallel = step_parallel * (self.trust_radius * norm_g / norm_step)

            # 6. Apply Parameters
            idx = 0
            for _, p in param_tuples:
                numel = p.numel()
                p.data.add_(-self.lr_kn * step_parallel[idx:idx + numel].view_as(p.data))
                idx += numel

        tau_L = math.sqrt(total_tau_num) / (math.sqrt(total_tau_den) + 1e-12)
        return tau_L
