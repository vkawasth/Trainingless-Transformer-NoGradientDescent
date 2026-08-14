import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# MODEL DEFINITION (P = 3,672 parameters)
# =====================================================================
class TinyModel(nn.Module):
    def __init__(self, d_in=16, d_hidden=64, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_hidden, bias=False)
        self.fc3 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x, y):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.fc3(h)
        return F.cross_entropy(logits, y)

# =====================================================================
# COMPLEMENT FRACTION OPTIMIZER
# =====================================================================
class ComplementFractionKrylovAdam:
    def __init__(
        self, 
        model, 
        lr_kn=0.40, 
        lr_ortho=0.02, 
        damping=0.10, 
        alpha_ortho=1.0, 
        k_per_block=4, 
        trust_radius=0.50,
        min_eig=1e-3
    ):
        self.model = model
        self.lr_kn = lr_kn
        self.lr_ortho = lr_ortho
        self.damping = damping
        self.alpha_ortho = alpha_ortho
        self.k_per_block = k_per_block
        self.trust_radius = trust_radius
        self.min_eig = min_eig

        self.block_params = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            block_key = name.split('.')[0] if '.' in name else 'head'
            if block_key not in self.block_params:
                self.block_params[block_key] = []
            self.block_params[block_key].append((name, param))

    def compute_block_hvp(self, loss, block_p_list, v_flat):
        grads = torch.autograd.grad(loss, block_p_list, create_graph=True, retain_graph=True)
        g_flat = torch.cat([g.flatten() for g in grads])
        grad_v = torch.dot(g_flat, v_flat)
        hvp_grads = torch.autograd.grad(grad_v, block_p_list, retain_graph=True)
        return torch.cat([h.flatten() for h in hvp_grads]).detach()

    def build_real_block_krylov(self, loss, block_p_list):
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
        T_damped = T_b + self.damping * torch.eye(T_b.shape[0], device=T_b.device)
        eigvals, eigvecs = torch.linalg.eigh(T_damped)
        clamped_eig = torch.clamp(eigvals, min=self.min_eig)
        inv_eig = 1.0 / clamped_eig
        return eigvecs @ torch.diag(inv_eig) @ eigvecs.T, eigvals

    def step(self, loss):
        for block_key, param_tuples in self.block_params.items():
            block_p_list = [p for _, p in param_tuples]
            Q_b, T_b, g_b = self.build_real_block_krylov(loss, block_p_list)
            T_b_inv, _ = self.invert_operator(T_b)

            rhs = Q_b.T @ g_b
            step_parallel = Q_b @ (T_b_inv @ rhs)

            norm_step = torch.linalg.norm(step_parallel)
            norm_g = torch.linalg.norm(g_b) + 1e-12
            if norm_step / norm_g > self.trust_radius:
                step_parallel = step_parallel * (self.trust_radius * norm_g / norm_step)

            g_parallel = Q_b @ rhs
            g_ortho = g_b - g_parallel
            step_ortho = torch.sign(g_ortho)

            net_step = (self.lr_kn * step_parallel) + (self.alpha_ortho * self.lr_ortho * step_ortho)

            idx = 0
            for _, p in param_tuples:
                numel = p.numel()
                p.data.add_(-net_step[idx:idx + numel].view_as(p.data))
                idx += numel

# =====================================================================
# BENCHMARK RUNNER
# =====================================================================
def run_bench():
    alpha_grid = [1.0, 0.5, 0.2, 0.05, 0.0]
    seeds = [0, 1, 2]
    
    print(f"{'Alpha (α)':<10} | {'Seed 0':<8} | {'Seed 1':<8} | {'Seed 2':<8} | {'Mean Loss':<10} | {'Std Dev':<10}")
    print("-" * 65)

    for alpha in alpha_grid:
        seed_losses = []
        for s in seeds:
            torch.manual_seed(s)
            model = TinyModel()
            
            # Synthetic task
            torch.manual_seed(999)
            x = torch.randn(32, 16)
            y = torch.randint(0, 10, (32,))
            
            torch.manual_seed(s)
            opt = ComplementFractionKrylovAdam(model, alpha_ortho=alpha)

            for step in range(120):
                model.zero_grad()
                loss = model(x, y)
                opt.step(loss)

            seed_losses.append(loss.item())

        mean_l = float(np.mean(seed_losses))
        std_l = float(np.std(seed_losses))
        print(f"{alpha:<10.2f} | {seed_losses[0]:<8.4f} | {seed_losses[1]:<8.4f} | {seed_losses[2]:<8.4f} | {mean_l:<10.4f} | {std_l:<10.4f}")

if __name__ == "__main__":
    run_bench()
