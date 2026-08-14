import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. 6-LAYER TRANSFORMER MODEL (4.3M PARAMETER SCALE COMPATIBLE)
# =====================================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.g = nn.Linear(d_model, 2 * d_model, bias=False)
        self.u = nn.Linear(d_model, 2 * d_model, bias=False)
        self.w = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, x):
        h = self.n1(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        attn = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(x.shape[-1]), dim=-1)
        x = x + self.o(attn @ v)
        h = self.n2(x)
        return x + self.w(F.silu(self.g(h)) * self.u(h))

class ScaledTransformer(nn.Module):
    def __init__(self, vocab_size=1024, d_model=128, num_layers=6):
        super().__init__()
        self.te = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model) for _ in range(num_layers)])
        self.nf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, y):
        h = self.te(x)
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(self.nf(h))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        return logits, loss

# =====================================================================
# 2. BLOCK-LOCAL KRYLOV-NEWTON OPTIMIZER (Pearlmutter HVP + Clip Rate Tracker)
# =====================================================================
class BlockLocalKrylovNewtonAdam:
    def __init__(
        self, 
        model, 
        lr_kn=0.40, 
        damping=0.10, 
        k_per_block=8, 
        trust_radius=0.50
    ):
        self.model = model
        self.lr_kn = lr_kn
        self.damping = damping
        self.k_per_block = k_per_block
        self.trust_radius = trust_radius
        
        # Partition parameters into block groups
        self.block_params = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Group by layer / embedding / head
            block_key = name.split('.')[0] if '.' in name else 'head'
            if block_key not in self.block_params:
                self.block_params[block_key] = []
            self.block_params[block_key].append((name, param))

        self.step_count = 0
        self.clipped_steps = 0

    def compute_block_hvp(self, loss, block_p_list, v_flat):
        """Pearlmutter exact Hessian-Vector Product for a single block."""
        grads = torch.autograd.grad(loss, block_p_list, create_graph=True, retain_graph=True)
        g_flat = torch.cat([g.flatten() for g in grads])
        
        grad_v = torch.dot(g_flat, v_flat)
        hvp_grads = torch.autograd.grad(grad_v, block_p_list, retain_graph=True)
        return torch.cat([h.flatten() for h in hvp_grads]).detach()

    def build_block_krylov(self, loss, block_p_list):
        """Executes Block-Local Lanczos to build Q_b and T_b (k=8)."""
        grads = torch.autograd.grad(loss, block_p_list, retain_graph=True)
        g_flat = torch.cat([g.flatten() for g in grads]).detach()
        
        norm_g = torch.linalg.norm(g_flat)
        if norm_g < 1e-12:
            P_b = g_flat.shape[0]
            return torch.zeros((P_b, 1), device=g_flat.device), torch.ones((1, 1), device=g_flat.device)

        v = g_flat / norm_g
        V_cols = [v]
        alphas, betas = [], []

        v_prev = torch.zeros_like(v)
        beta_prev = 0.0

        for j in range(self.k_per_block):
            # Block-wise HVP
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

        return Q_b, T_b

    def step(self, loss):
        self.step_count += 1
        clipped = False

        for block_key, param_tuples in self.block_params.items():
            block_p_list = [p for _, p in param_tuples]
            grads = torch.autograd.grad(loss, block_p_list, retain_graph=True)
            g_b = torch.cat([g.flatten() for g in grads]).detach()

            # 1. Local Block-Lanczos Subspace
            Q_b, T_b = self.build_block_krylov(loss, block_p_list)

            # 2. Curvature Inversion with Damping
            T_b_damped = T_b + self.damping * torch.eye(T_b.shape[0], device=T_b.device)
            eigvals, eigvecs = torch.linalg.eigh(T_b_damped)
            eigvals_clamped = torch.clamp(eigvals, min=1e-4)
            T_b_inv = eigvecs @ torch.diag(1.0 / eigvals_clamped) @ eigvecs.T

            # 3. Block Newton Update
            rhs = Q_b.T @ g_b
            step_b = Q_b @ (T_b_inv @ rhs)

            # 4. Trust Radius Safety Clip
            norm_step = torch.linalg.norm(step_b)
            norm_g = torch.linalg.norm(g_b) + 1e-12
            
            if norm_step / norm_g > self.trust_radius:
                step_b = step_b * (self.trust_radius * norm_g / norm_step)
                clipped = True

            # Apply parameter update
            idx = 0
            for _, p in param_tuples:
                numel = p.numel()
                p.data.add_(-self.lr_kn * step_b[idx:idx + numel].view_as(p.data))
                idx += numel

        if clipped:
            self.clipped_steps += 1

        return self.get_clip_rate()

    def get_clip_rate(self):
        return (self.clipped_steps / self.t) if hasattr(self, 't') and self.t > 0 else (self.clipped_steps / self.step_count)

# =====================================================================
# 3. EXECUTION DEMO & DIAGNOSTICS
# =====================================================================
def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ScaledTransformer(vocab_size=256, d_model=64, num_layers=4).to(device)
    P_total = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {P_total:,}")

    opt = BlockLocalKrylovNewtonAdam(model, lr_kn=0.40, damping=0.10, k_per_block=8)

    x = torch.randint(0, 256, (8, 16), device=device)
    y = torch.randint(0, 256, (8, 16), device=device)

    print("\nStarting Training & Damping Clip-Rate Health Tracking...")
    print(f"{'Step':>6} | {'Loss':>12} | {'Clip Rate':>12} | {'Damping (λ)':>12}")
    print("-" * 52)

    for step in range(1, 21):
        model.zero_grad()
        _, loss = model(x, y)
        clip_rate = opt.step(loss)
        
        if step % 5 == 0:
            print(f"{step:6d} | {loss.item():12.4f} | {clip_rate:12.1%} | {opt.damping:12.2f}")

if __name__ == "__main__":
    main()
