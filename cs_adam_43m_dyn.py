import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. TRANSFORMER MODEL ARCHITECTURE
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
    def __init__(self, vocab_size=256, d_model=64, num_layers=4):
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
# 2. ADAPTIVE BLOCK-LOCAL KRYLOV-NEWTON OPTIMIZER
# =====================================================================
class AdaptiveBlockKrylovNewton:
    def __init__(
        self, 
        model, 
        lr_kn=0.40, 
        initial_damping=0.10, 
        min_damping=1e-4,
        max_damping=1e2,
        k_per_block=8, 
        trust_radius=0.50,
        target_clip_min=0.15,
        target_clip_max=0.35,
        window_size=10
    ):
        self.model = model
        self.lr_kn = lr_kn
        self.damping = initial_damping
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.k_per_block = k_per_block
        self.trust_radius = trust_radius
        self.target_clip_min = target_clip_min
        self.target_clip_max = target_clip_max
        self.window_size = window_size

        # Partition parameters by block
        self.block_params = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            block_key = name.split('.')[0] if '.' in name else 'head'
            if block_key not in self.block_params:
                self.block_params[block_key] = []
            self.block_params[block_key].append((name, param))

        self.step_history = []  # Tracking boolean clips per step
        self.step_count = 0

    def compute_block_hvp(self, loss, block_p_list, v_flat):
        """Pearlmutter exact Hessian-Vector Product for a single block."""
        grads = torch.autograd.grad(loss, block_p_list, create_graph=True, retain_graph=True)
        g_flat = torch.cat([g.flatten() for g in grads])
        
        grad_v = torch.dot(g_flat, v_flat)
        hvp_grads = torch.autograd.grad(grad_v, block_p_list, retain_graph=True)
        return torch.cat([h.flatten() for h in hvp_grads]).detach()

    def build_block_krylov(self, loss, block_p_list):
        """Executes Block-Local Lanczos to construct Q_b and T_b (k=8)."""
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

    def adjust_damping_controller(self):
        """Updates λ dynamically based on trailing window clip rate."""
        if len(self.step_history) < 3:
            return

        recent_window = self.step_history[-self.window_size:]
        clip_rate = sum(recent_window) / len(recent_window)

        if clip_rate > self.target_clip_max:
            # Step too aggressive / frequently clipped -> Increase Damping
            self.damping = min(self.damping * 1.5, self.max_damping)
        elif clip_rate < self.target_clip_min:
            # Step under-utilized / rarely clipped -> Relax Damping
            self.damping = max(self.damping / 1.2, self.min_damping)

    def step(self, loss):
        self.step_count += 1
        any_block_clipped = False
        max_rel_step = 0.0

        for block_key, param_tuples in self.block_params.items():
            block_p_list = [p for _, p in param_tuples]
            grads = torch.autograd.grad(loss, block_p_list, retain_graph=True)
            g_b = torch.cat([g.flatten() for g in grads]).detach()

            # 1. Local Block Lanczos Subspace
            Q_b, T_b = self.build_block_krylov(loss, block_p_list)

            # 2. Curvature Inversion with Active Damping λ
            T_b_damped = T_b + self.damping * torch.eye(T_b.shape[0], device=T_b.device)
            eigvals, eigvecs = torch.linalg.eigh(T_b_damped)
            eigvals_clamped = torch.clamp(eigvals, min=1e-4)
            T_b_inv = eigvecs @ torch.diag(1.0 / eigvals_clamped) @ eigvecs.T

            # 3. Newton Update Vector
            rhs = Q_b.T @ g_b
            step_b = Q_b @ (T_b_inv @ rhs)

            # 4. Trust Radius Check
            norm_step = torch.linalg.norm(step_b)
            norm_g = torch.linalg.norm(g_b) + 1e-12
            rel_step = (norm_step / norm_g).item()
            max_rel_step = max(max_rel_step, rel_step)

            if rel_step > self.trust_radius:
                step_b = step_b * (self.trust_radius / rel_step)
                any_block_clipped = True

            # 5. Apply Parameters
            idx = 0
            for _, p in param_tuples:
                numel = p.numel()
                p.data.add_(-self.lr_kn * step_b[idx:idx + numel].view_as(p.data))
                idx += numel

        # Record clip status for current step
        self.step_history.append(1 if any_block_clipped else 0)

        # Execute Dynamic Controller
        self.adjust_damping_controller()

        recent_window = self.step_history[-self.window_size:]
        rolling_clip_rate = sum(recent_window) / len(recent_window)

        return rolling_clip_rate, self.damping, max_rel_step

# =====================================================================
# 3. RUNTIME DIAGNOSTICS & VERIFICATION
# =====================================================================
def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ScaledTransformer(vocab_size=256, d_model=64, num_layers=4).to(device)
    P_total = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {P_total:,}")

    opt = AdaptiveBlockKrylovNewton(
        model, 
        lr_kn=0.40, 
        initial_damping=0.10, 
        k_per_block=8,
        trust_radius=0.50
    )

    x = torch.randint(0, 256, (8, 16), device=device)
    y = torch.randint(0, 256, (8, 16), device=device)

    print("\nExecuting Dynamic Damping Controller Sweep...")
    print(f"{'Step':>6} | {'Loss':>10} | {'Window Clip%':>12} | {'Damping (λ)':>12} | {'Max ||Δθ||/||g||':>16}")
    print("-" * 68)

    for step in range(1, 41):
        model.zero_grad()
        _, loss = model(x, y)
        clip_rate, damping, max_rel_step = opt.step(loss)

        if step % 2 == 0:
            print(f"{step:6d} | {loss.item():10.4f} | {clip_rate:12.1%} | {damping:12.4f} | {max_rel_step:16.2f}")

if __name__ == "__main__":
    main()
