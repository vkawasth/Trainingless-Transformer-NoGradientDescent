import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. SYNTHETIC TRANSFORMER ARCHITECTURE
# =====================================================================
V, D, L, T_LEN = 40, 12, 2, 16
K_DIM = 4  # Krylov dimension k = 4

class Blk(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D); self.n2 = nn.LayerNorm(D)
        self.q = nn.Linear(D, D, bias=False); self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False); self.o = nn.Linear(D, D, bias=False)
        self.g = nn.Linear(D, 2 * D, bias=False); self.u = nn.Linear(D, 2 * D, bias=False)
        self.w = nn.Linear(2 * D, D, bias=False)

    def forward(self, x):
        h = self.n1(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        attn = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(D)
                         + torch.triu(torch.full((x.shape[1], x.shape[1]), -1e9), 1), dim=-1)
        x = x + self.o(attn @ v)
        h = self.n2(x)
        return x + self.w(F.silu(self.g(h)) * self.u(h))

class SyntheticTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = nn.Embedding(V, D); self.pe = nn.Embedding(T_LEN, D)
        self.b = nn.ModuleList([Blk() for _ in range(L)])
        self.nf = nn.LayerNorm(D)

    def forward(self, x, y):
        h = self.te(x) + self.pe(torch.arange(x.shape[1], device=x.device))
        for blk in self.b:
            h = blk(h)
        lo = self.nf(h) @ self.te.weight.T
        return lo, F.cross_entropy(lo.reshape(-1, V), y.reshape(-1))

rg = np.random.default_rng(1)
rules = {i: [(i * 3 + 1) % V, (i * 5 + 2) % V] for i in range(V)}
s_ = [0]
for _ in range(6000):
    c = s_[-1]
    s_.append(int(rules[c][0] if (len(s_) > 1 and s_[-2] % 2 == 0) else rules[c][1]))
seq = np.array(s_)

def get_batch(n=24):
    i = rg.integers(0, len(seq) - T_LEN - 1, n)
    return (torch.tensor(np.stack([seq[j:j + T_LEN] for j in i])),
            torch.tensor(np.stack([seq[j + 1:j + T_LEN + 1] for j in i])))

# =====================================================================
# 2. ADAPTIVE KRYLOV-NEWTON OPTIMIZER (Tikhonov Regularized \lambda=1e-2)
# =====================================================================
class AdaptiveKrylovSubspaceSignAdam:
    def __init__(
        self, 
        params, 
        lr_kn=0.50,               # High Newton step scale
        lr_ortho_max=2e-3,        # Initial complement scale
        lr_ortho_min=8e-4,        # Floor scale (8e-4)
        beta_ema=0.75,            # H* = 4 EMA gradient filter
        beta1_ortho=0.95,         # High-smoothing momentum
        beta2_ortho=0.999, 
        eps=1e-8, 
        k_dim=4,                  # k = 4
        soft_floor_ratio=0.50,    # Soft floor trigger
        tikhonov_lambda=1e-2      # Tikhonov regularization parameter
    ):
        self.params = list(params)
        self.P = sum(p.numel() for p in self.params)
        self.lr_kn = lr_kn
        self.lr_ortho_max = lr_ortho_max
        self.lr_ortho_min = lr_ortho_min
        self.beta_ema = beta_ema
        self.beta1_ortho = beta1_ortho
        self.beta2_ortho = beta2_ortho
        self.eps = eps
        self.k_dim = k_dim
        self.soft_floor_ratio = soft_floor_ratio
        self.tikhonov_lambda = tikhonov_lambda

        self.g_ema = torch.zeros(self.P)
        self.m_perp = torch.zeros(self.P)
        self.v_perp = torch.zeros(self.P)
        self.t = 0

    def _get_flat_params(self):
        return torch.cat([p.data.flatten() for p in self.params])

    def _set_flat_params(self, flat_p):
        idx = 0
        for p in self.params:
            numel = p.numel()
            p.data.copy_(flat_p[idx:idx + numel].view_as(p.data))
            idx += numel

    def _get_flat_grads(self):
        return torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros_like(p.data.flatten())) for p in self.params])

    def update_ema_grad(self, g_raw):
        self.g_ema = self.beta_ema * self.g_ema + (1.0 - self.beta_ema) * g_raw
        return self.g_ema

    def get_current_lr_ortho(self, total_steps=120):
        """Cosine decay schedule for lr_ortho down to 8e-4."""
        if self.t >= total_steps:
            return self.lr_ortho_min
        progress = self.t / float(total_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_ortho_min + (self.lr_ortho_max - self.lr_ortho_min) * cosine_decay

    def step(self, Q_K, T_K=None, total_steps=120):
        self.t += 1
        g_raw = self._get_flat_grads()
        p = self._get_flat_params()

        # Update EMA Gradient Seed
        g_seeded = self.update_ema_grad(g_raw)

        # 1. In-Subspace Projection
        g_parallel = Q_K @ (Q_K.T @ g_seeded)

        # 2. Subspace Newton Step with Tikhonov Regularization: (T_K + \lambda I)^-1
        if T_K is not None:
            rhs = Q_K.T @ g_seeded
            damped_T_K = T_K + self.tikhonov_lambda * torch.eye(T_K.shape[0], device=T_K.device)
            eigvals, eigvecs = torch.linalg.eigh(damped_T_K)
            T_K_inv = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T
            step_kn = Q_K @ (T_K_inv @ rhs)
        else:
            step_kn = g_parallel

        # 3. Orthogonal Complement Separation
        g_perp = g_seeded - g_parallel

        # 4. Decoupled Orthogonal Momentum Filtering (\beta_{1,\perp} = 0.95)
        self.m_perp = self.beta1_ortho * self.m_perp + (1.0 - self.beta1_ortho) * g_perp
        self.v_perp = self.beta2_ortho * self.v_perp + (1.0 - self.beta2_ortho) * (g_perp ** 2)

        m_hat = self.m_perp / (1.0 - self.beta1_ortho ** self.t)
        v_hat = self.v_perp / (1.0 - self.beta2_ortho ** self.t)

        step_ortho = torch.sign(m_hat / (torch.sqrt(v_hat) + self.eps))

        # Compute Cosine Decayed lr_ortho for Current Step
        current_lr_ortho = self.get_current_lr_ortho(total_steps)

        # 5. Compute Scaled Updates
        kn_vec = self.lr_kn * step_kn
        ortho_vec = current_lr_ortho * step_ortho

        norm_kn = float(torch.linalg.norm(kn_vec))
        norm_ortho_raw = float(torch.linalg.norm(ortho_vec)) + 1e-30
        raw_ratio = norm_kn / norm_ortho_raw

        # Dynamic Soft-Floor Norm Scheduler (floor = 0.50)
        if raw_ratio < self.soft_floor_ratio:
            max_ortho_norm = norm_kn / self.soft_floor_ratio
            ortho_vec = ortho_vec * (max_ortho_norm / norm_ortho_raw)

        norm_ortho_final = float(torch.linalg.norm(ortho_vec)) + 1e-30
        final_norm_ratio = norm_kn / norm_ortho_final

        total_update = kn_vec + ortho_vec
        self._set_flat_params(p - total_update)

        return total_update, g_raw, kn_vec, ortho_vec, final_norm_ratio, current_lr_ortho

# =====================================================================
# 3. MANIFOLD TRACKER & DIAGNOSTICS (k=4)
# =====================================================================
class UnifiedManifoldTracker:
    def __init__(self, model, block_indices, k_dim=4):
        self.model = model
        self.block_indices = block_indices
        self.k = k_dim
        self.prev_frames = {}
        self.prev_grad = None

    def build_frame(self, update_window, idxs):
        A = torch.stack([u[idxs] for u in update_window], dim=1)
        U, _, _ = torch.linalg.svd(A, full_matrices=False)
        return U[:, :self.k]

    def gr_distance(self, Q1, Q2):
        sv = torch.linalg.svdvals(Q1.T @ Q2).cpu().numpy()
        sv = np.clip(sv, 0.0, 1.0)
        return float(np.sqrt((np.arccos(sv) ** 2).sum()))

    def diagnose(self, step, loss_val, update_window, current_grad, full_update, kn_vec, ortho_vec):
        P_total = current_grad.shape[0]
        current_frames = {b_name: self.build_frame(update_window, idxs) 
                          for b_name, idxs in self.block_indices.items()}

        rotations = {}
        omega_norms = {}
        if self.prev_frames:
            for b_name in self.block_indices:
                Q_prev, Q_curr = self.prev_frames[b_name], current_frames[b_name]
                rotations[b_name] = self.gr_distance(Q_prev, Q_curr)
                Omega = Q_prev.T @ (Q_curr - Q_prev)
                omega_norms[b_name] = float(torch.linalg.norm(Omega))

        proj_energy = 0.0
        total_energy = float((current_grad * current_grad).sum()) + 1e-30
        for b_name, idxs in self.block_indices.items():
            gb = current_grad[idxs]
            Qb = current_frames[b_name]
            pb = Qb @ (Qb.T @ gb)
            proj_energy += float((pb * pb).sum())
        gamma = 1.0 - (proj_energy / total_energy)

        num_blocks = len(self.block_indices)
        Q_glob = torch.zeros(P_total, num_blocks * self.k, device=current_grad.device)
        col_idx = 0
        for b_name, idxs in self.block_indices.items():
            Qb = current_frames[b_name]
            Q_glob[idxs, col_idx : col_idx + self.k] = Qb
            col_idx += self.k
        Q_glob = torch.linalg.qr(Q_glob)[0]

        u_proj = Q_glob @ (Q_glob.T @ full_update)
        denom = -float((current_grad * full_update).sum()) + 1e-30
        rho_F = -float((current_grad * u_proj).sum()) / denom

        if self.prev_grad is not None:
            S_t = float((torch.sign(current_grad) == torch.sign(self.prev_grad)).float().mean())
        else:
            S_t = 1.0

        P_ortho = torch.eye(P_total, device=current_grad.device) - Q_glob @ Q_glob.T
        ortho_leakage = P_ortho @ ortho_vec
        in_subspace_norm = torch.linalg.norm(Q_glob @ (Q_glob.T @ kn_vec)) + 1e-30
        tau_L = float((torch.linalg.norm(ortho_leakage) / in_subspace_norm) ** 2)

        self.prev_frames = current_frames
        self.prev_grad = current_grad.clone()

        return {
            "step": step, "loss": loss_val, "gamma": gamma, "rho_F": rho_F,
            "S_t": S_t, "tau_L": tau_L, "rotations": rotations,
            "omega_norms": omega_norms, "Q_glob": Q_glob
        }

# =====================================================================
# 4. EXECUTION LOOP
# =====================================================================
def main():
    torch.manual_seed(0)
    TOTAL_STEPS = 120
    model = SyntheticTransformer()
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    span = {}; i = 0
    for nm, p in model.named_parameters():
        span[nm] = (i, i + p.numel()); i += p.numel()

    block_indices = {"ATTN0": [], "FF0": [], "ATTN1": [], "FF1": []}
    for nm, (a, b) in span.items():
        if "b.0." in nm and any(k in nm for k in ["q", "k", "v", "o"]):
            block_indices["ATTN0"].append(torch.arange(a, b))
        elif "b.0." in nm and any(k in nm for k in ["g", "u", "w"]):
            block_indices["FF0"].append(torch.arange(a, b))
        elif "b.1." in nm and any(k in nm for k in ["q", "k", "v", "o"]):
            block_indices["ATTN1"].append(torch.arange(a, b))
        elif "b.1." in nm and any(k in nm for k in ["g", "u", "w"]):
            block_indices["FF1"].append(torch.arange(a, b))

    block_indices = {k: torch.cat(v) for k, v in block_indices.items() if len(v) > 0}

    # Config: lr_kn = 0.50, lr_ortho = 2e-3 -> 8e-4 (Cosine), k = 4, lambda = 1e-2
    opt = AdaptiveKrylovSubspaceSignAdam(
        params, 
        lr_kn=0.50, 
        lr_ortho_max=2e-3, 
        lr_ortho_min=8e-4, 
        beta_ema=0.75, 
        beta1_ortho=0.95, 
        k_dim=K_DIM,
        soft_floor_ratio=0.50,
        tikhonov_lambda=1e-2
    )
    tracker = UnifiedManifoldTracker(model, block_indices, k_dim=K_DIM)

    print("=" * 125)
    print(f" INTEGRATED FISHER-GRASSMANN OPTIMIZER (P={P}, k={K_DIM}) | TIKHONOV REGULARIZATION (λ = 1e-2)")
    print(" Config: lr_kn = 0.50 | lr_ortho = 2e-3 -> 8e-4 (Cosine) | floor = 0.50 | lambda = 1e-2")
    print("=" * 125)
    print(f"{'Step':>6} | {'Loss':>8} | {'lr_ortho':>10} | {'Gamma (γ)':>10} | {'Rho_F (ρ)':>10} | {'Sign S_t':>8} | {'Torsion τ_L':>11} | {'||Δθ_||/||Δθ_⊥||':>15}")
    print("-" * 125)

    update_window = []
    WINDOW_SIZE = 8

    K_total = len(block_indices) * K_DIM
    Q_K = torch.randn(P, K_total)
    Q_K = torch.linalg.qr(Q_K)[0]

    for step in range(1, TOTAL_STEPS + 1):
        x, y = get_batch()
        model.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        loss_val = float(loss)

        T_K = torch.diag(torch.linspace(2.0, 0.5, K_total))

        full_update, g_flat, kn_vec, ortho_vec, norm_ratio, current_lr_ortho = opt.step(Q_K, T_K, total_steps=TOTAL_STEPS)

        update_window.append(full_update)
        if len(update_window) > WINDOW_SIZE:
            update_window.pop(0)

        if step >= WINDOW_SIZE and step % 10 == 0:
            diag = tracker.diagnose(step, loss_val, update_window, g_flat, full_update, kn_vec, ortho_vec)
            Q_K = diag["Q_glob"]

            print(f"{step:6d} | {loss_val:8.4f} | {current_lr_ortho:10.2e} | {diag['gamma']:10.4f} | {diag['rho_F']:10.4f} | {diag['S_t']:8.4f} | {diag['tau_L']:11.4e} | {norm_ratio:15.4f}")

    print("-" * 125)

if __name__ == "__main__":
    main()
