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
# 2. ADAPTIVE KRYLOV-NEWTON OPTIMIZER (Hard Clamped min_eig=1e-4)
# =====================================================================
class AdaptiveKrylovSubspaceSignAdam:
    def __init__(
        self, 
        params, 
        lr_kn=0.50,               # Configurable Newton scale
        lr_ortho_max=2e-3,        # Initial complement scale
        lr_ortho_min=8e-4,        # Floor scale (8e-4)
        beta_ema=0.75,            # H* = 4 EMA gradient filter
        beta1_ortho=0.95,         # High-smoothing momentum
        beta2_ortho=0.999, 
        eps=1e-8, 
        k_dim=4,                  # k = 4
        soft_floor_ratio=0.50,    # Soft floor trigger
        eig_clamp_min=1e-4        # Hard eigenvalue clamping floor
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
        self.eig_clamp_min = eig_clamp_min

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
        if self.t >= total_steps:
            return self.lr_ortho_min
        progress = self.t / float(total_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_ortho_min + (self.lr_ortho_max - self.lr_ortho_min) * cosine_decay

    def step(self, Q_K, T_K=None, total_steps=120):
        self.t += 1
        g_raw = self._get_flat_grads()
        p = self._get_flat_params()

        g_seeded = self.update_ema_grad(g_raw)

        # 1. In-Subspace Projection
        g_parallel = Q_K @ (Q_K.T @ g_seeded)

        # 2. Subspace Newton Step with Hard Eigenvalue Clamping (1e-4)
        if T_K is not None:
            rhs = Q_K.T @ g_seeded
            eigvals, eigvecs = torch.linalg.eigh(T_K)
            eigvals_clamped = torch.clamp(eigvals, min=self.eig_clamp_min)
            T_K_inv = eigvecs @ torch.diag(1.0 / eigvals_clamped) @ eigvecs.T
            step_kn = Q_K @ (T_K_inv @ rhs)
        else:
            step_kn = g_parallel

        # 3. Orthogonal Complement Separation
        g_perp = g_seeded - g_parallel

        # 4. Decoupled Orthogonal Momentum Filtering
        self.m_perp = self.beta1_ortho * self.m_perp + (1.0 - self.beta1_ortho) * g_perp
        self.v_perp = self.beta2_ortho * self.v_perp + (1.0 - self.beta2_ortho) * (g_perp ** 2)

        m_hat = self.m_perp / (1.0 - self.beta1_ortho ** self.t)
        v_hat = self.v_perp / (1.0 - self.beta2_ortho ** self.t)

        step_ortho = torch.sign(m_hat / (torch.sqrt(v_hat) + self.eps))

        current_lr_ortho = self.get_current_lr_ortho(total_steps)

        # 5. Compute Scaled Updates & Norm-Ratio Coupling
        kn_vec = self.lr_kn * step_kn
        ortho_vec = current_lr_ortho * step_ortho

        norm_kn = float(torch.linalg.norm(kn_vec))
        norm_ortho_raw = float(torch.linalg.norm(ortho_vec)) + 1e-30
        raw_ratio = norm_kn / norm_ortho_raw

        if raw_ratio < self.soft_floor_ratio:
            max_ortho_norm = norm_kn / self.soft_floor_ratio
            ortho_vec = ortho_vec * (max_ortho_norm / norm_ortho_raw)

        norm_ortho_final = float(torch.linalg.norm(ortho_vec)) + 1e-30
        final_norm_ratio = norm_kn / norm_ortho_final

        total_update = kn_vec + ortho_vec
        self._set_flat_params(p - total_update)

        return total_update, g_raw, kn_vec, ortho_vec, final_norm_ratio, current_lr_ortho

# =====================================================================
# 3. MANIFOLD TRACKER (k=4)
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

    def diagnose(self, step, loss_val, update_window, current_grad, full_update, kn_vec, ortho_vec):
        P_total = current_grad.shape[0]
        current_frames = {b_name: self.build_frame(update_window, idxs) 
                          for b_name, idxs in self.block_indices.items()}

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

        P_ortho = torch.eye(P_total, device=current_grad.device) - Q_glob @ Q_glob.T
        ortho_leakage = P_ortho @ ortho_vec
        in_subspace_norm = torch.linalg.norm(Q_glob @ (Q_glob.T @ kn_vec)) + 1e-30
        tau_L = float((torch.linalg.norm(ortho_leakage) / in_subspace_norm) ** 2)

        self.prev_frames = current_frames
        self.prev_grad = current_grad.clone()

        return {
            "step": step, "loss": loss_val, "gamma": gamma, "rho_F": rho_F,
            "tau_L": tau_L, "Q_glob": Q_glob
        }

# =====================================================================
# 4. GRID SWEEP EXECUTION LOOP
# =====================================================================
def run_trial(lr_kn_val, total_steps=120):
    torch.manual_seed(0)
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

    opt = AdaptiveKrylovSubspaceSignAdam(
        params, 
        lr_kn=lr_kn_val, 
        lr_ortho_max=2e-3, 
        lr_ortho_min=8e-4, 
        beta_ema=0.75, 
        beta1_ortho=0.95, 
        k_dim=K_DIM,
        soft_floor_ratio=0.50,
        eig_clamp_min=1e-4
    )
    tracker = UnifiedManifoldTracker(model, block_indices, k_dim=K_DIM)

    update_window = []
    WINDOW_SIZE = 8

    K_total = len(block_indices) * K_DIM
    Q_K = torch.randn(P, K_total)
    Q_K = torch.linalg.qr(Q_K)[0]

    last_diag = None

    for step in range(1, total_steps + 1):
        x, y = get_batch()
        model.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        loss_val = float(loss)

        T_K = torch.diag(torch.linspace(2.0, 0.5, K_total))

        full_update, g_flat, kn_vec, ortho_vec, norm_ratio, _ = opt.step(Q_K, T_K, total_steps=total_steps)

        update_window.append(full_update)
        if len(update_window) > WINDOW_SIZE:
            update_window.pop(0)

        if step >= WINDOW_SIZE and step % 10 == 0:
            last_diag = tracker.diagnose(step, loss_val, update_window, g_flat, full_update, kn_vec, ortho_vec)
            Q_K = last_diag["Q_glob"]

    return last_diag, loss_val

def main():
    lr_kn_list = [0.40, 0.50, 0.60, 0.70]
    results = {}

    print("=" * 105)
    print(f" KRYLOV-NEWTON SCALE SWEEP (k=4, min_eig=1e-4) | lr_ortho: 2e-3 -> 8e-4")
    print("=" * 105)
    print(f"{'lr_kn':>8} | {'Step 120 Loss':>15} | {'Rho_F (ρ)':>12} | {'Gamma (γ)':>12} | {'Torsion τ_L':>15}")
    print("-" * 105)

    for lr in lr_kn_list:
        diag, final_loss = run_trial(lr)
        results[lr] = (final_loss, diag)
        print(f"{lr:8.2f} | {final_loss:15.4f} | {diag['rho_F']:12.4f} | {diag['gamma']:12.4f} | {diag['tau_L']:15.4e}")

    print("-" * 105)

if __name__ == "__main__":
    main()
