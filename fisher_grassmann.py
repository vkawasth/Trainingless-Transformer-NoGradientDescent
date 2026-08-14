import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. SYNTHETIC MODEL & DATA PREPARATION
# =====================================================================
V, D, L, T_LEN = 40, 12, 2, 16
K_DIM = 3

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
# 2. ADAPTIVE KRYLOV-NEWTON + ORTHOGONAL ADAM OPTIMIZER
# =====================================================================
class AdaptiveKrylovSubspaceAdam:
    r"""
    Implements the unified update rule:
        \Delta \theta = (Q_K T_K^{-1} Q_K^T g) + (m_\perp / \sqrt{v_\perp})

    where Q_K T_K^{-1} Q_K^T g is the exact Krylov-Newton step on the low-rank
    subspace Q_K, and m_\perp / \sqrt{v_\perp} is the Adam preconditioned step
    confined strictly to the orthogonal complement.
    """
    def __init__(self, params, lr_kn=1e-2, lr_ortho=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, k_dim=3):
        self.params = list(params)
        self.P = sum(p.numel() for p in self.params)
        self.lr_kn = lr_kn
        self.lr_ortho = lr_ortho
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.k_dim = k_dim

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

    def step(self, Q_K, T_K=None):
        self.t += 1
        g = self._get_flat_grads()
        p = self._get_flat_params()

        # 1. Project gradient into subspace Q_K
        g_parallel = Q_K @ (Q_K.T @ g)

        # 2. Compute Subspace Newton step: Q_K T_K^{-1} Q_K^T g
        if T_K is not None:
            rhs = Q_K.T @ g
            y = torch.linalg.solve(T_K + 1e-4 * torch.eye(T_K.shape[0]), rhs)
            step_kn = Q_K @ y
        else:
            step_kn = g_parallel

        # 3. Separate orthogonal complement: g_perp = g - g_parallel
        g_perp = g - g_parallel

        # 4. Adam preconditioning on the orthogonal complement
        self.m_perp = self.beta1 * self.m_perp + (1.0 - self.beta1) * g_perp
        self.v_perp = self.beta2 * self.v_perp + (1.0 - self.beta2) * (g_perp ** 2)

        m_hat = self.m_perp / (1.0 - self.beta1 ** self.t)
        v_hat = self.v_perp / (1.0 - self.beta2 ** self.t)

        step_ortho = m_hat / (torch.sqrt(v_hat) + self.eps)

        # 5. Combined parameter update
        total_update = (self.lr_kn * step_kn) + (self.lr_ortho * step_ortho)
        self._set_flat_params(p - total_update)
        return total_update, g

# =====================================================================
# 3. UNIFIED FISHER-GRASSMANN MANIFOLD TRACKER
# =====================================================================
class UnifiedManifoldTracker:
    def __init__(self, model, block_indices, k_dim=3):
        self.model = model
        self.block_indices = block_indices
        self.k = k_dim
        self.prev_frames = {}

    def build_frame(self, update_window, idxs):
        A = torch.stack([u[idxs] for u in update_window], dim=1)
        U, _, _ = torch.linalg.svd(A, full_matrices=False)
        return U[:, :self.k]

    def gr_distance(self, Q1, Q2):
        sv = torch.linalg.svdvals(Q1.T @ Q2).cpu().numpy()
        sv = np.clip(sv, 0.0, 1.0)
        return float(np.sqrt((np.arccos(sv) ** 2).sum()))

    def face_alignment(self, Q1, Q2):
        return float((Q1.T @ Q2).pow(2).sum() / self.k)

    def diagnose(self, step, loss_val, update_window, current_grad, full_update):
        P_total = current_grad.shape[0]
        current_frames = {}
        for b_name, idxs in self.block_indices.items():
            current_frames[b_name] = self.build_frame(update_window, idxs)

        # 1. Rotation Velocities
        rotations = {}
        if self.prev_frames:
            for b_name in self.block_indices:
                rotations[b_name] = self.gr_distance(self.prev_frames[b_name], current_frames[b_name])

        # 2. Cross-Block Face Alignments
        b_names = list(self.block_indices.keys())
        alignments = {}
        for i in range(len(b_names)):
            for j in range(i + 1, len(b_names)):
                b1, b2 = b_names[i], b_names[j]
                if current_frames[b1].shape[0] == current_frames[b2].shape[0]:
                    alignments[f"{b1}_vs_{b2}"] = self.face_alignment(current_frames[b1], current_frames[b2])

        # 3. Subspace Escape Rate (\gamma)
        proj_energy = 0.0
        total_energy = float((current_grad * current_grad).sum()) + 1e-30
        for b_name, idxs in self.block_indices.items():
            gb = current_grad[idxs]
            Qb = current_frames[b_name]
            pb = Qb @ (Qb.T @ gb)
            proj_energy += float((pb * pb).sum())
        gamma = 1.0 - (proj_energy / total_energy)

        # 4. Global Subspace Basis Q_glob (P_total x (num_blocks * k))
        num_blocks = len(self.block_indices)
        Q_glob = torch.zeros(P_total, num_blocks * self.k, device=current_grad.device)
        col_idx = 0
        for b_name, idxs in self.block_indices.items():
            Qb = current_frames[b_name]
            Q_glob[idxs, col_idx : col_idx + self.k] = Qb
            col_idx += self.k

        # Orthonormalize full global subspace
        Q_glob = torch.linalg.qr(Q_glob)[0]

        # 5. Fisher Descent Share (\rho_F)
        u_proj = Q_glob @ (Q_glob.T @ full_update)
        denom = -float((current_grad * full_update).sum()) + 1e-30
        rho_F = -float((current_grad * u_proj).sum()) / denom

        self.prev_frames = current_frames
        return {
            "step": step,
            "loss": loss_val,
            "gamma": gamma,
            "rho_F": rho_F,
            "rotations": rotations,
            "alignments": alignments,
            "Q_glob": Q_glob
        }

# =====================================================================
# 4. EXECUTION LOOP & LIVE MANIFOLD REPORTING
# =====================================================================
def main():
    torch.manual_seed(0)
    model = SyntheticTransformer()
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    # Param indices partitioning for structural blocks
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

    opt = AdaptiveKrylovSubspaceAdam(params, lr_kn=5e-2, lr_ortho=1e-3, k_dim=K_DIM)
    tracker = UnifiedManifoldTracker(model, block_indices, k_dim=K_DIM)

    print("=" * 88)
    print(f" UNIFIED MANIFOLD OPTIMIZATION (P={P}, k={K_DIM})")
    print(" Engine: Subspace Krylov-Newton + Adaptive Orthogonal Complement Adam")
    print("=" * 88)
    print(f"{'Step':>6} | {'Loss':>8} | {'Gamma (γ)':>10} | {'Rho_F (ρ)':>10} | {'Rot ATTN0':>10} | {'Face ATTN0/1':>12}")
    print("-" * 88)

    update_window = []
    WINDOW_SIZE = 8

    # Initial global active basis Q_K: shape (P, num_blocks * k)
    K_total = len(block_indices) * K_DIM
    Q_K = torch.randn(P, K_total)
    Q_K = torch.linalg.qr(Q_K)[0]

    for step in range(1, 121):
        x, y = get_batch()
        model.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        loss_val = float(loss)

        # Tridiagonal Krylov curvature matrix matching Q_K's total rank (K_total)
        T_K = torch.diag(torch.linspace(2.0, 0.5, K_total))

        # Execute Krylov-Newton + Orthogonal Adam step
        full_update, g_flat = opt.step(Q_K, T_K)

        update_window.append(full_update)
        if len(update_window) > WINDOW_SIZE:
            update_window.pop(0)

        # Diagnostic sweep every 10 steps
        if step >= WINDOW_SIZE and step % 10 == 0:
            diag = tracker.diagnose(step, loss_val, update_window, g_flat, full_update)
            Q_K = diag["Q_glob"]  # Recalibrate Krylov basis from global Fisher frame

            rot_val = diag["rotations"].get("ATTN0", 0.0)
            align_val = diag["alignments"].get("ATTN0_vs_ATTN1", 0.0)

            print(f"{step:6d} | {loss_val:8.4f} | {diag['gamma']:10.4f} | {diag['rho_F']:10.4f} | {rot_val:10.4f} | {align_val:12.4f}")

    print("-" * 88)

if __name__ == "__main__":
    main()
