"""
CALIBRATED SUBSPACE ADAM WITH DIMENSIONAL DOMINANCE ANALYSIS

1. Calibrates Adam in full space first: g_hat = g_t / (sqrt(v_t) + eps)
2. Builds basis Q_r from preconditioned step history (isotropic space)
3. Measures Dimensional Dominance: SVD energy distribution (sigma_i^2 / sum(sigma^2))
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# --- CONFIGURATION ---
V, D, L, T = 40, 12, 2, 16
K = 5
STEPS = 300
WARMUP = 20
WIN = 20
RECAL_INTERVAL = 10

LR = 3e-3
BETA1 = 0.9
BETA2 = 0.95
EPS = 1e-8
WD = 0.1


class Blk(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1 = nn.LayerNorm(D)
        s.n2 = nn.LayerNorm(D)
        s.q = nn.Linear(D, D, bias=False)
        s.k = nn.Linear(D, D, bias=False)
        s.v = nn.Linear(D, D, bias=False)
        s.o = nn.Linear(D, D, bias=False)
        s.g = nn.Linear(D, 2 * D, bias=False)
        s.u = nn.Linear(D, 2 * D, bias=False)
        s.w = nn.Linear(2 * D, D, bias=False)

    def forward(s, x):
        h = s.n1(x)
        q, k, v = s.q(h), s.k(h), s.v(h)
        a = Fn.softmax(
            q @ k.transpose(-2, -1) / math.sqrt(D)
            + torch.triu(torch.full((x.shape[1], x.shape[1]), -1e9), 1),
            dim=-1,
        )
        x = x + s.o(a @ v)
        h = s.n2(x)
        return x + s.w(Fn.silu(s.g(h)) * s.u(h))


class M(nn.Module):
    def __init__(s):
        super().__init__()
        s.te = nn.Embedding(V, D)
        s.pe = nn.Embedding(T, D)
        s.b = nn.ModuleList([Blk() for _ in range(L)])
        s.nf = nn.LayerNorm(D)

    def forward(s, x, y):
        h = s.te(x) + s.pe(torch.arange(x.shape[1]))
        for b in s.b:
            h = b(h)
        lo = s.nf(h) @ s.te.weight.T
        return lo, Fn.cross_entropy(lo.reshape(-1, V), y.reshape(-1))


rg = np.random.default_rng(1)
rules = {i: [(i * 3 + 1) % V, (i * 5 + 2) % V] for i in range(V)}
s_ = [0]
for _ in range(6000):
    c = s_[-1]
    s_.append(int(rules[c][0] if (len(s_) > 1 and s_[-2] % 2 == 0) else rules[c][1]))
seq = np.array(s_)


def batch(n=24):
    i = rg.integers(0, len(seq) - T - 1, n)
    return (
        torch.tensor(np.stack([seq[j : j + T] for j in i])),
        torch.tensor(np.stack([seq[j + 1 : j + T + 1] for j in i])),
    )


def run_calibrated_subspace_adam(seed=0):
    torch.manual_seed(seed)
    m = M()
    ps = list(m.parameters())

    def flat():
        return torch.cat([p.data.flatten() for p in ps]).clone()

    def setth(t):
        with torch.no_grad():
            j = 0
            for p in ps:
                q = p.numel()
                p.data.copy_(t[j : j + q].view_as(p))
                j += q

    def gv():
        return torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for p in ps]).clone()

    P_total = sum(p.numel() for p in ps)
    m_full = torch.zeros(P_total)
    v_full = torch.zeros(P_total)

    recent_preconditioned = []
    Q = None

    print("=" * 95)
    print("STEP | LOSS   | TYPE | ALIGNMENT | DIM DOMINANCE RATIO (% ENERGY IN K=5 / TOTAL)")
    print("     |        |      | (COS SIM) | [sigma_1^2, sigma_2^2, sigma_3^2, sigma_4^2, sigma_5^2]")
    print("=" * 95)

    for st in range(STEPS):
        th = flat()
        x, y = batch()
        m.zero_grad()
        _, l = m(x, y)
        l.backward()
        g = gv() + WD * th
        t_step = st + 1

        # 1. Update Full-Space Adam Moments
        m_full = BETA1 * m_full + (1.0 - BETA1) * g
        v_full = BETA2 * v_full + (1.0 - BETA2) * (g * g)

        m_hat = m_full / (1.0 - BETA1 ** t_step)
        v_hat = v_full / (1.0 - BETA2 ** t_step)

        # 2. FULL-SPACE CALIBRATION: Preconditioned Step Vector u_t
        u_t = -LR * (m_hat / (torch.sqrt(v_hat) + EPS))

        # Store preconditioned vector in history buffer
        recent_preconditioned.append(u_t.clone())
        if len(recent_preconditioned) > WIN:
            recent_preconditioned.pop(0)

        # 3. Dynamic Basis Construction & Dimensional Dominance Measurement
        should_recal = (st == WARMUP - 1) or (st >= WARMUP and (st + 1 - WARMUP) % RECAL_INTERVAL == 0)

        top_k_energy_pct = 0.0
        singular_spectrum_pct = []

        if should_recal:
            # Construct SVD on preconditioned history matrix H [P x WIN]
            H = torch.stack(recent_preconditioned, dim=1)
            U, S, _ = torch.linalg.svd(H, full_matrices=False)
            
            Q = U[:, :K]

            # Measure Dimensional Dominance Spectrum
            energy_per_dim = S ** 2
            total_energy = energy_per_dim.sum().item()
            
            if total_energy > 0:
                top_k_energy_pct = (energy_per_dim[:K].sum().item() / total_energy) * 100.0
                singular_spectrum_pct = [(e.item() / total_energy) * 100.0 for e in energy_per_dim[:K]]

        if st < WARMUP:
            setth(th + u_t)
            if (st + 1) % 10 == 0 or st == WARMUP - 1:
                print(f"{st+1:>4d} | {l.item():.4f} | FULL | 1.0000    | N/A (Warmup)")
        else:
            # 4. Project Preconditioned Step onto Calibrated Subspace Q
            nu_t = Q @ (Q.T @ u_t)

            norm_u = torch.norm(u_t).item()
            norm_nu = torch.norm(nu_t).item()
            cos_sim = (torch.dot(u_t, nu_t) / max(norm_u * norm_nu, 1e-12)).item()

            setth(th + nu_t)

            if (st + 1) % 20 == 0 or should_recal:
                spec_str = "[" + ", ".join(f"{p:.1f}%" for p in singular_spectrum_pct) + "]" if singular_spectrum_pct else "..."
                print(f"{st+1:>4d} | {l.item():.4f} | SUB  | {cos_sim:.4f}    | Top-5: {top_k_energy_pct:.2f}% | Spectrum: {spec_str}")


if __name__ == "__main__":
    run_calibrated_subspace_adam(seed=0)
