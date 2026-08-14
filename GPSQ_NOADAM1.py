"""
GLOBAL-STATE SUBSPACE ADAM TRACKER (smallgps_global_state_subspace_adam.py)

Tracks BOTH m_t and v_t in full parameter space R^P to maintain uninterrupted 
differential momentum trajectories. Projects ONLY the final Adam step vector 
onto the low-rank subspace frame Q_r during constrained steps.
"""

import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# --- CONFIGURATION ---
V, D, L, T = 40, 12, 2, 16
K = 5
STEPS = 300
DETECT = 20
WIN = 8
CHECK = 5
CAP = 80

# Subspace Optimizer Hyperparameters
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


# Synthetic Context Data Generator
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


def role(nm):
    if nm.startswith("te") or nm.startswith("pe"):
        return "EMB"
    if "n1" in nm or "n2" in nm or nm.startswith("nf"):
        return "LN"
    for pat, lab in ((r"\.q\.", "W_Q"), (r"\.k\.", "W_K"), (r"\.v\.", "W_V"), (r"\.o\.", "W_O")):
        if re.search(pat, nm):
            return lab
    if re.search(r"\.(g|u|w)\.", nm):
        return "FF"
    return "other"


ROLES = ["EMB", "LN", "W_Q", "W_K", "W_V", "W_O", "FF"]


def run_global_state_subspace_adam(theta=0.5, seed=0):
    torch.manual_seed(seed)
    m = M()
    named = [(n, p) for n, p in m.named_parameters()]
    ps = [p for _, p in named]

    span = {}
    i = 0
    for nm, p in named:
        span[nm] = (i, i + p.numel())
        i += p.numel()

    idx = {r: [] for r in ROLES}
    for nm, (a, b) in span.items():
        if role(nm) in idx:
            idx[role(nm)].append(torch.arange(a, b))
    idx = {r: torch.cat(v) for r, v in idx.items() if len(v)}

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

    EV = [batch(48) for _ in range(3)]

    def loss():
        t = 0.0
        with torch.no_grad():
            for x, y in EV:
                t += float(m(x, y)[1])
        return t / len(EV)

    # --- STATE TENSORS (FULL PARAMETER SPACE R^P) ---
    P_total = sum(p.numel() for p in ps)
    m_full = torch.zeros(P_total)
    v_full = torch.zeros(P_total)

    Q = None
    recent = {r: [] for r in idx}
    quotient_history = {r: [] for r in idx}
    used = 0
    recals = []

    print("=" * 80)
    print("RUNNING GLOBAL-STATE SUBSPACE ADAM (Global M & V + Subspace Step Projection)")
    print("=" * 80)

    for st in range(STEPS):
        full = False
        if st < DETECT:
            full = True
        elif Q is not None and st % CHECK == 0 and used < CAP:
            th = flat()
            x, y = batch()
            m.zero_grad()
            _, l = m(x, y)
            l.backward()
            g = gv()
            m.zero_grad()
            setth(th)

            num = den = 0.0
            for r, ii in idx.items():
                gi = g[ii]
                e = float((gi * gi).sum())

                pj = Q[r] @ (Q[r].T @ gi)
                q_i = gi - pj
                q_norm_sq = float((q_i * q_i).sum())

                quotient_history[r].append(q_i.clone())
                if len(quotient_history[r]) > WIN:
                    quotient_history[r].pop(0)

                num += q_norm_sq
                den += e

            gamma_leak = num / max(den, 1e-30)

            if gamma_leak > theta:
                recals.append(st)
                full = True

        # Forward & Backward Pass
        th = flat()
        x, y = batch()
        m.zero_grad()
        _, l = m(x, y)
        l.backward()
        g = gv()
        g = g + WD * th

        t_step = st + 1

        # 1. UNINTERRUPTED GLOBAL ADAM STATE UPDATE IN R^P
        m_full = BETA1 * m_full + (1.0 - BETA1) * g
        v_full = BETA2 * v_full + (1.0 - BETA2) * (g * g)

        bc1 = 1.0 - (BETA1 ** t_step)
        bc2 = 1.0 - (BETA2 ** t_step)

        m_hat = m_full / bc1
        v_hat = v_full / bc2

        # Standard Unconstrained Adam Step Vector in R^P
        u_t = -LR * (m_hat / (torch.sqrt(v_hat) + EPS))

        if full and used < CAP:
            used += 1

            for r, ii in idx.items():
                recent[r].append(u_t[ii].clone())
                if len(recent[r]) > max(WIN, DETECT):
                    recent[r].pop(0)

            # Re-estimate basis frame Q_r via SVD on historical step vectors
            if st == DETECT - 1 or (st in recals):
                Q = {
                    r: torch.linalg.svd(torch.stack(recent[r], 1), full_matrices=False)[0][:, :K]
                    for r in idx
                }

            # Apply Full Rank Update
            setth(th + u_t)

        else:
            # 2. PROJECT FULL ADAM STEP VECTOR ONTO ACTIVE SUBSPACE Q_r
            nu = torch.zeros_like(g)

            for r, ii in idx.items():
                u_r = u_t[ii]
                Q_r = Q[r]  # (P_r, K)

                # Project the preconditioned momentum update onto basis Q_r
                nu[ii] = Q_r @ (Q_r.T @ u_r)

            setth(th + nu)

        if (st + 1) % 100 == 0:
            print(f"Step {st+1:>3d} | Val Loss: {loss():.4f} | Full Steps Used: {used}/{CAP}")


if __name__ == "__main__":
    run_global_state_subspace_adam(theta=0.5, seed=0)
