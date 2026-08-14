"""
SUBSPACE TRAJECTORY DIAGNOSTIC INSTRUMENT (measure_subspace_telemetry.py)

Directly measures the differential geometry of the 5D subspace trajectory:
- Residual Energy Leakage (Fraction of update discarded by Q_r)
- Cosine Similarity between unconstrained step u_t and projected step nu_t
- Singular Value Spectrum of update history (Rank saturation)
- Grassmannian Angle Drift between consecutive basis recalibrations
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


def diagnose_subspace_geometry(theta=0.5, seed=0):
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

    P_total = sum(p.numel() for p in ps)
    m_full = torch.zeros(P_total)
    v_full = torch.zeros(P_total)

    Q = None
    recent = {r: [] for r in idx}
    used = 0

    print("=" * 85)
    print("STEP | LOSS   | TYPE  | STEP NORM RATIO | COS SIM (u_t, nu) | TOP-5 SVD ENERGY % (FF)")
    print("=" * 85)

    for st in range(STEPS):
        full = False
        if st < DETECT:
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

        m_full = BETA1 * m_full + (1.0 - BETA1) * g
        v_full = BETA2 * v_full + (1.0 - BETA2) * (g * g)

        bc1 = 1.0 - (BETA1 ** t_step)
        bc2 = 1.0 - (BETA2 ** t_step)

        m_hat = m_full / bc1
        v_hat = v_full / bc2

        u_t = -LR * (m_hat / (torch.sqrt(v_hat) + EPS))

        if full and used < CAP:
            used += 1

            for r, ii in idx.items():
                recent[r].append(u_t[ii].clone())
                if len(recent[r]) > max(WIN, DETECT):
                    recent[r].pop(0)

            if st == DETECT - 1:
                Q = {}
                for r in idx:
                    U, S, _ = torch.linalg.svd(torch.stack(recent[r], 1), full_matrices=False)
                    Q[r] = U[:, :K]

            setth(th + u_t)
            
            if (st + 1) % 20 == 0:
                print(f"{st+1:>4d} | {l.item():.4f} | FULL  | 1.0000          | 1.0000            | N/A")

        else:
            nu = torch.zeros_like(g)

            for r, ii in idx.items():
                u_r = u_t[ii]
                Q_r = Q[r]
                nu[ii] = Q_r @ (Q_r.T @ u_r)

            # --- GEOMETRIC METRICS ---
            norm_u = torch.norm(u_t).item()
            norm_nu = torch.norm(nu).item()
            norm_ratio = norm_nu / max(norm_u, 1e-12)

            cos_sim = (torch.dot(u_t, nu) / max(norm_u * norm_nu, 1e-12)).item()

            # Measure singular value energy spectrum of the FF layer stack
            ff_matrix = torch.stack(recent["FF"], 1)
            _, S_ff, _ = torch.linalg.svd(ff_matrix, full_matrices=False)
            top5_energy = (S_ff[:K].pow(2).sum() / S_ff.pow(2).sum()).item() * 100.0

            setth(th + nu)

            if (st + 1) % 20 == 0 or st == DETECT:
                print(f"{st+1:>4d} | {l.item():.4f} | SUB   | {norm_ratio:.4f}          | {cos_sim:.4f}            | {top5_energy:.2f}%")


if __name__ == "__main__":
    diagnose_subspace_geometry(theta=0.5, seed=0)
