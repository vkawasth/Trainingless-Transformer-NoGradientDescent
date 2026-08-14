"""
INSTRUMENTED QUOTIENT TRAJECTORY TRACKER (smallgps_instrumented.py)

Instruments key dimensions and coherence structures across individual 
architectural roles (Embeddings, LayerNorm, Attention, Feed-Forward).
"""

import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

V, D, L, T = 40, 12, 2, 16
K = 5
STEPS = 300
DETECT = 20
WIN = 8
CHECK = 5
CAP = 80


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


def instrumented_quotient_tracker(theta=0.5, seed=0):
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

    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)

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

    Q = None
    recent = {r: [] for r in idx}
    quotient_history = {r: [] for r in idx}
    used = 0
    recals = []

    print("=" * 80)
    print(f"{'Step':<6} | {'Role':<6} | {'Leak (γ)':<8} | {'k90':<4} | {'Part Ratio':<10} | {'Spec Entropy':<12} | {'Coherence (cos)':<15}")
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
            
            # Temporary storage for step summary printout
            step_metrics = {}

            for r, ii in idx.items():
                gi = g[ii]
                e = float((gi * gi).sum())

                # Project and compute Quotient Vector
                pj = Q[r] @ (Q[r].T @ gi)
                q_i = gi - pj
                q_norm_sq = float((q_i * q_i).sum())

                quotient_history[r].append(q_i.clone())
                if len(quotient_history[r]) > WIN:
                    quotient_history[r].pop(0)

                # --- DIMENSION & COHERENCE METRICS ---
                gamma_r = q_norm_sq / max(e, 1e-30)
                
                # Directional Coherence (Cosine similarity of consecutive quotient vectors)
                if len(quotient_history[r]) >= 2:
                    v1 = quotient_history[r][-1]
                    v2 = quotient_history[r][-2]
                    norm_prod = (v1.norm() * v2.norm()).item()
                    cos_sim = float((v1 @ v2).item() / max(norm_prod, 1e-12))
                else:
                    cos_sim = 1.0

                # Spectral Metrics on Quotient Space History Matrix
                if len(quotient_history[r]) >= 2:
                    Q_mat = torch.stack(quotient_history[r], 1)
                    svals = torch.linalg.svdvals(Q_mat).numpy()
                    ev = svals**2
                    ev_norm = ev / max(ev.sum(), 1e-12)

                    # 1. Effective Rank (k90)
                    cum = np.cumsum(ev_norm)
                    k90 = int(np.searchsorted(cum, 0.90) + 1)

                    # 2. Participation Ratio: (sum λ)^2 / sum(λ^2)
                    pr = float((ev.sum() ** 2) / max((ev**2).sum(), 1e-12))

                    # 3. Spectral Entropy: -sum p log(p)
                    nz = ev_norm[ev_norm > 1e-12]
                    h_spec = float(-np.sum(nz * np.log(nz)))
                else:
                    k90, pr, h_spec = 1, 1.0, 0.0

                step_metrics[r] = {
                    "gamma": gamma_r,
                    "k90": k90,
                    "pr": pr,
                    "h_spec": h_spec,
                    "cos": cos_sim,
                }

                num += q_norm_sq
                den += e

            gamma_leak = num / max(den, 1e-30)

            # Print instrumentation metrics every 10 steps during analysis window
            if st in [20, 30, 40, 50]:
                for r in ROLES:
                    m_r = step_metrics[r]
                    print(
                        f"{st:<6d} | {r:<6} | {m_r['gamma']:<8.4f} | {m_r['k90']:<4d} | "
                        f"{m_r['pr']:<10.2f} | {m_r['h_spec']:<12.3f} | {m_r['cos']:<15.4f}"
                    )
                print("-" * 80)

            if gamma_leak > theta:
                recals.append(st)
                full = True

        th = flat()
        x, y = batch()
        _, l = m(x, y)
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        du = flat() - th

        if full and used < CAP:
            used += 1
            for r, ii in idx.items():
                recent[r].append(du[ii].clone())
                if len(recent[r]) > max(WIN, DETECT):
                    recent[r].pop(0)

            if st == DETECT - 1 or (st in recals):
                Q = {
                    r: torch.linalg.svd(torch.stack(recent[r], 1), full_matrices=False)[0][:, :K]
                    for r in idx
                }
        else:
            nu = torch.zeros_like(du)
            for r, ii in idx.items():
                nu[ii] = Q[r] @ (Q[r].T @ du[ii])
            setth(th + nu)


# --- Execution ---
instrumented_quotient_tracker(theta=0.5, seed=0)
