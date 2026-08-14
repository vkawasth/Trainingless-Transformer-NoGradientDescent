"""
NON-ADAM CALIBRATED SUBSPACE EXPERIMENT

Demonstrates subspace optimization with non-Adam preconditioners:
1. Instantaneous Sign Preconditioning: g_hat = sign(g)
2. Empirical Fisher Preconditioning: g_hat = g / (|g| + eps)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# Config
V, D, L, T = 40, 12, 2, 16
K = 5
STEPS = 100
WARMUP = 20
WIN = 20
LR = 1e-3
EPS = 1e-6

class Blk(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.q = nn.Linear(D, D, bias=False)
        s.k = nn.Linear(D, D, bias=False)
        s.v = nn.Linear(D, D, bias=False)
        s.o = nn.Linear(D, D, bias=False)

    def forward(s, x):
        h = s.n1(x)
        q, k, v = s.q(h), s.k(h), s.v(h)
        a = Fn.softmax(
            q @ k.transpose(-2, -1) / math.sqrt(D)
            + torch.triu(torch.full((x.shape[1], x.shape[1]), -1e9), 1),
            dim=-1,
        )
        return x + s.o(a @ v)

class M(nn.Module):
    def __init__(s):
        super().__init__()
        s.te = nn.Embedding(V, D)
        s.pe = nn.Embedding(T, D)
        s.b = nn.ModuleList([Blk() for _ in range(L)])
        s.nf = nn.LayerNorm(D)

    def forward(s, x, y):
        h = s.te(x) + s.pe(torch.arange(x.shape[1]))
        for b in s.b: h = b(h)
        lo = s.nf(h) @ s.te.weight.T
        return lo, Fn.cross_entropy(lo.reshape(-1, V), y.reshape(-1))

rg = np.random.default_rng(1)
rules = {i: [(i * 3 + 1) % V, (i * 5 + 2) % V] for i in range(V)}
s_ = [0]
for _ in range(2000):
    c = s_[-1]
    s_.append(int(rules[c][0] if (len(s_) > 1 and s_[-2] % 2 == 0) else rules[c][1]))
seq = np.array(s_)

def batch(n=24):
    i = rg.integers(0, len(seq) - T - 1, n)
    return torch.tensor(np.stack([seq[j : j + T] for j in i])), torch.tensor(np.stack([seq[j + 1 : j + T + 1] for j in i]))

def run_sign_subspace():
    torch.manual_seed(0)
    m = M()
    ps = list(m.parameters())

    def flat(): return torch.cat([p.data.flatten() for p in ps]).clone()
    def setth(t):
        with torch.no_grad():
            j = 0
            for p in ps:
                q = p.numel()
                p.data.copy_(t[j : j + q].view_as(p))
                j += q
    def gv(): return torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for p in ps]).clone()

    history = []
    Q = None

    print("\n--- RUNNING SUBSPACE OPTIMIZATION WITH INSTANTANEOUS SIGN PRECONDITIONING (NO ADAM) ---")
    for st in range(STEPS):
        th = flat()
        x, y = batch()
        m.zero_grad()
        _, l = m(x, y)
        l.backward()
        g = gv()

        # Instantaneous Sign Preconditioning (Zero state memory)
        u_t = -LR * torch.sign(g)

        history.append(u_t.clone())
        if len(history) > WIN: history.pop(0)

        if st < WARMUP:
            setth(th + u_t)
        else:
            if st == WARMUP or st % 10 == 0:
                H = torch.stack(history, dim=1)
                U, S, _ = torch.linalg.svd(H, full_matrices=False)
                Q = U[:, :K]
            
            # Project sign-preconditioned step onto subspace Q
            nu_t = Q @ (Q.T @ u_t)
            setth(th + nu_t)

        if (st + 1) % 20 == 0:
            print(f"Step {st+1:>3d} | Loss: {l.item():.4f}")

if __name__ == "__main__":
    run_sign_subspace()
