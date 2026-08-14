"""
CALIBRATED SUBSPACE ADAM EXPERIMENT (test_calibrated_adam.py)

Tests True Subspace Preconditioning:
1. Method 1: K x K Subspace Covariance Preconditioner (H_t^{-1/2})
2. Method 2: Projected Metric Preconditioner (Q^T D_t Q)^{-1}
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# --- CONFIGURATION ---
V, D, L, T = 40, 12, 2, 16
K = 5
STEPS = 200
DETECT = 20
WIN = 20

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

def test_calibrated_method(method_type="projected_metric", seed=0):
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
    
    # Subspace states
    m_sub = torch.zeros(K)
    H_sub = torch.eye(K) * 1e-4

    recent = []
    Q = None

    print(f"\n--- RUNNING CALIBRATED ADAM: METHOD = {method_type.upper()} ---")

    for st in range(STEPS):
        th = flat()
        x, y = batch()
        m.zero_grad()
        _, l = m(x, y)
        l.backward()
        g = gv() + WD * th
        t_step = st + 1

        # Track full space moments for basis tracking
        m_full = BETA1 * m_full + (1.0 - BETA1) * g
        v_full = BETA2 * v_full + (1.0 - BETA2) * (g * g)

        m_hat = m_full / (1.0 - BETA1 ** t_step)
        v_hat = v_full / (1.0 - BETA2 ** t_step)

        u_unconstrained = -LR * (m_hat / (torch.sqrt(v_hat) + EPS))
        recent.append(u_unconstrained.clone())
        if len(recent) > WIN:
            recent.pop(0)

        if st < DETECT:
            setth(th + u_unconstrained)
            if st == DETECT - 1:
                U, _, _ = torch.linalg.svd(torch.stack(recent, 1), full_matrices=False)
                Q = U[:, :K]
        else:
            if method_type == "subspace_covariance":
                # Method 1: Project raw gradient into Q, maintain KxK covariance H_t
                g_sub = Q.T @ g
                m_sub = BETA1 * m_sub + (1.0 - BETA1) * g_sub
                H_sub = BETA2 * H_sub + (1.0 - BETA2) * torch.outer(g_sub, g_sub)
                
                m_sub_hat = m_sub / (1.0 - BETA1 ** t_step)
                H_sub_hat = H_sub / (1.0 - BETA2 ** t_step)
                
                # Compute H_hat^{-1/2} via Eigendecomposition
                L_evals, V_evecs = torch.linalg.eigh(H_sub_hat)
                L_inv_sqrt = torch.diag(1.0 / (torch.sqrt(torch.clamp(L_evals, min=1e-8)) + EPS))
                H_inv_sqrt = V_evecs @ L_inv_sqrt @ V_evecs.T
                
                delta_alpha = -LR * (H_inv_sqrt @ m_sub_hat)
                nu = Q @ delta_alpha

            elif method_type == "projected_metric":
                # Method 2: Project full-space Adam diagonal D_t into Q via Q^T D_t Q
                d_diag = 1.0 / (torch.sqrt(v_hat) + EPS)
                # Compute Q^T D_t Q efficiently: Q^T @ (d_diag * Q)
                M_K = Q.T @ (d_diag.unsqueeze(1) * Q)
                
                # Subspace update step
                m_sub = Q.T @ m_hat
                M_K_inv = torch.linalg.inv(M_K + EPS * torch.eye(K))
                delta_alpha = -LR * (M_K_inv @ m_sub)
                nu = Q @ delta_alpha

            setth(th + nu)

        if (st + 1) % 20 == 0:
            print(f"Step {st+1:>3d} | Loss: {l.item():.4f}")

if __name__ == "__main__":
    test_calibrated_method("subspace_covariance")
    test_calibrated_method("projected_metric")
