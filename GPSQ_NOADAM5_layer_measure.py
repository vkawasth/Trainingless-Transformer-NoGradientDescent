"""
PER-LAYER PRECONDITIONED SUBSPACE ALIGNMENT DIAGNOSTIC

Evaluates preconditioned update trajectory alignment and spectral energy distribution
independently across 4 module families:
1. Embeddings (token + pos)
2. Attention Projection Weights (Q, K, V, Out)
3. MLP Dense Layers (Gate, Up, Down)
4. LayerNorm Scales (n1, n2, nf)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# --- CONFIGURATION ---
V, D, L, T = 40, 12, 2, 16
K_SUB = 3  # Track top-3 sub-components per layer
STEPS = 250
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


def classify_param(name):
    if "te" in name or "pe" in name:
        return "Embeddings"
    elif any(k in name for k in [".q", ".k", ".v", ".o"]):
        return "Attention"
    elif any(k in name for k in [".g", ".u", ".w"]):
        return "MLP"
    elif "n1" in name or "n2" in name or "nf" in name:
        return "LayerNorm"
    return "Other"


def run_per_layer_analysis(seed=0):
    torch.manual_seed(seed)
    m = M()

    groups = {"Embeddings": [], "Attention": [], "MLP": [], "LayerNorm": []}
    param_map = {}

    for name, p in m.named_parameters():
        grp = classify_param(name)
        groups[grp].append((name, p))
        param_map[p] = grp

    # Tracking per-group states
    states = {}
    for grp, p_list in groups.items():
        total_p = sum(p.numel() for _, p in p_list)
        states[grp] = {
            "p_list": p_list,
            "total_p": total_p,
            "m_full": torch.zeros(total_p),
            "v_full": torch.zeros(total_p),
            "history": [],
            "Q": None,
            "cos_sim": 1.0,
            "top3_energy": 0.0,
            "sigma_sq": [],
        }

    print("=" * 110)
    print(f"{'STEP':>4} | {'MODULE GROUP':<12} | {'ALIGNMENT (cos θ)':<18} | {'TOP-3 ENERGY (%)':<18} | {'SPECTRUM [σ1², σ2², σ3²]'}")
    print("=" * 110)

    for st in range(STEPS):
        x, y = batch()
        m.zero_grad()
        _, l = m(x, y)
        l.backward()
        t_step = st + 1

        should_recal = (st == WARMUP - 1) or (st >= WARMUP and (st + 1 - WARMUP) % RECAL_INTERVAL == 0)
        print_step = (st + 1) % 50 == 0 or st == WARMUP - 1

        for grp, st_dict in states.items():
            # Flatten group params & grads
            g_list = []
            th_list = []
            for _, p in st_dict["p_list"]:
                th_list.append(p.data.flatten())
                g_list.append(p.grad.flatten() if p.grad is not None else torch.zeros_like(p.data.flatten()))

            th = torch.cat(th_list)
            g = torch.cat(g_list) + WD * th

            # 1. Per-Group Preconditioning
            st_dict["m_full"] = BETA1 * st_dict["m_full"] + (1.0 - BETA1) * g
            st_dict["v_full"] = BETA2 * st_dict["v_full"] + (1.0 - BETA2) * (g * g)

            m_hat = st_dict["m_full"] / (1.0 - BETA1 ** t_step)
            v_hat = st_dict["v_full"] / (1.0 - BETA2 ** t_step)

            u_t = -LR * (m_hat / (torch.sqrt(v_hat) + EPS))

            st_dict["history"].append(u_t.clone())
            if len(st_dict["history"]) > WIN:
                st_dict["history"].pop(0)

            # 2. Per-Group Subspace Basis Recalibration
            if should_recal and len(st_dict["history"]) >= 5:
                H = torch.stack(st_dict["history"], dim=1)
                U, S, _ = torch.linalg.svd(H, full_matrices=False)
                st_dict["Q"] = U[:, :K_SUB]

                e_per_dim = S ** 2
                tot_e = e_per_dim.sum().item()
                if tot_e > 0:
                    st_dict["top3_energy"] = (e_per_dim[:K_SUB].sum().item() / tot_e) * 100.0
                    st_dict["sigma_sq"] = [(e.item() / tot_e) * 100.0 for e in e_per_dim[:K_SUB]]

            # 3. Apply Update
            if st < WARMUP or st_dict["Q"] is None:
                delta = u_t
                st_dict["cos_sim"] = 1.0
            else:
                delta = st_dict["Q"] @ (st_dict["Q"].T @ u_t)
                norm_u = torch.norm(u_t).item()
                norm_d = torch.norm(delta).item()
                st_dict["cos_sim"] = (torch.dot(u_t, delta) / max(norm_u * norm_d, 1e-12)).item()

            # Write updated values back to parameter tensors
            with torch.no_grad():
                idx = 0
                for _, p in st_dict["p_list"]:
                    num = p.numel()
                    p.data.add_(delta[idx : idx + num].view_as(p))
                    idx += num

        if print_step:
            for grp in ["Embeddings", "Attention", "MLP", "LayerNorm"]:
                st_dict = states[grp]
                spec_str = "[" + ", ".join(f"{p:.1f}%" for p in st_dict["sigma_sq"]) + "]" if st_dict["sigma_sq"] else "N/A"
                print(f"{st+1:>4d} | {grp:<12} | {st_dict['cos_sim']:<18.4f} | {st_dict['top3_energy']:<18.2f} | {spec_str}")
            print("-" * 110)


if __name__ == "__main__":
    run_per_layer_analysis(seed=0)
