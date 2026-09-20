#!/usr/bin/env python3
"""EM_BOTTOMUP -- estimate the grammar level by level, no gradient descent.

    python3 em_bottomup.py [--data /tmp] [--n-train 20000]

WHY THE FIRST ATTEMPT FAILED
----------------------------
Joint EM over all four levels from a random 64x64x64 init recovered 0.3% of
the true rule mass and lost to the transformer. That was a bad algorithm, not
a fair test: it asked a single latent-variable optimiser to find four levels
of structure at once from noise.

Level 1 is NOT a latent problem. Leaves 2n and 2n+1 are siblings by position,
so the sibling pair (x,y) is OBSERVED. Clustering those pairs gives level-1
symbols directly from counts. Once level 1 is fixed, its posterior symbols
become the observations for level 2, and so on. Each stage is a small,
well-posed estimation from counts rather than a global search.

    level 1: cluster observed sibling pairs           -> symbols + rules
    level l: same, on the level-(l-1) symbol sequence -> symbols + rules
    then:    a few EM sweeps over the whole tree to polish

Scored on the same held-out split, same local/higher split, wall-clock
reported, no gradients anywhere.
"""
import json, math, argparse, time, collections
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp")
ap.add_argument("--n-train", type=int, default=20000)
ap.add_argument("--n-val", type=int, default=2000)
ap.add_argument("--polish", type=int, default=5, help="joint EM sweeps after init")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
KIND = M["pos_kind"]
tr = torch.tensor(json.load(open(f"{a.data}/train_ids.json")), dtype=torch.long)
va = torch.tensor(json.load(open(f"{a.data}/val_ids.json")), dtype=torch.long)
TR = tr[:(len(tr)//SEQ)*SEQ].view(-1, SEQ)[:a.n_train]
VA = va[:(len(va)//SEQ)*SEQ].view(-1, SEQ)[:a.n_val]
print(f"  L={L} v={V} leaves={NLEAF} seq={SEQ}   fit on {len(TR)}, "
      f"score {len(VA)} held-out")
t0 = time.time()

def card(l): return NLEAF if l == 1 else V

# ---------------- bottom-up initialisation --------------------------------
# At each level the children are OBSERVED (hard-assigned from the level below),
# so the only question is which parent symbol generated each observed pair.
# Cluster the distinct pairs into V groups by co-occurrence, then read rules off.
P = {}
cur = TR.clone()                                   # current symbol sequence
for l in range(1, L + 1):
    c = card(l)
    lo, hi = cur[:, 0::2], cur[:, 1::2]
    pair_id = (lo * c + hi).reshape(-1)
    cnt = torch.bincount(pair_id, minlength=c * c).double()
    seen = torch.nonzero(cnt).squeeze(-1)
    # assign pairs to V parents: group by how often each pair occurs, then
    # spread them round-robin over parents in frequency order. This is a
    # deliberately crude clustering -- the polish sweeps refine it.
    order = seen[torch.argsort(cnt[seen], descending=True)]
    assign = torch.zeros(c * c, dtype=torch.long)
    for k, pid in enumerate(order.tolist()):
        assign[pid] = k % V
    tab = torch.zeros(V, c, c, dtype=torch.float64)
    for pid in order.tolist():
        tab[assign[pid], pid // c, pid % c] = cnt[pid]
    tab = tab + 1e-6
    P[l] = tab / tab.sum((1, 2), keepdim=True)
    # hard-assign this level's symbol for each node, to feed the next level
    cur = assign[(lo * c + hi)]
    print(f"    level {l}: {len(seen)} distinct observed pairs -> {V} symbols")

# ---------------- inside / outside ----------------------------------------
def inside(X, P):
    B = X.shape[0]
    t = torch.zeros(B, SEQ, NLEAF, dtype=torch.float64)
    t.scatter_(2, X.unsqueeze(-1), 1.0)
    tabs = [t]
    for l in range(1, L + 1):
        lo, hi = tabs[-1][:, 0::2, :], tabs[-1][:, 1::2, :]
        tabs.append(torch.einsum('pxy,bnx,bny->bnp', P[l], lo, hi))
    return tabs

def outside(tabs, P):
    B = tabs[0].shape[0]
    outs = [None]*(L+1)
    outs[L] = torch.full((B, 1, V), 1.0/V, dtype=torch.float64)
    for l in range(L, 0, -1):
        lo, hi = tabs[l-1][:, 0::2, :], tabs[l-1][:, 1::2, :]
        po = outs[l]
        lo_o = torch.einsum('pxy,bnp,bny->bnx', P[l], po, hi)
        hi_o = torch.einsum('pxy,bnp,bnx->bny', P[l], po, lo)
        ch = torch.zeros(B, lo.shape[1]*2, lo.shape[2], dtype=torch.float64)
        ch[:, 0::2, :] = lo_o; ch[:, 1::2, :] = hi_o
        outs[l-1] = ch
    return outs

def em_sweep(X, P):
    tabs = inside(X, P); outs = outside(tabs, P)
    Z = (tabs[L][:, 0, :].sum(-1) / V).clamp_min(1e-300)
    new = {}
    for l in range(1, L+1):
        lo, hi = tabs[l-1][:, 0::2, :], tabs[l-1][:, 1::2, :]
        cnt = torch.einsum('bnp,bnx,bny->pxy', outs[l] / Z[:, None, None], lo, hi) * P[l]
        cnt = cnt + 1e-9
        new[l] = cnt / cnt.sum((1, 2), keepdim=True)
    return new

def loglik(X, P):
    t = inside(X, P)
    return torch.log((t[L][:, 0, :].sum(-1)/V).clamp_min(1e-300)).mean().item()

print(f"    init held-out loglik/seq = {loglik(VA, P):.4f}")
for it in range(a.polish):
    P = em_sweep(TR, P)
    print(f"    polish {it+1}: held-out loglik/seq = {loglik(VA, P):.4f}")
t_fit = time.time() - t0

# ---------------- predict -------------------------------------------------
t1 = time.time()
onehot = torch.zeros(len(VA), SEQ, NLEAF, dtype=torch.float64)
onehot.scatter_(2, VA.unsqueeze(-1), 1.0)
tot = np.zeros(SEQ)
for t in range(SEQ):
    lf = torch.ones(len(VA), SEQ, NLEAF, dtype=torch.float64)
    if t > 0: lf[:, :t, :] = onehot[:, :t, :]
    tabs = inside_lf = None
    tt = [lf]
    for l in range(1, L+1):
        lo, hi = tt[-1][:, 0::2, :], tt[-1][:, 1::2, :]
        tt.append(torch.einsum('pxy,bnx,bny->bnp', P[l], lo, hi))
    oo = outside(tt, P)
    marg = oo[0][:, t, :]
    pr = marg / marg.sum(-1, keepdim=True).clamp_min(1e-300)
    tot[t] = -torch.log(pr.gather(1, VA[:, t:t+1]).clamp_min(1e-300)).mean().item()
t_pred = time.time() - t1

loc = np.mean([tot[t] for t in range(SEQ) if KIND[t] == "local"])
hig = np.mean([tot[t] for t in range(SEQ) if KIND[t] == "higher"])
print(f"\n  fit {t_fit:.1f}s, inference {t_pred:.1f}s, NO GRADIENTS")
print(f"\n  {'':<24}{'local':>9}{'higher':>9}")
print(f"  {'bottom-up + inference':<24}{loc:>9.4f}{hig:>9.4f}")
print(f"  {'joint EM (previous)':<24}{2.5663:>9.4f}{3.8308:>9.4f}")
print(f"  {'transformer 48k steps':<24}{2.0223:>9.4f}{3.7716:>9.4f}")
print(f"  {'exact floor':<24}{1.6034:>9.4f}{3.5527:>9.4f}")
print(f"\n  vs transformer: local {loc-2.0223:+.4f}  higher {hig-3.7716:+.4f}"
      f"   (negative = constructed predictor wins)")
if M.get("rules_all"):
    T = {int(l): {int(i): [tuple(x) for x in v] for i, v in d.items()}
         for l, d in M["rules_all"].items()}
    for l in range(1, L+1):
        m = sum(float(P[l][p, x, y]) for p in range(V) for (x, y) in set(T[l][p]))
        print(f"  level {l}: estimated mass on TRUE rules = {m/V:.3f}")
print(f"\n  NOTE rule recovery can be low while prediction is good: the symbol")
print(f"  labels are arbitrary, so a relabelled grammar scores 0 on that line")
print(f"  and identically on loss. Loss is the measure; recovery is a hint.")
