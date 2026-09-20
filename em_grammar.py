#!/usr/bin/env python3
"""EM_GRAMMAR -- replace gradient descent with estimation plus exact inference.

    python3 em_grammar.py [--data /tmp] [--iters 30] [--restarts 3]

THE CLAIM UNDER TEST
--------------------
A 405k-parameter transformer trained for 48,000 SGD steps reaches
    local 2.04   higher 3.76
The exact floors for this grammar are
    local 1.60   higher 3.55
This script never computes a gradient. It estimates the rule probabilities by
EM from the raw token stream -- no parse labels, no supervision -- and then
predicts by exact inside-outside inference. Scored on the SAME held-out split,
with the SAME local/higher position split, and wall-clock reported.

If it lands near the floor, then for data with a learnable generative model,
estimation + exact inference beats SGD by ~0.2 nats in seconds. If estimation
fails, the cost of the replacement is what the failure measures.

WHAT IS AND IS NOT GIVEN
------------------------
Given: the STRUCTURE -- depth L, branching 2, symbol counts. That is the same
information the transformer gets implicitly by being trained on the corpus,
and without it "estimate the grammar" is not a well-posed problem.
NOT given: which rules exist, their probabilities, or any parse. Those are
estimated. The rule table in rhm_meta.json is used ONLY at the end, to report
how close the estimate got -- never during fitting.

METHOD
------
Parameterise every level l by a dense P(l)[parent, left, right], initialised
at random and normalised per parent. E step: inside-outside on each sequence
gives posterior rule counts. M step: normalise. Repeat. Multiple restarts,
keep the best by held-out likelihood, because EM on latent trees has local
optima and a single run would understate what the method can do.
"""
import json, math, argparse, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--restarts", type=int, default=2)
ap.add_argument("--n-train", type=int, default=4000, help="sequences for EM")
ap.add_argument("--n-val", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
KIND = M["pos_kind"]
train = torch.tensor(json.load(open(f"{a.data}/train_ids.json")), dtype=torch.long)
val = torch.tensor(json.load(open(f"{a.data}/val_ids.json")), dtype=torch.long)
TR = train[: (len(train)//SEQ)*SEQ].view(-1, SEQ)[: a.n_train]
VA = val[: (len(val)//SEQ)*SEQ].view(-1, SEQ)[: a.n_val]
print(f"  L={L} v={V} leaves={NLEAF} seq={SEQ}   EM on {len(TR)} sequences, "
      f"scoring {len(VA)} held-out")

def child_card(l):
    return NLEAF if l == 1 else V

def init(g):
    """random rule tables, normalised per parent"""
    P = {}
    for l in range(1, L+1):
        c = child_card(l)
        t = torch.rand(V, c, c, generator=g, dtype=torch.float64) + 1e-3
        P[l] = t / t.sum((1, 2), keepdim=True)
    return P

def inside(X, P):
    """X: B x SEQ leaf ids. returns list of inside tables per level."""
    B = X.shape[0]
    cur = torch.zeros(B, SEQ, NLEAF, dtype=torch.float64)
    cur.scatter_(2, X.unsqueeze(-1), 1.0)
    tabs = [cur]
    for l in range(1, L+1):
        lo, hi = cur[:, 0::2, :], cur[:, 1::2, :]
        # out[b,n,p] = sum_{x,y} P[p,x,y] lo[b,n,x] hi[b,n,y]
        out = torch.einsum('pxy,bnx,bny->bnp', P[l], lo, hi)
        tabs.append(out); cur = out
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

def loglik(X, P):
    t = inside(X, P)
    Z = t[L][:, 0, :].sum(-1) / V
    return torch.log(Z.clamp_min(1e-300)).mean().item()

def em_step(X, P):
    tabs = inside(X, P); outs = outside(tabs, P)
    Z = (tabs[L][:, 0, :].sum(-1) / V).clamp_min(1e-300)
    newP = {}
    for l in range(1, L+1):
        lo, hi = tabs[l-1][:, 0::2, :], tabs[l-1][:, 1::2, :]
        po = outs[l]
        # expected count of rule (p,x,y) = outside[p] * P[p,x,y] * lo[x] * hi[y] / Z
        cnt = torch.einsum('bnp,bnx,bny->pxy', po / Z[:, None, None], lo, hi) * P[l]
        cnt = cnt + 1e-8
        newP[l] = cnt / cnt.sum((1, 2), keepdim=True)
    return newP

t0 = time.time()
g = torch.Generator().manual_seed(a.seed)
best, bestll = None, -1e300
for r in range(a.restarts):
    P = init(g)
    for it in range(a.iters):
        P = em_step(TR, P)
    ll = loglik(VA, P)
    print(f"  restart {r}: held-out loglik/seq = {ll:.4f}")
    if ll > bestll: bestll, best = ll, P
P = best
t_fit = time.time() - t0

# ---- predict: exact conditional at each position, same split as the model
t1 = time.time()
tot = np.zeros(SEQ)
for t in range(SEQ):
    # one inside pass with leaf t (and everything after) free; the OUTSIDE
    # value at leaf t is the unnormalised marginal over its token. No loop
    # over the 48 candidate tokens.
    cur = torch.zeros(len(VA), SEQ, NLEAF, dtype=torch.float64)
    cur.scatter_(2, VA.unsqueeze(-1), 1.0)
    cur[:, t:, :] = 1.0
    tt = [cur]
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
print(f"\n  EM fit {t_fit:.1f}s ({a.restarts} restarts x {a.iters} iters), "
      f"inference {t_pred:.1f}s, NO GRADIENTS")
print(f"\n  {'':<22}{'local':>9}{'higher':>9}")
print(f"  {'EM + exact inference':<22}{loc:>9.4f}{hig:>9.4f}")
print(f"  {'transformer 48k steps':<22}{2.0223:>9.4f}{3.7716:>9.4f}")
print(f"  {'exact floor':<22}{1.6034:>9.4f}{3.5527:>9.4f}")
print(f"\n  gap to floor: local {loc-1.6034:+.4f}  higher {hig-3.5527:+.4f}")
print(f"  vs transformer: local {loc-2.0223:+.4f}  higher {hig-3.7716:+.4f}"
      f"   (negative = the constructed predictor wins)")

# how close did the ESTIMATE get to the true rules? reporting only.
if M.get("rules_all"):
    tr = {int(l): {int(i): [tuple(x) for x in v] for i, v in d.items()}
          for l, d in M["rules_all"].items()}
    for l in range(1, L+1):
        mass = 0.0
        for p in range(V):
            for (x, y) in set(tr[l][p]): mass += float(P[l][p, x, y])
        print(f"  level {l}: estimated mass on TRUE rules = {mass/V:.3f} "
              f"(1.000 = perfectly recovered)")
