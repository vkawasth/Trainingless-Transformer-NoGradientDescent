#!/usr/bin/env python3
"""FLOOR_EXACT -- per-position conditional entropy H(x_t | x_<t), exactly.

    python3 floor_exact.py [--data /tmp] [--samples 2000]

The builder prints H_seq_exact/SEQ as "the floor". That is the average over a
whole sequence and is NOT the floor for predicting token t from tokens 0..t-1.
The `higher` positions are the expensive ones -- each opens a fresh subtree --
and t=0 is a uniform draw over the root alphabet, irreducibly ln(V) nats, and
is labelled `higher`.

METHOD: inside-outside on the fixed positional tree, batched.
  inside[l][n]  = P(observed leaves under node n | node n has symbol s)
  outside[l][n] = P(everything outside that subtree | node n has symbol s)
  marginal at a FREE leaf t  =  outside[0][t], normalised
One pass per position gives the whole distribution over x_t -- no loop over
candidate tokens. Symbol sums are a sparse bilinear contraction, not a Python
loop, and all sequences run as a batch.
"""
import json, math, argparse
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp")
ap.add_argument("--samples", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
KIND = M["pos_kind"]
if "rules_all" not in M:
    raise SystemExit("rebuild the corpus with the updated build_corpus_syn.py")
R = {int(l): {int(i): [tuple(t) for t in v] for i, v in d.items()}
     for l, d in M["rules_all"].items()}
g = torch.Generator().manual_seed(a.seed)

# rule tensors: Rl[l] is a list of (parent, left, right, weight)
RT = {}
for l in range(1, L + 1):
    par, le, ri, w = [], [], [], []
    for s in range(V):
        pr = R[l][s]; wt = 1.0 / len(pr)
        for (x, y) in pr:
            par.append(s); le.append(x); ri.append(y); w.append(wt)
    RT[l] = (torch.tensor(par), torch.tensor(le), torch.tensor(ri),
             torch.tensor(w, dtype=torch.float64))

def sample(nb):
    """batch of sequences from the grammar"""
    sym = torch.randint(0, V, (nb,), generator=g)
    cur = [sym]
    for l in range(L, 0, -1):
        nxt = []
        for node in cur:
            pr = [R[l][int(s)] for s in node]
            pick = [p[torch.randint(0, len(p), (1,), generator=g).item()] for p in pr]
            nxt.append(torch.tensor([q[0] for q in pick]))
            nxt.append(torch.tensor([q[1] for q in pick]))
        cur = nxt
    return torch.stack(cur, 1)                      # nb x SEQ

def inside(leafvec):
    """leafvec: B x SEQ x NLEAF (ones where free, one-hot where observed).
    returns list over levels of B x nodes x V, plus root scalar."""
    lv = [leafvec]
    cur = leafvec
    for l in range(1, L + 1):
        p, le, ri, w = RT[l]
        nn = cur.shape[1] // 2
        lo = cur[:, 0::2, :]; hi = cur[:, 1::2, :]
        prod = lo[:, :, le] * hi[:, :, ri] * w             # B x nn x nrules
        out = torch.zeros(cur.shape[0], nn, V, dtype=torch.float64)
        out.index_add_(2, p, prod)
        lv.append(out); cur = out
    return lv

def outside(lv):
    """top-down; root prior uniform over V"""
    B = lv[0].shape[0]
    outs = [None] * (L + 1)
    outs[L] = torch.full((B, 1, V), 1.0 / V, dtype=torch.float64)
    for l in range(L, 0, -1):
        p, le, ri, w = RT[l]
        nn = lv[l].shape[1]
        lo_in = lv[l-1][:, 0::2, :]; hi_in = lv[l-1][:, 1::2, :]
        par_out = outs[l]                                   # B x nn x V
        msg = par_out[:, :, p] * w                          # B x nn x nrules
        lo_out = torch.zeros(B, nn, lv[l-1].shape[2], dtype=torch.float64)
        hi_out = torch.zeros(B, nn, lv[l-1].shape[2], dtype=torch.float64)
        lo_out.index_add_(2, le, msg * hi_in[:, :, ri])
        hi_out.index_add_(2, ri, msg * lo_in[:, :, le])
        child = torch.zeros(B, nn * 2, lv[l-1].shape[2], dtype=torch.float64)
        child[:, 0::2, :] = lo_out; child[:, 1::2, :] = hi_out
        outs[l-1] = child
    return outs

print(f"  L={L} v={V} leaves={NLEAF} seq={SEQ}  averaging {a.samples} sequences")
seqs = sample(a.samples)
B = a.samples
tot = torch.zeros(SEQ, dtype=torch.float64)
onehot = torch.zeros(B, SEQ, NLEAF, dtype=torch.float64)
onehot.scatter_(2, seqs.unsqueeze(-1), 1.0)
for t in range(SEQ):
    lf = torch.ones(B, SEQ, NLEAF, dtype=torch.float64)
    if t > 0: lf[:, :t, :] = onehot[:, :t, :]
    lv = inside(lf)
    outs = outside(lv)
    marg = outs[0][:, t, :]                        # B x NLEAF, unnormalised
    p = marg / marg.sum(-1, keepdim=True).clamp_min(1e-300)
    tot[t] = -torch.log(p.gather(1, seqs[:, t:t+1]).clamp_min(1e-300)).mean()

# ---- IS THE LATENT PARSE EVEN DETERMINED BY THE PREFIX? ----------------
# The transformer decodes the level-1 symbol at 0.48 and levels 3-4 at chance,
# and 48k steps of direct supervision with free labels could not install them.
# Before concluding that is an architectural failure, ask what the BEST
# POSSIBLE decoder could do: H(level-l symbol above position t | x_<t).
#
# The level-l node above leaf t covers leaves [t0, t0+2^l). Its posterior
# given the prefix is  inside(node) * outside(node), normalised.
lat = {l: np.zeros(SEQ) for l in range(1, L + 1)}
for t in range(SEQ):
    lf = torch.ones(B, SEQ, NLEAF, dtype=torch.float64)
    if t > 0: lf[:, :t, :] = onehot[:, :t, :]
    lv = inside(lf); outs = outside(lv)
    for l in range(1, L + 1):
        n = t >> l                                   # which level-l node
        post = lv[l][:, n, :] * outs[l][:, n, :]
        post = post / post.sum(-1, keepdim=True).clamp_min(1e-300)
        lat[l][t] = float(-(post * torch.log(post.clamp_min(1e-300))).sum(-1).mean())

H = tot.numpy()
loc = [H[t] for t in range(SEQ) if KIND[t] == "local"]
hig = [H[t] for t in range(SEQ) if KIND[t] == "higher"]
print("\n  EXACT H(x_t | x_<t), nats")
for t in range(SEQ): print(f"    t={t:<3} {KIND[t]:<7} {H[t]:.4f}")
print(f"\n  mean `local`            {np.mean(loc):.4f}   (model ~2.1)")
print(f"  mean `higher`           {np.mean(hig):.4f}   (model ~3.8)  <- real floor")
print(f"  t=0 alone               {H[0]:.4f}   (ln V = {math.log(V):.4f})")
print(f"  `higher` excluding t=0  {np.mean(hig[1:]):.4f}")

print(f"\n  LATENT PARSE: H(level-l symbol above position t | x_<t), nats")
print(f"  chance (uniform over {V} symbols) = {math.log(V):.4f}")
print(f"    {'t':>3} {'kind':<7}" + "".join(f"{'L'+str(l):>8}" for l in range(1, L+1)))
for t in range(SEQ):
    print(f"    {t:>3} {KIND[t]:<7}" + "".join(f"{lat[l][t]:>8.3f}" for l in range(1, L+1)))
print(f"\n    mean over `higher` positions:")
for l in range(1, L + 1):
    m = np.mean([lat[l][t] for t in range(SEQ) if KIND[t] == "higher"])
    frac = m / math.log(V)
    print(f"      level {l}: {m:.4f}   ({frac:.0%} of chance)")
print(f"\n  near chance  => the prefix does not determine that symbol, so NO")
print(f"                  decoder of any architecture can recover it, and the")
print(f"                  transformer's failure is a fact about the data")
print(f"  near zero    => the information IS in the prefix and the model is")
print(f"                  failing to extract it -- an architectural limit")
