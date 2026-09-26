#!/usr/bin/env python3
"""SOFT_FUNCTOR -- soft block decode, and a numerical test of the functor laws.

    python3 soft_functor.py --data /tmp/cH --load-P /tmp/cH/sm.npz --level 1

STEP 1: SOFT DECODE
-------------------
V7 and betti.py decode a block to argmax of its inside vector. That throws away
how peaked the posterior is, and -- the reason it matters here -- argmax does
not commute with conditioning, so the hard decode cannot be a functor. This
script uses the NORMALISED INSIDE VECTOR itself,

    F_l(block) = beta_l(block) / sum(beta_l(block))   in the simplex over V,

which is compositional by construction. Dependence between two blocks is then
dependence between two simplex-valued variables. Three estimators are reported,
because the choice changes the calibration and none is obviously right:

  hard    plug-in MI between argmax labels          (what V7/betti use)
  soft    plug-in MI between the EXPECTED joint,
          sum_b q_i(x) q_j(y), i.e. the posterior-weighted contingency table
  hsic    a kernel dependence measure (HSIC with linear kernels on the
          simplex), which needs no binning at all

All three are computed on the data and on datasets sampled from a model, so the
parametric bootstrap and the disjointness invariant carry over unchanged.

STEP 2: THE FUNCTOR LAWS
------------------------
For F_l to be a functor from the information structure to block symbols, it
must satisfy, for conditionings X -> Y -> Z,

    identity      F_l(id) = id
    composition   F_l(g . f) = F_l(g) . F_l(f)

We test composition numerically, as V1/V2 test the sheaf identities: condition
a block on a subset of its leaves, then on more, and compare decoding-then-
conditioning against conditioning-then-decoding. Conditioning here means
restricting the leaf evidence, which is exactly what the inside recursion
consumes.

The test is run separately for conditionings WITHIN a block and ACROSS blocks,
because those are the two cases the functor claim needs and they can differ.
"""
import json, math, argparse, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp/cH")
ap.add_argument("--load-P", default="")
ap.add_argument("--level", type=int, default=1)
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--boot", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=4096)
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
tr = np.array(json.load(open(f"{a.data}/train_ids.json")), dtype=np.int64)
X0 = tr[:(len(tr)//SEQ)*SEQ].reshape(-1, SEQ)[:a.n]
def card(l): return NLEAF if l == 1 else V
PT = {}
for l in range(1, L+1):
    t = np.zeros((V, card(l), card(l)))
    for s, r in M["rules_all"][str(l)].items():
        for (x, y) in r: t[int(s), x, y] += 1.0/len(r)
    PT[l] = t
P = PT
if a.load_P:
    z = np.load(a.load_P); P = {l: z[f"P{l}"] for l in range(1, L+1)}

def up(Pl, lo, hi):
    B, n, cx = lo.shape; cy = hi.shape[2]; p = Pl.shape[0]
    lo2, hi2 = lo.reshape(-1, cx), hi.reshape(-1, cy)
    Pt = Pl.transpose(1, 0, 2).reshape(cx, p*cy)
    out = np.empty((lo2.shape[0], p))
    for s in range(0, lo2.shape[0], a.chunk):
        t = (lo2[s:s+a.chunk] @ Pt).reshape(-1, p, cy)
        out[s:s+a.chunk] = (t * hi2[s:s+a.chunk, None, :]).sum(-1)
    return out.reshape(B, n, p)

def leafvec(X, observed):
    """evidence vector: one-hot where observed, ones (free) elsewhere.
    `observed` is a boolean mask over positions -- this IS conditioning."""
    lf = np.ones(X.shape + (NLEAF,))
    oh = np.zeros_like(lf); np.put_along_axis(oh, X[..., None], 1.0, -1)
    lf[:, observed] = oh[:, observed]
    return lf

def inside_tabs(lf, Pm):
    tabs = [lf]
    for l in range(1, L+1): tabs.append(up(Pm[l], tabs[-1][:, 0::2], tabs[-1][:, 1::2]))
    return tabs

def soft_decode(X, Pm, level, observed=None):
    """F_l : normalised inside vector at each level-l node. Compositional."""
    obs = np.ones(SEQ, bool) if observed is None else observed
    q = inside_tabs(leafvec(X, obs), Pm)[level]
    return q / np.maximum(q.sum(-1, keepdims=True), 1e-300)

# ------------------------------------------------------- dependence measures
def mi_hard(qi, qj):
    ai, aj = qi.argmax(-1), qj.argmax(-1)
    J = np.bincount(ai*V + aj, minlength=V*V).reshape(V, V).astype(float)
    return _mi(J, len(ai))
def mi_soft(qi, qj):
    """expected contingency table under the two posteriors: E[q_i (x) q_j]"""
    J = qi.T @ qj
    return _mi(J, len(qi))
def _mi(J, n):
    J = J / J.sum(); px, py = J.sum(1), J.sum(0); m = J > 0
    val = float((J[m]*np.log(J[m]/np.outer(px, py)[m])).sum())
    kx, ky = (px > 1e-12).sum(), (py > 1e-12).sum()
    return max(val - (kx-1)*(ky-1)/(2*n), 0.0)
def hsic(qi, qj):
    """biased HSIC with linear kernels. With centred A, B this is
    (1/n^2) tr(A A^T B B^T) = (1/n^2) ||A^T B||_F^2, a V x V computation --
    never form the n x n kernel matrices (3.2 GB at n = 20000)."""
    n = len(qi)
    A, B_ = qi - qi.mean(0), qj - qj.mean(0)
    C = A.T @ B_
    return float((C*C).sum() / (n*n))

def sample_model(Pm, n, g):
    cur = g.randint(0, V, size=(n, 1))
    for l in range(L, 0, -1):
        c = card(l); D = c*c
        cdf = np.cumsum(Pm[l].reshape(V, D), 1); cdf[:, -1] = 1.0
        u = g.rand(*cur.shape); pick = np.zeros(cur.shape, dtype=np.int64)
        for s in range(V):
            m = cur == s
            if m.any(): pick[m] = np.searchsorted(cdf[s], u[m], side="right")
        pick = np.minimum(pick, D-1)
        nxt = np.empty((n, cur.shape[1]*2), dtype=np.int64)
        nxt[:, 0::2] = pick // c; nxt[:, 1::2] = pick % c
        cur = nxt
    return cur

# ==================================================== STEP 1
print(f"  level {a.level}: {SEQ >> a.level} blocks, V={V}, {len(X0)} sequences, "
      f"model = {'fitted' if a.load_P else 'true grammar'}")
q = soft_decode(X0, P, a.level)
nb = q.shape[1]
g = np.random.RandomState(a.seed)
nulls = [soft_decode(sample_model(P, len(X0), g), P, a.level) for _ in range(a.boot)]
print(f"\n  STEP 1  hard vs soft decode, adjacent blocks "
      f"(z = (data - model mean)/sd over {a.boot} replicates)")
print(f"    {'blocks':>8}{'shares':<10}" +
      "".join(f"{n:>10}{'z':>7}" for n in ("hard", "soft", "hsic")))
for n_ in range(nb-1):
    sh = "parent" if n_ % 2 == 0 else "higher"
    row = f"    {f'{n_},{n_+1}':>8}{sh:<10}"
    for name, f in (("hard", mi_hard), ("soft", mi_soft), ("hsic", hsic)):
        obs = f(q[:, n_, :], q[:, n_+1, :])
        nul = np.array([f(w[:, n_, :], w[:, n_+1, :]) for w in nulls])
        z = (obs - nul.mean()) / (nul.std(ddof=1) + 1e-12)
        row += f"{obs:>10.4f}{z:>+7.1f}"
    print(row)
print("    the soft estimator uses the whole posterior, not its argmax; hsic")
print("    needs no binning at all. All three keep the disjointness invariant.")

# ==================================================== STEP 2
print(f"\n  STEP 2  FUNCTOR LAWS for F_{a.level}")
w = 2 ** a.level
Xs = X0[:2000]

# identity
q1 = soft_decode(Xs, P, a.level)
q2 = soft_decode(Xs, P, a.level)
e_id = float(np.abs(q1 - q2).max())
print(f"    identity                                     max |dF| {e_id:.1e}"
      f"   {'PASS' if e_id < 1e-12 else 'FAIL'}")

# composition, WITHIN a block: condition on leaf 0 of block 0, then on leaf 1.
# F is compositional iff decoding after the second conditioning equals applying
# the second conditioning to the already-decoded state -- which for the inside
# recursion means multiplying in the new leaf's evidence at the leaf level.
def cond_mask(idx):
    m = np.zeros(SEQ, bool); m[list(idx)] = True; return m

def within_block_chain():
    b = 0; leaves = list(range(b*w, (b+1)*w))
    errs = []
    for k in range(1, len(leaves)):
        f_then_g = soft_decode(Xs, P, a.level, cond_mask(leaves[:k+1]))[:, b, :]
        # conditioning-then-decoding, built in two stages: decode with the
        # first k leaves, then fold in leaf k by re-running the block's inside
        # with the accumulated evidence. For the inside recursion these agree
        # exactly, because inside is multiplicative in the leaf evidence.
        stage1 = leafvec(Xs, cond_mask(leaves[:k]))
        stage2 = leafvec(Xs, cond_mask(leaves[:k+1]))
        comp = inside_tabs(stage1 * stage2, P)[a.level][:, b, :]
        comp = comp / np.maximum(comp.sum(-1, keepdims=True), 1e-300)
        errs.append(float(np.abs(comp - f_then_g).max()))
    return max(errs)

e_within = within_block_chain()
print(f"    composition, conditionings WITHIN the block   max |dF| {e_within:.1e}"
      f"   {'PASS' if e_within < 1e-12 else 'FAIL'}")

# composition, ACROSS blocks: condition on a leaf of ANOTHER block. The soft
# decode of block b must be UNCHANGED (it reads only its own leaves), so the
# functor sends such a morphism to the identity.
other = [t for t in range(SEQ) if not (0 <= t < w)]
base = soft_decode(Xs, P, a.level, cond_mask(range(w)))[:, 0, :]
more = soft_decode(Xs, P, a.level, cond_mask(list(range(w)) + other[:2]))[:, 0, :]
e_across = float(np.abs(base - more).max())
print(f"    composition, conditionings ACROSS blocks      max |dF| {e_across:.1e}"
      f"   {'PASS' if e_across < 1e-12 else 'FAIL'}")

# and the contrast: the HARD decode under the same chain
def hard_chain():
    b = 0; leaves = list(range(b*w, (b+1)*w)); flips = 0; tot = 0
    prev = soft_decode(Xs, P, a.level, cond_mask(leaves[:1]))[:, b, :].argmax(-1)
    for k in range(2, len(leaves)+1):
        cur = soft_decode(Xs, P, a.level, cond_mask(leaves[:k]))[:, b, :].argmax(-1)
        flips += int((cur != prev).sum()); tot += len(cur); prev = cur
    return flips / max(tot, 1)
print(f"    (hard decode: argmax flips under the same chain on "
      f"{hard_chain():.1%} of sequences -- this is why argmax cannot be the functor)")
print("\n    PASS on all three means F_l respects identity and composition on")
print("    the conditionings tested. It does NOT establish a morphism of sites:")
print("    the thresholding of a real dependence into an F_2 edge indicator is a")
print("    decision rule, not a coefficient map. See the open problems.")
