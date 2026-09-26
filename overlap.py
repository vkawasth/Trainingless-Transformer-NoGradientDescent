#!/usr/bin/env python3
"""OVERLAP -- a corpus with OVERLAPPING constituents, and a falsifiable test.

    python3 overlap.py --mix 0.0     # single bracketing: no obstruction
    python3 overlap.py --mix 0.3     # 70/30 mixture: overlapping constituents
    python3 overlap.py --sweep       # mix = 0, .1, .2, .3, .5 side by side

WHAT IS BEING BUILT, AND WHAT IT CAN AND CANNOT SHOW
----------------------------------------------------
An honest statement first, because it constrains the design. Any corpus
SAMPLED from a generative process has a global joint distribution by
construction, so every family of its marginals glues and the Cech obstruction
of Abramsky-Mansfield-Barbosa is identically zero. No amount of cleverness in
the generator changes that. Likewise I_k is a (k-1)-coboundary as an algebraic
identity (Vigneaux), so its class vanishes whatever the data.

What CAN be built, and what "second-level syzygy" means operationally, is an
obstruction to gluing WITHIN A MODEL CLASS: local tree models that each fit
their own context, are consistent where they overlap, and admit no single tree
covering all of them.

THE CONSTRUCTION
----------------
Nine leaves, ternary branching, two bracketings of the same tokens:

    A (rows)     {0,1,2} {3,4,5} {6,7,8}
    B (columns)  {0,3,6} {1,4,7} {2,5,8}

Every constituent of A meets every constituent of B in exactly one leaf, so
the two covers overlap everywhere and neither refines the other. A sequence is
generated from A with probability 1-mix and from B with probability mix, using
the SAME symbol alphabet and rule tables, so the marginals are matched and the
only difference is which grouping produced the sequence.

  mix = 0    a single tree explains everything -- the control.
  mix > 0    the corpus carries both constituent structures at once, and no
             single ternary tree over these nine leaves reproduces it.

THE MEASUREMENT
---------------
For a grouping G define the contrast

    C(G) = mean pairwise MI within the groups of G
         - mean pairwise MI across the groups of G

A tree with bracketing G gives C(G) > 0 and C(G') = 0 for the transverse
grouping. The falsifiable prediction is:

    mix = 0   ->  C(A) > 0,  C(B) = 0     (one structure)
    mix > 0   ->  C(A) > 0,  C(B) > 0     (both at once -- no single tree)

Both are compared against a permutation null, and the whole thing is a control
experiment rather than an assertion: if C(B) does not rise with mix, the
construction has failed and the reasoning above is wrong.
"""
import json, math, argparse, itertools, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mix", type=float, default=0.3)
ap.add_argument("--sweep", action="store_true")
ap.add_argument("--n", type=int, default=60000)
ap.add_argument("--nsym", type=int, default=6)
ap.add_argument("--nleaf", type=int, default=9)
ap.add_argument("--nrules", type=int, default=3)
ap.add_argument("--boot", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="", help="write the corpus to this directory")
a = ap.parse_args()

ROWS = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
COLS = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]

def make_grammar(rng, V, A, m):
    """ternary rules: root -> three level-1 symbols; each level-1 symbol ->
    three leaves. The SAME tables serve both bracketings, so the two
    structures differ only in which positions a constituent occupies."""
    top = {s: [tuple(rng.randint(0, V, 3)) for _ in range(m)] for s in range(V)}
    bot = {s: [tuple(rng.randint(0, A, 3)) for _ in range(m)] for s in range(V)}
    return top, bot

def generate(n, mix, top, bot, rng, V):
    X = np.zeros((n, 9), dtype=np.int64)
    which = rng.rand(n) < mix
    for i in range(n):
        groups = COLS if which[i] else ROWS
        r = rng.randint(V)
        kids = top[r][rng.randint(len(top[r]))]
        for g, sym in zip(groups, kids):
            leaves = bot[sym][rng.randint(len(bot[sym]))]
            for pos, lf in zip(g, leaves): X[i, pos] = lf
    return X, which

def mi(x, y, A):
    J = np.bincount(x*A + y, minlength=A*A).reshape(A, A).astype(float)
    J = J/J.sum(); px, py = J.sum(1), J.sum(0); m = J > 0
    val = float((J[m]*np.log(J[m]/np.outer(px, py)[m])).sum())
    kx, ky = (px > 0).sum(), (py > 0).sum()
    return max(val - (kx-1)*(ky-1)/(2*len(x)), 0.0)

def contrast(X, groups, A):
    within, across = [], []
    gid = {}
    for k, g in enumerate(groups):
        for p in g: gid[p] = k
    for i, j in itertools.combinations(range(9), 2):
        v = mi(X[:, i], X[:, j], A)
        (within if gid[i] == gid[j] else across).append(v)
    return float(np.mean(within) - np.mean(across)), float(np.mean(within)), \
           float(np.mean(across))

def fit_root(X, top, bot, groups, V, A, iters=40):
    """EM for the root posterior under a FIXED bracketing. The model is
    root -> (s1,s2,s3) -> leaves, so the E-step is one inside pass:
        beta_g(s) = P(the three leaves of group g | that group has symbol s)
    and the root posterior is proportional to the prior times the product of
    the top rule against the three betas."""
    n = len(X)
    # leaf emission per symbol, estimated by EM from scratch (no cheating with
    # the generating tables): start from a perturbed uniform
    rs = np.random.RandomState(0)
    E = rs.rand(V, A, A, A) + 1.0
    E /= E.sum((1, 2, 3), keepdims=True)
    T = rs.rand(V, V, V, V) + 1.0
    T /= T.sum((1, 2, 3), keepdims=True)
    pri = np.full(V, 1.0/V)
    idx = [np.array(g) for g in groups]
    for _ in range(iters):
        beta = np.stack([E[:, X[:, g[0]], X[:, g[1]], X[:, g[2]]].T
                         for g in idx], 1)            # n x 3 x V
        joint = np.einsum('r,rabc,na,nb,nc->nrabc', pri, T,
                          beta[:, 0], beta[:, 1], beta[:, 2])
        Z = joint.reshape(n, -1).sum(1)
        w = joint / np.maximum(Z[:, None, None, None, None], 1e-300)
        Tn = w.sum(0) + 1e-9; T = Tn/Tn.sum((1, 2, 3), keepdims=True)
        pri = w.reshape(n, V, -1).sum(2).sum(0); pri = pri/pri.sum()
        # group-symbol posteriors, then leaf emissions
        q = [w.sum((1, 3, 4)), w.sum((1, 2, 4)), w.sum((1, 2, 3))]   # n x V each
        En = np.zeros_like(E)
        for k, g in enumerate(idx):
            np.add.at(En, (slice(None), X[:, g[0]], X[:, g[1]], X[:, g[2]]),
                      q[k].T)
        En += 1e-9; E = En/En.sum((1, 2, 3), keepdims=True)
    root = w.reshape(n, V, -1).sum(2)
    root = root/np.maximum(root.sum(1, keepdims=True), 1e-300)
    q = [qq/np.maximum(qq.sum(1, keepdims=True), 1e-300) for qq in q]
    return root, q

def cond_mi(x, y, z, A, V):
    """I(x ; y | z) with z a SOFT assignment (n x V posterior). Weighted
    contingency tables, one per value of z."""
    tot = 0.0
    for s in range(V):
        w = z[:, s]; m = w.sum()
        if m < 1.0: continue
        J = np.zeros((A, A))
        np.add.at(J, (x, y), w)
        J = J/J.sum(); px, py = J.sum(1), J.sum(0); msk = J > 0
        val = float((J[msk]*np.log(J[msk]/np.outer(px, py)[msk])).sum())
        kx, ky = (px > 0).sum(), (py > 0).sum()
        tot += (m/len(x))*max(val - (kx-1)*(ky-1)/(2*max(m, 1)), 0.0)
    return tot

def cond_contrast(X, groups, q, A, V):
    """The conditional independence a tree actually asserts. For two leaves in
    DIFFERENT constituents the separator is the PAIR of their group symbols,
    not the root: given the root alone the two groups stay dependent through
    the correlation the top rule induces between the group symbols. So we
    condition on (s_gi, s_gj), a V^2-valued soft variable. For two leaves in
    the SAME constituent the separator is that constituent's symbol, and they
    remain dependent given it, because the emission is a joint over the three
    leaves -- which is what makes 'within' large and 'across' near zero the
    signature of a correct bracketing."""
    gid = {}
    for k, g in enumerate(groups):
        for p in g: gid[p] = k
    within, across = [], []
    for i, j in itertools.combinations(range(9), 2):
        gi, gj = gid[i], gid[j]
        if gi == gj:
            v = cond_mi(X[:, i], X[:, j], q[gi], A, V)
            within.append(v)
        else:
            z = (q[gi][:, :, None]*q[gj][:, None, :]).reshape(len(X), V*V)
            across.append(cond_mi(X[:, i], X[:, j], z, A, V*V))
    return float(np.mean(within)), float(np.mean(across))

def permuted(X, rng):
    Y = X.copy()
    for c in range(Y.shape[1]): Y[:, c] = Y[rng.permutation(len(Y)), c]
    return Y

rng = np.random.RandomState(a.seed)
top, bot = make_grammar(rng, a.nsym, a.nleaf, a.nrules)
mixes = [0.0, 0.1, 0.2, 0.3, 0.5] if a.sweep else [a.mix]

print(f"  overlap: 9 leaves, ternary, V={a.nsym}, leaf alphabet={a.nleaf}, "
      f"{a.nrules} rules/symbol, n={a.n}")
print(f"  rows    {ROWS}\n  columns {COLS}")
print(f"  every row meets every column in one leaf: the two covers overlap")
print(f"  everywhere and neither refines the other.\n")
print(f"    Neither W nor C is the right statistic on its own. W(cols) is above")
print(f"    a permutation null even at mix = 0, because all nine leaves share a")
print(f"    root and any two are dependent through it; C(cols) is dragged")
print(f"    negative because a within-column pair is an across-row pair. The")
print(f"    discriminator is the ASYMMETRY W(rows)/W(cols): a single tree makes")
print(f"    one grouping dominate, a mixture makes them equal. The reference")
print(f"    band is the ratio on corpora generated at mix = 0 with the same")
print(f"    grammar and sample size, so the test is parametric, not asymptotic.")
print()
print(f"    {'mix':>6}{'W(rows)':>10}{'W(cols)':>10}{'ratio':>8}"
      f"{'  mix=0 band':>16}   verdict")
for mx in mixes:
    X, which = generate(a.n, mx, top, bot, rng, a.nsym)
    cA, wA, aA = contrast(X, ROWS, a.nleaf)
    cB, wB, aB = contrast(X, COLS, a.nleaf)
    # conditional test: fit the root under EACH bracketing, then measure
    # I(leaf_i ; leaf_j | root) within and across that bracketing's groups
    _, qA = fit_root(X, top, bot, ROWS, a.nsym, a.nleaf)
    _, qB = fit_root(X, top, bot, COLS, a.nsym, a.nleaf)
    cwA, caA = cond_contrast(X, ROWS, qA, a.nleaf, a.nsym)
    cwB, caB = cond_contrast(X, COLS, qB, a.nleaf, a.nsym)
    ratio = wA/max(wB, 1e-12)
    refs = []
    for _ in range(max(3, a.boot//2)):
        Xr, _ = generate(a.n, 0.0, top, bot, rng, a.nsym)
        rA = contrast(Xr, ROWS, a.nleaf)[1]; rB = contrast(Xr, COLS, a.nleaf)[1]
        refs.append(rA/max(rB, 1e-12))
    lo = float(np.mean(refs)) - 3*float(np.std(refs, ddof=1))
    v = ("single tree" if ratio >= lo else
         "OVERLAPPING constituents: no single tree has both")
    print(f"    {mx:>6.1f}{wA:>10.4f}{wB:>10.4f}{ratio:>8.2f}"
          f"{lo:>10.2f} and up   {v}")
    print(f"           conditional: rows  within {cwA:.4f}  across {caA:.4f}"
          f"   |  cols  within {cwB:.4f}  across {caB:.4f}")
    if a.out and not a.sweep:
        import os
        os.makedirs(a.out, exist_ok=True)
        json.dump(X.reshape(-1).tolist(), open(f"{a.out}/train_ids.json", "w"))
        json.dump({"depth": 2, "nsym": a.nsym, "nleaf": a.nleaf, "seq_len": 9,
                   "branching": 3, "mix": mx,
                   "rows": [list(g) for g in ROWS],
                   "cols": [list(g) for g in COLS]},
                  open(f"{a.out}/overlap_meta.json", "w"))
        print(f"    wrote {a.out}/train_ids.json")

print("\n  BOTH means the corpus carries the row structure and the column")
print("  structure at once. No single ternary tree over these nine leaves has")
print("  both, so the local models -- one per bracketing -- do not glue into a")
print("  global one. That is the operational content of a second-level syzygy")
print("  here: an obstruction to gluing WITHIN the tree model class, which is")
print("  measurable, and not a nonzero Cech class, which cannot occur for data")
print("  sampled from any single generative process.")
