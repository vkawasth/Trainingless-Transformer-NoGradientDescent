#!/usr/bin/env python3
"""ESTIMATOR4 -- estimator3 plus V6: calibrated detection of where the tree fails.

    python3 estimator3.py --data /tmp --n-train 4000 --polish 20

Pure numpy (no torch), gradient-free. Fits the grammar exactly as estimator2
(spectral init + soft hand-off + EM), then verifies, on the fitted model:

  V0  EM ascent          train loglik never decreases across sweeps
                         (Dempster-Laird-Rubin)
  V1  sheaf condition    local sections restrict and glue:
                         (a) a free subtree has inside value exactly 1
                         (b) restriction = marginalisation: the left-half
                             local model equals the full model with the
                             right half free
                         (c) gluing = independence join: the full likelihood
                             equals  sum_uv mu(u,v) beta_L(u) beta_R(v)
                         (Simpson Thm 6.4 / Prop 8.5; Wright s7.1)
  V2  supports / soft    predicting x_t from the leaves equals predicting it
      hand-off           from the NORMALISED posterior of each completed
                         subtree, with those leaves deleted: the hand-off is
                         a sufficient statistic (Simpson Def 7.1)
  V3  hierarchy check    mutual information between leaf positions across
      (data-facing)      subtree boundaries: empirical vs fitted vs true grammar
  V4  merges             (a) identity: the drop in expected complete-data
                             loglik from merging symbols into their mixture
                             row equals  sum n_s KL(R_s || R_class)
                             = N * I(children ; B | q(B))
                         (b) sweep K per level: conditional MI paid, and the
                             held-out loglik actually lost

  V6  obstruction        for each pair of leaf positions, empirical MI in the
      detector           data against MI in datasets SAMPLED FROM THE FITTED
                         MODEL at the same size (parametric bootstrap). A large
                         positive z = dependence the tree model cannot produce.
                         Calibrated by two controls (true grammar vs its own
                         samples; true grammar vs the data) and validated by a
                         PLANTED obstruction: --plant i,j,q copies leaf i into
                         leaf j with probability q, a dependence no depth-L tree
                         of this shape carries directly.

V0-V2 and V4a are identities of the tree model class. Passing them verifies the
implementation and that the fitted object IS a sheaf section; it does not say
the data are tree-generated. V3 and V4b are the data-facing measurements.
"""
import json, math, argparse, time, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp")
ap.add_argument("--n-train", type=int, default=4000)
ap.add_argument("--n-val", type=int, default=2000)
ap.add_argument("--polish", type=int, default=20)
ap.add_argument("--ks", default="64,32,16,8,4,2")
ap.add_argument("--refit", type=int, default=0, help="EM sweeps after tying")
ap.add_argument("--n-check", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--save", default="")
ap.add_argument("--plant", default="", help="i,j,q: copy leaf i into leaf j w.p. q")
ap.add_argument("--boot", type=int, default=20, help="bootstrap replicates for V6")
ap.add_argument("--boot-n", type=int, default=40000, help="sequences per replicate")
ap.add_argument("--track", default="0,1;1,2;3,4;7,8",
                help="pairs whose data-vs-model MI gap is printed every sweep")
ap.add_argument("--skip-v4", action="store_true")
ap.add_argument("--z-flag", type=float, default=4.0)
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
KIND = M["pos_kind"]
if "rules_all" not in M:
    raise SystemExit("rhm_meta.json has no rules_all: rebuild with the updated "
                     "build_corpus_syn.py")
tr = np.array(json.load(open(f"{a.data}/train_ids.json")), dtype=np.int64)
va = np.array(json.load(open(f"{a.data}/val_ids.json")), dtype=np.int64)
TRALL = tr[:(len(tr)//SEQ)*SEQ].reshape(-1, SEQ)
TR = TRALL[:a.n_train]
VA = va[:(len(va)//SEQ)*SEQ].reshape(-1, SEQ)[:a.n_val]
rng = np.random.RandomState(a.seed)
PLANT = None
if a.plant:
    pi_, pj_, pq_ = a.plant.split(","); PLANT = (int(pi_), int(pj_), float(pq_))
    prng = np.random.RandomState(12345)
    for X_ in (TRALL, VA):
        m_ = prng.rand(len(X_)) < PLANT[2]
        X_[m_, PLANT[1]] = X_[m_, PLANT[0]]
    TR = TRALL[:a.n_train]
    print(f"  PLANTED obstruction: leaf {PLANT[0]} copied into leaf {PLANT[1]} "
          f"with probability {PLANT[2]}")
def card(l): return NLEAF if l == 1 else V
TOL = 1e-9
verdict = collections.OrderedDict()

print(f"  L={L} v={V} leaves={NLEAF} seq={SEQ}   fit {len(TR)} seqs, "
      f"score {len(VA)}, empirical MI from {len(TRALL)}")

# ============================================================ contractions
# every level is the same contraction  out[bn,p] = sum_xy P[p,x,y] lo[bn,x] hi[bn,y]
def up(P, lo, hi):
    B, n, cx = lo.shape; cy = hi.shape[2]; p = P.shape[0]
    lo2, hi2 = lo.reshape(-1, cx), hi.reshape(-1, cy)
    Pt = P.transpose(1, 0, 2).reshape(cx, p * cy)
    out = np.empty((lo2.shape[0], p))
    for s in range(0, lo2.shape[0], a.chunk):
        t = (lo2[s:s+a.chunk] @ Pt).reshape(-1, p, cy)
        out[s:s+a.chunk] = (t * hi2[s:s+a.chunk, None, :]).sum(-1)
    return out.reshape(B, n, p)

def down(P, po, lo, hi):
    """messages to the two children given the parent's outside message"""
    B, n, p = po.shape; cx, cy = lo.shape[2], hi.shape[2]
    po2, lo2, hi2 = po.reshape(-1, p), lo.reshape(-1, cx), hi.reshape(-1, cy)
    Pf = P.reshape(p, cx * cy)
    lo_o = np.empty((po2.shape[0], cx)); hi_o = np.empty((po2.shape[0], cy))
    for s in range(0, po2.shape[0], a.chunk):
        t = (po2[s:s+a.chunk] @ Pf).reshape(-1, cx, cy)
        lo_o[s:s+a.chunk] = (t * hi2[s:s+a.chunk, None, :]).sum(-1)
        hi_o[s:s+a.chunk] = (t * lo2[s:s+a.chunk, :, None]).sum(-2)
    return lo_o.reshape(B, n, cx), hi_o.reshape(B, n, cy)

def counts(P, po, lo, hi, w):
    """expected rule counts  C[p,x,y] = P[p,x,y] * sum_bn w_b po lo hi"""
    B, n, p = po.shape; cx, cy = lo.shape[2], hi.shape[2]
    po2 = (po * w[:, None, None]).reshape(-1, p)
    lo2, hi2 = lo.reshape(-1, cx), hi.reshape(-1, cy)
    acc = np.zeros((p, cx * cy))
    for s in range(0, po2.shape[0], a.chunk):
        o = (lo2[s:s+a.chunk, :, None] * hi2[s:s+a.chunk, None, :]).reshape(-1, cx*cy)
        acc += po2[s:s+a.chunk].T @ o
    return acc.reshape(p, cx, cy) * P

def onehot(X):
    o = np.zeros(X.shape + (NLEAF,)); np.put_along_axis(o, X[..., None], 1.0, -1)
    return o

def inside(lf, P, inject=None):
    tabs = [lf]
    for l in range(1, L+1):
        out = up(P[l], tabs[-1][:, 0::2], tabs[-1][:, 1::2])
        if inject:
            for (ll, n), vec in inject.items():
                if ll == l: out[:, n, :] = vec
        tabs.append(out)
    return tabs

def outside(tabs, P):
    B = tabs[0].shape[0]
    outs = [None]*(L+1)
    outs[L] = np.full((B, 1, V), 1.0/V)
    for l in range(L, 0, -1):
        lo, hi = tabs[l-1][:, 0::2], tabs[l-1][:, 1::2]
        lo_o, hi_o = down(P[l], outs[l], lo, hi)
        ch = np.zeros((B, lo.shape[1]*2, lo.shape[2]))
        ch[:, 0::2] = lo_o; ch[:, 1::2] = hi_o
        outs[l-1] = ch
    return outs

def Zof(tabs): return tabs[L][:, 0, :].sum(-1) / V

def loglik(X, P):
    return float(np.log(np.maximum(Zof(inside(onehot(X), P)), 1e-300)).mean())

def estep(X, P):
    tabs = inside(onehot(X), P); outs = outside(tabs, P)
    Z = np.maximum(Zof(tabs), 1e-300)
    C = {l: counts(P[l], outs[l], tabs[l-1][:, 0::2], tabs[l-1][:, 1::2], 1.0/Z)
         for l in range(1, L+1)}
    return C, float(np.log(Z).mean())

def mstep(C, cls=None):
    P = {}
    for l in range(1, L+1):
        c = C[l] + 1e-9
        if cls is not None and cls.get(l) is not None:
            k = cls[l]; pooled = np.zeros_like(c)
            for s in range(V): pooled[k[s]] += c[s]
            c = pooled[k]                       # every member gets its class row
        P[l] = c / c.sum((1, 2), keepdims=True)
    return P

# ============================================================ init (estimator2)
t0 = time.time()
soft = onehot(TR)
P = {}
for l in range(1, L+1):
    c = card(l)
    hard = soft.argmax(-1)
    lo, hi = hard[:, 0::2], hard[:, 1::2]
    nn_ = lo.shape[1]
    pid = lo * c + hi
    ctx = collections.defaultdict(collections.Counter)
    for row in pid.tolist():
        for i, p_ in enumerate(row):
            if i > 0: ctx[p_][row[i-1]] += 1
            if i + 1 < len(row): ctx[p_][row[i+1]] += 1
    cnt_all = collections.Counter(pid.reshape(-1).tolist())
    pairs = sorted(cnt_all); idx = {p_: i for i, p_ in enumerate(pairs)}
    use_ctx = nn_ >= 4 and len(ctx) >= 2
    if use_ctx:
        Cm = np.zeros((len(pairs), len(pairs)))
        for p_, d in ctx.items():
            for q, n in d.items():
                if q in idx: Cm[idx[p_], idx[q]] = n
        rs = Cm.sum(1, keepdims=True); rs[rs == 0] = 1
        U, S, _ = np.linalg.svd(Cm / rs, full_matrices=False)
        k = max(2, min(V, min(Cm.shape) - 1))
        emb = U[:, :k] * S[:k]
    else:
        emb = np.zeros((len(pairs), 2 * c))
        for p_ in pairs:
            emb[idx[p_], p_ // c] = 1.0; emb[idx[p_], c + (p_ % c)] = 1.0
    ncl = max(1, min(V, len(pairs)))
    g_ = np.random.RandomState(a.seed)           # as estimator2: re-seeded per level
    cen = emb[g_.choice(len(pairs), ncl, replace=False)].astype(float)
    for _ in range(25):
        lab = ((emb[:, None, :] - cen[None]) ** 2).sum(-1).argmin(1)
        for j in range(ncl):
            m = lab == j
            if m.any(): cen[j] = emb[m].mean(0)
    tab = np.zeros((V, c, c))
    for p_ in pairs: tab[lab[idx[p_]] % V, p_ // c, p_ % c] += cnt_all[p_]
    tab += 1e-6
    P[l] = tab / tab.sum((1, 2), keepdims=True)
    print(f"    init level {l}: {len(pairs)} distinct pairs "
          f"[{'context' if use_ctx else 'child-identity'}]")
    soft = up(P[l], soft[:, 0::2], soft[:, 1::2])
    soft = soft / np.maximum(soft.sum(-1, keepdims=True), 1e-300)

# ============================================================ MI helpers
def mi(J):
    J = J / J.sum(); px, py = J.sum(1), J.sum(0); m = J > 0
    return float((J[m] * np.log(J[m] / np.outer(px, py)[m])).sum())
def model_joint(P, i, j):
    A = np.repeat(np.arange(NLEAF), NLEAF); Bv = np.tile(np.arange(NLEAF), NLEAF)
    lf = np.ones((NLEAF * NLEAF, SEQ, NLEAF))
    lf[:, i] = 0; lf[np.arange(len(A)), i, A] = 1
    lf[:, j] = 0; lf[np.arange(len(A)), j, Bv] = 1
    return Zof(inside(lf, P)).reshape(NLEAF, NLEAF)
def emp_mi(X, i, j):
    J = np.bincount(X[:, i] * NLEAF + X[:, j], minlength=NLEAF * NLEAF)
    return mi(J.reshape(NLEAF, NLEAF).astype(float))
TRACK = [tuple(int(v) for v in q.split(",")) for q in a.track.split(";") if q]
def emp_mi_mm(X, i, j):
    """Miller-Madow bias-corrected plug-in MI (for the tracker only; the
    bootstrap needs no correction because both sides share the estimator)"""
    J = np.bincount(X[:, i] * NLEAF + X[:, j], minlength=NLEAF*NLEAF).reshape(NLEAF, NLEAF)
    kx, ky = (J.sum(1) > 0).sum(), (J.sum(0) > 0).sum()
    return mi(J.astype(float)) - (kx - 1) * (ky - 1) / (2 * len(X))
EMP_TRACK = {pq: emp_mi_mm(TRALL, *pq) for pq in TRACK}

# ============================================================ V0: EM ascent
print("\n  V0  EM ASCENT (train loglik must never decrease)")
hist = []
best_P, best_va, best_it = {l: P[l].copy() for l in P}, loglik(VA, P), 0
for it in range(a.polish):
    C, ll_tr = estep(TR, P)
    P = mstep(C)
    hist.append(ll_tr)
    va_ = loglik(VA, P)
    if va_ > best_va: best_P, best_va, best_it = {l: P[l].copy() for l in P}, va_, it + 1
    if it < 3 or it == a.polish - 1 or (it + 1) % 10 == 0:
        gaps = "  ".join(f"{pq}:{EMP_TRACK[pq] - mi(model_joint(P, *pq)):+.4f}"
                         for pq in TRACK)
        print(f"    sweep {it+1:3d}: train {ll_tr:.5f}   held-out {loglik(VA, P):.5f}"
              f"   data-model MI gap  {gaps}")
C, ll_final = estep(TR, P)
hist.append(ll_final)
d = np.diff(hist)
P = best_P                                       # as estimator2: best held-out sweep
C, ll_final = estep(TR, P)
print(f"    keeping sweep {best_it} (best held-out {best_va:.5f}), as estimator2 does")
worst = float(d.min()) if len(d) else 0.0
verdict["V0 EM ascent"] = worst > -1e-8
print(f"    smallest step {worst:+.2e}  over {len(d)} sweeps  "
      f"-> {'PASS' if verdict['V0 EM ascent'] else 'FAIL'}")
t_fit = time.time() - t0

# ============================================================ scoring
def score(P, X):
    oh = onehot(X); tot = np.zeros(SEQ)
    for t in range(SEQ):
        lf = np.ones_like(oh); lf[:, :t] = oh[:, :t]
        marg = outside(inside(lf, P), P)[0][:, t, :]
        pr = marg / np.maximum(marg.sum(-1, keepdims=True), 1e-300)
        tot[t] = -np.log(np.maximum(pr[np.arange(len(X)), X[:, t]], 1e-300)).mean()
    lo_ = np.mean([tot[t] for t in range(SEQ) if KIND[t] == "local"])
    hi_ = np.mean([tot[t] for t in range(SEQ) if KIND[t] == "higher"])
    return lo_, hi_

PT = {}
for l in range(1, L+1):
    t_ = np.zeros((V, card(l), card(l)))
    for s, rules in M["rules_all"][str(l)].items():
        for (x, y) in rules: t_[int(s), x, y] += 1.0 / len(rules)
    PT[l] = t_
lo_f, hi_f = score(P, VA)
lo_t, hi_t = score(PT, VA)
print(f"\n  SCORES  (fit {t_fit:.0f}s, no gradients)")
print(f"    {'':<24}{'local':>9}{'higher':>9}{'loglik/seq':>12}")
print(f"    {'fitted':<24}{lo_f:>9.4f}{hi_f:>9.4f}{loglik(VA, P):>12.4f}"
      f"   (train {ll_final:.4f})")
print(f"    {'true grammar':<24}{lo_t:>9.4f}{hi_t:>9.4f}{loglik(VA, PT):>12.4f}")
print(f"    {'transformer 48k steps':<24}{2.0223:>9.4f}{3.7716:>9.4f}")

# ============================================================ V1: sheaf condition
print("\n  V1  SHEAF CONDITION on the fitted model")
X = VA[:a.n_check]; oh = onehot(X); H = SEQ // 2
full = Zof(inside(oh, P))
free = inside(np.ones_like(oh), P)
e_a = max(float(np.abs(free[l] - 1).max()) for l in range(1, L+1))
# left and right halves as local sections (inside at the two level-(L-1) nodes)
lfL = oh.copy(); lfL[:, H:] = 1.0
lfR = oh.copy(); lfR[:, :H] = 1.0
tL, tR = inside(lfL, P), inside(lfR, P)
bL, bR = tL[L-1][:, 0, :], tR[L-1][:, 1, :]            # P(half | u), P(half | v)
mu = P[L].sum(0) / V                                   # joint prior on (u, v)
glued = np.einsum('uv,bu,bv->b', mu, bL, bR)
e_c = float(np.abs(glued / full - 1).max())
restr_local = bL @ mu.sum(1)                           # left-local model
restr_full = Zof(tL)                                   # full model, right free
e_b = float(np.abs(restr_local / restr_full - 1).max())
for key, e, meaning in (("V1a free subtree = 1", e_a, "free subtree has inside value 1"),
                        ("V1b restriction", e_b, "restriction = marginalisation"),
                        ("V1c gluing", e_c, "gluing = independence join")):
    verdict[key] = e < TOL
    print(f"    {meaning:<34} max err {e:.1e}  -> {'PASS' if e < TOL else 'FAIL'}")

# ============================================================ V2: supports
print("\n  V2  SOFT HAND-OFF IS A SUFFICIENT STATISTIC")
def dyadic_blocks(t):
    out, s = [], 0
    while s < t:
        l = 0
        while l < L and s % (2 ** (l+1)) == 0 and s + 2 ** (l+1) <= t: l += 1
        out.append((l, s >> l)); s += 2 ** l
    return out
tabs_full = inside(oh, P)
e2 = 0.0
for t in (4, 8, 12, 15):
    lf = np.ones_like(oh); lf[:, :t] = oh[:, :t]
    ref = outside(inside(lf, P), P)[0][:, t, :]
    ref = ref / ref.sum(-1, keepdims=True)
    blocks = dyadic_blocks(t)
    lf2 = np.ones_like(oh); inj = {}
    for (l, n) in blocks:
        if l == 0: lf2[:, n] = oh[:, n]                # single leaf: keep observed
        else:
            beta = tabs_full[l][:, n, :]
            inj[(l, n)] = beta / beta.sum(-1, keepdims=True)   # posterior shape only
    got = outside(inside(lf2, P, inj), P)[0][:, t, :]
    got = got / got.sum(-1, keepdims=True)
    e = float(np.abs(got - ref).max()); e2 = max(e2, e)
    desc = " + ".join(f"L{l}" if l else "leaf" for l, _ in blocks)
    print(f"    t={t:<3} prefix -> {desc:<22} max |dP| {e:.1e}")
verdict["V2 hand-off sufficient"] = e2 < TOL
print(f"    -> {'PASS' if e2 < TOL else 'FAIL'}: the leaves can be deleted once each "
      f"completed subtree is replaced by its posterior")

# ============================================================ V3: hierarchy (data)
print("\n  V3  HIERARCHY CHECK: MI between leaf positions (data-facing)")
print(f"    {'pair':<9}{'boundary':<16}{'empirical':>10}{'(MM corr)':>11}"
      f"{'fitted':>9}{'true':>9}")
names = {1: "sibling", 2: "level-1", 3: "level-2", 4: "root"}
for (i, j) in ((0, 1), (1, 2), (3, 4), (7, 8), (0, 8), (0, 15)):
    lvl = next(l for l in range(1, L+1) if i >> l == j >> l)
    J = np.zeros((NLEAF, NLEAF)); np.add.at(J, (TRALL[:, i], TRALL[:, j]), 1)
    emp = mi(J)
    kx, ky = (J.sum(1) > 0).sum(), (J.sum(0) > 0).sum()
    mm = emp - (kx - 1) * (ky - 1) / (2 * len(TRALL))
    mt = mi(model_joint(PT, i, j))
    flag = "" if mt > 1e-3 else "   (true ~0: uninformative)"
    print(f"    ({i:>2},{j:>2})  {names[lvl]:<16}{emp:>10.4f}{mm:>11.4f}"
          f"{mi(model_joint(P, i, j)):>9.4f}{mt:>9.4f}{flag}")
print("    Single-leaf MI vanishes above level 1 even under the TRUE grammar, so")
print("    this checks only the sibling and level-1 boundaries. Upper levels are")
print("    checked by `higher` in SCORES: fitted against the true grammar.")

if a.skip_v4: print("\n  V4  skipped (--skip-v4)")
# ============================================================ V4: merges
if not a.skip_v4: print("\n  V4  MERGES: mixture rows, conditional MI, held-out cost")
def nH(c):
    n = c.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(c > 0, c * np.log(c / np.maximum(n, 1e-300)), 0.0)
    return -t.sum(-1)
def agglomerate(Cl, ks):
    D = Cl.shape[1] * Cl.shape[2]
    cc = Cl.reshape(V, D).copy(); groups = [[s] for s in range(V)]
    E = nH(cc); tot = 0.0; rec = {}
    if V in ks: rec[V] = ([g[:] for g in groups], 0.0)
    first = None
    while len(groups) > min(ks):
        cost = nH(cc[:, None, :] + cc[None, :, :]) - E[:, None] - E[None, :]
        np.fill_diagonal(cost, np.inf)
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        i, j = min(i, j), max(i, j)
        if first is None: first = (groups[i][0], groups[j][0], float(cost[i, j]))
        tot += float(cost[i, j])
        groups[i] += groups[j]; cc[i] += cc[j]
        del groups[j]; cc = np.delete(cc, j, 0)
        E = nH(cc)
        if len(groups) in ks: rec[len(groups)] = ([g[:] for g in groups], tot)
    return rec, first
def cls_of(groups):
    k = np.zeros(V, dtype=int)
    for ci, g in enumerate(groups):
        for s in g: k[s] = ci
    return k
def Q(Cl, rows): return float((Cl * np.log(np.maximum(rows, 1e-300))).sum())

ks = sorted({int(k) for k in a.ks.split(",") if 1 <= int(k) <= V}, reverse=True)
P0 = mstep(C); ll0 = loglik(VA, P0); N = len(TR)
e4 = 0.0
for l in (range(1, L+1) if not a.skip_v4 else []):
    Cl = C[l] + 1e-9
    rec, (s1, s2, predicted) = agglomerate(Cl, ks)
    # V4a: formula vs direct recomputation of Q with the two rows tied
    untied = Cl / Cl.sum((1, 2), keepdims=True)
    tied_c = Cl.copy(); tied_c[s1] = tied_c[s2] = Cl[s1] + Cl[s2]
    tied = tied_c / tied_c.sum((1, 2), keepdims=True)
    direct = Q(Cl, untied) - Q(Cl, tied)
    e = abs(direct - predicted) / max(abs(direct), 1e-12); e4 = max(e4, e)
    nodes = N * (SEQ >> l)
    print(f"\n    level {l}  ({SEQ >> l} nodes/seq)   first merge {s1}+{s2}: "
          f"formula {predicted:.6g}  direct {direct:.6g}  rel err {e:.1e}")
    print(f"      {'K':>4}{'N*I (nats)':>13}{'I/node':>11}{'Q lost/seq':>12}"
          f"{'held-out change/seq':>21}" + (f"{'after refit':>13}" if a.refit else ""))
    for K in ks:
        groups, tot = rec[K]
        cls = {l: cls_of(groups)}
        Pk = mstep(C, cls); dll = loglik(VA, Pk) - ll0
        row = (f"      {K:>4}{tot:>13.2f}{tot/nodes:>11.5f}{tot/N:>12.4f}"
               f"{dll:>+21.4f}")
        if a.refit:
            Pr = Pk
            for _ in range(a.refit):
                Cr, _ = estep(TR, Pr); Pr = mstep(Cr, cls)
            row += f"{loglik(VA, Pr) - ll0:>13.4f}"
        print(row)
if not a.skip_v4: verdict["V4a merge identity"] = e4 < 1e-6
if not a.skip_v4: print(f"\n    V4a merge identity  max rel err {e4:.1e}  -> "
      f"{'PASS' if e4 < 1e-6 else 'FAIL'}")
if not a.skip_v4:
    print("    I/node is the witness of  Leaves _||_ B | q(B):  zero iff the merge is")
    print("    lossless (strong lumpability). Q lost is paid on TRAIN by construction.")
    print("    held-out change: + = better. A positive value means the unmerged model")
    print("    was overfitting and the merge removed parameters it could not support.")

# ============================================================ V6: obstruction detector
print("\n  V6  WHERE DOES THE TREE FAIL?  parametric bootstrap, per leaf pair")
def sample_model(Pm, n, g):
    """draw n sequences from a tree model: uniform root, then each node picks
    a child pair from its rule row"""
    cur = g.randint(0, V, size=(n, 1))
    for l in range(L, 0, -1):
        c = card(l); D = c * c
        cdf = np.cumsum(Pm[l].reshape(V, D), 1); cdf[:, -1] = 1.0
        u = g.rand(*cur.shape); pick = np.zeros(cur.shape, dtype=np.int64)
        for s_ in range(V):
            m_ = cur == s_
            if m_.any(): pick[m_] = np.searchsorted(cdf[s_], u[m_], side="right")
        pick = np.minimum(pick, D - 1)
        nxt = np.empty((n, cur.shape[1] * 2), dtype=np.int64)
        nxt[:, 0::2] = pick // c; nxt[:, 1::2] = pick % c
        cur = nxt
    return cur
PAIRS = [(t, t + 1) for t in range(SEQ - 1)] + [(0, 15), (3, 12), (1, 9)]
nb = min(len(TRALL), a.boot_n)
Xd = TRALL[:nb]
def lvl_of(i, j): return next(l for l in range(1, L+1) if i >> l == j >> l)
def test(Pm, Xdata, tag, g):
    obs = np.array([emp_mi(Xdata, i, j) for i, j in PAIRS])
    null = np.array([[emp_mi(Xs, i, j) for i, j in PAIRS]
                     for Xs in (sample_model(Pm, len(Xdata), g) for _ in range(a.boot))])
    mu_, sd_ = null.mean(0), null.std(0, ddof=1) + 1e-12
    return obs, mu_, (obs - mu_) / sd_
g6 = np.random.RandomState(777)
Xsyn = sample_model(PT, nb, np.random.RandomState(999))
runs = [("control A: true grammar vs its own samples", PT, Xsyn),
        ("control B: true grammar vs the data", PT, Xd),
        ("fitted model vs the data", P, Xd)]
res = {}
for tag, Pm, Xx in runs:
    res[tag] = test(Pm, Xx, tag, g6)
print(f"    {nb} sequences per dataset, {a.boot} replicates; z = (data - model mean)/sd")
print(f"    {'pair':<9}{'boundary':<10}{'data MI':>9}" +
      "".join(f"{'z '+k:>11}" for k in ("ctrl A", "ctrl B", "fitted")))
names6 = {1: "sibling", 2: "level-1", 3: "level-2", 4: "root"}
for k, (i, j) in enumerate(PAIRS):
    zs = [res[t][2][k] for t, _, _ in runs]
    mark = "  <== obstruction" if zs[2] > a.z_flag else ""
    if PLANT and (i, j) in ((PLANT[0], PLANT[1]), (PLANT[1], PLANT[0])): mark += "  [planted]"
    print(f"    ({i:>2},{j:>2})  {names6[lvl_of(i, j)]:<10}{res[runs[2][0]][0][k]:>9.4f}" +
          "".join(f"{z:>+11.1f}" for z in zs) + mark)
zA = np.abs(res[runs[0][0]][2]); zB = np.abs(res[runs[1][0]][2])
verdict["V6 control A calibrated"] = bool(zA.max() < a.z_flag)
verdict["V6 control B (true = generator)"] = bool(zB.max() < a.z_flag) if not PLANT else \
    bool(res[runs[1][0]][2][PAIRS.index((PLANT[0], PLANT[1]))] > a.z_flag) \
    if (PLANT[0], PLANT[1]) in PAIRS else True
if PLANT and (PLANT[0], PLANT[1]) in PAIRS:
    zf = res[runs[2][0]][2]; ip = PAIRS.index((PLANT[0], PLANT[1]))
    verdict["V6 planted obstruction found"] = bool(zf[ip] > a.z_flag)
    others = [zf[k] for k in range(len(PAIRS)) if k != ip]
    print(f"    planted pair z = {zf[ip]:+.1f}; largest other z = {max(others):+.1f}")
print("    control A |z| < flag     => the test is calibrated (no false alarms)")
print("    control B                => unplanted: the true grammar explains the data;")
print("                                planted: even the true grammar is refuted there")
print("    fitted, z >> 0           => dependence the fitted tree cannot produce: an")
print("                                obstruction localised to that boundary")

# ============================================================ V7: hierarchy-level detector
print("\n  V7  DEPENDENCE BETWEEN HIERARCHY UNITS (not raw tokens)")
print("    Pairwise leaf MI is ~0 above level 1 even under the true grammar, so V6")
print("    is blind there. Here each block of 2^l leaves is DECODED to the level-l")
print("    symbol the fitted model gives it (argmax of that block's own inside")
print("    vector), and dependence is measured between adjacent decoded blocks.")
print("    The same decoder is applied to data and to model samples, so the")
print("    bootstrap stays calibrated; the decoder being fitted costs power, not")
print("    validity.")
def decode(X, lmax, Pm):
    """block -> MAP level-l symbol from that block's own evidence, under Pm"""
    tabs = inside(onehot(X), Pm)
    return {l: tabs[l].argmax(-1) for l in range(1, lmax + 1)}
def mi_codes(A, B_, K):
    J = np.bincount(A * K + B_, minlength=K * K).reshape(K, K).astype(float)
    kx, ky = (J.sum(1) > 0).sum(), (J.sum(0) > 0).sum()
    return mi(J), mi(J) - (kx - 1) * (ky - 1) / (2 * len(A))
LMAX = L - 1
CELLS = [(l, n) for l in range(1, LMAX + 1) for n in range((SEQ >> l) - 1)]
def stats7(X, Pdec):
    d = decode(X, LMAX, Pdec)
    return np.array([mi_codes(d[l][:, n], d[l][:, n+1], V) for (l, n) in CELLS])
g7 = np.random.RandomState(4242)
# two references. TRUE decoder says how much block dependence the data really
# has; FITTED decoder says how much the fitted model's own symbols capture.
res7 = {}
for tag, Pm in (("true", PT), ("fitted", P)):
    obs = stats7(Xd, Pm)
    null = np.array([stats7(sample_model(Pm, len(Xd), g7), Pm)[:, 0]
                     for _ in range(a.boot)])
    res7[tag] = (obs, null.mean(0), (obs[:, 0] - null.mean(0)) /
                 (null.std(0, ddof=1) + 1e-12))
obsA7 = stats7(Xsyn, PT)[:, 0]
nullA = np.array([stats7(sample_model(PT, len(Xd), g7), PT)[:, 0]
                  for _ in range(a.boot)])
zA7 = (obsA7 - nullA.mean(0)) / (nullA.std(0, ddof=1) + 1e-12)
print(f"\n    {'level':>6}{'blocks':>8}{'shares':<10}" +
      f"{'data|true':>10}{'true mdl':>10}{'z ctrlA':>9}" +
      f"{'data|fit':>10}{'fit mdl':>9}{'z fitted':>10}")
for k, (l, n) in enumerate(CELLS):
    sh = "parent" if n % 2 == 0 else f"level-{l+2}+"
    ot, mt, _ = res7["true"]; of, mf, zf = res7["fitted"]
    print(f"    {l:>6}{f'{n},{n+1}':>8}  {sh:<8}{ot[k,1]:>10.4f}{mt[k]:>10.4f}"
          f"{zA7[k]:>+9.1f}{of[k,1]:>10.4f}{mf[k]:>9.4f}{zf[k]:>+10.1f}")
print("    data|true  = block dependence really present (MM-corrected), decoded")
print("                 with the TRUE grammar; compare to `true mdl`")
print("    data|fit   = what the FITTED model's own symbols capture")
print("    data|true large but data|fit ~ 0  =>  the hierarchy IS in the data at")
print("    that level and the fitted model did not recover it")
verdict["V7 control A calibrated"] = bool(np.abs(zA7).max() < a.z_flag)

# ============================================================ summary
print("\n  SUMMARY")
for k, v in verdict.items(): print(f"    {k:<26} {'PASS' if v else 'FAIL'}")
print("  V0-V2, V4a are identities of the tree model class: they verify the code")
print("  and that the fitted object is a sheaf section. V3 and V4b are the")
print("  measurements about the data.")
if a.save:
    np.savez(a.save, **{f"P{l}": P[l] for l in P}); print(f"  saved {a.save}")
