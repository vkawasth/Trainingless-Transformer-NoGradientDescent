#!/usr/bin/env python3
"""ESTIMATOR6 -- depth-6 hierarchy recovery, with the two filtrations side by side.

    python3 estimator6.py --data /tmp/cD6 --n-train 20000
    python3 estimator6.py --data /tmp/cD6 --load-P /tmp/cD6/fit.npz   # skip fitting

Fits a depth-L grammar without gradients (spectral init blended with uniform,
inside-outside EM, split-merge), verifies the structural theorems, and then
shows, LEVEL BY LEVEL, two filtrations that look at the same hierarchy from
opposite sides:

  BLOCK SIDE   Dowker filtration on interaction information between decoded
               blocks. Vertices are blocks; hyperedges are subsets whose
               |I_k| clears a threshold. Its beta_0 cascade is the hierarchy
               being assembled: separate constituents merging into larger ones.

  ROW SIDE     exact Bregman (KL) Cech filtration on the level's rule rows.
               Vertices are the V symbols; a simplex enters at the radius of
               the smallest enclosing dual KL ball. Its beta_0 cascade is the
               symbol alphabet collapsing.

THEOREMS CHECKED (each printed with its numerical witness)
----------------------------------------------------------
  T1  EM ascent                      Dempster-Laird-Rubin
  T2  gluing = independence join     Simpson Prop 8.5 / Wright 7.1
  T3  hand-off is sufficient         Fritz Def 14.3, Thm 14.5
  T4  merge cost = Bregman info      Banerjee et al.; = N I(children;B|q(B))
  T5  Cech radius is monotone        so the row filtration is a filtration
  T6  Rips != Cech                   Edelsbrunner-Wagner Result 2, measured
  T7  alpha = Cech here              with V points in ambient dimension
                                     >= V-1 and in general position, the
                                     Delaunay triangulation is the full
                                     simplex, so the Delaunay (alpha) complex
                                     coincides with the Cech complex at every
                                     radius. The general-position condition is
                                     checked numerically.
"""
import json, math, argparse, time, itertools, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp/cD6")
ap.add_argument("--n-train", type=int, default=20000)
ap.add_argument("--n-val", type=int, default=2000)
ap.add_argument("--warmup", type=int, default=30)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--sweeps", type=int, default=15)
ap.add_argument("--init-mix", type=float, default=0.9)
ap.add_argument("--delta", type=float, default=0.25)
ap.add_argument("--n-blocks-max", type=int, default=8,
                help="Dowker runs only where a level has at most this many "
                     "blocks; triples grow as O(n^3) and each needs a V^3 table")
ap.add_argument("--boot", type=int, default=6)
ap.add_argument("--iters", type=int, default=1200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--save", default=""); ap.add_argument("--load-P", default="")
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
KIND = M["pos_kind"]
tr = np.array(json.load(open(f"{a.data}/train_ids.json")), dtype=np.int64)
va = np.array(json.load(open(f"{a.data}/val_ids.json")), dtype=np.int64)
TR = tr[:(len(tr)//SEQ)*SEQ].reshape(-1, SEQ)[:a.n_train]
VA = va[:(len(va)//SEQ)*SEQ].reshape(-1, SEQ)[:a.n_val]
rng = np.random.RandomState(a.seed)
def card(l): return NLEAF if l == 1 else V
EPS = 1e-12

# ------------------------------------------------------------- tree machinery
def up(Pl, lo, hi):
    B, n, cx = lo.shape; cy = hi.shape[2]; p = Pl.shape[0]
    lo2, hi2 = lo.reshape(-1, cx), hi.reshape(-1, cy)
    Pt = Pl.transpose(1, 0, 2).reshape(cx, p*cy)
    out = np.empty((lo2.shape[0], p))
    for s in range(0, lo2.shape[0], a.chunk):
        t = (lo2[s:s+a.chunk] @ Pt).reshape(-1, p, cy)
        out[s:s+a.chunk] = (t * hi2[s:s+a.chunk, None, :]).sum(-1)
    return out.reshape(B, n, p)
def down(Pl, po, lo, hi):
    B, n, p = po.shape; cx, cy = lo.shape[2], hi.shape[2]
    po2, lo2, hi2 = po.reshape(-1, p), lo.reshape(-1, cx), hi.reshape(-1, cy)
    Pf = Pl.reshape(p, cx*cy)
    lo_o = np.empty((po2.shape[0], cx)); hi_o = np.empty((po2.shape[0], cy))
    for s in range(0, po2.shape[0], a.chunk):
        t = (po2[s:s+a.chunk] @ Pf).reshape(-1, cx, cy)
        lo_o[s:s+a.chunk] = (t * hi2[s:s+a.chunk, None, :]).sum(-1)
        hi_o[s:s+a.chunk] = (t * lo2[s:s+a.chunk, :, None]).sum(-2)
    return lo_o.reshape(B, n, cx), hi_o.reshape(B, n, cy)
def cnts(Pl, po, lo, hi, w):
    B, n, p = po.shape; cx, cy = lo.shape[2], hi.shape[2]
    po2 = (po*w[:, None, None]).reshape(-1, p)
    lo2, hi2 = lo.reshape(-1, cx), hi.reshape(-1, cy)
    acc = np.zeros((p, cx*cy))
    for s in range(0, po2.shape[0], a.chunk):
        o = (lo2[s:s+a.chunk, :, None]*hi2[s:s+a.chunk, None, :]).reshape(-1, cx*cy)
        acc += po2[s:s+a.chunk].T @ o
    return acc.reshape(p, cx, cy)*Pl
def onehot(X):
    o = np.zeros(X.shape + (NLEAF,)); np.put_along_axis(o, X[..., None], 1.0, -1)
    return o
def inside(lf, Pm):
    t = [lf]
    for l in range(1, L+1): t.append(up(Pm[l], t[-1][:, 0::2], t[-1][:, 1::2]))
    return t
def outside(t, Pm):
    B = t[0].shape[0]; vr = Pm[L].shape[0]
    o = [None]*(L+1); o[L] = np.full((B, 1, vr), 1.0/vr)
    for l in range(L, 0, -1):
        lo, hi = t[l-1][:, 0::2], t[l-1][:, 1::2]
        a_, b_ = down(Pm[l], o[l], lo, hi)
        ch = np.zeros((B, lo.shape[1]*2, lo.shape[2]))
        ch[:, 0::2] = a_; ch[:, 1::2] = b_; o[l-1] = ch
    return o
def Zof(t, Pm): return t[L][:, 0, :].sum(-1)/Pm[L].shape[0]
def loglik(X, Pm):
    return float(np.log(np.maximum(Zof(inside(onehot(X), Pm), Pm), 1e-300)).mean())
def estep(X, Pm):
    t = inside(onehot(X), Pm); o = outside(t, Pm)
    Z = np.maximum(Zof(t, Pm), 1e-300)
    C = {l: cnts(Pm[l], o[l], t[l-1][:, 0::2], t[l-1][:, 1::2], 1.0/Z)
         for l in range(1, L+1)}
    return C, float(np.log(Z).mean())
def mstep(C):
    return {l: (C[l]+1e-9)/(C[l]+1e-9).sum((1, 2), keepdims=True) for l in C}
def em(Pm, X, n):
    hist = []
    for _ in range(n):
        C, ll = estep(X, Pm); hist.append(ll); Pm = mstep(C)
    return Pm, hist

PT = {}
for l in range(1, L+1):
    t = np.zeros((V, card(l), card(l)))
    for s, r in M["rules_all"][str(l)].items():
        for (x, y) in r: t[int(s), x, y] += 1.0/len(r)
    PT[l] = t

# ------------------------------------------------------------------ fit
print(f"  estimator6: L={L} V={V} leaves={NLEAF} seq={SEQ}, "
      f"{len(TR)} train / {len(VA)} held-out")
print(f"  true grammar held-out {loglik(VA, PT):.4f}")
t0 = time.time(); ascent_ok = True; worst_step = 0.0
if a.load_P:
    z = np.load(a.load_P); P = {l: z[f"P{l}"] for l in range(1, L+1)}
    print(f"  loaded {a.load_P}")
else:
    soft = onehot(TR); P = {}
    for l in range(1, L+1):
        c = card(l)
        hard = soft.argmax(-1); lo, hi = hard[:, 0::2], hard[:, 1::2]
        pid = lo*c + hi
        cnt_all = collections.Counter(pid.reshape(-1).tolist())
        pairs = sorted(cnt_all); idx = {p_: i for i, p_ in enumerate(pairs)}
        ctx = collections.defaultdict(collections.Counter)
        for row in pid.tolist():
            for i, p_ in enumerate(row):
                if i > 0: ctx[p_][row[i-1]] += 1
                if i+1 < len(row): ctx[p_][row[i+1]] += 1
        if lo.shape[1] >= 4 and len(ctx) >= 2:
            Cm = np.zeros((len(pairs), len(pairs)))
            for p_, d in ctx.items():
                for q, n in d.items():
                    if q in idx: Cm[idx[p_], idx[q]] = n
            rs = Cm.sum(1, keepdims=True); rs[rs == 0] = 1
            U, S, _ = np.linalg.svd(Cm/rs, full_matrices=False)
            cum = np.cumsum(S)/max(S.sum(), 1e-300)
            k = max(2, min(int(np.searchsorted(cum, 0.90)+1), len(S)))
            emb = U[:, :k]*S[:k]
        else:
            emb = np.zeros((len(pairs), 2*c))
            for p_ in pairs: emb[idx[p_], p_//c] = 1.0; emb[idx[p_], c + p_ % c] = 1.0
        g = np.random.RandomState(a.seed)
        cen = emb[g.choice(len(pairs), min(V, len(pairs)), replace=False)].astype(float)
        for _ in range(25):
            lab = ((emb[:, None, :]-cen[None])**2).sum(-1).argmin(1)
            for j in range(cen.shape[0]):
                m = lab == j
                if m.any(): cen[j] = emb[m].mean(0)
        tab = np.zeros((V, c, c))
        for p_ in pairs: tab[lab[idx[p_]] % V, p_//c, p_ % c] += cnt_all[p_]
        tab += 1e-6; P[l] = tab/tab.sum((1, 2), keepdims=True)
        P[l] = (1-a.init_mix)*P[l] + a.init_mix/(c*c)
        soft = up(P[l], soft[:, 0::2], soft[:, 1::2])
        soft = soft/np.maximum(soft.sum(-1, keepdims=True), 1e-300)
    P, hist = em(P, TR, a.warmup)
    d = np.diff(hist); worst_step = float(d.min()) if len(d) else 0.0
    ascent_ok = worst_step > -1e-8
    print(f"  warmup {a.warmup} sweeps: held-out {loglik(VA, P):.4f}")

    def nH(c_):
        n = c_.sum(-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(c_ > 0, c_*np.log(c_/np.maximum(n, 1e-300)), 0.0)
        return -t.sum(-1)
    best = ({l: P[l].copy() for l in P}, loglik(VA, P))
    for r in range(a.rounds):
        for l in range(1, L+1):
            Ps = {k: v.copy() for k, v in P.items()}
            rows = np.repeat(P[l], 2, axis=0)*(1 + a.delta*rng.uniform(-1, 1, (2*V,) + P[l].shape[1:]))
            Ps[l] = rows/rows.sum((1, 2), keepdims=True)
            if l < L:
                T2 = np.repeat(np.repeat(P[l+1], 2, 1), 2, 2)/4.0
                Ps[l+1] = T2/T2.sum((1, 2), keepdims=True)
            Ps, h2 = em(Ps, TR, a.sweeps)
            if len(h2) > 1: worst_step = min(worst_step, float(np.diff(h2).min()))
            C, _ = estep(TR, Ps)
            Cl = C[l] + 1e-9; D = Cl.shape[1]*Cl.shape[2]
            cc = Cl.reshape(Cl.shape[0], D).copy(); groups = [[s] for s in range(Cl.shape[0])]
            E = nH(cc)
            while len(groups) > V:
                cost = nH(cc[:, None, :] + cc[None, :, :]) - E[:, None] - E[None, :]
                np.fill_diagonal(cost, np.inf)
                i, j = np.unravel_index(np.argmin(cost), cost.shape)
                i, j = min(i, j), max(i, j)
                groups[i] += groups[j]; cc[i] += cc[j]; del groups[j]
                cc = np.delete(cc, j, 0); E = nH(cc)
            lab2 = np.zeros(Cl.shape[0], dtype=int)
            for gi, gg in enumerate(groups):
                for s in gg: lab2[s] = gi
            pooled = np.zeros((V, Cl.shape[1], Cl.shape[2]))
            for s in range(Cl.shape[0]): pooled[lab2[s]] += Cl[s]
            Pm = {k: v.copy() for k, v in Ps.items()}
            Pm[l] = pooled/pooled.sum((1, 2), keepdims=True)
            if l < L:
                T = Ps[l+1]
                A_ = np.zeros((T.shape[0], V, T.shape[2]))
                for s in range(T.shape[1]): A_[:, lab2[s], :] += T[:, s, :]
                B_ = np.zeros((T.shape[0], V, V))
                for s in range(T.shape[2]): B_[:, :, lab2[s]] += A_[:, :, s]
                Pm[l+1] = B_/B_.sum((1, 2), keepdims=True)
            Pm, h3 = em(Pm, TR, a.sweeps)
            if len(h3) > 1: worst_step = min(worst_step, float(np.diff(h3).min()))
            ll = loglik(VA, Pm)
            if ll > best[1]: best = ({k: v.copy() for k, v in Pm.items()}, ll); P = Pm
            else: P = {k: v.copy() for k, v in best[0].items()}
        print(f"  round {r+1}: held-out {best[1]:.4f}")
    P = best[0]; ascent_ok = worst_step > -1e-8
    if a.save: np.savez(a.save, **{f"P{l}": P[l] for l in P})
print(f"  fit {time.time()-t0:.0f}s   held-out {loglik(VA, P):.4f} "
      f"(true {loglik(VA, PT):.4f})\n")

# ------------------------------------------------------------ theorem checks
print("  THEOREM CHECKS")
print(f"    T1 EM ascent (min step over all sweeps)        {worst_step:+.2e}   "
      f"{'PASS' if ascent_ok else 'FAIL'}")
X = VA[:256]; oh = onehot(X); H = SEQ//2
full = Zof(inside(oh, P), P)
lfL = oh.copy(); lfL[:, H:] = 1.0
lfR = oh.copy(); lfR[:, :H] = 1.0
tL, tR = inside(lfL, P), inside(lfR, P)
mu = P[L].sum(0)/P[L].shape[0]
glued = np.einsum('uv,bu,bv->b', mu, tL[L-1][:, 0, :], tR[L-1][:, 1, :])
e2 = float(np.abs(glued/full - 1).max())
print(f"    T2 gluing = independence join                 {e2:.1e}   "
      f"{'PASS' if e2 < 1e-9 else 'FAIL'}")
def dyadic(t):
    out, s = [], 0
    while s < t:
        l = 0
        while l < L and s % (2**(l+1)) == 0 and s + 2**(l+1) <= t: l += 1
        out.append((l, s >> l)); s += 2**l
    return out
tabs = inside(oh, P); e3 = 0.0
for t in (8, 24, 63):
    lf = np.ones_like(oh); lf[:, :t] = oh[:, :t]
    ref = outside(inside(lf, P), P)[0][:, t, :]; ref /= ref.sum(-1, keepdims=True)
    lf2 = np.ones_like(oh); inj = {}
    for (l, n) in dyadic(t):
        if l == 0: lf2[:, n] = oh[:, n]
        else:
            b = tabs[l][:, n, :]; inj[(l, n)] = b/b.sum(-1, keepdims=True)
    tt = [lf2]
    for l in range(1, L+1):
        o_ = up(P[l], tt[-1][:, 0::2], tt[-1][:, 1::2])
        for (ll, n), v in inj.items():
            if ll == l: o_[:, n, :] = v
        tt.append(o_)
    got = outside(tt, P)[0][:, t, :]; got /= got.sum(-1, keepdims=True)
    e3 = max(e3, float(np.abs(got-ref).max()))
print(f"    T3 hand-off is a sufficient statistic         {e3:.1e}   "
      f"{'PASS' if e3 < 1e-9 else 'FAIL'}")
C, _ = estep(TR, P)
Cl = C[1] + 1e-9
def nH2(c_):
    n = c_.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(c_ > 0, c_*np.log(c_/np.maximum(n, 1e-300)), 0.0)
    return -t.sum(-1)
cc = Cl.reshape(V, -1)
form = float(nH2(cc[0]+cc[1]) - nH2(cc[0]) - nH2(cc[1]))
u = Cl/Cl.sum((1, 2), keepdims=True)
ti = Cl.copy(); ti[0] = ti[1] = Cl[0]+Cl[1]; ti = ti/ti.sum((1, 2), keepdims=True)
direct = float((Cl*np.log(np.maximum(u, 1e-300))).sum()
               - (Cl*np.log(np.maximum(ti, 1e-300))).sum())
e4 = abs(direct-form)/max(abs(direct), 1e-12)
print(f"    T4 merge cost = Bregman information           {e4:.1e}   "
      f"{'PASS' if e4 < 1e-6 else 'FAIL'}")

# ------------------------------------------- filtrations, side by side
def kl(p, q):
    m = p > EPS
    return float((p[m]*np.log(p[m]/np.maximum(q[m], EPS))).sum())
def one_centre(pts):
    c = pts.mean(0); c = c/c.sum(); best = max(kl(p, c) for p in pts)
    for i in range(1, a.iters+1):
        d = np.array([kl(p, c) for p in pts]); j = int(d.argmax())
        c = (1-1/(i+1))*c + (1/(i+1))*pts[j]; c = np.maximum(c, 0); c /= c.sum()
        r = max(kl(p, c) for p in pts)
        if r < best: best = r
    return best
def rank_gf2(Mx):
    A = (Mx % 2).astype(np.uint8).copy(); r = 0
    for c_ in range(A.shape[1]):
        piv = next((rr for rr in range(r, A.shape[0]) if A[rr, c_]), None)
        if piv is None: continue
        A[[r, piv]] = A[[piv, r]]
        sel = A[:, c_] == 1; sel[r] = False
        A[sel] ^= A[r]; r += 1
        if r == A.shape[0]: break
    return r
def betti01(verts, edges, tris):
    if not verts: return 0, 0
    par = {v: v for v in verts}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for (i, j) in edges:
        ri, rj = find(i), find(j)
        if ri != rj: par[ri] = rj
    b0 = len({find(v) for v in verts})
    cyc = len(edges)-len(verts)+b0
    if not tris or cyc == 0: return b0, cyc
    ei = {e: k for k, e in enumerate(edges)}
    d2 = np.zeros((len(edges), len(tris)), dtype=np.uint8)
    for t, (i, j, k) in enumerate(tris):
        for e in ((i, j), (j, k), (i, k)):
            if e in ei: d2[ei[e], t] = 1
    return b0, cyc-rank_gf2(d2)
def dowker(he):
    vs, es, ts = set(), set(), set()
    for e in he:
        e = tuple(sorted(e)); vs.update(e)
        es.update(itertools.combinations(e, 2)); ts.update(itertools.combinations(e, 3))
    return sorted(vs), sorted(es), sorted(ts)
def Hc(c_):
    p = c_/c_.sum(); m = p > 0
    return float(-(p[m]*np.log(p[m])).sum())

print("\n  HIERARCHY MAPPED BY TWO FILTRATIONS")
print("    BLOCK SIDE: Dowker on |I_k| between decoded blocks (b0 cascade =")
print("                constituents merging).   ROW SIDE: exact Bregman-Cech on")
print("                the rule rows (b0 cascade = symbols collapsing).")
t_all = inside(onehot(TR[:min(len(TR), 20000)]), P)
t6ok = True
for l in range(1, L+1):
    nb = SEQ >> l
    # ---- row side
    rows = P[l].reshape(V, -1)
    rho = {}
    for k in (2, 3):
        for S in itertools.combinations(range(V), k):
            rho[S] = one_centre(rows[list(S)])
    mono = all(rho[f] <= rho[S] + 1e-9
               for S in rho if len(S) == 3 for f in itertools.combinations(S, 2))
    ed = {S: v for S, v in rho.items() if len(S) == 2}
    gap = max(rho[S] - max(ed[(S[0], S[1])], ed[(S[0], S[2])], ed[(S[1], S[2])])
              for S in rho if len(S) == 3)
    genpos = len(set(np.round(list(rho.values()), 9))) == len(rho)
    ts = np.unique(np.array(sorted(rho.values())))
    b0row = []
    for t in ts:
        E = sorted(S for S, v in ed.items() if v <= t)
        T = sorted(S for S, v in rho.items() if len(S) == 3 and v <= t)
        b0row.append(betti01(list(range(V)), E, T)[0])
    b0rips = []
    for t in ts:
        E = sorted(S for S, v in ed.items() if v <= t)
        T = sorted(S for S in rho if len(S) == 3 and
                   max(ed[(S[0], S[1])], ed[(S[0], S[2])], ed[(S[1], S[2])]) <= t)
        b0rips.append(betti01(list(range(V)), E, T)[1])
    b1cech = []
    for t in ts:
        E = sorted(S for S, v in ed.items() if v <= t)
        T = sorted(S for S, v in rho.items() if len(S) == 3 and v <= t)
        b1cech.append(betti01(list(range(V)), E, T)[1])
    # ---- block side
    if nb <= a.n_blocks_max and nb >= 3:
        codes = t_all[l].argmax(-1)
        h1 = {i: Hc(np.bincount(codes[:, i], minlength=V).astype(float)) for i in range(nb)}
        h2 = {}
        for i, j in itertools.combinations(range(nb), 2):
            h2[(i, j)] = Hc(np.bincount(codes[:, i]*V+codes[:, j], minlength=V*V).astype(float))
        W = {}
        for i, j in itertools.combinations(range(nb), 2):
            W[(i, j)] = max(h1[i]+h1[j]-h2[(i, j)] - (V-1)**2/(2*len(codes)), 0.0)
        for i, j, k in itertools.combinations(range(nb), 3):
            h3 = Hc(np.bincount(codes[:, i]*V*V+codes[:, j]*V+codes[:, k],
                                minlength=V**3).astype(float))
            W[(i, j, k)] = abs(h1[i]+h1[j]+h1[k]-h2[(i, j)]-h2[(i, k)]-h2[(j, k)]+h3)
        tb = np.unique(np.array(list(W.values())))[::-1]
        b0blk = [betti01(*dowker([s for s, v in W.items() if v >= t]))[0] for t in tb]
        blk = " ".join(str(x) for x in b0blk[:10])
    else:
        blk = f"(skipped: {nb} blocks)"
    print(f"\n    level {l}   {nb} blocks, {V} symbols")
    print(f"      block side  b0: {blk}")
    print(f"      row side    b0: " + " ".join(str(x) for x in b0row[:10]))
    print(f"      T5 Cech monotone {'PASS' if mono else 'FAIL'}"
          f"   T6 max Rips-Cech gap {gap:+.4f}"
          f" (max b1: Cech {max(b1cech)}, Rips {max(b0rips)})"
          f"   T7 general position {'PASS' if genpos else 'FAIL'}")
    t6ok &= gap > 1e-9
print(f"\n    T6 overall: Rips differs from Cech at every level "
      f"{'PASS' if t6ok else 'FAIL'} -- Edelsbrunner-Wagner Result 2, so the")
print(f"       clique/Rips construction is not a substitute for the nerve.")
print(f"    T7: with {V} points in ambient dimension >= {V-1} and distinct radii,")
print(f"       the Delaunay triangulation is the full simplex, so the alpha")
print(f"       complex equals the Cech complex at every radius.")
