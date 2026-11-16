"""Study 38: each decision rule has its own arrangement; the one-scalar decision witness; one-sided vanishing edges (sec:witness,
sec:lefschetz). Pre-registered in study38_witness/PREREG_study38.md, frozen by SHA-256 (study38_witness/FROZEN_PREREG_study38.txt)
before any code for it ran; a note written after freezing and before the run (study38_witness/NOTE_study38.md) predicts the robust
half of DQ9 from prop:coarsen. New script; reuses helpers of check_four6.py (code before its population block); modifies nothing.
  DQ1 IDENTITY lem:ownarr(i)      DQ2 IDENTITY lem:ownarr(ii)     DQ3 IDENTITY thm:witness(i)    DQ4 IDENTITY thm:witness(ii)
  DQ5 IDENTITY thm:witness(iii)   DQ6 IDENTITY prop:onesided      DQ7 HYPOTHESIS converse        DQ8 RECORD common refinement
  DQ9 RECORD decision descent under merging rows {1,2}
Check ids: the pre-registration names them DW1-DW9; after the run they were renamed DQ1-DQ9 because DW1-DW7 already belong to
check_datawalls.py. Labels only; nothing else changed (study38_witness/RENAME_study38.md).
First run (the only run): 8/9. DQ5 FAILED on its pre-registered count: 28 grid points (not 30) had a gap zero in their leximin
region; in all 28 the scalar was sufficient and its direction necessary (0 violations). The threshold is unchanged; the shortfall is a
design error (the 91-point grid is too coarse for 30 qualifying points with one draw of utilities), recorded as a failure.
DQ6 tested only 4 one-cell sides (all walls f_i = g_j + g_k, side f_i > s); DQ7 tested 76 sides, all with a vanishing edge.
DQ9: the merged selected point varied over the compatible margins at 11 of 25 values of f3; the selected-point decision stayed
constant at 23 (so at 9 of those 11); the robust decision at 25, as NOTE_study38.md predicted from prop:coarsen.
Output: check_witness_output.txt, study38_witness/witness.json.
"""
import os, sys, json, itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SRC = open(os.path.join(HERE, "check_four6.py")).read(); exec(SRC.split("# population ----")[0])
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
cells, A, N, D = shape(3, 3); g = [Fr(24, 60), Fr(21, 60), Fr(15, 60)]; gv = np.array([float(x) for x in g])
S_LEV = sorted({float(x) for x in g} | {float(a + b) for a, b in itertools.combinations(g, 2)})
def is_tree(cl):
    par = list(range(6))
    def fd(x):
        while par[x] != x: x = par[x]
        return x
    for (i, j) in cl:
        a, b = fd(i), fd(3 + j)
        if a == b: return False
        par[a] = b
    return len(cl) == 5
TREES = [B for B in itertools.combinations(range(9), 5) if is_tree([cells[k] for k in B])]
Af = np.array(A, float)
def vertices(fv):
    b = np.concatenate([fv, gv[:2]]); out = []
    for B in TREES:
        xb = np.linalg.solve(Af[:, list(B)], b)
        if (xb >= -1e-13).all():
            x = np.zeros(9); x[list(B)] = xb; out.append((B, x))
    return out
def chamber(fv): return frozenset(B for B, _ in vertices(fv))
def shortest_edge(fv):
    V = vertices(fv); best = np.inf
    for (B1, x1), (B2, x2) in itertools.combinations(V, 2):
        if len(set(B1) & set(B2)) == 4:
            d = np.linalg.norm(x1 - x2)
            if d > 1e-14: best = min(best, d)
    return best, len(V)
CACHE = {}
def lex(fv):
    key = tuple(np.round(fv, 15))
    if key not in CACHE:
        f_ = [float(x) for x in fv]; Sf = facets_float(f_, g, cells, A); lv, x, _, _ = leximin_float(f_, g, cells, A, Sf)
        CACHE[key] = (tuple(frozenset(S_) for _, S_ in lv), x)
    return CACHE[key]
def phi(w, fv): return min(float(w @ x) for _, x in vertices(fv))
def rand_point(rng, lo=0.05):
    while True:
        p = rng.dirichlet([3, 3, 3])
        if p.min() > lo: return p
def rand_dir(rng):
    d = rng.normal(size=3); d -= d.mean(); return d / np.linalg.norm(d)
rng = np.random.default_rng(38)
U = rng.normal(size=(3, 9))
TH = np.zeros(9); TH[cells.index((0, 0))] = 1.0
for j in range(3): TH[cells.index((0, j))] -= 0.4                                   # n - tau d, n = cell (0,0), d = row 0, tau = 0.4
def dec_sel(fv): x = lex(fv)[1]; return int(np.argmax(U @ x))
def dec_rob(fv): return int(np.argmax([phi(U[a], fv) for a in range(3)]))
def gap_sel(fv, a, b): x = lex(fv)[1]; return float((U[a] - U[b]) @ x)
def gap_rob(fv, a, b): return phi(U[a], fv) - phi(U[b], fv)
def reg_sel(fv): return lex(fv)[0]
def reg_rob(fv): return chamber(fv)
RULES = dict(selected=(dec_sel, gap_sel, reg_sel), robust=(dec_rob, gap_rob, reg_rob))
OUT = {}
# DQ1, DQ2 ---------------------------------------------------------------
def segsearch(want, need=30, cap=20000):
    found = []; tries = 0; worst = 0.0
    while len(found) < need and tries < cap:
        tries += 1; p = rand_point(rng); d = rand_dir(rng); q = p + 0.03 * d
        if q.min() <= 0.02: continue
        ts = np.linspace(0, 1, 31); pts = [p + t * (q - p) for t in ts]
        chs = [chamber(x) for x in pts]; ch_change = len(set(chs)) > 1
        if want == "chamber_only" and not ch_change: continue
        if want == "lex_only" and ch_change: continue
        sgs = [lex(x)[0] for x in pts]; lx_change = len(set(sgs)) > 1
        if want == "chamber_only" and lx_change: continue
        if want == "lex_only" and not lx_change: continue
        if want == "chamber_only":
            X = np.array([lex(x)[1] for x in pts]); err = float(np.abs(X[2:] - 2 * X[1:-1] + X[:-2]).max())
        else:
            err = 0.0
            for w in list(U) + [TH]:
                ph = np.array([phi(w, x) for x in pts]); err = max(err, float(np.abs(ph[2:] - 2 * ph[1:-1] + ph[:-2]).max()))
        found.append(err); worst = max(worst, err)
    return found, tries, worst
f1, t1, w1 = segsearch("chamber_only")
check("DQ1 IDENTITY lem:ownarr(i): across chamber walls inside one leximin region the selected point is affine", len(f1) >= 20 and w1 < 1e-7,
      f"{len(f1)} segments in {t1} candidates; max second difference {w1:.2e}")
f2, t2, w2 = segsearch("lex_only")
check("DQ2 IDENTITY lem:ownarr(ii): across leximin walls inside one chamber every robust quantity is affine", len(f2) >= 20 and w2 < 1e-10,
      f"{len(f2)} segments in {t2} candidates; max second difference {w2:.2e}")
OUT["dw12"] = dict(dw1=dict(n=len(f1), tries=t1, worst=w1), dw2=dict(n=len(f2), tries=t2, worst=w2))
# DQ3, DQ4 ---------------------------------------------------------------
viol3 = 0; flips = []; n4 = 0; viol4 = 0; worst4 = 0.0; done = 0
while done < 40:
    p = rand_point(rng, 0.08); d = rand_dir(rng); q = p + 0.15 * d
    if q.min() <= 0.03: continue
    done += 1; ts = np.linspace(0, 1, 101); pts = [p + t * (q - p) for t in ts]
    for rule, (dec, gap, reg) in RULES.items():
        ds = [dec(x) for x in pts]
        for k in range(100):
            if ds[k] == ds[k + 1]: continue
            a, b = ds[k], ds[k + 1]; gl, gr = gap(pts[k], a, b), gap(pts[k + 1], a, b)
            ok = gl >= -1e-12 and gr <= 1e-12
            lo_, hi_ = ts[k], ts[k + 1]
            while hi_ - lo_ > 1e-10:
                mid = (lo_ + hi_) / 2
                if dec(p + mid * (q - p)) == a: lo_ = mid
                else: hi_ = mid
            tf = (lo_ + hi_) / 2; gf = gap(p + tf * (q - p), a, b); ok &= abs(gf) < 1e-6; viol3 += not ok
            flips.append(dict(rule=rule, seg=done, k=k, tflip=tf, gap=gf, ok=bool(ok)))
            regs = [reg(x) for x in pts[:k + 2]]
            if len(set(regs)) == 1:
                g0, g1_ = gap(pts[0], a, b), gap(pts[1], a, b); slope = (g1_ - g0) / (ts[1] - ts[0])
                if slope != 0:
                    tp = -g0 / slope; e4 = abs(tp - tf); n4 += 1; worst4 = max(worst4, e4); viol4 += e4 > 1e-6
check("DQ3 IDENTITY thm:witness(i): every decision flip is a sign change of the gap between the two decisions", viol3 == 0 and len(flips) > 0,
      f"40 segments x 2 rules; {len(flips)} flips ({sum(f['rule'] == 'selected' for f in flips)} selected, {sum(f['rule'] == 'robust' for f in flips)} robust); violations {viol3}")
check("DQ4 IDENTITY thm:witness(ii): inside the own region the flip time is predicted by one affine scalar", n4 >= 10 and viol4 == 0,
      f"{n4} flips with the region fixed up to the flip; max error {worst4:.2e}; violations {viol4}")
OUT["dw34"] = dict(flips=flips, n4=n4, worst4=worst4)
# DQ5 --------------------------------------------------------------------
n_ = 15; grid = [(a, b, n_ - a - b) for a in range(1, n_) for b in range(1, n_ - a) if n_ - a - b >= 1]
E1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2); E2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6); h = 1e-4
n5 = 0; viol5 = 0; rec5 = []
for pt in grid:
    fv = np.array(pt, float) / n_; R = reg_sel(fv); hit = None
    for (a, b) in ((0, 1), (0, 2), (1, 2)):
        nb = [fv + s * h * e for e in (E1, E2) for s in (1, -1)]
        if any(reg_sel(x) != R for x in nb): break
        G0 = gap_sel(fv, a, b); al = np.array([(gap_sel(fv + h * E1, a, b) - gap_sel(fv - h * E1, a, b)) / (2 * h),
                                              (gap_sel(fv + h * E2, a, b) - gap_sel(fv - h * E2, a, b)) / (2 * h)])
        if np.linalg.norm(al) < 1e-9: continue
        dirn = (al[0] * E1 + al[1] * E2) / np.linalg.norm(al); b0 = fv - G0 / np.linalg.norm(al) * dirn
        if b0.min() <= 0.02 or reg_sel(b0) != R: continue
        perp = (-al[1] * E1 + al[0] * E2) / np.linalg.norm(al); ok = True
        for s in (0.002, -0.002, 0.004, -0.004):
            x = fv + s * perp
            if x.min() > 0.02 and reg_sel(x) == R: ok &= abs(gap_sel(x, a, b) - G0) < 1e-7
        up, dn = b0 + 1e-4 * dirn, b0 - 1e-4 * dirn
        if reg_sel(up) == R and reg_sel(dn) == R: ok &= gap_sel(up, a, b) > 0 > gap_sel(dn, a, b)
        hit = (a, b, bool(ok)); break
    if hit is not None:
        n5 += 1; viol5 += not hit[2]; rec5.append(dict(pt=list(pt), pair=hit[:2], ok=hit[2]))
    if n5 >= 30: break
check("DQ5 IDENTITY thm:witness(iii): one scalar is sufficient, and the direction it measures is necessary", n5 >= 30 and viol5 == 0,
      f"{n5} grid points with a gap zero in their leximin region; violations {viol5}")
OUT["dw5"] = rec5
# DQ6, DQ7 ---------------------------------------------------------------
def cut_of(s):
    for j in range(3):
        if abs(float(g[j]) - s) < 1e-12: return {j}
    for j, k in itertools.combinations(range(3), 2):
        if abs(float(g[j] + g[k]) - s) < 1e-12: return {j, k}
def walls_crossed(p, q):
    out = []
    for i in range(3):
        for s in S_LEV:
            a, b = p[i] - s, q[i] - s
            if a * b < 0: out.append((i, s, a / (a - b)))
    return out
rng6 = np.random.default_rng(3801); done = 0; viol6 = 0; viol7 = 0; n6 = 0; n7 = 0; rec67 = []
while done < 40:
    p = rand_point(rng6); d = rand_dir(rng6); q = p + 0.05 * d
    if q.min() <= 0.02: continue
    W = walls_crossed(p, q)
    if len(W) != 1: continue
    done += 1; i, s, ts_ = W[0]; J = cut_of(s); wall = p + ts_ * (q - p)
    Zp, Zm = 3 - len(J), 2 * len(J)
    for side in (-1, 1):
        rs = []; es = []
        for k in range(1, 10):
            dist = k * 1e-4; pt = wall + side * np.sign(d[i]) * (dist / abs(d[i])) * d
            e, nv = shortest_edge(pt); es.append(e); rs.append(e / abs(pt[i] - s))
        rs = np.array(rs); kappa_sign = np.sign(pt[i] - s); Z = Zp if kappa_sign > 0 else Zm
        prop_ = bool(np.isfinite(rs).all() and (rs.max() - rs.min()) / rs.mean() < 1e-6)
        if Z == 1:
            n6 += 1; viol6 += not ((not prop_) or min(es) >= 1e-3)
        else:
            n7 += 1; viol7 += not prop_
        rec67.append(dict(i=i, s=s, J=sorted(J), side=int(kappa_sign), Z=Z, vanishing=prop_, min_edge=float(min(es))))
check("DQ6 IDENTITY prop:onesided: a side whose half of the cut has one cell carries no vanishing edge", n6 > 0 and viol6 == 0,
      f"{n6} such sides on 40 segments; violations {viol6}")
check("DQ7 HYPOTHESIS converse: every side whose half of the cut has at least two cells carries a vanishing edge", n7 > 0 and viol7 == 0,
      f"{n7} such sides; violations {viol7}")
OUT["dw67"] = rec67
# DQ8 --------------------------------------------------------------------
G_ = {pt: np.array(pt, float) / n_ for pt in grid}
pairs = {(reg_sel(G_[pt]), chamber(G_[pt])) for pt in grid}
nsig = len({reg_sel(G_[pt]) for pt in grid}); nch = len({chamber(G_[pt]) for pt in grid})
check("DQ8 RECORD common refinement on the 91-point grid", True, f"{len(pairs)} (signature, chamber) pairs; {nsig} signatures; {nch} chambers")
OUT["dw8"] = dict(pairs=len(pairs), signatures=nsig, chambers=nch)
# DQ9 --------------------------------------------------------------------
Uc = U.copy()
for a in range(3):
    for j in range(3): Uc[a, cells.index((1, j))] = Uc[a, cells.index((0, j))]
def merged(x): y = np.zeros(6); y[:3] = x[0:3] + x[3:6]; y[3:] = x[6:9]; return y
vary = 0; sel_const = 0; rob_const = 0; rec9 = []
for c in np.linspace(0.08, 0.84, 25):
    pts = [np.array([t, 1 - c - t, c]) for t in np.linspace(0.02 + 1e-3, 1 - c - 0.02 - 1e-3, 41)]
    M = np.array([merged(lex(x)[1]) for x in pts]); v = float(np.abs(M - M[0]).max()) > 1e-6
    ds = [int(np.argmax(Uc @ lex(x)[1])) for x in pts]; dr = [int(np.argmax([phi(Uc[a], x) for a in range(3)])) for x in pts]
    vary += v; sel_const += len(set(ds)) == 1; rob_const += len(set(dr)) == 1
    rec9.append(dict(f3=float(c), merged_varies=v, sel_const=len(set(ds)) == 1, rob_const=len(set(dr)) == 1))
check("DQ9 RECORD decision descent under merging rows {1,2}", True,
      f"25 values of f3: merged selected point varies in {vary}; selected-point decision constant in {sel_const}; robust decision constant in {rob_const}")
OUT["dw9"] = rec9
json.dump(OUT, open(os.path.join(HERE, "study38_witness", "witness.json"), "w"), default=str)
print(f"{sum(results)}/{len(results)} passed")
