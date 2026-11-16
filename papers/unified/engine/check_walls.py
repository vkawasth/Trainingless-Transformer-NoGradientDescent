"""Study 37: two arrangements on margin space; vanishing edges; tropical smoothing of leximin walls (sec:lefschetz). Pre-registered in
study37_walls/PREREG_study37.md, frozen by SHA-256 (study37_walls/FROZEN_PREREG_study37.txt) before any code for it ran. New script; reuses
helpers of check_four6.py (code before its population block); modifies nothing.
  WX1 IDENTITY prop:vanishedge     WX2 RECORD two arrangements     WX3 IDENTITY prop:realmono
  WX4 HYPOTHESIS cor:width          WX5 RECORD curvature location
First run (the only run): 3/5. WX1 FAILED in 8 of 40 segments: in all 8 the wall was of the form f_i = g_j + g_k (levels 0.60, 0.65),
and on the side where the vertex count drops (18 -> 15 or 15 -> 12) no edge shrinks; on the other side the shortest edge was exactly
2 x |f_i - s| (sqrt of the cycle length 4), as predicted. The proposition as pre-registered (a vanishing edge on both sides) is false;
the paper states the corrected, proved version (edge length = sqrt(cycle length) x |cut value|). WX4 FAILED vacuously: the threshold
10/L exceeds the segment length 0.04 for L = 60 and 240, so no point was tested and both errors were 0 (a design error, recorded).
Post hoc (not a check): at three random slice points L * u_L tends to c_P = 0 exponentially (L = 60..960), as thm:selection(d) says;
and the c_P routine had a shape bug for levels with fewer than four cells (svd full_matrices), never reached in the run.
Output: check_walls_output.txt, study37_walls/walls.json.
"""
import os, sys, json, itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog, minimize
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "check_four6.py")).read(); exec(src.split("# population ----")[0])
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
def walls_crossed(p, q):
    out = []
    for i in range(3):
        for s in S_LEV:
            a, b = p[i] - s, q[i] - s
            if a * b < 0: out.append((i, s, a / (a - b)))
    return out
def rand_point(rng):
    while True:
        p = rng.dirichlet([3, 3, 3])
        if p.min() > 0.05: return p
def rand_dir(rng):
    d = rng.normal(size=3); d -= d.mean(); return d / np.linalg.norm(d)
# WX1 ------------------------------------------------------------------
rng = np.random.default_rng(37); done = 0; bad1 = 0; info1 = []
while done < 40:
    p = rand_point(rng); d = rand_dir(rng); q = p + 0.05 * d
    if q.min() <= 0.02: continue
    W = walls_crossed(p, q)
    if len(W) != 1: continue
    i, s, ts = W[0]; wall = p + ts * (q - p); ok = True; ratios = {}; counts = {}
    for side in (-1, 1):
        rs = []
        for k in range(1, 10):
            dist = k * 1e-4; pt = wall + side * (dist / abs(d[i])) * d                 # |f_i - s| = dist
            e, nv = shortest_edge(pt); rs.append(e / abs(pt[i] - s)); counts[side] = nv
        rs = np.array(rs); ratios[side] = rs; ok &= (rs.max() - rs.min()) / rs.mean() < 1e-6
    changed = counts[-1] != counts[1] or chamber(wall - 1e-4 * d) != chamber(wall + 1e-4 * d)
    bad1 += not (ok and changed); done += 1
check("WX1 IDENTITY prop:vanishedge: at a chamber wall the shortest edge shrinks linearly to zero and the combinatorics change", bad1 == 0, f"40 segments; violations {bad1}")
# WX2 ------------------------------------------------------------------
def sig(fv):
    f_ = [Fr(x).limit_denominator(10 ** 9) for x in fv]; Sf = facets_float(f_, g, cells, A); lv, x, _, _ = leximin_float(f_, g, cells, A, Sf)
    return tuple(frozenset(S_) for _, S_ in lv), x, lv, Sf
grid = []
n = 15
for a in range(1, n):
    for b in range(1, n - a):
        c_ = n - a - b
        if c_ >= 1: grid.append((a, b, c_))
G = {pt: np.array(pt, float) / n for pt in grid}; SIG = {pt: sig(G[pt])[0] for pt in grid}; CH = {pt: chamber(G[pt]) for pt in grid}
lex_edges = 0; ch_edges = 0; both = 0
for pt in grid:
    for dlt in ((1, -1, 0), (1, 0, -1), (0, 1, -1)):
        nb = tuple(pp + dd for pp, dd in zip(pt, dlt))
        if nb in G:
            L1 = SIG[pt] != SIG[nb]; C1 = CH[pt] != CH[nb]; lex_edges += L1; ch_edges += C1; both += L1 and C1
check("WX2 RECORD two arrangements on margin space", True,
      f"{len(grid)} grid points; distinct leximin signatures {len(set(SIG.values()))}, distinct chambers {len(set(CH.values()))}; grid edges with a leximin change {lex_edges}, "
      f"with a chamber change {ch_edges}, both {both}")
# WX3 ------------------------------------------------------------------
bad3 = 0
for lp in range(20):
    c0 = rand_point(rng); ang = np.sort(rng.uniform(0, 2 * np.pi, 6)); e1 = np.array([1, -1, 0]) / np.sqrt(2); e2 = np.array([1, 1, -2]) / np.sqrt(6)
    poly = [c0 + 0.08 * (np.cos(t) * e1 + np.sin(t) * e2) for t in ang]
    if min(p_.min() for p_ in poly) <= 0.02: poly = [c0 + 0.03 * (np.cos(t) * e1 + np.sin(t) * e2) for t in ang]
    path = []
    for k in range(6):
        for t in np.linspace(0, 1, 20, endpoint=False): path.append(poly[k] + t * (poly[(k + 1) % 6] - poly[k]))
    path.append(poly[0]); s0, x0, _, _ = sig(path[0]); sigs = [sig(p_)[0] for p_ in path[:: 10]]; s1, x1, _, _ = sig(path[-1])
    bad3 += s0 != s1 or np.abs(x0 - x1).max() > 1e-9
check("WX3 IDENTITY prop:realmono: around closed loops the leximin point and its level partition return unchanged", bad3 == 0, f"20 loops; violations {bad3}")
# WX4 + WX5 ------------------------------------------------------------
Fidx = None
def crit_u(L, fv, xlex, Sf):
    Fi = [cells.index(cl) for cl in Sf]; Nm = np.array([N[cl] for cl in Sf], float); c0 = xlex[Fi]
    def F(u):
        z = -L * (c0 + Nm @ u); m = z.max(); w = np.exp(z - m); return w.sum(), m, w
    u = np.zeros(4)
    for _ in range(200):
        _, m, w = F(u); gr = -L * (w @ Nm); H = L * L * (Nm.T * w) @ Nm; st = np.linalg.solve(H, gr); u = u - st
        if np.abs(st).max() < 1e-15: break
    return u
def c_P(lv, Sf):
    """first-order coefficient of thm:selection(d), weights 1: level by level minimisers over the projected normals."""
    Nm = {cl: np.array(N[cl], float) for cl in Sf}; xi_sum = np.zeros(4); earlier = []
    for h, S_ in lv:
        S_ = [cl for cl in S_ if cl in Nm]
        if earlier:
            Q, _ = np.linalg.qr(np.array(earlier).T); P = np.eye(4) - Q @ Q.T
        else: P = np.eye(4)
        V = np.array([P @ Nm[cl] for cl in S_]); base = np.array([Nm[cl] @ xi_sum for cl in S_])
        U, sv, _ = np.linalg.svd(V.T); Bs = U[:, sv > 1e-10]
        if Bs.shape[1] > 0:
            obj = lambda z: np.exp(-base - V @ (Bs @ z)).sum()
            jac = lambda z: -(Bs.T @ (V.T @ np.exp(-base - V @ (Bs @ z))))
            z = minimize(obj, np.zeros(Bs.shape[1]), jac=jac, method="BFGS", options=dict(gtol=1e-13)).x; xi_sum = xi_sum + Bs @ z
        earlier += [Nm[cl] for cl in S_]
    return xi_sum
done = 0; better = 0; rec5 = []; tries = 0
while done < 20 and tries < 4000:
    tries += 1; p = rand_point(rng); d = rand_dir(rng); q = p + 0.04 * d
    if q.min() <= 0.03: continue
    ts = np.linspace(0, 1, 41); pts = [p + t * (q - p) for t in ts]
    if len({chamber(x) for x in pts}) != 1: continue
    sg = [sig(x)[0] for x in pts]
    if len(set(sg)) != 2 or sum(sg[k] != sg[k + 1] for k in range(40)) != 1: continue
    k = next(k for k in range(40) if sg[k] != sg[k + 1]); lo_, hi_ = ts[k], ts[k + 1]
    for _ in range(40):
        mid = (lo_ + hi_) / 2
        if sig(p + mid * (q - p))[0] == sg[0]: lo_ = mid
        else: hi_ = mid
    tw = (lo_ + hi_) / 2; errs = {}
    for L in (60, 240):
        e = 0.0
        for t in ts:
            dist = abs(t - tw) * 0.04
            if dist <= 10 / L: continue
            fv = p + t * (q - p); _, xl, lv, Sf = sig(fv); u = crit_u(L, fv, xl, Sf); cp = c_P(lv, Sf)
            Nm9 = np.array([N[cl] for cl in cells], float); e = max(e, float(np.abs(Nm9 @ (u - cp / L)).max()))
        errs[L] = e
    better += errs[240] < errs[60]; loc = {}
    for L in (60, 120, 240):
        xs = []
        for t in ts:
            fv = p + t * (q - p); _, xl, lv, Sf = sig(fv); u = crit_u(L, fv, xl, Sf); Nm9 = np.array([N[cl] for cl in cells], float); xs.append(xl + Nm9 @ u)
        xs = np.array(xs); curv = np.abs(xs[2:] - 2 * xs[1:-1] + xs[:-2]).max(1); loc[L] = float(abs(ts[1 + int(np.argmax(curv))] - tw))
    rec5.append(dict(err60=errs[60], err240=errs[240], loc=loc)); done += 1
check("WX4 HYPOTHESIS cor:width: away from a leximin wall (distance > 10/L) the first-order law x(L) = x_lex + c_P/L improves from L = 60 to 240 in >= 18 of 20 segments",
      done == 20 and better >= 18, f"{done} segments; improved in {better}")
check("WX5 RECORD location of maximal curvature of x(L) relative to the leximin wall (segment-parameter units)", True,
      "; ".join(f"L={L}: median {np.median([r_['loc'][L] for r_ in rec5]):.3f}" for L in (60, 120, 240)))
json.dump(dict(wx2=dict(signatures=len(set(SIG.values())), chambers=len(set(CH.values())), lex_edges=lex_edges, ch_edges=ch_edges, both=both), wx4=rec5,
               grid=[[list(pt), [sorted(map(list, S_)) for S_ in SIG[pt]]] for pt in grid]), open(os.path.join(HERE, "study37_walls", "walls.json"), "w"))
print(f"{sum(results)}/{len(results)} passed")
