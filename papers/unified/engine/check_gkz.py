"""Study 39a: the GKZ identification -- which walls the potential can see (sec:gkz: prop:shiftinv, prop:gkz). Pre-registered in
study39a_gkz/PREREG_study39a.md and frozen by SHA-256 before any code for it ran. Reuses helpers of check_four6.py (code before its
population block); modifies nothing.
  GZ1 IDENTITY prop:shiftinv(i)    GZ2 IDENTITY prop:shiftinv(iii)    GZ3 IDENTITY prop:gkz(a)    GZ4 IDENTITY prop:gkz(b)
  GZ5 IDENTITY first-level walls on circuit walls    GZ6 HYPOTHESIS all-level walls on circuit walls    GZ7, GZ8 RECORD
Output: check_gkz_output.txt, study39a_gkz/gkz.json.
First run (study39a_gkz/check_gkz_output_run1.txt): GZ1 PASS, GZ2 FAIL (0 segments: vacuous by design), GZ3 PASS, GZ4 FAIL (a code error:
the lower-hull test read the qhull offset instead of the normal), then a crash in the GZ5 bisection (leximin stalled on a wall).
The fixes are recorded in study39a_gkz/DEVIATION_study39a.md before the second run; no threshold, sample size or seed changed.
Second run (check_gkz_output.txt): 6/8. GZ1-GZ4 PASS. GZ5 FAILED on its count: 3x3 found 17 first-level changes, not 20 (2x3: 38);
0 violations, but 48 (2x3) and 1 (3x3) bisections stalled and were not tested. GZ6 FAILED: the hypothesis is false; 15 of 126 (2x3)
and 27 of 64 (3x3) signature changes at deeper levels lie off every circuit wall (up to 0.03 away). GZ7: 6 of 150 grid edges with a
leximin change cross an unbalanced wall and no circuit wall; the prediction of 0 assumed GZ6. GZ8: 14 subdivisions, 5 tie sets.
"""
import os, sys, json, itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.spatial import ConvexHull
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SRC = open(os.path.join(HERE, "check_four6.py")).read(); exec(SRC.split("# population ----")[0])
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
OUT = {}

class Shape:
    def __init__(s, r, c):
        s.r, s.c = r, c; s.cells, A, Nd, s.d = shape(r, c); s.A = np.array(A, float); s.n = len(s.cells)
        s.V = np.array([Nd[cl] for cl in s.cells], float)                          # exponents v_k (rows of the kernel basis)
        s.A1 = s.A @ np.ones(s.n)
        H = np.hstack([np.ones((s.n, 1)), s.V]); s.circuits = []
        for m in range(2, s.d + 3):
            for S in itertools.combinations(range(s.n), m):
                M = H[list(S)].T; _, sv, vt = np.linalg.svd(M)
                null = vt[np.sum(sv > 1e-9):]
                if len(null) != 1: continue
                cvec = null[0]
                if np.min(np.abs(cvec)) < 1e-9: continue
                full = np.zeros(s.n); full[list(S)] = cvec
                mu = np.linalg.lstsq(s.A.T, full, rcond=None)[0]
                assert np.abs(s.A.T @ mu - full).max() < 1e-9
                s.circuits.append((S, mu / np.linalg.norm(mu)))
        s.cuts = []
        for a in range(1, r + 1):
            for I in itertools.combinations(range(r), a):
                for bb in range(0, c + 1):
                    for J in itertools.combinations(range(c), bb):
                        if (a, bb) in ((r, c),): continue
                        if a * c - bb * r == 0 and (a, bb) == (0, 0): continue
                        s.cuts.append((I, J))
        # keep one of each complementary pair, nontrivial
        seen = set(); cuts = []
        for I, J in s.cuts:
            key = (I, J); comp = (tuple(i for i in range(r) if i not in I), tuple(j for j in range(c) if j not in J))
            if comp in seen or not I or len(I) == r: continue
            seen.add(key); cuts.append((I, J))
        s.cuts = cuts
    def b_of(s, f, g): return np.array(list(f) + list(g[:-1]), float)
    def kappa(s, I, J, f, g): return sum(f[i] for i in I) - sum(g[j] for j in J)
    def balanced(s, I, J): return len(I) * s.c - len(J) * s.r == 0
    def x0(s, b): return np.linalg.lstsq(s.A, b, rcond=None)[0]
    def lex(s, f, g):
        lv, x, _, _ = leximin_float(list(f), list(g), s.cells, s.A.tolist(), s.cells)
        return tuple(frozenset(S_) for _, S_ in lv), x, lv
    def circuit_dist(s, b): return min(abs(mu @ b) for _, mu in s.circuits)
    def feasible_bases(s, b):
        out = set()
        for B in itertools.combinations(range(s.n), s.A.shape[0]):
            AB = s.A[:, list(B)]
            if abs(np.linalg.det(AB)) < 1e-9: continue
            xb = np.linalg.solve(AB, b)
            if (xb >= -1e-12).all(): out.add(frozenset(B))
        return out
    def cell_bases(s, b):
        x0 = s.x0(b); out = set()
        for sig in itertools.combinations(range(s.n), s.d):
            Vs = s.V[list(sig)]
            if abs(np.linalg.det(Vs)) < 1e-9: continue
            u = np.linalg.solve(Vs, -x0[list(sig)])                                 # l_k = x0_k + <v_k, u> = 0 on sigma
            l = x0 + s.V @ u
            if (l >= -1e-12).all(): out.add(frozenset(range(s.n)) - frozenset(sig))
        return out
    def hull_K0(s, b):
        x0 = s.x0(b); P = np.hstack([s.V, x0[:, None]]); hull = ConvexHull(P); sets = []
        for eq, simp in zip(hull.equations, hull.simplices):
            if eq[-2] >= -1e-12: continue                                       # lower facet: normal points down
            Vt = s.V[simp]                                                          # projected simplex: does it contain 0?
            M = np.vstack([Vt.T, np.ones(len(simp))]); lam = np.linalg.lstsq(M, np.r_[np.zeros(s.d), 1.0], rcond=None)[0]
            if np.abs(M @ lam - np.r_[np.zeros(s.d), 1.0]).max() > 1e-9 or lam.min() < -1e-9: continue
            on = frozenset(k for k in range(s.n) if abs(eq[:-1] @ P[k] + eq[-1]) < 1e-9); sets.append(on)
        K = frozenset.intersection(*sets); return frozenset(s.cells[k] for k in K)
    def subdivision(s, b):
        x0 = s.x0(b); P = np.hstack([s.V, x0[:, None]]); hull = ConvexHull(P); cellsets = set()
        for eq in hull.equations:
            if eq[-2] >= -1e-12: continue                                       # lower facet: normal points down
            cellsets.add(frozenset(k for k in range(s.n) if abs(eq[:-1] @ P[k] + eq[-1]) < 1e-9))
        return frozenset(cellsets)

S23 = Shape(2, 3); S33 = Shape(3, 3)
G33 = [Fr(24, 60), Fr(21, 60), Fr(15, 60)]
def sample23(rng):
    while True:
        f1 = rng.uniform(0.06, 0.94); g = rng.dirichlet([2, 2, 2])
        f = np.array([f1, 1 - f1])
        if g.min() > 0.06 and f.max() + g.max() < 0.98: return f, g
def sample33(rng):
    gv = np.array([float(x) for x in G33])
    while True:
        f = rng.dirichlet([3, 3, 3])
        if f.min() > 0.04 and f.max() + gv.max() < 0.98: return f, gv
def ok23(f, g): return f.min() > 0.02 and g.min() > 0.02 and f.max() + g.max() < 0.99
def ok33(f, g): return f.min() > 0.02 and f.max() + g.max() < 0.99
def seg23(rng, length):
    while True:
        f, g = sample23(rng); d = rng.normal(size=3); d /= np.linalg.norm(d)
        f2 = np.array([f[0] + length * d[0], f[1] - length * d[0]]); g2 = g + length * np.array([d[1], d[2], -d[1] - d[2]])
        if ok23(f2, g2): return (f, g), (f2, g2)
def seg33(rng, length):
    while True:
        f, g = sample33(rng); d = rng.normal(size=3); d -= d.mean(); d /= np.linalg.norm(d); f2 = f + length * d
        if ok33(f2, g): return (f, g), (f2, g)
SHAPES = [("2x3", S23, sample23, seg23, np.random.default_rng(3901), 0.15), ("3x3", S33, sample33, seg33, np.random.default_rng(3902), 0.10)]
def at(p, q, t): return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

# GZ1 -------------------------------------------------------------------
def crit(S, x0, L):
    u = np.zeros(S.d)
    for _ in range(200):
        z = -L * (x0 + S.V @ u); m = z.max(); w = np.exp(z - m); gr = -L * (w @ S.V); H = L * L * (S.V.T * w) @ S.V
        st = np.linalg.solve(H, gr); u = u - st
        if np.abs(st).max() < 1e-15: break
    return u
w1x = w1u = 0.0
for name, S, samp, seg, rng, _ in SHAPES:
    for _ in range(50):
        f, g = samp(rng); _, x, _ = S.lex(f, g)
        for s_ in (0.01, 0.05):
            f2 = f + s_ * S.c; g2 = g + s_ * S.r
            _, x2, _ = S.lex(f2, g2); w1x = max(w1x, float(np.abs(x2 - (x + s_)).max()))
            x0 = S.x0(S.b_of(f, g)); u1 = crit(S, x0, 40.0); u2 = crit(S, x0 + s_, 40.0); w1u = max(w1u, float(np.abs(u1 - u2).max()))
check("GZ1 IDENTITY prop:shiftinv(i): adding s to every cell shifts the leximin point by s and leaves the critical point unchanged",
      w1x < 1e-8 and w1u < 1e-9, f"2 shapes x 50 margins x 2 shifts; max leximin error {w1x:.1e}; max critical-point error {w1u:.1e}")
OUT["gz1"] = dict(lex=w1x, crit=w1u)
# GZ2 -------------------------------------------------------------------
def seg2(name, rng, length):                                                 # deviation 2: no all-facets restriction
    while True:
        if name == "2x3":
            f1 = rng.uniform(0.03, 0.97); g = rng.dirichlet([2, 2, 2]); f = np.array([f1, 1 - f1])
            if g.min() <= 0.03: continue
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            f2 = np.array([f[0] + length * d[0], f[1] - length * d[0]]); g2 = g + length * np.array([d[1], d[2], -d[1] - d[2]])
            if f2.min() > 0.02 and g2.min() > 0.02: return (f, g), (f2, g2)
        else:
            gv_ = np.array([float(x) for x in G33]); f = rng.dirichlet([3, 3, 3])
            if f.min() <= 0.03: continue
            d = rng.normal(size=3); d -= d.mean(); d /= np.linalg.norm(d); f2 = f + length * d
            if f2.min() > 0.02: return (f, gv_), (f2, gv_)
info2 = []; ok2 = True; rec2 = {}
for name, S, samp, seg, rng, length in SHAPES:
    found = 0; worst = 0.0; sigdiff = 0; tries = 0
    while found < 40 and tries < 20000:
        tries += 1; p, q = seg2(name, rng, length)
        cross = []
        for I, J in S.cuts:
            a, bq = S.kappa(I, J, *p), S.kappa(I, J, *q)
            if a * bq < 0: cross.append((I, J, a / (a - bq)))
        if len(cross) != 1 or S.balanced(cross[0][0], cross[0][1]): continue
        t0 = cross[0][2]; pc = at(p, q, t0)
        if S.circuit_dist(S.b_of(*pc)) <= 1e-3: continue
        step = 1e-4 / np.linalg.norm(np.r_[q[0] - p[0], q[1] - p[1]])
        ts = t0 + step * np.arange(-10, 11)
        if ts.min() < 0 or ts.max() > 1: continue
        X = []; sigs = []
        for t in ts:
            sg, x, _ = S.lex(*at(p, q, t)); X.append(x); sigs.append(sg)
        X = np.array(X); err = float(np.abs(X[2:] - 2 * X[1:-1] + X[:-2]).max()); worst = max(worst, err); sigdiff += sigs[0] != sigs[-1]; found += 1
    good = found >= 40 and worst < 1e-8 and sigdiff == 0; ok2 &= good; rec2[name] = dict(found=found, tries=tries, worst=worst, sigdiff=sigdiff)
    info2.append(f"{name}: {found} segments, max second difference {worst:.1e}, signature changes {sigdiff}")
check("GZ2 IDENTITY prop:shiftinv(iii): an unbalanced chamber wall changes nothing in the leximin point", ok2, "; ".join(info2))
OUT["gz2"] = rec2
# GZ3, GZ4 --------------------------------------------------------------
mm3 = 0; mm4 = 0; n34 = 0
for name, S, samp, seg, rng, _ in SHAPES:
    for _ in range(100):
        f, g = samp(rng); b = S.b_of(f, g)
        mm3 += S.feasible_bases(b) != S.cell_bases(b)
        sg, x, lv = S.lex(f, g); mm4 += frozenset(lv[0][1]) != S.hull_K0(b); n34 += 1
check("GZ3 IDENTITY prop:gkz(a): vertices of the fibre = complements of the cells of the regular subdivision of the vector configuration",
      mm3 == 0, f"{n34} margins over 2 shapes; mismatches {mm3}")
check("GZ4 IDENTITY prop:gkz(b): the first-level tie set = the minimal cell containing 0 of the lower hull of the lifted exponents",
      mm4 == 0, f"{n34} margins over 2 shapes; mismatches {mm4}")
OUT["gz34"] = dict(mm3=mm3, mm4=mm4)
# GZ5, GZ6 --------------------------------------------------------------
info5 = []; info6 = []; ok5 = True; ok6 = True; rec56 = {}; unresolved = {}
for name, S, samp, seg, rng, length in SHAPES:
    n5 = v5 = n6 = v6 = 0; d5 = []; d6 = []
    for _ in range(40):
        p, q = seg(rng, length); ts = np.linspace(0, 1, 41); sg = [S.lex(*at(p, q, t)) for t in ts]
        for k in range(40):
            for lvl, (cnt, key) in enumerate((("first", lambda z: z[2][0][1] and frozenset(z[2][0][1])), ("all", lambda z: z[0]))):
                a, b_ = key(sg[k]), key(sg[k + 1])
                if a == b_: continue
                lo_, hi_ = ts[k], ts[k + 1]
                stalled = False
                while hi_ - lo_ > 1e-10:
                    mid = (lo_ + hi_) / 2
                    try: z = S.lex(*at(p, q, mid))
                    except RuntimeError:
                        try: z = S.lex(*at(p, q, mid + 1e-12))                          # deviation 3: nudge once
                        except RuntimeError: stalled = True; break
                    if key(z) == a: lo_ = mid
                    else: hi_ = mid
                if stalled: unresolved[name] = unresolved.get(name, 0) + 1; continue
                dist = S.circuit_dist(S.b_of(*at(p, q, (lo_ + hi_) / 2)))
                if cnt == "first": n5 += 1; v5 += dist >= 1e-7; d5.append(dist)
                else: n6 += 1; v6 += dist >= 1e-7; d6.append(dist)
    ok5 &= n5 >= 20 and v5 == 0; ok6 &= v6 == 0
    rec56[name] = dict(n5=n5, v5=v5, n6=n6, v6=v6, max5=max(d5) if d5 else None, max6=max(d6) if d6 else None)
    info5.append(f"{name}: {n5} first-level changes, violations {v5}, max distance {max(d5) if d5 else float('nan'):.1e}")
    info6.append(f"{name}: {n6} signature changes, violations {v6}, max distance {max(d6) if d6 else float('nan'):.1e}")
check("GZ5 IDENTITY prop:gkz(b): every first-level change lies on a circuit wall of the point configuration", ok5, "; ".join(info5) + f"; unresolved bisections {unresolved}")
check("GZ6 HYPOTHESIS every change of the full leximin signature lies on a circuit wall", ok6, "; ".join(info6))
OUT["gz56"] = rec56
# GZ7 -------------------------------------------------------------------
rec7 = {}
for name, S, *_ in SHAPES:
    rec7[name] = dict(cuts=len(S.cuts), balanced=sum(S.balanced(I, J) for I, J in S.cuts), circuits=len(S.circuits),
                      circuit_walls=len({tuple(np.round(mu * np.sign(mu[np.argmax(np.abs(mu))]), 9)) for _, mu in S.circuits}))
n_ = 15; grid = [(a, b, n_ - a - b) for a in range(1, n_) for b in range(1, n_ - a) if n_ - a - b >= 1]
gv = np.array([float(x) for x in G33]); P = {pt: np.array(pt, float) / n_ for pt in grid}; SIG = {pt: S33.lex(P[pt], gv)[0] for pt in grid}
bad7 = 0; edges7 = 0
for pt in grid:
    for dlt in ((1, -1, 0), (1, 0, -1), (0, 1, -1)):
        qt = tuple(np.add(pt, dlt))
        if qt not in P or SIG[pt] == SIG[qt]: continue
        fa, fb = P[pt], P[qt]
        unb = any(not S33.balanced(I, J) and S33.kappa(I, J, fa, gv) * S33.kappa(I, J, fb, gv) < 0 for I, J in S33.cuts)
        circ = any((mu @ S33.b_of(fa, gv)) * (mu @ S33.b_of(fb, gv)) <= 0 for _, mu in S33.circuits)
        edges7 += 1; bad7 += unb and not circ
rec7["grid3x3"] = dict(edges_with_leximin_change=edges7, crossing_unbalanced_and_no_circuit=bad7)
check("GZ7 RECORD walls per shape, and Study 37 grid edges with a leximin change across an unbalanced wall and no circuit wall", True,
      f"2x3: {rec7['2x3']}; 3x3: {rec7['3x3']}; grid: {edges7} edges with a leximin change, {bad7} of them cross an unbalanced chamber wall and no circuit wall")
OUT["gz7"] = rec7
# GZ8 -------------------------------------------------------------------
rng8 = np.random.default_rng(3903); subs = set(); firsts = set()
for _ in range(2000):
    f, g = sample23(rng8); b = S23.b_of(f, g); subs.add(S23.subdivision(b)); firsts.add(S23.hull_K0(b))
check("GZ8 RECORD 2x3: regular subdivisions of the hexagon and first-level tie sets met over 2000 margins", True,
      f"{len(subs)} subdivisions; {len(firsts)} first-level tie sets: {sorted(sorted(map(str, k)) for k in firsts)}")
OUT["gz8"] = dict(subdivisions=len(subs), firsts=[sorted(map(str, k)) for k in firsts])
json.dump(OUT, open(os.path.join(HERE, "study39a_gkz", "gkz.json"), "w"), default=str)
print(f"{sum(results)}/{len(results)} passed")
