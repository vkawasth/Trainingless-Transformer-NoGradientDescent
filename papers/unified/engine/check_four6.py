"""Study 35: leximin and Theorem 196 at 4 x 6; cheap exact structure; coarsening (sec:four6). Pre-registered in study35_four6/PREREG_study35.md,
frozen by SHA-256 (study35_four6/FROZEN_PREREG_study35.txt) before any code for it ran. New script; uses certify_max from pce/cpl33.py
(generic exact LP certification); modifies nothing.
  LZ1 IDENTITY prop:twowaycomplete      LZ2 IDENTITY prop:coarsen      LZ3 IDENTITY max-ent merge consistency
  LZ4 RECORD census of leximin patterns  LZ5 IDENTITY Theorem 196 at 4 x 6 (exact levels, spectral check)
  LZ6 RECORD leximin coarsening defect   LZ7 HYPOTHESIS defect > 1e-9 in >= 50% of the census
First run (the only run): 6/7. LZ5 FAILED at tolerance 0.01 (tables 22: 0.0105, 45: 0.0140; six others <= 0.0081); threshold unchanged.
Post-hoc diagnosis in check_four6_addendum.py (LZ8): at L = 960 both are within 0.0016 (finite-L bias). The Frechet pruning never fired
(0 saved), contrary to expectation.
Output: check_four6_output.txt, study35_four6/four6.json.
"""
import os, sys, json, itertools, time
from collections import Counter
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce.cpl33 import certify_max
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def margins(rng, k, den):
    v = rng.dirichlet([2] * k) * den; x = np.maximum(1, np.floor(v)).astype(int); x[int(np.argmax(x))] += den - x.sum()
    return [Fr(int(y), den) for y in x] if x.min() >= 1 else None
def nondeg(f, g):
    fs = {sum(c) for k in range(1, len(f)) for c in itertools.combinations(f, k)}; gs = {sum(c) for k in range(1, len(g)) for c in itertools.combinations(g, k)}
    return not (fs & gs)
def shape(r, c):
    cells = [(i, j) for i in range(r) for j in range(c)]
    A = [[1 if cl[0] == i else 0 for cl in cells] for i in range(r)] + [[1 if cl[1] == j else 0 for cl in cells] for j in range(c - 1)]
    D = (r - 1) * (c - 1); idx = {(i, j): k for k, (i, j) in enumerate([(i, j) for i in range(r - 1) for j in range(c - 1)])}; N = {}
    for (i, j) in cells:
        v = [0] * D
        if i < r - 1 and j < c - 1: v[idx[(i, j)]] = 1
        elif i < r - 1:
            for jj in range(c - 1): v[idx[(i, jj)]] = -1
        elif j < c - 1:
            for ii in range(r - 1): v[idx[(ii, j)]] = -1
        else: v = [1] * D
        N[(i, j)] = v
    return cells, A, N, D
def facets_float(f, g, cells, A):
    b = [float(x) for x in f] + [float(x) for x in g[:-1]]; out = []; n = len(cells)
    for ci, (i, j) in enumerate(cells):
        if f[i] + g[j] > 1: continue
        c = np.zeros(n + 1); c[-1] = -1; Aeq = np.hstack([np.array(A, float), np.zeros((len(A), 1))]); row = np.zeros(n + 1); row[ci] = 1
        Aeq = np.vstack([Aeq, row]); beq = b + [0.0]; Aub = []
        for k in range(n):
            if k != ci: r_ = np.zeros(n + 1); r_[k] = -1; r_[-1] = 1; Aub.append(r_)
        r = linprog(c, A_ub=np.array(Aub), b_ub=np.zeros(len(Aub)), A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * n + [(None, 1)], method="highs")
        if r.status == 0 and -r.fun > 1e-9: out.append((i, j))
    return out
def leximin_float(f, g, cells, A, S):
    n = len(cells); b = [float(x) for x in f] + [float(x) for x in g[:-1]]; fixed = {}; free = set(cells.index(c) for c in S); levels = []; lps = 0; saved = 0
    fre = {k: float(min(f[cells[k][0]], g[cells[k][1]])) for k in range(n)}
    while free:
        c = np.zeros(n + 1); c[-1] = -1; Aeq = np.hstack([np.array(A, float), np.zeros((len(A), 1))]); beq = list(b)
        for k, v in fixed.items(): row = np.zeros(n + 1); row[k] = 1; Aeq = np.vstack([Aeq, row]); beq.append(v)
        Aub = []
        for k in sorted(free): row = np.zeros(n + 1); row[k] = -1; row[-1] = 1; Aub.append(row)
        r = linprog(c, A_ub=np.array(Aub), b_ub=np.zeros(len(Aub)), A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * n + [(None, None)], method="highs"); lps += 1; t = r.x[-1]
        newly = []
        for k in sorted(free):
            if fre[k] <= t + 1e-9: newly.append(k); saved += 1; continue
            c2 = np.zeros(n); c2[k] = -1; Aub2 = []
            for kk in sorted(free): row = np.zeros(n); row[kk] = -1; Aub2.append(row)
            Aeq2 = np.array(A, float); beq2 = list(b)
            for kk, v in fixed.items(): row = np.zeros(n); row[kk] = 1; Aeq2 = np.vstack([Aeq2, row]); beq2.append(v)
            r2 = linprog(c2, A_ub=np.array(Aub2), b_ub=-t * np.ones(len(Aub2)), A_eq=Aeq2, b_eq=beq2, bounds=[(0, None)] * n, method="highs"); lps += 1
            if -r2.fun <= t + 1e-9: newly.append(k)
        if not newly: raise RuntimeError("stalled")
        for k in newly: fixed[k] = t; free.discard(k)
        levels.append((t, [cells[k] for k in newly]))
    Aeq2 = np.array(A, float); beq2 = list(b)
    for kk, v in fixed.items(): row = np.zeros(n); row[kk] = 1; Aeq2 = np.vstack([Aeq2, row]); beq2.append(v)
    x = linprog(np.zeros(n), A_eq=Aeq2, b_eq=beq2, bounds=[(0, None)] * n, method="highs").x
    return levels, x, lps, saved
def r_pattern(levels, N):
    out = []; prev = []
    for h, Sj in levels:
        r0 = np.linalg.matrix_rank(np.array(prev, float)) if prev else 0; prev = prev + [N[c] for c in Sj]; out.append(int(np.linalg.matrix_rank(np.array(prev, float)) - r0))
    return out
# population -----------------------------------------------------------
r, c = 4, 6; cells, A, N, D = shape(r, c); rng = np.random.default_rng(20261135); pop = []
while len(pop) < 2000:
    f, g = margins(rng, r, 240), margins(rng, c, 240)
    if f and g and nondeg(f, g): pop.append((f, g))
Af = np.array(A, float)
# LZ1 ------------------------------------------------------------------
rr = np.random.default_rng(1); bad1 = 0; n1 = 0
for t in range(500):
    f, g = pop[t]; b = [float(x) for x in f] + [float(x) for x in g[:-1]]; FS = float(f[0] + f[1]); FC = 1 - FS
    lo = np.array([max(0.0, float(gj) - FC) for gj in g]); hi = np.array([min(float(gj), FS) for gj in g])
    for _ in range(20):
        w = rr.normal(size=c); s = lo.copy(); rem = FS - lo.sum()
        for j in np.argsort(-w):
            add = min(hi[j] - lo[j], rem); s[j] += add; rem -= add
        closed = w @ s; obj = np.array([w[cl[1]] if cl[0] in (0, 1) else 0.0 for cl in cells])
        lp = -linprog(-obj, A_eq=Af, b_eq=b, bounds=[(0, None)] * len(cells), method="highs").fun; bad1 += abs(closed - lp) > 1e-9; n1 += 1
check("LZ1 IDENTITY prop:twowaycomplete: column-given-row-block region = box cut by the plane, any number of outcomes", bad1 == 0, f"{n1} support values; mismatches {bad1}")
# LZ2 ------------------------------------------------------------------
RG = [[0, 1], [2, 3]]; CG = [[0, 1], [2, 3], [4, 5]]; bad2 = 0; n2 = 0
for t in range(500):
    f, g = pop[t]; b = [float(x) for x in f] + [float(x) for x in g[:-1]]
    for I, rows in enumerate(RG):
        fI = float(sum(f[i] for i in rows))
        for J, cols in enumerate(CG):
            gJ = float(sum(g[j] for j in cols)); obj = np.array([1.0 if (cl[0] in rows and cl[1] in cols) else 0.0 for cl in cells])
            mn = linprog(obj, A_eq=Af, b_eq=b, bounds=[(0, None)] * len(cells), method="highs").fun; mx = -linprog(-obj, A_eq=Af, b_eq=b, bounds=[(0, None)] * len(cells), method="highs").fun
            bad2 += abs(mn - max(0.0, fI + gJ - 1)) > 1e-9 or abs(mx - min(fI, gJ)) > 1e-9; n2 += 1
check("LZ2 IDENTITY prop:coarsen: merge-respecting bands on the fine fibre equal those on the coarse fibre", bad2 == 0, f"{n2} coarse-cell bands; mismatches {bad2}")
# LZ3 ------------------------------------------------------------------
bad3 = 0
for f, g in pop:
    F = np.array([float(x) for x in f]); G = np.array([float(x) for x in g]); M = np.outer(F, G)
    Mc = np.array([[M[np.ix_(rows, cols)].sum() for cols in CG] for rows in RG]); Fc = np.array([F[rows].sum() for rows in RG]); Gc = np.array([G[cols].sum() for cols in CG])
    bad3 += np.abs(Mc - np.outer(Fc, Gc)).max() > 1e-15
check("LZ3 IDENTITY maximum entropy commutes with merging", bad3 == 0, f"2000 tables; violations {bad3}")
# LZ4 + LZ6 census -----------------------------------------------------
cc, Ac, Nc, Dc = shape(2, 3); pats = Counter(); nlev = Counter(); lps_tot = 0; saved_tot = 0; defects = []; full = 0; examples = {}; t0 = time.perf_counter()
for t, (f, g) in enumerate(pop):
    S = facets_float(f, g, cells, A); lv, x, lps, sv = leximin_float(f, g, cells, A, S); lps_tot += lps; saved_tot += sv
    rp = r_pattern(lv, N); pat = tuple(v for v in rp if v > 0); pats[pat] += 1; nlev[len(lv)] += 1; full += sum(rp) == D
    examples.setdefault(pat, t)
    fc = [f[0] + f[1], f[2] + f[3]]; gc = [g[0] + g[1], g[2] + g[3], g[4] + g[5]]; Sc = facets_float(fc, gc, cc, Ac); lvc, xc, _, _ = leximin_float(fc, gc, cc, Ac, Sc)
    X = x.reshape(r, c); merged = np.array([[X[np.ix_(rows, cols)].sum() for cols in CG] for rows in RG]); defects.append(float(np.abs(merged - xc.reshape(2, 3)).max()))
t_census = time.perf_counter() - t0; defects = np.array(defects)
check("LZ4 RECORD census of leximin patterns at 4 x 6", True,
      f"{len(pop)} tables in {t_census:.0f}s; patterns (dimensions removed per level) {dict(pats.most_common(10))}; levels {dict(sorted(nlev.items()))}; all 15 dimensions removed in {full}; "
      f"programmes {lps_tot}, blocked tests saved by Frechet pruning {saved_tot}")
# LZ5 Theorem 196 ------------------------------------------------------
def exact_leximin(f, g, cells, A, S):
    n = len(cells); AF = [[Fr(x) for x in row] for row in A]; b = list(f) + list(g[:-1]); fixed = {}; free = set(cells.index(cl) for cl in S); levels = []
    fre = {k: min(f[cells[k][0]], g[cells[k][1]]) for k in range(n)}
    while free:
        cvec = [Fr(0)] * n + [Fr(1)]; A_eq = [row + [Fr(0)] for row in AF] + [[Fr(1) if k == i else Fr(0) for k in range(n)] + [Fr(0)] for i in fixed]; b_eq = b + [fixed[i] for i in fixed]
        A_ub = [[Fr(-1) if k == i else Fr(0) for k in range(n)] + [Fr(1)] for i in sorted(free)]; tval, _ = certify_max(cvec, A_eq, b_eq, A_ub, [Fr(0)] * len(free), free={n})
        newly = []
        for i in sorted(free):
            if fre[i] <= tval: newly.append(i); continue
            c2 = [Fr(1) if k == i else Fr(0) for k in range(n)]; A_eq2 = AF + [[Fr(1) if k == j else Fr(0) for k in range(n)] for j in fixed]; b_eq2 = b + [fixed[j] for j in fixed]
            A_ub2 = [[Fr(-1) if k == j else Fr(0) for k in range(n)] for j in sorted(free)]; v, _ = certify_max(c2, A_eq2, b_eq2, A_ub2, [-tval] * len(free), free=set())
            if v == tval: newly.append(i)
        for i in newly: fixed[i] = tval; free.discard(i)
        levels.append((tval, [cells[i] for i in newly]))
    A_eq = AF + [[Fr(1) if k == j else Fr(0) for k in range(n)] for j in fixed]; b_eq = b + [fixed[j] for j in fixed]
    _, x = certify_max([Fr(0)] * n, A_eq, b_eq, None, None, free=set()); return levels, x
mp.mp.dps = 120
def scales_at(L, F, Nn, xstar, cells, Dd, u0=None):
    rows = [[mp.mpf(Nn[cl][i]) for i in range(Dd)] for cl in F]; c0 = [mp.mpf(xstar[cells.index(cl)].numerator) / xstar[cells.index(cl)].denominator for cl in F]
    u = [mp.mpf(0)] * Dd if u0 is None else list(u0)
    for _ in range(300):
        e = [mp.e ** (-L * (c0[a] + mp.fsum(rows[a][i] * u[i] for i in range(Dd)))) for a in range(len(F))]
        gv = mp.matrix([-L * mp.fsum(e[a] * rows[a][i] for a in range(len(F))) for i in range(Dd)]); H = mp.matrix(Dd, Dd)
        for i in range(Dd):
            for j in range(i, Dd): H[i, j] = H[j, i] = L * L * mp.fsum(e[a] * rows[a][i] * rows[a][j] for a in range(len(F)))
        st = mp.lu_solve(H, gv); u = [u[i] - st[i] for i in range(Dd)]
        if mp.norm(st) < mp.mpf(10) ** (-80): break
    ev = mp.eigsy(H / (L * L * mp.fsum(e)))[0]; return sorted(float(-mp.log(abs(ev[i])) / L) for i in range(Dd)), u
picked = [examples[p] for p, _ in pats.most_common(8)]; worst = 0.0; info5 = []; spec = {}
for t in picked:
    f, g = pop[t]; S = facets_float(f, g, cells, A)
    lv, xs = exact_leximin(f, g, cells, A, S); rp = r_pattern(lv, N); target = sorted(float(h - lv[0][0]) for (h, Sj), rr_ in zip(lv, rp) for _ in range(rr_))
    vals = {}; u = None
    for L in (60, 120, 240): vals[L], u = scales_at(L, S, N, xs, cells, D, u)
    ext = sorted(float(np.polyfit([1 / L for L in (60, 120, 240)], [vals[L][i] for L in (60, 120, 240)], 1)[1]) for i in range(D))
    err = max(abs(a - b_) for a, b_ in zip(ext, target)) if len(target) == D else 1.0; worst = max(worst, err)
    spec[str(t)] = dict(pattern=[v for v in rp if v > 0], levels=[str(h) for h, _ in lv], target=target, extrapolated=ext, error=err)
    info5.append(f"table {t}: r = {tuple(v for v in rp if v > 0)}, error {err:.4f}")
check("LZ5 IDENTITY Theorem 196 at 4 x 6: scale valuations = h_j - h_1 with multiplicity r_j (exact levels; extrapolated in 1/L)", worst < 0.01 and len(picked) >= 1, "; ".join(info5))
z = (defects <= 1e-9)
check("LZ6 RECORD leximin coarsening defect", True, f"share with defect <= 1e-9: {z.mean():.3f}; defect median {np.median(defects):.4f}, max {defects.max():.4f}")
check("LZ7 HYPOTHESIS leximin is not merge consistent in general: defect > 1e-9 in >= 50% of the census", (~z).mean() >= 0.5, f"share with defect > 1e-9: {(~z).mean():.3f}")
json.dump(dict(patterns={str(k): v for k, v in pats.items()}, levels=dict(nlev), full=full, lps=lps_tot, saved=saved_tot, t_census=t_census,
               defect_share_zero=float(z.mean()), defect_median=float(np.median(defects)), defect_max=float(defects.max()), spectral=spec, defects=defects.tolist()),
          open(os.path.join(HERE, "study35_four6", "four6.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
