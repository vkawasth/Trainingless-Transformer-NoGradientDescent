"""Study 32: the unified-geometry frontispiece (fig:frontispiece, sec:overview). Pre-registered in study32_frontispiece/PREREG_study32.md,
frozen by SHA-256 (study32_frontispiece/FROZEN_PREREG_study32.txt) before any code for it ran. One example, E1 of sec:threebythree,
followed from margins to decision; every panel is computed and checked. New script; uses pce/cpl33.py and pce/credalledger.py and the
spectral routine of check_threebythree.py (copied, unchanged in substance); modifies nothing.
  UG1 IDENTITY base of margins: chamber walls in the slice g = g(E1) are the lines f_i = subset sum of g; vertex count constant per cell
  UG2 IDENTITY the fibre: vertices = feasible spanning-tree bases (count = TT2 record), simple, adjacency = 4 shared basic cells
  UG3 IDENTITY dual bases at the marked vertex: spanning tree; its 4 edges <-> its 4 zero cells
  UG4 IDENTITY Theorem 196 on E1: levels and r = TT2; scale valuations -> h_j - h_1 with multiplicity r_j; x(L) -> leximin point
  UG5 IDENTITY the decision: conditional region = hull of vertex images = product of bands; after evidence = hull of transported images;
               band ends = direct programmes
First run (the only run): 5/5.
Output: check_frontispiece_output.txt, study32_frontispiece/frontispiece.json.
"""
import os, sys, json, itertools
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
from scipy.spatial import ConvexHull
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import cpl33 as C, credalledger as CL
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
f = [Fr(30, 60), Fr(20, 60), Fr(10, 60)]; g = [Fr(24, 60), Fr(21, 60), Fr(15, 60)]; CELLS = C.CELLS
TT = json.load(open(os.path.join(HERE, "threebythree_results.json")))["examples"]["E1"]
def is_tree(cells):
    par = list(range(6))
    def fd(x):
        while par[x] != x: x = par[x]
        return x
    for (i, j) in cells:
        a, b = fd(i), fd(3 + j)
        if a == b: return False
        par[a] = b
    return len(cells) == 5
TREES = [B for B in itertools.combinations(range(9), 5) if is_tree([CELLS[k] for k in B])]
AM = [[Fr(1) if CELLS[k][0] == i else Fr(0) for k in range(9)] for i in range(3)] + [[Fr(1) if CELLS[k][1] == j else Fr(0) for k in range(9)] for j in range(2)]
def vertex(B, f, g):
    rhs = list(f) + list(g[:2]); xb = CL.gsolve([[AM[i][k] for k in B] for i in range(5)], rhs); x = [Fr(0)] * 9
    for a, k in enumerate(B): x[k] = xb[a]
    return x
def vertices(f, g):
    out = []
    for B in TREES:
        x = vertex(B, f, g)
        if min(x) >= 0: out.append((B, x))
    return out
# UG1 -------------------------------------------------------------------
S = sorted({float(x) for x in g} | {float(a + b) for a, b in itertools.combinations(g, 2)}); rng = np.random.default_rng(32)
def count_at(ff):
    n = 0
    for B in TREES:
        rhs = list(ff) + [float(x) for x in g[:2]]; M = np.array([[float(AM[i][k]) for k in B] for i in range(5)]); xb = np.linalg.solve(M, rhs)
        n += (xb >= -1e-12).all()
    return int(n)
cellmap = {}; bad1 = 0
for _ in range(2000):
    ff = rng.dirichlet([1, 1, 1]); key = tuple((ff[i] > s) for i in range(3) for s in S)
    if min(abs(ff[i] - s) for i in range(3) for s in S) < 1e-9: continue
    n = count_at(ff); bad1 += key in cellmap and cellmap[key] != n; cellmap.setdefault(key, n)
interior = min(abs(float(f[i]) - s) for i in range(3) for s in S) > 1e-9
grid = []
for a in np.linspace(0, 1, 121):
    for b in np.linspace(0, 1 - a, max(2, int(121 * (1 - a)))):
        c = 1 - a - b
        if min(a, b, c) > 1e-6 and min(abs(x - s) for x in (a, b, c) for s in S) > 1e-6: grid.append([a, b, c, count_at([a, b, c])])
check("UG1 IDENTITY chamber walls of the slice are the lines f_i = subset sums of g; vertex count constant on each cell; E1 interior",
      bad1 == 0 and interior, f"{len(cellmap)} cells met by 2000 points, inconsistent {bad1}; wall levels {[round(s, 4) for s in S]}; E1 interior {interior}; E1 vertex count {count_at([float(x) for x in f])}")
# UG2 -------------------------------------------------------------------
V = vertices(f, g); nv = len(V); adj = [[j for j in range(nv) if j != i and len(set(V[i][0]) & set(V[j][0])) == 4] for i in range(nv)]
deg = sorted({len(a) for a in adj})
check("UG2 IDENTITY vertices are the feasible spanning-tree bases; count = TT2; simple; adjacency shares 4 basic cells", nv == TT["nverts"] and deg == [4],
      f"{len(TREES)} spanning trees, {nv} feasible (TT2: {TT['nverts']}); degrees {deg}")
# UG5 prerequisites: decision map q = column distribution given rows {0,1}
den = f[0] + f[1]
def q_of(x, Lw=None):
    w = [Fr(1)] * 9 if Lw is None else Lw; num = [sum(w[k] * x[k] for k in range(9) if CELLS[k][0] in (0, 1) and CELLS[k][1] == j) for j in range(3)]; t = sum(num)
    return [n / t for n in num]
Lw = [Fr(1) + Fr((i + 2 * j) % 3, 2) for (i, j) in CELLS]
Q0 = np.array([[float(v) for v in q_of(x)] for _, x in V]); Q1 = np.array([[float(v) for v in q_of(x, Lw)] for _, x in V])
mark = int(np.argmax(Q0[:, 0]))
# UG3 -------------------------------------------------------------------
Bm, xm = V[mark]; pos = [CELLS[k] for k in Bm]; zer = [k for k in range(9) if k not in Bm]
entering = sorted(list(set(V[j][0]) - set(Bm))[0] for j in adj[mark])
check("UG3 IDENTITY the marked vertex: positive cells a spanning tree; its 4 edges <-> its 4 zero cells", is_tree(pos) and entering == sorted(zer),
      f"marked vertex = upper end of the band of q0; positive cells {pos}; zero cells {[CELLS[k] for k in zer]}; entering cells of the 4 edges {[CELLS[k] for k in entering]}")
# UG4 -------------------------------------------------------------------
F = C.exact_facets(f, g); lv, xstar = C.exact_leximin(f, g, F); r = C.r_pattern(lv); N = C.normals()
target = sorted(float(h - lv[0][0]) for (h, Sj), rr in zip(lv, r) for _ in range(rr))
mp.mp.dps = 160
def crit(L, u0=None):
    rows = [[mp.mpf(int(N[c][i])) for i in range(4)] for c in F]; c0 = [mp.mpf(xstar[CELLS.index(c)].numerator) / xstar[CELLS.index(c)].denominator for c in F]
    u = [mp.mpf(0)] * 4 if u0 is None else list(u0)
    for _ in range(300):
        e = [mp.e ** (-L * (c0[a] + mp.fsum(rows[a][i] * u[i] for i in range(4)))) for a in range(len(F))]
        gv = mp.matrix([-L * mp.fsum(e[a] * rows[a][i] for a in range(len(F))) for i in range(4)]); H = mp.matrix(4, 4)
        for i in range(4):
            for j in range(4): H[i, j] = L * L * mp.fsum(e[a] * rows[a][i] * rows[a][j] for a in range(len(F)))
        st = mp.lu_solve(H, gv); u = [u[i] - st[i] for i in range(4)]
        if mp.norm(st) < mp.mpf(10) ** (-100): break
    ev = mp.eigsy(H / (L * L * mp.fsum(e)))[0]
    x = [float(xstar[k]) + sum(float(N[CELLS[k]][i]) * float(u[i]) for i in range(4)) for k in range(9)]
    return u, x, sorted(float(-mp.log(abs(ev[i])) / L) for i in range(4))
Ls = [5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240]; path = []; vals = {}; u = None
for L in Ls:
    u, x, sv = crit(L, u); path.append(x); vals[L] = sv
xs = np.array([float(v) for v in xstar]); dist = [float(np.linalg.norm(np.array(p) - xs)) for p in path]
ext = [float(np.polyfit([1 / L for L in (60, 120, 240)], [vals[L][i] for L in (60, 120, 240)], 1)[1]) for i in range(4)]
err = max(abs(a - b) for a, b in zip(sorted(ext), target)); levels_ok = [str(h) for h, _ in lv] == [h for h, _ in TT["levels"]] and r == TT["r"]
mono = all(b <= a + 1e-12 for a, b in zip(dist, dist[1:]))
check("UG4 IDENTITY Theorem 196 on E1: levels and rank increments; scale valuations; x(L) -> leximin point", levels_ok and err < 0.01 and mono and dist[-1] < 0.01,
      f"levels {[str(h) for h, _ in lv]}, r {r} (TT2 match {levels_ok}); targets {[round(t, 4) for t in target]}, extrapolated {[round(t, 4) for t in sorted(ext)]}, error {err:.4f}; "
      f"distance to the leximin point {[round(d, 4) for d in dist]}")
# UG5 -------------------------------------------------------------------
A_std = np.array([[1 if CELLS[k][0] == i else 0 for k in range(9)] for i in range(3)] + [[1 if CELLS[k][1] == j else 0 for k in range(9)] for j in range(2)], float)
b_std = np.array([float(v) for v in list(f) + list(g[:2])])
def band(j, w):
    n = np.array([float(w[k]) if CELLS[k][0] in (0, 1) and CELLS[k][1] == j else 0 for k in range(9)]); d = np.array([float(w[k]) if CELLS[k][0] in (0, 1) else 0 for k in range(9)])
    out = []
    for s in (-1, +1):
        z = CL.solve_ratio(A_std, b_std, n, d, s); out.append(float(n @ z / (d @ z)))
    return out
ones = [Fr(1)] * 9; bad5 = 0; bands0 = []; bands1 = []
for j in range(3):
    b0 = band(j, ones); b1 = band(j, Lw); bands0.append(b0); bands1.append(b1)
    bad5 += abs(b0[0] - Q0[:, j].min()) > 1e-9 or abs(b0[1] - Q0[:, j].max()) > 1e-9 or abs(b1[0] - Q1[:, j].min()) > 1e-9 or abs(b1[1] - Q1[:, j].max()) > 1e-9
def tern(Q): Q = np.atleast_2d(Q); return np.column_stack([Q[:, 1] + Q[:, 2] / 2, Q[:, 2] * np.sqrt(3) / 2])
def prod_area(bands):
    pts = []
    for a, b in itertools.combinations(range(3), 2):
        for va in bands[a]:
            for vb in bands[b]:
                q = np.zeros(3); q[a] = va; q[b] = vb; q[3 - a - b] = 1 - va - vb
                if all(bands[k][0] - 1e-12 <= q[k] <= bands[k][1] + 1e-12 for k in range(3)): pts.append(q)
    return ConvexHull(tern(np.array(pts))).volume
a0 = ConvexHull(tern(Q0)).volume; a1 = ConvexHull(tern(Q1)).volume; r0 = a0 / prod_area(bands0); r1 = a1 / prod_area(bands1)
check("UG5 IDENTITY the decision region: hull of vertex images = product of bands; transported after evidence; band ends = direct programmes",
      bad5 == 0 and abs(r0 - 1) < 1e-9, f"band ends mismatched {bad5}; region/product area before {r0:.6f}, after evidence {r1:.3f}; bands before {[[round(v, 4) for v in b] for b in bands0]}")
OUT = dict(f=[str(x) for x in f], g=[str(x) for x in g], wall_levels=S, grid=grid, vertices=[[float(v) for v in x] for _, x in V], bases=[list(B) for B, _ in V],
           adj=adj, mark=mark, pos=[list(c) for c in pos], zero=[list(CELLS[k]) for k in zer], xm=[str(v) for v in xm], levels=[[str(h), [list(c) for c in Sj]] for h, Sj in lv], r=r,
           target=target, scale_values={str(L): v for L, v in vals.items()}, extrapolated=sorted(ext), path=path, Ls=Ls, leximin=[float(v) for v in xstar],
           Q0=Q0.tolist(), Q1=Q1.tolist(), bands0=bands0, bands1=bands1, ratio_after=r1, facets=[list(c) for c in F])
json.dump(OUT, open(os.path.join(HERE, "study32_frontispiece", "frontispiece.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
