"""Study 31: the shape of a conditioned state (sec:credalshape). Pre-registered in study31_shape/PREREG_study31.md, frozen by SHA-256
(study31_shape/FROZEN_PREREG_study31.txt) before any code for it ran. Exact geometry: identities of lem:cographic, prop:ratiocone and
prop:condpolytope, and records for fig:credalshape. New script; reuses the helpers of check_structbound.py (its code before the families
block, which itself reuses check_tolerance.py) and pce.stratledger; modifies nothing.
  SO1 IDENTITY lem:cographic (spanning-tree support; zero cells independent on the circulation space; cycle in the complement -> dependent)
  SO2 IDENTITY prop:ratiocone (segments and scalings stay in a ratio basis region)
  SO3 IDENTITY prop:condpolytope: vertices of K, block image, outcome polygon = hull of vertex images = support-function polygon; bands
  SO4 IDENTITY evidence commutes with conditioning; Hilbert distances preserved; after-evidence polygon = hull of transported images
  SO5 IDENTITY Fisher-Rao image on the unit sphere; band ends are extreme latitudes
  SO6 RECORD   vertex counts, areas, Hilbert diameters, active facets, changes
First run (the only run): 6/6. Making the figure then showed that, in all three figure tables, the before-evidence outcome region equals
the product of its coordinate bands; this was proved afterwards (prop:bandcomplete) and recorded on 100 more tables in
check_credalshape_addendum.py (SO7, post hoc). RuntimeWarnings from the reused helpers (logit at band ends) are filtered from the output
file and are benign (see check_structbound.py).
Output: check_credalshape_output.txt, study31_shape/credalshape.json (aggregates and the three figure tables' geometry; synthetic data).
"""
import os, sys, json, itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.spatial import ConvexHull
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
exec(open(os.path.join(HERE, "check_structbound.py")).read().split("FAM = {")[0])
results.clear()
from pce import stratledger as SL
# ------------------------------------------------------------------ SO1 lem:cographic
pop = population(3, 3, 120, 20261126); r, c = 3, 3; cells33 = [(i, j) for i in range(r) for j in range(c)]
Mrg = np.array([[1 if cl[0] == i else 0 for cl in cells33] for i in range(r)] + [[1 if cl[1] == j else 0 for cl in cells33] for j in range(c)], float)
_, sv, Vt = np.linalg.svd(Mrg); circ = Vt[np.linalg.matrix_rank(Mrg):]                  # basis of the circulation space (rows)
def is_tree(edges):
    par = list(range(r + c))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for (i, j) in edges:
        a, b = f(i), f(r + j)
        if a == b: return False
        par[a] = b
    return len(edges) == r + c - 1
from scipy.optimize import linprog
rng = np.random.default_rng(31); bad1 = 0; n1 = 0
for t in rng.choice(len(pop), 300, replace=False):
    f, g = pop[t]; b = [float(x) for x in f] + [float(x) for x in g]; w = rng.normal(size=9)
    x = linprog(-w, A_eq=Mrg, b_eq=b, bounds=[(0, None)] * 9, method="highs").x
    pos = [cells33[k] for k in range(9) if x[k] > 1e-10]; zer = [k for k in range(9) if x[k] <= 1e-10]
    ok = len(pos) == r + c - 1 and is_tree(pos) and np.linalg.matrix_rank(circ[:, zer]) == (r - 1) * (c - 1); bad1 += not ok; n1 += 1
dep_bad = 0; dep_n = 0; ind_bad = 0
for S in itertools.combinations(range(9), 4):
    comp = [cells33[k] for k in range(9) if k not in S]; rk = np.linalg.matrix_rank(circ[:, list(S)])
    if is_tree(comp): ind_bad += rk != 4
    else: dep_n += 1; dep_bad += rk == 4
check("SO1 IDENTITY lem:cographic: optimal vertices have spanning-tree support and independent zero-cell normals; cycles give dependence", bad1 == 0 and ind_bad == 0 and dep_bad == 0,
      f"vertices {n1}, failures {bad1}; all 126 four-cell sets: complement a tree -> independent (failures {ind_bad}); complement with a cycle ({dep_n} sets) -> dependent (failures {dep_bad})")
# ------------------------------------------------------------------ SO2 prop:ratiocone
P = SL.Problem(3, 3); Bm = np.array([P.b(f, g) for f, g in pop]).T; rr = np.random.default_rng(77); bad2 = 0; n2 = 0; bad2s = 0
for o in range(5):
    dv = rr.integers(1, 10, size=P.N).astype(float); nv = np.floor(dv * rr.uniform(0, 1, size=P.N)); L = SL.Library(P)
    asg, _, _, _ = SL.build(P, L, Bm, nv, dv, +1, np.random.default_rng(3).permutation(Bm.shape[1])); rp = np.random.default_rng(300 + o)
    ok2, _ = (None, None)
    for _ in range(400):
        k = asg[rp.integers(len(asg))]; mem = np.where(asg == k)[0]
        if len(mem) < 2: continue
        a, b_ = rp.choice(mem, 2, replace=False); lam = np.linspace(0, 1, 7)[1:-1]
        seg = np.array([(1 - l) * Bm[:, a] + l * Bm[:, b_] for l in lam]).T; ok, _ = L.certify(k, seg, nv, dv, +1); bad2 += int((~ok).sum()); n2 += len(lam)
        ok, _ = L.certify(k, 2 * Bm[:, [a]], nv, dv, +1); bad2s += int(not ok[0])
check("SO2 IDENTITY prop:ratiocone: ratio basis regions are convex cones", bad2 == 0 and bad2s == 0, f"segment points {n2}, outside {bad2}; scalings outside {bad2s}")
# ------------------------------------------------------------------ SO3-SO6 on P332
F = family_pairs(3, 3, 2, 600, 20261129); R = run_family("P332", F, np.random.default_rng(0)); cells = F["cells"]; nx = F["nx"]
FT = family_pairs(3, 3, 2, 600, 20261129, tol=E)
blk = F["den"]; Lc = np.array([float(x) for x in F["L"]]); AGG = np.array([[1.0 if cells[k][0] == a else 0.0 for k in blk] for a in range(3)])
H = []
for i in range(2):
    for j in range(2): H.append([(1 if x == i else -1 if x == 2 else 0) * (1 if y == j else -1 if y == 2 else 0) * (1 if z == 0 else -1) for (x, y, z) in cells])
H = np.array(H, int).T; HF = [[Fr(int(v)) for v in row] for row in H]
w0 = R["vals"]["hi0"] - R["vals"]["lo0"]; qs = np.percentile(w0, [25, 50, 75]); picks = [int(np.argmin(np.abs(w0 - q))) for q in qs]
def exact_vertex(q, key):
    k_ = R["asg"][key][q]; B = list(R["libs"][key].bases[k_]); AF = [[Fr(int(x)) for x in row] for row in F["A"]]
    xb = CL.gsolve([[AF[i][j] for j in B] for i in range(len(AF))], F["bF"][R["Di"][q]]); x = [Fr(0)] * F["M"]
    for i, j in enumerate(B): x[j] = xb[i]
    return x[:nx]
def enumerate_vertices(xb):
    V = []; seen = set()
    for S in itertools.combinations(range(nx), 4):
        th = CL.gsolve([HF[s] for s in S], [-xb[s] for s in S])
        if th is None: continue
        x = [xb[k] + sum(HF[k][p] * th[p] for p in range(4)) for k in range(nx)]
        if min(x) < 0: continue
        key_ = tuple(x)
        if key_ not in seen: seen.add(key_); V.append(x)
    return V
def to_q(x, Lw=None):
    xb = np.array([float(x[k]) for k in blk]) * (1 if Lw is None else Lw[blk]); pi = xb / xb.sum(); return pi, AGG @ pi
def hilbert(p, q):
    s = (p > 1e-15) | (q > 1e-15)
    if ((p > 1e-15) != (q > 1e-15)).any(): return np.inf
    l = np.log(p[s] / q[s]); return float(l.max() - l.min())
def support_polygon(FF, b, Lw, ndir=720):
    M = FF["M"]; pts = []
    for ang in np.linspace(0, 2 * np.pi, ndir, endpoint=False):
        w = np.array([np.cos(ang), np.sin(ang)]); n = np.zeros(M); d = np.zeros(M); Lf = np.ones(M); Lf[:nx] = Lw
        for k in blk: d[k] = Lf[k]; n[k] = Lf[k] * (w[0] * (cells[k][0] == 0) + w[1] * (cells[k][0] == 1))
        z = CL.solve_ratio(FF["A"], b, n, d, +1); pi = z[blk] * Lf[blk] / (z[blk] * Lf[blk]).sum(); pts.append(AGG @ pi)
    return np.array(pts)
def hull(P2):
    h = ConvexHull(P2[:, :2]); return P2[h.vertices], h.volume
def hull_close(A_, B_, tol=1e-9):
    dirs = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 360, endpoint=False)])
    return float(np.abs((A_[:, :2] @ dirs.T).max(0) - (B_[:, :2] @ dirs.T).max(0)).max())
ok3 = True; ok4 = True; ok5 = True; info3 = []; FIG = {}; info4 = []
for tag, q in zip(("p25", "p50", "p75"), picks):
    t = R["Di"][q]; xb = exact_vertex(q, "lo0"); V = enumerate_vertices(xb); VA = np.array([[float(v) for v in x] for x in V])
    marg_ok = all(np.allclose(F["RA"] @ VA[i], F["RA"] @ VA[0], atol=0) for i in range(len(V))) and np.allclose(F["RA"] @ VA[0], F["RA"] @ np.array([float(v) for v in xb]))
    Q0 = np.array([to_q(x)[1] for x in V]); PI0 = np.array([to_q(x)[0] for x in V]); Q1 = np.array([to_q(x, Lc)[1] for x in V]); PI1 = np.array([to_q(x, Lc)[0] for x in V])
    S0 = support_polygon(F, F["Bm"][:, t], np.ones(nx)); S1 = support_polygon(F, F["Bm"][:, t], Lc); ST = support_polygon(FT, FT["Bm"][:, t], np.ones(nx))
    g0 = hull_close(Q0, S0); g1 = hull_close(Q1, S1)
    band = (Q0[:, 0].min(), Q0[:, 0].max()); bandok = abs(band[0] - R["vals"]["lo0"][q]) < 1e-9 and abs(band[1] - R["vals"]["hi0"][q]) < 1e-9
    band1 = (Q1[:, 0].min(), Q1[:, 0].max()); band1ok = abs(band1[0] - R["vals"]["lo1"][q]) < 1e-9 and abs(band1[1] - R["vals"]["hi1"][q]) < 1e-9
    ok3 &= marg_ok and g0 < 1e-9 and bandok; info3.append(f"{tag}: |V(K)| {len(V)}, margins {marg_ok}, hull gap {g0:.1e}, band ends match {bandok}")
    # SO4
    comm = max(np.abs(to_q([v * l for v, l in zip(x, F['L'])])[0] - to_q(x, Lc)[0]).max() for x in V)
    hd_bad = 0; hd_n = 0
    for i, j in itertools.combinations(range(len(V)), 2):
        a0, b0 = hilbert(PI0[i], PI0[j]), hilbert(PI1[i], PI1[j])
        if np.isfinite(a0): hd_n += 1; hd_bad += abs(a0 - b0) > 1e-9
        else: hd_bad += np.isfinite(b0)
    ok4 &= comm < 1e-12 and hd_bad == 0 and g1 < 1e-9 and band1ok; info4.append(f"{tag}: commute {comm:.1e}, Hilbert pairs {hd_n} finite, changed {hd_bad}, after-evidence hull gap {g1:.1e}, bands {band1ok}")
    # SO5
    H0v, A0 = hull(np.column_stack([Q0[:, :2], Q0[:, 2]])); sph = np.sqrt(np.clip(Q0, 0, None)); ok5 &= np.allclose((sph ** 2).sum(1), 1) and abs(np.sqrt(band[1]) - sph[:, 0].max()) < 1e-12
    # SO6 records
    H1v, A1 = hull(Q1); HTv, AT = hull(ST)
    def diam(P_):
        return max((hilbert(a, b) for a, b in itertools.combinations(P_, 2)), default=0.0)
    def active(Qset, Vset):
        out = []
        for v in hull(Qset)[0]:
            pre = [k for k in range(len(Vset)) if np.abs(Qset[k] - v).max() < 1e-12]
            z = set.intersection(*[set(np.where(Vset[k] <= 0)[0].tolist()) for k in pre]); out.append(dict(q=v.tolist(), preimages=len(pre), zero_cells=sorted(z)))
        return out
    act0 = active(Q0, VA); act1 = active(Q1, VA)
    edges = []; tight = [set(np.where(VA[i] <= 1e-15)[0].tolist()) for i in range(len(V))]
    for i, j in itertools.combinations(range(len(V)), 2):
        T_ = sorted(tight[i] & tight[j])
        if T_ and np.linalg.matrix_rank(H[T_].astype(float)) == 3: edges.append([i, j])
    FIG[tag] = dict(table=int(t), width=float(w0[q]), band0=list(band), band1=list(band1), bandT=[float(ST[:, 0].min()), float(ST[:, 0].max())],
                    vertices_K=len(V), vertices_pi=int(len(np.unique(np.round(PI0, 12), axis=0))), polygon0=H0v.tolist(), polygon1=H1v.tolist(), polygonT=HTv.tolist(),
                    area0=float(A0), area1=float(A1), areaT=float(AT), hdiam0=diam(H0v), hdiam1=diam(H1v), hdiamT=diam(HTv),
                    coord_bands0=[[float(Q0[:, a].min()), float(Q0[:, a].max())] for a in range(3)], active0=act0, active1=act1,
                    K_vertices=VA.tolist(), K_edges=edges, Q0=Q0.tolist(), Q1=Q1.tolist())
check("SO3 IDENTITY prop:condpolytope: outcome polygon = hull of vertex images = support-function polygon; band ends", ok3, "; ".join(info3))
check("SO4 IDENTITY evidence commutes with conditioning, preserves Hilbert distances, transports the polygon", ok4, "; ".join(info4))
check("SO5 IDENTITY Fisher-Rao image on the unit sphere; band ends are extreme latitudes", ok5)
check("SO6 RECORD vertex counts, areas, Hilbert diameters, active facets", True,
      "; ".join(f"{k}: |V(K)| {v['vertices_K']}, |V(pi K)| {v['vertices_pi']}, polygon vertices {len(v['polygon0'])}/{len(v['polygon1'])}/{len(v['polygonT'])} (before/after/tolerance), "
                f"area {v['area0']:.4f}/{v['area1']:.4f}/{v['areaT']:.4f}, Hilbert diameter {v['hdiam0']:.3f}/{v['hdiam1']:.3f}/{v['hdiamT']:.3f}" for k, v in FIG.items()))
json.dump(FIG, open(os.path.join(HERE, "study31_shape", "credalshape.json"), "w"), indent=1, default=float)
print(f"{sum(results)}/{len(results)} passed")
