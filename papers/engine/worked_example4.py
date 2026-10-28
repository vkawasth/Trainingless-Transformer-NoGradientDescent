"""worked_example4.py -- the worked example with a fourth harm, computed in stages (lab paper, Section "The example with four variables").

Truth: weights (0.6, 0.4); harm probabilities a1 = (0.5, 0.1, 0.4, 0.2), a2 = (0.05, 0.5, 0.1, 0.7) for A, B, C, D. The ABC face is the
three-variable example of Section 17 exactly. Every number in the section is computed here, written to ../lab/worked_example4.json, and
the W4 checks assert it.

Stage 1  the law and its four faces (each face a 2x2x2 table with its own line, capacities, Birch point, hyperdeterminant; one common r)
Stage 2  three-way interactions: the truth's, each face's Birch point, and those induced on the faces by the global pairwise
         maximum-entropy law (all interactions of order >= 3 removed)
Stage 3  the pair fibre: a 5-dimensional polytope; its lattice, vertices, facets, simplicity, smoothness (Delzant), volume; Kaehler data
Stage 4  the two-type model inside it: a conic arc (three linear equations and one quadric); the four-variable segment R4 and what the
         fourth harm adds to the ABC prediction
Stage 5  the decision and the hidden channel with four variables: capacities on 16 cells, the walls, and the second-triple test
Stage 6  chambers: neighbouring choices of the fourth harm give a smooth chamber, a wall (not simple) and orbifold chambers
"""
import itertools, json, os, sys
import numpy as np
from fractions import Fraction as Fr
from numpy.polynomial import polynomial as Pn
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce.lab import law, moments, fit_v, segment, explanation, r_of_t
from pce import segment_decision as SD
from pce import injection as IJ

OUT = {}; results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def f6(x): return float(np.round(float(x), 6))

W = (Fr(6, 10), Fr(4, 10)); A1 = [Fr(5, 10), Fr(1, 10), Fr(4, 10), Fr(2, 10)]; A2 = [Fr(5, 100), Fr(5, 10), Fr(1, 10), Fr(7, 10)]
X = IJ.outcomes(4); NAMES = "ABCD"; TRIPLES = list(itertools.combinations(range(4), 3))
def cellq(x, a): return np.prod([a[i] if x[i] else 1 - a[i] for i in range(4)])
pq = [W[0] * cellq(x, A1) + W[1] * cellq(x, A2) for x in X]           # exact rationals
P = law(0.6, np.array([float(z) for z in A1]), np.array([float(z) for z in A2])); p = IJ.as_vec(P)
mu, M, mS = moments(P); v = fit_v(mu, M); v = v if v[0] > 0 else -v        # label the types as in Section 17 (v_A > 0)
r0 = mS((0, 1, 2)) / np.prod(v[:3])
lab = lambda S: "".join(NAMES[i] for i in S)

# ---------------- Stage 1: the law and its faces ----------------
OUT["cells"] = {"".join(map(str, x)): str(c) for x, c in zip(X, pq)}
faces = {}
for S in TRIPLES:
    F = P.sum(axis=tuple(i for i in range(4) if i not in S)); f = IJ.as_vec(F); c3 = IJ.chi(3, (0, 1, 2))
    lo, hi = IJ.stealth_range(F, (0, 1, 2)); muF, MF, mSF = moments(F)
    Dm = MF[0, 1] * MF[0, 2] * MF[1, 2]; mT = mSF((0, 1, 2))
    odds = np.prod(f[c3 < 0]) / np.prod(f[c3 > 0])
    g = lambda s: np.sum(c3 * np.log(f + s * c3)); from scipy.optimize import brentq
    sb = brentq(g, lo + 1e-12, hi - 1e-12)
    faces[lab(S)] = {"capacity_plus": f6(hi), "capacity_minus": f6(-lo), "m_S": f6(mT), "v_S": f6(np.prod(v[list(S)])), "r_S": f6(mT / np.prod(v[list(S)])),
                     "odds_ratio": f6(odds), "birch_s": f6(sb), "Delta_m": float(Dm), "Det_roots_imag": f6(2 * np.sqrt(Dm)) if Dm > 0 else None,
                     "Det_positive_on_line": bool(Dm > 0)}
OUT["faces"] = faces
abc3 = law(0.6, np.array([.5, .1, .4]), np.array([.05, .5, .1]))
check("W4-1 the ABC face is the three-variable example exactly; all four faces share r = -0.408 (one latent coordinate)",
      np.allclose(P.sum(axis=3), abc3, atol=1e-15) and all(abs(faces[k]["r_S"] - r0) < 1e-6 for k in faces) and abs(r0 + 0.2 / np.sqrt(0.24)) < 1e-9,
      ", ".join(f"{k}: r {faces[k]['r_S']}" for k in faces))
check("W4-2 each face's hyperdeterminant m_S^2 + 4 Delta_S is positive on its whole line (Delta_S > 0 on every face)", all(faces[k]["Det_positive_on_line"] for k in faces))

# ---------------- Stage 2: three-way interactions ----------------
def ipf_pairwise(P, iters=5000):
    Q = np.full_like(P, 1 / 16)
    for _ in range(iters):
        for i, j in itertools.combinations(range(4), 2):
            ax = tuple(k for k in range(4) if k not in (i, j)); t = P.sum(axis=ax); q = Q.sum(axis=ax)
            shp = [1] * 4; shp[i] = 2; shp[j] = 2; Q = Q * (t / q).reshape(shp)
    return Q
Pme = ipf_pairwise(P); pme = IJ.as_vec(Pme); muE, ME, mSE = moments(Pme)
def odds3(F):
    f = IJ.as_vec(F); c3 = IJ.chi(3, (0, 1, 2)); return np.prod(f[c3 < 0]) / np.prod(f[c3 > 0])
induced = {}
for S in TRIPLES:
    FE = Pme.sum(axis=tuple(i for i in range(4) if i not in S)); induced[lab(S)] = {"odds_ratio": f6(odds3(FE)), "m_S": f6(mSE(S)),
        "face_birch_m_S": f6(faces[lab(S)]["m_S"] + faces[lab(S)]["birch_s"])}
c4 = IJ.chi(4, (0, 1, 2, 3)); four_way_truth = np.prod(p[c4 > 0]) / np.prod(p[c4 < 0])
log_lin_ok = True
lp = np.log(pme); H = np.array([[np.prod([2 * x[i] - 1 for i in S]) for S in [()] + [(i,) for i in range(4)] + list(itertools.combinations(range(4), 2))] for x in X])
coef, res, *_ = np.linalg.lstsq(H, lp, rcond=None); log_lin_ok = np.max(np.abs(H @ coef - lp)) < 1e-9
OUT["three_way"] = {"truth_odds_ratios": {k: faces[k]["odds_ratio"] for k in faces}, "global_maxent_face_odds_ratios": {k: induced[k]["odds_ratio"] for k in induced},
                    "global_maxent_m_S": {k: induced[k]["m_S"] for k in induced}, "face_birch_m_S": {k: induced[k]["face_birch_m_S"] for k in induced},
                    "truth_four_way_odds_ratio": f6(four_way_truth), "global_maxent_cells": [f6(z) for z in pme]}
check("W4-3 the global pairwise maximum-entropy law has exactly the six pair tables and no interactions of order >= 3 (log-linear in pairs)",
      log_lin_ok and all(np.allclose(Pme.sum(axis=tuple(k for k in range(4) if k not in (i, j))), P.sum(axis=tuple(k for k in range(4) if k not in (i, j))), atol=1e-12) for i, j in itertools.combinations(range(4), 2)))
diffs = [abs(induced[k]["odds_ratio"] - 1) for k in induced]; offbirch = [abs(induced[k]["m_S"] - induced[k]["face_birch_m_S"]) for k in induced]
check("W4-4 marginalising induces three-way interaction: the global maximum-entropy law's faces are not their own Birch points (odds ratios != 1)",
      min(diffs) > 1e-3 and min(offbirch) > 1e-6, "face odds ratios " + ", ".join(f"{k} {induced[k]['odds_ratio']:.4f}" for k in induced))

# ---------------- Stage 3: the pair fibre polytope ----------------
chis = [np.array([int(np.prod([2 * x[i] - 1 for i in S])) for x in X]) for S in TRIPLES + [(0, 1, 2, 3)]]
def gcd_minors(Bm):
    from math import gcd
    g = 0
    for rows in itertools.combinations(range(16), 5):
        d = int(round(np.linalg.det(Bm[list(rows), :]))); g = gcd(g, abs(d))
    return g
cands = []
for cs in itertools.product((-1, 0, 1), repeat=5):
    if not any(cs): continue
    w_ = sum(c * ch for c, ch in zip(cs, chis))
    for k in (4, 3, 2, 1, 0):
        if np.all(w_ % (2 ** k) == 0): cands.append(tuple(w_ // (2 ** k))); break
cands = sorted(set(cands), key=lambda c: (np.abs(c).sum(), c)); basis = []
for c in cands:
    t = basis + [np.array(c)]
    if np.linalg.matrix_rank(np.array(t).T) == len(t): basis = t
    if len(basis) == 5: break
B = np.array(basis).T; Bchi = np.array(chis).T
idx_chi = gcd_minors(Bchi); idx_B = gcd_minors(B)
Amarg = np.array([[1] * 16] + [[int(x[i]) for x in X] for i in range(4)] + [[int(x[i] and x[j]) for x in X] for i, j in itertools.combinations(range(4), 2)])
def enumerate_vertices(pv, tol=1e-12):
    verts = []
    for rows in itertools.combinations(range(16), 5):
        Mx = B[list(rows), :]
        if abs(np.linalg.det(Mx)) < 1e-9: continue
        y = np.linalg.solve(Mx, -pv[list(rows)]); vals = pv + B @ y
        if vals.min() < -tol: continue
        if not any(np.allclose(y, u[0], atol=1e-10) for u in verts): verts.append((y, tuple(np.where(np.abs(vals) < tol)[0])))
    return verts
verts = enumerate_vertices(p)
tight = [t for _, t in verts]; simple = all(len(t) == 5 for t in tight)
dets = [abs(int(round(np.linalg.det(B[list(t[:5]), :])))) for t in tight]; smooth = simple and all(d == 1 for d in dets)
facet_cells = sorted({i for t in tight for i in t}); nonfacets = [i for i in range(16) if i not in facet_cells]
# lattice volume (in the coordinates of the saturated basis) by triangulating the convex hull
from scipy.spatial import ConvexHull, Delaunay
Y = np.array([y for y, _ in verts]); tri = Delaunay(Y); vol = sum(abs(np.linalg.det(Y[s_[1:]] - Y[s_[0]])) / 120 for s_ in tri.simplices)
nvol = vol * 120                                                       # normalised volume (5! times Euclidean)
OUT["fibre"] = {"dimension": 5, "chi_lattice_index": int(idx_chi), "saturated_basis_index": int(idx_B), "vertices": len(verts), "simple": bool(simple),
                "smooth_Delzant": bool(smooth), "facet_cells": ["".join(map(str, X[i])) for i in facet_cells], "non_facet_cells": ["".join(map(str, X[i])) for i in nonfacets],
                "kaehler_facet_values": {"".join(map(str, X[i])): str(pq[i]) for i in facet_cells}, "lattice_volume": float(vol), "normalised_volume": float(nvol),
                "euler_characteristic": len(verts)}
check("W4-5 the pair fibre is 5-dimensional; the four triple characters and the four-way character span a sublattice of index 16, and a saturated basis (entries 0, +-1) exists",
      np.linalg.matrix_rank(Amarg) == 11 and idx_chi == 16 and idx_B == 1)
check("W4-6 at this truth the fibre is a simple, smooth (Delzant) polytope with 20 vertices and 9 facets: a smooth toric 5-fold with Euler characteristic 20",
      simple and smooth and len(verts) == 20 and len(facet_cells) == 9, f"facet cells {OUT['fibre']['facet_cells']}; volume {vol:.3e}")
# f-vector, h-vector and Betti numbers of the smooth toric 5-fold (b_{2i} = h_i of the simple polytope)
from math import comb
fv = {}
for j in range(0, 6):
    subs = set()
    for t in tight:
        for c_ in itertools.combinations(sorted(t), j): subs.add(c_)
    fv[5 - j] = len(subs)                                                  # faces of dimension 5 - j
fvec = [fv[k] for k in range(6)]                                           # f_0 (vertices) ... f_5 (the polytope)
# h-vector of a simple d-polytope: sum_i h_i t^i = sum_k f_k (t - 1)^k
d_ = 5; hpoly = np.zeros(d_ + 1)
for k in range(d_ + 1):
    for i in range(k + 1): hpoly[i] += fvec[k] * comb(k, i) * (-1) ** (k - i)
hvec = [int(round(z)) for z in hpoly]
OUT["fibre"]["f_vector"] = fvec; OUT["fibre"]["h_vector_betti"] = hvec
pairs_faces = {}
for S1, S2 in itertools.combinations(TRIPLES, 2):
    c1 = np.array([np.prod([x[i] - mu[i] for i in S1]) for x in X]); c2 = np.array([np.prod([x[i] - mu[i] for i in S2]) for x in X])
    pts2 = np.array([[c1 @ (p + B @ y), c2 @ (p + B @ y)] for y, _ in verts]); hull = ConvexHull(pts2)
    rng1 = np.ptp(pts2[:, 0]); rng2 = np.ptp(pts2[:, 1]); pairs_faces[lab(S1) + "x" + lab(S2)] = bool(abs(hull.volume - rng1 * rng2) < 1e-12 and len(hull.vertices) == 4)
    if not pairs_faces[lab(S1) + "x" + lab(S2)]:
        OUT.setdefault("fibre_cut", {})[lab(S1) + "x" + lab(S2)] = {"hull_vertices": [[f6(a_), f6(b_)] for a_, b_ in pts2[hull.vertices]],
            "area_ratio": f6(hull.volume / (rng1 * rng2)), "box": [[f6(pts2[:, 0].min()), f6(pts2[:, 0].max())], [f6(pts2[:, 1].min()), f6(pts2[:, 1].max())]]}
OUT["fibre"]["two_face_projections_are_products"] = pairs_faces
check("W4-6b the f-vector is %s; the h-vector, which gives the Betti numbers b_0, b_2, ..., b_10 of the toric 5-fold, is %s: symmetric (Dehn-Sommerville / Poincare duality) and summing to 20" % (fvec, hvec),
      hvec == hvec[::-1] and sum(hvec) == 20 and all(z >= 0 for z in hvec))
nonprod = [k for k, v_ in pairs_faces.items() if not v_]
check("W4-6c five of the six two-face projections are products of the faces' lines; ACD x BCD is not: its corners are cut, so those two faces' three-way moments are coupled",
      nonprod == ["ACDxBCD"], json.dumps(OUT.get("fibre_cut", {})))
check("W4-7 the Kaehler data at the truth are the probabilities of the nine facet cells, and every cell with a facet is a cell of positive probability",
      all(pq[i] > 0 for i in facet_cells))

# ---------------- Stage 4: the model's conic and the four-variable segment ----------------
lo4, hi4 = segment(mu, v); R4 = (r_of_t(lo4), r_of_t(hi4)); R3 = (-0.7626304870470775, 0.573117792381961)
walls4 = []
for t_ in (lo4, hi4):
    w_, b1, b2 = explanation(mu, v, t_); walls4.append({"w1": f6(w_), "a1": [f6(z) for z in b1], "a2": [f6(z) for z in b2]})
any_abc = lambda x: float(x[0] or x[1] or x[2])
pany = SD.U_poly(mu, v, lambda x: any_abc(x), range(4))
int4 = sorted([Pn.polyval(R4[0], pany), Pn.polyval(R4[1], pany)])
vS = {lab(S): np.prod(v[list(S)]) for S in TRIPLES}; vQ = np.prod(v)
quad_ok = abs(mS((0, 1, 2, 3)) - vQ * (r0 ** 2 + 1)) < 1e-12
OUT["model"] = {"R4": [f6(R4[0]), f6(R4[1])], "R3": [f6(R3[0]), f6(R3[1])], "walls4": walls4, "abc_harm_interval_4": [f6(int4[0]), f6(int4[1])],
                "abc_harm_interval_3": [0.660769, 0.66925], "conic": "m_S = v_S r (four triples), m_ABCD = v_ABCD (r^2 + 1)",
                "v_S": {k: f6(z) for k, z in vS.items()}, "v_ABCD": f6(vQ), "m_ABCD": f6(mS((0, 1, 2, 3)))}
check("W4-8 the model is a conic arc in the fibre: three linear equations r_S = r_S' and the quadric m_ABCD = v_ABCD (r^2 + 1) hold at the truth",
      quad_ok and all(abs(mS(S) - vS[lab(S)] * r0) < 1e-12 for S in TRIPLES))
tD = sorted([abs(v[3]) / (1 - mu[3]) if v[3] < 0 else abs(v[3]) / mu[3], mu[3] / abs(v[3]) if v[3] < 0 else (1 - mu[3]) / abs(v[3])])
OUT["model"]["D_walls_t"] = [f6(tD[0]), f6(tD[1])]
check("W4-9 the fourth harm's walls are slack here (t in [0.408, 1.633] contains [0.689, 1.327]): R4 = R3 and the ABC harm interval is unchanged",
      abs(R4[0] - R3[0]) < 1e-9 and abs(R4[1] - R3[1]) < 1e-9 and tD[0] < lo4 and hi4 < tD[1] and abs(int4[0] - 0.660769) < 1e-6,
      f"R4 = [{R4[0]:.4f}, {R4[1]:.4f}] vs R3 = [{R3[0]:.4f}, {R3[1]:.4f}]; ABC harm [{int4[0]:.5f}, {int4[1]:.5f}]")

# ---------------- Stage 5: the decision and the hidden channel with four variables ----------------
acts = [lambda x: 1 - float(x[0] or x[1] or x[2]), lambda x: 0.872 - float(x[0] or x[1])]
ps = [SD.U_poly(mu, v, u, range(4)) for u in acts]; pre = SD.set_rule(ps, R4, 0.30); orc = SD.oracle(ps, r0, 0.30)
rc = (ps[1][0] - ps[0][0]) / ps[0][1] if len(ps[0]) > 1 else None
lo16, hi16 = IJ.stealth_range(P, (0, 1, 2)); cap16 = hi16                 # chi_ABC on 16 cells; s > 0 pushes the apparent ABC harm up
shift_per_eps = 2 * cap16 / abs(vS["ABC"])                                # |r| shift per unit eps (2^(m-3) s / v_S)
eps_star = abs(rc - r0) / shift_per_eps; eps_wall = abs(R4[0] - r0) / shift_per_eps
# m_ABC range over the whole fibre (all laws with these pair tables) by linear programming
cABC = np.array([np.prod([x[i] - mu[i] for i in (0, 1, 2)]) for x in X])
bounds = [linprog(sgn * cABC, A_eq=Amarg, b_eq=Amarg @ p, bounds=[(0, None)] * 16, method="highs").fun * sgn for sgn in (1, -1)]
# second-triple test at n = 1e5: Monte Carlo of rhat_ABC - rhat_ABD
rng = np.random.default_rng(4); n = 100000; WIN = list(itertools.combinations(range(4), 2)) + [(0, 1, 2), (0, 1, 3)]
def rhat_pair(Pl, rng):
    tabs = {}
    for S in WIN:
        T = Pl.sum(axis=tuple(k for k in range(4) if k not in S)).ravel(); tabs[S] = (rng.multinomial(n, T / T.sum()) / n).reshape((2,) * len(S))
    muh = np.zeros(4); cnt = np.zeros(4); Mh = np.zeros((4, 4))
    for (i, j) in WIN[:6]:
        T = tabs[(i, j)]; pi_, pj_ = T[1].sum(), T[:, 1].sum(); muh[i] += pi_; muh[j] += pj_; cnt[i] += 1; cnt[j] += 1; Mh[i, j] = Mh[j, i] = T[1, 1] - pi_ * pj_
    muh /= cnt; vh = fit_v(muh, Mh); out = []
    for S in WIN[6:]:
        T = tabs[S]; m3 = [T.sum(axis=tuple(k for k in range(3) if k != q))[1] for q in range(3)]
        mm = sum(T[x] * np.prod([x[q] - m3[q] for q in range(3)]) for x in itertools.product((0, 1), repeat=3)); out.append(mm / np.prod(vh[list(S)]))
    return out
diffs0 = np.array([np.subtract(*rhat_pair(P, rng)) for _ in range(1500)]); sd_diff = float(np.std(diffs0))
eps_test = 1.96 * sd_diff / shift_per_eps                                  # the injection's shift equals the test's critical value
# the model-invisible injection: records chosen so that the observed law is exactly the two-type law at r (Corollary 3(b)); exact budget
law_at = lambda r_: law(*explanation(mu, v, t_of_r(r_)))
from pce.lab import t_of_r
eps_model = lambda r_: float(1 - np.min(IJ.as_vec(law_at(r_)) / p))
eps_star_model = eps_model(rc); eps_wall_model = eps_model(R4[0])
OUT["decision4_model_channel"] = {"eps_star": f6(eps_star_model), "eps_wall": f6(eps_wall_model),
                                  "eps_star_3var_check": f6(1 - np.min(IJ.as_vec(law(*explanation(*(lambda m3: (m3[0], SD.v_closed(m3[1])))(moments(abc3)), t_of_r(rc)))) / IJ.as_vec(abc3)))}
OUT["decision4"] = {"pre_probe_act": ["T", "P"][pre], "oracle_act": ["T", "P"][orc], "r_c": f6(rc), "capacity16_plus": f6(cap16), "capacity8_plus": 0.027, "capacity16_in_face_units": f6(2 * cap16),
                    "shift_r_per_eps": f6(shift_per_eps), "eps_star": f6(eps_star), "eps_wall": f6(eps_wall),
                    "mABC_range_fibre": [f6(bounds[0]), f6(bounds[1])], "mABC_truth": f6(mS((0, 1, 2))), "sd_rdiff_n1e5": f6(sd_diff), "eps_second_triple_n1e5": f6(eps_test)}
check("W4-10 with four variables the decision is unchanged (pair before the probe, triple at the truth), and the ABC channel's capacity on 16 cells is smaller than on 8",
      pre == 1 and orc == 0 and cap16 < 0.027, f"capacity {cap16:.4f} (vs 0.027); eps* {eps_star:.4f}; wall at {eps_wall:.4f}")
check("W4-11 a second triple sees the ABC channel: at n = 1e5 the difference test reaches its critical value before the decision flips",
      eps_test < eps_star, f"sd of rhat_ABC - rhat_ABD {sd_diff:.4f}; test at eps {eps_test:.4f} vs flip at {eps_star:.4f}")

check("W4-13 the model-invisible injection (observed law = the two-type law at r) has exact budget 1 - min_x law_r(x)/P(x); on three variables it reproduces 0.037",
      abs(OUT["decision4_model_channel"]["eps_star_3var_check"] - 0.037037) < 1e-5 and eps_star_model < eps_wall_model,
      f"four variables: flip at {eps_star_model:.4f}, wall at {eps_wall_model:.4f}")

# ---------------- Stage 6: chambers of the fourth harm ----------------
cham = {}
for d2 in (0.5, 0.6, 0.7):
    pv = IJ.as_vec(law(0.6, np.array([.5, .1, .4, .2]), np.array([.05, .5, .1, d2]))); vv = enumerate_vertices(pv)
    tt = [t for _, t in vv]; si = all(len(t) == 5 for t in tt); dd = sorted({abs(int(round(np.linalg.det(B[list(t[:5]), :])))) for t in tt})
    cham[str(d2)] = {"vertices": len(vv), "simple": bool(si), "vertex_dets": dd, "facets": len({i for t in tt for i in t})}
for d1, d2 in ((0.3, 0.7),):
    pv = IJ.as_vec(law(0.6, np.array([.5, .1, .4, d1]), np.array([.05, .5, .1, d2]))); vv = enumerate_vertices(pv); tt = [t for _, t in vv]
    cham[f"{d1},{d2}"] = {"vertices": len(vv), "simple": bool(all(len(t) == 5 for t in tt)), "vertex_dets": sorted({abs(int(round(np.linalg.det(B[list(t[:5]), :])))) for t in tt}),
                         "facets": len({i for t in tt for i in t})}
OUT["chambers"] = cham
check("W4-12 chambers: a2D = 0.5 and 0.7 give smooth fibres; a2D = 0.6 lies on a wall (not simple); a1D = 0.3 gives a simple orbifold (vertex index 2)",
      cham["0.5"]["simple"] and cham["0.5"]["vertex_dets"] == [1] and cham["0.7"]["vertex_dets"] == [1] and not cham["0.6"]["simple"]
      and cham["0.3,0.7"]["simple"] and 2 in cham["0.3,0.7"]["vertex_dets"], json.dumps(cham))

rng2 = np.random.default_rng(1); rand = []
for k in range(50):
    if k < 30:
        w_ = rng2.uniform(.2, .8); pv = IJ.as_vec(law(w_, rng2.uniform(.05, .95, 4), rng2.uniform(.05, .95, 4)))
    else:
        pv = rng2.dirichlet(np.ones(16))
    vv = enumerate_vertices(pv); tt = [t for _, t in vv]; si = all(len(t) == 5 for t in tt)
    sm = si and all(abs(int(round(np.linalg.det(B[list(t), :])))) == 1 for t in tt); rand.append((si, sm))
OUT["random_fibres"] = {"simple": int(sum(a_ for a_, _ in rand)), "smooth": int(sum(b_ for _, b_ in rand)), "n": 50}
check("W4-14 on 50 random laws and two-type truths the four-variable pair fibre is always simple and never smooth (an orbifold)",
      OUT["random_fibres"]["simple"] == 50 and OUT["random_fibres"]["smooth"] == 0)
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "worked_example4.json"), "w"), indent=1, default=float)
print(f"{sum(results)}/{len(results)} passed")
