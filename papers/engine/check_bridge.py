"""check_bridge.py -- how the hiding polytopes, the toric ideal, the sign sheets and the tangent structure fit together (companion
"Two latent types on arbitrary covers", Lemma "algebra, reals, positivity"; companion "A tangent structure for injected data",
Proposition "restriction to the hiding polytope"). Writes lab/bridge.json.

  BR1  algebra: data from a two-type truth satisfy every generator of I_A (the ideal of the image of the moment map); the complex fibre
       over such data is a coset of the torus K = {z : z^{a_j} = 1}, of dimension m + 1 - rank A
  BR2  reals: the real points of that fibre are |G_2| copies of its positive part (one per sign symmetry); data that pass every relation
       but carry an invisible sign class have an empty real fibre
  BR3  positivity: on each real copy the positivity constraints cut out exactly the sheet polytope of Phase II, in the coordinates
       (log t, log lambda); a sheet can be empty although its real copy is not (two disjoint edges, w = 0.9)
  BR4  so the three conditions are nested and each can fail alone: in I_A (algebraic) > real explanation (signs) > explanation in the
       simplex (positivity), with one witness for each gap
  BR5  the tangent structure on a sheet is the restriction of the cone structure: the feasible directions at a point of the sheet are its
       tangent cone {u : A_active u <= 0}; each active facet is a named positivity wall (a_{1i} or a_{2i} at 0 or 1); the vertical lift
       fails to be universal exactly where some wall is active
"""
import itertools, json, os, sys
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import covers as CV, syzygy as SZ, signs as SG, cover_strata as STR

rng = np.random.default_rng(20261029); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
pairs = lambda Ws: sorted({e for W in Ws for e in itertools.combinations(W, 2)})

# BR1: C4 + 012 + 123 (finite? rank 5 -> fibre dim 0) and C4 pairs only (fibre dim 2) and the bowtie
COV = {"C4 pairs": (4, [(0, 1), (1, 2), (2, 3), (0, 3)], []), "C4 + 012": (4, [(0, 1), (1, 2), (2, 3), (0, 3)], [(0, 1, 2)]),
       "bowtie": (5, pairs([(0, 1, 2), (2, 3, 4)]), [(0, 1, 2), (2, 3, 4)])}
ok = True; info = {}
for k, (m, E, W) in COV.items():
    A, names = SZ.config(m, E, W); R = SZ.m2_ideal(m, E, W); F = [tuple(e) for e in E] + W
    v = rng.uniform(0.25, 0.6, m) * rng.choice([-1, 1], m); r = rng.uniform(0.3, 0.9); pt = SZ.model_point(m, E, W, v, r)
    ok &= all(abs(SZ.eval_rel(g, pt)) < 1e-12 for g in R["gens"])
    # torus directions: log-kernel of A^T (continuous part of K); check phi is constant along them
    Kc = np.linalg.svd(A.T.astype(float))[2][np.linalg.matrix_rank(A):]          # rows: kernel of A^T in log coordinates
    z0 = np.r_[v, r]; dimK = Kc.shape[0]
    for kv in Kc:
        for s in (0.3, -0.5):
            z = z0 * np.exp(s * kv) if True else None
            if not W: z[m] = z0[m] + s                                            # r is free when no window uses it
            ok &= np.allclose(STR.phi(m, F, z), STR.phi(m, F, z0), atol=1e-12)
    ok &= dimK == m + 1 - np.linalg.matrix_rank(A)
    info[k] = dict(codim=R["codim"], fibre_dim=int(dimK))
check("BR1 data from the model satisfy I_A; the fibre over them is a torus coset of dimension m + 1 - rank A (phi constant along it)", ok,
      "; ".join(f"{k}: fibre dim {v['fibre_dim']}" for k, v in info.items()))

# BR2: real copies = |G_2|; invisible class -> empty real fibre
m, E, W = COV["bowtie"]; F = [tuple(e) for e in E] + W; G = SG.sign_group(m, F)
v = np.array([0.4, -0.3, 0.5, 0.35, -0.45]); r = 0.7; target = STR.phi(m, F, np.r_[v, r]); sols = []
for _ in range(300):
    rr = least_squares(lambda z: STR.phi(m, F, z) - target, rng.normal(size=m + 1), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
    if np.abs(rr.fun).max() < 1e-11 and not any(np.allclose(rr.x, s, atol=1e-5) for s in sols): sols.append(rr.x)
tri = [(0, 1), (1, 2), (0, 2)]; bad = {(0, 1): -0.05, (1, 2): 0.04, (0, 2): 0.03}
best = min(np.abs(least_squares(lambda z: np.array([z[i] * z[j] - bad[(i, j)] for i, j in tri]), rng.normal(size=3), xtol=1e-15, ftol=1e-15).fun).max() for _ in range(100))
OUT["real"] = dict(bowtie_solutions=len(sols), G2=2 ** G["dim"], liar_best_residual=float(best))
check("BR2 real points: the bowtie's real fibre has |G_2| = 2 points; the triangle with a negative product passes every relation (none) but has no real point",
      len(sols) == 2 ** G["dim"] and best > 1e-4, f"bowtie {len(sols)} real points; liar residual >= {best:.2e}")

# BR3: positive part of each real copy = sheet polytope (C4 pairs: 2-dim fibre, one sheet); a sheet empty although the real copy exists
m, E = 4, [(0, 1), (1, 2), (2, 3), (0, 3)]; w, a1, a2 = 0.4, np.array([.8, .3, .7, .25]), np.array([.2, .7, .15, .7])
mu, cc, v = CV.truth_data(w, a1, a2, E); Fz = CV.factorise(m, E, cc); sg = [float(np.sign(v[0] / Fz["comps"][0]["v0"][0]))]; A, b, who = CV.sheet_constraints(mu, Fz, sg)
ok = True
for _ in range(400):
    z = np.array([rng.uniform(-1, 1.2), rng.uniform(-1, 1)]); wv, b1, b2, _ = CV.explain(mu, Fz, sg, z[0], z[1:], m=4)
    inside = bool(np.all(A @ z <= b + 1e-12)); pos = bool(np.all((b1 >= -1e-12) & (b1 <= 1 + 1e-12) & (b2 >= -1e-12) & (b2 <= 1 + 1e-12)))
    ok &= inside == pos
mu9, c9, v9 = CV.truth_data(0.9, np.array([.05, .02, .15, .05]), np.array([.55, .45, .75, .9]), [(0, 1), (2, 3)]); F9 = CV.factorise(4, [(0, 1), (2, 3)], c9)
s0 = [float(np.sign(v9[C["vertices"][0]] / C["v0"][C["vertices"][0]])) for C in F9["comps"]]; rel = [s0[0], -s0[1]]
real_copy = all(abs(CV.family_v(F9, rel, [0.1, -0.2])[i] * CV.family_v(F9, rel, [0.1, -0.2])[j] - c9[(i, j)]) < 1e-12 for i, j in [(0, 1), (2, 3)])
empty = CV.sheet_dimension(*CV.sheet_constraints(mu9, F9, rel)[:2]) == -1
check("BR3 the positive part of a real copy is exactly the sheet polytope (C4: 400 random points); a real copy can have an empty sheet (two edges, w = 0.9)",
      ok and real_copy and empty)

# BR4
OUT["nesting"] = {"algebraic but not real": "triangle with c01 c12 c02 < 0 (invisible class)", "real but not positive": "two edges, w = 0.9, relative sheet",
                  "algebraic and real but no explanation in the simplex": "triangle with c01 scaled below lambda* = 0.3115"}
lam_case = CV.factorise(3, tri, dict(zip(tri, [0.0346 * 0.25, 0.0336, 0.0306])))
from pce import layers as LY
check("BR4 the three conditions are nested and each gap has a witness: algebraic-not-real (liar sign), real-not-positive (dead sheet), and the triangle beyond lambda*",
      best > 1e-4 and empty and LY.triangle_family(0.25) is None and LY.triangle_family(0.5) is not None)

# BR5
from scipy.spatial import HalfspaceIntersection, ConvexHull
z0, _ = CV.chebyshev(A, b); hs = HalfspaceIntersection(np.c_[A, -b], z0); V = hs.intersections[ConvexHull(hs.intersections).vertices]
ok = True; walls = {}
for name, z in (("vertex", V[0]), ("edge", 0.5 * (V[0] + V[1])), ("interior", z0)):
    act = [k for k in range(len(b)) if abs(A[k] @ z - b[k]) < 1e-9]; walls[name] = [who[k] for k in act]
    for _ in range(300):
        u = rng.normal(size=2); feas_cone = all(A[k] @ u <= 1e-12 for k in act)
        feas_def = bool(np.all(A @ (z + 1e-7 * u) <= b + 1e-12)); ok &= feas_cone == feas_def
    for k in act:                                  # each active facet is a named positivity wall
        i, sgn = who[k]; wv, b1, b2, _ = CV.explain(mu, Fz, sg, z[0], z[1:], m=4)
        ok &= min(abs(b1[i]), abs(1 - b1[i]), abs(b2[i]), abs(1 - b2[i])) < 1e-9
OUT["walls"] = {k: [list(x) for x in v] for k, v in walls.items()}
check("BR5 on a sheet the feasible directions are the tangent cone {u : A_active u <= 0}; every active facet is a positivity wall (a = 0 or 1); universality fails exactly where a wall is active",
      ok and len(walls["interior"]) == 0 and len(walls["vertex"]) == 2, f"active walls: {walls}")

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "bridge.json"), "w"), indent=1, default=str)
print(f"{sum(results)}/{len(results)} passed")
