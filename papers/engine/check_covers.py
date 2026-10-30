"""check_covers.py -- two latent types on arbitrary covers, Phases I-II (companion paper "Two latent types on arbitrary covers").
Writes lab/covers.json.

  CV1  rank B_G = |V| - b(G) on 300 random graphs (b = bipartite components, isolated vertices included)
  CV2  existence: a real factorisation c_ij = v_i v_j exists iff the sign cocycle is a coboundary and log|c| lies in im B (on each
       component of the nonzero graph); compared with a direct nonlinear solve on the zoo with corrupted data
  CV3  the factorisation set: rigid (+-v) on non-bipartite components, one scale per bipartite component; dimension of the solution
       variety = number of bipartite components of G*; every member reproduces c exactly
  CV4  zero strata: a coincidence v_2 = 0 on the triangle turns the nonzero graph into one edge and creates a scale freedom; a zero
       moment between two vertices with nonzero moments is infeasible
  CV5  the hiding set of each sign sheet is the polytope A[y, x] <= b with slopes +-1: every sampled point gives a two-type law with
       exactly the observed singles and pairs and a1, a2 in [0, 1]; points beyond a facet leave [0, 1]
  CV6  for a connected bipartite pair graph the hiding set is a rectangle in u = log(lambda t), w = log(lambda / t)
  CV7  generic dimension b(G) + 1 on the zoo (C3, C4, C4 + diagonal, C5, P4, star, two edges, triangle + edge, triangle + isolated,
       K4, K5); sheets {+-1}^c / (label swap), some killed by positivity
  CV9  positivity can kill a relative-sign sheet (two disjoint edges, w = 0.9)
  CV8  the lab recovered: on K3 the y-interval is log of the lab's segment [t_lo, t_hi]
"""
import itertools, json, os, sys
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import covers as CV
from pce.lab import segment

rng = np.random.default_rng(20261029); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

ZOO = {"C3": (3, [(0, 1), (1, 2), (0, 2)]), "C4": (4, [(0, 1), (1, 2), (2, 3), (0, 3)]), "C4+diag": (4, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]),
       "C5": (5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]), "P4": (4, [(0, 1), (1, 2), (2, 3)]), "star": (4, [(0, 1), (0, 2), (0, 3)]),
       "two edges": (4, [(0, 1), (2, 3)]), "triangle+edge": (5, [(0, 1), (1, 2), (0, 2), (3, 4)]), "triangle+isolated": (4, [(0, 1), (1, 2), (0, 2)]),
       "K4": (4, list(itertools.combinations(range(4), 2))), "K5": (5, list(itertools.combinations(range(5), 2)))}

def truth(m):
    w = rng.uniform(0.25, 0.75)
    while True:
        a1 = rng.uniform(0.08, 0.92, m); a2 = rng.uniform(0.08, 0.92, m)
        if np.min(np.abs(a1 - a2)) > 0.12: return w, a1, a2

# CV1
ok = True
for _ in range(300):
    m = int(rng.integers(1, 9)); E = [e for e in itertools.combinations(range(m), 2) if rng.random() < rng.uniform(0.1, 0.6)]
    ok &= np.linalg.matrix_rank(CV.incidence(m, E)) == m - CV.b_count(m, E) if E else CV.b_count(m, E) == m
check("CV1 rank B_G = |V| - b(G) on 300 random graphs", ok)

# CV2
def nl_exists(m, E, c):
    best = np.inf
    for _ in range(40):
        f = lambda v: np.array([v[i] * v[j] - c[(i, j)] for i, j in E])
        r = least_squares(f, rng.normal(size=m) * 0.3, xtol=1e-15, ftol=1e-15, gtol=1e-15); best = min(best, np.abs(r.fun).max())
    return best < 1e-9
agree = 0; tot = 0; kinds = {"sign": 0, "modulus": 0, "ok": 0}
for name, (m, E) in ZOO.items():
    for trial in range(6):
        w, a1, a2 = truth(m); mu, c, v = CV.truth_data(w, a1, a2, E); c = dict(c); e = E[int(rng.integers(len(E)))]
        if trial % 3 == 1: c[e] = -c[e]
        elif trial % 3 == 2: c[e] = c[e] * 1.7
        F = CV.factorise(m, E, c); ex = nl_exists(m, E, c); agree += (F["ok"] == ex); tot += 1
        kinds["ok" if F["ok"] else ("sign" if "sign" in F["reason"] else "modulus")] += 1
check("CV2 existence = (sign class trivial) and (modulus class zero), against a direct nonlinear solve on the zoo with corrupted data",
      agree == tot, f"{agree}/{tot}; verdicts {kinds}")

# CV3
ok = True; dims = {}
for name, (m, E) in ZOO.items():
    w, a1, a2 = truth(m); mu, c, v = CV.truth_data(w, a1, a2, E); F = CV.factorise(m, E, c)
    for _ in range(5):
        sig = list(rng.choice([-1.0, 1.0], size=len(F["comps"]))); x = rng.normal(size=F["n_scale"]); vv = CV.family_v(F, sig, x)
        ok &= all(abs(vv[i] * vv[j] - c[(i, j)]) < 1e-12 * max(1, abs(c[(i, j)])) for i, j in E)
    N = F["N"]; Jm = np.array([[(v[j] if k == i else v[i] if k == j else 0.0) for k in N] for i, j in E])
    dims[name] = (len(N) - int(np.linalg.matrix_rank(Jm)), F["n_scale"])
    ok &= dims[name][0] == dims[name][1]
OUT["factor_dims"] = dims
check("CV3 factorisation set: rigid on non-bipartite components, one scale per bipartite component; tangent dimension = number of bipartite components of G*",
      ok, "; ".join(f"{k} {v[0]}" for k, v in dims.items()))

# CV4
m, E = ZOO["C3"]; w = 0.4; a1 = np.array([.8, .3, .6]); a2 = np.array([.2, .7, .6]); mu, c, v = CV.truth_data(w, a1, a2, E); F = CV.factorise(m, E, c)
_, cg, _ = CV.truth_data(0.4, np.array([.8, .3, .6]), np.array([.2, .7, .1]), E); cz = dict(cg); cz[(0, 1)] = 0.0; Fz = CV.factorise(m, E, cz)
check("CV4 zero strata: on the triangle a coincidence v_2 = 0 leaves G* = {01}, bipartite, so a scale appears (n_scale 0 -> 1); c_01 = 0 with c_02, c_12 != 0 is infeasible",
      F["ok"] and F["zero"] == [2] and F["n_scale"] == 1 and not Fz["ok"] and "zero moment" in Fz["reason"], f"zero {F['zero']}; reason '{Fz['reason']}'")

# CV5, CV6, CV7
def sample_sheet(A, b, n, z0):
    """hit-and-run from an interior point"""
    pts = []; z = z0.copy()
    for _ in range(n):
        d = rng.normal(size=len(z)); d /= np.linalg.norm(d); Ad = A @ d; sl = (b - A @ z)
        hi = np.min(sl[Ad > 1e-12] / Ad[Ad > 1e-12]) if np.any(Ad > 1e-12) else 1.0; lo = np.max(sl[Ad < -1e-12] / Ad[Ad < -1e-12]) if np.any(Ad < -1e-12) else -1.0
        z = z + rng.uniform(lo, hi) * d; pts.append(z.copy())
    return pts
ok5 = True; ok_out = True; ok6 = True; table = {}
for name, (m, E) in ZOO.items():
    w, a1, a2 = truth(m); mu, c, v = CV.truth_data(w, a1, a2, E); F = CV.factorise(m, E, c); P0 = CV.law_m(w, a1, a2)
    nc = len(F["comps"]); sheets = []
    for sig in itertools.product([1.0, -1.0], repeat=nc):
        if sig[0] < 0: continue                                          # label swap: (sigma, y) ~ (-sigma, -y)
        A, b, who = CV.sheet_constraints(mu, F, list(sig)); z0, rad = CV.chebyshev(A, b); dim = CV.sheet_dimension(A, b)
        sheets.append((sig, dim, rad))
        if dim < 0: continue
        for z in sample_sheet(A, b, 60, z0):
            ww, b1, b2, vv = CV.explain(mu, F, list(sig), z[0], z[1:], m=m); P = CV.law_m(ww, b1, b2)
            ok5 &= np.all(b1 >= -1e-12) and np.all(b1 <= 1 + 1e-12) and np.all(b2 >= -1e-12) and np.all(b2 <= 1 + 1e-12)
            ok5 &= all(np.allclose(CV.marg(P, e), CV.marg(P0, e), atol=1e-12) for e in E) and all(np.allclose(CV.marg(P, (i,)), CV.marg(P0, (i,)), atol=1e-12) for i in range(m))
        # beyond each facet: push the Chebyshev centre across row r
        for r in range(A.shape[0]):
            z = z0 + (b[r] - A[r] @ z0 + 1e-3) * A[r] / (A[r] @ A[r]); ww, b1, b2, _ = CV.explain(mu, F, list(sig), z[0], z[1:], m=m)
            i = who[r][0]; ok_out &= not (0 <= b1[i] <= 1 and 0 <= b2[i] <= 1)
        if len(F["comps"]) == 1 and F["n_scale"] == 1:                    # rectangle in (u, w)
            Muw = np.array([[0.5, 0.5], [0.5, -0.5]])                     # [y, x] = Muw^{-1}...: u = x + y, w = x - y -> y = (u - w)/2, x = (u + w)/2
            Ay = A @ np.array([[0.5, -0.5], [0.5, 0.5]])                  # rows in (u, w)
            ok6 &= all(min(abs(r_[0]), abs(r_[1])) < 1e-12 for r_ in Ay)
    nfree = len(F["free"]); gdim = max(d for _, d, _ in sheets) + nfree
    table[name] = dict(b=CV.b_count(m, E), dim=int(gdim), sheets=len(sheets), feasible=sum(1 for _, d, _ in sheets if d >= 0),
                       full=sum(1 for _, d, _ in sheets if d == F["n_scale"] + 1))
OUT["zoo"] = table
check("CV5 every sampled point of every sheet polytope is a two-type law with the observed singles and pairs; beyond each facet a1 or a2 leaves [0, 1]",
      ok5 and ok_out)
check("CV6 connected bipartite pair graphs (C4, P4, star): the hiding set is a rectangle in u = log(lambda t), w = log(lambda/t)", ok6)
check("CV7 generic dimension b(G) + 1 on the zoo; sheets {+-1}^c / swap, some killed by positivity",
      all(v["dim"] == v["b"] + 1 for v in table.values()), "; ".join(f"{k}: b {v['b']}, dim {v['dim']}, sheets {v['feasible']}/{v['sheets']}" for k, v in table.items()))

# CV8
w, a1, a2 = 0.6, np.array([.5, .1, .4]), np.array([.05, .5, .1]); m, E = ZOO["C3"]; mu, c, v = CV.truth_data(w, a1, a2, E); F = CV.factorise(m, E, c)
sg = [float(np.sign(v[F["comps"][0]["vertices"][0]] / F["comps"][0]["v0"][F["comps"][0]["vertices"][0]]))]
A, b, _ = CV.sheet_constraints(mu, F, sg); ylo = -min(b[A[:, 0] < 0]); yhi = min(b[A[:, 0] > 0]); lo, hi = segment(mu, v)
OUT["K3"] = {"y": [ylo, yhi], "lab_t": [lo, hi]}
check("CV8 on K3 the sheet is the interval y in [log t_lo, log t_hi] of the lab's segment", abs(np.exp(ylo) - lo) < 1e-12 and abs(np.exp(yhi) - hi) < 1e-12,
      f"t in [{np.exp(ylo):.4f}, {np.exp(yhi):.4f}] vs lab [{lo:.4f}, {hi:.4f}]")

# CV9
E2 = [(0, 1), (2, 3)]; mu9, c9, v9 = CV.truth_data(0.9, np.array([.05, .02, .15, .05]), np.array([.55, .45, .75, .9]), E2); F9 = CV.factorise(4, E2, c9)
d9 = [CV.sheet_dimension(*CV.sheet_constraints(mu9, F9, list(s))[:2]) for s in [(1.0, 1.0), (1.0, -1.0)]]
OUT["killed_sheet"] = d9
check("CV9 positivity can kill a sheet: on two disjoint edges with w = 0.9 the relative-sign sheet is empty (dimensions 3 and -1)", d9 == [3, -1], f"{d9}")

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "covers.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(f"{sum(results)}/{len(results)} passed")
