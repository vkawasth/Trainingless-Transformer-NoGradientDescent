"""check_birth.py -- where contextuality is born in the representation scheme (note: level2/partial_gluing.pdf, Section 15).

CHSH scenario = the 4-cycle C4 (A side {a, c}, B side {b, d}); W = D_inf x D_inf. Hermitian models split into tensor products of irreducibles of
the two factors (characters, and 2-dimensional irreducibles with reflection angle theta). A model can violate CHSH only if it has a 2 (x) 2
block; as theta -> 0 or pi such a block degenerates to four characters {alpha, alpha'} x {beta, beta'} with alpha alpha' = (-,-) on A and
beta beta' = (-,-) on B. A classical point (direct sum of characters) containing such a block is a birth point.

  BR1 Rep_3: no birth point (a 2 (x) 2 block needs dimension 4), yet 16 singular points; random nearby models never violate CHSH
  BR2 Rep_4: every birth point is singular (all four birth types); most singular points are not birth points
  BR3 nearby models (Gauss-Newton projection onto the Hermitian locus of Rep_4 from random perturbations): CHSH violations occur near birth
      points, and never near non-birth singular points or smooth points
  BR4 along the birth curve (both reflection angles = t) the largest CHSH value is exactly 2 sqrt(1 + sin^2 t) = 2 + t^2 + O(t^4):
      contextuality switches on at second order, the order of the obstruction
  BR5 Rep_5: every classical point containing a birth block is still singular
Writes ../level2/birth_results.json.  Run: python check_birth.py
"""
import itertools, json, os, sys, warnings
import numpy as np
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce.coxeter import analyse, diagonal_points

rng = np.random.default_rng(20261025); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
G = (list("abcd"), [(0, 1), (1, 2), (2, 3), (3, 0)])
A = lambda u: (u[0], u[2]); B = lambda u: (u[1], u[3])
def birth(key):
    ch = [tuple(c) for c in key]
    for quad in itertools.combinations(range(len(ch)), 4):
        S = [ch[m] for m in quad]
        if len(set(S)) < 4: continue
        for p in itertools.permutations(S):
            if A(p[0]) == A(p[2]) and A(p[1]) == A(p[3]) and B(p[0]) == B(p[1]) and B(p[2]) == B(p[3]) and \
               tuple(np.multiply(A(p[0]), A(p[1]))) == (-1, -1) and tuple(np.multiply(B(p[0]), B(p[2]))) == (-1, -1):
                return True
    return False
SIGNS = [s for s in itertools.product((1, -1), repeat=4) if s.count(-1) % 2 == 1]
def chsh(X):
    a, b, c, d = X; best = -np.inf
    for s in SIGNS:
        Bop = s[0] * a @ b + s[1] * b @ c + s[2] * c @ d + s[3] * d @ a; best = max(best, np.linalg.eigvalsh((Bop + Bop.T) / 2).max())
    return best
def nearby(X0, eps, m):
    n = X0[0].shape[0]; iu = np.triu_indices(n); L = len(iu[0])
    def unpack(z):
        out = []
        for k in range(4):
            M = np.zeros((n, n)); M[iu] = z[k * L:(k + 1) * L]; out.append(M + M.T - np.diag(np.diag(M)))
        return out
    def F(z):
        X = unpack(z); r = [X[k] @ X[k] - np.eye(n) for k in range(4)] + [X[u] @ X[v] - X[v] @ X[u] for u, v in G[1]]
        return np.concatenate([q.ravel() for q in r])
    z0 = np.concatenate([X[iu] for X in X0]); vals = []
    for _ in range(m):
        r = least_squares(F, z0 + eps * rng.normal(size=z0.size), xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if np.linalg.norm(F(r.x)) < 1e-9: vals.append(chsh(unpack(r.x)) - 2)
    return vals
def classify(n):
    rows = []
    for X, key in diagonal_points(G, n):
        rows.append((X, key, birth(key), analyse(G, X, 3, rng)["obstructed"]))
    return rows

# BR1
R3 = classify(3); nb3 = sum(r[2] for r in R3); ob3 = sum(r[3] for r in R3)
v3 = [v for X, k, b, o in [r for r in R3 if r[3]] for v in nearby(X, 0.08, 10)]
check("BR1 Rep_3: no birth points; singular points exist; nearby models never violate CHSH", nb3 == 0 and ob3 > 0 and max(v3) < 1e-9,
      f"{ob3} singular points, {len(v3)} nearby models, max CHSH - 2 = {max(v3):.1e}")

# BR2
R4 = classify(4); births = [r for r in R4 if r[2]]; sing = [r for r in R4 if r[3]]
types = {tuple(sorted(r[1])) for r in births}
check("BR2 Rep_4: every birth point is singular; most singular points are not birth points", len(births) == 4 and all(r[3] for r in births)
      and len(sing) > 10 * len(births), f"{len(R4)} classical points; {len(births)} birth points (all singular); {len(sing)} singular points")
OUT["rep4"] = {"points": len(R4), "birth": len(births), "singular": len(sing)}

# BR3
nonb = [r for r in sing if not r[2]]; smooth = [r for r in R4 if not r[3]]
pick = lambda L, k: [L[i] for i in rng.choice(len(L), min(k, len(L)), replace=False)]
vb = [v for r in births for v in nearby(r[0], 0.08, 25)]
vn = [v for r in pick(nonb, 30) for v in nearby(r[0], 0.08, 25)]
vs = [v for r in pick(smooth, 30) for v in nearby(r[0], 0.08, 25)]
share = lambda v: float(np.mean(np.array(v) > 1e-6))
OUT["nearby"] = {"birth": {"n": len(vb), "share": share(vb), "max": max(vb)}, "singular_not_birth": {"n": len(vn), "share": share(vn), "max": max(vn)},
                 "smooth": {"n": len(vs), "share": share(vs), "max": max(vs)}}
check("BR3 CHSH violations appear near birth points only", share(vb) > 0.1 and max(vn) < 1e-9 and max(vs) < 1e-9,
      f"birth: {share(vb):.2f} of {len(vb)} nearby models contextual; non-birth singular: {share(vn):.2f} of {len(vn)}; smooth: {share(vs):.2f} of {len(vs)}")

# BR4
Z = np.diag([1., -1]); Xp = np.array([[0, 1.], [1, 0]]); I2 = np.eye(2); ts = np.linspace(0, 0.6, 13); dev = 0.0; curve = []
for t in ts:
    Rt = np.cos(t) * Z + np.sin(t) * Xp; Xs = [np.kron(Z, I2), np.kron(I2, Z), np.kron(Rt, I2), np.kron(I2, Rt)]
    v = chsh(Xs); curve.append([float(t), float(v)]); dev = max(dev, abs(v - 2 * np.sqrt(1 + np.sin(t) ** 2)))
start = [np.kron(Z, I2), np.kron(I2, Z), np.kron(Z, I2), np.kron(I2, Z)]
key0 = tuple(sorted(tuple(int(np.diag(X)[j]) for X in start) for j in range(4)))
OUT["curve"] = curve
check("BR4 birth curve: CHSH = 2 sqrt(1 + sin^2 t) = 2 + t^2 + O(t^4), from a birth point", dev < 1e-12 and birth(key0), f"largest deviation {dev:.1e}")

# BR5
R5 = [(key, b, o) for X, key in diagonal_points(G, 5) for b in [birth(key)] if b for o in [analyse(G, X, 3, rng)["obstructed"]]]
check("BR5 Rep_5: every classical point containing a birth block is singular", len(R5) > 0 and all(o for _, _, o in R5), f"{len(R5)} such points")
OUT["rep5_birth_points"] = len(R5)

json.dump(OUT, open(os.path.join(HERE, "..", "level2", "birth_results.json"), "w"), default=float)
print(f"\n{sum(results)}/{len(results)} checks passed")
