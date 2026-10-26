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
  The 5-cycle (KCBS) has no block classification; contextuality of a model is defined instead by the complete n-cycle inequalities
  (Araujo et al.): a model is contextual for some state iff max over odd sign patterns of the top eigenvalue of sum_i g_i X_i X_{i+1}
  exceeds n - 2.
  BR6 the witness over Hermitian Rep_3(C5) reaches the quantum 5-cycle bound (KCBS in a qutrit)
  BR7 classical points of Rep_3(C5), up to the dihedral symmetry, sign flips and reordering (33 classes): exactly one class has contextual
      models arbitrarily close (local maximum of the witness ~ 0.7 eps^2 down to eps = 0.01), and that class is a SMOOTH point of the full
      scheme (tangent dimension = dimension at nearby points; tangent vectors extend to third order)
  BR8 the singular class of Rep_3(C5) (tangent dimension above the nearby dimension) has no contextual models nearby: in the 5-cycle,
      birth and singularity are disjoint in Rep_3, so "born only at singular points" is specific to CHSH
  What degenerates (scan_c5.py writes ../level2/c5_scan_n3.json and c5_scan_n4.json; block sizes of nearby Hermitian models from the
  eigenspaces of a random element of the commutant):
  BR9  Rep_3(C5): every model near the birth class is irreducible (one 3-dimensional block) and contextual; every model near the
       singular class splits as 2 + 1 and none is contextual
  BR10 a degenerating irreducible block is necessary but not sufficient: in Rep_3 six classes have irreducible models nearby and only one
       gives birth; in Rep_4 every birth class has a nearby block of size >= 3, but so do 99 classes of which 29 give birth; and
       irreducible models of W_C5 are often noncontextual globally (random sample in Rep_3)
  BR11 Rep_4(C5): birth and singularity overlap only partly (neither contains the other)
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

# ---------- BR6-BR8: the 5-cycle ----------
from scipy.optimize import minimize
from pce.coxeter import linearisation, _null, quadratic
k = 5; G5 = ([str(i) for i in range(k)], [(i, (i + 1) % k) for i in range(k)])
GAM = [g_ for g_ in itertools.product((1, -1), repeat=k) if g_.count(-1) % 2 == 1]
def witness(X):
    return max(np.linalg.eigvalsh((lambda B: (B + B.T) / 2)(sum(g_[i] * X[i] @ X[(i + 1) % k] for i in range(k)))).max() for g_ in GAM) - (k - 2)
n5 = 3; iu5 = np.triu_indices(n5); L5 = len(iu5[0])
def unpack5(z):
    out = []
    for v in range(k):
        M = np.zeros((n5, n5)); M[iu5] = z[v * L5:(v + 1) * L5]; out.append(M + M.T - np.diag(np.diag(M)))
    return out
pack5 = lambda X: np.concatenate([x[iu5] for x in X])
def F5(z):
    X = unpack5(z); r = [X[v] @ X[v] - np.eye(n5) for v in range(k)] + [X[u] @ X[w] - X[w] @ X[u] for u, w in G5[1]]
    return np.concatenate([q.ravel() for q in r])
best = -9
for s_ in range(40):
    z = least_squares(F5, rng.normal(size=k * L5)).x
    for it in range(3):
        res = minimize(lambda zz: -witness(unpack5(zz)) + 50 * np.sum(F5(zz) ** 2), z, method="BFGS", options={"maxiter": 400}); z = least_squares(F5, res.x).x
    if np.linalg.norm(F5(z)) < 1e-8: best = max(best, witness(unpack5(z)))
qb = (3 * k * np.cos(np.pi / k) - k) / (1 + np.cos(np.pi / k)) - (k - 2)
check("BR6 C5: the witness over Hermitian Rep_3 reaches the quantum 5-cycle bound (KCBS in a qutrit)", abs(best - qb) < 1e-5, f"{best:.6f} vs {qb:.6f}")
def canon(key):
    bestc = None; cols = [tuple(c) for c in key]
    for r_ in range(k):
        for refl in (False, True):
            perm = [((-i if refl else i) + r_) % k for i in range(k)]
            for fl in itertools.product((1, -1), repeat=k):
                cc = tuple(sorted(tuple(fl[v] * c[perm[v]] for v in range(k)) for c in cols))
                if bestc is None or cc < bestc: bestc = cc
    return bestc
reps = {}; sizes = {}
for X, key in diagonal_points(G5, 3):
    c = canon(key); reps.setdefault(c, X); sizes[c] = sizes.get(c, 0) + 1
def local_max(X0, eps, starts):
    z0 = pack5(X0); bestw = -np.inf
    for s_ in range(starts):
        z = least_squares(F5, z0 + eps * 0.5 * rng.normal(size=z0.size)).x
        cons = [{"type": "ineq", "fun": lambda zz: eps ** 2 - np.sum((zz - z0) ** 2)}]
        res = minimize(lambda zz: -witness(unpack5(zz)) + 200 * np.sum(F5(zz) ** 2), z, method="SLSQP", constraints=cons, options={"maxiter": 300})
        zz = least_squares(F5, res.x).x
        if np.linalg.norm(F5(zz)) < 1e-8 and np.sum((zz - z0) ** 2) <= (1.05 * eps) ** 2: bestw = max(bestw, witness(unpack5(zz)))
    return bestw
def Ffull(z):
    X = [z[v * 9:(v + 1) * 9].reshape(3, 3) for v in range(k)]
    r = [X[v] @ X[v] - np.eye(3) for v in range(k)] + [X[u] @ X[w] - X[w] @ X[u] for u, w in G5[1]]
    return np.concatenate([q.ravel() for q in r])
def tdim(z):
    J = np.array([(Ffull(z + 1e-7 * e) - Ffull(z - 1e-7 * e)) / 2e-7 for e in np.eye(len(z))]).T
    sv = np.linalg.svd(J, compute_uv=False); return len(z) - int(np.sum(sv > 1e-6 * sv.max()))
def smooth(X0):
    z0 = np.concatenate([x.ravel() for x in X0]); d0 = tdim(z0); near = set()
    for _ in range(20):
        z = least_squares(Ffull, z0 + 0.02 * rng.normal(size=z0.size), xtol=1e-15, ftol=1e-15, gtol=1e-15).x
        if np.linalg.norm(Ffull(z)) < 1e-10: near.add(tdim(z))
    return d0, sorted(near)
rows = []
for c, X in reps.items():
    rows.append({"class": [list(map(int, col)) for col in c], "size": sizes[c], "w03": local_max(X, 0.3, 6), "w01": local_max(X, 0.1, 6),
                 "dims": smooth(X)})
born = [r for r in rows if r["w01"] > 1e-4]
sing = [r for r in rows if r["dims"][0] > max(r["dims"][1])]
ok = len(born) == 1
if ok:
    Xb = reps[tuple(tuple(col) for col in born[0]["class"])]
    ws = [local_max(Xb, e, 8) for e in (0.1, 0.03, 0.01)]; ratios = [ws[i] / ws[i + 1] for i in range(2)]
    K_, _ = _null(linearisation(G5, Xb)); Lb = linearisation(G5, Xb); third = True
    tol9 = lambda y: [y[i * 9:(i + 1) * 9].reshape(3, 3) for i in range(k)]
    def bil(Y, Z):
        return np.concatenate([(Y[i] @ Z[i] + Z[i] @ Y[i]).ravel() for i in range(k)] + [(Y[u] @ Z[w] - Z[w] @ Y[u] + Z[u] @ Y[w] - Y[w] @ Z[u]).ravel() for u, w in G5[1]])
    for _ in range(4):
        y1 = K_ @ rng.normal(size=K_.shape[1]); Y1 = tol9(y1); q2 = -quadratic(G5, Y1); y2, *_ = np.linalg.lstsq(Lb, q2, rcond=None)
        A_ = np.hstack([Lb, np.array([bil(Y1, tol9(K_[:, j])) for j in range(K_.shape[1])]).T]); rhs = -bil(Y1, tol9(y2))
        sol, *_ = np.linalg.lstsq(A_, rhs, rcond=None); third &= np.linalg.norm(Lb @ y2 - q2) < 1e-9 and np.linalg.norm(A_ @ sol - rhs) < 1e-9
    d0, near = born[0]["dims"]
    ok = ok and d0 == max(near) and min(near) == d0 and third and all(5 < x < 15 for x in ratios)
    OUT["c5_birth"] = {"class": born[0]["class"], "size": born[0]["size"], "witness_eps": dict(zip(["0.1", "0.03", "0.01"], ws)), "dims": [d0, near]}
check("BR7 C5, Rep_3: exactly one class of classical points gives birth to contextual models, it does so at second order, and it is a smooth point",
      ok, (f"{len(rows)} classes; birth class of {born[0]['size']} points, witness {ws[0]:.1e}, {ws[1]:.1e}, {ws[2]:.1e} at eps 0.1, 0.03, 0.01; "
           f"tangent dim {born[0]['dims'][0]} = nearby {born[0]['dims'][1]}; extends to third order") if born else f"{len(born)} birth classes")
ok8 = len(sing) >= 1 and all(r["w03"] < 1e-6 and r["w01"] < 1e-6 for r in sing) and not any(r in sing for r in born)
OUT["c5_classes"] = rows
check("BR8 C5, Rep_3: the singular class gives no birth; birth and singularity are disjoint", ok8,
      "; ".join(f"singular class of {r['size']} points: tangent dim {r['dims'][0]} > nearby {r['dims'][1]}, max nearby witness {max(r['w03'], r['w01']):.1e}" for r in sing))

# ---------- BR9-BR11: what degenerates ----------
def commutant_dim(X):
    n_ = X[0].shape[0]; Cm = np.vstack([np.kron(x, np.eye(n_)) - np.kron(np.eye(n_), x.T) for x in X])
    return int(np.sum(np.linalg.svd(Cm, compute_uv=False) < 1e-7))
Xb = reps[tuple(tuple(col) for col in born[0]["class"])] if born else None
Xs = reps[tuple(tuple(col) for col in sing[0]["class"])] if sing else None
def near_stats(X0, m=100):
    z0 = pack5(X0); cds = []; ctx = 0
    for _ in range(m):
        z = least_squares(F5, z0 + 0.05 * rng.normal(size=z0.size), xtol=1e-14, ftol=1e-14, gtol=1e-14).x
        if np.linalg.norm(F5(z)) < 1e-9:
            X = unpack5(z); cds.append(commutant_dim(X)); ctx += witness(X) > 1e-6
    return cds, ctx / max(len(cds), 1)
cb, sb = near_stats(Xb); cs, ss = near_stats(Xs)
check("BR9 Rep_3(C5): near the birth class every model is irreducible and contextual; near the singular class every model is 2+1 and none is",
      set(cb) == {1} and sb > 0.95 and set(cs) == {2} and ss == 0.0,
      f"birth: commutant dims {sorted(set(cb))}, contextual share {sb:.2f}; singular: commutant dims {sorted(set(cs))} (2 blocks), contextual share {ss:.2f}")
SC3 = json.load(open(os.path.join(HERE, "..", "level2", "c5_scan_n3.json"))); SC4 = json.load(open(os.path.join(HERE, "..", "level2", "c5_scan_n4.json")))
isborn = lambda r: r["local_max_witness"] > 1e-6 or r["random_ctx_share"] > 0
maxblk = lambda r: max(eval(d)[0] for d in r["nearby_blocks"]) if r["nearby_blocks"] else 0
irr3 = [r for r in SC3 if maxblk(r) >= 3]; irr4 = [r for r in SC4 if maxblk(r) >= 3]; b3 = [r for r in SC3 if isborn(r)]; b4 = [r for r in SC4 if isborn(r)]
# global sample of irreducible models of W_C5 in dimension 3
irrN = irrC = 0
for _ in range(400):
    z0 = np.concatenate([(lambda Q, D: (Q @ D @ Q.T)[iu5])(np.linalg.qr(rng.normal(size=(3, 3)))[0], np.diag(rng.choice([-1, 1], 3))) for _ in range(k)])
    z = least_squares(F5, z0, xtol=1e-14, ftol=1e-14, gtol=1e-14).x
    if np.linalg.norm(F5(z)) < 1e-9 and commutant_dim(unpack5(z)) == 1:
        irrN += 1; irrC += witness(unpack5(z)) > 1e-7
ok10 = all(maxblk(r) >= 3 for r in b3 + b4) and len(irr3) > len(b3) and len(irr4) > len(b4) and irrN > 20 and irrC < irrN
OUT["what_degenerates"] = {"n3": {"classes": len(SC3), "irreducible_nearby": len(irr3), "birth": len(b3)},
                           "n4": {"classes": len(SC4), "block>=3_nearby": len(irr4), "birth": len(b4)}, "global_irreducible_n3": [irrN, irrC]}
check("BR10 a degenerating block of size >= 3 is necessary for birth but not sufficient (Rep_3 and Rep_4); irreducible models are often noncontextual",
      ok10, f"Rep_3: {len(irr3)} classes with irreducible neighbours, {len(b3)} birth; Rep_4: {len(irr4)} with a block >= 3, {len(b4)} birth; "
            f"global Rep_3 sample: {irrC} of {irrN} irreducible models contextual")
s4 = [r for r in SC4 if r["singular"]]; both = [r for r in b4 if r["singular"]]
pts = lambda L: sum(r["size"] for r in L)
OUT["rep4_loci"] = {"singular": [len(s4), pts(s4)], "birth": [len(b4), pts(b4)], "both": [len(both), pts(both)]}
check("BR11 Rep_4(C5): birth and singularity overlap only partly", 0 < len(both) < min(len(s4), len(b4)),
      f"{len(SC4)} classes ({pts(SC4)} points): singular {len(s4)} ({pts(s4)}), birth {len(b4)} ({pts(b4)}), both {len(both)} ({pts(both)})")

json.dump(OUT, open(os.path.join(HERE, "..", "level2", "birth_results.json"), "w"), default=float)
print(f"\n{sum(results)}/{len(results)} checks passed")
