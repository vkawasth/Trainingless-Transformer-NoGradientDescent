"""check_bands.py -- every mathematical claim of "Bands" and the synthetic validation of its clinical tests.

  B1  discounting a law p with reliability alpha gives exactly the element band [alpha p, alpha p + 1 - alpha]
  B2  tightening of an element band: every tightened bound is attained (LP)
  B3  Frechet bounds over banded margins: cell bounds of couplings of banded margins equal the closed form (LP, 2x3)
  B4  the total-variation distance from (f, g) to the walls is min |f(I) - g(J)|, and to the boundary is min f_y (LP)
  B5  upper entropy of an element band is water-filling; lower entropy is attained at a vertex (vertex enumeration)
  B6  entropy continuity |H(p) - H(q)| <= t log(n-1) + h(t) holds and is attained
  B7  across a wall the entropy of the comonotone vertex coupling has a kink; the product coupling's entropy is smooth;
      near the boundary the gradient of H diverges
  B8  level-two decomposition H(mean) = E H + I, and the gluing gap H(filler) - H(P) = I(X;Z|Y)
  B9  conditioning on events commutes and is idempotent; Jeffrey updates do not commute in general, commute for a
      product law; a loop of Jeffrey updates leaves a residue, preserves the odds ratio, and iterated fitting returns
  B10 dilation: a band that is a point becomes [0,1] after conditioning on either value of a second variable
  B11 Dobrushin contraction of the band diameter under a kernel
  B13 over an element band, min p(I) = max(sum_I lo, 1 - sum_{I^c} hi) and max p(I) = min(sum_I hi, 1 - sum_{I^c} lo); a
      pair of independent bands is tear-safe iff every wall function keeps its sign between these extremes
  P1  the pilot numbers and the synthetic and example numbers quoted in the paper match the saved outputs and this run
  B12 tear-time bound: steps of TV size <= v cannot reach a wall in fewer than ceil(m/v) steps (random walks)
  S1  G-test of conditional independence (test 3): size close to 0.05 under the null, power under a planted effect
  S2  selection creates marginal incompatibility (test 4): available-case tables exceed the bootstrap null; inverse-
      probability weighting brings them back; contextual fraction and L1 inconsistency agree in direction
  S3  planted walls: the chamber is identified from samples once n m^2 is large (theory Prop. 12.2)
"""
import itertools, math, random
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.stats import chi2

R = np.random.default_rng(7); res = []
def check(name, cond, info=""):
    res.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))
def H(p):
    p = np.asarray(p, float).ravel(); p = p[p > 0]; return float(-(p * np.log(p)).sum())
def dist(n): x = R.random(n) + 0.05; return x / x.sum()

# ---------------------------------------------------------------- B1 discounting
ok = True
for _ in range(50):
    n = R.integers(2, 6); p = dist(n); a = R.uniform(0.1, 0.95); lo, hi = a * p, a * p + 1 - a
    q = dist(n); r = a * p + (1 - a) * q                       # every discounted law lies in the band
    ok &= np.all(r >= lo - 1e-12) and np.all(r <= hi + 1e-12) and abs(r.sum() - 1) < 1e-12
    # every band member is a discounted law: q = (r - a p)/(1 - a) is a distribution
    x = R.random(n); x = lo + (hi - lo) * x; x = x / x.sum()       # random point, then project onto the sum by LP feasibility
    res_lp = linprog(np.zeros(n), A_eq=np.ones((1, n)), b_eq=[1], bounds=list(zip(lo, hi)), method="highs")
    s = res_lp.x; qq = (s - a * p) / (1 - a); ok &= np.all(qq >= -1e-12) and abs(qq.sum() - 1) < 1e-9
check("B1 discounting with reliability alpha gives exactly the band [alpha p, alpha p + 1 - alpha]", ok)

# ---------------------------------------------------------------- B2 tightening
def tighten(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    lo2 = np.maximum(lo, 1 - (hi.sum() - hi)); hi2 = np.minimum(hi, 1 - (lo.sum() - lo)); return lo2, hi2
def random_band(n):
    p = dist(n); w = R.uniform(0.02, 0.4, n); lo = np.clip(p - w * R.random(n), 0, 1); hi = np.clip(p + w * R.random(n), 0, 1)
    return lo, hi
ok = True
for _ in range(60):
    n = R.integers(2, 6); lo, hi = random_band(n); lo2, hi2 = tighten(lo, hi)
    for i in range(n):
        c = np.zeros(n); c[i] = 1
        mn = linprog(c, A_eq=np.ones((1, n)), b_eq=[1], bounds=list(zip(lo, hi)), method="highs").fun
        mx = -linprog(-c, A_eq=np.ones((1, n)), b_eq=[1], bounds=list(zip(lo, hi)), method="highs").fun
        ok &= abs(mn - lo2[i]) < 1e-9 and abs(mx - hi2[i]) < 1e-9
check("B2 tightened bounds max(lo_i, 1 - sum_{k!=i} hi_k), min(hi_i, 1 - sum_{k!=i} lo_k) are attained", ok)

# ---------------------------------------------------------------- B3 Frechet over banded margins
ok = True
for _ in range(25):
    (lf, hf), (lg, hg) = random_band(2), random_band(3); lf, hf = tighten(lf, hf); lg, hg = tighten(lg, hg)
    nv = 2 + 3 + 6                                           # f (2), g (3), alpha (2x3)
    A = []; b = []
    for i in range(2):                                       # row sums of alpha = f
        r = np.zeros(nv); r[i] = -1; r[5 + 3 * i:5 + 3 * i + 3] = 1; A.append(r); b.append(0)
    for j in range(3):
        r = np.zeros(nv); r[2 + j] = -1; r[5 + j:11:3] = 1; A.append(r); b.append(0)
    r = np.zeros(nv); r[:2] = 1; A.append(r); b.append(1)
    bounds = list(zip(lf, hf)) + list(zip(lg, hg)) + [(0, 1)] * 6
    for i in range(2):
        for j in range(3):
            c = np.zeros(nv); c[5 + 3 * i + j] = 1
            mn = linprog(c, A_eq=np.array(A), b_eq=b, bounds=bounds, method="highs").fun
            mx = -linprog(-c, A_eq=np.array(A), b_eq=b, bounds=bounds, method="highs").fun
            ok &= abs(mn - max(0, lf[i] + lg[j] - 1)) < 1e-9 and abs(mx - min(hf[i], hg[j])) < 1e-9
check("B3 couplings of banded margins: cell bounds [max(0, lo_f + lo_g - 1), min(hi_f, hi_g)] after tightening", ok)

# ---------------------------------------------------------------- B4 distance to walls and to the boundary
def walls(n, m):
    for k in range(1, n):
        for I in itertools.combinations(range(n), k):
            for l in range(1, m):
                for J in itertools.combinations(range(m), l): yield I, J
def tear_margin(f, g): return min(abs(f[list(I)].sum() - g[list(J)].sum()) for I, J in walls(len(f), len(g)))
def tv_to_wall_lp(f, g, I, J):
    n, m = len(f), len(g); nv = 2 * (n + m)                   # f', g', and slacks for |f - f'|, |g - g'|
    A_eq, b_eq, A_ub, b_ub = [], [], [], []
    r = np.zeros(nv); r[list(I)] = 1; r[[n + j for j in J]] = -1; A_eq.append(r); b_eq.append(0)
    r = np.zeros(nv); r[:n] = 1; A_eq.append(r); b_eq.append(1)
    r = np.zeros(nv); r[n:n + m] = 1; A_eq.append(r); b_eq.append(1)
    for k in range(n + m):
        v = np.concatenate([f, g])[k]
        r = np.zeros(nv); r[k] = 1; r[n + m + k] = -1; A_ub.append(r); b_ub.append(v)
        r = np.zeros(nv); r[k] = -1; r[n + m + k] = -1; A_ub.append(r); b_ub.append(-v)
    c = np.concatenate([np.zeros(n + m), 0.5 * np.ones(n + m)])
    return linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq, bounds=[(0, 1)] * (n + m) + [(0, None)] * (n + m), method="highs").fun
ok = True; ex = None
for _ in range(15):
    f, g = dist(R.integers(2, 4)), dist(R.integers(2, 5))
    d = min(tv_to_wall_lp(f, g, I, J) for I, J in walls(len(f), len(g)))
    ok &= abs(d - tear_margin(f, g)) < 1e-9; ex = ex or (f, g, d)
# boundary: TV distance to {p: p_y = 0} is p_y (move p_y elsewhere); verified by LP
okb = True
for _ in range(20):
    p = dist(4); y = R.integers(4)
    c = np.concatenate([np.zeros(4), 0.5 * np.ones(4)]); A_ub = []; b_ub = []
    for k in range(4):
        r = np.zeros(8); r[k] = 1; r[4 + k] = -1; A_ub.append(r); b_ub.append(p[k])
        r = np.zeros(8); r[k] = -1; r[4 + k] = -1; A_ub.append(r); b_ub.append(-p[k])
    A_eq = [np.r_[np.ones(4), np.zeros(4)], np.r_[np.eye(4)[y], np.zeros(4)]]
    v = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=[1, 0], bounds=[(0, 1)] * 4 + [(0, None)] * 4, method="highs").fun
    okb &= abs(v - p[y]) < 1e-9
check("B4 TV distance from (f,g) to the walls equals min |f(I) - g(J)|; TV distance to the face p_y = 0 equals p_y", ok and okb,
      f"example margins {np.round(ex[0],3)}, {np.round(ex[1],3)}: distance {ex[2]:.4f}")

# ---------------------------------------------------------------- B5 upper and lower entropy of an element band
def upper_entropy_point(lo, hi):
    a, b = 0.0, 1.0
    for _ in range(200):
        c = (a + b) / 2; s = np.clip(c, lo, hi).sum()
        if s > 1: b = c
        else: a = c
    return np.clip((a + b) / 2, lo, hi)
def band_vertices(lo, hi):
    n = len(lo); V = []
    for i in range(n):
        others = [k for k in range(n) if k != i]
        for choice in itertools.product([0, 1], repeat=n - 1):
            p = np.zeros(n)
            for k, ch in zip(others, choice): p[k] = hi[k] if ch else lo[k]
            p[i] = 1 - p[others].sum()
            if lo[i] - 1e-12 <= p[i] <= hi[i] + 1e-12: V.append(p)
    return V
ok_up = ok_lo = True
for _ in range(30):
    n = R.integers(3, 6); lo, hi = tighten(*random_band(n))
    pu = upper_entropy_point(lo, hi)
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    opt = minimize(lambda x: -H(np.clip(x, 1e-15, 1)), (lo + hi) / 2 / ((lo + hi) / 2).sum(), bounds=list(zip(lo, hi)), constraints=cons, method="SLSQP")
    ok_up &= H(pu) >= -opt.fun - 1e-7 and abs(pu.sum() - 1) < 1e-9
    V = band_vertices(lo, hi); hmin = min(H(v) for v in V)
    samples = []
    for _ in range(400):
        w = R.dirichlet(np.ones(len(V))); samples.append(H(sum(wi * v for wi, v in zip(w, V))))
    ok_lo &= min(samples) >= hmin - 1e-12
check("B5 upper entropy of a band = water-filling clip(c, lo, hi); lower entropy attained at a vertex", ok_up and ok_lo)

# ---------------------------------------------------------------- B6 continuity of entropy
def hb(t): return 0.0 if t in (0, 1) else -t * math.log(t) - (1 - t) * math.log(1 - t)
ok = True
for _ in range(2000):
    n = R.integers(2, 7); p, q = R.dirichlet(np.ones(n) * 0.5), R.dirichlet(np.ones(n) * 0.5)
    t = 0.5 * np.abs(p - q).sum()
    if t <= 1 - 1 / n: ok &= abs(H(p) - H(q)) <= t * math.log(n - 1) + hb(t) + 1e-12
n, t = 5, 0.3; p = np.r_[1, np.zeros(4)]; q = np.r_[1 - t, np.full(4, t / 4)]
tight = abs(abs(H(p) - H(q)) - (t * math.log(n - 1) + hb(t))) < 1e-12
check("B6 |H(p) - H(q)| <= t log(n-1) + h(t) on 2000 random pairs, with equality at p = delta, q = (1-t, t/(n-1), ...)", ok and tight)

# ---------------------------------------------------------------- B7 kinks at walls, smoothness elsewhere, divergence at the boundary
g1 = 0.3
def comonotone_entropy(f1):          # 2x2: the vertex with alpha_11 = min(f1, g1)
    a = min(f1, g1); return H([a, f1 - a, g1 - a, 1 - f1 - g1 + a])
def product_entropy(f1): return H([f1, 1 - f1]) + H([g1, 1 - g1])
e = 1e-6
dL = (comonotone_entropy(g1) - comonotone_entropy(g1 - e)) / e; dR = (comonotone_entropy(g1 + e) - comonotone_entropy(g1)) / e
pL = (product_entropy(g1) - product_entropy(g1 - e)) / e; pR = (product_entropy(g1 + e) - product_entropy(g1)) / e
# away from the wall (f1 = 0.5) the comonotone entropy is smooth
aL = (comonotone_entropy(0.5) - comonotone_entropy(0.5 - e)) / e; aR = (comonotone_entropy(0.5 + e) - comonotone_entropy(0.5)) / e
grad = [abs(-math.log(y) - 1) for y in (1e-2, 1e-4, 1e-6)]
slopes = [(comonotone_entropy(g1 + ee) - comonotone_entropy(g1)) / ee for ee in (1e-3, 1e-5, 1e-7)]
cusp = all(abs(slopes[k + 1] - slopes[k] - math.log(100)) < 0.05 for k in range(2))   # slope grows like log(1/eps)
def countermonotone_entropy(f1):
    a = max(0.0, f1 + g1 - 1); return H([a, f1 - a, g1 - a, 1 - f1 - g1 + a])
# inside the chamber 0.3 < f1 < 0.7 the two vertex entropies cross at f1 = 0.5, so the lower entropy has an ordinary kink there
lower = lambda f1: min(comonotone_entropy(f1), countermonotone_entropy(f1))
kinkL = (lower(0.5) - lower(0.5 - e)) / e; kinkR = (lower(0.5 + e) - lower(0.5)) / e
cusp &= abs(comonotone_entropy(0.5) - countermonotone_entropy(0.5)) < 1e-12 and kinkR - kinkL < -0.5
check("B7 comonotone-coupling entropy has a d log(1/d) cusp at the wall f1 = g1 (one-sided slope grows like log 1/eps); "
      "it is smooth away from walls; product entropy smooth; |dH/dp_y| diverges as p_y -> 0",
      cusp and abs(pR - pL) < 1e-4 and abs(aR - aL) < 1e-4 and grad[0] < grad[1] < grad[2],
      f"one-sided slopes at eps = 1e-3, 1e-5, 1e-7: {[round(x, 2) for x in slopes]}")

# ---------------------------------------------------------------- B8 level-two decomposition and the gluing gap
ok = True
for _ in range(50):
    k, n = R.integers(2, 5), R.integers(2, 5); w = dist(k); rho = [dist(n) for _ in range(k)]
    J = np.array([w[i] * rho[i] for i in range(k)]); mean = J.sum(0)
    I = sum(J[i, x] * math.log(J[i, x] / (w[i] * mean[x])) for i in range(k) for x in range(n))
    ok &= abs(H(mean) - (sum(w[i] * H(rho[i]) for i in range(k)) + I)) < 1e-12
    P = R.random((3, 4, 2)); P /= P.sum(); Pxy, Pyz, Py = P.sum(2), P.sum(0), P.sum((0, 2))
    F = Pxy[:, :, None] * Pyz[None, :, :] / Py[None, :, None]
    cmi = float((P * np.log(P * Py[None, :, None] / (Pxy[:, :, None] * Pyz[None, :, :]))).sum())
    ok &= abs((H(F) - H(P)) - cmi) < 1e-12 and np.allclose(F.sum(2), Pxy) and np.allclose(F.sum(0), Pyz)
check("B8 H(mean) = E_Pi H(rho) + I(model; outcome); H(gluing filler) - H(P) = I(X;Z|Y) with the same (X,Y), (Y,Z) margins", ok)

# ---------------------------------------------------------------- B9 the update monoid
def cond(p, mask): q = p * mask; return q / q.sum()
def jeffrey_rows(P, q):  return P / P.sum(1, keepdims=True) * np.asarray(q)[:, None]
def jeffrey_cols(P, r):  return P / P.sum(0, keepdims=True) * np.asarray(r)[None, :]
ok_cond = True
for _ in range(50):
    p = dist(6); A = R.random(6) < 0.6; B = R.random(6) < 0.6
    if (p * A * B).sum() == 0: continue
    ok_cond &= np.allclose(cond(cond(p, A), B), cond(cond(p, B), A)) and np.allclose(cond(cond(p, A), A), cond(p, A))
P0 = np.array([[0.30, 0.10], [0.15, 0.45]]); q1, r1 = [0.6, 0.4], [0.35, 0.65]
noncomm = np.abs(jeffrey_cols(jeffrey_rows(P0, q1), r1) - jeffrey_rows(jeffrey_cols(P0, r1), q1)).sum() / 2
Pprod = np.outer([0.4, 0.6], [0.7, 0.3])
comm_prod = np.allclose(jeffrey_cols(jeffrey_rows(Pprod, q1), r1), jeffrey_rows(jeffrey_cols(Pprod, r1), q1))
q0, r0 = P0.sum(1), P0.sum(0)
Pend = jeffrey_cols(jeffrey_rows(jeffrey_cols(jeffrey_rows(P0, q1), r1), q0), r0)
residue = np.abs(Pend - P0).sum() / 2
odds = lambda P: P[0, 0] * P[1, 1] / (P[0, 1] * P[1, 0])
Pit = Pend.copy()
for _ in range(200): Pit = jeffrey_cols(jeffrey_rows(Pit, q0), r0)
check("B9 conditioning commutes and is idempotent; Jeffrey updates do not commute (TV defect) but commute on a product law; "
      "a loop leaves a residue, keeps the odds ratio, and iterated fitting returns to the start",
      ok_cond and noncomm > 1e-3 and comm_prod and residue > 1e-3 and abs(odds(Pend) - odds(P0)) < 1e-12 and np.abs(Pit - P0).max() < 1e-10,
      f"commutator defect {noncomm:.4f}, loop residue {residue:.4f}, odds ratio {odds(P0):.3f} kept")

# ---------------------------------------------------------------- B10 dilation
# joints on (X, Y) in {0,1}^2 with P(X=1) = P(Y=1) = 1/2: band for P(X=1 | Y=y)
ok = True; bands = []
for y in (0, 1):
    for sgn in (1, -1):
        # variables p00, p01, p10, p11; P(X=1) = p10 + p11 = 1/2, P(Y=1) = p01 + p11 = 1/2, sum = 1
        A_eq = [[0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1]]
        c = np.zeros(4); c[2 + y] = sgn                       # P(X=1, Y=y) = p_{1y}; P(Y=y) = 1/2 fixed
        v = linprog(c, A_eq=A_eq, b_eq=[0.5, 0.5, 1], bounds=[(0, 1)] * 4, method="highs").fun
        bands.append(sgn * v / 0.5)
ok = abs(bands[0]) < 1e-12 and abs(bands[1] - 1) < 1e-12 and abs(bands[2]) < 1e-12 and abs(bands[3] - 1) < 1e-12
check("B10 dilation: P(X=1) is the point 1/2, yet P(X=1 | Y=y) ranges over [0, 1] for both y", ok)

# ---------------------------------------------------------------- B11 Dobrushin contraction of band diameter
ok = True
for _ in range(30):
    n = R.integers(3, 5); lo, hi = tighten(*random_band(n)); V = band_vertices(lo, hi)
    K = R.dirichlet(np.ones(n), size=n); delta = max(0.5 * np.abs(K[i] - K[j]).sum() for i in range(n) for j in range(n))
    diam = max(0.5 * np.abs(u - v).sum() for u in V for v in V); diamK = max(0.5 * np.abs(K.T @ u - K.T @ v).sum() for u in V for v in V)
    ok &= diamK <= delta * diam + 1e-12
check("B11 the TV diameter of a band shrinks by at least the Dobrushin coefficient under a kernel", ok)

# ---------------------------------------------------------------- B12 tear-time bound
ok = True
for _ in range(200):
    f, g = dist(2), dist(3); m = tear_margin(f, g); v = R.uniform(0.005, 0.05)
    side = lambda f, g: tuple(np.sign(f[list(I)].sum() - g[list(J)].sum()) for I, J in walls(2, 3))
    s0 = side(f, g); steps = 0
    while steps < 500:
        df = R.normal(size=2); df -= df.mean(); df *= v / (0.5 * np.abs(df).sum()) / 2
        dg = R.normal(size=3); dg -= dg.mean(); dg *= v / (0.5 * np.abs(dg).sum()) / 2
        if np.any(f + df <= 0) or np.any(g + dg <= 0): break
        f, g = f + df, g + dg; steps += 1
        if side(f, g) != s0:
            ok &= steps >= math.ceil(m / v); break
check("B12 random walks with TV steps <= v never leave the chamber before ceil(m_T / v) steps", ok)

# ---------------------------------------------------------------- B13 extremes of event probabilities over a band
def pI_range(lo, hi, I):
    I = list(I); Ic = [k for k in range(len(lo)) if k not in I]
    return max(lo[I].sum(), 1 - hi[Ic].sum()), min(hi[I].sum(), 1 - lo[Ic].sum())
ok = True
for _ in range(60):
    n = R.integers(2, 6); lo, hi = random_band(n)
    for k in range(1, n):
        for I in itertools.combinations(range(n), k):
            c = np.zeros(n); c[list(I)] = 1
            mn = linprog(c, A_eq=np.ones((1, n)), b_eq=[1], bounds=list(zip(lo, hi)), method="highs").fun
            mx = -linprog(-c, A_eq=np.ones((1, n)), b_eq=[1], bounds=list(zip(lo, hi)), method="highs").fun
            a, b = pI_range(lo, hi, I); ok &= abs(mn - a) < 1e-9 and abs(mx - b) < 1e-9
# tear-safety: sign constancy of f(I) - g(J) over independent bands, compared with dense sampling of the bands
oks = True
for _ in range(20):
    (lf, hf), (lg, hg) = random_band(2), random_band(3)
    safe = all((pI_range(lf, hf, I)[0] - pI_range(lg, hg, J)[1]) > 0 or (pI_range(lf, hf, I)[1] - pI_range(lg, hg, J)[0]) < 0
               for I, J in walls(2, 3))
    Vf, Vg = band_vertices(*tighten(lf, hf)), band_vertices(*tighten(lg, hg))
    chambers = {tuple(np.sign(u[list(I)].sum() - v[list(J)].sum()) for I, J in walls(2, 3)) for u in Vf for v in Vg}
    oks &= (not safe) or len(chambers) == 1
    if not safe:   # some wall is reached: then there are band points on both sides or on the wall
        oks &= len(chambers) > 1 or any(0 in ch for ch in chambers)
check("B13 min/max of p(I) over an element band in closed form; the sign test decides whether a band pair meets a wall", ok and oks)

# ================================================================= synthetic validation of the clinical tests
R = np.random.default_rng(2026)          # separate seed, so the quoted synthetic numbers do not depend on the checks above
# ---------------------------------------------------------------- S1 G-test of conditional independence
def g_test(X, Y, Z, kx, ky, kz):
    N = len(X); P = np.zeros((kx, ky, kz)); np.add.at(P, (X, Y, Z), 1); P /= N
    Pxy, Pyz, Py = P.sum(2), P.sum(0), P.sum((0, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = P * np.log(P * Py[None, :, None] / (Pxy[:, :, None] * Pyz[None, :, :]))
    cmi = float(np.nansum(t)); stat = 2 * N * cmi; df = (kx - 1) * (kz - 1) * ky
    return cmi, chi2.sf(stat, df)
def sample(N, beta):
    Y = R.integers(0, 3, N); X = (R.random(N) < 0.3 + 0.2 * Y / 2).astype(int)
    pz = 0.2 + 0.3 * Y / 2 + beta * X; Z = (R.random(N) < np.clip(pz, 0, 1)).astype(int); return X, Y, Z
rej0 = np.mean([g_test(*sample(2000, 0.0), 2, 3, 2)[1] < 0.05 for _ in range(400)])
rej1 = np.mean([g_test(*sample(2000, 0.08), 2, 3, 2)[1] < 0.05 for _ in range(200)])
check("S1 G-test 2N I(X;Z|Y): size near 0.05 under X _||_ Z | Y, high power for a planted 8-point effect", 0.025 <= rej0 <= 0.08 and rej1 > 0.8,
      f"size {rej0:.3f}, power {rej1:.3f} at N = 2000")

# ---------------------------------------------------------------- S2 selection and marginal incompatibility
CELLS = list(itertools.product(range(3), range(2), range(2)))     # (X, Y, Z)
def pair_rows(S, kS):
    return np.array([[1.0 if all(c[i] == v for i, v in zip(S, vals)) else 0.0 for c in CELLS] for vals in itertools.product(*[range(k) for k in kS])])
PAIRS = [((0, 1), (3, 2)), ((1, 2), (2, 2)), ((0, 2), (3, 2))]
def l1_inconsistency(tables):
    M = np.vstack([pair_rows(S, k) for S, k in PAIRS]); b = np.concatenate(tables); k = len(b); nC = len(CELLS)
    A_eq = np.vstack([np.hstack([M, -np.eye(k), np.eye(k)]), np.hstack([np.ones(nC), np.zeros(2 * k)])])
    return linprog(np.r_[np.zeros(nC), np.ones(2 * k)], A_eq=A_eq, b_eq=np.r_[b, 1], bounds=[(0, None)] * (nC + 2 * k), method="highs").fun
def contextual_fraction(tables):
    """Abramsky-Barbosa-Mansfield: CF = 1 - max lambda with lambda * (margins of a joint) <= the empirical tables."""
    M = np.vstack([pair_rows(S, k) for S, k in PAIRS]); b = np.concatenate(tables); nC = len(CELLS)
    # variables: sub-normalised joint x >= 0 with M x <= b; maximise sum x  (sum x = lambda)
    r = linprog(-np.ones(nC), A_ub=M, b_ub=b, bounds=[(0, None)] * nC, method="highs")
    return 1 - (-r.fun)                                      # every context table of x has total mass sum(x) = lambda
def population(N):
    X = R.integers(0, 3, N); Y = (R.random(N) < 0.2 + 0.25 * X).astype(int); Z = (R.random(N) < 0.15 + 0.2 * X + 0.15 * Y).astype(int)
    return X, Y, Z
def tables_from(X, Y, Z, obs, w=None):
    w = np.ones(len(X)) if w is None else w; out = []
    for S, kS in PAIRS:
        use = np.ones(len(X), bool) if 2 not in S else obs
        ww = np.ones(len(X)) if 2 not in S else w            # weights only for tables that need the lab Z
        V = [X, Y, Z]; t = np.zeros(kS)
        np.add.at(t, tuple(V[i][use] for i in S), ww[use]); out.append((t / t.sum()).ravel())
    return out
N = 3000
X, Y, Z = population(N)
p_obs = np.array([0.9, 0.55, 0.2])[X]                     # Z (a lab) measured far more often when X is low: MAR on X
obs = R.random(N) < p_obs
inc_ac = l1_inconsistency(tables_from(X, Y, Z, obs)); cf_ac = contextual_fraction(tables_from(X, Y, Z, obs))
ipw = np.zeros(N);
for x in range(3): ipw[X == x] = 1 / obs[X == x].mean()     # estimated P(observed | X)
inc_ipw = l1_inconsistency(tables_from(X, Y, Z, obs, ipw)); cf_ipw = contextual_fraction(tables_from(X, Y, Z, obs, ipw))
def ipw_weights(Xs, obs_s):
    w_ = np.zeros(len(Xs))
    for x in range(3): w_[Xs == x] = 1 / obs_s[Xs == x].mean()     # estimated P(observed | X)
    return w_
null, null_ipw, null_cf = [], [], []
for _ in range(200):
    Xb, Yb, Zb = population(N)
    ob = R.random(N) < obs.mean()                         # same rate, missing completely at random
    null.append(l1_inconsistency(tables_from(Xb, Yb, Zb, ob))); null_cf.append(contextual_fraction(tables_from(Xb, Yb, Zb, ob)))
    ob2 = R.random(N) < np.array([0.9, 0.55, 0.2])[Xb]    # same MAR design, corrected by IPW: sampling noise of the weighted estimator
    null_ipw.append(l1_inconsistency(tables_from(Xb, Yb, Zb, ob2, ipw_weights(Xb, ob2))))
q95 = float(np.quantile(null, 0.95)); q95_ipw = float(np.quantile(null_ipw, 0.95))
med, med_ipw = float(np.median(null)), float(np.median(null_ipw)); q95_cf = float(np.quantile(null_cf, 0.95))
check("S2 MAR missingness: available-case tables are incompatible far beyond the sampling null; after IPW the inconsistency is "
      "of the same order as the sampling null of complete-at-random missingness; the contextual fraction drops with it",
      inc_ac > 5 * max(q95, q95_ipw) and 0.5 < q95_ipw / q95 < 2 and 0.5 < med_ipw / med < 2 and cf_ac > 5 * cf_ipw and cf_ac > 5 * q95_cf,
      f"L1 inconsistency {inc_ac:.4f} (available cases); null median / 95%: MCAR {med:.4f} / {q95:.4f}, MAR + IPW {med_ipw:.4f} / {q95_ipw:.4f}; "
      f"CF {cf_ac:.4f} (available cases) vs {cf_ipw:.4f} (IPW, same sample), CF null 95% {q95_cf:.4f}")
SVALS = {"rej0": rej0, "rej1": rej1, "inc_ac": inc_ac, "inc_ipw": inc_ipw, "q95": q95, "q95_ipw": q95_ipw, "med": med, "med_ipw": med_ipw, "q95_cf": q95_cf, "cf_ac": cf_ac, "cf_ipw": cf_ipw}
# ---------------------------------------------------------------- S3 planted walls
def chamber(f, g): return tuple(np.sign(f[list(I)].sum() - g[list(J)].sum()) for I, J in walls(len(f), len(g)))
f0, g0 = np.array([0.46, 0.54]), np.array([0.30, 0.38, 0.32]); m0 = tear_margin(f0, g0)
rates = {}
for nm2 in (0.5, 4, 16):
    N = int(math.ceil(nm2 / m0 ** 2)); wrong = 0
    for _ in range(300):
        fh = R.multinomial(N, f0) / N; gh = R.multinomial(N, g0) / N; wrong += chamber(fh, gh) != chamber(f0, g0)
    rates[nm2] = wrong / 300
check("S3 the chamber of planted margins is identified once N m_T^2 is large; errors are common when N m_T^2 is small",
      rates[16] < 0.05 and rates[0.5] > 0.1, f"m_T = {m0:.3f}; error rate by N m_T^2: {rates}")

# ---------------------------------------------------------------- P1 pilot numbers
import json
nh = json.load(open("pilot/out/nhanes_summary.json")); fh = json.load(open("pilot/out/fhub_summary.json"))
quoted = {
    "nhanes target bands": [nh["target"]["bands"][k] for k in ("0", "1", "2", "3")] == [[0.0, 1.0], [0.0, 0.8805], [0.1699, 0.4304], [0.3019, 0.3107]],
    "nhanes target risk": round(nh["target"]["risk"], 3) == 0.308 and nh["target"]["n"] == 462,
    "nhanes mean widths": [round(nh["ladder_width_mean"][k], 2) for k in ("0", "1", "2", "3")] == [0.95, 0.73, 0.33, 0.04],
    "nhanes revision": nh["revision"]["cells_new_estimate_inside_old_95band"] == 11 and len(nh["revision"]["side_changes"]) == 4,
    "nhanes inconsistency": round(nh["inconsistency"]["mixed_cycles"], 3) == 0.084 and round(nh["inconsistency"]["split_half_within_2009_10_median"], 3) == 0.179,
    "fhub coverage": fh["coverage80_by_h"] == {"1": 0.782, "2": 0.795, "3": 0.799, "4": 0.778},
    "fhub widening": fh["fraction_bands_wider_after_update"] == 0.263 and fh["fraction_new_band_not_inside_old"] == 0.642,
    "fhub tau": [fh["tear_rate_by_tau"][k]["tear_rate"] for k in ("[0.0, 0.5)", "[0.5, 1.0)", "[1.0, 2.0)", "[2.0, 4.0)", "[4.0, inf)")] == [0.332, 0.2, 0.111, 0.1, 0.031],
    "fhub auc": fh["auc_tear_from_small_tau"] == 0.778 and fh["auc_tear_from_small_relative_margin"] == 0.8,
    "fhub backfill": {k: fh["coverage80_h1_by_data_reliability"][k] for k in ("revised_le_25pct", "revised_gt_25pct", "n_bad", "n_good")} == {"revised_le_25pct": 0.793, "revised_gt_25pct": 0.432, "n_bad": 74, "n_good": 2278},
}
quoted.update({
    "nhanes basics": nh["n_adults"] == 10730 and round(nh["prevalence_Y"], 3) == 0.128,
    "nhanes target sampling band": [round(x, 2) for x in nh["target_sampling_band"]] == [0.25, 0.37],
    "nhanes target revision": [round(x, 2) for x in nh["target_revision"]] == [0.25, 0.36],
    "nhanes expected inside": round(nh["revision"]["expected_inside_if_no_change_cells_with_se"], 1) == 11.3 and nh["revision"]["cells_with_positive_se_both"] == 15
        and nh["revision"]["crossed_wall"] == 0,
    "nhanes out of sample": nh["out_of_sample_ladder"]["inside"] == {"0": 16, "1": 16, "2": 15, "3": 0} and nh["out_of_sample_ladder"]["interval_meets_band"] == {"0": 16, "1": 16, "2": 16, "3": 12},
    "nhanes simpson": nh["simpson"]["sign_change_in_one_stratum"][0] == {"exposure": "overweight", "stratifier": "55+", "rd_marginal": 0.035, "rd_Z0": 0.045, "se_Z0": 0.007, "rd_Z1": -0.033, "se_Z1": 0.018},
    "fhub basics": fh["n_forecasts"] == 9072 and fh["n_locations"] == 56 and fh["forecast_dates_used"] == ["2021-06-07", "2022-03-21"],
    "fhub tau sample": (fh["decisive_with_next"], fh["jumps_across_wall"], fh["decisive_bands"], fh["tau_sample_excluded_h4"], fh["tau_sample_excluded_zero_speed"]) == (3143, 3, 1954, 936, 253),
    "fhub backfill share": fh["backfill"]["state_weeks"] == 2408 and fh["backfill"]["frac_revised_more_than_25pct"] == 0.032
        and fh["coverage80_h1_by_data_reliability"]["ci_bad"] == [0.326, 0.546],
    "S1 size/power": (round(SVALS["rej0"], 3), round(SVALS["rej1"], 2)) == (0.048, 0.88),
    "S2 values": (round(SVALS["inc_ac"], 3), round(SVALS["cf_ac"], 3), round(SVALS["med"], 3), round(SVALS["q95"], 3), round(SVALS["cf_ipw"], 3),
                  round(SVALS["med_ipw"], 3), round(SVALS["q95_ipw"], 3)) == (0.439, 0.220, 0.021, 0.040, 0.021, 0.015, 0.042),
    "S2 CF null": round(SVALS["q95_cf"], 3) == 0.020,
    "S3 values": (round(m0, 3), round(rates[0.5], 2), rates[4], rates[16]) == (0.08, 0.24, 0.0, 0.0),
    "B7 slopes": [round(x, 2) for x in slopes] == [7.55, 12.16, 16.76],
    "B9 values": (round(noncomm, 3), round(residue, 3), round(odds(P0), 6)) == (0.12, 0.073, 9.0),
})
check("P1 the pilot numbers quoted in the paper match the saved outputs", all(quoted.values()), ", ".join(k for k, v in quoted.items() if not v) or "all match")

print(f"\n{sum(res)}/{len(res)} checks passed")
