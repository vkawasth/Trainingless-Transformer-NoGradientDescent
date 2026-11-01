"""Checks for Section "Strata and jumps" of "Probability, Context and Differentiation: The Layers" (Step 2).
  SJ1  support theorem (exact): supp pi^A = L_A = {l : pi_l p_l(A) > 0}; pi^A_l = pi_l p_l(A) / P(A); the odds identity; the rows of
       C_{L x A}(Lambda) are pi^A_l C_A(p_l); sources outside L_A have no row; forgetting commutes with conditioning
  SJ2  the first two questions (exact): pi^A = pi iff p_l(A) is the same for every source of positive weight; pooling with the old weights
       commutes with conditioning iff sum_l (pi^A_l - pi_l) C_A(p_l) = 0, which holds with changed weights when the C_A(p_l) coincide
       and is equivalent to equal p_l(A) when the C_A(p_l) are affinely independent
  SJ3  fibres (exact, and ranks): conditioning maps the fibre of forgetting over P onto the fibre over C_A(P) (explicit lift); dimensions
       (n-1)|supp P|, (n-1)|supp P n A| and (n-1)|supp P minus A| for the fibre, its image and the fibres of the induced map; every posterior
       provenance w in Delta_L is compatible with C_A(P); with known sources, independence of the C_A(p_l) implies independence of the
       p_l, and conditioning can destroy identifiability (two sources that agree on A)
  SJ4  the regular regime (floating point): along random smooth paths, |d/dt C_{LxA}(Lambda_t)|_1 <= 2 |dLambda_t/dt|_1 / Lambda_t(L x A);
       the change of the posterior provenance is bounded by the integral; source extinction: C_{LxA} is continuous while the extinct
       source's own conditional has no limit
  SJ5  the wall (exact): along polynomial paths the limit of C_{LxA}(Lambda_t) as t -> 0+ is the conditional of the leading jet on L x A,
       with error O(t); the attachment's example tends to (1/2, 1/2); a two-sided analytic passage is removable (equal one-sided limits);
       a kinked path jumps (one-sided jets differ)
  SJ6  near the wall (exact): on the near-miss path with gap eta, the posterior weight of source 1 moves from 3/4 to 1/4 over an interval
       of length exactly 4 eta, with slope -1/(4 eta) at the crossing; the replacement bound holds; eta = 0 is a jump from 1 to 0
"""
import itertools, os, sys, json, random
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import context as CT
from pce import jumps as JP
random.seed(20261031); rng = np.random.default_rng(20261031); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
X = ["a", "b", "c", "d"]; A = lambda x: x in ("a", "b")
def rlaw(support):
    w = [Fr(random.randint(1, 9)) for _ in support]; s = sum(w); return {x: v / s for x, v in zip(support, w)}
def family(n, zero_pi=True, zero_A=True):
    pis = [Fr(random.randint(0 if zero_pi else 1, 5)) for _ in range(n)]
    if sum(pis) == 0: pis[0] = Fr(1)
    s = sum(pis); pis = [p / s for p in pis]; ps = []
    for l in range(n):
        sup = random.choice([X, ["c", "d"], ["a", "c"], ["b", "d", "a"]]) if zero_A else X; ps.append(rlaw(sup))
    return pis, ps
def labelled(pis, ps): return {(l, x): pis[l] * v for l in range(len(ps)) for x, v in ps[l].items() if pis[l] * v != 0}

# SJ1
ok = True; cnt = 0
for _ in range(300):
    pis, ps = family(4); Lam = labelled(pis, ps); PA = JP.mass_A(Lam, A)
    if PA == 0: continue
    cnt += 1; C = JP.cond_lab(Lam, A); LA = {l for l in range(4) if pis[l] * CT.prob(ps[l], A) > 0}
    w = JP.weights(C); ok &= set(w) == LA == JP.support_A(Lam, A)
    for l in LA:
        ok &= w[l] == pis[l] * CT.prob(ps[l], A) / PA
        ok &= {x: v / w[l] for (k, x), v in C.items() if k == l} == CT.cond(ps[l], A)
        for j in LA: ok &= w[l] / w[j] == (pis[l] / pis[j]) * (CT.prob(ps[l], A) / CT.prob(ps[j], A))
    ok &= JP.forget(C) == CT.cond(JP.forget(Lam), A)
check("SJ1 support theorem: supp pi^A = {l : pi_l p_l(A) > 0}; Bayes weights and odds; rows; forgetting commutes (exact)", ok and cnt > 200, f"{cnt} families")

# SJ2
ok = True; eq = neq = 0
for _ in range(300):
    pis, ps = family(3, zero_pi=False, zero_A=False)
    if random.random() < 0.4:                        # force equal p_l(A): rescale the A-part of each source to mass 1/2
        new = []
        for p in ps:
            a = CT.prob(p, A); new.append({x: (v / a / 2 if A(x) else v / (1 - a) / 2) for x, v in p.items()})
        ps = new
    Lam = labelled(pis, ps); w = JP.weights(JP.cond_lab(Lam, A)); same = len({CT.prob(p, A) for p in ps}) == 1
    ok &= (all(w[l] == pis[l] for l in range(3)) == same); eq += same; neq += not same
    old = CT.pool(pis, [CT.cond(p, A) for p in ps]); new = CT.cond(CT.pool(pis, ps), A)
    M = np.array([[float(CT.cond(p, A).get(x, 0)) for x in X] for p in ps]); indep = np.linalg.matrix_rank(np.vstack([M[1] - M[0], M[2] - M[0]])) == 2
    if indep: ok &= (old == new) == same
p1 = {"a": Fr(1, 4), "b": Fr(1, 4), "c": Fr(1, 2)}; p2 = {"a": Fr(3, 8), "b": Fr(3, 8), "d": Fr(1, 4)}       # same C_A, different p(A)
pis = [Fr(1, 2), Fr(1, 2)]; w = JP.weights(JP.cond_lab(labelled(pis, [p1, p2]), A))
ok &= w[0] != pis[0] and CT.pool(pis, [CT.cond(p1, A), CT.cond(p2, A)]) == CT.cond(CT.pool(pis, [p1, p2]), A)
check("SJ2 weights preserved iff equal p_l(A); old-weight pooling commutes iff sum (pi^A - pi) C_A p_l = 0 (and can, with changed weights)", ok,
      f"{eq} equal, {neq} unequal; counterexample weights {w[0]}, {w[1]}")
OUT["SJ2_counter"] = [str(w[0]), str(w[1])]

# SJ3
ok = True; dims = []
for _ in range(60):
    n = random.randint(2, 4); sup = random.choice([X, ["a", "c", "d"], ["a", "b", "c"], ["b", "d"]]); P = rlaw(sup)
    if CT.prob(P, A) == 0: continue
    CP = CT.cond(P, A)
    Lp = {}                                            # a random labelled split of C_A(P)
    for x, v in CP.items():
        cut = sorted(Fr(random.randint(0, 10), 10) for _ in range(n - 1)); parts = [b - a for a, b in zip([Fr(0)] + cut, cut + [Fr(1)])]
        for l in range(n):
            if parts[l]: Lp[(l, x)] = v * parts[l]
    Lam = JP.fibre_lift(Lp, P, A, 0); ok &= JP.forget(Lam) == P and JP.cond_lab(Lam, A) == Lp and all(v >= 0 for v in Lam.values())
    # dimensions: affine span of random points of each fibre
    def fibre_points(cells, k=40):
        pts = []
        for _ in range(k):
            v = []
            for x in cells:
                r = rng.random(n); r = r / r.sum() * float(P[x]); v.extend(r)
            pts.append(v)
        pts = np.array(pts); return np.linalg.matrix_rank(pts[1:] - pts[0], 1e-9)
    sA = [x for x in sup if A(x)]; sN = [x for x in sup if not A(x)]
    d_full, d_A, d_off = fibre_points(sup), fibre_points(sA), fibre_points(sN)
    ok &= d_full == (n - 1) * len(sup) and d_A == (n - 1) * len(sA) and d_off == (n - 1) * len(sN) and d_full == d_A + d_off; dims.append((n, len(sup), d_full))
    wv = [Fr(random.randint(1, 5)) for _ in range(n)]; s = sum(wv); wv = [x / s for x in wv]                    # any posterior provenance
    Lw = {(l, x): wv[l] * v for l in range(n) for x, v in CP.items()}; Lam2 = JP.fibre_lift(Lw, P, A, 0)
    ok &= JP.forget(Lam2) == P and JP.weights(JP.cond_lab(Lam2, A)) == {l: wv[l] for l in range(n)}
# known sources: C_A-independence implies independence; destruction example
for _ in range(200):
    ps = [rlaw(random.choice([X, ["a", "b", "c"], ["a", "c", "d"]])) for _ in range(3)]
    M = np.array([[float(p.get(x, 0)) for x in X] for p in ps]); MC = np.array([[float(CT.cond(p, A).get(x, 0)) for x in X] for p in ps])
    if np.linalg.matrix_rank(MC) == 3: ok &= np.linalg.matrix_rank(M) == 3
Md = np.array([[float(p.get(x, 0)) for x in X] for p in (p1, p2)]); MdC = np.array([[float(CT.cond(p, A).get(x, 0)) for x in X] for p in (p1, p2)])
ok &= np.linalg.matrix_rank(Md) == 2 and np.linalg.matrix_rank(MdC) == 1
check("SJ3 conditioning maps fibres of forgetting onto fibres; dimensions split as on A + off A; any posterior provenance fits; identifiability only lost", ok,
      f"e.g. (n, |supp P|, dim) = {[(a, b, int(c)) for a, b, c in dims[:3]]}")

# SJ4
def C_float(L, Amask):
    a = (L * Amask).sum(); return L * Amask / a
ok = True; worst = 0.0
nL, nX = 3, 4; Am = np.array([[1, 1, 0, 0]] * nL, float)
for _ in range(100):
    c0 = rng.random((nL, nX)) + 0.05; c0 /= c0.sum(); c1 = rng.normal(size=(nL, nX)); c1 -= c1.mean(); c2 = rng.normal(size=(nL, nX)); c2 -= c2.mean()
    path = lambda t: c0 + 0.02 * t * c1 + 0.02 * t * t * c2
    ts = np.linspace(0, 1, 401); tv = 0.0; integ = 0.0
    for t0, t1 in zip(ts[:-1], ts[1:]):
        L0, L1 = path(t0), path(t1)
        if L0.min() < 0 or L1.min() < 0: break
        dC = np.abs(C_float(L1, Am) - C_float(L0, Am)).sum(); dL = np.abs(L1 - L0).sum(); a = min((L0 * Am).sum(), (L1 * Am).sum())
        ok &= dC <= 2 * dL / a * (1 + 1e-6) + 1e-12; worst = max(worst, dC / (2 * dL / a))
        tv += np.abs(C_float(L1, Am).sum(axis=1) - C_float(L0, Am).sum(axis=1)).sum(); integ += 2 * dL / a
    ok &= tv <= integ + 1e-12
# source extinction: source 1's mass on A is t^2 (1 +- sin(1/t)) / 2, source 2 keeps 1/4 on A
def ext(t):
    s = np.sin(1 / t); return np.array([[t * t * (1 + s) / 2, t * t * (1 - s) / 2, 0.5 - t * t, 0], [0.125, 0.125, 0, 0.25]])
lim = np.array([[0, 0, 0, 0], [0.5, 0.5, 0, 0]], float); shares = []
for t in np.geomspace(1e-1, 1e-6, 400):
    L = ext(t); ok &= np.abs(C_float(L, Am[:2]) - lim).sum() <= 16 * t * t + 1e-12; shares.append(L[0, 0] / (L[0, 0] + L[0, 1]))
ok &= min(shares) < 0.01 and max(shares) > 0.99
check("SJ4 regular regime: |dC| <= 2|dLambda|/P(A) on random paths; provenance change bounded by the integral; extinction is continuous", ok,
      f"largest ratio to the bound {worst:.3f}; extinct source's own a-share ranges over [{min(shares):.3f}, {max(shares):.3f}] as t -> 0")

# SJ5
ok = True; cells = [(l, x) for l in range(2) for x in X]
def rand_path_coeffs(m):
    """c_0 vanishes on L x A, c_1..c_{m-1} vanish on L x A, c_m >= 0 and nonzero on L x A; off A anything keeping positivity"""
    cs = []
    off = [(l, x) for (l, x) in cells if not A(x)]; on = [(l, x) for (l, x) in cells if A(x)]
    base = rlaw(off); cs.append(dict(base))
    for j in range(1, 4):
        c = {}
        if j == m:
            for k in on:
                if random.random() < 0.7: c[k] = Fr(random.randint(1, 5))
            if not c: c[on[0]] = Fr(1)
        elif j > m:
            for k in on: c[k] = Fr(random.randint(-3, 3))
        s = sum(c.values()); k0 = off[0]; c[k0] = c.get(k0, Fr(0)) - s           # zero-sum correction off A
        cs.append(c)
    return cs
errs = []
for _ in range(200):
    m = random.randint(1, 3); cs = rand_path_coeffs(m); mm, limv = JP.jet_limit(cs, A); ok &= mm == m
    for t in (Fr(1, 10 ** 4), Fr(1, 10 ** 8)):
        C = JP.cond_lab(JP.poly_eval(cs, t), A); err = sum(abs(C.get(k, Fr(0)) - limv.get(k, Fr(0))) for k in set(C) | set(limv))
        errs.append(float(err / t)); ok &= err <= 1000 * t
pt = lambda t: {"a": t, "b": t * (1 - t), "c": 1 - t - t * (1 - t)}
ok &= CT.cond(pt(Fr(1, 10 ** 9)), A)["a"] - Fr(1, 2) < Fr(1, 10 ** 8) and JP.jet_limit([{"c": Fr(1)}, {"a": Fr(1), "b": Fr(1), "c": Fr(-2)}, {"b": Fr(-1), "c": Fr(1)}], A)[1] == {"a": Fr(1, 2), "b": Fr(1, 2)}
# two-sided analytic passage: cells on A = t^2 (alpha + beta t); removable
two = [{(0, "c"): Fr(1, 2), (1, "c"): Fr(1, 2)}, {}, {(0, "a"): Fr(2), (1, "b"): Fr(1), (0, "c"): Fr(-3)}, {(0, "a"): Fr(1), (1, "b"): Fr(-1)}]
left = JP.cond_lab(JP.poly_eval(two, Fr(-1, 10 ** 9)), A); right = JP.cond_lab(JP.poly_eval(two, Fr(1, 10 ** 9)), A)
ok &= abs(left[(0, "a")] - right[(0, "a")]) < Fr(1, 10 ** 8) and abs(left[(0, "a")] - Fr(2, 3)) < Fr(1, 10 ** 8)
# kinked path (eta = 0): source 1 explains A before the crossing, source 2 after
kink = lambda t, eta: {(0, "a"): eta + max(-t, Fr(0)), (0, "c"): Fr(1, 2) - eta - max(-t, Fr(0)), (1, "b"): eta + max(t, Fr(0)), (1, "c"): Fr(1, 2) - eta - max(t, Fr(0))}
lw = JP.weights(JP.cond_lab(kink(Fr(-1, 10 ** 9), Fr(0)), A)); rw = JP.weights(JP.cond_lab(kink(Fr(1, 10 ** 9), Fr(0)), A))
ok &= lw == {0: Fr(1)} and rw == {1: Fr(1)}
check("SJ5 the wall: limit = conditional of the leading jet (error O(t)); attachment example -> (1/2,1/2); analytic passage removable; a kink jumps", ok,
      f"max error/t {max(errs):.2f}; kinked path: posterior provenance {dict(lw)} -> {dict(rw)}")

# SJ6
ok = True; rows = []
for eta in (Fr(1, 10), Fr(1, 40), Fr(1, 100)):
    w1 = lambda t: JP.weights(JP.cond_lab(kink(t, eta), A)).get(0, Fr(0))
    ok &= w1(-2 * eta) == Fr(3, 4) and w1(2 * eta) == Fr(1, 4) and w1(Fr(0)) == Fr(1, 2)
    h = Fr(1, 10 ** 12); ok &= abs((w1(h) - w1(Fr(0))) / h + 1 / (4 * eta)) < Fr(1, 10 ** 6) and abs((w1(Fr(0)) - w1(-h)) / h + 1 / (4 * eta)) < Fr(1, 10 ** 6)
    # replacement bound over [-1/4, 1/4]: |Delta pi^A|_1 <= int 2 |dLambda|_1 / P_t(A) dt, |dLambda/dt|_1 = 2, P_t(A) = 2 eta + |t|
    T = Fr(1, 4); dpi = 2 * abs(w1(-T) - w1(T)); bound = 2 * 2 * 2 * float(np.log(float((2 * eta + T) / (2 * eta))))
    ok &= float(dpi) <= bound; rows.append((str(eta), str(w1(-T)), str(w1(T)), round(bound, 2)))
check("SJ6 near the wall: pi^A_1 goes 3/4 -> 1/4 over length 4 eta, slope -1/(4 eta); replacement bound holds; eta = 0 jumps (SJ5)", ok, f"(eta, pi^A_1(-1/4), pi^A_1(1/4), bound) {rows}")
OUT["SJ6"] = rows
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "jumps.json"), "w"))
print(f"{sum(results)}/{len(results)} passed")
