"""check_tropical.py -- exact checks for "From barycentre fibres to secant geometry: the positive valuation map".

Field: real Puiseux series K = R{{t}}; we work with exponents in (1/N)Z by writing t = s^N, so an element is a finite Laurent
polynomial in s with rational coefficients (dict exponent -> Fraction). Only + and x are needed for the exact identities; valuations
of quotients use leading terms. Positive cone: leading coefficient > 0.

  T1  D_T (normalised min-plus distributions) satisfies the monad laws
  T2  val is a monad morphism D_{K>0} => D_T: val(eta x) = eta_T x and val(mu Pi) = mu_T(D(val) Pi) exactly (no cancellation)
  T3  fibre lifting: every point of the tropical k-fibre over val(rho) lifts to an exact positive k-decomposition of rho
  T4  Berkovich square on the skeleton: -log of the Gauss norm of each mixture coordinate at s(v) equals mu_T(v)
  T5  the Bayes posterior tropicalises to the hard (argmin) assignment: val P(i|x) = w_i + u_i(x) - v(x) >= 0, = 0 iff i attains
  T6  tropical vs classical secant dimension (generic Jacobian rank), 3x3 independence model, k = 1, 2, 3
  T7  same for the binary 2x2x2 independence model, k = 1, 2
  T8  walls: inside one assignment cell mu_T is affine; crossing a tie changes the assignment and the linear piece
"""
from fractions import Fraction as F
import random, itertools, hashlib, json, sys
import numpy as np

OUT = {}
def check(name, cond, info=""):
    OUT[name] = bool(cond); print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""))
rng = random.Random(20261013)
INF = float("inf")

# ---------------------------------------------------------------- Puiseux arithmetic (t = s^N, integer exponents in s)
def padd(a, b):
    c = dict(a)
    for e, v in b.items():
        c[e] = c.get(e, 0) + v
        if c[e] == 0: del c[e]
    return c
def pmul(a, b):
    c = {}
    for e1, v1 in a.items():
        for e2, v2 in b.items():
            c[e1 + e2] = c.get(e1 + e2, 0) + v1 * v2
    return {e: v for e, v in c.items() if v != 0}
def psub(a, b): return padd(a, {e: -v for e, v in b.items()})
def val(a): return min(a) if a else INF
def lead(a): return a[min(a)]
def positive(a): return bool(a) and lead(a) > 0
def mono(e, c=F(1)): return {e: F(c)}
def rand_pos(e, extra=2):
    a = mono(e, F(rng.randint(1, 5)))
    for _ in range(rng.randint(0, extra)): a = padd(a, mono(e + rng.randint(1, 4), F(rng.randint(-3, 5))))
    return a

# ---------------------------------------------------------------- T1: the tropical monad
def norm_T(v):
    m = min(v.values()); return {k: x - m for k, x in v.items()}
def eta_T(x, X): return tuple(0 if y == x else INF for y in X)
def mu_T(V, X):  # V: dict {u (tuple over X): weight val}
    return tuple(min(w + u[i] for u, w in V.items()) for i in range(len(X)))
X = (0, 1, 2)
def rand_T():
    v = [rng.randint(0, 4) if rng.random() < .8 else INF for _ in X]
    if min(v) == INF: v[0] = 0
    m = min(v); return tuple(x - m for x in v)
okT1 = True
for _ in range(300):
    V = {}
    for _ in range(rng.randint(1, 3)): V[rand_T()] = rng.randint(0, 3)
    m = min(V.values()); V = {u: w - m for u, w in V.items()}
    okT1 &= min(mu_T(V, X)) == 0                                   # normalised
    u = rand_T()
    okT1 &= mu_T({eta_T(x, X): u[x] for x in X if u[x] < INF}, X) == u      # mu . D(eta) = id
    okT1 &= mu_T({u: 0}, X) == u                                         # mu . eta = id
    # associativity on a random level-3 element: mu(mu_D W) = mu(D mu W)
    W = {}
    for _ in range(rng.randint(1, 3)):
        Vk = {}
        for _ in range(rng.randint(1, 3)): Vk[rand_T()] = rng.randint(0, 3)
        mk = min(Vk.values()); W[tuple(sorted({u: w - mk for u, w in Vk.items()}.items()))] = rng.randint(0, 3)
    mw = min(W.values()); W = {k: w - mw for k, w in W.items()}
    flat = {}
    for Vt, a in W.items():
        for u_, b in Vt: flat[u_] = min(flat.get(u_, INF), a + b)
    lhs = mu_T(flat, X)
    inner = {}
    for Vt, a in W.items():
        m_ = mu_T(dict(Vt), X); inner[m_] = min(inner.get(m_, INF), a)
    okT1 &= lhs == mu_T(inner, X)
check("T1 D_T is a monad (unit laws, associativity, normalisation)", okT1)

# ---------------------------------------------------------------- T2: val is a monad morphism (unnormalised cone; normalisation commutes)
okT2 = True
for _ in range(300):
    k = rng.randint(1, 4)
    comps = [{x: rand_pos(rng.randint(0, 5)) for x in X} for _ in range(k)]
    wts = [rand_pos(rng.randint(0, 4)) for _ in range(k)]
    mix = {x: {} for x in X}
    for w, r in zip(wts, comps):
        for x in X: mix[x] = padd(mix[x], pmul(w, r[x]))
    okT2 &= all(positive(mix[x]) for x in X)
    okT2 &= tuple(val(mix[x]) for x in X) == tuple(min(val(w) + val(r[x]) for w, r in zip(wts, comps)) for x in X)
    # normalisation: val(r/sum r) = val r - min val r  (leading terms divide)
    tot = {}
    for x in X: tot = padd(tot, comps[0][x])
    okT2 &= val(tot) == min(val(comps[0][x]) for x in X)
check("T2 val(mixture) = min-plus mixture of vals (positive cone, no cancellation); normalisation commutes", okT2)

# ---------------------------------------------------------------- T3: fibre lifting
okT3 = True; n_ties = 0
for _ in range(300):
    k = rng.randint(1, 3)
    u = [tuple(rng.randint(0, 6) for _ in X) for _ in range(k)]
    u = [tuple(c - min(ui) for c in ui) for ui in u]
    w = [rng.randint(0, 4) for _ in range(k)]; m = min(w); w = [a - m for a in w]
    v = tuple(min(w[i] + u[i][x] for i in range(k)) for x in X)
    # a positive rho (unnormalised is enough for the lifting identity) with val rho = v, generic higher terms
    rho = {x: rand_pos(v[x]) for x in X}
    q = [[None] * len(X) for _ in range(k)]
    for x in X:
        att = [i for i in range(k) if w[i] + u[i][x] == v[x]]; n_ties += len(att) > 1
        rest = {}
        for i in range(k):
            if i not in att:
                q[i][x] = mono(w[i] + u[i][x], F(1, rng.randint(1, 4)))
                rest = padd(rest, q[i][x])
        rem = psub(rho[x], rest)
        # split the remainder among attainers: (1/|att|) each, except a perturbation that keeps leading coefficients positive
        share = {e: c / len(att) for e, c in rem.items()}
        for i in att: q[i][x] = share
    for x in X:
        tot = {}
        for i in range(k): tot = padd(tot, q[i][x])
        okT3 &= tot == rho[x]                                           # exact decomposition
        okT3 &= all(positive(q[i][x]) and val(q[i][x]) == w[i] + u[i][x] for i in range(k))
    for i in range(k):
        wi = {}
        for x in X: wi = padd(wi, q[i][x])
        okT3 &= positive(wi) and val(wi) == w[i]                         # weight valuation (min u_i = 0)
        okT3 &= all(val(q[i][x]) - val(wi) == u[i][x] for x in X)       # component valuation u_i = val(q_i/w_i)
check("T3 every tropical fibre point lifts to an exact positive decomposition of rho", okT3, f"{n_ties} coordinates with ties")

# ---------------------------------------------------------------- T4: Gauss norms on the skeleton
import sympy as sp
okT4 = True
for _ in range(60):
    k = rng.randint(1, 3)
    W = sp.symbols(f"w0:{k}"); R = [[sp.Symbol(f"r{i}_{x}") for x in X] for i in range(k)]
    vw = [F(rng.randint(-3, 3), rng.randint(1, 3)) for _ in range(k)]
    vr = [[F(rng.randint(-3, 3), rng.randint(1, 3)) for _ in X] for _ in range(k)]
    vmap = {**{W[i]: vw[i] for i in range(k)}, **{R[i][x]: vr[i][x] for i in range(k) for x in X}}
    for x in X:
        poly = sp.Poly(sp.expand(sum(W[i] * R[i][x] for i in range(k))), *W, *[R[i][y] for i in range(k) for y in X])
        gens = poly.gens
        # Gauss norm at s(v): max_m |a_m| e^{-<m,v>}; |a_m| = 1 for nonzero integers (residue characteristic 0)
        neglog = min(sum(F(e) * vmap[g] for e, g in zip(m, gens)) for m, c in poly.terms() if c != 0)
        okT4 &= neglog == min(vw[i] + vr[i][x] for i in range(k))
check("T4 -log Gauss norm of each mixture coordinate at the skeleton point s(v) equals mu_T(v)", okT4)

# ---------------------------------------------------------------- T5: posterior -> hard assignment
okT5 = True
for _ in range(300):
    k = rng.randint(2, 4)
    comps = [{x: rand_pos(rng.randint(0, 5)) for x in X} for _ in range(k)]
    wts = [rand_pos(rng.randint(0, 4)) for _ in range(k)]
    for x in X:
        terms = [pmul(wts[i], comps[i][x]) for i in range(k)]
        tot = {}
        for tm in terms: tot = padd(tot, tm)
        vals = [val(tm) - val(tot) for tm in terms]                    # val of posterior P(i|x) = term / total
        okT5 &= min(vals) == 0 and all(v_ >= 0 for v_ in vals)
        okT5 &= set(i for i in range(k) if vals[i] == 0) == set(i for i in range(k) if val(wts[i]) + val(comps[i][x]) == val(tot))
        # leading coefficient of the posterior on the attainers sums to 1: the zero-temperature posterior is uniform-on-ties weighted
        lc = sum(lead(terms[i]) for i in range(k) if vals[i] == 0)
        okT5 &= lc == lead(tot)
check("T5 the Bayes posterior tropicalises to the argmin (MAP) assignment; leading coefficients split ties", okT5)

# ---------------------------------------------------------------- T6/T7: dimensions
def trop_rank(fun, npar, trials=40):
    best = 0
    for _ in range(trials):
        p = np.array([rng.uniform(-3, 3) for _ in range(npar)]); h = 1e-6
        J = np.array([(np.array(fun(p + h * e)) - np.array(fun(p - h * e))) / (2 * h) for e in np.eye(npar)]).T
        best = max(best, np.linalg.matrix_rank(J, tol=1e-6))
    return best
def class_rank(fun, npar, trials=10):
    best = 0
    for _ in range(trials):
        p = np.array([rng.uniform(.2, 2) for _ in range(npar)]); h = 1e-6
        J = np.array([(np.array(fun(p + h * e)) - np.array(fun(p - h * e))) / (2 * h) for e in np.eye(npar)]).T
        best = max(best, np.linalg.matrix_rank(J, tol=1e-7))
    return best
dims = {}
for k in (1, 2, 3):
    npar = k * 7  # w_k, a_k in R^3, b_k in R^3
    def ft(p, k=k):
        return [min(p[7 * c] + p[7 * c + 1 + i] + p[7 * c + 4 + j] for c in range(k)) for i in range(3) for j in range(3)]
    def fc(p, k=k):
        return [sum(p[7 * c] * p[7 * c + 1 + i] * p[7 * c + 4 + j] for c in range(k)) for i in range(3) for j in range(3)]
    dims[k] = (trop_rank(ft, npar), class_rank(fc, npar))
check("T6 3x3 independence: tropical and classical affine mixture dimensions agree (5, 8, 9)",
      dims == {1: (5, 5), 2: (8, 8), 3: (9, 9)}, str(dims))
dims2 = {}
for k in (1, 2):
    npar = k * 7
    idx = list(itertools.product((0, 1), repeat=3))
    def ft(p, k=k):
        return [min(p[7 * c] + p[7 * c + 1 + a] + p[7 * c + 3 + b] + p[7 * c + 5 + d] for c in range(k)) for a, b, d in idx]
    def fc(p, k=k):
        return [sum(p[7 * c] * p[7 * c + 1 + a] * p[7 * c + 3 + b] * p[7 * c + 5 + d] for c in range(k)) for a, b, d in idx]
    dims2[k] = (trop_rank(ft, npar), class_rank(fc, npar))
check("T7 2x2x2 independence: dimensions (tropical, classical)", dims2[1] == (4, 4) and dims2[2][1] == 8, str(dims2))
T7_trop2 = dims2[2][0]

# ---------------------------------------------------------------- T8: walls
okT8 = True; crossings = 0
for _ in range(200):
    k = 2
    u = [np.array([rng.randint(0, 5) for _ in X], float) for _ in range(k)]
    w = np.array([0.0, rng.uniform(-2, 2)])
    def v_of(w1): return np.array([min(0 + u[0][x], w1 + u[1][x]) for x in X])
    def assign(w1): return tuple(tuple(i for i, s in enumerate((u[0][x], w1 + u[1][x])) if abs(s - min(u[0][x], w1 + u[1][x])) < 1e-12) for x in X)
    ties = sorted(set(u[0][x] - u[1][x] for x in X))
    pts = [ties[0] - 1] + [(a + b) / 2 for a, b in zip(ties, ties[1:])] + [ties[-1] + 1]
    for p0 in pts:   # affine inside a cell: midpoint rule
        a, b = p0 - 0.1, p0 + 0.1
        okT8 &= np.allclose(v_of(p0), (v_of(a) + v_of(b)) / 2)
    for t in ties:   # crossing a tie changes the assignment
        crossings += assign(t - 1e-3) != assign(t + 1e-3)
        okT8 &= assign(t - 1e-3) != assign(t + 1e-3)
check("T8 mu_T is affine on assignment cells and the assignment changes across every tie wall", okT8, f"{crossings} wall crossings")

npass = sum(OUT.values()); print(f"\n{npass}/{len(OUT)} checks pass; tropical dim of 2-mixtures of 2x2x2 = {T7_trop2}")
print("results SHA-256:", hashlib.sha256(json.dumps(OUT, sort_keys=True).encode()).hexdigest())
sys.exit(0 if npass == len(OUT) else 1)
