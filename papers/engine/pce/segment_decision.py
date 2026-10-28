"""Decisions on the hiding segment (lab paper, Theorem D). Two latent types on m binary variables, pair cover.

Expected utilities along the segment are polynomials in r (Theorem A). For utilities on a variable set T, with eps(1) = +1, eps(0) = -1
and the exact centred-moment expansion of a binary law,
    P_T(x) = sum_{S subset T} m_S prod_{i in S} eps(x_i) prod_{i in T minus S} p_i(x_i),   m_empty = 1, m_{i} = 0, m_S = v_S s_|S|(r),
so U_a(r) = sum_S c_a(S) v_S s_|S|(r) with c_a(S) = sum_x u_a(x) prod_{S} eps(x_i) prod_{T minus S} p_i(x_i). For |T| = 3 the slope is
beta_a = v_T * sum_x u_a(x) chi(x), chi(x) = prod eps(x_i): the top Walsh coefficient of the utility.

Rules on an evidence interval I (in r):
  set_rule(acts, I, u0)        the act admissible on I (worst value >= u0) with the largest worst value; None = abstain. Depends on I only.
  nominal_rule(acts, rhat, R, u0)   info-gap: the act with the largest robustness horizon around a nominal rhat (ties: nominal value).
  settle_radius(acts, r0, R, u0)    the largest h with set_rule([r0 - h, r0 + h] cap R) equal to the oracle act for every smaller h.
  settle_constraints(acts, r0, u0)  for affine utilities: the linear constraints (g, g', gamma) that define settling (see Theorem D(4)).
"""
from __future__ import annotations
import itertools
import numpy as np
from numpy.polynomial import polynomial as Pn
from .lab import segment, r_of_t

EPS = {0: -1.0, 1: 1.0}


def s_poly(k):
    """s_k(r) as polynomial coefficients (low to high): s_0 = 1 (empty set), s_2 = 1, s_3 = r, s_(k+1) = r s_k + s_(k-1)"""
    if k == 0: return np.array([1.0])
    if k == 1: return np.array([0.0])
    a, b = np.array([1.0]), np.array([0.0, 1.0])            # s_2, s_3
    if k == 2: return a
    for _ in range(k - 3): a, b = b, Pn.polyadd(Pn.polymulx(b), a)
    return b


def U_poly(mu, v, u, T):
    """coefficients (low to high, in r) of E[u] along the segment; u maps a tuple on T to a number"""
    T = tuple(T); out = np.array([0.0])
    for k in range(len(T) + 1):
        if k == 1: continue
        for S in itertools.combinations(T, k):
            c = 0.0
            for x in itertools.product((0, 1), repeat=len(T)):
                xd = dict(zip(T, x))
                c += u(x) * np.prod([EPS[xd[i]] for i in S]) * np.prod([mu[i] if xd[i] else 1 - mu[i] for i in T if i not in S])
            vS = np.prod([v[i] for i in S]) if S else 1.0
            out = Pn.polyadd(out, c * vS * s_poly(k))
    return Pn.polytrim(out, 1e-15) if np.any(out) else np.array([0.0])


def walsh_top(u, k):
    return float(sum(u(x) * np.prod([EPS[b] for b in x]) for x in itertools.product((0, 1), repeat=k)))


def r_walls(mu, v):
    lo, hi = segment(mu, v); return float(r_of_t(lo)), float(r_of_t(hi))


def worst(p, I):
    """min of the polynomial p on I = (a, b): endpoints and interior critical points"""
    a, b = I; cands = [a, b]
    if len(p) > 2:
        for z in Pn.polyroots(Pn.polyder(p)):
            if abs(z.imag) < 1e-12 and a < z.real < b: cands.append(z.real)
    return float(min(Pn.polyval(c, p) for c in cands))


def clip(I, R):
    return (max(I[0], R[0]), min(I[1], R[1]))


def set_rule(polys, I, u0):
    vals = [worst(p, I) for p in polys]; adm = [k for k, x in enumerate(vals) if x >= u0 - 1e-12]
    return None if not adm else max(adm, key=lambda k: (vals[k], -k))


def horizon(p, rhat, R, u0, tol=1e-12):
    """info-gap robustness: sup{h : worst(p, [rhat - h, rhat + h] cap R) >= u0}; inf when the act is admissible on all of R"""
    if Pn.polyval(rhat, p) < u0: return -np.inf
    if worst(p, R) >= u0: return np.inf
    lo, hi = 0.0, max(rhat - R[0], R[1] - rhat)
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if worst(p, clip((rhat - m, rhat + m), R)) >= u0: lo = m
        else: hi = m
        if hi - lo < tol: break
    return lo


def nominal_rule(polys, rhat, R, u0):
    h = [horizon(p, rhat, R, u0) for p in polys]; nv = [Pn.polyval(rhat, p) for p in polys]
    ok = [k for k in range(len(polys)) if h[k] > -np.inf]
    return None if not ok else max(ok, key=lambda k: (h[k], nv[k], -k))


def oracle(polys, r0, u0):
    return set_rule(polys, (r0, r0), u0)


def settle_radius(polys, r0, R, u0, tol=1e-12):
    """largest h such that set_rule on [r0 - h, r0 + h] cap R equals the oracle act for all h' < h (bisection on a fine scan)"""
    a0 = oracle(polys, r0, u0); hmax = max(r0 - R[0], R[1] - r0)
    hs = np.linspace(0, hmax, 4001)[1:]; bad = [h for h in hs if set_rule(polys, clip((r0 - h, r0 + h), R), u0) != a0]
    if not bad: return np.inf
    lo, hi = (hs[hs < bad[0]][-1] if np.any(hs < bad[0]) else 0.0), bad[0]
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if set_rule(polys, clip((r0 - m, r0 + m), R), u0) == a0: lo = m
        else: hi = m
        if hi - lo < tol: break
    return lo


def settle_constraints(polys, r0, u0):
    """affine utilities, interval [c - h, c + h] inside R: set_rule = oracle a iff every constraint g + g1 (c - r0) - gamma h >= 0 holds
    (up to ties). Constraint 0: worst value of a >= u0. Constraint b: worst value of a >= worst value of b."""
    a = oracle(polys, r0, u0)
    if a is None: return a, []
    lin = lambda p: (Pn.polyval(r0, p), (p[1] if len(p) > 1 else 0.0))
    Ua, ba = lin(polys[a]); out = [(Ua - u0, ba, abs(ba))]
    for b, p in enumerate(polys):
        if b == a: continue
        Ub, bb = lin(p); out.append((Ua - Ub, ba - bb, abs(ba) - abs(bb)))
    return a, out


def settle_radius_affine(polys, r0, u0):
    a, cons = settle_constraints(polys, r0, u0)
    rad = [g / gam for g, _, gam in cons if gam > 0]
    return min(rad) if rad else np.inf


def settle_probability(cons, tau, z, n):
    """P(all constraints hold) at c = r0 + tau Z / sqrt(n), h = z tau / sqrt(n), Z standard normal (closed form)"""
    from scipy.stats import norm
    lo, hi = -np.inf, np.inf; s = tau / np.sqrt(n)
    for g, g1, gam in cons:
        rhs = gam * z * s - g                                   # need g1 s Z >= rhs
        if abs(g1) < 1e-15:
            if rhs > 0: return 0.0
            continue
        bnd = rhs / (g1 * s)
        if g1 > 0: lo = max(lo, bnd)
        else: hi = min(hi, bnd)
    return float(max(0.0, norm.cdf(hi) - norm.cdf(lo)))


# ---- the m = 3 estimator used by Study 11b (closed form; pair windows and the triple window sampled independently, n each) ----
def v_closed(M):
    """v for m = 3 from the three pair covariances: |v_i| = sqrt|M_ij M_ik / M_jk|; signs from M; largest |v_i| positive"""
    a = np.array([np.sqrt(abs(M[0, 1] * M[0, 2] / M[1, 2])), np.sqrt(abs(M[0, 1] * M[1, 2] / M[0, 2])), np.sqrt(abs(M[0, 2] * M[1, 2] / M[0, 1]))])
    k = int(np.argmax(a)); s = np.ones(3)
    for i in range(3):
        if i != k: s[i] = np.sign(M[min(i, k), max(i, k)]) or 1.0
    return s * a


def tau_r(P):
    """asymptotic sd of rhat = mhat_S / vhat_S (times sqrt n) for m = 3: influence functions of the centred triple moment (own means) and of
    log|v_S| = (1/2) sum log|M_ij| (each pair window with its own means)"""
    mu = np.array([P.sum(axis=tuple(j for j in range(3) if j != i))[1] for i in range(3)]); M = np.zeros((3, 3)); xs = list(itertools.product((0, 1), repeat=3))
    for i, j in itertools.combinations(range(3), 2):
        M[i, j] = M[j, i] = sum(P[x] * (x[i] - mu[i]) * (x[j] - mu[j]) for x in xs)
    mS = sum(P[x] * np.prod([x[i] - mu[i] for i in range(3)]) for x in xs); v = v_closed(M); vS = np.prod(v); r = mS / vS
    IF = lambda x: np.prod([x[i] - mu[i] for i in range(3)]) - mS - sum(M[j, k] * (x[i] - mu[i]) for i, (j, k) in zip(range(3), ((1, 2), (0, 2), (0, 1))))
    var_m = sum(P[x] * IF(x) ** 2 for x in xs)
    var_logv = 0.25 * sum(sum(P[x] * ((x[i] - mu[i]) * (x[j] - mu[j]) - M[i, j]) ** 2 for x in xs) / M[i, j] ** 2 for i, j in itertools.combinations(range(3), 2))
    return float(np.sqrt(var_m / vS ** 2 + r ** 2 * var_logv)), float(r), v, mu


WINDOWS = ((0, 1), (0, 2), (1, 2), (0, 1, 2))


def sample_tables(P, n, rng):
    """three pair windows and the triple window, n draws each (independent samples)"""
    tabs = {}
    for S in WINDOWS:
        T = P.sum(axis=tuple(j for j in range(3) if j not in S)).ravel(); tabs[S] = (rng.multinomial(n, T / T.sum()) / n).reshape((2,) * len(S))
    return tabs


def exact_tables(P):
    return {S: P.sum(axis=tuple(j for j in range(3) if j not in S)) for S in WINDOWS}


def from_tables(tabs):
    """the regulator's estimates: mu pooled from the pair windows, v from the pair covariances (each window with its own means), rhat from
    the triple window's centred moment, and the plug-in tauhat; None if the pair tables admit no two-type explanation"""
    mu = np.zeros(3); c = np.zeros(3); M = np.zeros((3, 3))
    for (i, j) in WINDOWS[:3]:
        T = tabs[(i, j)]; pi, pj = T[1].sum(), T[:, 1].sum(); mu[i] += pi; mu[j] += pj; c[i] += 1; c[j] += 1
        M[i, j] = M[j, i] = T[1, 1] - pi * pj
    mu /= c
    if min(abs(M[0, 1]), abs(M[0, 2]), abs(M[1, 2])) < 1e-12 or M[0, 1] * M[0, 2] * M[1, 2] <= 0: return None
    v = v_closed(M); vS = float(np.prod(v)); P3 = tabs[(0, 1, 2)]; xs = list(itertools.product((0, 1), repeat=3))
    m3 = [P3.sum(axis=tuple(j for j in range(3) if j != i))[1] for i in range(3)]
    mS = sum(P3[x] * np.prod([x[i] - m3[i] for i in range(3)]) for x in xs); rhat = mS / vS
    Mt = {(i, j): sum(P3[x] * (x[i] - m3[i]) * (x[j] - m3[j]) for x in xs) for i, j in itertools.combinations(range(3), 2)}
    other = ((1, 2), (0, 2), (0, 1))
    IF = lambda x: np.prod([x[i] - m3[i] for i in range(3)]) - mS - sum(Mt[other[i]] * (x[i] - m3[i]) for i in range(3))
    var_m = sum(P3[x] * IF(x) ** 2 for x in xs); var_logv = 0.0
    for (i, j) in WINDOWS[:3]:
        T = tabs[(i, j)]; pi, pj = T[1].sum(), T[:, 1].sum()
        var_logv += 0.25 * sum(T[a, b] * ((a - pi) * (b - pj) - M[i, j]) ** 2 for a in (0, 1) for b in (0, 1)) / M[i, j] ** 2
    tauhat = float(np.sqrt(var_m / vS ** 2 + rhat ** 2 * var_logv))
    try: R = r_walls(mu, v)
    except Exception: return None
    if not (np.isfinite(R[1]) and R[1] > R[0]): return None
    return mu, v, float(rhat), tauhat, R


def estimate(P, n, rng):
    e = from_tables(sample_tables(P, n, rng)); return None if e is None else e[:4]
