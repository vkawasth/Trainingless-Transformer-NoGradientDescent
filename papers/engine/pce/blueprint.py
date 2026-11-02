"""A computational blueprint connecting contextuality to categorical probability (layers paper, Section "A computational blueprint").
New module. One synthetic world: three binary readings (X0, X1, X2) per record, two sources, a flag event.

  source1(kappa)                  a coherent two-type source: w = 0.4, types a1 = 1/2 + kappa d, a2 = 1/2 - kappa d (d = (0.3, 0.2, 0.25))
  source2(kappa)                  a stale source with source 1's single margins: (X0, X2) counter-monotone (they disagree as often as
                                  the margins allow), X1 independent
  window_tables(theta, kappa)     pair tables measured in three contexts: windows 01 and 12 from source 1, window 02 from the mixture
                                  (1 - theta) source1 + theta source2 (context-dependent provenance)
  glues(tables)                   linear programme: is there a joint law on {0,1}^3 with these pair tables? (also returns the slack)
  cells_lab()                     the 16 cells (l, x0, x1, x2); marginal_matrix_lab(level) the observation map at each level of the
                                  exclusion filtration: pooled singles, pooled pairs, labelled pairs, labelled full
  twotype_law(w, a1, a2)          the law of a two-type latent class model on {0,1}^3
  fibre_points(P, starts, rng)    explanations (w, a1, a2) of P by multi-start least squares, clustered (b0 of a finite fibre)
  FLAG_CELLS, FLAG_GAP            the flag event's labelled cells as polynomials in (s, t) and the decision gap on them
"""
from __future__ import annotations
import itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog, least_squares

CELLS3 = list(itertools.product((0, 1), repeat=3)); D_SEP = np.array([0.3, 0.2, 0.25])


def twotype_law(w, a1, a2):
    P = np.zeros(8)
    for k, x in enumerate(CELLS3):
        P[k] = w * np.prod([a1[i] if x[i] else 1 - a1[i] for i in range(3)]) + (1 - w) * np.prod([a2[i] if x[i] else 1 - a2[i] for i in range(3)])
    return P


def source1(kappa): return twotype_law(0.4, 0.5 + kappa * D_SEP, 0.5 - kappa * D_SEP)


def margins(P): return np.array([sum(P[k] for k, x in enumerate(CELLS3) if x[i]) for i in range(3)])


def source2(kappa):
    p = margins(source1(kappa)); q11 = max(0.0, p[0] + p[2] - 1); T = {(1, 1): q11, (1, 0): p[0] - q11, (0, 1): p[2] - q11}
    T[(0, 0)] = 1 - sum(T.values()); P = np.zeros(8)
    for k, x in enumerate(CELLS3): P[k] = T[(x[0], x[2])] * (p[1] if x[1] else 1 - p[1])
    return P


def pair_table(P, i, j):
    T = np.zeros((2, 2))
    for k, x in enumerate(CELLS3): T[x[i], x[j]] += P[k]
    return T


def window_tables(theta, kappa):
    S1 = source1(kappa); S2 = source2(kappa); mix = (1 - theta) * S1 + theta * S2
    return {(0, 1): pair_table(S1, 0, 1), (1, 2): pair_table(S1, 1, 2), (0, 2): pair_table(mix, 0, 2)}


def glues(tables):
    A = []; b = []
    for (i, j), T in tables.items():
        for xi, xj in itertools.product((0, 1), repeat=2):
            A.append([1.0 if (x[i], x[j]) == (xi, xj) else 0.0 for x in CELLS3]); b.append(T[xi, xj])
    # maximise the slack t: P >= t (feasible with t >= 0 iff a joint exists); minimise -t
    A = np.array(A); n = 8; c = np.r_[np.zeros(n), -1.0]
    A_eq = np.hstack([A, np.zeros((len(A), 1))]); A_ub = np.hstack([-np.eye(n), np.ones((n, 1))])
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=b, bounds=[(None, None)] * (n + 1), method="highs")
    return (r.status == 0 and -r.fun >= -1e-12), (-r.fun if r.status == 0 else float("-inf"))


def boole(tables):
    """the closed form for three binary readings with consistent single margins: pair tables glue iff the four Boole (Bell) inequalities
    hold, with p_i the margins and p_ij = P(Xi = 1, Xj = 1):  p_ij + p_ik - p_jk <= p_i (each i)  and  p_0 + p_1 + p_2 - p_01 - p_02 - p_12 <= 1.
    Returns the largest violation (<= 0 iff the tables glue)."""
    pij = {k: T[1, 1] for k, T in tables.items()}; pij.update({(j, i): v for (i, j), v in list(pij.items())})
    p = [tables[(0, 1)][1].sum(), tables[(0, 1)][:, 1].sum(), tables[(1, 2)][:, 1].sum()]
    viol = [pij[(i, j)] + pij[(i, k)] - pij[(j, k)] - p[i] for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1))]
    viol.append(sum(p) - pij[(0, 1)] - pij[(0, 2)] - pij[(1, 2)] - 1); return max(viol), viol


def theta_wall(kappa, lo=0.0, hi=1.0):
    """the least share theta of stale provenance in window 02 at which the windows stop gluing (None if they glue up to theta = 1)"""
    if boole(window_tables(hi, kappa))[0] <= 0: return None
    v0 = boole(window_tables(0.0, kappa))[0]; v1 = boole(window_tables(1.0, kappa))[0]   # each inequality is affine in theta
    vs0 = boole(window_tables(0.0, kappa))[1]; vs1 = boole(window_tables(1.0, kappa))[1]
    ts = [-a / (b - a) for a, b in zip(vs0, vs1) if b > 0 and a < 0]; return min(ts)


def cells_lab(): return [(l,) + x for l in (0, 1) for x in CELLS3]


def marginal_matrix_lab(level):
    C = cells_lab(); rows = []
    def add(f):
        keys = sorted({f(c) for c in C})
        for k in keys: rows.append([1.0 if f(c) == k else 0.0 for c in C])
    if level >= 0:
        for i in range(3): add(lambda c, i=i: c[1 + i])                          # pooled singles
    if level >= 1:
        for i, j in itertools.combinations(range(3), 2): add(lambda c, i=i, j=j: (c[1 + i], c[1 + j]))   # pooled pairs
    if level >= 2:
        for i, j in itertools.combinations(range(3), 2): add(lambda c, i=i, j=j: (c[0], c[1 + i], c[1 + j]))  # labelled pairs
    if level >= 3: add(lambda c: c)                                              # labelled full
    return np.array(rows)


def fibre_points(P, starts, rng, tol=1e-5):
    def res(th):
        w = th[0]; a1 = th[1:4]; a2 = th[4:7]; return twotype_law(w, a1, a2) - P
    sols = []
    for _ in range(starts):
        th0 = rng.uniform(0.02, 0.98, 7); r = least_squares(res, th0, bounds=(0, 1), xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if np.max(np.abs(r.fun)) < 1e-9: sols.append(r.x)
    clusters = []
    for x in sols:
        if not any(np.max(np.abs(x - y)) < tol for y in clusters): clusters.append(x)
    return sols, clusters


# the flag event: labelled cells on L x A as polynomials in (s, t) (exponent tuples (a_s, a_t)), and the decision gap on them
FLAG_CELLS = {("source 1", "X0=1"): {(Fr(3), Fr(0)): Fr(1)}, ("source 1", "X0=0"): {(Fr(2), Fr(1)): Fr(7, 4)},
              ("source 2", "no reading"): {(Fr(1), Fr(1)): Fr(1)}, ("source 2", "X0=0"): {(Fr(0), Fr(3)): Fr(1)}}
FLAG_GAP = {("source 1", "X0=1"): Fr(1), ("source 1", "X0=0"): Fr(-1), ("source 2", "no reading"): Fr(0), ("source 2", "X0=0"): Fr(-1)}


# ------------------------------------------------------------------------------------------------ how much data the decision needs, cone by cone
GAP_VEC = np.array([1.0, -1.0, 0.0, -1.0])                                   # cells in FLAG_CELLS order: s^3, 7/4 s^2 t, ts, t^3


def flag_masses(s, t): return np.array([s ** 3, 1.75 * s * s * t, t * s, t ** 3])


def p_wrong(N, s, t):
    """exact P(wrong act) for the plug-in rule sign(n_1 - n_2 - n_4) with Poisson cell counts N * mass (ties broken by a fair coin):
    the gap estimate is a Skellam variable; the uninformative cell (gap 0) adds no noise"""
    from scipy.stats import skellam
    m = N * flag_masses(s, t); mu1, mu2 = m[0], m[1] + m[3]
    if mu1 + mu2 > 1e8:                                                      # Skellam underflows; the normal limit is exact to O(mu^-1/2)
        from scipy.stats import norm
        return float(norm.cdf(-abs(mu1 - mu2) / np.sqrt(mu1 + mu2)))
    half = 0.5 * skellam.pmf(0, mu1, mu2)
    return float((skellam.cdf(-1, mu1, mu2) if m[0] - m[1] - m[3] > 0 else skellam.sf(0, mu1, mu2)) + half)


def n_star(s, t, target=0.1):
    """the least sample size N with P(wrong act) = target"""
    from scipy.optimize import brentq
    m = flag_masses(s, t); V = m[0] + m[1] + m[3]; G = abs(m[0] - m[1] - m[3]); L0 = np.log(V / G ** 2)
    f = lambda L: p_wrong(np.exp(L), s, t) - target; lo, hi = L0 - 3, L0 + 3
    while f(lo) < 0: lo -= 2
    while f(hi) > 0: hi += 2
    return float(np.exp(brentq(f, lo, hi, xtol=1e-7)))


def sample_exponents(w):
    """orders along the arc s = tau^w_s, t = tau^w_t: v(P), v(N), the decisive order v_dec (least order among cells with a non-zero gap,
    the order of the gap's variance), the tie order m = v(N) - v(P), the predicted sample exponent e = 2 v(N) - v_dec, and the exponent
    2 v(N) - v(P) that a fixed conditional spread would give"""
    ords = [3 * w[0], 2 * w[0] + w[1], w[0] + w[1], 3 * w[1]]
    vP = min(ords); vdec = min(o for o, g in zip(ords, GAP_VEC) if g != 0)
    lead = sum(c * g for o, c, g in zip(ords, (1.0, 1.75, 1.0, 1.0), GAP_VEC) if abs(o - vdec) < 1e-12)   # monomial arc, coefficients 1
    vN = vdec if abs(lead) > 1e-12 else None                                 # cancellation happens only on the wall ray at c = 1/2
    return dict(vP=vP, vN=vN, vdec=vdec, m=None if vN is None else vN - vP, e=None if vN is None else 2 * vN - vdec,
                fixed_spread=None if vN is None else 2 * vN - vP)
