"""The geometry of decisions and of the value of information on marginal data (companion note, section 12.23). New module; it modifies
nothing and uses pce/margsec.py and pce/poisonflip.py.
Actions a carry utilities u_a(omega) on joint readings, so E_P u_a = u_a . P is linear in the joint law P and the utility image of a fibre
is a polytope. Two kinds of information are kept apart:
  structural: a new context is observed (a pair never measured together); it shrinks the fibre, J_d -> J_d'
  instance:   a signal Z about this unit's state, with the population law fixed
  utility_interval(d, ua, ub)     [L, U] of (u_b - u_a) . P over the fibre
  identified(d, U)                True when one action is optimal on the whole fibre (the image lies in one cell)
  regret_value(d, U)              R(d) = min_a max_b max_P (u_b - u_a) . P: the minimax-regret value of resolving the fibre
  act_abstain(L, U, tau)          the symmetric example: R = 2 min((tau - L)+, (U - tau)+), randomisation gap = R / 2
  signal_voi(phi, tau, alpha)     the instance value of a signal of accuracy alpha about T = 1[A0 = A1] under P(T) = phi
  robust_signal_voi(L, U, tau, a) the value to a robust decision-maker who commits to a rule z -> action
  mutual_info(phi, alpha)         I(T; Z) in bits
"""
from __future__ import annotations
import itertools
import numpy as np
from . import margsec as G
from . import poisonflip as P


def utility_interval(d, ua, ub, rows=None):
    return P.lower_upper(d, np.asarray(ub, float) - np.asarray(ua, float), rows=rows)


def identified(d, U, rows=None, tol=1e-12):
    """some action a has (u_b - u_a) . P <= 0 for every b and every P in the fibre"""
    return any(all(utility_interval(d, U[a], U[b], rows)[1] <= tol for b in range(len(U)) if b != a) for a in range(len(U)))


def regret_value(d, U, rows=None):
    return min(max(max(utility_interval(d, U[a], U[b], rows)[1], 0.0) if b != a else 0.0 for b in range(len(U))) for a in range(len(U)))


def act_abstain_utilities(tau):
    """act: 1[A0 = A1] - tau; abstain: tau - 1[A0 = A1] (symmetric regret around the threshold)"""
    t = np.array([1.0 if c[0] == c[1] else 0.0 for c in G.CELLS]); return [tau - t, t - tau]      # index 0 abstain, 1 act


def act_abstain(L, U, tau):
    o1, o2 = max(tau - L, 0.0), max(U - tau, 0.0)
    return dict(R=2 * min(o1, o2), gap=min(o1, o2) if L < tau < U else 0.0, pure=max(L - tau, tau - U))


def mixed_maxmin(L, U, tau):
    """max over x in [0, 1] (probability of acting) of min over phi in [L, U] of x (phi - tau) + (1 - x)(tau - phi)"""
    best = -np.inf
    for x in np.linspace(0, 1, 2001):
        v = min(x * (p - tau) + (1 - x) * (tau - p) for p in (L, U)); best = max(best, v)
    return best


def _a(phi, tau, alpha):
    """a_z = P(T = 1, Z = z) - tau P(Z = z), for Z = T with probability alpha"""
    p11, p10 = alpha * phi, (1 - alpha) * phi; p01, p00 = (1 - alpha) * (1 - phi), alpha * (1 - phi)
    return (p11 - tau * (p11 + p01), p10 - tau * (p10 + p00))


def signal_voi(phi, tau, alpha):
    """with payoffs act: T - tau, abstain: tau - T the value is sum_z |a_z| with the signal and |sum_z a_z| without; VOI = their difference"""
    a1, a0 = _a(phi, tau, alpha); return abs(a1) + abs(a0) - abs(a1 + a0)


def robust_signal_voi(L, U, tau, alpha):
    """max over committed rules sigma: z -> {act, abstain} of min over phi in [L, U] of the expected payoff, minus the same over constant rules.
    Payoffs are linear in phi, so the inner minimum is at an endpoint."""
    def pay(sig, phi):
        a1, a0 = _a(phi, tau, alpha); return (a1 if sig[0] else -a1) + (a0 if sig[1] else -a0)
    best = lambda sigs: max(min(pay(s, p) for p in (L, U)) for s in sigs)
    return best(list(itertools.product((0, 1), repeat=2))) - best([(0, 0), (1, 1)])


def h2(x):
    x = min(max(x, 1e-300), 1 - 1e-16); return -(x * np.log2(x) + (1 - x) * np.log2(1 - x)) if 0 < x < 1 else 0.0


def mutual_info(phi, alpha):
    return h2(phi * alpha + (1 - phi) * (1 - alpha)) - h2(alpha)


def alpha_star(phi, tau):
    """the least accuracy at which the signal can move the decision (one posterior crosses tau): smallest alpha in [1/2, 1] with VOI > 0"""
    lo, hi = 0.5, 1.0
    if signal_voi(phi, tau, 1.0) <= 1e-15: return None
    for _ in range(80):
        mid = (lo + hi) / 2; lo, hi = (lo, mid) if signal_voi(phi, tau, mid) > 1e-15 else (mid, hi)
    return hi


# ------------------------------------------------------------------ stability: fibres, dilation, envelopes
def fibre_vertices(d, tol=1e-10):
    """all vertices of J_d: basic feasible solutions with rank-many basic cells (exhaustive over bases; 16 cells, rank 9 on CHSH)"""
    M = np.array(G.M, float); A = np.vstack([M, np.ones(16)]); b = np.r_[np.asarray(d, float), 1.0]; r = np.linalg.matrix_rank(A); out = {}
    for B in itertools.combinations(range(16), r):
        AB = A[:, list(B)]
        if np.linalg.matrix_rank(AB) < r: continue
        xb = np.linalg.lstsq(AB, b, rcond=None)[0]
        if np.abs(AB @ xb - b).max() > 1e-9 or xb.min() < -tol: continue
        x = np.zeros(16); x[list(B)] = np.maximum(xb, 0); out[tuple(np.round(x, 9))] = x
    return list(out.values())


def tv_to_fibre(x, d):
    """min over y in J_d of TV(x, y) (a linear program)"""
    from scipy.optimize import linprog
    M = np.array(G.M, float); A = np.vstack([M, np.ones(16)]); b = np.r_[np.asarray(d, float), 1.0]
    c = np.r_[np.zeros(16), 0.5 * np.ones(16)]
    Aub = np.vstack([np.c_[np.eye(16), -np.eye(16)], np.c_[-np.eye(16), -np.eye(16)]]); bub = np.r_[x, -x]
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=np.c_[A, np.zeros((A.shape[0], 16))], b_eq=b, bounds=[(0, None)] * 32, method="highs")
    return r.fun


def hausdorff_tv(d1, d2):
    """Hausdorff distance between J_d1 and J_d2 in total variation: the distance to a polytope is convex, so its sup is at a vertex"""
    return max(max(tv_to_fibre(x, d2) for x in fibre_vertices(d1)), max(tv_to_fibre(y, d1) for y in fibre_vertices(d2)))


def conditional_interval(d, w_joint, w_given, p_given):
    """the identified interval of P(w | given) when P(given) = p_given is fixed by the data (an observed margin): [L, U] of the joint
    event divided by p_given"""
    lo, hi = P.lower_upper(d, np.asarray(w_joint, float)); return lo / p_given, hi / p_given


def signal_envelope(L, U, tau, alpha, n=2001):
    """[min, max] over phi in [L, U] of the classical (fixed-measure) value of the signal (Iskandar's envelope)"""
    v = [signal_voi(p, tau, alpha) for p in np.linspace(L, U, n)]; return min(v), max(v)


def das_gap(d, z, sel=None):
    """Das's square for the event E = {B0 = z}: TV between s(d)(. | E) (commit, then condition) and the law re-selected from the data that
    remain observable after conditioning. Only the contexts containing B0 ((A0, B0), (A1, B0)) give conditional tables, so the conditioned
    problem knows only P(A0 | E) and P(A1 | E); its maximum-entropy law is their product with B1 uniform. Returns (TV, committed P(A0 = A1 | E),
    re-selected P(A0 = A1 | E))."""
    x = G.maxent(d) if sel is None else sel(d); C = G.CELLS
    m = np.array([1.0 if c[2] == z else 0.0 for c in C]); pz = float(m @ x); xc = x * m / pz
    a0 = sum(xc[i] for i, c in enumerate(C) if c[0] == 0); a1 = sum(xc[i] for i, c in enumerate(C) if c[1] == 0)
    y = np.array([((a0 if c[0] == 0 else 1 - a0) * (a1 if c[1] == 0 else 1 - a1) * 0.5) if c[2] == z else 0.0 for c in C])
    t = np.array([1.0 if c[0] == c[1] else 0.0 for c in C])
    return 0.5 * float(np.abs(xc - y).sum()), float(t @ xc), float(t @ y)


def robust_signal_voi_mixed(L, U, tau, alpha, n=4001):
    """the robust value of the signal when mixed policies are allowed with and without it. By the minimax theorem (payoffs linear in phi and
    in the mixture), the robust value is min over phi in [L, U] of the Bayes value: sum_z |a_z(phi)| with the signal, |phi - tau| without.
    It is 0 for an uninformative signal (alpha = 1/2), unlike the pure-policy value, whose excess is the value of randomising."""
    ph = np.linspace(L, U, n) if U > L else np.array([L])
    with_ = min(abs(a1) + abs(a0) for a1, a0 in (_a(p, tau, alpha) for p in ph)); without = min(abs(p - tau) for p in ph)
    return with_ - without


# ------------------------------------------------------------------ when do local data determine the value of information? (section 12.23)
def _fibre_lp(c, d, Aub=None, bub=None, extra_eq=None):
    from scipy.optimize import linprog
    M = np.array(G.M, float); A = np.vstack([M, np.ones(16)]); b = np.r_[np.asarray(d, float), 1.0]
    return linprog(c, A_ub=Aub, b_ub=bub, A_eq=A, b_eq=b, bounds=[(0, None)] * 16, method="highs")


def fibre_support(d, tol=1e-10):
    """cells that are positive for some law in J_d (the support of the fibre's relative interior)"""
    return [c for c in range(16) if -_fibre_lp(-np.eye(16)[c], d).fun > tol]


def fibre_directions(d):
    """an orthonormal basis of the direction space of J_d: v supported on the fibre's support, M v = 0, 1.v = 0"""
    S = fibre_support(d); M = np.array(G.M, float); A = np.vstack([M, np.ones(16)])[:, S]
    if not S: return np.zeros((0, 16))
    u, s, vt = np.linalg.svd(A); r = int((s > 1e-10).sum()); K = vt[r:]
    out = np.zeros((K.shape[0], 16)); out[:, S] = K; return out


def policies(nA, nZ):
    """(a0, (a_z)_z): the action without the signal and the rule with it"""
    return [(a0, rule) for a0 in range(nA) for rule in itertools.product(range(nA), repeat=nZ)]


def value_functional(U, zmap, pol):
    """g = sum_z 1[z(omega) = z] (u_{a_z} - u_{a_0}): VOI on the policy's cell is g . P"""
    a0, rule = pol; z = np.asarray(zmap); return sum((z == k) * (U[rule[k]] - U[a0]) for k in range(len(rule)))


def cell_constraints(U, zmap, pol):
    """linear inequalities (A_ub P <= 0) saying a0 is optimal without the signal and a_z is optimal on each z"""
    a0, rule = pol; z = np.asarray(zmap); rows = []
    for b in range(len(U)):
        if b != a0: rows.append(U[b] - U[a0])
        for k in range(len(rule)):
            if b != rule[k]: rows.append((z == k) * (U[b] - U[rule[k]]))
    return np.array(rows)


def voi_at(P, U, zmap):
    z = np.asarray(zmap); P = np.asarray(P, float)
    return sum(max((z == k) * u @ P for u in U) for k in sorted(set(zmap))) - max(u @ P for u in U)


def voi_envelope(d, U, zmap):
    """exact [min, max] of the classical VOI over J_d: on each policy cell VOI is the linear function g . P, so one LP per cell and sense"""
    nZ = len(set(zmap)); lo, hi = np.inf, -np.inf
    for pol in policies(len(U), nZ):
        A = cell_constraints(U, zmap, pol); g = value_functional(U, zmap, pol)
        r1 = _fibre_lp(g, d, A, np.zeros(len(A))); r2 = _fibre_lp(-g, d, A, np.zeros(len(A)))
        if r1.status == 0: lo = min(lo, r1.fun)
        if r2.status == 0: hi = max(hi, -r2.fun)
    return lo, hi


def full_dimensional_cells(d, U, zmap, tol=1e-9):
    """policies whose cell has nonempty interior relative to J_d: maximise a common slack t on the cell's strict inequalities and on the
    support cells, inside J_d (positive t certifies a relative-interior point)"""
    from scipy.optimize import linprog
    S = fibre_support(d); M = np.array(G.M, float); A = np.vstack([M, np.ones(16)]); b = np.r_[np.asarray(d, float), 1.0]; out = []
    D = fibre_directions(d)
    for pol in policies(len(U), len(set(zmap))):
        C = cell_constraints(U, zmap, pol)
        rows = [r for r in C if np.linalg.norm(D @ r) > 1e-12]                         # constraints constant on the fibre need no slack
        const = [r for r in C if np.linalg.norm(D @ r) <= 1e-12]
        Aub = [np.r_[r, 1.0] for r in rows] + [np.r_[-np.eye(16)[c], 1.0] for c in S]
        bub = [0.0] * len(Aub)
        Aub += [np.r_[r, 0.0] for r in const]; bub += [0.0] * len(const)
        r_ = linprog(np.r_[np.zeros(16), -1.0], A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.c_[A, np.zeros(len(A))], b_eq=b,
                     bounds=[(0, None)] * 16 + [(None, 1.0)], method="highs")
        if r_.status == 0 and -r_.fun > tol: out.append(pol)
    return out


def determination_criterion(d, U, zmap, tol=1e-9):
    """True when every full-dimensional policy cell's value functional is orthogonal to the fibre's directions (cover-additive on its face)"""
    D = fibre_directions(d)
    if D.shape[0] == 0: return True, []
    cells = full_dimensional_cells(d, U, zmap)
    return all(np.linalg.norm(D @ value_functional(U, zmap, p)) < tol for p in cells), cells


# ------------------------------------------------------------------ value-based distance = Hausdorff distance (section 12.23)
def support_value(d, u):
    """the robust value of a single action with utility u: min over J_d of u . P (the support function of the fibre)"""
    return _fibre_lp(np.asarray(u, float), d).fun


def worst_direction(d1, d2):
    """the utility u in [-1, 1]^16 maximising |h_{d1}(u) - h_{d2}(u)|, from the vertex of one fibre farthest from the other (LP dual of the
    l1 distance). Returns (gap, u)."""
    from scipy.optimize import linprog
    best = (0.0, None)
    for (da, db) in ((d1, d2), (d2, d1)):
        for x in fibre_vertices(da):
            # max over u in [-1,1]^16 of (min_{Q in J_db} u.Q - u.x) = l1 distance; solve via the dual: max u.(-x) + h(u) is not linear, so
            # take u = sign(y* - x) for the nearest point y* of J_db (an l1 subgradient), and evaluate the gap exactly
            M = np.array(G.M, float); A = np.vstack([M, np.ones(16)]); b = np.r_[np.asarray(db, float), 1.0]
            r = linprog(np.r_[np.zeros(16), np.ones(16)], A_ub=np.vstack([np.c_[np.eye(16), -np.eye(16)], np.c_[-np.eye(16), -np.eye(16)]]),
                        b_ub=np.r_[x, -x], A_eq=np.c_[A, np.zeros((A.shape[0], 16))], b_eq=b, bounds=[(0, None)] * 32, method="highs")
            # dual multipliers of the |y - x| constraints give the optimal u
            lam = r.ineqlin.marginals; u = -(lam[:16] - lam[16:])
            gap = abs(support_value(da, u) - support_value(db, u))
            if gap > best[0]: best = (gap, u)
    return best
