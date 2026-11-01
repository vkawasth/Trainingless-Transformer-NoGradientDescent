"""What does not matter: the decision-relevant quotient, decision value as a coboundary, and the ambiguity gap (lab Section
"What does not matter"). New module.

Laws are arrays on {0,1}^m (C order). A decision is a list of utility arrays u_a on {0,1}^m (or losses l_a = -u_a).

  marginal_matrix(m, windows)          K: the linear map from laws to the stacked window tables
  rho_range(P, windows, du)            [min, max] of <du, Q> over the fibre {Q >= 0 : K Q = K P} (two linear programmes)
  settled(P, windows, U)               the act that is optimal for every law in the fibre, or None
  relevant_rank(m, windows, U)         rank of the utility differences restricted to ker K: the decision-relevant dimension of the fibre
  redundant_windows(P, windows, U)     windows whose removal leaves every utility-difference range unchanged
  bayes_risk(P, L)                     R(P) = min_a E_P l_a
  condition(P, axes, x)                P(. | X_axes = x) (None if probability zero)
  act_on(f, P, axes)                   Vigneaux's action (alpha = 1): sum_x P(X = x) f(P | X = x)
  value(P, axes, L)                    V[X](P) = R(P) - X.R(P), the value of observing X before deciding
  ante_and_updated(Q, axes, U)         for a finite set Q of laws: the ex-ante maxmin value of observing X (mixed policies, linear
                                       programme) and the ex-ante worst-case value of the policy chosen by prior-by-prior updating
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog


def cells(m): return list(itertools.product((0, 1), repeat=m))


def marginal_matrix(m, windows):
    C = cells(m); rows = []
    for W in windows:
        for x in itertools.product((0, 1), repeat=len(W)):
            rows.append([1.0 if all(c[i] == xi for i, xi in zip(W, x)) else 0.0 for c in C])
    return np.array(rows)


def rho_range(P, windows, du):
    m = P.ndim; K = marginal_matrix(m, windows); b = K @ P.ravel(); d = np.asarray(du).ravel(); out = []
    for sgn in (1, -1):
        r = linprog(sgn * d, A_eq=K, b_eq=b, bounds=[(0, None)] * len(d), method="highs"); out.append(sgn * r.fun)
    return out[0], out[1]


def settled(P, windows, U):
    for a in range(len(U)):
        if all(rho_range(P, windows, U[a] - U[b])[0] >= -1e-12 for b in range(len(U)) if b != a): return a
    return None


def kernel(m, windows):
    K = marginal_matrix(m, windows); U_, s, Vt = np.linalg.svd(K); r = int((s > 1e-10).sum()); return Vt[r:].T


def relevant_rank(m, windows, U):
    N = kernel(m, windows); D = np.array([(U[a] - U[0]).ravel() for a in range(1, len(U))])
    return int(np.linalg.matrix_rank(D @ N, 1e-10)) if N.size else 0


def redundant_windows(P, windows, U, tol=1e-10):
    base = [rho_range(P, windows, U[a] - U[b]) for a, b in itertools.combinations(range(len(U)), 2)]; out = []
    for j in range(len(windows)):
        rest = windows[:j] + windows[j + 1:]
        if not rest: continue
        rr = [rho_range(P, rest, U[a] - U[b]) for a, b in itertools.combinations(range(len(U)), 2)]
        if all(abs(x[0] - y[0]) < tol and abs(x[1] - y[1]) < tol for x, y in zip(base, rr)): out.append(j)
    return out


def bayes_risk(P, L): return min(float((P * l).sum()) for l in L)


def condition(P, axes, x):
    idx = [slice(None)] * P.ndim
    for a, xi in zip(axes, x): idx[a] = xi
    mass = P[tuple(idx)].sum()
    if mass <= 0: return None, 0.0
    Q = np.zeros_like(P); Q[tuple(idx)] = P[tuple(idx)] / mass; return Q, float(mass)


def act_on(f, P, axes):
    tot = 0.0
    for x in itertools.product((0, 1), repeat=len(axes)):
        Q, p = condition(P, axes, x)
        if p > 0: tot += p * f(Q)
    return tot


def value(P, axes, L): return bayes_risk(P, L) - act_on(lambda Q: bayes_risk(Q, L), P, axes)


def ante_and_updated(Qs, axes, U):
    """maxmin over mixed policies x -> Delta(acts) of min_{P in Qs} E_P u(policy); and the same evaluation for the policy that, after x,
    plays the maxmin mixed act of the conditional set {P | x}"""
    k = len(U); xs = list(itertools.product((0, 1), repeat=len(axes)))
    def eu_block(P, x):            # E_P[u_a ; X = x] for each act
        idx = [slice(None)] * P.ndim
        for a, xi in zip(axes, x): idx[a] = xi
        return np.array([float((P[tuple(idx)] * Ua[tuple(idx)]).sum()) for Ua in U])
    # ex ante: variables pi[x, a] >= 0, sum_a pi[x, a] = 1, t; max t s.t. sum_x sum_a pi E_P[u_a; x] >= t for all P
    nv = len(xs) * k + 1; c = np.zeros(nv); c[-1] = -1
    A_ub = []; b_ub = []
    for P in Qs:
        row = np.zeros(nv)
        for j, x in enumerate(xs): row[j * k:(j + 1) * k] = -eu_block(P, x)
        row[-1] = 1; A_ub.append(row); b_ub.append(0)
    A_eq = []; b_eq = []
    for j in range(len(xs)):
        row = np.zeros(nv); row[j * k:(j + 1) * k] = 1; A_eq.append(row); b_eq.append(1)
    r = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq, bounds=[(0, None)] * (nv - 1) + [(None, None)], method="highs")
    ante = -r.fun
    # prior-by-prior updating: after x, maxmin mixed act against {P | x}
    pol = np.zeros((len(xs), k))
    for j, x in enumerate(xs):
        conds = [condition(P, axes, x)[0] for P in Qs]; conds = [Q for Q in conds if Q is not None]
        if not conds: pol[j] = np.ones(k) / k; continue
        M = np.array([[float((Q * Ua).sum()) for Ua in U] for Q in conds])
        cc = np.zeros(k + 1); cc[-1] = -1; Aub = np.hstack([-M, np.ones((len(conds), 1))])
        rr = linprog(cc, A_ub=Aub, b_ub=np.zeros(len(conds)), A_eq=[list(np.ones(k)) + [0]], b_eq=[1], bounds=[(0, None)] * k + [(None, None)], method="highs")
        pol[j] = rr.x[:k]
    upd = min(sum(pol[j] @ eu_block(P, x) for j, x in enumerate(xs)) for P in Qs)
    return ante, upd
