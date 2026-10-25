"""Partial gluing, and level two: mixtures of latent types, their singular fibres, second-order obstructions and monodromy.

Level one, graded (sitting at a gluing failure):
  explanation set   E_alpha = {q a law on the atoms : alpha * marginal_w(q) <= t_w for every window w}: laws that account for a fraction
                    alpha of every table. Non-empty iff alpha <= lam*.
  predict(domains, windows, event, alpha)   [min, max] of q(event) over E_alpha (two LPs), or None if alpha > lam*

Level two, latent class model (actors = latent types): k types, m binary variables, weights w_h, type h has P(X_i = 1 | h) = a_hi and
the X_i independent given the type. The observed law F(w, a) = sum_h w_h prod_i Bern(a_hi) is a polynomial map; its fibre over an
observed law P is the real algebraic set F^{-1}(P).
  lcm_law(w, a)                 the law over the 2^m atoms (axis order X_1..X_m)
  jacobian(w, a)                dF (2^m x (k + k m)), exact
  second(w, a, v)               D^2 F (v, v), exact (the quadratic term of F along v)
  kuranishi(w, a, tol)          kernel K of dF restricted to the parameter directions keeping sum w = 1, and a basis C of the cokernel
                                of dF inside the sum-zero directions; kappa(v) = C^T D^2F(v, v) is the second-order obstruction: a direction
                                v in ker dF is tangent to a path in the fibre only if kappa(v) = 0
  fit(P, k, init)               least-squares fit of (w, a) to P from an initial point (for tracking along paths)
  track(path, k, init)          follow the fibre along a path of observed laws, each step started from the previous solution, and
                                return the parameters at every step; labels are carried by continuity
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog, least_squares
from .coherence import _constraints
from .integrate import gluable_fraction


# ------------------------------------------------------------------------------------------------ level one, graded
def predict(domains, windows, event, alpha):
    A, t, atoms, index = _constraints(domains, windows); n = A.shape[1]; names = list(domains)
    e = np.array([1.0 if event(dict(zip(names, a))) else 0.0 for a in atoms])
    Aub = alpha * A; Aeq = np.ones((1, n))
    out = []
    for sgn in (1.0, -1.0):
        r = linprog(sgn * e, A_ub=Aub, b_ub=t, A_eq=Aeq, b_eq=[1.0], bounds=(0, None), method="highs")
        if r.status != 0: return None
        out.append(sgn * r.fun)
    return out[0], out[1]


def most_probable(domains, windows, alpha):
    """the maximum-entropy alpha-explanation: argmax H(q) over E_alpha (the least committal law that still accounts for a fraction alpha
    of every table). None if alpha > lam*."""
    from scipy.optimize import minimize
    A, t, atoms, index = _constraints(domains, windows); n = A.shape[1]
    r0 = linprog(np.zeros(n), A_ub=alpha * A, b_ub=t, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=(0, None), method="highs")
    if r0.status != 0: return None, atoms
    x0 = np.clip(r0.x, 1e-9, None); x0 /= x0.sum()
    f = lambda x: float(np.sum(x * np.log(np.maximum(x, 1e-300)))); g = lambda x: np.log(np.maximum(x, 1e-300)) + 1
    cons = [{"type": "ineq", "fun": lambda x: t - alpha * A @ x, "jac": lambda x: -alpha * A},
            {"type": "eq", "fun": lambda x: x.sum() - 1, "jac": lambda x: np.ones((1, n))}]
    r = minimize(f, x0, jac=g, constraints=cons, bounds=[(0, 1)] * n, method="SLSQP", options={"ftol": 1e-14, "maxiter": 1000})
    x = np.clip(r.x, 0, None); x /= x.sum()
    if np.any(alpha * A @ x > t + 1e-7): x = x0
    return x, atoms


# ------------------------------------------------------------------------------------------------ level two: latent classes
def _bern(a): return np.array([1 - a, a])


def lcm_law(w, a):
    w = np.asarray(w, float); a = np.asarray(a, float); k, m = a.shape; P = np.zeros((2,) * m)
    for h in range(k):
        T = np.array(1.0)
        for i in range(m): T = np.multiply.outer(T, _bern(a[h, i]))
        P = P + w[h] * T
    return P


def _prod(a_h, replace=None):
    """product law of one type, with factor i replaced by (-1, +1) for each i in `replace`."""
    T = np.array(1.0)
    for i, x in enumerate(a_h):
        f = np.array([-1.0, 1.0]) if replace and i in replace else _bern(x)
        T = np.multiply.outer(T, f)
    return T


def jacobian(w, a):
    w = np.asarray(w, float); a = np.asarray(a, float); k, m = a.shape; cols = []
    for h in range(k): cols.append(_prod(a[h]).ravel())
    for h in range(k):
        for i in range(m): cols.append(w[h] * _prod(a[h], {i}).ravel())
    return np.array(cols).T


def second(w, a, v):
    """D^2 F(v, v) for a direction v = (dw (k), da (k*m))."""
    w = np.asarray(w, float); a = np.asarray(a, float); k, m = a.shape; v = np.asarray(v, float)
    dw, da = v[:k], v[k:].reshape(k, m); out = np.zeros(2 ** m)
    for h in range(k):
        for i in range(m): out += 2 * dw[h] * da[h, i] * _prod(a[h], {i}).ravel()
        for i, j in itertools.combinations(range(m), 2): out += 2 * w[h] * da[h, i] * da[h, j] * _prod(a[h], {i, j}).ravel()
    return out


def _null(M, tol=1e-9):
    u, s, vt = np.linalg.svd(M); r = int((s > tol * max(1.0, s[0] if len(s) else 1.0)).sum()); return vt[r:].T, r


def kuranishi(w, a, tol=1e-9):
    """returns K (kernel of dF within sum dw = 0), C (cokernel basis within the sum-zero target), and the ranks."""
    w = np.asarray(w, float); a = np.asarray(a, float); k, m = a.shape; J = jacobian(w, a); n = J.shape[1]
    S = np.zeros((1, n)); S[0, :k] = 1.0                                   # keep sum w = 1
    K, _ = _null(np.vstack([J, S]), tol)
    Z, _ = _null(np.ones((1, 2 ** m)), tol)                               # sum-zero directions of the target (tangent to the simplex)
    Jz = Z.T @ J @ _null(S, tol)[0]                                        # dF from the constrained domain into the target tangent
    u, s, vt = np.linalg.svd(Jz); r = int((s > tol * max(1.0, s[0])).sum())
    C = Z @ u[:, r:]                                                       # cokernel, as vectors of the target
    return K, C, r


def kappa(w, a, v, C):
    return C.T @ second(w, a, v)


def fit(P, k, init, bounded=False):
    P = np.asarray(P, float).ravel(); w0, a0 = init; m = int(np.log2(P.size))
    def unpack(z):
        w = np.empty(k); w[:-1] = z[:k - 1]; w[-1] = 1 - z[:k - 1].sum(); return w, z[k - 1:].reshape(k, m)
    z0 = np.concatenate([np.asarray(w0, float)[:-1], np.asarray(a0, float).ravel()])
    kw = {"bounds": (np.zeros(z0.size), np.ones(z0.size))} if bounded else {}
    if bounded: z0 = np.clip(z0, 1e-6, 1 - 1e-6)
    r = least_squares(lambda z: lcm_law(*unpack(z)).ravel() - P, z0, xtol=1e-14, ftol=1e-14, gtol=1e-14, **kw)
    return unpack(r.x)


def fiber_range(target_fn, target, k, m, objective, w_grid, starts=20, seed=0, tol=1e-9):
    """range of objective(w, a) over the latent-class models with target_fn(law) = target, found by fixing w_1 on a grid (k = 2)
    and solving for the type parameters from several starts (bounded least squares). Returns (min, max, solutions)."""
    rng = np.random.default_rng(seed); sols = []
    for w1 in w_grid:
        w = np.array([w1, 1 - w1])
        for _ in range(starts):
            r = least_squares(lambda z: target_fn(lcm_law(w, z.reshape(k, m))) - target, rng.uniform(0.02, 0.98, k * m),
                              bounds=(np.zeros(k * m), np.ones(k * m)), xtol=1e-15, ftol=1e-15, gtol=1e-15)
            if np.abs(r.fun).max() < tol: sols.append((w, r.x.reshape(k, m)))
    vals = [objective(w, a) for w, a in sols]
    return (min(vals), max(vals), sols) if vals else (None, None, sols)


def track(path, k, init):
    out = [init]
    for P in path[1:]: out.append(fit(P, k, out[-1]))
    return out


def lam2(target_fn, target, k, m, starts=40, seed=0):
    """level-two gluable fraction (lower bound): the largest alpha with alpha * target_fn(F(w, a)) <= target for some k-type model,
    by multi-start SLSQP over (w, a, alpha). Returns (alpha, w, a); the value is attained, hence a lower bound on lam*_2."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed); best = (0.0, None, None); target = np.asarray(target, float)
    def unpack(z):
        w = np.empty(k); w[:-1] = z[:k - 1]; w[-1] = 1 - z[:k - 1].sum(); return w, z[k - 1:-1].reshape(k, m)
    for _ in range(starts):
        z0 = np.r_[rng.dirichlet(np.ones(k))[:-1], rng.uniform(0.02, 0.98, k * m), 0.5]
        G = lambda z: target_fn(lcm_law(*unpack(z)))
        cons = [{"type": "ineq", "fun": lambda z: target - z[-1] * G(z)}, {"type": "ineq", "fun": lambda z: 1 - z[:k - 1].sum()}]
        r = minimize(lambda z: -z[-1], z0, constraints=cons, bounds=[(0, 1)] * z0.size, method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
        if np.all(target - r.x[-1] * G(r.x) > -1e-9) and r.x[-1] > best[0]: best = (float(r.x[-1]), *unpack(r.x))
    return best
