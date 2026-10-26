"""Decisions on a time-varying sheaf of evidence, with covers chosen by robust satisficing.

At each time t the evidence is a family of windows (scopes and tables) over variables with finite domains: a presheaf of local laws on the
context complex whose faces are the windows. A step t -> t' may
  * add windows (new contexts observed),
  * refine outcomes (awareness growth): a map r_v from the new outcomes of variable v onto its old outcomes, and
  * drift: the new tables, coarsened along r and restricted to the old windows, differ from the old tables by at most d in total variation.
The decision maker chooses a cover (a subset of the observed windows to glue) and an act.

Level one, for a cover C and a fraction alpha <= lam*(C):
  E_alpha(C) = {q a law on the global outcomes : alpha q|_w <= t_w for w in C}         (the graded explanations of Section 2)
  V(a; alpha) = alpha min_{q in E_alpha} E_q[u_a] + (1 - alpha) min_x u_a(x)            (epsilon-contamination with epsilon = 1 - alpha:
                                                                                       the unexplained part is treated as adversarial)
  robustness h(a, C) = the largest eps in [1 - lam*(C), 1] such that V(a; 1 - eps') >= u0 for every eps' in [1 - lam*(C), eps]
                       (info-gap robustness; -inf when even the best-supported explanation set fails the requirement)
  robust satisficing: choose the (cover, act) with the largest robustness.

Functions
  System(domains, windows)                        a snapshot of the evidence
  lam(S, cover=None)                              lam* of the cover (default: all windows)
  interval(S, event, alpha, cover=None)           [min, max] of q(event) over E_alpha(cover)
  lam_under(S, f, bound, cover=None)              support of a hypothesis E_q[f] <= bound: the largest alpha whose E_alpha meets it
  polygon_under(S, fx, fy, alpha, f, bound)       the projection of E_alpha cut by the hypothesis
  polygon(S, fx, fy, alpha, cover=None, n=120)    the projection of E_alpha onto (E_q fx, E_q fy) (support-function LPs)
  robust_value(S, u, alpha, cover=None)           V(a; alpha) for the utility u (a function of the global outcome)
  robustness(S, u, u0, cover=None, grid=101)      h(a, C)
  covers(S)                                       all non-empty subsets of the windows (as index tuples)
  frontier(points)                                the non-dominated covers for (lam* high, interval width low)
  coarsen(q, atoms_new, atoms_old, refine)        push a law on the new outcomes to the old ones (refine: {var: {new: old}})
  drift(S_old, S_new, refine, pairs)              max TV over corresponding windows after coarsening
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog
from . import Window
from .coherence import _constraints
from .integrate import gluable_fraction


class System:
    def __init__(self, domains, windows, names=None):
        self.domains = dict(domains); self.windows = list(windows)
        self.names = names or [",".join(w.vars) for w in self.windows]

    def sub(self, cover):
        cover = tuple(range(len(self.windows))) if cover is None else tuple(cover)
        used = {v for i in cover for v in self.windows[i].vars}
        # keep every variable of the system so that outcome spaces (and utilities) are comparable across covers
        return self.domains, [self.windows[i] for i in cover]


def lam(S, cover=None):
    d, W = S.sub(cover); return float(gluable_fraction(d, W).lam)


def _lp_setup(S, cover):
    d, W = S.sub(cover); A, t, atoms, index = _constraints(d, W); names = list(d)
    return A, t, atoms, names


def _vec(f, atoms, names): return np.array([float(f(dict(zip(names, a)))) for a in atoms])


def lam_under(S, constraint_vec_fn, bound, cover=None):
    """hypothesis support: the largest alpha such that E_alpha(cover) contains a law q with E_q[f] <= bound (f = constraint_vec_fn);
    one LP in the sub-measure b = alpha q: max sum b, A b <= t, f.b <= bound * sum b"""
    A, t, atoms, names = _lp_setup(S, cover); f = _vec(constraint_vec_fn, atoms, names); m = len(atoms)
    Aub = np.vstack([A, (f - bound)[None, :]]); bub = np.r_[t, 0.0]
    r = linprog(-np.ones(m), A_ub=Aub, b_ub=bub, bounds=(0, None), method="highs")
    return float(-r.fun) if r.status == 0 else 0.0


def polygon_under(S, fx, fy, alpha, constraint_vec_fn, bound, cover=None, n=120):
    """projection of E_alpha intersected with the hypothesis {E_q[f] <= bound}; None if empty"""
    A, t, atoms, names = _lp_setup(S, cover); X = _vec(fx, atoms, names); Y = _vec(fy, atoms, names); f = _vec(constraint_vec_fn, atoms, names)
    m = len(atoms); pts = []
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        r = linprog(-(np.cos(th) * X + np.sin(th) * Y), A_ub=np.vstack([alpha * A, f[None, :]]), b_ub=np.r_[t, bound],
                    A_eq=np.ones((1, m)), b_eq=[1.0], bounds=(0, None), method="highs")
        if r.status != 0: return None
        pts.append((float(X @ r.x), float(Y @ r.x)))
    return np.array(pts)


def interval(S, event, alpha, cover=None):
    A, t, atoms, names = _lp_setup(S, cover); e = _vec(event, atoms, names); n = len(atoms); out = []
    for sg in (1.0, -1.0):
        r = linprog(sg * e, A_ub=alpha * A, b_ub=t, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=(0, None), method="highs")
        if r.status != 0: return None
        out.append(sg * r.fun)
    return out[0], out[1]


def polygon(S, fx, fy, alpha, cover=None, n=120):
    A, t, atoms, names = _lp_setup(S, cover); X = _vec(fx, atoms, names); Y = _vec(fy, atoms, names); m = len(atoms); pts = []
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        c = -(np.cos(th) * X + np.sin(th) * Y)
        r = linprog(c, A_ub=alpha * A, b_ub=t, A_eq=np.ones((1, m)), b_eq=[1.0], bounds=(0, None), method="highs")
        if r.status != 0: return None
        pts.append((float(X @ r.x), float(Y @ r.x)))
    return np.array(pts)


def robust_value(S, u, alpha, cover=None):
    A, t, atoms, names = _lp_setup(S, cover); U = _vec(u, atoms, names); m = len(atoms)
    r = linprog(U, A_ub=alpha * A, b_ub=t, A_eq=np.ones((1, m)), b_eq=[1.0], bounds=(0, None), method="highs")
    if r.status != 0: return None
    return float(alpha * r.fun + (1 - alpha) * U.min())


def robustness(S, u, u0, cover=None, grid=101):
    l = lam(S, cover) - 1e-9; eps = np.linspace(1 - l, 1, grid); h = -np.inf
    for e in eps:
        v = robust_value(S, u, 1 - e, cover)
        if v is None or v < u0 - 1e-12: break
        h = float(e)
    return h


def covers(S):
    k = len(S.windows); return [c for r in range(1, k + 1) for c in itertools.combinations(range(k), r)]


def frontier(points):
    """points: list of (cover, lam, width); returns those not dominated (another cover with lam >= and width <=, one strictly)"""
    out = []
    for c, l, w in points:
        if not any((l2 >= l - 1e-12 and w2 <= w + 1e-12) and (l2 > l + 1e-12 or w2 < w - 1e-12) for c2, l2, w2 in points): out.append((c, l, w))
    return out


def coarsen(q, atoms_new, names_new, atoms_old, names_old, refine):
    idx = {a: i for i, a in enumerate(atoms_old)}; out = np.zeros(len(atoms_old))
    for a, p in zip(atoms_new, q):
        x = dict(zip(names_new, a)); y = tuple(refine.get(v, {}).get(x[v], x[v]) if v in refine else x[v] for v in names_old)
        out[idx[y]] += p
    return out


def coarsen_table(table, vars_, refine):
    T = np.asarray(table, float)
    for ax, v in enumerate(vars_):
        if v in refine:
            mp = refine[v]; k_old = max(mp.values()) + 1; shp = list(T.shape); shp[ax] = k_old; R = np.zeros(shp)
            for new, old in mp.items():
                sl_new = [slice(None)] * T.ndim; sl_new[ax] = new; sl_old = [slice(None)] * T.ndim; sl_old[ax] = old
                R[tuple(sl_old)] += T[tuple(sl_new)]
            T = R
    return T


def drift(S_old, S_new, refine, pairs):
    """pairs: list of (index in S_old, index in S_new) of corresponding windows (same scope)"""
    d = 0.0
    for i, j in pairs:
        wo, wn = S_old.windows[i], S_new.windows[j]
        d = max(d, 0.5 * float(np.abs(coarsen_table(wn.table, wn.vars, refine) - wo.table).sum()))
    return d
