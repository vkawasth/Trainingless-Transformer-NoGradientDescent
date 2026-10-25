"""Cech cochains on the fibres of the tower: separating local incongruence, positivity obstructions, and level-two obstructions.

Level one. A cover is a family of windows (contexts) U_i with tables t_i. For every index set J = (i_0 < ... < i_q), U_J is the intersection
of the U_i (possibly empty; the empty product is one point, carrying total mass). The cochains are

    C^q = direct sum over |J| = q + 1 of the real functions on X_{U_J},

with the alternating Cech differential built from marginalisation (summing out variables). The local tables form a 0-cochain t.

  cech(domains, vars_list, qmax)        the differentials d^0, ..., d^qmax as dense matrices, and the index of every coordinate
  cohomology_dims(...)                   dim H^q = dim ker d^q - rank d^(q-1)
  hodge(...)                             t = t_par + t_perp with t_par in ker d^0 (the nearest compatible family) and t_perp orthogonal;
                                         the local incongruence d^0 t, its energy |d^0 t|^2 and its total variation on each overlap
  signed_extension(domains, windows)     for a compatible family, an explicit signed global law with the given marginals (built from the
                                         Efron-Stein / ANOVA components); None if the family is not compatible
  negativity(domains, windows)           nu* = least negative mass of a signed global law with the given marginals (0 iff the family glues)
  separate(domains, windows)             the obstruction profile: local (a coboundary), positivity (convex, invisible to linear
                                         cohomology), or none (glues, with its fibre dimension)

Level two. Each context carries a finite law over tables. Restriction pushes it forward along marginalisation. Cochains are functions on
the finite set of restricted tables that occur. mu (mixing a law over tables into one table) is a cochain map to level one.

  cech2(...), mu_map(...)               the level-two d^0 and the cochain map mu: C^0_(2) -> C^0_(1) and C^1_(2) -> C^1_(1)
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linprog
from .coherence import Window, _constraints
from .integrate import gluable_fraction


def _space(domains, U):
    U = tuple(sorted(U))
    return U, list(itertools.product(*[range(domains[v]) for v in U]))


def _res(domains, U, V):
    """marginalisation matrix R^{X_U} -> R^{X_V}, V subset of U."""
    U, aU = _space(domains, U); V, aV = _space(domains, V)
    pos = [U.index(v) for v in V]; iV = {a: k for k, a in enumerate(aV)}
    M = np.zeros((len(aV), len(aU)))
    for j, a in enumerate(aU): M[iV[tuple(a[p] for p in pos)], j] = 1.0
    return M


def cech(domains, vars_list, qmax=2):
    n = len(vars_list)
    simp = {q: list(itertools.combinations(range(n), q + 1)) for q in range(qmax + 2)}
    U = {J: frozenset.intersection(*[frozenset(vars_list[i]) for i in J]) for q in simp for J in simp[q]}
    off = {}
    for q in simp:
        o = 0; off[q] = {}
        for J in simp[q]:
            k = len(_space(domains, U[J])[1]); off[q][J] = (o, o + k); o += k
        off[q]["dim"] = o
    ds = []
    for q in range(qmax + 1):
        d = np.zeros((off[q + 1]["dim"], off[q]["dim"]))
        for J in simp[q + 1]:
            r0, r1 = off[q + 1][J]
            for k in range(len(J)):
                F = J[:k] + J[k + 1:]; c0, c1 = off[q][F]
                d[r0:r1, c0:c1] += (-1) ** k * _res(domains, U[F], U[J])
        ds.append(d)
    return ds, off, U


def cohomology_dims(ds, tol=1e-9):
    out = []
    for q, d in enumerate(ds):
        rk = np.linalg.matrix_rank(d, tol) if d.size else 0
        prev = np.linalg.matrix_rank(ds[q - 1], tol) if q > 0 and ds[q - 1].size else 0
        out.append(d.shape[1] - rk - prev)
    return out


def _flat(w: Window):
    """the table of a window as a vector over the atoms of its sorted variables (the coordinates of cech())."""
    U = tuple(sorted(w.vars)); perm = [list(w.vars).index(v) for v in U]
    return np.transpose(np.asarray(w.table, float), perm).ravel()


def cochain(domains, windows):
    """the 0-cochain of the local tables, in the coordinates of cech()."""
    return np.concatenate([_flat(w) for w in windows])


@dataclass
class Hodge:
    t_par: np.ndarray
    t_perp: np.ndarray
    energy: float                 # |d^0 t|^2 = t' L t, L = d0' d0 the sheaf Laplacian
    edge_tv: dict                 # overlap (i, j) -> total variation between the two restrictions


def hodge(domains, windows):
    ds, off, U = cech(domains, [w.vars for w in windows], qmax=0)
    d0 = ds[0]; t = cochain(domains, windows)
    P = np.linalg.pinv(d0) @ d0                                          # projector onto the row space = (ker d0)-perp
    t_perp = P @ t; t_par = t - t_perp; e = d0 @ t
    edge = {J: 0.5 * float(np.abs(e[off[1][J][0]:off[1][J][1]]).sum()) for J in off[1] if J != "dim"}
    return Hodge(t_par, t_perp, float(e @ e), edge)


# ------------------------------------------------------------------------------------------------- signed extension
def _anova(table, U, N):
    """Efron-Stein components of a function on X_U (uniform reference): {S: (S, {x_S: value})}, with f = sum over S of f_S."""
    out = {}
    def cond_mean(T):
        axes = tuple(k for k, v in enumerate(U) if v not in T)
        return table.mean(axis=axes) if axes else table
    for k in range(len(U) + 1):
        for S in itertools.combinations(U, k):
            vals = {}
            for xs in itertools.product(*[range(N[v]) for v in S]):
                acc = 0.0
                for j in range(len(S) + 1):
                    for T in itertools.combinations(S, j):
                        m = cond_mean(T)
                        acc += (-1) ** (len(S) - len(T)) * float(m[tuple(xs[S.index(v)] for v in U if v in T)] if T else m)
                vals[xs] = acc
            out[frozenset(S)] = (S, vals)
    return out


def signed_extension(domains, windows, tol=1e-9):
    """for a compatible family, the signed global law g = sum over S of g_S with g_S = (t_i)_S / N_{V minus U_i} for any window i whose
    variables contain S (well defined by compatibility), g_S = 0 if no window contains S. Returns (g, atoms) or None if incompatible."""
    if hodge(domains, windows).energy > tol: return None
    names = list(domains); N = dict(domains); NV = float(np.prod([N[v] for v in names]))
    comp = {}
    for w in windows:
        U = tuple(w.vars); NU = float(np.prod([N[v] for v in U]))
        for S, (Sv, vals) in _anova(np.asarray(w.table, float), U, N).items():
            comp.setdefault(S, (Sv, {k: v * NU / NV for k, v in vals.items()}))
    atoms = list(itertools.product(*[range(N[v]) for v in names]))
    g = np.array([sum(vals[tuple(dict(zip(names, a))[v] for v in Sv)] for Sv, vals in comp.values()) for a in atoms])
    return g, atoms


def negativity(domains, windows):
    A, t, atoms, _ = _constraints(domains, windows)
    n = A.shape[1]; Aeq = np.vstack([np.hstack([A, -A]), np.concatenate([np.ones(n), -np.ones(n)])])
    beq = np.concatenate([t, [1.0]])
    r = linprog(np.concatenate([np.zeros(n), np.ones(n)]), A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    return float(r.fun) if r.status == 0 else None


@dataclass
class Profile:
    kind: str                     # "local", "positivity", "none"
    energy: float
    edge_tv: dict
    lam: float
    nu: float | None
    witness: dict | None = None
    details: dict = field(default_factory=dict)


def separate(domains, windows, tol=1e-9):
    h = hodge(domains, windows); g = gluable_fraction(domains, windows)
    if h.energy > tol:
        return Profile("local", h.energy, h.edge_tv, g.lam, None)
    nu = negativity(domains, windows)
    if g.lam < 1 - 1e-7:
        return Profile("positivity", 0.0, h.edge_tv, g.lam, nu, g.witness)
    return Profile("none", 0.0, h.edge_tv, 1.0, nu)


# --------------------------------------------------------------------------------------------------------------- level two
def _tab(domains, vars_, T, V):
    """marginal of table T (axes in the order vars_) on the sorted variable set V, as a vector."""
    return _res(domains, tuple(sorted(vars_)), V) @ _flat(Window(tuple(vars_), np.asarray(T, float)))


def cech2(domains, contexts):
    """contexts: list of (vars, [(table, weight), ...]). Returns (d0 at level two, basis per overlap, the level-two 0-cochain,
    d0 at level one, and the matrices of mu on C^0 and C^1)."""
    n = len(contexts)
    own = [tuple(sorted(v)) for v, _ in contexts]
    pairs = list(itertools.combinations(range(n), 2))
    inter = {(i, j): tuple(sorted(set(own[i]) & set(own[j]))) for i, j in pairs}
    # bases: level-two coordinates are the distinct tables occurring (own context) or as restrictions (overlaps)
    def key(v): return tuple(np.round(v, 12))
    base0 = []
    for i, (vs, Pi) in enumerate(contexts):
        b = {}
        for T, w in Pi: b.setdefault(key(_tab(domains, vs, T, own[i])), _tab(domains, vs, T, own[i]))
        base0.append(b)
    base1 = {}
    for (i, j) in pairs:
        b = {}
        for k in (i, j):
            for kk, v in base0[k].items():
                r = _res(domains, own[k], inter[(i, j)]) @ v; b.setdefault(key(r), r)
        base1[(i, j)] = b
    o0 = np.cumsum([0] + [len(b) for b in base0]); o1 = np.cumsum([0] + [len(base1[p]) for p in pairs])
    d0 = np.zeros((o1[-1], o0[-1]))
    for pi, (i, j) in enumerate(pairs):
        keys1 = list(base1[(i, j)])
        for sgn, k in ((-1, i), (1, j)):
            for c, (kk, v) in enumerate(base0[k].items()):
                r = _res(domains, own[k], inter[(i, j)]) @ v
                d0[o1[pi] + keys1.index(key(r)), o0[k] + c] += sgn
    c0 = np.zeros(o0[-1])
    for i, (vs, Pi) in enumerate(contexts):
        keys0 = list(base0[i])
        for T, w in Pi: c0[o0[i] + keys0.index(key(_tab(domains, vs, T, own[i])))] += w
    # level one
    dims0 = [len(_space(domains, u)[1]) for u in own]; dims1 = [len(_space(domains, inter[p])[1]) for p in pairs]
    p0 = np.cumsum([0] + dims0); p1 = np.cumsum([0] + dims1)
    d0_1 = np.zeros((p1[-1], p0[-1]))
    for pi, (i, j) in enumerate(pairs):
        d0_1[p1[pi]:p1[pi + 1], p0[i]:p0[i + 1]] -= _res(domains, own[i], inter[(i, j)])
        d0_1[p1[pi]:p1[pi + 1], p0[j]:p0[j + 1]] += _res(domains, own[j], inter[(i, j)])
    mu0 = np.zeros((p0[-1], o0[-1]))
    for i in range(n):
        for c, v in enumerate(base0[i].values()): mu0[p0[i]:p0[i + 1], o0[i] + c] = v
    mu1 = np.zeros((p1[-1], o1[-1]))
    for pi, p in enumerate(pairs):
        for c, v in enumerate(base1[p].values()): mu1[p1[pi]:p1[pi + 1], o1[pi] + c] = v
    return {"d0_2": d0, "c0_2": c0, "d0_1": d0_1, "mu0": mu0, "mu1": mu1, "pairs": pairs}
