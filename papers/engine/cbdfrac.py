"""Gluing fractions that allow for direct influence: the classical and quantum fractions of the consistified system.

Contextuality-by-Default keeps one copy of a variable per window. Consistification (Dzhafarov and Kujala) adds, for every variable seen
in two windows, a connection: a further window on the pair of copies whose table is their maximal coupling (the coupling under which the
copies agree as often as their marginals allow). The consistified system is consistently connected, so the ordinary fractions apply to it.

  maximal_coupling(p, q)          2x2 table of two binary copies with P(+) = p and q, agreeing with probability 1 - |p - q|
  consistify(windows)             (copies, edges): copies are (variable, window index); edges are (i, j, 2x2 table) for every window
                                  (a pair of copies) and every connection; each window must hold exactly two binary variables
  lam_classical_cbd(windows)      largest mass of a sub-measure on the copies' joint outcomes lying below every edge table:
                                  1 - CNTF in the sense of Dzhafarov, Kujala and Cervantes. It is 1 iff the system is noncontextual (c = 0)
  edge_npa_fraction(n, edges, level, extra)
                                  NPA relaxation: +-1 observables X_0..X_{n-1}, X_i X_j = X_j X_i exactly along edges (the right-angled
                                  Coxeter relations), unnormalised moment matrix with <1> = lam, every edge's cells between 0 and the
                                  edge table; maximise lam. level: "1+E" (1, X_i, products of commuting pairs) or an integer word length
  lam_quantum_cbd(windows, level, parties)
                                  edge_npa_fraction of the consistified system: the dimension-free quantum fraction allowing direct
                                  influence. Copies commute with what they are measured with (window partner, other copy) and, if
                                  parties is given (variable -> label; bipartite() for A<x>/B<y> names), with every copy of a variable of
                                  another party. With parties, perfectly coupled copies reduce to the Bell quantum set (lam_Q,cbd = lam_Q
                                  when there is no direct influence). Without parties (minimal commutation) the copies may agree on the
                                  state without being the same operator, and the quantum set is larger even with no direct influence
  edge_npa_value(n, edges, coeffs, level)
                                  max of sum_e coeffs[e] <X_i X_j> over the relaxation (normalised), used to test the relaxation
  decomposition(domains, windows) 1 - lam* split into direct influence (lam_C,cbd - lam*), quantum-explainable contextuality
                                  (lam_Q,cbd - lam_C,cbd) and contextuality beyond quantum (1 - lam_Q,cbd)

Copies are coded 0 -> +1, 1 -> -1 (row/column 0 of a table is the +1 outcome).
"""
from __future__ import annotations
import itertools
from functools import lru_cache
import numpy as np
from scipy.optimize import linprog


def maximal_coupling(p, q):
    a = min(p, q); d = min(1 - p, 1 - q)
    return np.array([[a, p - a], [q - a, d]])


def _marg(table, k):
    T = np.asarray(table, float); return T.sum(axis=1 - k)


def consistify(windows):
    copies, edges, occ = [], [], {}
    for wi, w in enumerate(windows):
        assert len(w.vars) == 2, "pairwise windows only"
        T = np.asarray(w.table, float); assert T.shape == (2, 2), "binary variables only"
        idx = []
        for k, v in enumerate(w.vars):
            copies.append((v, wi)); idx.append(len(copies) - 1); occ.setdefault(v, []).append((len(copies) - 1, _marg(T, k)[0]))
        edges.append((idx[0], idx[1], T))
    for v, oc in occ.items():
        assert len(oc) <= 2, "each variable may appear in at most two windows"
        if len(oc) == 2:
            (i, p), (j, q) = oc; edges.append((i, j, maximal_coupling(p, q)))
    return copies, edges


def lam_classical_cbd(windows):
    copies, edges = consistify(windows); n = len(copies); atoms = list(itertools.product((0, 1), repeat=n))
    A, b = [], []
    for i, j, T in edges:
        for a, c in itertools.product((0, 1), repeat=2):
            A.append([1.0 if (at[i] == a and at[j] == c) else 0.0 for at in atoms]); b.append(float(T[a, c]))
    r = linprog(-np.ones(len(atoms)), A_ub=np.array(A), b_ub=np.array(b), bounds=(0, None), method="highs")
    return float(-r.fun)


# ---------- NPA over a right-angled Coxeter group (commutation along edges, X^2 = 1) ----------
def _make_reducer(n, comm):
    @lru_cache(maxsize=None)
    def canon(w):
        seen = {w}; stack = [w]
        while stack:
            u = stack.pop()
            for k in range(len(u) - 1):
                if u[k] == u[k + 1]: nb = u[:k] + u[k + 2:]
                elif (u[k], u[k + 1]) in comm: nb = u[:k] + (u[k + 1], u[k]) + u[k + 2:]
                else: continue
                if nb not in seen: seen.add(nb); stack.append(nb)
        m = min(len(u) for u in seen); return min(u for u in seen if len(u) == m)
    def key(w):  # real relaxation: identify a word with its reverse (adjoint)
        a = canon(tuple(w)); b = canon(tuple(reversed(w))); return min(a, b)
    return key, canon


def _words(n, comm, level):
    if level == "1+E":
        W = [()] + [(i,) for i in range(n)] + sorted({(min(i, j), max(i, j)) for i, j in comm})
    else:
        W = [()]
        for L in range(1, int(level) + 1): W += list(itertools.product(range(n), repeat=L))
    return W


def _relaxation(n, edges, level, extra=()):
    import cvxpy as cp
    comm = set()
    for i, j in [(i, j) for i, j, _ in edges] + list(extra): comm.add((i, j)); comm.add((j, i))
    key, canon = _make_reducer(n, comm); W0 = _words(n, comm, level)
    W, seen = [], set()
    for w in W0:                      # one row per group element
        c = canon(tuple(w))
        if c not in seen: seen.add(c); W.append(c)
    m = len(W); var = {}; entries = {}
    for a in range(m):
        for b_ in range(a, m):
            k = key(tuple(reversed(W[a])) + tuple(W[b_])); entries[(a, b_)] = k; var.setdefault(k, None)
    names = list(var); index = {k: t for t, k in enumerate(names)}; y = cp.Variable(len(names))
    G = [[None] * m for _ in range(m)]
    for (a, b_), k in entries.items(): G[a][b_] = y[index[k]]; G[b_][a] = y[index[k]]
    Gm = cp.bmat(G); val = lambda w: y[index[key(tuple(w))]]
    return cp, Gm, val, m


_CACHE = {}


def _fraction_problem(n, pairs, level, extra):
    """The relaxation depends on the tables only through the right-hand sides; build it once per structure, with the tables as
    parameters (compiled once by cvxpy), and reuse it.

    Numerical note. Every maximal coupling has a cell of probability zero, so the program has no strictly feasible point. Interior-point
    and first-order solvers then stop at 'optimal_inaccurate', with values reproducible to about 2e-4 (measured: Clarabel against SCS,
    and under recoding). One step of facial reduction did not restore strict feasibility; this is left open."""
    k = (n, pairs, level, tuple(extra))
    if k in _CACHE: return _CACHE[k]
    import cvxpy as cp
    import scipy.sparse as sp
    comm = set()
    for i, j in list(pairs) + list(extra): comm.add((i, j)); comm.add((j, i))
    key, canon = _make_reducer(n, comm); W, seen = [], set()
    for w in _words(n, comm, level):
        c = canon(tuple(w))
        if c not in seen: seen.add(c); W.append(c)
    m = len(W); index = {}; rows, cols = [], []
    for a in range(m):
        for b_ in range(m):
            kk = key(tuple(reversed(W[a])) + tuple(W[b_])); t = index.setdefault(kk, len(index)); rows.append(a * m + b_); cols.append(t)
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(m * m, len(index)))
    y = cp.Variable(len(index)); G = cp.reshape(M @ y, (m, m), order="C"); val = lambda w: y[index[key(tuple(w))]]
    T = cp.Parameter(4 * len(pairs)); lam = val(()); mus = []
    for e, (i, j) in enumerate(pairs):
        for ia, a in enumerate((1, -1)):
            for ib, b in enumerate((1, -1)):
                mus.append((lam + a * val((i,)) + b * val((j,)) + a * b * val((i, j))) / 4)
    mu = cp.hstack(mus)
    pr = cp.Problem(cp.Maximize(lam), [G >> 0, mu <= T, mu >= 0])
    _CACHE[k] = (pr, T); return _CACHE[k]


def edge_npa_fraction(n, edges, level="1+E", extra=(), solver="CLARABEL"):
    pairs = tuple((int(i), int(j)) for i, j, _ in edges)
    pr, T = _fraction_problem(n, pairs, level, tuple(extra))
    T.value = np.concatenate([np.clip(np.asarray(t, float).ravel(), 0, None) for _, _, t in edges])
    try:
        pr.solve(solver=solver) if solver != "SCS" else pr.solve(solver="SCS", eps=1e-9, max_iters=200000)
        if pr.status not in ("optimal", "optimal_inaccurate"): raise RuntimeError(pr.status)
    except Exception:
        pr.solve(solver="SCS", eps=1e-9, max_iters=200000)
    return float(pr.value)


def _edge_npa_fraction_reference(n, edges, level="1+E", extra=()):
    """the original (uncached, scalar-built) implementation, kept to check that the cached one is identical"""
    cp, Gm, val, m = _relaxation(n, edges, level, extra); lam = val(())
    cons = [(Gm + Gm.T) / 2 >> 0]
    for i, j, T in edges:
        for ia, a in enumerate((1, -1)):
            for ib, b in enumerate((1, -1)):
                mu = (lam + a * val((i,)) + b * val((j,)) + a * b * val((i, j))) / 4
                cons += [mu <= float(T[ia, ib]), mu >= 0]
    pr = cp.Problem(cp.Maximize(lam), cons); pr.solve(solver="SCS", eps=1e-9, max_iters=200000)
    return float(pr.value)


def edge_npa_value(n, edges, coeffs, level="1+E"):
    cp, Gm, val, m = _relaxation(n, edges, level)
    cons = [(Gm + Gm.T) / 2 >> 0, val(()) == 1]
    obj = sum(c * val((i, j)) for (i, j, _), c in zip(edges, coeffs))
    pr = cp.Problem(cp.Maximize(obj), cons); pr.solve(solver="SCS", eps=1e-9, max_iters=200000)
    return float(pr.value)


def party_commutations(copies, parties):
    """extra commuting pairs: copies of variables belonging to different parties (parties: variable -> label)"""
    return [(i, j) for i in range(len(copies)) for j in range(i + 1, len(copies)) if parties(copies[i][0]) != parties(copies[j][0])]


def lam_quantum_cbd(windows, level="1+E", parties=None):
    copies, edges = consistify(windows)
    extra = party_commutations(copies, parties) if parties is not None else ()
    return edge_npa_fraction(len(copies), edges, level, extra)


def bipartite(v):
    """party of a variable named A<x> or B<y> (Bell-type systems)"""
    return v[0]


def decomposition(domains, windows, level="1+E", parties=None):
    from .integrate import gluable_fraction
    lam = float(gluable_fraction(domains, windows).lam); lc = lam_classical_cbd(windows); lq = lam_quantum_cbd(windows, level, parties)
    return {"lam_classical": lam, "lam_classical_cbd": lc, "lam_quantum_cbd": lq,
            "share_direct_influence": lc - lam, "share_quantum_contextual": lq - lc, "share_beyond_quantum": 1 - lq}
