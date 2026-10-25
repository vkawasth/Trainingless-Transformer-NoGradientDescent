"""The overlap algebra at level two: merging laws over hypotheses.

Context i carries Pi_i, a finite law over tables p on its variables (its hypotheses). A level-two lift (Theorem 19(c)) is a coupling x
of the Pi_i supported on admissible tuples (tuples of tables that glue at level one), together with a filler for each coupled tuple.
So the level-two fibre has two parts:
  * the coupling polytope C(J): couplings of the Pi_i, i in J, on admissible tuples (which hypothesis goes with which);
  * the filler fibres Phi_1(tau) of the coupled tuples (how each matched set of hypotheses is joined).
Flattening sends a level-two lift to a level-one lift of the mu(Pi_i), so level two can only narrow the merge.

  admissible(domains, contexts, tol)      the tuples tau with lam*(tau) >= 1 - tol (tol = 0: exact gluing; tol > 0: within noise)
  coupling_polytope(...)                  tuples, the marginal constraints, and lam2* (largest admissible sub-coupling)
  coupling_dim(...)                       dimension of C(J) (0 = the matching of hypotheses is forced)
  overlap_dim(domains, contexts, J, K)    dim of the projection of C(K) onto the tuples of J, and dim C(J)
  merge2(domains, contexts, tol)          a level-two merge: the maximum-entropy coupling on admissible tuples, each tuple glued by its
                                          maximum-entropy filler; returns the law over global tables and its flattening
  merge1(domains, contexts)               the level-one shortcut: average each context first, then glue (maximum entropy)
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog
from .coherence import Window
from .integrate import gluable_fraction, maxent_merge
from .lift import _canon


def _setup(domains, contexts):
    ctx = [(tuple(v), _canon(Pi, v)) for v, Pi in contexts]
    names = sorted({v for vs, _ in ctx for v in vs})
    return ctx, names, {v: domains[v] for v in names}


def admissible(domains, contexts, tol=0.0):
    ctx, names, dom = _setup(domains, contexts)
    out = []
    for tau in itertools.product(*[range(len(Pi)) for _, Pi in ctx]):
        W = [Window(ctx[i][0], ctx[i][1][tau[i]][0]) for i in range(len(ctx))]
        if gluable_fraction(dom, W).lam >= 1 - tol - 1e-9: out.append(tau)
    return out, ctx, names, dom


def coupling_polytope(domains, contexts, tol=0.0):
    tuples, ctx, names, dom = admissible(domains, contexts, tol)
    rows, rhs = [], []
    for i, (_, Pi) in enumerate(ctx):
        for s, (_, w) in enumerate(Pi):
            rows.append([1.0 if tau[i] == s else 0.0 for tau in tuples]); rhs.append(w)
    A = np.array(rows) if tuples else np.zeros((len(rhs), 0)); b = np.array(rhs)
    lam2 = 0.0
    if tuples:
        r = linprog(-np.ones(len(tuples)), A_ub=A, b_ub=b, bounds=(0, None), method="highs"); lam2 = float(-r.fun)
    return {"tuples": tuples, "A": A, "b": b, "lam2": min(lam2, 1.0), "ctx": ctx, "names": names, "dom": dom}


def _affine_coupling(cp, tol=1e-9):
    A, b, n = cp["A"], cp["b"], len(cp["tuples"])
    if n == 0 or cp["lam2"] < 1 - 1e-7: return None
    S = []
    for k in range(n):
        e = np.zeros(n); e[k] = -1
        r = linprog(e, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        if -r.fun > tol: S.append(k)
    M = A[:, S]; u, s, vt = np.linalg.svd(M); rank = int((s > 1e-10).sum())
    return S, vt[rank:].T


def coupling_dim(domains, contexts, tol=0.0):
    aff = _affine_coupling(coupling_polytope(domains, contexts, tol))
    return -1 if aff is None else int(aff[1].shape[1])


def overlap_dim(domains, contexts, J, K, tol=0.0):
    """dim of pi_J C(K) (projection of couplings of the contexts in K onto those in J) and dim C(J); -1 = empty."""
    cK = coupling_polytope(domains, [contexts[k] for k in K], tol)
    cJ = coupling_polytope(domains, [contexts[j] for j in J], tol)
    aJ = _affine_coupling(cJ); dJ = -1 if aJ is None else int(aJ[1].shape[1])
    aK = _affine_coupling(cK)
    if aK is None: return -1, dJ
    S, N = aK; pos = [list(K).index(j) for j in J]
    tJ = {t: i for i, t in enumerate(cJ["tuples"])}
    P = np.zeros((len(cJ["tuples"]), len(S)))
    for c, k in enumerate(S):
        sub = tuple(cK["tuples"][k][p] for p in pos)
        if sub in tJ: P[tJ[sub], c] = 1.0
    return (int(np.linalg.matrix_rank(P @ N, 1e-9)) if N.size else 0), dJ


def _ipf_coupling(cp, sweeps=5000, tol=1e-12):
    """maximum-entropy coupling on the admissible tuples (iterative proportional fitting from uniform)."""
    tuples, ctx = cp["tuples"], cp["ctx"]
    x = np.full(len(tuples), 1.0 / len(tuples))
    for _ in range(sweeps):
        for i, (_, Pi) in enumerate(ctx):
            for s, (_, w) in enumerate(Pi):
                idx = [k for k, t in enumerate(tuples) if t[i] == s]; m = x[idx].sum()
                if m > 0: x[idx] *= w / m
        err = max(abs(sum(x[k] for k, t in enumerate(tuples) if t[i] == s) - w) for i, (_, Pi) in enumerate(ctx) for s, (_, w) in enumerate(Pi))
        if err < tol: break
    return x


def merge2(domains, contexts, tol=0.0):
    cp = coupling_polytope(domains, contexts, tol)
    if cp["lam2"] < 1 - 1e-7: return None, None, cp
    x = _ipf_coupling(cp); ctx, names, dom = cp["ctx"], cp["names"], cp["dom"]
    law = []
    for tau, xv in zip(cp["tuples"], x):
        if xv <= 1e-12: continue
        W = [Window(ctx[i][0], ctx[i][1][tau[i]][0]) for i in range(len(ctx))]
        q, atoms, _, _ = maxent_merge(dom, W)
        T = np.zeros([dom[v] for v in names])
        for a, v in zip(atoms, q): T[a] = v
        law.append((T, float(xv)))
    flat = sum(w * T for T, w in law)
    return law, flat, cp


def merge1(domains, contexts):
    ctx, names, dom = _setup(domains, contexts)
    W = [Window(vs, sum(w * np.asarray(T, float) for T, w in Pi)) for vs, Pi in ctx]
    q, atoms, _, lam = maxent_merge(dom, W)
    T = np.zeros([dom[v] for v in names])
    for a, v in zip(atoms, q): T[a] = v
    return T, lam


def report2(domains, contexts, tol=0.0):
    """for every subfamily J of the contexts: lam2*(J), lam*(J) of the flattened windows, and whether J is a level-two-only obstruction
    (joins after flattening, not as laws over hypotheses)."""
    out = {}
    n = len(contexts)
    for k in range(1, n + 1):
        for J in itertools.combinations(range(n), k):
            sub = [contexts[j] for j in J]
            cp = coupling_polytope(domains, sub, tol)
            ctx, names, dom = _setup(domains, sub)
            lam1 = gluable_fraction(dom, [Window(vs, sum(w * np.asarray(T, float) for T, w in Pi)) for vs, Pi in ctx]).lam
            out[J] = {"lam2": cp["lam2"], "lam1": lam1,
                      "level_two_only": lam1 > 1 - 1e-7 and cp["lam2"] < 1 - 1e-7}
    return out
