"""Contextuality beyond direct influence (Contextuality-by-Default, Dzhafarov and Kujala), for windows of binary variables in which each
variable appears in at most two windows (cyclic systems and their sub-systems).

Why it is needed. In PCE a variable's marginal must agree across the windows that contain it; if it does not (the elicited P(X1) moves
with the partner X0 or X2 -- "direct influence", "signalling", "inconsistent connectedness"), the family cannot glue and lam* < 1 for a
reason that has nothing to do with contextuality. Contextuality-by-Default treats each variable in each window as its own copy and asks
whether the copies of the same variable can be coupled as closely as their marginals allow (maximal coupling) while every window keeps its
table. If yes, the incoherence is explained by direct influence alone.

  copies(windows)                      the copies: (variable, window index) for every occurrence
  signalling(windows)                  s_q = TV between the marginals of variable q's two copies (0 for a variable in one window)
  cbd(windows)                         M = max over couplings of the copies (window tables fixed) of sum_q P(copies of q agree);
                                       c = sum_q (1 - s_q) - M >= 0. c = 0: noncontextual (direct influence explains everything);
                                       c > 0: contextual. One LP over the 2^(number of copies) atoms.
  cbd_tolerance(eps)                   4 sum_w eps_w: if each window moves by at most eps_w in TV, c moves by at most this
  kd_cyclic(windows_in_cycle_order)    the Kujala-Dzhafarov criterion for cyclic systems of rank n:
                                       contextual iff s_odd(<R_i R_i+1>) > n - 2 + Delta, Delta = sum_i |<R_i>_ctx1 - <R_i>_ctx2| (+-1 units)
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog


def copies(windows):
    return [(v, wi) for wi, w in enumerate(windows) for v in w.vars]


def _marg(w, v):
    T = np.asarray(w.table, float); k = list(w.vars).index(v)
    return T.sum(axis=tuple(i for i in range(T.ndim) if i != k))


def signalling(windows):
    occ = {}
    for wi, w in enumerate(windows):
        for v in w.vars: occ.setdefault(v, []).append(wi)
    s = {}
    for v, ws in occ.items():
        assert len(ws) <= 2, "each variable may appear in at most two windows"
        s[v] = 0.0 if len(ws) == 1 else 0.5 * float(np.abs(_marg(windows[ws[0]], v) - _marg(windows[ws[1]], v)).sum())
    return s, occ


def cbd(windows):
    s, occ = signalling(windows)
    C = copies(windows); n = len(C); atoms = list(itertools.product((0, 1), repeat=n))
    rows, rhs = [], []
    for wi, w in enumerate(windows):
        idx = [C.index((v, wi)) for v in w.vars]
        for cell in itertools.product((0, 1), repeat=len(w.vars)):
            rows.append([1.0 if tuple(a[i] for i in idx) == cell else 0.0 for a in atoms]); rhs.append(float(np.asarray(w.table)[cell]))
    obj = np.zeros(len(atoms))
    pairs = {v: (C.index((v, ws[0])), C.index((v, ws[1]))) for v, ws in occ.items() if len(ws) == 2}
    for v, (i, j) in pairs.items():
        obj += np.array([1.0 if a[i] == a[j] else 0.0 for a in atoms])
    Aeq = np.vstack([np.array(rows), np.ones(len(atoms))]); beq = np.concatenate([rhs, [1.0]])
    r = linprog(-obj, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    if r.status != 0: return {"s": s, "M": float("nan"), "c": float("nan"), "status": "infeasible"}
    M = float(-r.fun); best = sum(1 - s[v] for v in pairs)
    return {"s": s, "M": M, "c": max(0.0, best - M), "max_agree": best, "status": "ok"}


def cbd_tolerance(eps):
    return 4.0 * float(sum(eps))


def kd_cyclic(windows):
    """windows in cycle order: window i holds (R_i, R_{i+1}), indices mod n, binary 0/1 mapped to +1/-1 as 1 - 2x."""
    n = len(windows); corr, mean_in, mean_out = [], {}, {}
    for i, w in enumerate(windows):
        T = np.asarray(w.table, float); sgn = np.array([1.0, -1.0])
        corr.append(float(sum(T[a, b] * sgn[a] * sgn[b] for a in (0, 1) for b in (0, 1))))
        a, b = w.vars
        mean_in.setdefault(a, []).append(float(sgn @ T.sum(axis=1))); mean_in.setdefault(b, []).append(float(sgn @ T.sum(axis=0)))
    delta = sum(abs(m[0] - m[1]) for m in mean_in.values() if len(m) == 2)
    sodd = max(sum(sg * x for sg, x in zip(signs, corr)) for signs in itertools.product((1, -1), repeat=n) if signs.count(-1) % 2 == 1)
    return sodd > n - 2 + delta + 1e-12, sodd, delta


def cbd_soft(windows, eps):
    """theta_c: the smallest theta such that some noncontextual system (copies maximally coupled) has window tables within theta * eps_w
    of the observed ones, in TV. theta_c = 0 iff c = 0; theta_c > 1: contextual beyond the tolerances. Exact: a maximal coupling of two
    binary copies has one discordant cell empty, so the noncontextual systems are a union of 2^k polytopes (k shared variables), and
    theta_c is the minimum of one LP per orientation. Returns (theta_c, orientation attaining it)."""
    s, occ = signalling(windows)
    C = copies(windows); n = len(C); atoms = list(itertools.product((0, 1), repeat=n)); na = len(atoms)
    cells = []
    for wi, w in enumerate(windows):
        idx = [C.index((v, wi)) for v in w.vars]
        for cell in itertools.product((0, 1), repeat=len(w.vars)):
            cells.append((wi, [1.0 if tuple(a[i] for i in idx) == cell else 0.0 for a in atoms], float(np.asarray(w.table)[cell])))
    m = len(cells); k = len(windows)
    pairs = [(C.index((v, ws[0])), C.index((v, ws[1]))) for v, ws in occ.items() if len(ws) == 2]
    best, arg = np.inf, None
    for orient in itertools.product((0, 1), repeat=len(pairs)):
        # variables: Q (na), s+ (m), s- (m), theta
        nv = na + 2 * m + 1; c = np.zeros(nv); c[-1] = 1.0
        Aeq = np.zeros((m + 1, nv)); beq = np.zeros(m + 1)
        for r, (wi, row, val) in enumerate(cells):
            Aeq[r, :na] = row; Aeq[r, na + r] = -1.0; Aeq[r, na + m + r] = 1.0; beq[r] = val
        Aeq[m, :na] = 1.0; beq[m] = 1.0
        Aub = np.zeros((k, nv))
        for r, (wi, _, _) in enumerate(cells): Aub[wi, na + r] = 0.5; Aub[wi, na + m + r] = 0.5
        for wi in range(k): Aub[wi, -1] = -max(eps[wi], 1e-12)
        ub = [(0, None)] * nv
        for (i, j), o in zip(pairs, orient):                                # the discordant cell (copy_i = o, copy_j = 1 - o) is empty
            for a_i, a in enumerate(atoms):
                if a[i] == o and a[j] == 1 - o: ub[a_i] = (0, 0)
        r = linprog(c, A_ub=Aub, b_ub=np.zeros(k), A_eq=Aeq, b_eq=beq, bounds=ub, method="highs")
        if r.status == 0 and r.fun < best: best, arg = float(r.fun), orient
    return best, arg
