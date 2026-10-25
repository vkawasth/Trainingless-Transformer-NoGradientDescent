"""The gluability complex filtered by gradation, its persistent homology, and how it moves with the evidence (vineyards).

Vertices are windows; a set J of windows is a simplex entering at level phi(J) = 1 - lam*(J) (0 = glues exactly). phi is monotone
(J in K => phi(J) <= phi(K)), so K_eps = {J : phi(J) <= eps} is a simplicial complex for every eps, and eps -> K_eps is a filtration.
A hollow cell (all faces glue better than the cell) is a class born at the largest face value and killed when the cell enters.

  filtration(domains, windows, tol)   {J: phi(J)} for every non-empty J (sorted tuples of window indices); tolerances (optional) give
                                      the graded version phi(J) = theta*(J)
  persistence(filt)                   Z/2 persistent homology: {dim: [(birth, death, creator, killer)]}, death = inf for essential classes
  bottleneck(D1, D2)                  bottleneck distance between two diagrams of one dimension (finite points; small exact matching)
  vineyard(domains, windows_of_s, s_grid)   the diagrams along a path of evidence
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from .integrate import gluable_fraction
from .cover import glue_soft


def _vars(windows, J):
    return sorted({v for j in J for v in windows[j].vars})


def filtration(domains, windows, eps=None):
    out = {}
    n = len(windows)
    for k in range(1, n + 1):
        for J in itertools.combinations(range(n), k):
            dom = {v: domains[v] for v in _vars(windows, J)}; W = [windows[j] for j in J]
            if eps is None: out[J] = max(0.0, 1.0 - gluable_fraction(dom, W).lam)
            else: out[J] = glue_soft(dom, W, [eps[j] for j in J])["theta"]
    for J in sorted(out, key=len):                                       # enforce monotonicity against round-off
        for K in itertools.combinations(J, len(J) - 1):
            if K: out[J] = max(out[J], out[K])
    return out


def persistence(filt, tol=1e-12):
    simp = sorted(filt, key=lambda J: (filt[J], len(J), J)); index = {J: i for i, J in enumerate(simp)}
    cols = []
    for J in simp:
        cols.append(set(index[K] for K in itertools.combinations(J, len(J) - 1)) if len(J) > 1 else set())
    low = {}; pairs = []; reduced = []
    for j, c in enumerate(cols):
        c = set(c)
        while c and max(c) in low: c ^= reduced[low[max(c)]]
        reduced.append(c)
        if c: low[max(c)] = j; pairs.append((max(c), j))
    paired = {i for p in pairs for i in p}; out = {}
    for i, j in pairs:
        b, d = filt[simp[i]], filt[simp[j]]
        if d - b > tol: out.setdefault(len(simp[i]) - 1, []).append((b, d, simp[i], simp[j]))
    for i, J in enumerate(simp):
        if i not in paired: out.setdefault(len(J) - 1, []).append((filt[J], np.inf, J, None))
    return out


def bottleneck(D1, D2):
    """exact bottleneck distance between finite diagrams [(b, d)] (points may also match the diagonal)."""
    A = [(b, d) for b, d in D1 if np.isfinite(d)]; B = [(b, d) for b, d in D2 if np.isfinite(d)]
    if not A and not B: return 0.0
    m, n = len(A), len(B); N = m + n
    def diag(p): return (p[1] - p[0]) / 2
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i < m and j < n: C[i, j] = max(abs(A[i][0] - B[j][0]), abs(A[i][1] - B[j][1]))
            elif i < m: C[i, j] = diag(A[i])
            elif j < n: C[i, j] = diag(B[j])
            else: C[i, j] = 0.0
    vals = np.unique(C)
    lo, hi = 0, len(vals) - 1
    while lo < hi:                                                       # smallest threshold with a perfect matching
        mid = (lo + hi) // 2; M = (C <= vals[mid] + 1e-15).astype(float)
        r, c = linear_sum_assignment(-M)
        if M[r, c].sum() == N: hi = mid
        else: lo = mid + 1
    return float(vals[lo])


def vineyard(domains, windows_of_s, s_grid, eps=None):
    return [(float(s), persistence(filtration(domains, windows_of_s(s), eps))) for s in s_grid]


def filtration_k(domains, windows, kmax, eps=None):
    """phi(J) for every set J of at most kmax windows (the (kmax - 1)-skeleton of the filtered complex). Homology of the skeleton
    equals that of the full complex in dimensions <= kmax - 2, so persistence there is exact; the top dimension is not reported."""
    out = {}; n = len(windows)
    for k in range(1, min(kmax, n) + 1):
        for J in itertools.combinations(range(n), k):
            dom = {v: domains[v] for v in _vars(windows, J)}; W = [windows[j] for j in J]
            if eps is None: out[J] = max(0.0, 1.0 - gluable_fraction(dom, W).lam)
            else: out[J] = glue_soft(dom, W, [eps[j] for j in J])["theta"]
    for J in sorted(out, key=len):
        for K in itertools.combinations(J, len(J) - 1):
            if K: out[J] = max(out[J], out[K])
    return out


def persistence_k(filt, kmax):
    """persistence of the truncated filtration, keeping only dimensions <= kmax - 2 (exact there)."""
    P = persistence(filt); return {d: v for d, v in P.items() if d <= kmax - 2}


def parts(windows):
    """the variable-disjoint parts of a family: connected components of the graph 'windows share a variable'."""
    n = len(windows); comp = list(range(n))
    def find(i):
        while comp[i] != i: comp[i] = comp[comp[i]]; i = comp[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if set(windows[i].vars) & set(windows[j].vars): comp[find(i)] = find(j)
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def join_diagram(diagrams):
    """reduced persistence of a join of filtered complexes from the parts' reduced diagrams ({dim: [(b, d), ...]}, finite bars plus
    essential bars d = inf of the reduced homology). Bars [b1, d1) in dim i and [b2, d2) in dim j give [max b, min d) in dim i + j + 1
    (field coefficients). A part with no reduced bars at all (contractible at every level) makes the join empty."""
    out = None
    for D in diagrams:
        bars = [(k, b, d) for k, v in D.items() for b, d in v if d - b > 1e-12]
        if out is None: out = bars; continue
        out = [(i + j + 1, max(b1, b2), min(d1, d2)) for i, b1, d1 in out for j, b2, d2 in bars if min(d1, d2) - max(b1, b2) > 1e-12]
    res = {}
    for k, b, d in out or []: res.setdefault(k, []).append((b, d))
    return res


def reduced(P):
    """reduced persistence: drop the one essential H0 bar (the whole complex is non-empty from the start)."""
    out = {k: [(b, d) for b, d, _, _ in v] for k, v in P.items()}
    if 0 in out:
        ess = [x for x in out[0] if not np.isfinite(x[1])]
        if ess: out[0].remove(min(ess))
        if not out[0]: del out[0]
    return out
