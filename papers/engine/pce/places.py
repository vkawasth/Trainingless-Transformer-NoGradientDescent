"""Separating obstructions by place: the trivial point (supports), the tropical scale (orders of magnitude), and the archimedean place
(probabilities).

Write v = -log t for the window tables (v = +inf on zero cells). Marginalising a global law q sums over the N_c atoms of a cell c, and
-log of a sum is within log N_c of the minimum of the -logs. So at the tropical scale a gluing is a solution x of the min-plus system
    min over atoms a in c of x(a) = v_w(c)     for every window w and cell c.
  principal(domains, windows)   the smallest x with x(a) >= v_w(c) for every cell containing a: x*(a) = max over windows of v_w(cell of a)
  tropical_defect(...)          delta = max over finite cells of (min over a in c of x*(a)) - v_w(c); delta = 0 iff the min-plus system
                                is solvable (Cuninghame-Green), delta = +inf iff some support cell has no admissible atom
  support_class(...)            at the trivial point: 'strong' (no global assignment inside all supports), 'logical' (some support cell
                                does not extend), or 'none'
  avn_z2(...)                   binary windows only: stack the affine Z2 equations satisfied by each window's support; inconsistent =
                                an All-versus-Nothing obstruction, which Abramsky et al. show is witnessed by Cech cohomology
  layer(domains, windows)       'support' (delta = inf), 'tropical' (log N < delta < inf), 'archimedean' (lam* < 1, delta <= log N),
                                or 'coherent'
"""
from __future__ import annotations
import itertools
import numpy as np
from .coherence import Window, _constraints
from .integrate import gluable_fraction


def _cells(domains, windows):
    A, t, atoms, index = _constraints(domains, windows)
    return A, t, atoms, index


def principal(domains, windows):
    A, t, atoms, index = _cells(domains, windows)
    with np.errstate(divide="ignore"):
        v = np.where(t > 0, -np.log(np.where(t > 0, t, 1.0)), np.inf)
    x = np.array([max(v[r] for r in np.nonzero(A[:, k])[0]) for k in range(A.shape[1])])
    return x, v, A, atoms, index


def tropical_defect(domains, windows):
    x, v, A, atoms, index = principal(domains, windows)
    worst = 0.0
    for r in range(A.shape[0]):
        if not np.isfinite(v[r]): continue
        m = min(x[k] for k in np.nonzero(A[r])[0])
        worst = max(worst, m - v[r])
    N = max(int(A[r].sum()) for r in range(A.shape[0]))
    return float(worst), N


def support_class(domains, windows):
    A, t, atoms, index = _cells(domains, windows)
    ok_atom = np.array([all(t[r] > 0 for r in np.nonzero(A[:, k])[0]) for k in range(A.shape[1])])
    if not ok_atom.any(): return "strong"
    for r in range(A.shape[0]):
        if t[r] > 0 and not any(ok_atom[k] for k in np.nonzero(A[r])[0]): return "logical"
    return "none"


def _z2_rank(M):
    M = M.copy() % 2; r = 0
    for c in range(M.shape[1]):
        piv = next((i for i in range(r, M.shape[0]) if M[i, c]), None)
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]: M[i] ^= M[r]
        r += 1
    return r


def avn_z2(domains, windows):
    """True if the affine Z2 equations satisfied by every window's support are jointly inconsistent (binary variables only)."""
    names = sorted(domains); rows = []
    for w in windows:
        vs = list(w.vars); supp = [c for c in itertools.product((0, 1), repeat=len(vs)) if w.table[c] > 0]
        for k in range(1, len(vs) + 1):
            for S in itertools.combinations(range(len(vs)), k):
                vals = {sum(c[i] for i in S) % 2 for c in supp}
                if len(vals) == 1:
                    row = np.zeros(len(names) + 1, dtype=np.uint8)
                    for i in S: row[names.index(vs[i])] = 1
                    row[-1] = vals.pop(); rows.append(row)
    if not rows: return False
    M = np.array(rows)
    return _z2_rank(M[:, :-1]) < _z2_rank(M)


def layer(domains, windows):
    d, N = tropical_defect(domains, windows)
    if not np.isfinite(d): return "support", d, N, None
    lam = gluable_fraction(domains, windows).lam
    if d > np.log(N) + 1e-12: return "tropical", d, N, lam
    return ("archimedean" if lam < 1 - 1e-7 else "coherent"), d, N, lam
