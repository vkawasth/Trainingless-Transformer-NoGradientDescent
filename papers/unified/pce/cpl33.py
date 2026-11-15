"""Four-dimensional coupling fibres: 3 x 3 tables (section sec:threebythree). New module; it modifies nothing.
Exact rational leximin with certificates: every linear programme is solved numerically, its primal and dual solutions are rounded to
rationals, and optimality is then verified in exact arithmetic (primal feasibility, dual feasibility, equal objective values). A level
is accepted only when its certificate verifies.
Fibre coordinates: u = (u00, u01, u10, u11), x_ij = x0_ij + <n_ij, u> with n_ij = e_ij (i, j < 2), n_i2 = -(e_i0 + e_i1),
n_2j = -(e_0j + e_1j), n_22 = sum of all four (a unimodular basis of the move lattice).
  certify_max(c, A_eq, b_eq, A_ub, b_ub, free)   exact optimum of max c.z (Fractions) or raises
  exact_facets(f, g)                             the cells whose constraint is a facet (certified); a cell with f_i + g_j > 1 is
                                                 bounded below by f_i + g_j - 1 > 0 (Frechet) and is not a facet
  exact_leximin(f, g, S)                         levels [(h_j, S_j)] and the exact point, every step certified
  normals()                                      the nine cell normals in the fibre lattice
  r_pattern(levels)                              dimensions removed by each level (exact ranks)
  vertex_sets(f, g, facets, support=None)        the 4-subsets of facets tight at vertices (support numbers: margins, or all 1 for -K)
  smooth(f, g, facets), fano(f, g, facets)       Delzant (unimodular vertex cones) and Fano (same vertex structure with support numbers 1)
"""
from __future__ import annotations
import itertools
from fractions import Fraction as Fr
import numpy as np
import sympy as sp
from scipy.optimize import linprog
CELLS = [(i, j) for i in range(3) for j in range(3)]
def normals():
    e = {(0, 0): (1, 0, 0, 0), (0, 1): (0, 1, 0, 0), (1, 0): (0, 0, 1, 0), (1, 1): (0, 0, 0, 1)}; out = {}
    for (i, j) in CELLS:
        if i < 2 and j < 2: out[(i, j)] = e[(i, j)]
        elif i < 2 and j == 2: out[(i, j)] = tuple(-a - b for a, b in zip(e[(i, 0)], e[(i, 1)]))
        elif i == 2 and j < 2: out[(i, j)] = tuple(-a - b for a, b in zip(e[(0, j)], e[(1, j)]))
        else: out[(i, j)] = (1, 1, 1, 1)
    return out
def margins_A():
    A = []
    for i in range(3): A.append([1 if c[0] == i else 0 for c in CELLS])
    for j in range(3): A.append([1 if c[1] == j else 0 for c in CELLS])
    return A
def _rat(v, D):
    return [Fr(float(x)).limit_denominator(D) for x in v]
def certify_max(c, A_eq, b_eq, A_ub, b_ub, free, Ds=(10 ** 4, 10 ** 6, 10 ** 9)):
    """max c.z s.t. A_eq z = b_eq, A_ub z <= b_ub, z_j >= 0 for j not in free. All data Fractions. Returns (value, z) exactly."""
    n = len(c); A_ub = A_ub or []; b_ub = b_ub or []; A_eq = A_eq or []; b_eq = b_eq or []
    fl = lambda M: np.array([[float(x) for x in r] for r in M]) if M else None
    bounds = [(None, None) if j in free else (0, None) for j in range(n)]
    r = linprog(-np.array([float(x) for x in c]), A_eq=fl(A_eq), b_eq=[float(x) for x in b_eq] if A_eq else None, A_ub=fl(A_ub),
                b_ub=[float(x) for x in b_ub] if A_ub else None, bounds=bounds, method="highs")
    if r.status != 0: raise ValueError(f"primal status {r.status}")
    me, mu = len(A_eq or []), len(A_ub or [])
    # dual: min b_eq.y + b_ub.w, w >= 0, (A_eq^T y + A_ub^T w)_j >= c_j (j not free), = c_j (j free)
    AT = [[float(A_eq[i][j]) for i in range(me)] + [float(A_ub[i][j]) for i in range(mu)] for j in range(n)]
    Aub_d = [[-x for x in AT[j]] for j in range(n) if j not in free]; bub_d = [-float(c[j]) for j in range(n) if j not in free]
    Aeq_d = [AT[j] for j in range(n) if j in free]; beq_d = [float(c[j]) for j in range(n) if j in free]
    rd = linprog(np.array([float(x) for x in b_eq] + [float(x) for x in b_ub]), A_ub=np.array(Aub_d) if Aub_d else None, b_ub=bub_d or None,
                 A_eq=np.array(Aeq_d) if Aeq_d else None, b_eq=beq_d or None, bounds=[(None, None)] * me + [(0, None)] * mu, method="highs")
    if rd.status != 0: raise ValueError(f"dual status {rd.status}")
    for D in Ds:
        z = _rat(r.x, D); yw = _rat(rd.x, D); y, w = yw[:me], yw[me:]
        ok = all(sum(A_eq[i][j] * z[j] for j in range(n)) == b_eq[i] for i in range(me))
        ok &= all(sum(A_ub[i][j] * z[j] for j in range(n)) <= b_ub[i] for i in range(mu)) and all(z[j] >= 0 for j in range(n) if j not in free)
        ok &= all(x >= 0 for x in w)
        for j in range(n):
            lhs = sum(A_eq[i][j] * y[i] for i in range(me)) + sum(A_ub[i][j] * w[i] for i in range(mu))
            ok &= (lhs == c[j]) if j in free else (lhs >= c[j])
        pv = sum(c[j] * z[j] for j in range(n)); dv = sum(b_eq[i] * y[i] for i in range(me)) + sum(b_ub[i] * w[i] for i in range(mu))
        if ok and pv == dv: return pv, z
    raise ValueError("certificate did not verify")
def exact_facets(f, g):
    """cell c is a facet iff max eps s.t. x_c = 0 and x_k >= eps (k != c) is positive (certified)."""
    A = margins_A(); b = list(f) + list(g); out = []
    for ci, c in enumerate(CELLS):
        if f[c[0]] + g[c[1]] > 1: continue          # x_c >= f_i + g_j - 1 > 0 on the whole polytope (exact Frechet bound): never 0
        n = 10; cvec = [Fr(0)] * 9 + [Fr(1)]
        A_eq = [r + [0] for r in A] + [[1 if k == ci else 0 for k in range(9)] + [0]]; b_eq = b + [Fr(0)]
        A_ub = [[-1 if k == kk else 0 for k in range(9)] + [1] for kk in range(9) if kk != ci] + [[0] * 9 + [1]]; b_ub = [Fr(0)] * 8 + [Fr(1)]
        v, _ = certify_max(cvec, A_eq, b_eq, A_ub, b_ub, free={9})
        if v > 0: out.append(c)
    return out
def exact_leximin(f, g, S):
    A = margins_A(); b = list(f) + list(g); Si = [CELLS.index(c) for c in S]; fixed = {}; free = set(Si); levels = []
    while free:
        cvec = [Fr(0)] * 9 + [Fr(1)]
        A_eq = [r + [0] for r in A] + [[1 if k == i else 0 for k in range(9)] + [0] for i in fixed]; b_eq = b + [fixed[i] for i in fixed]
        A_ub = [[-1 if k == i else 0 for k in range(9)] + [1] for i in sorted(free)]; b_ub = [Fr(0)] * len(free)
        t, _ = certify_max(cvec, A_eq, b_eq, A_ub, b_ub, free={9})
        newly = []
        for i in sorted(free):
            c2 = [Fr(1) if k == i else Fr(0) for k in range(9)]
            A_eq2 = [r for r in A] + [[1 if k == j else 0 for k in range(9)] for j in fixed]; b_eq2 = b + [fixed[j] for j in fixed]
            A_ub2 = [[-1 if k == j else 0 for k in range(9)] for j in sorted(free)]; b_ub2 = [-t] * len(free)
            v, _ = certify_max(c2, A_eq2, b_eq2, A_ub2, b_ub2, free=set())
            if v == t: newly.append(i)
        if not newly: raise ValueError("no blocked cell: recursion stalled")
        for i in newly: fixed[i] = t; free.discard(i)
        levels.append((t, [CELLS[i] for i in newly]))
    A_eq = [r for r in A] + [[1 if k == j else 0 for k in range(9)] for j in fixed]; b_eq = b + [fixed[j] for j in fixed]
    v, x = certify_max([Fr(0)] * 9, A_eq, b_eq, None, None, free=set())
    # uniqueness of the point: the fixed facet cells pin u (rank 4 of their normals) - checked by the caller via r_pattern
    return levels, x
def r_pattern(levels):
    N = normals(); out = []; prev = []
    for h, Sj in levels:
        r0 = sp.Matrix(prev).rank() if prev else 0; prev = prev + [list(N[c]) for c in Sj]; out.append(sp.Matrix(prev).rank() - r0)
    return out
def base_point(f, g):
    """an exact rational point x0 of Cpl(f, g) (north-west corner rule)."""
    f = list(f); g = list(g); x = {c: Fr(0) for c in CELLS}; i = j = 0
    while i < 3 and j < 3:
        t = min(f[i], g[j]); x[(i, j)] = t; f[i] -= t; g[j] -= t
        if f[i] == 0: i += 1
        else: j += 1
    return x
def vertex_sets(f, g, facets, support=None):
    """4-subsets of facets whose normals are independent and whose solution satisfies all facet inequalities. support: dict of
    support numbers lam_k (polytope <n_k, u> >= -lam_k); default the margins' polytope (lam_k = x0_k)."""
    N = normals(); x0 = base_point(f, g); lam = support or {c: x0[c] for c in facets}; out = set()
    for Sq in itertools.combinations(facets, 4):
        M = sp.Matrix([list(N[c]) for c in Sq])
        if M.det() == 0: continue
        u = M.LUsolve(sp.Matrix([-lam[c] for c in Sq]))
        if all(sum(N[c][k] * u[k] for k in range(4)) >= -lam[c] for c in facets): out.add(frozenset(Sq))
    return out
def smooth(f, g, facets):
    N = normals(); V = vertex_sets(f, g, facets)
    return len(V) > 0 and all(abs(sp.Matrix([list(N[c]) for c in Sq]).det()) == 1 for Sq in V), V
def fano(f, g, facets, V=None):
    V = V or vertex_sets(f, g, facets); VK = vertex_sets(f, g, facets, support={c: Fr(1) for c in facets}); return V == VK
SYMS = []
for pr in itertools.permutations(range(3)):
    for pc in itertools.permutations(range(3)):
        for tr in (False, True): SYMS.append((pr, pc, tr))
def canon(facets):
    best = None
    for pr, pc, tr in SYMS:
        img = tuple(sorted(((pc[j], pr[i]) if tr else (pr[i], pc[j])) for i, j in facets))
        if best is None or img < best: best = img
    return best

def canon_fan(V):
    """canonical form of a normal fan (its maximal cones as 4-sets of facet cells) under row, column and transpose symmetry."""
    best = None
    for pr, pc, tr in SYMS:
        m = lambda c: (pc[c[1]], pr[c[0]]) if tr else (pr[c[0]], pc[c[1]])
        img = tuple(sorted(tuple(sorted(m(c) for c in Sq)) for Sq in V))
        if best is None or img < best: best = img
    return best
