"""exact transportation-polytope helpers (vertices by spanning-tree enumeration)."""
import itertools
from fractions import Fraction as F
def rank_solve(A, b):
    A = [row[:] + [bb] for row, bb in zip(A, b)]; n = len(A[0]) - 1; piv = []; r = 0
    for c in range(n):
        p = next((i for i in range(r, len(A)) if A[i][c] != 0), None)
        if p is None: continue
        A[r], A[p] = A[p], A[r]; A[r] = [x / A[r][c] for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c] != 0: A[i] = [x - A[i][c] * y for x, y in zip(A[i], A[r])]
        piv.append(c); r += 1
    if any(all(x == 0 for x in row[:-1]) and row[-1] != 0 for row in A): return r, None, n - r
    sol = [F(0)] * n
    for i, c in enumerate(piv): sol[c] = A[i][-1]
    return r, sol, n - r
def polytope(f, g):
    n, m = len(f), len(g); cells = [(i, j) for i in range(n) for j in range(m)]; V = {}
    for S in itertools.combinations(cells, n + m - 1):
        A = [[F(int(c[0] == i)) for c in S] for i in range(n)] + [[F(int(c[1] == j)) for c in S] for j in range(m)]
        r, sol, nul = rank_solve(A, list(f) + list(g))
        if sol is None or nul > 0 or any(x < 0 for x in sol): continue
        V[tuple(sol[S.index(c)] if c in S else F(0) for c in cells)] = None
    return cells, list(V)
def facet_cells(V, cells):
    Zs = {c: frozenset(i for i, v in enumerate(V) if v[cells.index(c)] == 0) for c in cells}
    return [c for c in cells if Zs[c] and not any(d != c and Zs[d] > Zs[c] for d in cells)]
def fan(f, g):
    """rays = facet cells; maximal cones = sets of facet cells vanishing at each vertex"""
    cells, V = polytope(f, g); fac = facet_cells(V, cells)
    cones = frozenset(frozenset(c for c in fac if v[cells.index(c)] == 0) for v in V)
    return frozenset(fac), cones, len(V)
def walls(n, m):
    out = []
    for r in range(1, n):
        for I in itertools.combinations(range(n), r):
            for s in range(1, m):
                for J in itertools.combinations(range(m), s):
                    if 0 in I: out.append((I, J))          # (I,J) ~ (I^c, J^c): keep the representative with 0 in I
    return out
def signs(f, g, W):
    return tuple((sum(f[i] for i in I) > sum(g[j] for j in J)) - (sum(f[i] for i in I) < sum(g[j] for j in J)) for I, J in W)
