"""Linear-programming helpers for coupling polytopes (shared by check_scales.py and check_leximinext.py). New module; modifies nothing.
  margins_A(n, m)            the margin constraint matrix of n x m tables
  facets(f, g)               the cells whose constraint x_c >= 0 is a facet of Cpl(f, g)
  leximin_levels(f, g, S)    the leximin recursion over the cells S: the point and the list of (h_j, S_j)
"""
import numpy as np
from scipy.optimize import linprog
def margins_A(n, m):
    A = []
    for i in range(n): r = np.zeros(n * m); r[i * m:(i + 1) * m] = 1; A.append(r)
    for j in range(m): r = np.zeros(n * m); r[j::m] = 1; A.append(r)
    return np.array(A)
def facets(f, g):
    n, m = len(f), len(g); A = margins_A(n, m); b = np.r_[f, g]; N = n * m; out = []
    for c in range(N):
        cc = np.zeros(N + 1); cc[-1] = -1; Aub = []; bub = []
        for k in range(N):
            if k == c: continue
            r = np.zeros(N + 1); r[k] = -1; r[-1] = 1; Aub.append(r); bub.append(0)
        r = np.zeros(N + 1); r[c] = 1
        res = linprog(cc, A_ub=Aub, b_ub=bub, A_eq=np.r_[np.c_[A, np.zeros(len(A))], [r]], b_eq=np.r_[b, 0], bounds=[(0, None)] * N + [(None, 1)], method="highs")
        if res.status == 0 and -res.fun > 1e-9: out.append(c)
    return out
def leximin_levels(f, g, S):
    """the facet-leximin recursion: returns the point x and the list of (h_j, S_j)."""
    n, m = len(f), len(g); A = margins_A(n, m); b = np.r_[f, g]; N = n * m; fixed = {}; free = set(S); levels = []
    while free:
        Aeq = np.c_[A, np.zeros(len(A))]; beq = b.copy()
        for i, v in fixed.items(): r = np.zeros(N + 1); r[i] = 1; Aeq = np.r_[Aeq, [r]]; beq = np.r_[beq, v]
        Aub = [];
        for i in free: r = np.zeros(N + 1); r[i] = -1; r[-1] = 1; Aub.append(r)
        c = np.zeros(N + 1); c[-1] = -1
        t = linprog(c, A_ub=Aub, b_ub=np.zeros(len(Aub)), A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * N + [(None, None)], method="highs").x[-1]
        newly = []
        for i in free:
            c2 = np.zeros(N + 1); c2[i] = -1; Aub2 = []
            for k in free: r = np.zeros(N + 1); r[k] = -1; Aub2.append(r)
            res2 = linprog(c2, A_ub=Aub2, b_ub=-t * np.ones(len(Aub2)) + 1e-12, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * N + [(None, None)], method="highs")
            if res2.status != 0 or -res2.fun <= t + 1e-9: newly.append(i)
        for i in newly: fixed[i] = t; free.discard(i)
        levels.append((t, newly))
    Aeq = A.copy(); beq = b.copy()
    for i, v in fixed.items(): r = np.zeros(N); r[i] = 1; Aeq = np.r_[Aeq, [r]]; beq = np.r_[beq, v]
    x = linprog(np.zeros(N), A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * N, method="highs").x
    return x, levels
