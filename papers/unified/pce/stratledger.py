"""The stratified ledger (section sec:ledger): exact parametric evaluation of credal bands over many tables, with basis libraries,
certificates, boundary data and certified top-k search. New module; it modifies nothing.
Credal set of a table: the coupling polytope Cpl(f, g) of r x c tables with margins (f, g) (exact margins). The margin system is
A x = b with b = (f, g without its last entry); A (r + c - 1 rows) has full row rank and is totally unimodular; a basis is a set of
r + c - 1 cells forming a spanning tree of K_{r,c}.
  Problem(r, c)                       constraint matrix, cells, basis utilities
  Library                             bases found so far, with exact integer inverses; certify(B, tables, n, d, sense) tests, for every table
                                      at once, primal feasibility of B and optimality of its vertex for max (or min) n.x / d.x (Dinkelbach:
                                      reduced costs of n - lambda d with lambda the vertex ratio); d = 0 gives a linear objective n.x
  solve_ratio(P, b, n, d, sense)      Charnes-Cooper linear programme; returns the optimal vertex and its support
  exact_value(P, B, b, n, d, sense)   exact rational value and exact certificate check (Fractions)
"""
from __future__ import annotations
import itertools
from fractions import Fraction as Fr
import numpy as np
import sympy as sp
from scipy.optimize import linprog
class Problem:
    def __init__(self, r, c):
        self.r, self.c = r, c; self.cells = [(i, j) for i in range(r) for j in range(c)]; self.N = r * c; self.m = r + c - 1
        A = []
        for i in range(r): A.append([1 if cl[0] == i else 0 for cl in self.cells])
        for j in range(c - 1): A.append([1 if cl[1] == j else 0 for cl in self.cells])
        self.A = np.array(A, int)
    def b(self, f, g): return np.array([float(x) for x in list(f) + list(g)[:-1]])
    def bF(self, f, g): return [Fr(x) for x in list(f) + list(g)[:-1]]
    def is_basis(self, B): return len(B) == self.m and np.linalg.matrix_rank(self.A[:, list(B)]) == self.m
class Library:
    def __init__(self, P):
        self.P = P; self.bases = []; self.inv = []; self.invF = []; self.detabs = []
    def add(self, B):
        B = tuple(sorted(B))
        if B in self.bases: return self.bases.index(B)
        AB = sp.Matrix(self.P.A[:, list(B)].tolist()); det = AB.det(); inv = AB.inv()
        self.bases.append(B); self.inv.append(np.array(inv.tolist(), float)); self.invF.append([[Fr(int(x.p), int(x.q)) for x in row] for row in inv.tolist()])
        self.detabs.append(abs(int(det))); return len(self.bases) - 1
    def certify(self, k, Bm, n, d, sense, tol=1e-11):
        """Bm: (m, T) right-hand sides. Returns (mask, values) for basis k over all T tables."""
        B = list(self.bases[k]); Nn = [j for j in range(self.P.N) if j not in B]; inv = self.inv[k]; A = self.P.A
        X = inv @ Bm                                                      # basic values, (m, T)
        feas = (X >= -tol).all(0)
        nB, dB = n[B], d[B]; num = nB @ X; den = dB @ X if d.any() else np.ones(X.shape[1])
        lam = np.where(den > tol, num / np.where(den > tol, den, 1), np.nan) if d.any() else np.zeros(X.shape[1])
        M = inv @ A[:, Nn]                                                # (m, |N|)
        rn = n[Nn] - nB @ M; rd = (d[Nn] - dB @ M) if d.any() else np.zeros(len(Nn))
        red = rn[:, None] - rd[:, None] * lam[None, :]                    # reduced costs of n - lam d, (|N|, T)
        opt = (red <= tol).all(0) if sense > 0 else (red >= -tol).all(0)
        ok = feas & opt & ~np.isnan(lam)
        return ok, (lam if d.any() else num)
def solve_ratio(P, b, n, d, sense):
    """max (sense=+1) or min (sense=-1) of n.x / d.x over Cpl (d = 0: linear n.x). Returns x."""
    N = P.N
    if not d.any():
        r = linprog(-sense * n, A_eq=P.A, b_eq=b, bounds=[(0, None)] * N, method="highs"); return r.x
    c = np.concatenate([-sense * n, [0.0]]); Aeq = np.vstack([np.hstack([P.A, -b[:, None]]), np.concatenate([d, [0.0]])[None, :]])
    beq = np.concatenate([np.zeros(P.m), [1.0]]); r = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (N + 1), method="highs")
    if r.status != 0: return None
    return r.x[:N] / r.x[N]
def bases_from_vertex(P, x, tol=1e-10):
    S = [j for j in range(P.N) if x[j] > tol]
    if len(S) == P.m and P.is_basis(S): yield tuple(S); return
    rest = [j for j in range(P.N) if j not in S]
    for add in itertools.combinations(rest, P.m - len(S)):
        B = tuple(sorted(S + list(add)))
        if P.is_basis(B): yield B
def build(P, L, Bm, n, d, sense, order, max_solves=10 ** 6):
    """assign every table a certifying basis; solve a linear programme only for tables no library basis certifies. Returns
    (assign, value, solves, growth) with growth[i] = library size after the i-th table in 'order'."""
    T = Bm.shape[1]; assign = -np.ones(T, int); value = np.full(T, np.nan); solves = 0; growth = []
    for k in range(len(L.bases)):
        ok, v = L.certify(k, Bm, n, d, sense); new = ok & (assign < 0); assign[new] = k; value[new] = v[new]
    for t in order:
        if assign[t] < 0 and solves < max_solves:
            x = solve_ratio(P, Bm[:, t], n, d, sense); solves += 1
            if x is not None:
                for B in bases_from_vertex(P, x):
                    k = L.add(B); ok, v = L.certify(k, Bm, n, d, sense); new = ok & (assign < 0); assign[new] = k; value[new] = v[new]
                    if assign[t] >= 0: break
        growth.append(solves)
    return assign, value, solves, growth
def exact_check(P, L, k, bF, nF, dF, sense):
    """exact rational re-verification of basis k at one table: feasibility, ratio, reduced costs. Returns (ok, value)."""
    B = list(L.bases[k]); Nn = [j for j in range(P.N) if j not in B]; inv = L.invF[k]; m = P.m
    xB = [sum(inv[i][j] * bF[j] for j in range(m)) for i in range(m)]
    if any(x < 0 for x in xB): return False, None
    num = sum(nF[B[i]] * xB[i] for i in range(m)); den = sum(dF[B[i]] * xB[i] for i in range(m)) if any(dF) else Fr(1)
    if den <= 0: return False, None
    lam = num / den if any(dF) else Fr(0)
    for j in Nn:
        col = [int(P.A[i][j]) for i in range(m)]; Mj = [sum(inv[i][q] * col[q] for q in range(m)) for i in range(m)]
        rc = (nF[j] - lam * dF[j]) - sum((nF[B[i]] - lam * dF[B[i]]) * Mj[i] for i in range(m))
        if (sense > 0 and rc > 0) or (sense < 0 and rc < 0): return False, None
    return True, (lam if any(dF) else num)
