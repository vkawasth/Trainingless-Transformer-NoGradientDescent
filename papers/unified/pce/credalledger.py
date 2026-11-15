"""A basis ledger for general polyhedral credal sets (Study 29; sec:ledger). It generalises pce.stratledger, which is specific to exact
margins, and modifies nothing. The credal set of one table is {z >= 0 : A z = b}, where A is a fixed integer matrix of full row rank,
the first nx coordinates of z are the cells and the rest are slack variables, and b varies from table to table. Objectives are n.z / d.z
(ratio, Dinkelbach certificates) or n.z (d = 0), with n and d zero on slacks.
  Library(A)                         bases found so far, with float inverses; certify(k, Bm, n, d, sense) tests every table at once
  solve_ratio(A, b, n, d, sense)     Charnes-Cooper linear programme; returns the optimal z (cells and slacks)
  completions(A, z, rng, tries)      bases containing the support of a vertex z: natural order first, then random orders
  build(L, Bm, n, d, sense, order)   assign every table a certifying basis, solving only uncertified tables; a table no completion
                                     certifies keeps its LP value and is counted as a fallback (assign = -1)
  exact_check(A, B, bF, nF, dF, sense)  exact rational check by Gaussian elimination (no stored inverse)
  tolerance_form(r, c)               A for row and column sums within +-e of (f, g), total 1; b_tol(f, g, e)
  pairwise_form(a, b, c)             A for the three pairwise margins of an a x b x c table (independent rows only); b_pair(joint)
"""
from __future__ import annotations
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
class Library:
    def __init__(self, A):
        self.A = np.asarray(A, float); self.m, self.M = self.A.shape; self.bases = []; self.inv = []; self.index = {}
    def add(self, B):
        B = tuple(sorted(B))
        if B in self.index: return self.index[B]
        self.bases.append(B); self.inv.append(np.linalg.inv(self.A[:, list(B)])); self.index[B] = len(self.bases) - 1; return len(self.bases) - 1
    def certify(self, k, Bm, n, d, sense, tol=1e-9):
        B = list(self.bases[k]); Nn = [j for j in range(self.M) if j not in self.index_set(k)]; inv = self.inv[k]
        X = inv @ Bm; feas = (X >= -tol).all(0); lin = not d.any()
        nB, dB = n[B], d[B]; num = nB @ X; den = np.ones(X.shape[1]) if lin else dB @ X
        lam = np.zeros(X.shape[1]) if lin else np.where(den > tol, num / np.where(den > tol, den, 1), np.nan)
        Mx = inv @ self.A[:, Nn]; rn = n[Nn] - nB @ Mx; rd = np.zeros(len(Nn)) if lin else d[Nn] - dB @ Mx
        red = rn[:, None] - rd[:, None] * lam[None, :]
        opt = (red <= tol).all(0) if sense > 0 else (red >= -tol).all(0)
        return feas & opt & ~np.isnan(lam), (num if lin else lam)
    def index_set(self, k):
        return set(self.bases[k])
def solve_ratio(A, b, n, d, sense):
    A = np.asarray(A, float); m, M = A.shape
    if not d.any():
        r = linprog(-sense * n, A_eq=A, b_eq=b, bounds=[(0, None)] * M, method="highs"); return r.x if r.status == 0 else None
    c = np.concatenate([-sense * n, [0.0]]); Aeq = np.vstack([np.hstack([A, -np.asarray(b, float)[:, None]]), np.concatenate([d, [0.0]])[None, :]])
    beq = np.concatenate([np.zeros(m), [1.0]]); r = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (M + 1), method="highs")
    if r.status != 0 or r.x[M] <= 1e-14: return None
    return r.x[:M] / r.x[M]
def completions(A, z, rng, tries=30, tol=1e-10):
    A = np.asarray(A, float); m, M = A.shape; S = [j for j in range(M) if z[j] > tol]
    if np.linalg.matrix_rank(A[:, S]) < len(S): S = S[:m]
    seen = set()
    for t in range(tries):
        rest = [j for j in range(M) if j not in S]
        if t > 0: rest = list(rng.permutation(rest))
        cur = list(S); rk = np.linalg.matrix_rank(A[:, cur]) if cur else 0
        for j in rest:
            if rk == m: break
            r2 = np.linalg.matrix_rank(A[:, cur + [j]])
            if r2 > rk: cur.append(j); rk = r2
        B = tuple(sorted(cur))
        if rk == m and B not in seen: seen.add(B); yield B
def build(L, Bm, n, d, sense, order, rng=None, tries=30):
    rng = rng or np.random.default_rng(0); T = Bm.shape[1]; assign = -np.ones(T, int); value = np.full(T, np.nan)
    done = np.zeros(T, bool); solves = 0; fallback = 0
    for k in range(len(L.bases)):
        ok, v = L.certify(k, Bm, n, d, sense); new = ok & ~done; assign[new] = k; value[new] = v[new]; done |= new
    for t in order:
        if done[t]: continue
        z = solve_ratio(L.A, Bm[:, t], n, d, sense); solves += 1
        if z is None: done[t] = True; continue
        for B in completions(L.A, z, rng, tries):
            k = L.add(B); ok, v = L.certify(k, Bm, n, d, sense); new = ok & ~done; assign[new] = k; value[new] = v[new]; done |= new
            if done[t]: break
        if not done[t]:
            value[t] = (n @ z) if not d.any() else (n @ z) / (d @ z); done[t] = True; fallback += 1
    return assign, value, solves, fallback
def certify_one(L, b, n, d, sense):
    for k in range(len(L.bases)):
        ok, v = L.certify(k, b[:, None], n, d, sense)
        if ok[0]: return k, float(v[0])
    return -1, None
def gsolve(M, rhs):
    n = len(M); A = [list(M[i]) + [rhs[i]] for i in range(n)]
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] != 0), None)
        if p is None: return None
        A[c], A[p] = A[p], A[c]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c] / A[c][c]; A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return [A[i][n] / A[i][i] for i in range(n)]
def exact_check(A, B, bF, nF, dF, sense):
    """exact feasibility and optimality of basis B at right-hand side bF; returns (ok, value)."""
    A = [[Fr(int(x)) for x in row] for row in np.asarray(A)]; m = len(A); M = len(A[0]); B = list(B)
    xB = gsolve([[A[i][j] for j in B] for i in range(m)], bF)
    if xB is None or any(x < 0 for x in xB): return False, None
    lin = not any(dF); num = sum(nF[B[i]] * xB[i] for i in range(m)); den = Fr(1) if lin else sum(dF[B[i]] * xB[i] for i in range(m))
    if den <= 0: return False, None
    lam = Fr(0) if lin else num / den; cp = [nF[j] - lam * dF[j] for j in range(M)]
    y = gsolve([[A[i][j] for i in range(m)] for j in B], [cp[j] for j in B])
    if y is None: return False, None
    for j in range(M):
        if j in B: continue
        rc = cp[j] - sum(y[i] * A[i][j] for i in range(m))
        if (sense > 0 and rc > 0) or (sense < 0 and rc < 0): return False, None
    return True, (num if lin else lam)
def tolerance_form(r, c):
    cells = [(i, j) for i in range(r) for j in range(c)]; N = r * c; M = N + 2 * r + 2 * c; rows = []; s = N
    for i in range(r):
        for sign in (+1, -1):
            row = [1 if cl[0] == i else 0 for cl in cells] + [0] * (M - N); row[s] = sign; s += 1; rows.append(row)
    for j in range(c):
        for sign in (+1, -1):
            row = [1 if cl[1] == j else 0 for cl in cells] + [0] * (M - N); row[s] = sign; s += 1; rows.append(row)
    rows.append([1] * N + [0] * (M - N)); return np.array(rows, int), cells
def b_tol(f, g, e):
    out = []
    for x in list(f) + list(g): out += [x + e, x - e]
    return out + [Fr(1)]
def pairwise_form(a, b, c):
    cells = [(i, j, k) for i in range(a) for j in range(b) for k in range(c)]; rows = []; keys = []
    for i in range(a):
        for j in range(b): rows.append([1 if (cl[0], cl[1]) == (i, j) else 0 for cl in cells]); keys.append(("AB", i, j))
    for j in range(b):
        for k in range(c): rows.append([1 if (cl[1], cl[2]) == (j, k) else 0 for cl in cells]); keys.append(("BC", j, k))
    for i in range(a):
        for k in range(c): rows.append([1 if (cl[0], cl[2]) == (i, k) else 0 for cl in cells]); keys.append(("AC", i, k))
    keep = []; rk = 0
    for q in range(len(rows)):
        r2 = np.linalg.matrix_rank(np.array([rows[p] for p in keep + [q]], float))
        if r2 > rk: keep.append(q); rk = r2
    return np.array([rows[q] for q in keep], int), cells, [keys[q] for q in keep]
def b_pair(joint, cells, keys):
    out = []
    for key in keys:
        if key[0] == "AB": out.append(sum(joint[k] for k, cl in enumerate(cells) if (cl[0], cl[1]) == key[1:]))
        elif key[0] == "BC": out.append(sum(joint[k] for k, cl in enumerate(cells) if (cl[1], cl[2]) == key[1:]))
        else: out.append(sum(joint[k] for k, cl in enumerate(cells) if (cl[0], cl[2]) == key[1:]))
    return out
