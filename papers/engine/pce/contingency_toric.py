"""Part II of the companion note: contingency tables read back from the toric side. New module; it imports pce/kahlerbridge.py and
pce/datawalls.py and modifies neither.

Two configurations belong to one log-linear model and must be kept apart.
  cells   the columns of the design matrix A (one per cell of the table). Their relations are the moves between tables with the same margins;
          for the r x c independence model there are (r - 1)(c - 1) of them (Diaconis-Sturmfels). This is the Tier A support S.
  facets  the inward normals v_k of the moment polytope Q, here a product of simplices. W is built from these. For independence there is
          exactly one circuit among them.

  cell_config(r, c)                  the r c design columns (row indicator, column indicator)
  cell_relation_dim(r, c)            r c - rank A
  basic_move(r, c, i, j, k, l)       the table with +1 at (i,j), (k,l) and -1 at (i,l), (k,j)
  facet_config(r, c, L1, L2)         normals and exponents of Delta_(r-1)(L1) x Delta_(c-1)(L2): coordinates are the first r - 1 row margins and the
                                     first c - 1 column margins; l_k(b) is then exactly the k-th margin
  margin_point(rows, cols)           those coordinates for given margins
  ell(v, lam, b)                     all l_k(b)
  mle(table)                         the independence MLE, rows x cols / n^2 (Birch: its margins are the observed ones)
  fisher_blocks(pr, pc, n1, n2)      the Fisher information of the independence family in the design's natural coordinates (row and column surveys
                                     of sizes n1, n2): block diagonal, n1 Cov(row) + n2 Cov(col)
  fr_simplex(p, q)                   Fisher-Rao distance on the simplex, 2 arccos sum sqrt(p q)
  fr_independence(p, q, n1, n2)      sqrt(n1 d_row^2 + n2 d_col^2) for product laws p = (pr, pc), q = (qr, qc)
"""
from __future__ import annotations
import numpy as np


def cell_config(r, c):
    out = []
    for i in range(r):
        for j in range(c):
            x = np.zeros(r + c); x[i] = 1; x[r + j] = 1; out.append(x)
    return np.array(out)


def cell_relation_dim(r, c): return r * c - int(np.linalg.matrix_rank(cell_config(r, c)))


def basic_move(r, c, i, j, k, l):
    M = np.zeros((r, c)); M[i, j] += 1; M[k, l] += 1; M[i, l] -= 1; M[k, j] -= 1; return M


def facet_config(r, c, L1, L2):
    n = r + c - 2; v = []; lam = []
    for i in range(r - 1):
        x = np.zeros(n); x[i] = 1; v.append(tuple(x)); lam.append(0.0)
    x = np.zeros(n); x[:r - 1] = -1; v.append(tuple(x)); lam.append(-float(L1))
    for j in range(c - 1):
        x = np.zeros(n); x[r - 1 + j] = 1; v.append(tuple(x)); lam.append(0.0)
    x = np.zeros(n); x[r - 1:] = -1; v.append(tuple(x)); lam.append(-float(L2))
    return v, lam


def margin_point(rows, cols): return np.r_[np.array(rows, float)[:-1], np.array(cols, float)[:-1]]


def ell(v, lam, b): return np.array(v, float) @ np.array(b, float) - np.array(lam, float)


def mle(table):
    t = np.array(table, float); n = t.sum(); return np.outer(t.sum(1), t.sum(0)) / n ** 2


def fisher_blocks(pr, pc, n1, n2):
    def cov(p):
        p = np.array(p, float); return np.diag(p) - np.outer(p, p)
    r, c = len(pr), len(pc); F = np.zeros((r + c, r + c)); F[:r, :r] = n1 * cov(pr); F[r:, r:] = n2 * cov(pc); return F


def fr_simplex(p, q): return 2 * np.arccos(np.clip(np.sum(np.sqrt(np.array(p, float) * np.array(q, float))), -1, 1))


def fr_independence(p, q, n1, n2):
    (pr, pc), (qr, qc) = p, q; return np.sqrt(n1 * fr_simplex(pr, qr) ** 2 + n2 * fr_simplex(pc, qc) ** 2)
