"""Fast geometry of 3 x 3 coupling polytopes (PREANALYSIS_PHASE4.md, WP1/WP3): vertices by basic solutions, wall signs, tear margin,
break count and entropy-brane steepness (largest eigenvalue of the Fisher form on the fibre at the product coupling)."""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from monitor import wall_values
CELLS = [(i, j) for i in range(3) for j in range(3)]
_A = np.array([[1.0 if c[0] == i else 0.0 for c in CELLS] for i in range(3)] + [[1.0 if c[1] == j else 0.0 for c in CELLS] for j in range(3)])
_SOLVERS = []
for Z in itertools.combinations(range(9), 4):
    A = np.vstack([_A] + [np.eye(9)[z][None] for z in Z])
    if np.linalg.matrix_rank(A) == 9: _SOLVERS.append(np.linalg.pinv(A)[:, :6])      # x = pinv(A)[:, :6] @ (f, g) when consistent
_SOLVERS = np.array(_SOLVERS)                                                          # (bases, 9, 6)
# orthonormal basis of the circulations (row and column sums zero)
_U, _S, _Vt = np.linalg.svd(_A); _CIRC = _Vt[np.sum(_S > 1e-10):].T                  # (9, 4)

def n_vertices(f, g, tol=1e-9):
    b = np.concatenate([f, g]); X = _SOLVERS @ b                                        # (bases, 9)
    ok = (X.min(1) > -tol) & (np.abs(X @ _A.T - b).max(1) < 1e-8)
    return len({tuple(np.round(x, 7)) for x in X[ok]})
def steepness(f, g):
    if min(f.min(), g.min()) <= 0: return np.inf
    D = 1.0 / np.outer(f, g).ravel(); Q = _CIRC.T @ (D[:, None] * _CIRC); return float(np.linalg.eigvalsh(Q).max())
def signs(f, g, tol=1e-9):
    v = wall_values(f, g); return np.where(v > tol, 1, np.where(v < -tol, -1, 0))
def tear_margin(f, g): return float(np.abs(wall_values(f, g)).min())
def breaks(f, g): return int((np.asarray(f) <= 0).sum() + (np.asarray(g) <= 0).sum())

if __name__ == '__main__':      # self-check against the slow enumeration of bridges/fukaya/nef33.py
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'bridges', 'fukaya'))
    from nef33 import vertices as slow
    rng = np.random.default_rng(1); ok = True
    for _ in range(60):
        f, g = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3)); ok &= n_vertices(f, g) == len(slow(f, g))
    for f, g in (([.5, .5, 0], [.2, .3, .5]), ([1, 0, 0], [.2, .3, .5]), ([1/3] * 3, [1/3] * 3)):
        ok &= n_vertices(np.array(f), np.array(g)) == len(slow(np.array(f), np.array(g)))
    print('n_vertices agrees with nef33.vertices:', ok)
