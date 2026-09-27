"""Monodromy of transport operators: a representation of pi_1 of the context graph, and
character-twisted cohomology.

Transport along an edge a -> b is the conditional-expectation operator T_ab[x, y] = P(b = y | a = x)
(a permutation matrix for a noiseless operation, a stochastic matrix for a lossy one; the reverse
edge uses the Bayesian inverse). Composing around the fundamental cycles of a spanning tree gives
    rho : pi_1(Gamma, root) = F_{beta_1}  ->  End(C^n),      loop_i |-> M_i,
a representation of the free group on the basis loops (unitary iff every edge is a bijection).

What the representation adds to the class [g] in H^1(Gamma, A):
  * abelian A (shifts on Z/n): nothing up to conjugacy -- M_i = T_{h_i}, so rho is determined by
    the loop values h_i, i.e. by [g]; in the Fourier basis it is diagonal, diag(w^{k h_i});
    reversing a loop inverts the phases.
  * non-abelian A (e.g. Aff(Z/p)): more than any abelian class -- traces of words and commutators
    separate systems that share their abelianised class; a multiplier a = -1 acts on characters
    as the reflection chi_k -> chi_{-k}.
  * lossy edges: |eigenvalues| < 1; the phase of the character eigenvalue is still the holonomy.

Character-twisted cohomology: for a Z/n-valued operation system and a character k, the local
system with edge phase w^{k g_e} has coboundary (delta f)_e = f(b) - w^{k g_e} f(a). Then
    h0_k = 1 if k [g] = 0 else 0 (connected graph),     h1_k = E - V + h0_k.
So the twisted H^0 over C is exactly the global-section test: torsion made visible over C by a
character, which requires complex (or R^2-rotation) coefficients unless n = 2.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .holonomy import Graph
from .channels import loop_walk, edge_operator, shift


# ---------------------------------------------------------------- representation
def loop_matrices(graph: Graph, K: Sequence[np.ndarray]) -> List[np.ndarray]:
    """rho(loop_i) for the fundamental cycles (cotree edge a->b, tree path back to a).
    K[e] is the transport for edge e in its stored orientation."""
    out = []
    for i in range(graph.beta1):
        w = loop_walk(graph, i)
        M = np.eye(K[0].shape[0])
        for u, v in zip(w[:-1], w[1:]):
            M = M @ edge_operator(graph, K, u, v)
        out.append(M)
    return out


def word(Ms: Sequence[np.ndarray], w: Sequence[int]) -> np.ndarray:
    """rho of a word in the free group: +i for loop i, -i-1 style avoided: use (i, +1/-1) pairs."""
    M = np.eye(Ms[0].shape[0])
    for i, s in w:
        M = M @ (Ms[i] if s > 0 else np.linalg.inv(Ms[i]))
    return M


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B @ np.linalg.inv(A) @ np.linalg.inv(B)


def is_unitary(M: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.allclose(M.conj().T @ M, np.eye(M.shape[0]), atol=tol))


def fourier(n: int) -> np.ndarray:
    return np.exp(2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n) / np.sqrt(n)


def in_character_basis(M: np.ndarray) -> np.ndarray:
    """F^* M F: diagonal for a shift, a monomial (permutation x phase) matrix for an affine map."""
    F = fourier(M.shape[0])
    return F.conj().T @ M @ F


def monomial_part(Mc: np.ndarray, tol: float = 1e-9):
    """If Mc is monomial, return (permutation sigma with Mc[sigma(j), j] != 0, phases); else None."""
    n = Mc.shape[0]
    sigma, ph = [], []
    for j in range(n):
        nz = np.flatnonzero(np.abs(Mc[:, j]) > tol)
        if len(nz) != 1:
            return None
        sigma.append(int(nz[0])); ph.append(Mc[nz[0], j])
    return sigma, np.array(ph)


def affine_perm_matrix(a: int, b: int, p: int) -> np.ndarray:
    """Transport matrix of x -> a x + b on Z/p (row x has its 1 in column a x + b)."""
    M = np.zeros((p, p))
    for x in range(p):
        M[x, (a * x + b) % p] = 1.0
    return M


# ---------------------------------------------------------------- twisted cohomology
def twisted_coboundary(graph: Graph, g: Sequence[int], n: int, k: int) -> np.ndarray:
    w = np.exp(2j * np.pi * k / n)
    D = np.zeros((graph.E, graph.n_vertices), dtype=complex)
    for e, (a, b) in enumerate(graph.edges):
        D[e, b] += 1.0
        D[e, a] -= w ** (int(g[e]) % n)
    return D


def twisted_betti(graph: Graph, g: Sequence[int], n: int, k: int) -> Tuple[int, int]:
    D = twisted_coboundary(graph, g, n, k)
    r = np.linalg.matrix_rank(D, tol=1e-9)
    return graph.n_vertices - r, graph.E - r


# ---------------------------------------------------------------- the mod-p grid
def grid_transports(p: int, row_res: Sequence[int], col_res: Sequence[int]):
    """Context graph of the 3x3 grid (rows R0..R2 = vertices 0..2, columns C0..C2 = 3..5; edge
    R_i - C_j through the shared cell x_ij).  Transport R_i -> C_j is the overlap-conditional
    expectation: from a row section, keep x_ij, then draw the column section uniformly among
    those with the same x_ij.  Sections of a context are the p^2 triples with the right sum."""
    def sections(r):
        return [s for s in np.ndindex(p, p, p) if sum(s) % p == r % p]
    rows = [sections(r) for r in row_res]
    cols = [sections(c) for c in col_res]
    edges, K = [], []
    for i in range(3):
        for j in range(3):
            T = np.zeros((len(rows[i]), len(cols[j])))
            for a, s in enumerate(rows[i]):
                match = [b for b, t in enumerate(cols[j]) if t[i] == s[j]]
                T[a, match] = 1.0 / len(match)
            edges.append((i, 3 + j)); K.append(T)
    return Graph(6, edges), K


def grid_loop_matrices(graph: Graph, K: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Like loop_matrices, but the edge operators map between different section spaces, so
    the reverse edge uses the Bayesian inverse T^T normalised by rows (uniform sections)."""
    def op(u, v):
        for e, (a, b) in enumerate(graph.edges):
            if (a, b) == (u, v):
                return K[e]
            if (a, b) == (v, u):
                R = K[e].T.copy()
                return R / R.sum(1, keepdims=True)
        raise KeyError
    out = []
    for i in range(graph.beta1):
        w = loop_walk(graph, i)
        M = None
        for u, v in zip(w[:-1], w[1:]):
            T = op(u, v)
            M = T if M is None else M @ T
        out.append(M)
    return out


# ---------------------------------------------------------------- certificate phase (local data only)
# Each context C carries LOCAL linear data over F_p:  sum_v A[C, v] x_v = b_C   (a row of A and its
# right-hand side).  Operation edges x_b - a x_a = g, parity rows sum x = r, literals x = c all fit.
# Fredholm alternative over F_p: the family glues iff y.b = 0 for every y with yA = 0.  When the left
# null space is spanned by one certificate y, the phase
#       Phi(b) = sum_C y_C b_C  (mod p)
# is a COMPLETE invariant of the obstruction, and it is a sum of local contributions y_C b_C.
# A local perturbation b_C -> b_C + eps moves it by y_C eps; contexts outside supp(y) do not enter.
def _rref_left_null(A: np.ndarray, p: int) -> np.ndarray:
    """Basis of {y : y A = 0 mod p} (rows)."""
    m, n = A.shape
    M = np.concatenate([A % p, np.eye(m, dtype=np.int64)], 1).astype(np.int64)
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, m) if M[i, c] % p), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p
        for i in range(m):
            if i != r and M[i, c]:
                M[i] = (M[i] - M[i, c] * M[r]) % p
        r += 1
    return M[r:, n:] % p


def certificate_phase(A: np.ndarray, b: np.ndarray, p: int):
    """Returns (Y basis of certificates, Phi = Y b mod p, local contributions Y[i, C] b_C)."""
    Y = _rref_left_null(np.asarray(A, dtype=np.int64), p)
    b = np.asarray(b, dtype=np.int64) % p
    return Y, (Y @ b) % p, (Y * b[None, :]) % p


def grid_system(row_res, col_res):
    """3x3 grid: variables x_ij (index 3i+j); contexts rows then columns."""
    A = np.zeros((6, 9), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            A[i, 3 * i + j] = 1
            A[3 + j, 3 * i + j] = 1
    return A, np.array(list(row_res) + list(col_res))


def operation_system(graph: Graph, ops, p: int):
    """Edge (u, v) with op x -> a x + g contributes the local row  x_v - a x_u = g."""
    A = np.zeros((graph.E, graph.n_vertices), dtype=np.int64)
    b = np.zeros(graph.E, dtype=np.int64)
    for e, ((u, v), (a, g)) in enumerate(zip(graph.edges, ops)):
        A[e, v] += 1; A[e, u] -= a; b[e] = g
    return A % p, b % p


def continuous_phase(Y_row: np.ndarray, b_real: np.ndarray, p: int) -> complex:
    """Real lift: local data b_C in R (e.g. estimated or perturbed); the shadow is the phase
    exp(2 pi i sum_C y_C b_C / p), continuous in every local b_C."""
    return complex(np.exp(2j * np.pi * float(Y_row @ b_real) / p))
