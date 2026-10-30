"""Real signs and parity for two-type covers (companion "Two latent types on arbitrary covers", Phase III, Stage 3). New module.

The sign symmetries of the parametrisation m_F = v_F s_|F|(r) are sigma in {+-1}^(m+1) (one sign per v_i and one for r) with
prod_{i in F} sigma_i * sigma_r^|F| = 1 for every observed face F, because s_k(-r) = (-1)^k s_k(r). Over F_2 this is P^T sigma = 0
with the parity matrix P: one column per face, 1_F + (|F| mod 2) e_r. For pairs and triples P is the extended incidence matrix mod 2.

  parity_matrix(m, faces)            P over F_2 ((m + 1) x n) and the list of coordinates that occur in some face
  gf2_nullspace(M)                   a basis of the right null space over F_2
  sign_group(m, faces)               G_2 = ker P^T restricted to the coordinates that occur (free coordinates are continuous), its
                                     dimension, the label swap (all ones), and the number of sheets |G_2| / 2
  determined(m, faces, combo)        is the sign combination combo (a set of coordinates) fixed by the cover, i.e. constant on G_2?
  smith_invariants(A)                the invariant factors of an integer matrix (sympy)
  invisible_classes(A)               rank_Q A - rank_2 A: the number of even invariant factors; 2^that is the number of data sign
                                     classes allowed by every relation but realised by no real explanation
  pair_split(m, edges)               for a pair graph: b_1 = visible (sign shadows of binomial relations) + invisible (one per
                                     non-bipartite component)
"""
from __future__ import annotations
import itertools
import numpy as np


def parity_matrix(m, faces):
    cols = []
    for F in faces:
        c = np.zeros(m + 1, dtype=np.uint8); c[list(F)] = 1; c[m] = len(F) % 2; cols.append(c)
    P = np.array(cols, dtype=np.uint8).T
    used = sorted({i for F in faces for i in F} | ({m} if any(len(F) >= 3 for F in faces) else set()))
    return P, used


def gf2_rank(M):
    M = (np.array(M, dtype=np.uint8) % 2).copy(); r = 0; rows, cols = M.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]: M[i] ^= M[r]
        r += 1
        if r == rows: break
    return r


def gf2_nullspace(M):
    """basis of {x : M x = 0 mod 2}"""
    M = (np.array(M, dtype=np.uint8) % 2).copy(); rows, cols = M.shape; piv_cols = []; r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]: M[i] ^= M[r]
        piv_cols.append(c); r += 1
        if r == rows: break
    free = [c for c in range(cols) if c not in piv_cols]; basis = []
    for f in free:
        x = np.zeros(cols, dtype=np.uint8); x[f] = 1
        for k, pc in enumerate(piv_cols):
            if M[k, f]: x[pc] = 1
        basis.append(x)
    return basis


def sign_group(m, faces):
    P, used = parity_matrix(m, faces); Pu = P[used, :]
    basis = gf2_nullspace(Pu.T); dim = len(basis)
    ones = np.ones(len(used), dtype=np.uint8); swap_in = gf2_rank(np.vstack(basis + [ones])) == dim if basis else False
    return dict(used=used, basis=basis, dim=dim, swap_in=bool(swap_in), sheets=2 ** dim // 2 if dim >= 1 else 1, P=P)


def group_elements(G):
    els = []
    for coeffs in itertools.product((0, 1), repeat=G["dim"]):
        x = np.zeros(len(G["used"]), dtype=np.uint8)
        for c, b in zip(coeffs, G["basis"]):
            if c: x ^= b
        els.append(x)
    return els


def determined(m, faces, combo):
    """combo: coordinates (indices into 0..m, m = r); the product of their signs is fixed iff it is 0 on every element of G_2"""
    G = sign_group(m, faces); idx = [G["used"].index(c) for c in combo if c in G["used"]]
    if len(idx) < len(combo): return False                     # a free coordinate is involved: not a discrete invariant
    return all(int(np.sum(g[idx]) % 2) == 0 for g in group_elements(G))


def smith_invariants(A):
    from sympy import Matrix
    from sympy.matrices.normalforms import smith_normal_form
    S = smith_normal_form(Matrix(np.asarray(A, dtype=int).tolist())); d = []
    for i in range(min(S.shape)):
        if S[i, i] != 0: d.append(abs(int(S[i, i])))
    return d


def invisible_classes(A):
    rq = int(np.linalg.matrix_rank(np.asarray(A, float))); r2 = gf2_rank(np.asarray(A) % 2)
    return rq - r2, rq, r2


def pair_split(m, edges):
    from .covers import components, bipartition
    B = np.zeros((m, len(edges)), dtype=int)
    for j, (a, b) in enumerate(edges): B[a, j] = B[b, j] = 1
    comps = components(m, edges); b1 = len(edges) - m + len(comps)
    inv, rq, r2 = invisible_classes(B); nonbip = sum(1 for C in comps if bipartition(C, edges) is None)
    return dict(b1=b1, invisible=inv, nonbipartite=nonbip, visible=b1 - inv, codim=len(edges) - rq)
