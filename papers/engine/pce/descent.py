"""The descent tower of the latent-actor lab (lab paper, Proposition "spectrum and descent" and Theorem M). New module.

Level 0 (ambient laws). Contexts = observed windows, arrows = marginalisation. Over the algebraic spectrum (signed laws) the limit of
the diagram is the affine space of consistent pseudo-marginals; the marginal map from signed global laws onto it is surjective and its
kernel is spanned by the Walsh characters chi_S of the sets S contained in no window (descent holds up to that kernel: the presheaf is not
separated). Over the real probability points the image is the marginal polytope, strictly inside the local polytope when the cover has
a cycle: the obstruction is positivity only (pce.cohomology.separate, the LP witness of pce.integrate).

Level 1 (two-type pair layer). M_ij = v_i v_j is a multiplicative 1-cochain on the graph of pairs, with the unsigned coboundary
(d+ v)_ij = v_i v_j. Write R^x = {+-1} x R_>0:
  sign part     over Z/2 the unsigned and signed coboundaries agree: kernel H^0(G; Z/2) = Z/2 (the label switch v -> -v), cokernel
                H^1(G; Z/2) (cycle classes; on the triangle the class is sign(M_01 M_02 M_12) = sign(Delta_m))
  modulus part  log|M| in R^E modulo the image of the unsigned incidence matrix B: zero cokernel on the triangle (no equation at
                m = 3), dimension 2 on K4 (the two tetrads); kernel 0 when the graph is connected and not bipartite
  walls         positivity: the hiding segment is non-empty (semialgebraic, not cohomological)
Level 2 (triple layer). r_S = m_S / v_S is a 0-cochain on the triple windows; its coboundary (differences on overlaps) is the
triple-consistency residual; the quadruple window adds m_0123 = v_0123 (r^2 + 1).

  pair_graph(m, windows)            edges and the unsigned incidence matrix B (E x m)
  descent_kernel(m, windows)        kernel of the marginal map on signed laws, and the Walsh characters that span it
  gf2_rank(A), gf2_solve(A, b)      linear algebra over Z/2
  sign_class(M, edges)              the class of the sign cochain in coker(B mod 2): a vector of fundamental-cycle parities
  modulus_class(M, edges, B)        the projection of log|M| onto coker(B) over R (zero iff a modulus solution exists)
  pair_realizable(P)                (sign class, modulus class, walls) for the pair tables of P (m = 3 or 4)
  triple_coboundary(P)              (delta r) on the triple cover (m = 4)
"""
from __future__ import annotations
import itertools
import numpy as np
from .lab import moments, segment, fit_v
from . import injection as IJ
from .cohomology import cech, cohomology_dims, _res, _space


def pair_graph(m, windows=None):
    edges = [tuple(sorted(w)) for w in windows if len(w) == 2] if windows else list(itertools.combinations(range(m), 2))
    B = np.zeros((len(edges), m))
    for k, (i, j) in enumerate(edges): B[k, i] = B[k, j] = 1.0
    return edges, B


def marginal_map(m, windows):
    """signed global laws R^{2^m} -> C^0 = sum of R^{X_U} over the windows (in pce.cohomology's coordinates)"""
    dom = {i: 2 for i in range(m)}; full = tuple(range(m))
    return np.vstack([_res(dom, full, tuple(sorted(w))) for w in windows])


def descent_kernel(m, windows):
    A = marginal_map(m, windows); u, s, vt = np.linalg.svd(A); rk = int(np.sum(s > 1e-9)); K = vt[rk:].T
    outs = IJ.outcomes(m); covered = lambda S: any(set(S) <= set(w) for w in windows)
    chis = [np.array([np.prod([2 * x[i] - 1 for i in S]) for x in outs], float) for k in range(1, m + 1) for S in itertools.combinations(range(m), k) if not covered(S)]
    return K, np.array(chis).T if chis else np.zeros((2 ** m, 0)), rk


def level0_cohomology(m, windows, qmax=3):
    dom = {i: 2 for i in range(m)}; ds, off, U = cech(dom, [tuple(w) for w in windows], qmax=qmax); return cohomology_dims(ds)


def consistent_space_dim(m, windows):
    """dimension of ker d^0 (the algebraic limit, as a linear space of consistent signed families)"""
    dom = {i: 2 for i in range(m)}; ds, _, _ = cech(dom, [tuple(w) for w in windows], qmax=0); d0 = ds[0]
    return d0.shape[1] - np.linalg.matrix_rank(d0, 1e-9)


# ------------------------------------------------------------------------------------------------ Z/2 linear algebra
def gf2_rank(A):
    A = (np.asarray(A) % 2).astype(np.uint8).copy(); r = 0; rows, cols = A.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c]), None)
        if piv is None: continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(rows):
            if i != r and A[i, c]: A[i] ^= A[r]
        r += 1
        if r == rows: break
    return r


def in_image_gf2(B, b):
    B = (np.asarray(B) % 2).astype(np.uint8); b = (np.asarray(b) % 2).astype(np.uint8).reshape(-1, 1)
    return gf2_rank(B) == gf2_rank(np.hstack([B, b]))


def cycle_basis(m, edges):
    """fundamental cycles of the graph (as 0/1 edge vectors), from a BFS spanning tree"""
    adj = {i: [] for i in range(m)}
    for k, (i, j) in enumerate(edges): adj[i].append((j, k)); adj[j].append((i, k))
    parent = {0: (None, None)}; order = [0]
    for u in order:
        for v, k in adj[u]:
            if v not in parent: parent[v] = (u, k); order.append(v)
    tree = {k for v, (u, k) in parent.items() if k is not None}
    def path(v):
        out = set()
        while parent[v][0] is not None: out ^= {parent[v][1]}; v = parent[v][0]
        return out
    cyc = []
    for k, (i, j) in enumerate(edges):
        if k in tree: continue
        c = path(i) ^ path(j) ^ {k}; z = np.zeros(len(edges), np.uint8); z[list(c)] = 1; cyc.append(z)
    return cyc


def sign_class(M, edges, m):
    """parities of the sign cochain around the fundamental cycles: all zero iff signs are a coboundary sigma_i sigma_j"""
    s = np.array([1 if M[i, j] < 0 else 0 for i, j in edges], np.uint8)
    return np.array([int((c & s).sum() % 2) for c in cycle_basis(m, edges)])


def modulus_class(M, edges, B):
    y = np.log(np.abs(np.array([M[i, j] for i, j in edges]))); proj = y - B @ np.linalg.lstsq(B, y, rcond=None)[0]; return proj


def pair_realizable(P, tol=1e-9):
    m = P.ndim; mu, M, _ = moments(P); edges, B = pair_graph(m); sc = sign_class(M, edges, m); mc = modulus_class(M, edges, B)
    algebraic = (not sc.any()) and np.linalg.norm(mc) < tol
    walls = None
    if algebraic:
        from .segment_decision import v_closed
        v = v_closed(M) if m == 3 else fit_v(mu, M); lo, hi = segment(mu, v); walls = bool(np.isfinite(hi) and hi > lo)
    return dict(sign=sc, modulus=mc, algebraic=algebraic, walls=walls, realizable=bool(algebraic and walls))


def triple_coboundary(P):
    """m >= 4: r_S = m_S / v_S on each triple window; returns the 0-cochain and its coboundary on overlapping pairs of triples"""
    m = P.ndim; mu, M, mS = moments(P); v = fit_v(mu, M); T = list(itertools.combinations(range(m), 3))
    r = {S: mS(S) / np.prod(v[list(S)]) for S in T}
    d = {(S, U): r[U] - r[S] for S, U in itertools.combinations(T, 2) if len(set(S) & set(U)) >= 2}
    return r, d


# ------------------------------------------------------------------------------------------------ the labelled layer
def labelled_coboundary(m_b, src):
    """batches carry the shared statistic m_T (a 0-cochain on the complete nerve of the batch cover: every pair of batches overlaps in
    the whole triple context). delta m on the edges, its energy, and the orthogonal split of the energy into between-source and
    within-source parts (the coarse and the fine cover)"""
    m_b = np.asarray(m_b, float); src = np.asarray(src); K = len(m_b)
    d = np.array([m_b[c] - m_b[b] for b, c in itertools.combinations(range(K), 2)]); energy = float(d @ d)
    mbar = m_b.mean(); labs = sorted(set(src.tolist())); between = sum((src == j).sum() * (m_b[src == j].mean() - mbar) ** 2 for j in labs)
    within = sum(((m_b[src == j] - m_b[src == j].mean()) ** 2).sum() for j in labs)
    return d, energy, float(between), float(within)


def two_by_two_realizable(T):
    """every 2x2 table is a mixture of two product laws (non-negative rank <= 2): the rows give the decomposition"""
    T = np.asarray(T, float); out = []
    for i in range(2):
        w = T[i].sum()
        if w > 0: out.append((w, np.outer(np.eye(2)[i], T[i] / w)))
    return out
