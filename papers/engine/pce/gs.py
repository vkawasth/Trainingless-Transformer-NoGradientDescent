"""Gerstenhaber-Schack cohomology of the diagram of function algebras over the context poset.

The diagram. Contexts form the intersection-closed poset P (pce.complex.poset). Order it by reverse inclusion, i <= j iff C_i contains
C_j, so that pulling functions back along the marginal maps is a presheaf of commutative algebras:
    A(i) = R^{X_{C_i}} (pointwise product),   phi_ij : A(j) -> A(i) for i <= j (pullback).
Probabilities are states on these algebras; marginalisation is restriction of states. The algebraic diagram does not know the states.

The Gerstenhaber-Schack double complex (coefficients in A itself), over strictly increasing chains s = (i_0 < ... < i_p):
    C^{p,q} = prod over s of Hom(A(i_p)^{(x) q}, A(i_0)),
with the Hochschild differential in q (A(i_p) acts on A(i_0) through phi) and the simplicial differential in p
    (d_S f)_s = phi_{i_0 i_1} f_{s minus i_0} + sum_{k=1..p} (-1)^k f_{s minus i_k} + (-1)^{p+1} f_{s minus i_{p+1}} o phi^{(x)q},
and total differential D = d_S + (-1)^p d_H.

  gs_complex(domains, contexts, nmax)   the total complex up to degree nmax + 1 (dense), with D o D = 0 checked by the caller
  gs_dims(...)                          dim H^n_GS for n <= nmax
  row_dims(...)                         the q = 0 row alone: cohomology of the order complex with coefficients in A
  predicted_dims(...)                   sum over variable sets S of dim E_S x dim H^n(Delta P, Delta(P minus U_S)), U_S = contexts containing S
  hochschild_vertex(dB, dM, phi, q)     the Hochschild differential of R^dB with coefficients in R^dM (through phi): HH^q = 0 for q >= 1

Tolerance.
  lam_tolerance(eps)                    lam* is Lipschitz: |lam*(t) - lam*(t')| <= sum over windows of TV(t_w, t'_w); so with window errors
                                        eps_w a family that truly glues has 1 - lam*(estimate) <= sum eps_w
  rank_weyl(H, noise_norm)              number of singular values above the spectral norm of the noise: by Weyl's inequality it never
                                        exceeds the rank of the noiseless matrix (whose flattening lies on the k-th secant variety)
"""
from __future__ import annotations
import itertools
import numpy as np
from .complex import poset
from .cohomology import _res, _space


# ------------------------------------------------------------------------------------------------------------- the diagram
def diagram(domains, contexts):
    uniq = {}
    for k, v in contexts.items():                                         # contexts with the same variables are one context
        if frozenset(v) not in uniq.values(): uniq[k] = frozenset(v)
    el, _, _ = poset(uniq)
    names = sorted(el, key=lambda k: (-len(el[k]), sorted(el[k])))
    sets = {k: tuple(sorted(el[k])) for k in names}
    leq = {(a, b) for a in names for b in names if set(sets[a]) >= set(sets[b])}       # a <= b iff C_a contains C_b
    dim = {k: len(_space(domains, sets[k])[1]) for k in names}
    phi = {(a, b): _res(domains, sets[a], sets[b]).T for a, b in leq}                  # pullback A(b) -> A(a)
    return names, sets, leq, dim, phi


def chains(names, leq, p):
    out = []
    for c in itertools.permutations(names, p + 1):
        if all((c[k], c[k + 1]) in leq and c[k] != c[k + 1] for k in range(p)): out.append(c)
    return out


# ------------------------------------------------------------------------------------------------- Hochschild differential
def hochschild_vertex(dB, dM, phi, q):
    """matrix of d_H : Hom(B^{(x)q}, M) -> Hom(B^{(x)(q+1)}, M), B = R^dB, M = R^dM, B acting through phi (dM x dB, 0/1 pullback);
    basis of B: idempotents e_x; a cochain is an array f[m, x_1..x_q]."""
    src = list(itertools.product(range(dB), repeat=q)); tgt = list(itertools.product(range(dB), repeat=q + 1))
    si = {t: k for k, t in enumerate(src)}
    D = np.zeros((dM * len(tgt), dM * len(src)))
    for ti, xs in enumerate(tgt):
        for m in range(dM):
            row = m * len(tgt) + ti
            a = phi[m, xs[0]]                                             # phi(e_{x1}) f(x2..)
            if a: D[row, m * len(src) + si[xs[1:]]] += a
            for k in range(q):                                            # f(.., x_k x_{k+1}, ..)
                if xs[k] == xs[k + 1]:
                    D[row, m * len(src) + si[xs[:k + 1] + xs[k + 2:]]] += (-1) ** (k + 1)
            b = phi[m, xs[-1]]                                            # f(x1..xq) phi(e_{x_{q+1}})
            if b: D[row, m * len(src) + si[xs[:-1]]] += (-1) ** (q + 1) * b
    return D


# --------------------------------------------------------------------------------------------------------- total complex
def gs_complex(domains, contexts, nmax=2):
    names, sets, leq, dim, phi = diagram(domains, contexts)
    blocks = {}                                                           # (p, q) -> list of (chain, offset, size)
    off = {}
    for n in range(nmax + 2):
        o = 0; off[n] = []
        for p in range(n + 1):
            q = n - p
            for c in chains(names, leq, p):
                sz = dim[c[0]] * dim[c[-1]] ** q; off[n].append(((p, q), c, o, sz)); o += sz
        off[n].append(("dim", o))
    idx = {n: {(pq, c): (o, sz) for pq, c, o, sz in off[n][:-1]} for n in off}
    Ds = []
    for n in range(nmax + 1):
        D = np.zeros((off[n + 1][-1][1], off[n][-1][1]))
        for (p, q), c, o, sz in off[n][:-1]:
            # Hochschild part: (p, q) -> (p, q + 1), sign (-1)^p
            dH = hochschild_vertex(dim[c[-1]], dim[c[0]], phi[(c[0], c[-1])], q)
            o2, sz2 = idx[n + 1][((p, q + 1), c)]
            D[o2:o2 + sz2, o:o + sz] += (-1) ** p * dH
            # simplicial part: (p, q) -> (p + 1, q); the image chain s has p + 2 elements and contains c as a face
            for s in chains(names, leq, p + 1):
                for k in range(p + 2):
                    if s[:k] + s[k + 1:] != c: continue
                    o2, sz2 = idx[n + 1][((p + 1, q), s)]
                    dB = dim[s[-1]]; dM = dim[s[0]]
                    if k == 0:                                            # pull the value from A(s_1) to A(s_0)
                        P = phi[(s[0], s[1])]                             # dM x dim(s_1)
                        M = np.kron(P, np.eye(dB ** q))
                    elif k == p + 1:                                      # pull the arguments from A(s_{p+1}) to A(s_p)
                        P = phi[(s[p], s[p + 1])]                         # dim(s_p) x dB
                        Pq = np.ones((1, 1))
                        for _ in range(q): Pq = np.kron(Pq, P)
                        M = np.kron(np.eye(dM), Pq.T) * (-1) ** (p + 1)
                    else:
                        M = np.eye(sz) * (-1) ** k
                    D[o2:o2 + sz2, o:o + sz] += M
        Ds.append(D)
    return Ds


def _ranks(Ds, tol=1e-8):
    return [np.linalg.matrix_rank(D, tol) if D.size else 0 for D in Ds]


def gs_dims(domains, contexts, nmax=2):
    Ds = gs_complex(domains, contexts, nmax)
    r = _ranks(Ds)
    return [Ds[n].shape[1] - r[n] - (r[n - 1] if n else 0) for n in range(nmax + 1)], Ds


# ------------------------------------------------------------------------------------------------------ the q = 0 row
def row_complex(domains, contexts, nmax=3):
    names, sets, leq, dim, phi = diagram(domains, contexts)
    ch = {p: chains(names, leq, p) for p in range(nmax + 2)}
    offs = {p: np.cumsum([0] + [dim[c[0]] for c in ch[p]]) for p in ch}
    Ds = []
    for p in range(nmax + 1):
        D = np.zeros((offs[p + 1][-1], offs[p][-1])); ci = {c: k for k, c in enumerate(ch[p])}
        for j, s in enumerate(ch[p + 1]):
            for k in range(p + 2):
                f = s[:k] + s[k + 1:]; i = ci[f]
                blk = phi[(s[0], s[1])] if k == 0 else np.eye(dim[s[0]])
                D[offs[p + 1][j]:offs[p + 1][j + 1], offs[p][i]:offs[p][i + 1]] += (-1) ** k * blk
        Ds.append(D)
    return Ds


def row_dims(domains, contexts, nmax=3):
    Ds = row_complex(domains, contexts, nmax); r = _ranks(Ds)
    return [Ds[n].shape[1] - r[n] - (r[n - 1] if n else 0) for n in range(nmax + 1)]


def _rel_cohomology(names, leq, keep, nmax):
    """dims of H^n(Delta P, Delta(P minus keep)) with real coefficients: cochains on chains whose first (largest) element is in keep."""
    ch = {p: [c for c in chains(names, leq, p) if c[0] in keep] for p in range(nmax + 2)}
    Ds = []
    for p in range(nmax + 1):
        ci = {c: k for k, c in enumerate(ch[p])}
        D = np.zeros((len(ch[p + 1]), len(ch[p])))
        for j, s in enumerate(ch[p + 1]):
            for k in range(p + 2):
                f = s[:k] + s[k + 1:]
                if f in ci: D[j, ci[f]] += (-1) ** k
        Ds.append(D)
    r = _ranks(Ds)
    return [Ds[n].shape[1] - r[n] - (r[n - 1] if n else 0) for n in range(nmax + 1)]


def predicted_dims(domains, contexts, nmax=3):
    names, sets, leq, dim, phi = diagram(domains, contexts)
    allv = sorted({v for s in sets.values() for v in s})
    tot = [0] * (nmax + 1); terms = {}
    covered = {frozenset(S) for s in sets.values() for k in range(len(s) + 1) for S in itertools.combinations(s, k)}
    for S in covered:
        keep = {a for a in names if set(S) <= set(sets[a])}
        h = _rel_cohomology(names, leq, keep, nmax); e = int(np.prod([domains[v] - 1 for v in S]))
        terms[tuple(sorted(S))] = h
        for n in range(nmax + 1): tot[n] += e * h[n]
    return tot, terms


def order_complex_dims(domains, contexts, nmax=3):
    names, sets, leq, dim, phi = diagram(domains, contexts)
    return _rel_cohomology(names, leq, set(names), nmax)


# ---------------------------------------------------------------------------------------------------------------- tolerance
def lam_tolerance(eps):
    return float(sum(eps))


def rank_weyl(H, noise_norm):
    s = np.linalg.svd(np.asarray(H, float), compute_uv=False)
    return int((s > noise_norm).sum())


def secant_distance(H, k):
    """Frobenius distance of H to the matrices of rank <= k (Eckart-Young); a lower bound on the distance of the Hankel flattening to
    the k-th secant variety of exchangeable laws with k hypotheses."""
    s = np.linalg.svd(np.asarray(H, float), compute_uv=False)
    return float(np.sqrt((s[k:] ** 2).sum()))
