"""Strata of two-type covers (companion "Two latent types on arbitrary covers", Phase III, Stage 4). New module.

A stratum is a set Z of variables with v_i = 0 (coincidences; zero pair moments in the model are exactly these), possibly with r = 0
(equal weights) or Z = everything (the merge).

  latent_rank(m, faces, v, r)        rank of the Jacobian of (v, r) -> (m_F) at a point (finite differences)
  latent_rank_pred(m, faces, Z)      rank of A on the faces avoiding Z (coordinates outside Z, and r) + #{j in Z : some face meets Z
                                     exactly in j}: faces meeting Z twice vanish to second order, faces meeting it once give e_j
  b_degree(gen, names, faces, i)     the A-degree of a generator in variable i: how many factors of each monomial contain i
  tests_df(gens, names, point)       rank of the generators' Jacobian at an observed point
  lobes(m, edges, i)                 the number of components C of G - i for which C + i is non-bipartite
  sheets_at_r0(m, faces)             the sign group when r = 0: odd faces carry no sign (m_T = 0), so G_2 of the even faces on v
"""
from __future__ import annotations
import itertools
import numpy as np
from . import covers as CV
from . import signs as SG
from . import syzygy as SZ


def phi(m, faces, z):
    return np.array([np.prod(z[list(F)]) * (1.0 if len(F) == 2 else CV.s_poly(len(F), z[m])) for F in faces])


def latent_rank(m, faces, v, r, h=1e-7):
    z0 = np.r_[v, r]; J = np.array([(phi(m, faces, z0 + h * e) - phi(m, faces, z0 - h * e)) / (2 * h) for e in np.eye(m + 1)]).T
    return int(np.linalg.matrix_rank(J, tol=1e-6))


def latent_rank_pred(m, faces, Z):
    Z = set(Z); keep = [F for F in faces if not (set(F) & Z)]; coords = [i for i in range(m) if i not in Z] + [m]
    A = np.zeros((len(coords), len(keep)))
    for j, F in enumerate(keep):
        for i in F: A[coords.index(i), j] = 1
        if len(F) >= 3: A[coords.index(m), j] = 1                    # r enters every higher window at a generic r != 0
    base = int(np.linalg.matrix_rank(A)) if keep else 0
    single = {j for j in Z if any(set(F) & Z == {j} for F in faces)}
    return base + len(single)


def b_degree(gen, names, faces, i):
    import sympy
    syms = sympy.symbols(names); P = sympy.Poly(sympy.sympify(gen.replace("^", "**"), locals=dict(zip(names, syms))), *syms)
    degs = set()
    for e, c in P.terms():
        degs.add(sum(k for k, nm in zip(e, faces) if i in nm))
    return degs


def tests_df(gens, names, point, h=1e-6):
    J = np.zeros((len(gens), len(names)))
    for j, nm in enumerate(names):
        p1 = dict(point); p2 = dict(point); p1[nm] += h; p2[nm] -= h
        J[:, j] = [(SZ.eval_rel(g, p1) - SZ.eval_rel(g, p2)) / (2 * h) for g in gens]
    return int(np.linalg.matrix_rank(J, tol=1e-7)), J


def lobes(m, edges, i):
    Ei = [e for e in edges if i not in e]; comps = [C for C in CV.components(m, Ei) if i not in C]; cnt = 0
    for C in comps:
        sub = [e for e in edges if all(x in C or x == i for x in e)]
        if any(i in e for e in sub) and CV.bipartition(sorted(set(C) | {i}), sub) is None: cnt += 1
    return cnt


def sheets_at_r0(m, faces):
    even = [F for F in faces if len(F) % 2 == 0]; G = SG.sign_group(m, even); used_v = [c for c in G["used"] if c < m]
    return G
