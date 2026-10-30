"""Layered restriction structure for corners (companion "A tangent structure for injected data", Section "Layers"). New module.

Three layers, each with its own restriction criterion:
  strata       the relatively open faces of a polyhedron; restriction idempotent = the indicator of a face; on each open face the cone
               structure is a tangent category (universal lift). The face a direction u enters from x is G(x, u) = supp x u {u > 0};
               the vertical fibre over (x, u) is the cone of that face.
  resolutions  a map singular on a stratum becomes smooth after lifting (blowing up): r-hat lifted to the parameter space at a
               coincidence; polar coordinates v = rho theta at the merge
  multi-maps   data -> explanations is a correspondence; its domain is cut out by the existence criteria (sign, modulus, positivity),
               its branches are the sign sheets, and its graph has corners where branches are born or die

  face_of(x), entered_face(x, u), face_cone_member(G, w)
  fibre_by_layer(x, U, W)            the fibre of T(p) over 0 at x, sorted by the entered layer
  open_face_universal(x, U, W)       universality for the open face of x as an object (directions tangent to the face)
  triangle_family(lam)               the triangle with c_01 scaled by lam: the explanation segment [y_lo, y_hi] in y = log t (or None)
  two_edge_sheet(w)                  the relative-sign sheet of two disjoint edges: its Chebyshev radius (0 or less = empty)
  pipeline_domain(P)                 M01 M02 M12 > 0: the post-probe pipeline's domain
"""
from __future__ import annotations
import itertools
import numpy as np
from . import tangent as TG
from . import covers as CV


def face_of(x, tol=1e-12): return frozenset(i for i in range(len(x)) if x[i] > tol)


def entered_face(x, u, tol=1e-12): return frozenset(i for i in range(len(x)) if x[i] > tol or u[i] > tol)


def face_cone_member(G, w, n):
    """w in the feasible cone at a relative-interior point of the face G: zero-sum and w_z >= 0 off G"""
    return abs(np.sum(w)) < 1e-9 and all(w[z] >= -1e-12 for z in range(n) if z not in G)


def fibre_by_layer(x, U, W):
    """returns (fibre from the lexicographic test, layered prediction): sets of (u, w) and a dict layer -> count"""
    n = len(x); fib = set(); pred = set(); layers = {}
    for u in U:
        if not TG.feasible_cone(x, u): continue
        G = entered_face(x, u)
        for w in W:
            if TG.in_T(TG.T2_point(x, u, np.zeros(n), w), 2): fib.add((tuple(u), tuple(w)))
            if face_cone_member(G, w, n): pred.add((tuple(u), tuple(w))); layers[tuple(sorted(G))] = layers.get(tuple(sorted(G)), 0) + 1
    return fib, pred, layers


def open_face_universal(x, U, W):
    """the open face of x as its own object: directions with u_z = 0 off the face; fibre of T(p) over 0 = image of the lift?"""
    F = face_of(x); n = len(x); tang = [u for u in U if abs(np.sum(u)) < 1e-9 and all(abs(u[z]) < 1e-12 for z in range(n) if z not in F)]
    fib = {(tuple(u), tuple(w)) for u in tang for w in tang}          # inside an open face every tangent pair is admissible
    img = {(tuple(u), tuple(w)) for u in tang for w in tang if TG.feasible_cone(x, u) and TG.feasible_cone(x, w)}
    return fib == img, len(fib)


# ------------------------------------------------------------------------------------------------ multi-maps
TRI = [(0, 1), (1, 2), (0, 2)]


def triangle_family(lam, w=0.4, a1=(.8, .3, .7), a2=(.2, .7, .15)):
    mu, c, v = CV.truth_data(w, np.array(a1), np.array(a2), TRI); cc = dict(c); cc[(0, 1)] = c[(0, 1)] * lam
    F = CV.factorise(3, TRI, cc)
    if not F["ok"]: return None
    out = None
    for sg in ([1.0], [-1.0]):
        A, b, _ = CV.sheet_constraints(mu, F, sg); ylo = -min(b[A[:, 0] < 0]); yhi = min(b[A[:, 0] > 0])
        if yhi >= ylo: out = (sg[0], float(ylo), float(yhi))
    return out


TWO = [(0, 1), (2, 3)]


def two_edge_sheet(w, a1=(.05, .02, .15, .05), a2=(.55, .45, .75, .9)):
    mu, c, v = CV.truth_data(w, np.array(a1), np.array(a2), TWO); F = CV.factorise(4, TWO, c)
    s0 = [float(np.sign(v[C["vertices"][0]] / C["v0"][C["vertices"][0]])) for C in F["comps"]]
    rel = [s0[0], -s0[1]]; A, b, _ = CV.sheet_constraints(mu, F, rel); z, rad = CV.chebyshev(A, b)
    At, bt, _ = CV.sheet_constraints(mu, F, s0); zt, radt = CV.chebyshev(At, bt)
    return rad, radt


def pipeline_domain(P):
    from .lab import moments
    _, M, _ = moments(P); return M[0, 1] * M[0, 2] * M[1, 2]
