"""Outcome toggling: which phase changes can flip an outcome, and how far away the flip is.

A bias cochain b on the source-topic edges splits W-orthogonally, b = X beta + h (exact / natural part + loops).
An outcome read off b by a threshold, o = sign f(b), f(b) = <a, b>_W - c, has the same split of its reading vector:
    a = a_exact + a_loop,   a_exact = W-orthogonal projection of a onto im X.
Proposition (toggle directions). A change Delta b restricted to a subspace V moves f by <P_V a, Delta b>_W, so the
smallest change in V that flips o has W-norm
    r_V = |f(b)| / ||P_V a||_W          (infinite when a is W-orthogonal to V).
In particular: natural changes (V = im X, re-calibrations u_s + v_t) cannot flip an outcome that reads only loops
(a_exact = 0), and loop changes cannot flip an outcome that reads only the exact part (a_loop = 0).
AMB (possibilistic) outcomes read supports: an AMB outcome flips only when some probability reaches 0 or leaves 0,
so its toggle radius in the probability simplex is the smallest probability that must vanish (or appear).
"""
from __future__ import annotations
import numpy as np


def split_functional(a, X, w):
    a = np.asarray(a, float); W = np.asarray(w, float)
    G = X.T @ (W[:, None] * X)
    coef = np.linalg.pinv(G) @ (X.T @ (W * a))
    ae = X @ coef
    return ae, a - ae


def wnorm(v, w):
    return float(np.sqrt((np.asarray(w) * np.asarray(v) ** 2).sum()))


def toggle_radius(a, b, X, w, c=0.0):
    """returns f(b), radius over all changes, over natural changes, over loop changes (W-norm)."""
    a = np.asarray(a, float); b = np.asarray(b, float); w = np.asarray(w, float)
    f = float((w * a * b).sum() - c)
    ae, al = split_functional(a, X, w)
    r = lambda v: abs(f) / wnorm(v, w) if wnorm(v, w) > 1e-12 else np.inf
    return dict(f=f, r_all=r(a), r_natural=r(ae), r_loop=r(al), share_exact=wnorm(ae, w) ** 2 / max(wnorm(a, w) ** 2, 1e-300))
