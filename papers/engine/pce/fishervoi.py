"""Fisher geometry against decision relevance (companion note, remark in section 12.23). New module; it modifies nothing and uses
pce/margsec.py and pce/holonomy.py.
  bc(p, q), fisher_rao(p, q), tv(p, q)       Bhattacharyya coefficient, Fisher-Rao distance 2 arccos BC on the simplex, total variation
  fubini_study(z1, z2)                       arccos |<z1, z2>| for unit complex amplitudes (z = sqrt(p) e^{i theta})
  observed_fisher(P)                         Fisher information (on the tangent space sum v = 0) of one reading in each observed context,
                                             I = sum_c A_c^T diag(1 / A_c P) A_c, A_c the 4 x 16 indicator rows of context c
  full_fisher(P)                             diag(1 / P): the Fisher information of a complete reading
  null_directions(P)                         an orthonormal basis of the null space of observed_fisher on the tangent space
  normal_complexity(k)                       sqrt(det g) of the normal family N(mu, sigma^2) in the coordinates (mu, v = sigma^k) (sympy, exact)
  normal_isometry()                          the pullback of the normal family's Fisher metric under (mu, sigma) -> (a mu + b, a sigma), minus the
                                             metric (sympy; zero matrix)
"""
from __future__ import annotations
import numpy as np
from . import holonomy as H


def bc(p, q): return float(np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None))))
def fisher_rao(p, q): return 2.0 * float(np.arccos(min(1.0, bc(p, q))))
def tv(p, q): return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())
def fubini_study(z1, z2): return float(np.arccos(min(1.0, abs(np.vdot(z1, z2)))))


def _tangent():
    """an orthonormal basis (15 x 16) of {v : sum v = 0}"""
    u, s, vt = np.linalg.svd(np.ones((1, 16))); return vt[1:]


def observed_fisher(P):
    P = np.asarray(P, float); I = np.zeros((16, 16))
    for k in range(4):
        A = H.ctx_rows(k); I += A.T @ np.diag(1.0 / (A @ P)) @ A
    B = _tangent(); return B @ I @ B.T, B


def full_fisher(P):
    B = _tangent(); return B @ np.diag(1.0 / np.asarray(P, float)) @ B.T, B


def null_directions(P, tol=1e-10):
    I, B = observed_fisher(P); w, V = np.linalg.eigh(I); return (V[:, w < tol * w.max()].T) @ B


def normal_complexity(k):
    import sympy as sp
    mu, s, v = sp.symbols("mu sigma v", positive=True)
    g = sp.diag(1 / s**2, 2 / s**2)                                            # Fisher metric of N(mu, sigma^2) in (mu, sigma)
    sig = v ** sp.Rational(1, k) if k != 0 else sp.exp(v)
    J = sp.diag(1, sp.diff(sig, v)); gv = (J.T * g.subs(s, sig) * J).applyfunc(sp.simplify)
    return sp.simplify(sp.sqrt(gv.det()).subs(v, s**k if k != 0 else sp.log(s)))


def normal_isometry():
    import sympy as sp
    mu, s, a, b = sp.symbols("mu sigma a b", positive=True)
    g = lambda m, t: sp.diag(1 / t**2, 2 / t**2)
    J = sp.Matrix([[a, 0], [0, a]])                                            # differential of (mu, sigma) -> (a mu + b, a sigma)
    return (J.T * g(a * mu + b, a * s) * J - g(mu, s)).applyfunc(sp.simplify)
