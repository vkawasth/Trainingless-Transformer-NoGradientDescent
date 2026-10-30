"""Using the tangent structure on decision boundaries (companion "A tangent structure for injected data", Sections "Non-affine maps at
corners" and "Decision boundaries"). New module.

  cone_K(x, grad)                    the cone K_y = C_x cap ker Df_y(x): its lineality basis L and its extreme rays R
  second_order_ok(f, x, tol)         the criterion: for every output cell y with f_y(x) = 0, the Hessian of f_y vanishes on L x K_y and
                                     is >= 0 on R x R. f maps HD points to a list of HD output cells
  direct_T2(f, x, samples, rng)      direct test: random points of T^2 Delta over x (integer coefficients), T^2 f, lexicographic test
  grad_hess(F, p, B)                 exact gradient and Hessian of a scalar F in the basis B of zero-sum directions (Weil algebra W_2)
  second_form(F, p0, B)              the second fundamental form of the level set {F = F(p0)} at p0 (Hessian on the tangent space / |grad|)
  mixture_flip(F, p0, B, xi, delta, eta)   P+- = p0 +- delta xi + eta n: two laws on one side whose midpoint is on the other side
  budget_jets(F, p, u)               F along P + eps u: first- and second-order budgets eps1 = -F/F', eps2 (root of the quadratic) and the
                                     exact budget (bisection)
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import brentq
from . import tangent as TG
from .tangent import HD


def _coef(z, S): return z.c[S] if isinstance(z, HD) else (np.asarray(z, float) if S == 0 else 0.0)


def outputs(f, X):
    """f applied to an HD point; returns the output coefficient array (2^k, p)"""
    ys = f(X); return np.stack([y.c if isinstance(y, HD) else np.r_[float(y), np.zeros(2 ** X.k - 1)] for y in ys], axis=1)


def cone_K(x, grad, tol=1e-10):
    n = len(x); supp = [i for i in range(n) if x[i] > tol]; Z = [i for i in range(n) if x[i] <= tol]; s0 = supp[0]
    L = [np.eye(n)[s] - np.eye(n)[s0] for s in supp[1:]]
    gs = grad[s0]; R = [np.eye(n)[z] - np.eye(n)[s0] for z in Z if abs(grad[z] - gs) < 1e-9]
    return L, R


def jets_scalar(fy, x, n):
    """gradient (n) and Hessian (n x n) of one output cell at x, exactly, from W_2 points x + e_i e1 + e_j e2"""
    I = np.eye(n); g = np.zeros(n); H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            Y = fy(HD.var(x, [I[i], I[j]])); H[i, j] = H[j, i] = Y.c[3]; g[i] = Y.c[1]
    return g, H


def second_order_ok(f, x, tol=1e-9):
    n = len(x); X0 = HD.var(x, []); vals = outputs(f, X0)[0]; ok = True; detail = []
    for y in range(len(vals)):
        if vals[y] > tol: continue
        fy = lambda Xh, y=y: f(Xh)[y]; g, H = jets_scalar(fy, x, n); L, R = cone_K(x, g)
        gens = L + R
        lin = max([abs(l @ H @ k) for l in L for k in gens] + [0.0]); ray = min([r @ H @ s for r in R for s in R] + [0.0])
        good = lin < tol and ray > -tol; ok &= good; detail.append((y, lin, ray, good))
    return ok, detail


def direct_T2(f, x, samples, rng, tries=4000):
    n = len(x); bad = 0; tested = 0
    for _ in range(tries):
        c = np.zeros((4, n)); c[0] = x
        for S in (1, 2, 3):
            z = rng.integers(-2, 3, size=n).astype(float); z[-1] -= z.sum(); c[S] = z
        if not TG.in_T(c, 2): continue
        img = outputs(f, HD(c, 2)); tested += 1; bad += not TG.in_T(img, 2)
        if tested >= samples: break
    return bad == 0, tested, bad


# ------------------------------------------------------------------------------------------------ decision boundaries
def grad_hess(F, p, B):
    k = B.shape[1]; g = np.zeros(k); H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            T = F(HD.var(p, [B[:, i], B[:, j]])); H[i, j] = H[j, i] = T.c[3]; g[i] = T.c[1]
    return g, H


def second_form(F, p0, B):
    g, H = grad_hess(F, p0, B); gn = g / np.linalg.norm(g); Pt = np.eye(len(g)) - np.outer(gn, gn)
    II = Pt @ H @ Pt / np.linalg.norm(g); ev, V = np.linalg.eigh(II); keep = [k for k in range(len(ev)) if abs(V[:, k] @ gn) < 1e-8]
    return ev[keep], V[:, keep], g, H


def mixture_flip(Fr, p0, B, xi, delta, eta):
    """Fr: real-valued F; xi, n in the basis B. Returns P+, P-, midpoint and the three values"""
    g = grad_hess(lambda X: Fr(X), p0, B)[0]; nvec = B @ (g / np.linalg.norm(g)); d = B @ xi
    Pp = p0 + delta * d + eta * nvec; Pm = p0 - delta * d + eta * nvec; Mid = 0.5 * (Pp + Pm)
    val = lambda p: Fr(HD.var(p, [])).c[0]
    return Pp, Pm, Mid, (val(Pp), val(Pm), val(Mid))


def budget_jets(F, p, u, emax=0.5):
    T = F(HD.var(p, [u, u])); f0, f1, f2 = T.c[0], T.c[1], T.c[3]          # F, F', F'' along u
    e1 = -f0 / f1 if f1 < 0 else np.inf
    disc = f1 * f1 - 2 * f0 * f2; roots = []
    if abs(f2) > 1e-15 and disc >= 0: roots = [r for r in ((-f1 - np.sqrt(disc)) / f2, (-f1 + np.sqrt(disc)) / f2) if r > 0]
    e2 = min(roots) if roots else (e1 if abs(f2) <= 1e-15 else np.inf)
    val = lambda e: F(HD.var(p + e * u, [])).c[0]
    es = np.linspace(0, emax, 2001)[1:]; ex = np.inf; prev = 0.0
    for e in es:
        if val(e) < 0: ex = brentq(val, prev, e, xtol=1e-15); break
        prev = e
    return dict(F=f0, dF=f1, d2F=f2, eps1=e1, eps2=e2, exact=ex)
