"""The second-order birth invariant at a classical point of a cycle's representation scheme.

Hermitian models of the n-cycle (+-1 observables X_0..X_{k-1}, X_i X_{i+1} = X_{i+1} X_i) in dimension n, coordinates z = upper triangles of
real symmetric matrices. The equations F(z) = 0 are quadratic, so F(z0 + tY) = t J Y + t^2 F2(Y) exactly. A tangent vector Y (J Y = 0)
extends to second order iff F2(Y) is in the image of J; its lift Y2 solves J Y2 = -F2(Y), and X(t) = X0 + t Y + t^2 Y2 + O(t^3) is a curve in
the scheme when the point is smooth (and along the extendable cone in general).

Contextuality witness: w = max over odd sign patterns g of lambda_max(B_g) - (k - 2), B_g = sum_i g_i X_i X_{i+1}. At a classical point X0
(diagonal, a direct sum of characters) B_g is diagonal, w <= 0, and the top eigenspace P is spanned by the characters saturating g. Along the
curve B(t) = B0 + t B1 + t^2 B2, and by degenerate perturbation theory the top eigenvalue near k - 2 is
    lambda(t) = (k - 2) + t lambda_1 + t^2 lambda_2 + O(t^3),
lambda_1 = top eigenvalue of P B1 P, and (where lambda_1 = 0 on the relevant subspace) lambda_2 = top eigenvalue of
    M(Y) = P B2 P + P B1 (lambda0 - B0)^+ B1 P.
The invariant is Lambda_2 = max over odd g and unit extendable Y of lambda_2. Lambda_2 > 0 (or lambda_1 > 0) means contextual models
arbitrarily close, with the witness growing like Lambda_2 eps^2.

  setup(k, n)                         the cycle's equations in symmetric coordinates
  first_order(sys, X0)                max over g and unit tangent Y of the top eigenvalue of P B1 P (expected 0 at classical points)
  invariant(sys, X0, samples, rng)    (Lambda_2, smooth?, tangent dim, best g); samples tangent directions (the extendable cone at singular
                                      points, by Gauss-Newton on the cokernel equations), then refines the best ones
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import least_squares, minimize


class CycleSystem:
    def __init__(self, k, n):
        self.k, self.n = k, n; self.iu = np.triu_indices(n); self.L = len(self.iu[0]); self.N = k * self.L
        self.E = [(i, (i + 1) % k) for i in range(k)]
        self.G = [g for g in itertools.product((1, -1), repeat=k) if g.count(-1) % 2 == 1]

    def unpack(self, z):
        out = []
        for v in range(self.k):
            M = np.zeros((self.n, self.n)); M[self.iu] = z[v * self.L:(v + 1) * self.L]; out.append(M + M.T - np.diag(np.diag(M)))
        return out

    def pack(self, X): return np.concatenate([x[self.iu] for x in X])

    def F(self, z):
        X = self.unpack(z); I = np.eye(self.n)
        r = [X[v] @ X[v] - I for v in range(self.k)] + [X[u] @ X[w] - X[w] @ X[u] for u, w in self.E]
        return np.concatenate([q.ravel() for q in r])

    def lin(self, z0):
        F0 = self.F(z0); J = np.zeros((len(F0), self.N))
        for j in range(self.N):
            e = np.zeros(self.N); e[j] = 1.0; J[:, j] = (self.F(z0 + e) - self.F(z0 - e)) / 2      # exact: F is quadratic
        return J

    def F2(self, z0, y): return (self.F(z0 + y) + self.F(z0 - y)) / 2 - self.F(z0)

    def Bterms(self, X0, Y, Y2, g):
        k = self.k; s = lambda M: (M + M.T) / 2
        B0 = s(sum(g[i] * X0[i] @ X0[(i + 1) % k] for i in range(k)))
        B1 = s(sum(g[i] * (Y[i] @ X0[(i + 1) % k] + X0[i] @ Y[(i + 1) % k]) for i in range(k)))
        B2 = s(sum(g[i] * (Y2[i] @ X0[(i + 1) % k] + X0[i] @ Y2[(i + 1) % k] + Y[i] @ Y[(i + 1) % k]) for i in range(k)))
        return B0, B1, B2


def setup(k, n): return CycleSystem(k, n)


def _null(M, tol=1e-9):
    u, s, vt = np.linalg.svd(M); r = int(np.sum(s > tol * max(1.0, s.max() if len(s) else 1.0))); return vt[r:].T, u[:, r:], r


def _lam2(sys, X0, z0, J, y, g):
    y2, *_ = np.linalg.lstsq(J, -sys.F2(z0, y), rcond=None)
    B0, B1, B2 = sys.Bterms(X0, sys.unpack(y), sys.unpack(y2), g)
    d = np.diag(B0); lam0 = sys.k - 2; top = np.where(np.abs(d - lam0) < 1e-9)[0]
    if len(top) == 0: return -np.inf, 0.0
    P = np.zeros((sys.n, len(top))); P[top, np.arange(len(top))] = 1.0
    l1 = np.linalg.eigvalsh(P.T @ B1 @ P).max()
    gap = lam0 - d; Rinv = np.diag([1.0 / x if x > 1e-9 else 0.0 for x in gap])
    M = P.T @ (B2 + B1 @ Rinv @ B1) @ P
    return float(np.linalg.eigvalsh((M + M.T) / 2).max()), float(l1)


def first_order(sys, X0):
    z0 = sys.pack(X0); J = sys.lin(z0); K, _, _ = _null(J); best = 0.0
    for g in sys.G:
        d = np.diag(sys.Bterms(X0, X0, X0, g)[0]); top = np.where(np.abs(d - (sys.k - 2)) < 1e-9)[0]
        if len(top) == 0: continue
        P = np.zeros((sys.n, len(top))); P[top, np.arange(len(top))] = 1.0
        for j in range(K.shape[1]):
            B1 = sys.Bterms(X0, sys.unpack(K[:, j]), [np.zeros((sys.n, sys.n))] * sys.k, g)[1]
            best = max(best, np.abs(P.T @ B1 @ P).max())
    return best


def invariant(sys, X0, samples=300, rng=None, refine=6):
    rng = rng or np.random.default_rng(0); z0 = sys.pack(X0); J = sys.lin(z0); K, C, r = _null(J)
    t = K.shape[1]
    if t == 0: return {"Lambda2": -np.inf, "smooth": True, "tangent": 0, "g": None, "lambda1": 0.0}
    coker = lambda c: C.T @ sys.F2(z0, K @ c)                        # second-order extendability: F2(Y) in image of J
    smooth = True
    for _ in range(4):
        c = rng.normal(size=t); smooth &= np.linalg.norm(coker(c / np.linalg.norm(c))) < 1e-9
    def cone_point():
        c = rng.normal(size=t); c /= np.linalg.norm(c)
        if smooth or C.shape[1] == 0: return c
        r_ = least_squares(lambda cc: np.r_[coker(cc), (cc @ cc - 1) * 10], c, xtol=1e-15, ftol=1e-15, gtol=1e-15)
        cc = r_.x / np.linalg.norm(r_.x)
        return cc if np.linalg.norm(coker(cc)) < 1e-9 else None
    best = (-np.inf, None, None, 0.0); cands = []
    for _ in range(samples):
        c = cone_point()
        if c is None: continue
        for g in sys.G:
            v, l1 = _lam2(sys, X0, z0, J, K @ c, g)
            if v > -np.inf: cands.append((v, tuple(g), c))
            if v > best[0]: best = (v, g, c, l1)
    cands.sort(key=lambda x: -x[0])
    for v, g, c in cands[:refine]:
        def obj(cc):
            cc = cc / np.linalg.norm(cc)
            pen = 0.0 if smooth else 1e3 * np.linalg.norm(coker(cc)) ** 2
            return -_lam2(sys, X0, z0, J, K @ cc, g)[0] + pen
        res = minimize(obj, c, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-12})
        cc = res.x / np.linalg.norm(res.x)
        if smooth or np.linalg.norm(coker(cc)) < 1e-7:
            v2, l1 = _lam2(sys, X0, z0, J, K @ cc, g)
            if v2 > best[0]: best = (v2, g, cc, l1)
    return {"Lambda2": float(best[0]), "smooth": bool(smooth), "tangent": int(t), "g": best[1], "lambda1": float(best[3])}
