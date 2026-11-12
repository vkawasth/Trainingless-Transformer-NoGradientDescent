"""When refitting and coarsening commute for every starting law. New module; it modifies nothing.

Everything is expressed through spans. On a finite fine space with n states, the data are a subspace H of functions (constants plus the
statistics whose expectations the data fix) and the values beta = B^T P of a basis B at a positive law P. A coarsening is a map
g: {0..n-1} -> {0..m-1}; the coarse data are a subspace H' of coarse functions whose pullback lies in H, with values read from g_* P.

  span_labels(labels_list, n)   basis of constants + indicators of the cells of each context (a context = an integer label per state)
  orth(B)                       orthonormal basis of the column span
  intersect(B1, B2)             basis of span(B1) intersected with span(B2)
  residual(f, B)                distance of f from span(B)
  pull(g, Bc)                   pull back coarse functions to the fine space
  push(g, P, m)                 pushforward of a law
  iproj(Q, B, beta)             I-projection of Q onto {P : B^T P = beta} (Newton on the convex dual)
  criterion(Hf, Hc, g, m)       the section criterion: every fine statistic read along a section of g is a coarse statistic; returned as
                                (holds, classes where some statistic varies, worst residual)
  criterion_sections(...)       the same, checked over every section (exponential; small examples only)
  defect(Q, Hf, Hc, g, m, P)    TV(g_* Pi_J(Q), Pi_J'(g_* Q))
"""
from __future__ import annotations
import itertools
import numpy as np


def span_labels(labels_list, n):
    cols = [np.ones(n)]
    for lab in labels_list:
        lab = np.asarray(lab)
        for v in np.unique(lab): cols.append((lab == v).astype(float))
    return np.array(cols).T


def orth(B, tol=1e-10):
    U, s, _ = np.linalg.svd(B, full_matrices=False); return U[:, s > tol * max(1.0, s.max())]


def intersect(B1, B2, tol=1e-10):
    A = np.column_stack([orth(B1), -orth(B2)]); _, s, Vt = np.linalg.svd(A)
    r = int((s > tol).sum()); N = Vt[r:].T
    if N.shape[1] == 0: return np.zeros((B1.shape[0], 0))
    return orth(orth(B1) @ N[:orth(B1).shape[1]])


def residual(f, B):
    if B.shape[1] == 0: return float(np.linalg.norm(f))
    x = np.linalg.lstsq(B, f, rcond=None)[0]; return float(np.linalg.norm(B @ x - f))


def pull(g, Bc): return Bc[np.asarray(g)]


def push(g, P, m): out = np.zeros(m); np.add.at(out, np.asarray(g), P); return out


def iproj(Q, B, beta, tol=1e-13, iters=200):
    """minimise D(R || Q) subject to B^T R = beta; R = Q exp(C theta)/Z with C an orthonormal basis of span(B) minus constants"""
    n = len(Q); one = np.ones(n) / np.sqrt(n)
    Bc = B - np.outer(one, one @ B); C = orth(Bc)
    if C.shape[1] == 0: return Q / Q.sum()
    # targets for C: C^T R, read from B^T R = beta (and sum R = 1)
    X = np.linalg.lstsq(np.column_stack([B, np.ones(n)]), C, rcond=None)[0]          # C = [B 1] X
    target = X[:-1].T @ beta + X[-1] * 1.0
    th = np.zeros(C.shape[1]); logQ = np.log(Q)
    def law(t):
        z = logQ + C @ t; z = z - z.max(); R = np.exp(z); return R / R.sum()
    def dual(t):
        z = logQ + C @ t; mx = z.max(); return mx + np.log(np.exp(z - mx).sum()) - t @ target
    for _ in range(iters):
        R = law(th); grad = C.T @ R - target
        if np.abs(grad).max() < tol: break
        Hs = C.T @ (R[:, None] * C) - np.outer(C.T @ R, C.T @ R)
        step = np.linalg.lstsq(Hs, grad, rcond=None)[0]
        if np.abs(grad).max() < 1e-7 and np.abs(C.T @ law(th - step) - target).max() < np.abs(grad).max():   # near the optimum: full Newton step (dual values
            th = th - step; continue                                            # differ below rounding there, so no line search)
        a = 1.0; f0 = dual(th)
        while dual(th - a * step) > f0 - 1e-4 * a * grad @ step and a > 1e-12: a /= 2
        th = th - a * step
    R = law(th)
    if np.abs(C.T @ R - target).max() > 1e-10: raise RuntimeError("I-projection did not converge")
    return R


def criterion(Hf, Hc, g, m, tol=1e-9):
    g = np.asarray(g); Bf = orth(Hf); varies = set(); worst = 0.0
    classes = [np.where(g == c)[0] for c in range(m)]
    sigma0 = np.array([cl[0] for cl in classes])
    for s in Bf.T:
        for c, cl in enumerate(classes):
            if np.ptp(s[cl]) > tol: varies.add(c)
        worst = max(worst, residual(s[sigma0], Hc))
    for c in varies:
        e = np.zeros(m); e[c] = 1.0; worst = max(worst, residual(e, Hc))
    return worst < 1e-7, sorted(varies), worst


def criterion_sections(Hf, Hc, g, m):
    g = np.asarray(g); Bf = orth(Hf); classes = [np.where(g == c)[0] for c in range(m)]; worst = 0.0
    for sigma in itertools.product(*classes):
        sigma = np.array(sigma)
        for s in Bf.T: worst = max(worst, residual(s[sigma], Hc))
    return worst < 1e-7, worst


def defect(Q, Hf, Hc, g, m, P):
    R = iproj(Q, Hf, Hf.T @ P); Pc = push(g, P, m); Rp = iproj(push(g, Q, m), Hc, Hc.T @ Pc)
    return 0.5 * float(np.abs(push(g, R, m) - Rp).sum())
