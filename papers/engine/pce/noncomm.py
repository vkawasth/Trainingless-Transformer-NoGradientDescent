"""Non-commutativity as structure: matrix-valued (quantum) probability on top of the gluing engine.

States are density matrices rho (Hermitian, PSD, trace 1). An observable is a projective measurement, a list of orthogonal projections
summing to the identity. Spatial arrangement enters through rotations: the spin-j component along a unit vector n.

Order and projections
  spin_ops(j)                      (Jx, Jy, Jz) for spin j (dimension 2j + 1)
  spectral(O)                      projective measurement of a Hermitian observable (eigenspaces)
  axis_measurement(j, n)           projections onto the eigenspaces of n . J (outcomes m = j, ..., -j)
  rotation(j, axis, angle)         exp(-i angle axis . J)
  sequential(rho, A, B)            P(a then b) = tr(B_b A_a rho A_a B_b)  (Lueders rule), as a table [a, b]
  order_effect(rho, P, Q)          P(P then Q) - P(Q then P) for two projections (both 'yes')
  halmos_angles(P, Q)              principal angles between the ranges (two-subspace theory)
  comm_norm(P, Q)                  operator norm of [P, Q]
The cover
  compatibility_cover(obs)         maximal sets of pairwise commuting observables (the contexts)
  context_tables(rho, obs, cover)  the joint table of each context (they agree on overlaps: no signalling)
The canonical section on the complex rung
  kirkwood_dirac(rho, A, B)        q(a, b) = tr(B_b A_a rho): complex, with marginals tr(A_a rho) and tr(B_b rho)
  kd_rung(q)                       'R>=0', 'R' (real, some negative) or 'C' (non-real), and the measures: negativity sum|q| - 1, imaginary mass
Loops
  bargmann(psis)                   tr(P1 P2 ... Pn) for pure states: its argument is the geometric (Pancharatnam) phase of the loop
  lueders_loop(rho, projs)         the (unnormalised) state after projecting around the loop, and its survival probability
  zeno_drag(j, n0, n1, N, m)       steer |j, m> along n0 to n1 by N intermediate projections: probability of staying on the path
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.linalg import expm


def spin_ops(j):
    d = int(round(2 * j + 1)); m = j - np.arange(d)
    Jz = np.diag(m).astype(complex)
    Jp = np.zeros((d, d), complex)
    for k in range(1, d): Jp[k - 1, k] = np.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    Jx = (Jp + Jp.conj().T) / 2; Jy = (Jp - Jp.conj().T) / (2j)
    return Jx, Jy, Jz


def axis_measurement(j, n):
    n = np.asarray(n, float); n = n / np.linalg.norm(n); Jx, Jy, Jz = spin_ops(j)
    H = n[0] * Jx + n[1] * Jy + n[2] * Jz; vals, vecs = np.linalg.eigh(H); order = np.argsort(-vals)
    return [np.outer(vecs[:, k], vecs[:, k].conj()) for k in order]


def spectral(O, tol=1e-9):
    """projective measurement of a Hermitian observable: one projection per distinct eigenvalue (largest first)."""
    v, U = np.linalg.eigh(O); out = []; vals = []
    for k in np.argsort(-v):
        if vals and abs(v[k] - vals[-1]) < tol: out[-1] = out[-1] + np.outer(U[:, k], U[:, k].conj())
        else: vals.append(v[k]); out.append(np.outer(U[:, k], U[:, k].conj()))
    return out


def rotation(j, axis, angle):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a); Jx, Jy, Jz = spin_ops(j)
    return expm(-1j * angle * (a[0] * Jx + a[1] * Jy + a[2] * Jz))


def sequential(rho, A, B):
    return np.array([[float(np.real(np.trace(Bb @ Aa @ rho @ Aa @ Bb))) for Bb in B] for Aa in A])


def order_effect(rho, P, Q):
    return float(np.real(np.trace(Q @ P @ rho @ P @ Q) - np.trace(P @ Q @ rho @ Q @ P)))


def comm_norm(X, Y):
    return float(np.linalg.norm(X @ Y - Y @ X, 2))


def halmos_angles(P, Q, tol=1e-9):
    def basis(R):
        v, U = np.linalg.eigh(R); return U[:, v > 0.5]
    s = np.linalg.svd(basis(P).conj().T @ basis(Q), compute_uv=False)
    return np.arccos(np.clip(s, -1, 1))


def _commute(A, B, tol=1e-9):
    return all(np.linalg.norm(a @ b - b @ a) < tol for a in A for b in B)


def compatibility_cover(obs, tol=1e-9):
    """maximal cliques of the graph 'observables commute' (Bron-Kerbosch on a small graph)."""
    n = len(obs); adj = {i: {k for k in range(n) if k != i and _commute(obs[i], obs[k], tol)} for i in range(n)}
    out = []
    def bk(R, P, X):
        if not P and not X: out.append(tuple(sorted(R))); return
        for v in list(P):
            bk(R | {v}, P & adj[v], X & adj[v]); P = P - {v}; X = X | {v}
    bk(set(), set(range(n)), set())
    return sorted(out)


def context_tables(rho, obs, cover):
    tabs = []
    for C in cover:
        T = np.zeros(tuple(len(obs[i]) for i in C))
        for idx in itertools.product(*[range(len(obs[i])) for i in C]):
            M = np.eye(rho.shape[0], dtype=complex)
            for i, k in zip(C, idx): M = M @ obs[i][k]
            T[idx] = float(np.real(np.trace(M @ rho)))
        tabs.append(T)
    return tabs


def kirkwood_dirac(rho, A, B):
    return np.array([[np.trace(Bb @ Aa @ rho) for Bb in B] for Aa in A])


def kd_rung(q, tol=1e-10):
    imag = float(np.abs(q.imag).sum()); neg = float(np.abs(q).sum() - 1)
    rung = "C" if imag > tol else ("R" if q.real.min() < -tol else "R>=0")
    return {"rung": rung, "nonpositivity": neg, "imag_mass": imag, "neg_real": float(-q.real[q.real < 0].sum())}


def bargmann(psis):
    v = 1.0 + 0j
    for a, b in zip(psis, psis[1:] + psis[:1]): v *= np.vdot(a, b)
    return v


def lueders_loop(rho, projs):
    s = rho.copy()
    for P in projs: s = P @ s @ P
    return s, float(np.real(np.trace(s)))


def zeno_drag(j, n0, n1, N, m_index=0):
    """projective measurements of n_k . J along the great circle from n0 to n1 (k = 0..N); probability of obtaining the tracked
    outcome m at every step, starting from that eigenstate of n0 . J."""
    n0 = np.asarray(n0, float) / np.linalg.norm(n0); n1 = np.asarray(n1, float) / np.linalg.norm(n1)
    Th = np.arccos(np.clip(n0 @ n1, -1, 1)); ax = np.cross(n0, n1); ax = ax / np.linalg.norm(ax)
    path = [np.cos(t) * n0 + np.sin(t) * np.cross(ax, n0) for t in np.linspace(0, Th, N + 1)]
    P0 = axis_measurement(j, path[0])[m_index]; rho = P0 / np.trace(P0).real; p = 1.0
    for n in path[1:]:
        P = axis_measurement(j, n)[m_index]; pk = float(np.real(np.trace(P @ rho))); p *= pk; rho = P @ rho @ P / pk
    return p, Th
