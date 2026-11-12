"""The selection theorem, uniform version (Theorem thm:selection): the lexicographic max-min point with its ordered levels, the explicit
constant of the uniform O(1/L) bound, the linear map on each chamber, and the first-order coefficient of the soft-min point.
New module; it modifies nothing (uses pce/selectorlink.py for a first full Newton step).

Notation as in the paper: l_k(u) = <v_k, u> - lam_k with v_1..v_m positively spanning R^n; u_L minimises sum_k a_k exp(-L l_k(u)).
  lex_levels(V, lam)          (u_lex, levels) with levels = [(t_1, I_1), (t_2, I_2), ...], t_1 < t_2 < ..., the ordered level partition
  softmin(V, lam, L, a=None)  u_L, by block coordinate descent over the level subspaces (exact to rounding at any L)
  chamber_map(V, parts)       the n x m matrix A_P with u_lex = A_P lam on the chamber of the ordered partition P (unique solution of
                              l_k(u) = t_i for k in I_i, in the unknowns u and t)
  coefficient(V, parts, b)    the limit of L (u_L - u_lex) for weights a = exp(b) inside a chamber: level by level, the minimiser of the
                              tied terms sum_{k in I_i} exp(b''_k - <v'_k, xi>) over the span of the projected normals
  K_bound(V, B)               the constant K(V, B) of the uniform bound |u_L - u_lex| <= K/L for weights with (1/2) log(C/c) <= B
  positive_relation(W)        a strictly positive q with sum q_k w_k = 0, maximising min q (None if none exists)
"""
from __future__ import annotations
import itertools, math
import numpy as np
from scipy.optimize import linprog
from . import selectorlink as SLK


def _lex(V, lam, band):
    m, n = V.shape; fixed = {}; free = set(range(m)); levels = []
    while free:
        Aub, bub = [], []
        for k in free: Aub.append(np.r_[-V[k], 1.0]); bub.append(-lam[k])
        for k, h in fixed.items():
            Aub.append(np.r_[-V[k], 0.0]); bub.append(-lam[k] - h + band); Aub.append(np.r_[V[k], 0.0]); bub.append(lam[k] + h + band)
        r = linprog(np.r_[np.zeros(n), -1.0], A_ub=np.array(Aub), b_ub=np.array(bub), bounds=[(None, None)] * (n + 1), method="highs")
        t = -r.fun; S = []
        for k in sorted(free):
            A2 = [-V[j] for j in free]; b2 = [-lam[j] - t + band for j in free]
            for j, h in fixed.items(): A2 += [-V[j], V[j]]; b2 += [-lam[j] - h + band, lam[j] + h + band]
            r2 = linprog(-V[k], A_ub=np.array(A2), b_ub=np.array(b2), bounds=[(None, None)] * n, method="highs")
            if r2.status == 0 and (-r2.fun - lam[k]) <= t + 1e-7: S.append(k)
        if not S: S = sorted(free)
        for k in S: fixed[k] = t; free.discard(k)
        levels.append((t, tuple(S)))
    K = sorted(fixed); u = np.linalg.lstsq(V[K], lam[K] + np.array([fixed[k] for k in K]), rcond=None)[0]
    return u, levels


def _lex_dual(V, lam, eps):
    """level by level, in coordinates on the affine set where the fixed functions take their levels (u = u0 + N z): the level LP gives
    t_j; the set fixed at level j is the union of supports of the optimal dual laws (strict complementarity), found by one homogenised LP
    that maximises sum_k min(q_k, 1) over the optimal dual face {q >= 0 : sum q_k N^T v_k = 0, sum q_k (l_k(u0) - t_j) = 0}"""
    m, n = V.shape; fixed = {}; free = list(range(m)); levels = []; u0 = np.zeros(n); N = np.eye(n)
    while free:
        nf = len(free); l0 = V[free] @ u0 - lam[free]; W = V[free] @ N; d = N.shape[1]
        if d == 0:
            t = l0.min(); S = [free[i] for i in range(nf) if l0[i] <= t + eps * max(1, np.abs(lam).max())]
        else:
            r = linprog(np.r_[np.zeros(d), -1.0], A_ub=np.hstack([-W, np.ones((nf, 1))]), b_ub=l0, bounds=[(None, None)] * (d + 1), method="highs")
            if r.status != 0: raise ValueError("level LP")
            t = -r.fun; gap = np.maximum(l0 + W @ r.x[:d] - t, 0.0)               # slacks at the optimum, >= 0
            nv = 2 * nf; c = np.r_[np.zeros(nf), -np.ones(nf)]
            A_ub = [np.r_[gap - eps, np.zeros(nf)]]; b_ub = [0.0]                   # sum q_k gap_k <= eps sum q_k, with sum q = nf
            for i in range(nf):
                row = np.zeros(nv); row[nf + i] = 1.0; row[i] = -1.0; A_ub.append(row); b_ub.append(0.0)
            Ae = np.vstack([np.hstack([W.T, np.zeros((d, nf))]), np.r_[np.ones(nf), np.zeros(nf)][None]])
            r2 = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=Ae, b_eq=np.r_[np.zeros(d), nf], bounds=[(0, None)] * nf + [(0, 1)] * nf, method="highs")
            if r2.status != 0: raise ValueError("support LP")
            S = [free[i] for i in range(nf) if r2.x[nf + i] > 1e-7]
        if not S: raise ValueError("empty level")
        for k in S: fixed[k] = t
        free = [k for k in free if k not in S]; levels.append((t, tuple(S)))
        K = sorted(fixed); u0 = np.linalg.lstsq(V[K], lam[K] + np.array([fixed[k] for k in K]), rcond=None)[0]
        sv = np.linalg.svd(V[K]); rk = int((sv[1] > 1e-10 * max(1, sv[1].max())).sum()); N = sv[2][rk:].T
    return u0, levels


def lex_levels(V, lam):
    V = np.asarray(V, float); lam = np.asarray(lam, float)
    tries = [(_lex_dual, e) for e in (1e-12, 1e-14, 1e-10)] + [(_lex, b) for b in (1e-10, 1e-8, 1e-6)]
    for fn, tol_ in tries:
        try: u, lv = fn(V, lam, tol_)
        except (TypeError, ValueError): continue
        merged = [list(lv[0][1])]
        for (t0, _), (t1, I1) in zip(lv, lv[1:]):
            if t1 - t0 < 1e-6 * max(1, np.abs(lam).max()): merged[-1] += list(I1)    # levels that meet at a wall
            else: merged.append(list(I1))
        vals = V @ u - lam; order = np.argsort(vals); cands = [[I for _, I in lv], [tuple(sorted(I)) for I in merged]]
        for tau in (1e-10, 1e-8, 1e-6):                                         # levels read off the values at u, clustered at tau
            cl = [[int(order[0])]]
            for a_, b_ in zip(order, order[1:]):
                if vals[b_] - vals[a_] < tau * max(1, np.abs(lam).max()): cl[-1].append(int(b_))
                else: cl.append([int(b_)])
            cands.append([tuple(sorted(c)) for c in cl])
        for parts in cands:
            m, n = V.shape; E = np.zeros((m, len(parts)))
            for i, I in enumerate(parts):
                for k in I: E[k, i] = 1.0
            x = np.linalg.lstsq(np.hstack([V, -E]), lam, rcond=None)[0]       # polish: the unique solution of l_k(u) = t_i on I_i
            if valid_levels(V, lam, x[:n], parts, x[n:]): return x[:n], [(x[n + i], tuple(I)) for i, I in enumerate(parts)]
    raise RuntimeError("lexicographic max-min: no tolerance gave a certified level partition")


def valid_levels(V, lam, u, parts, t, tol=1e-7):
    """certificate of the lexicographic point (Lemma lem:lexunique): the levels are strictly increasing, l_k(u) = t_i on I_i, and each I_i
    carries a strictly positive relation modulo the span of the earlier levels"""
    if np.any(np.diff(t) <= -tol * max(1, np.abs(lam).max())): return False    # levels within rounding of each other: at a wall
    F = []
    for i, I in enumerate(parts):
        if np.abs(V[list(I)] @ u - lam[list(I)] - t[i]).max() > tol * max(1, np.abs(lam).max()): return False
        if positive_relation(V[list(I)] @ _proj_out(V, F)) is None: return False
        F = F + list(I)
    return True




def softmin(V, lam, L, a=None, parts=None, sweeps=200, tol=1e-15):
    """u_L by block coordinate descent over the level subspaces of the lexicographic point. Within the block of level i only the terms of
    levels >= i vary, so each block is solved on its own scale (log-sum-exp over those terms); a plain Newton method on F_L loses the upper
    levels below rounding once L times the gap exceeds about 37."""
    V = np.asarray(V, float); lam = np.asarray(lam, float); m, n = V.shape
    b = np.zeros(m) if a is None else np.log(np.asarray(a, float))
    u, lv = lex_levels(V, lam + b / L)
    u = SLK.softmin(V, lam + b / L, L, u0=u)                                   # full Newton first: exact when L times the gaps is small
    parts = [I for _, I in lv] if parts is None else parts
    blocks = []; F = []
    for I in parts:
        P = _proj_out(V, F); W = V[list(I)] @ P; sv = np.linalg.svd(W)
        rk = int((sv[1] > 1e-10 * max(1, sv[1].max())).sum())
        if rk: blocks.append((sv[2][:rk].T, [k for k in range(m) if k not in F]))
        F = F + list(I)
    for _ in range(sweeps):
        moved = 0.0
        for Bm, act in blocks:
            Va = V[act] @ Bm
            def f(z):
                e = -L * (V[act] @ (u + Bm @ z) - lam[act]) + b[act]; mx = e.max(); return mx + np.log(np.exp(e - mx).sum())
            z = np.zeros(Bm.shape[1])
            for _ in range(40):
                e = -L * (V[act] @ (u + Bm @ z) - lam[act]) + b[act]; w = np.exp(e - e.max()); w /= w.sum()
                g = -L * (w @ Va); mu = w @ Va; H = L * L * ((Va - mu).T @ ((Va - mu) * w[:, None]))
                d = -np.linalg.lstsq(H, g, rcond=None)[0]
                if np.abs(d).max() < tol: break
                s_, f0 = 1.0, f(z)
                while f(z + s_ * d) > f0 + 1e-4 * s_ * (g @ d) and s_ > 1e-12: s_ /= 2
                if s_ <= 1e-12: break
                z = z + s_ * d
            u = u + Bm @ z; moved = max(moved, np.abs(Bm @ z).max())
        if moved < 10 * tol: break
    return u


def chamber_map(V, parts):
    V = np.asarray(V, float); m, n = V.shape; r = len(parts); E = np.zeros((m, r))
    for i, I in enumerate(parts):
        for k in I: E[k, i] = 1.0
    M = np.hstack([V, -E])                       # V u - E t = lam
    return np.linalg.pinv(M)[:n]


def _proj_out(V, F):
    """orthogonal projector onto the complement of span{v_k : k in F}"""
    n = V.shape[1]
    if not F: return np.eye(n)
    Q = np.linalg.qr(V[list(F)].T)[0]; s = np.linalg.svd(V[list(F)].T, compute_uv=False); rk = int((s > 1e-10 * max(1, s.max())).sum())
    U = np.linalg.svd(V[list(F)].T)[0][:, :rk]
    return np.eye(n) - U @ U.T


def positive_relation(W):
    W = np.asarray(W, float); k = W.shape[0]
    r = linprog(np.r_[np.zeros(k), -1.0], A_ub=np.hstack([-np.eye(k), np.ones((k, 1))]), b_ub=np.zeros(k),
                A_eq=np.vstack([np.hstack([W.T, np.zeros((W.shape[1], 1))]), np.r_[np.ones(k), 0.0][None]]), b_eq=np.r_[np.zeros(W.shape[1]), 1.0],
                bounds=[(None, None)] * (k + 1), method="highs")
    if r.status != 0 or r.x[-1] <= 1e-9: return None
    return r.x[:k]


def _tied_min(W, b):
    """minimiser over span(W rows) of sum_k exp(b_k - <w_k, xi>) (the rows positively span their span)"""
    s = np.linalg.svd(W, full_matrices=False); rk = int((s[1] > 1e-10 * max(1, s[1].max())).sum()); B = s[2][:rk].T   # basis of row span
    Wc = W @ B; z = np.zeros(rk)
    for _ in range(200):
        e = np.exp(b - Wc @ z); g = -(e @ Wc); H = Wc.T @ (Wc * e[:, None])
        d = -np.linalg.solve(H, g); z = z + d
        if np.abs(d).max() < 1e-15: break
    return B @ z


def coefficient(V, parts, b):
    V = np.asarray(V, float); b = np.asarray(b, float); c = np.zeros(V.shape[1]); F = []
    for I in parts:
        P = _proj_out(V, F); W = V[list(I)] @ P
        xi = _tied_min(W, b[list(I)] - V[list(I)] @ c) if np.abs(W).max() > 1e-12 else np.zeros(V.shape[1])
        c = c + xi; F = F + list(I)
    return c


def _level_term(W, Btot, mcur):
    q = positive_relation(W)
    if q is None: return None
    s = np.linalg.svd(W, compute_uv=False); sig = s[s > 1e-10 * max(1, s.max())].min()
    kappa = 2 * Btot + math.log(mcur); qm = q.min(); rho = kappa * max(1.0, (1 - qm) / qm)
    return math.sqrt(len(W)) * rho / sig


def K_bound(V, B, F=(), _memo=None):
    """sup over lam and over centred log-weights |b| <= B of |G - u_lex| (in L-scaled coordinates): recursion of the proof, maximised over
    every candidate next level I (a subset of the free indices whose projected normals have a strictly positive relation)"""
    V = np.asarray(V, float); m = V.shape[0]; free = [k for k in range(m) if k not in F]
    if not free: return 0.0
    P = _proj_out(V, list(F)); best = 0.0
    for r in range(1, len(free) + 1):
        for I in itertools.combinations(free, r):
            W = V[list(I)] @ P
            if np.linalg.norm(W) < 1e-12: continue
            R = _level_term(W, B, len(free))
            if R is None: continue
            rest = [k for k in free if k not in I]
            B2 = B + (max(np.linalg.norm(V[k] @ P) for k in rest) * R if rest else 0.0)
            best = max(best, math.hypot(R, K_bound(V, B2, tuple(F) + I)))
    return best
