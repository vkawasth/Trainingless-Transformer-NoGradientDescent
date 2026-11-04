"""Section 12 of the companion note: objects, morphisms and the categorical comparison across an outcome wall. New module; imports
kahlerbridge, datawalls, couplingfibre, and modifies none.

Everything here is computed on the B-side data of the potential W_T = sum_k T^(-lam_k) y^(v_k): its critical points, critical values and
Hessians. The A-side reading (torus fibres with local systems, Floer rings, Fukaya summands) is cited (Cho-Oh 2006, Cho 2005 Thm 5.6,
Smith 2018), never computed.

  f1_critical(alpha, T, dps)           the four critical points of F_1 from the quartic w^4 - T^(1/2) w - T^(1 - alpha) (y1 = T^(1/2)/w, y2 = w^2)
  objects(v, lam, T, pts)              for each critical point: valuation, type (positive / real-signed / complex), leading holonomy y / T^u,
                                       critical value W, Hessian Q = sum_k c_k v_k v_k^T (c_k the term values) and Q / W
  clifford_generators(Q)               2 x 2 complex matrices e_1, e_2 with e_i e_j + e_j e_i = Q_ij (Cho's normalisation), from a symmetric
                                       square root of Q; returns them and the dimension of the algebra they generate
  reduced_rank(Qn, T, tol_power)       the rank of Q / W modulo T: count eigenvalues whose valuation (log|ev| / log T) is ~ 0
  stokes_events(tracks, al, theta)     parameters where Im(e^(-i theta)(W_j - W_i)) changes sign along a path, with the pairs involved
  track(path, solver)                  critical points along a path, matched continuously (linear assignment on relative distance)
"""
from __future__ import annotations
import numpy as np
import mpmath as mp
from scipy.optimize import linear_sum_assignment


def f1_critical(alpha, T, dps=120):
    mp.mp.dps = dps; T = mp.mpf(T); a = mp.mpf(alpha); out = []
    for w in mp.polyroots([1, 0, 0, -mp.sqrt(T), -T ** (1 - a)], maxsteps=800, extraprec=4 * dps):
        out.append((mp.sqrt(T) / w, w * w))
    return out


def objects(v, lam, T, pts, tol=None):
    T = mp.mpf(T); V = [tuple(int(x) for x in r) for r in v]; out = []
    tol = tol if tol is not None else mp.mpf(10) ** (-(mp.mp.dps // 3))
    for y in pts:
        c = [T ** (-mp.mpf(l)) * mp.fprod(yi ** vi for yi, vi in zip(y, vk)) for vk, l in zip(V, lam)]; W = mp.fsum(c)
        n = len(y); Q = [[mp.fsum(ck * vk[i] * vk[j] for ck, vk in zip(c, V)) for j in range(n)] for i in range(n)]
        u = [float(mp.log(abs(yi)) / mp.log(T)) for yi in y]
        real = all(abs(mp.im(yi)) <= tol * abs(yi) for yi in y)
        typ = "positive" if real and all(mp.re(yi) > 0 for yi in y) else ("real-signed" if real else "complex")
        out.append(dict(y=y, u=np.array(u), type=typ, W=W, Q=Q, Qn=np.array([[complex(q / W) for q in row] for row in Q]),
                        hol=np.array([complex(yi / T ** mp.mpf(ui)) for yi, ui in zip(y, u)])))
    return out


def clifford_generators(Q):
    """e_i = sum_j A_ji g_j with Pauli-type g_1, g_2 (g_j g_k + g_k g_j = 2 delta) and A^T A = Q / 2, so e_i e_j + e_j e_i = Q_ij"""
    import scipy.linalg as sl
    Q = np.array(Q, complex); A = sl.sqrtm(Q / 2)                      # symmetric square root: A^T = A, A A = Q / 2
    g = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex)]
    e = [sum(A[j, i] * g[j] for j in range(2)) for i in range(2)]
    words = [np.eye(2), e[0], e[1], e[0] @ e[1]]; dim = np.linalg.matrix_rank(np.array([w.ravel() for w in words]), tol=1e-9 * max(1, np.abs(Q).max()))
    return e, dim


def reduced_rank(Qn_by_T, Ts):
    """Qn_by_T: the normalised Hessians Q/W at a decreasing sequence of T; an eigenvalue has valuation ~ slope of log|ev| against log T.
    Returns the valuations (sorted) and the number of them that are ~ 0 (the rank of the reduction mod T)."""
    ev = [np.sort(np.abs(np.linalg.eigvals(Q))) for Q in Qn_by_T]; lT = np.array([float(mp.log(mp.mpf(t))) for t in Ts])
    vals = [np.polyfit(lT, np.log([e[i] for e in ev]), 1)[0] for i in range(len(ev[0]))]
    return np.array(vals), int(sum(abs(x) < 0.02 for x in vals))


def track(params, solver):
    """solver(p) -> (points as list of tuples, values as list); returns arrays P[step, k, :], W[step, k] with continuous labels"""
    prevP = None; P_all = []; W_all = []
    for p in params:
        pts, Wv = solver(p); P = np.array([[complex(x) for x in y] for y in pts]); Wv = np.array([complex(w) for w in Wv])
        if prevP is not None:
            C = sum(np.abs(P[None, :, i] - prevP[:, None, i]) / np.abs(prevP[:, None, i]) for i in range(P.shape[1]))
            _, col = linear_sum_assignment(C); P = P[col]; Wv = Wv[col]
        prevP = P; P_all.append(P); W_all.append(Wv)
    return np.array(P_all), np.array(W_all)


def stokes_events(Wtr, params, theta):
    out = []; params = np.array(params)
    for i in range(Wtr.shape[1]):
        for j in range(i + 1, Wtr.shape[1]):
            z = np.exp(-1j * theta) * (Wtr[:, j] - Wtr[:, i]); s = np.sign(z.imag)
            for k in np.where(s[1:] * s[:-1] < 0)[0]:
                t = z.imag[k] / (z.imag[k] - z.imag[k + 1]); out.append((params[k] + t * (params[k + 1] - params[k]), i, j))
    return sorted(out)
