"""Given measurements, partly masked: how much can a fixed-dimension (two-qubit) explanation still recover?

Setting. Two qubits; Alice measures spin along K known directions a_i, Bob along K known directions b_j; every pair (i, j) is a context
whose 2x2 table of outcomes (+1/-1) is observed with N shots. A fraction of the directions is masked: their tables are still observed,
but the solver does not know the direction. The state is unknown. For a two-qubit state with Bloch vectors s, t and correlation matrix T,
  P(alpha, beta | a, b) = (1 + alpha s.a + beta t.b + alpha beta a^T T b) / 4,
which is linear in theta = (s, t, T) for fixed directions and linear in each direction for a fixed state (bilinear overall).

  bell_data(rho, A, B, N, rng)        observed tables (K x K x 2 x 2), from counts (N = None: exact)
  fit_state(A, B, P, N)                weighted least squares for theta given all directions (chi-square-like rss)
  batch_rss(Abatch, Bbatch, P, N)      the same for a batch of direction sets (vectorised normal equations)
  rho_from(theta), psd_fit(...)        the state, and the fit constrained to the positive semidefinite cone (rho = L L^* / tr)
  fib_sphere(L)                        a near-uniform lattice of L directions
  refine(...)                          joint least squares over theta and the hidden directions' angles from a starting point
  lattice_search(...)                  exhaustive lattice over the hidden directions (h <= 2): coarse cull, refinement of cluster
                                       representatives, chi-square cull, cone cull, distinct modes
  sweep_search(...)                    coordinate-wise lattice sweeps from random starts (any h)
  local_search(...)                    plain local least squares from random starts (the baseline)
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

SIG = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
I2 = np.eye(2)
OUT = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def rho_from(theta):
    s, t, T = theta[:3], theta[3:6], theta[6:].reshape(3, 3)
    R = np.kron(I2, I2).astype(complex)
    for k in range(3):
        R += s[k] * np.kron(SIG[k], I2) + t[k] * np.kron(I2, SIG[k])
        for l in range(3): R += T[k, l] * np.kron(SIG[k], SIG[l])
    return R / 4


def theta_from(rho):
    s = [np.trace(rho @ np.kron(S, I2)).real for S in SIG]; t = [np.trace(rho @ np.kron(I2, S)).real for S in SIG]
    T = [[np.trace(rho @ np.kron(S1, S2)).real for S2 in SIG] for S1 in SIG]
    return np.r_[s, t, np.ravel(T)]


def probs(theta, A, B):
    s, t, T = theta[:3], theta[3:6], theta[6:].reshape(3, 3)
    P = np.zeros((len(A), len(B), 4))
    for o, (al, be) in enumerate(OUT):
        P[:, :, o] = (1 + al * (A @ s)[:, None] + be * (B @ t)[None, :] + al * be * (A @ T @ B.T)) / 4
    return P


def bell_data(rho, A, B, N, rng):
    P = probs(theta_from(rho), A, B); P = np.clip(P, 0, None); P /= P.sum(axis=2, keepdims=True)
    if N is None: return P
    return np.array([[rng.multinomial(N, P[i, j]) / N for j in range(len(B))] for i in range(len(A))])


def _design(A, B):
    """X (M x R x 15) for a batch of direction sets A (M x K x 3), B (M x K x 3); rows ordered (i, j, outcome)."""
    M, K = A.shape[0], A.shape[1]; rows = []
    for al, be in OUT:
        pass
    Xs = []
    for i in range(K):
        for j in range(B.shape[1]):
            for al, be in OUT:
                Xs.append(np.concatenate([al * A[:, i], be * B[:, j], al * be * np.einsum("mk,ml->mkl", A[:, i], B[:, j]).reshape(M, 9)], axis=1) / 4)
    return np.stack(Xs, axis=1)


def _yw(P, N):
    y = P.reshape(-1) - 0.25
    w = (N / np.clip(P.reshape(-1), 0.5 / N, None)) if N else np.ones_like(y)
    return y, w


def batch_rss(Ab, Bb, P, N, chunk=3000):
    y, w = _yw(P, N); out = np.empty(Ab.shape[0]); th = np.empty((Ab.shape[0], 15))
    for c in range(0, Ab.shape[0], chunk):
        X = _design(Ab[c:c + chunk], Bb[c:c + chunk]); Xw = X * w[None, :, None]
        G = np.einsum("mri,mrj->mij", Xw, X) + 1e-10 * np.eye(15); h = np.einsum("mri,r->mi", Xw, y)
        t = np.linalg.solve(G, h[..., None])[..., 0]; r = y[None] - np.einsum("mri,mi->mr", X, t)
        out[c:c + chunk] = np.einsum("mr,r->m", r * r, w); th[c:c + chunk] = t
    return out, th


def fit_state(A, B, P, N):
    r, t = batch_rss(A[None], B[None], P, N); return t[0], float(r[0])


def rss_theta(theta, A, B, P, N):
    y, w = _yw(P, N); X = _design(A[None], B[None])[0]; r = y - X @ theta; return float((w * r * r).sum())


def psd_fit(A, B, P, N, theta0):
    """least squares over rho = L L^* / tr (positive semidefinite by construction), started from the PSD projection of theta0."""
    R = rho_from(theta0); v, U = np.linalg.eigh(R); v = np.clip(v, 1e-6, None); R = U @ np.diag(v) @ U.conj().T; R /= np.trace(R).real
    L0 = np.linalg.cholesky(R + 1e-9 * np.eye(4)); idx = np.tril_indices(4)
    z0 = np.r_[L0[idx].real, L0[idx].imag]
    y, w = _yw(P, N); X = _design(A[None], B[None])[0]; sw = np.sqrt(w)
    def unpack(z):
        L = np.zeros((4, 4), complex); n = len(idx[0]); L[idx] = z[:n] + 1j * z[n:]; Rr = L @ L.conj().T; return Rr / np.trace(Rr).real
    res = least_squares(lambda z: sw * (y - X @ theta_from(unpack(z))), z0, xtol=1e-12, ftol=1e-12)
    th = theta_from(unpack(res.x)); return th, rss_theta(th, A, B, P, N)


def fib_sphere(L):
    k = np.arange(L) + 0.5; phi = np.arccos(1 - 2 * k / L); th = np.pi * (1 + 5 ** 0.5) * k
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def _ang(v):
    v = v / np.linalg.norm(v); return np.array([np.arccos(np.clip(v[2], -1, 1)), np.arctan2(v[1], v[0])])


def _vec(a):
    return np.array([np.sin(a[0]) * np.cos(a[1]), np.sin(a[0]) * np.sin(a[1]), np.cos(a[0])])


def place(A, B, hidden, dirs):
    A2, B2 = A.copy(), B.copy()
    for (side, i), v in zip(hidden, dirs):
        (A2 if side == "A" else B2)[i] = v
    return A2, B2


def refine(A, B, P, N, hidden, start_dirs):
    y, w = _yw(P, N); sw = np.sqrt(w)
    def resid(z):
        dirs = [_vec(z[2 * k:2 * k + 2]) for k in range(len(hidden))]; A2, B2 = place(A, B, hidden, dirs)
        X = _design(A2[None], B2[None])[0]; return sw * (y - X @ z[2 * len(hidden):])
    A2, B2 = place(A, B, hidden, start_dirs); th0, _ = fit_state(A2, B2, P, N)
    z0 = np.r_[np.concatenate([_ang(d) for d in start_dirs]), th0]
    r = least_squares(resid, z0, xtol=1e-12, ftol=1e-12, max_nfev=400)
    dirs = [_vec(r.x[2 * k:2 * k + 2]) for k in range(len(hidden))]; return dirs, r.x[2 * len(hidden):], float((r.fun ** 2).sum())


def _dist(d1, d2):
    return max(np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for a, b in zip(d1, d2))


def lattice_search(A, B, P, N, hidden, L=300, conf=0.99, seed=0):
    """exhaustive lattice over the hidden directions (h <= 2). Returns a dict with the survivors and modes."""
    rng = np.random.default_rng(seed); h = len(hidden); G = fib_sphere(L); combos = np.array(list(itertools.product(range(L), repeat=h)))
    Ab = np.repeat(A[None], len(combos), 0); Bb = np.repeat(B[None], len(combos), 0)
    for k, (side, i) in enumerate(hidden): (Ab if side == "A" else Bb)[:, i] = G[combos[:, k]]
    rss, _ = batch_rss(Ab, Bb, P, N); best = int(np.argmin(rss)); spacing = np.sqrt(4 * np.pi / L)
    # lattice allowance: rss increase for a half-spacing displacement around the best lattice point (no knowledge of the truth)
    pert = []
    for _ in range(12):
        dirs = []
        for k in range(h):
            v = G[combos[best, k]] + rng.normal(0, 1, 3) * spacing / 2 / np.sqrt(3); dirs.append(v / np.linalg.norm(v))
        A2, B2 = place(A, B, hidden, dirs); pert.append(fit_state(A2, B2, P, N)[1] - rss[best])
    q = chi2.ppf(conf, 2 * h); allow = max(0.0, float(np.median(pert)))
    keep = np.where(rss <= rss[best] + q + 2 * allow)[0]
    # cluster survivors greedily (radius 2 spacings) and refine one representative per cluster
    order = keep[np.argsort(rss[keep])]; reps = []
    for c in order:
        d = [G[combos[c, k]] for k in range(h)]
        if all(_dist(d, [G[combos[r, k]] for k in range(h)]) > np.degrees(2 * spacing) for r in reps): reps.append(c)
    refined = [refine(A, B, P, N, hidden, [G[combos[r, k]] for k in range(h)]) for r in reps]
    rbest = min(x[2] for x in refined)
    modes = []
    for dirs, th, r in sorted(refined, key=lambda x: x[2]):
        if r > rbest + q: continue
        if any(_dist(dirs, m["dirs"]) < 2.0 for m in modes): continue
        thp, rp = psd_fit(*place(A, B, hidden, dirs), P, N, th)
        modes.append({"dirs": dirs, "theta": th, "rss": r, "psd_theta": thp, "psd_rss": rp,
                      "cone_ok": rp <= rbest + q, "min_eig": float(np.linalg.eigvalsh(rho_from(th)).min())})
    return {"L": L, "evaluated": int(len(combos)), "coarse_survivors": int(len(keep)), "coarse_fraction": len(keep) / len(combos),
            "clusters": len(reps), "modes": modes, "threshold": q, "allowance": allow}


def sweep_search(A, B, P, N, hidden, L=300, starts=8, sweeps=4, seed=0):
    """coordinate-wise lattice sweeps from random starts, then refinement; returns the refined optima and evaluation count."""
    rng = np.random.default_rng(seed); G = fib_sphere(L); h = len(hidden); outs = []; evals = 0
    for _ in range(starts):
        dirs = [v / np.linalg.norm(v) for v in rng.normal(size=(h, 3))]
        for _ in range(sweeps):
            for k, (side, i) in enumerate(hidden):
                Ab = np.repeat(place(A, B, hidden, dirs)[0][None], L, 0); Bb = np.repeat(place(A, B, hidden, dirs)[1][None], L, 0)
                (Ab if side == "A" else Bb)[:, i] = G; r, _ = batch_rss(Ab, Bb, P, N); evals += L; dirs[k] = G[int(np.argmin(r))]
        outs.append(refine(A, B, P, N, hidden, dirs))
    return outs, evals


def local_search(A, B, P, N, hidden, starts=30, seed=0):
    rng = np.random.default_rng(seed)
    return [refine(A, B, P, N, hidden, [v / np.linalg.norm(v) for v in rng.normal(size=(len(hidden), 3))]) for _ in range(starts)]


def refine_psd(A, B, P, N, hidden, start_dirs, rho0=None, seed=0):
    """joint least squares over the hidden directions' angles and a positive semidefinite state rho = L L^* / tr."""
    y, w = _yw(P, N); sw = np.sqrt(w); idx = np.tril_indices(4); n = len(idx[0]); h = len(hidden)
    def unpack(z):
        L = np.zeros((4, 4), complex); L[idx] = z[2 * h:2 * h + n] + 1j * z[2 * h + n:]; R = L @ L.conj().T; return R / np.trace(R).real
    def resid(z):
        dirs = [_vec(z[2 * k:2 * k + 2]) for k in range(h)]; A2, B2 = place(A, B, hidden, dirs)
        return sw * (y - _design(A2[None], B2[None])[0] @ theta_from(unpack(z)))
    rng = np.random.default_rng(seed)
    if rho0 is None:
        G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)); rho0 = G @ G.conj().T; rho0 /= np.trace(rho0).real
    L0 = np.linalg.cholesky(rho0 + 1e-6 * np.eye(4))
    z0 = np.r_[np.concatenate([_ang(d) for d in start_dirs]), L0[idx].real, L0[idx].imag]
    r = least_squares(resid, z0, xtol=1e-12, ftol=1e-12, max_nfev=2000)
    dirs = [_vec(r.x[2 * k:2 * k + 2]) for k in range(h)]; th = theta_from(unpack(r.x)); return dirs, th, float((r.fun ** 2).sum())


def span_rank(V, tol=0.15):
    """rank of a set of unit directions, with a tolerance for near-degenerate sets"""
    if len(V) == 0: return 0
    s = np.linalg.svd(np.asarray(V), compute_uv=False); return int((s > tol * max(1.0, s[0])).sum())


def cone_cull(A, B, P, N, hidden, L=300, tol=1e-9):
    """exhaustive lattice over the hidden directions; for each point the least-squares state theta and the smallest eigenvalue of
    rho(theta). Points whose state is not positive semidefinite would give negative probabilities to some measurement: they are culled
    by the cone. Returns the lattice, the combos, rss, min eigenvalues and the surviving mask."""
    G = fib_sphere(L); h = len(hidden); combos = np.array(list(itertools.product(range(L), repeat=h)))
    Ab = np.repeat(A[None], len(combos), 0); Bb = np.repeat(B[None], len(combos), 0)
    for k, (side, i) in enumerate(hidden): (Ab if side == "A" else Bb)[:, i] = G[combos[:, k]]
    rss, th = batch_rss(Ab, Bb, P, N)
    R = np.zeros((len(th), 4, 4), complex)
    s, t, T = th[:, :3], th[:, 3:6], th[:, 6:].reshape(-1, 3, 3)
    R += np.kron(I2, I2)[None]
    for k in range(3):
        R += s[:, k, None, None] * np.kron(SIG[k], I2)[None] + t[:, k, None, None] * np.kron(I2, SIG[k])[None]
        for l in range(3): R += T[:, k, l, None, None] * np.kron(SIG[k], SIG[l])[None]
    mins = np.linalg.eigvalsh(R / 4)[:, 0]
    return {"G": G, "combos": combos, "rss": rss, "theta": th, "min_eig": mins, "keep": mins >= -tol}


def _min_eig_at(A, B, P, N, hidden, dirs):
    A2, B2 = place(A, B, hidden, dirs); th, _ = fit_state(A2, B2, P, N); return float(np.linalg.eigvalsh(rho_from(th))[0]), th


def feasible_region(A, B, P, N, hidden, L=200, top=0.005, seeds=24, samples=4000, seed=0):
    """positivity-only search when the linear fit is uninformative: score lattice points by the smallest eigenvalue of the implied state,
    keep the best fraction, climb from cluster seeds to maximise it (Nelder-Mead over the hidden angles), and sample the feasible set
    (smallest eigenvalue >= 0) around each feasible optimum by random walks. Returns the feasible samples (directions, theta)."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed); h = len(hidden); cc = cone_cull(A, B, P, N, hidden, L=L)
    order = np.argsort(-cc["min_eig"]); G = cc["G"]; cand = order[:max(1, int(top * len(order)))]
    reps = []
    for c in cand:
        d = [G[cc["combos"][c, k]] for k in range(h)]
        if all(_dist(d, r) > 20 for r in reps): reps.append(d)
        if len(reps) >= seeds: break
    optima = []
    for d in reps:
        z0 = np.concatenate([_ang(v) for v in d])
        r = minimize(lambda z: -_min_eig_at(A, B, P, N, hidden, [_vec(z[2 * k:2 * k + 2]) for k in range(h)])[0], z0, method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
        optima.append((r.x, -r.fun))
    feas = [o for o in optima if o[1] >= 0]
    out = []
    for z, _ in feas:
        cur = z.copy(); step = 0.05
        for _ in range(samples // max(1, len(feas))):
            prop = cur + rng.normal(0, step, cur.size); dirs = [_vec(prop[2 * k:2 * k + 2]) for k in range(h)]
            e, th = _min_eig_at(A, B, P, N, hidden, dirs)
            if e >= 0: cur = prop; out.append((dirs, th))
    return {"lattice": cc, "optima": optima, "feasible_optima": len(feas), "samples": out}
