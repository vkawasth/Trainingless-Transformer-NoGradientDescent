"""Three latent types on five binary variables: the proof of concept for the arbitrary-cover theory (companion "Two latent types on
arbitrary covers", Section "Three types"). New module.

Parameters theta = (w1, w2, a_1, a_2, a_3) with w3 = 1 - w1 - w2 and a_k in [0, 1]^m (2 + 3m numbers; 17 for m = 5).

  law_k(w, A)                    the 2^m-cell law of the k-type mixture (w: k weights, A: k x m)
  unpack(theta, m, k)            weights and the k x m matrix
  window_tables(P, windows)      the concatenated marginal tables of the windows (the observation)
  obs_rank(theta, m, windows)    rank of the Jacobian of theta -> window tables (complex-step derivatives): the image dimension
  between_cov(w, A)              the between-type covariance V V^T (rank k - 1) and V
  pentad_m2()                    the pentad: Macaulay2 implicitisation of the rank-2 off-diagonal 5 x 5 pattern
  fit_windows(target, m, k, windows, starts, rng)   multistart least squares for k-type parameters reproducing window tables
  range_functional(ell, theta0, m, k, windows, sense, starts, rng)   sense = +1: maximum, -1: minimum, of of a linear decision functional ell . P over
                                 the k-type explanations of theta0's window tables (SLSQP with equality constraints)
"""
from __future__ import annotations
import itertools, os, subprocess, tempfile
import numpy as np
from scipy.optimize import least_squares, minimize


def law_k(w, A):
    k, m = A.shape; P = np.zeros((2,) * m, dtype=np.result_type(A, w, float))
    for x in itertools.product((0, 1), repeat=m):
        xx = np.array(x)
        P[x] = sum(w[j] * np.prod(np.where(xx == 1, A[j], 1 - A[j])) for j in range(k))
    return P


def unpack(theta, m, k=3):
    w = np.r_[theta[:k - 1], 1 - np.sum(theta[:k - 1])]; A = np.reshape(theta[k - 1:], (k, m)); return w, A


def pack(w, A): return np.r_[w[:-1], np.ravel(A)]


def window_tables(P, windows):
    m = P.ndim; return np.concatenate([P.sum(axis=tuple(i for i in range(m) if i not in W)).ravel() for W in windows])


def obs(theta, m, windows, k=3):
    w, A = unpack(theta, m, k); return window_tables(law_k(w, A), windows)


def obs_jacobian(theta, m, windows, k=3, h=1e-20):
    n = len(theta); cols = []
    for j in range(n):
        th = theta.astype(complex); th[j] += 1j * h; cols.append(np.imag(obs(th, m, windows, k)) / h)
    return np.array(cols).T


def obs_rank(theta, m, windows, k=3, tol=1e-9):
    J = obs_jacobian(theta, m, windows, k); s = np.linalg.svd(J, compute_uv=False); return int(np.sum(s > tol * s[0])), s


def between_cov(w, A):
    mu = w @ A; D = A - mu; C = (D.T * w) @ D; return C, mu


def random_theta(m, k, rng, wmin=0.15, gap=0.15):
    while True:
        w = rng.dirichlet(np.ones(k) * 3)
        if w.min() < wmin: continue
        A = rng.uniform(0.08, 0.92, (k, m))
        if min(np.min(np.abs(A[i] - A[j])) for i, j in itertools.combinations(range(k), 2)) > gap: return pack(w, A)


def pentad_m2(timeout=600):
    """the ideal of the off-diagonal entries of a rank-2 symmetric 5 x 5 matrix (Drton-Sturmfels-Sullivant's pentad)"""
    names = [f"c{i}{j}" for i, j in itertools.combinations(range(5), 2)]
    imgs = [f"x_{i}*x_{j}+y_{i}*y_{j}" for i, j in itertools.combinations(range(5), 2)]
    script = f"""R = QQ[x_0..x_4, y_0..y_4]; S = QQ[{",".join(names)}];
f = map(R, S, {{{",".join(imgs)}}}); I = ker f; print("GENS"); scan(flatten entries mingens I, g -> print toString g);
print("CODIM"); print toString codim I; print("END");"""
    with tempfile.NamedTemporaryFile("w", suffix=".m2", delete=False) as fh: fh.write(script); path = fh.name
    out = subprocess.run(["M2", "--script", path], capture_output=True, text=True, timeout=timeout).stdout; os.unlink(path)
    lines = out.splitlines(); g = lines[lines.index("GENS") + 1:lines.index("CODIM")]; return g, int(lines[lines.index("CODIM") + 1])


def fit_windows(target, m, k, windows, starts, rng, tol=1e-12):
    sols = []
    lb = np.r_[np.zeros(k - 1), np.zeros(k * m)]; ub = np.r_[np.ones(k - 1), np.ones(k * m)]
    for _ in range(starts):
        th0 = pack(rng.dirichlet(np.ones(k)), rng.uniform(0.05, 0.95, (k, m)))
        f = lambda th: np.r_[obs(th, m, windows, k) - target, 10 * min(0.0, 1 - np.sum(th[:k - 1]))]
        r = least_squares(f, th0, bounds=(lb, ub), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        if np.abs(r.fun).max() < 1e-9: sols.append(r.x)
    return sols


def canonical(theta, m, k=3):
    """sort the types by weight (then by a) to quotient the label permutations"""
    w, A = unpack(theta, m, k); order = np.lexsort(tuple(A.T[::-1]) + (w,)); return w[order], A[order]


def range_functional(ell, theta0, m, k, windows, sense, starts, rng, theta_init=None):
    target = obs(theta0, m, windows, k) if theta_init is None else obs(theta_init, m, windows, k)
    best = None
    cons = [{"type": "eq", "fun": lambda th: obs(th, m, windows, k) - target}, {"type": "ineq", "fun": lambda th: 1 - np.sum(th[:k - 1])}]
    bnds = [(0.0, 1.0)] * (k - 1 + k * m)
    for s in range(starts):
        th0 = (theta0 if s == 0 else theta0 + rng.normal(scale=0.05, size=len(theta0))).clip(0.01, 0.99)
        fobj = lambda th: -sense * float(ell @ law_k(*unpack(th, m, k)).ravel())
        r = minimize(fobj, th0, method="SLSQP", bounds=bnds, constraints=cons, options=dict(maxiter=500, ftol=1e-12))
        if np.abs(obs(r.x, m, windows, k) - target).max() < 1e-8 and np.all(r.x >= -1e-9) and np.all(r.x <= 1 + 1e-9) and np.sum(r.x[:k - 1]) <= 1 + 1e-9:
            val = -sense * r.fun
            if best is None or sense * val > sense * best[0]: best = (val, r.x)
    return best


def moments_obs(theta, m, faces, k=3):
    """means and the central moments m_S = sum_j w_j prod_{i in S} (a_ji - mu_i) on the faces (|S| >= 2): an independent
    parametrisation of the window tables (observing a window = observing the moments of its faces)"""
    w, A = unpack(theta, m, k); mu = w @ A; D = A - mu
    return np.r_[mu, [np.sum(w * np.prod(D[:, list(S)], axis=1)) for S in faces]]


def range_moments(ell, theta0, m, k, faces, sense, starts, rng, scale=0.05):
    target = moments_obs(theta0, m, faces, k); best = None
    cons = [{"type": "eq", "fun": lambda th: moments_obs(th, m, faces, k) - target}, {"type": "ineq", "fun": lambda th: 1 - np.sum(th[:k - 1])}]
    bnds = [(0.0, 1.0)] * (k - 1 + k * m)
    for s in range(starts):
        th0 = (theta0 if s == 0 else theta0 + rng.normal(scale=scale, size=len(theta0))).clip(0.01, 0.99)
        fobj = lambda th: -sense * float(ell @ law_k(*unpack(th, m, k)).ravel())
        try:
            r = minimize(fobj, th0, method="SLSQP", bounds=bnds, constraints=cons, options=dict(maxiter=800, ftol=1e-13))
        except Exception:
            continue
        x = r.x
        if np.abs(moments_obs(x, m, faces, k) - target).max() < 1e-9 and np.all(x >= -1e-9) and np.all(x <= 1 + 1e-9) and np.sum(x[:k - 1]) <= 1 + 1e-9:
            val = float(ell @ law_k(*unpack(x, m, k)).ravel())
            if best is None or sense * val > sense * best[0]: best = (val, x)
    return best


def fibre_points(theta0, m, k, faces, n, rng, tol=1e-11, tries=2000):
    """random points of the k-type fibre over theta0's moments on the faces: random starts projected by bounded least squares"""
    target = moments_obs(theta0, m, faces, k); pts = []
    lb = np.zeros(k - 1 + k * m); ub = np.ones(k - 1 + k * m)
    for _ in range(tries):
        th = pack(rng.dirichlet(np.ones(k) * 2), rng.uniform(0.02, 0.98, (k, m)))
        f = lambda x: np.r_[moments_obs(x, m, faces, k) - target, 5 * min(0.0, 1 - np.sum(x[:k - 1]))]
        r = least_squares(f, th, bounds=(lb, ub), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
        if np.abs(r.fun).max() < tol: pts.append(r.x)
        if len(pts) >= n: break
    return pts


def range_from(ell, starts, m, k, faces, target, sense):
    cons = [{"type": "eq", "fun": lambda th: moments_obs(th, m, faces, k) - target}, {"type": "ineq", "fun": lambda th: 1 - np.sum(th[:k - 1])}]
    bnds = [(0.0, 1.0)] * (k - 1 + k * m); best = None
    for th0 in starts:
        fobj = lambda th: -sense * float(ell @ law_k(*unpack(th, m, k)).ravel())
        try: r = minimize(fobj, th0, method="SLSQP", bounds=bnds, constraints=cons, options=dict(maxiter=600, ftol=1e-13))
        except Exception: continue
        x = r.x
        if np.abs(moments_obs(x, m, faces, k) - target).max() < 1e-9 and np.all(x >= -1e-9) and np.all(x <= 1 + 1e-9) and np.sum(x[:k - 1]) <= 1 + 1e-9:
            val = float(ell @ law_k(*unpack(x, m, k)).ravel())
            if best is None or sense * val > sense * best[0]: best = (val, x)
    return best
