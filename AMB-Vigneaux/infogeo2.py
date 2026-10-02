"""Second-order information geometry on the simplex of laws on K cells: component [G2].

First order is the Fisher metric (speed, distance, standard errors). Second order is how laws BEND: the two dual affine
connections (e and m) and the Levi-Civita (Fisher-Rao) connection give three notions of acceleration, hence three
curvatures of a path, and a curvature 2-form (holonomy) for a 2-parameter family. Everything here is closed form, IPF or
Newton on a scalar; nothing is fitted by gradient descent. Every function returns plain arrays/dicts for plotting.

Coordinates of a law p (all cells > 0):
    m (mixture)    p itself; tangent vectors sum to 0; metric g_m(v, w) = sum v w / p
    e (natural)    eta = log p up to an additive constant; metric I = diag(p) - p p^T
    sphere         x = 2 sqrt(p) on the radius-2 sphere; the Fisher metric is the round metric there, so the simplex
                   with Fisher-Rao distance d_FR(p, q) = 2 arccos(sum sqrt(p q)) is a piece of a sphere (curvature 1/4).
Paths s -> p(s):
    e-geodesic   eta affine in s  (exponential family; e.g. a covariate SHIFT)        gamma2_e = 0
    m-geodesic   p affine in s    (mixture with s-dependent weight; a PULL)            gamma2_m = 0
    FR geodesic  great circle on the sphere                                           kappa2_FR = 0
    gamma2_e is Efron's statistical curvature. All three are invariant under reparametrisation of s.
Second-order consequence (Efron-Hinkley): at the MLE, observed / expected information fluctuates with relative variance
gamma2_e / n; an e-flat family has observed = expected exactly.
Holonomy: on a grid of laws p_{uv}, the e-plaquette H = eta_{u+1,v+1} - eta_{u+1,v} - eta_{u,v+1} + eta_{uv} is the
curvature of 'transport the cell pattern additively in u and v'; H = 0 on every plaquette iff the no-three-way model
(cells x u x v) holds. |H|^2_I / 2 is the information (nats per unit) lost by assuming path-independent transport;
sampling noise adds sum_corners (K - 1) / n_corner to |H|^2_I, removed by debiasing. Same for the m-plaquette.
Top-order information: D_top(p) = KL(p || max-ent law with the same (n-1)-margins) (nats per unit) = what only the full
n-way joint carries; E[2 N D_top] = 1 under no top interaction (one degree of freedom), so the debiased value is
D_top - 1 / (2N). Co-information I_n = sum_{S != {}} (-1)^{|S|+1} H(X_S).
"""
from __future__ import annotations

import itertools

import numpy as np

from . import toric as T
from . import deform as D


# ------------------------------------------------------------------------------------------- basic geometry
def smooth(counts, prior=0.5):
    c = np.asarray(counts, float) + prior
    return c / c.sum(-1, keepdims=True)


def sphere(p):
    return 2 * np.sqrt(p)


def fr_dist(p, q):
    return float(2 * np.arccos(np.clip(np.sum(np.sqrt(p * q)), -1, 1)))


def kl(p, q):
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def fisher(p):
    return np.diag(p) - np.outer(p, p)


def geodesic(p, q, kind, t):
    """point at time t in [0, 1] on the e-, m- or FR-geodesic from p to q"""
    if kind == "e":
        r = p ** (1 - t) * q ** t; return r / r.sum()
    if kind == "m":
        return (1 - t) * p + t * q
    a, b = np.sqrt(p), np.sqrt(q); w = np.arccos(np.clip(a @ b, -1, 1))
    if w < 1e-12:
        return p.copy()
    x = (np.sin((1 - t) * w) * a + np.sin(t * w) * b) / np.sin(w)
    return x ** 2 / (x ** 2).sum()


# ------------------------------------------------------------------------------------------- jets and curvatures of a path
def _kappa2(v, a, G):
    """(|a|^2 |v|^2 - <a, v>^2) / |v|^6 in the inner product G (matrix or vector of diagonal weights)"""
    ip = (lambda x, y: x @ (G @ y)) if np.ndim(G) == 2 else (lambda x, y: np.sum(x * G * y))
    vv, aa, av = ip(v, v), ip(a, a), ip(a, v)
    return float((aa * vv - av * av) / vv ** 3) if vv > 0 else np.nan


def jets_e_to_m(p, d1, d2):
    """natural-coordinate jets (eta', eta'') at a law p -> (log p)', (log p)'', p', p''"""
    l1 = d1 - p @ d1
    l2 = d2 - p @ d2 - p @ (l1 ** 2)
    m1 = p * l1; m2 = p * (l2 + l1 ** 2)
    return l1, l2, m1, m2


def curvatures(p, d1, d2) -> dict:
    """the three curvatures of a path through p with natural jets d1 = eta', d2 = eta''"""
    l1, l2, m1, m2 = jets_e_to_m(p, d1, d2)
    ge = _kappa2(d1, d2, fisher(p))
    gm = _kappa2(m1, m2, 1.0 / p)
    x = sphere(p); x1 = m1 / np.sqrt(p); x2 = m2 / np.sqrt(p) - m1 ** 2 / (2 * p ** 1.5)
    a_tan = x2 - (x2 @ x) / (x @ x) * x                  # remove the sphere normal: covariant (Levi-Civita) acceleration
    gfr = _kappa2(x1, a_tan, np.ones_like(p))
    return dict(gamma2_e=ge, gamma2_m=gm, kappa2_fr=gfr, speed=float(np.sqrt(d1 @ fisher(p) @ d1)))


def poisson_irls(X, y, iters=200, tol=1e-10):
    """Poisson log-linear fit by iteratively reweighted least squares (each step a closed-form WLS)"""
    mu = y + 0.5; eta = np.log(mu)
    for _ in range(iters):
        z = eta + (y - mu) / mu; w = np.sqrt(mu)
        beta = np.linalg.lstsq(X * w[:, None], z * w, rcond=None)[0]
        new = np.clip(X @ beta, -50, 50)
        done = np.max(np.abs(new - eta)) < tol
        eta = new; mu = np.exp(eta)
        if done:
            break
    return mu, beta


def deviance(y, mu):
    m = y > 0
    return float(2 * (np.sum(y[m] * np.log(y[m] / mu[m])) - np.sum(y - mu)))


def local_jets(tab, s, i0, half=2, return_mu=False):
    """local quadratic Poisson fit of the path (bins x cells table, bin positions s) around bin i0:
    log mu_{t,x} = a_t + b_x + c_x (s_t - s0) + d_x (s_t - s0)^2. Returns p(s0), eta'(s0) = c, eta''(s0) = 2 d."""
    Tn, K = tab.shape; idx = np.arange(max(0, i0 - half), min(Tn, i0 + half + 1)); s0 = s[i0]
    sub = tab[idx]; u = s[idx] - s0
    tt = np.repeat(np.arange(len(idx)), K); xx = np.tile(np.arange(K), len(idx))
    A = np.eye(len(idx))[tt]; B = np.eye(K)[xx][:, 1:]
    X = np.hstack([A, B, B * u[tt][:, None], B * (u[tt] ** 2)[:, None]])
    mu, beta = poisson_irls(X, sub.ravel().astype(float) + 0.5)          # +1/2: empty cells stay finite
    L = len(idx); b = np.concatenate([[0], beta[L:L + K - 1]])
    c = np.concatenate([[0], beta[L + K - 1:L + 2 * (K - 1)]]); d = np.concatenate([[0], beta[L + 2 * (K - 1):]])
    p0 = np.exp(b - b.max()); p0 /= p0.sum()
    if return_mu:
        return p0, c, 2 * d, mu.reshape(len(idx), K), idx
    return p0, c, 2 * d


KEYS = ("gamma2_e", "gamma2_m", "kappa2_fr")


def local_curvatures(tab, s, i0, half=2, B=60, rng=None) -> dict:
    """curvatures at bin i0 with a parametric-bootstrap bias correction. Noise in the fitted acceleration inflates every
    curvature (|a|^2 picks up E|noise|^2), most where the path is slow. Simulate the fitted local model B times, refit:
    debiased = 2 * raw - mean(boot), sd = sd(boot)."""
    rng = rng or np.random.default_rng(0)
    p0, d1, d2, mu, idx = local_jets(tab, s, i0, half, return_mu=True)
    raw = curvatures(p0, d1, d2); boot = {k: [] for k in KEYS}
    sub_s = s[idx]; j0 = int(np.where(idx == i0)[0][0])
    for _ in range(B):
        tb = rng.poisson(mu)
        r = curvatures(*local_jets(tb, sub_s, j0, half))
        for k in KEYS:
            boot[k].append(r[k])
    out = dict(raw)
    for k in KEYS:
        bb = np.array(boot[k]); out[k + "_raw"] = raw[k]; out[k] = float(2 * raw[k] - bb.mean()); out[k + "_sd"] = float(bb.std())
    return out


def path_profile(tab, s, half=2, B=60, rng=None) -> dict:
    """bias-corrected curvatures along the whole path (one entry per interior bin) + Fisher-Rao lengths"""
    rng = rng or np.random.default_rng(0)
    P = smooth(tab); rows = []
    for i in range(half, len(s) - half):
        rows.append(dict(i=i, s=float(s[i]), **local_curvatures(tab, s, i, half, B, rng)))
    steps = [fr_dist(P[i], P[i + 1]) for i in range(len(s) - 1)]
    return dict(rows=rows, fr_steps=steps, fr_length=float(np.sum(steps)), fr_chord=fr_dist(P[0], P[-1]),
                e_dev=float(np.sum([kl(P[i], P[i + 1]) for i in range(len(s) - 1)])))


def path_tests(tab, s) -> dict:
    """global likelihood-ratio tests of the path shape: e-geodesic (eta affine in s) vs quadratic vs saturated"""
    Tn, K = tab.shape; y = tab.ravel().astype(float)
    tt = np.repeat(np.arange(Tn), K); xx = np.tile(np.arange(K), Tn)
    A = np.eye(Tn)[tt]; B = np.eye(K)[xx][:, 1:]; u = s[tt][:, None]
    X1 = np.hstack([A, B, B * u]); X2 = np.hstack([X1, B * u ** 2])
    D1 = deviance(y, poisson_irls(X1, y)[0]); D2 = deviance(y, poisson_irls(X2, y)[0])
    df1 = Tn * (K - 1) - 2 * (K - 1); df2 = df1 - (K - 1)
    return dict(dev_eflat=D1, df_eflat=df1, dev_quad=D2, df_quad=df2, lr_bend=D1 - D2, df_bend=K - 1)


# ------------------------------------------------------------------------------------------- mixture decomposition
def mixture_path_check(tabs_by_class, weights_by_bin):
    """If the pooled path is p(s) = sum_k w_k(s) q_k with FIXED class laws q_k it is an m-flat SURFACE (a line for two
    classes); compare the pooled law with that reconstruction: TV per bin of pooled vs fixed-class mixture."""
    Q = np.array([smooth(t.sum(0)) for t in tabs_by_class])            # class laws averaged over bins
    pooled = smooth(sum(tabs_by_class))
    recon = weights_by_bin @ Q
    return [float(0.5 * np.abs(pooled[i] - recon[i]).sum()) for i in range(len(recon))]


# ------------------------------------------------------------------------------------------- Efron-Hinkley information loss
def softmax(e):
    q = np.exp(e - e.max()); return q / q.sum()


def efron_hinkley(beta, c, d, n, reps=2000, rng=None, smax=4.0):
    """the 1-parameter family eta(s) = beta + c s + d s^2, truth s = 0. Simulate n draws, MLE s_hat by scalar Newton
    (closed-form score and Hessian), and return n * Var(J(s_hat) / (n I(s_hat))) -- Efron-Hinkley predict gamma2_e(0)."""
    rng = rng or np.random.default_rng(0); p0 = softmax(beta); R = []
    def parts(s):
        e1 = c + 2 * d * s; p = softmax(beta + c * s + d * s * s); m1 = p @ e1
        return p, e1, m1, p @ (e1 - m1) ** 2
    for _ in range(reps):
        x = rng.multinomial(n, p0); s = 0.0; ok = False
        for _ in range(60):
            p, e1, m1, v1 = parts(s)
            g = x @ e1 - n * m1                                   # score
            h = 2 * x @ d - n * (2 * p @ d + v1)                  # second derivative
            if h >= 0:
                break
            step = -g / h; s = float(np.clip(s + step, -smax, smax))
            if abs(step) < 1e-10:
                ok = True; break
        if not ok or abs(s) >= smax:
            continue
        p, e1, m1, v1 = parts(s)
        J = -(2 * x @ d - n * (2 * p @ d + v1))
        I = v1                                                    # expected information per draw at s (= Var_p(eta'))
        R.append(J / (n * I))
    R = np.array(R)
    return dict(n=n, kept=len(R), nvar=float(n * R.var()), mean=float(R.mean()))


# ------------------------------------------------------------------------------------------- holonomy 2-form
def plaquettes(C, prior=0.5) -> dict:
    """C: (U, V, K) counts on a 2-parameter grid. e- and m-plaquette holonomies, their Fisher norms, the sampling-noise
    level and the debiased information (nats per unit) lost by flat transport on each plaquette."""
    C = np.asarray(C, float); n = C.sum(-1); P = smooth(C, prior); L = np.log(P); K = C.shape[-1]
    He = L[1:, 1:] - L[1:, :-1] - L[:-1, 1:] + L[:-1, :-1]
    Hm = P[1:, 1:] - P[1:, :-1] - P[:-1, 1:] + P[:-1, :-1]
    Pc = (P[1:, 1:] + P[1:, :-1] + P[:-1, 1:] + P[:-1, :-1]) / 4
    he = np.einsum("uvk,uvk->uv", He * Pc, He) - np.einsum("uvk,uvk->uv", He, Pc) ** 2      # H^T (diag p - p p^T) H
    hm = np.einsum("uvk,uvk->uv", Hm, Hm / Pc)
    # sampling noise: Cov(log p_hat_c) ~ (diag(1/p_c) - 1 1^T)/n_c, Cov(p_hat_c) = (diag p_c - p_c p_c^T)/n_c, read in the
    # metric at the plaquette centre: E|H_e|^2 = sum_c (sum Pc/p_c - 1)/n_c,  E|H_m|^2 = sum_c (sum p_c/Pc - sum p_c^2/Pc)/n_c
    corners = ((slice(1, None), slice(1, None)), (slice(1, None), slice(None, -1)), (slice(None, -1), slice(1, None)), (slice(None, -1), slice(None, -1)))
    noise_e = sum(((Pc / P[a, b]).sum(-1) - 1) / n[a, b] for a, b in corners)
    noise_m = sum(((P[a, b] / Pc).sum(-1) - (P[a, b] ** 2 / Pc).sum(-1)) / n[a, b] for a, b in corners)
    return dict(he=he, hm=hm, noise=noise_e, noise_m=noise_m, nats_e=(he - noise_e) / 2, nats_m=(hm - noise_m) / 2,
                total_nats_e=float(((he - noise_e) / 2).sum()), total_nats_m=float(((hm - noise_m) / 2).sum()),
                z_e=float((he.sum() - noise_e.sum()) / np.sqrt(2 * (noise_e ** 2).sum() / (K - 1))))


def plaquette_null(C, B=100, rng=None) -> dict:
    """null distribution of the total debiased holonomy: resample grids from the flat (no-three-way) fit"""
    rng = rng or np.random.default_rng(0)
    M = flat_fit(C)
    tot = np.array([plaquettes(flat_null_grid(C, rng, M))["total_nats_e"] for _ in range(B)])
    obs = plaquettes(C)["total_nats_e"]
    return dict(obs=obs, null_mean=float(tot.mean()), null_sd=float(tot.std()), p=float((1 + (tot >= obs).sum()) / (B + 1)),
                excess=float(obs - tot.mean()))


def plaquette_bootstrap(C, B=200, rng=None):
    """parametric bootstrap of the debiased plaquette information (resample each grid cell from its own law)"""
    rng = rng or np.random.default_rng(0); C = np.asarray(C, float); n = C.sum(-1).astype(int); P = smooth(C)
    tot_e, tot_m, maps = [], [], []
    for _ in range(B):
        Cb = np.array([[rng.multinomial(n[u, v], P[u, v]) for v in range(C.shape[1])] for u in range(C.shape[0])])
        r = plaquettes(Cb); tot_e.append(r["total_nats_e"]); tot_m.append(r["total_nats_m"]); maps.append(r["nats_e"])
    return dict(total_e_sd=float(np.std(tot_e)), total_m_sd=float(np.std(tot_m)), map_sd=np.std(maps, 0))


def flat_fit(C):
    """the no-three-way (flat transport) fit of a (U, V, K) grid by IPF on its three 2-margins"""
    C = np.asarray(C, float); M = np.ones_like(C)
    for _ in range(3000):
        old = M.copy()
        for ax in (0, 1, 2):
            M = M * np.where(M.sum(ax, keepdims=True) > 0, C.sum(ax, keepdims=True) / np.maximum(M.sum(ax, keepdims=True), 1e-300), 0)
        if np.max(np.abs(M - old)) < 1e-9:
            break
    return M


def flat_null_grid(C, rng=None, M=None):
    """a grid with the same margins but NO holonomy: the flat fit, resampled (calibration)"""
    rng = rng or np.random.default_rng(0); C = np.asarray(C, float); M = flat_fit(C) if M is None else M
    n = C.sum(-1).astype(int)
    def draw(u, v):
        tot = M[u, v].sum()
        return rng.multinomial(n[u, v], M[u, v] / tot) if tot > 0 and n[u, v] > 0 else np.zeros(C.shape[-1])
    return np.array([[draw(u, v) for v in range(C.shape[1])] for u in range(C.shape[0])])


def holonomy_info(C, B=100, rng=None) -> dict:
    """information lost by flat (path-independent) transport, in nats per unit: KL(data || flat fit) summed over the grid,
    divided by N. G = 2 N * that is the LR statistic of the no-three-way model; its null is calibrated by refitting
    resampled flat grids, so the excess (observed - null mean) is the debiased holonomy information. Also a map of the
    per-grid-cell excess (where transport fails), for plotting."""
    rng = rng or np.random.default_rng(0); C = np.asarray(C, float); N = C.sum(); M = flat_fit(C)
    def cellG(Y, F):
        m = Y > 0; out = np.zeros(Y.shape[:2])
        out += np.where(m, Y * np.log(np.where(m, Y, 1) / np.maximum(F, 1e-300)), 0).sum(-1)
        return 2 * out
    Gmap = cellG(C, M); Gobs = float(Gmap.sum()); nulls, nmaps = [], []
    for _ in range(B):
        Cb = flat_null_grid(C, rng, M); Mb = flat_fit(Cb); g = cellG(Cb, Mb); nulls.append(g.sum()); nmaps.append(g)
    nulls = np.array(nulls); nm = np.mean(nmaps, 0)
    U, V, K = C.shape
    return dict(G=Gobs, df=(K - 1) * (U - 1) * (V - 1), nats=Gobs / (2 * N), null_nats=float(nulls.mean() / (2 * N)),
                excess_nats=float((Gobs - nulls.mean()) / (2 * N)), sd_nats=float(nulls.std() / (2 * N)),
                z=float((Gobs - nulls.mean()) / nulls.std()), p=float((1 + (nulls >= Gobs).sum()) / (B + 1)),
                map_excess=((Gmap - nm) / (2 * N)).tolist())


# ------------------------------------------------------------------------------------------- top-order information
def top_info(counts, prior=0.5) -> dict:
    c = np.asarray(counts, float); N = c.sum(); p = smooth(c, prior); n = int(np.log2(len(p)))
    q = T.point_from_margins(p, n)
    d = kl(p, q)
    return dict(D_top=d, D_top_debiased=d - 1 / (2 * N), theta=T.theta(p), coinfo=coinformation(p))


def coinformation(p):
    n = int(np.log2(len(p))); P = np.asarray(p).reshape((2,) * n); tot = 0.0
    for k in range(1, n + 1):
        for S in itertools.combinations(range(n), k):
            m = P.sum(axis=tuple(a for a in range(n) if a not in S)).ravel(); m = m[m > 0]
            tot += (-1) ** (k + 1) * float(-(m * np.log(m)).sum())
    return tot


def top_info_posterior(counts, B=200, rng=None):
    rng = rng or np.random.default_rng(0); c = np.asarray(counts, float); n = int(np.log2(len(c)))
    D_, I_ = [], []
    for P in rng.dirichlet(c + 0.5, size=B):
        q = T.point_from_margins(P, n); D_.append(kl(P, q)); I_.append(coinformation(P))
    D_ = np.array(D_) - 1 / (2 * c.sum())          # a posterior draw carries about one degree of freedom of noise
    return dict(D_lo=float(np.quantile(D_, .05)), D_hi=float(np.quantile(D_, .95)),
                I_lo=float(np.quantile(I_, .05)), I_hi=float(np.quantile(I_, .95)))


# ------------------------------------------------------------------------------------------- nodes
def node_posterior(counts, B=2000, rng=None, prior=0.5) -> dict:
    """posterior (Dirichlet) over which perfect matching the toggle/node picks: the cheapest even cell paired with the
    cheapest odd cell. Returns branch probabilities, the posterior of the partner margin |p_o1 - p_o2| / (p_o1 + p_o2),
    the node radius, and the observed-table margin."""
    rng = rng or np.random.default_rng(0); c = np.asarray(counts, float); n = int(np.log2(len(c)))
    ev, od = D.parity_classes(n); P = rng.dirichlet(c + prior, size=B)
    e = np.array(ev)[np.argmin(P[:, ev], 1)]; o_sorted = np.array(od)[np.argsort(P[:, od], 1)]
    o1, o2 = o_sorted[:, 0], o_sorted[:, 1]
    pairs = {}
    for a, b in zip(e, o1):
        pairs[(int(a), int(b))] = pairs.get((int(a), int(b)), 0) + 1 / B
    po1 = P[np.arange(B), o1]; po2 = P[np.arange(B), o2]; margin = np.abs(po1 - po2) / (po1 + po2)
    obs = D.radii(smooth(c, prior))
    top = max(pairs.values())
    return dict(branch_top=float(top), branch_entropy=float(-sum(v * np.log(v) for v in pairs.values())), n_branches=len(pairs),
                margin_obs=obs["partner_margin"], margin_lo=float(np.quantile(margin, .05)), margin_hi=float(np.quantile(margin, .95)),
                node_radius=obs["node_radius"], pairs={f"{T.cells(n)[a]}-{T.cells(n)[b]}": v for (a, b), v in sorted(pairs.items(), key=lambda kv: -kv[1])})
