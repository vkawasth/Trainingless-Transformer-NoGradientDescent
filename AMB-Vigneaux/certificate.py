"""Robustness certificate for a source x topic gluing obstruction ("the topology is not tearing").

Data: units with a source s, a topic t, a value y and a cluster id (the resampling unit).
Complex: the bipartite source-topic graph, one edge per cell (s,t) with >= min_n units; edge value b_e = cell mean,
weight w_e = n_e / sigma^2 (pooled within-cell variance). No 2-cells: every cycle is a potential loop.

Hodge split. b = X beta + h, X the (unsigned) incidence e=(s,t) -> u_s + v_t; beta by weighted least squares;
h = harmonic part (the loop residual). beta_1 = E - V + beta_0 = dim of the harmonic space.
  Q = sum_e w_e h_e^2   ~ chi^2(beta_1) under "no loops" (gluing).
Spectral statement. L0 = X^T W X (signless Laplacian = Laplacian for a bipartite graph, after flipping the sign of
topic nodes). Its kernel has dimension beta_0 (connected components). If a perturbation dL satisfies
||dL||_2 < lambda_2(L0) (Fiedler value) then, by Weyl, lambda_2(L0 + dL) > 0: the cover stays connected and beta_1,
the harmonic space and hence the loop coordinates are unchanged. The certificate compares lambda_2 with the 95%
quantile of ||L0* - L0||_2 over a cluster bootstrap (margin > 1 = certified).
Obstruction stability: SNR = ||h||_W / rms_boot ||h* - h||_W, median W-cosine between h and h*.
Cover refinement: split each topic's clusters at random into two halves (a refinement of the cover). A class
of the original cover pulls back to the refinement, so a real obstruction must stay significant (with the same data);
report the median p and the fraction of refinements with p < 0.05. Leave-one-source-out localises the obstruction.

Descent (discrete form of the homotopy-limit argument): a map of presheaves that is a bijection on every context and
every overlap is a bijection on glued sections, because glued sections are a limit. Contrapositive: any global change
is visible on some context or overlap -- which is what the tests above localise.
"""
from __future__ import annotations
import collections
import numpy as np
from scipy import stats


def cells(src, top, y, min_n=5):
    acc = collections.defaultdict(list)
    for s, t, v in zip(src, top, y):
        acc[(s, t)].append(v)
    return {k: (float(np.mean(v)), float(np.var(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) for k, v in acc.items() if len(v) >= min_n}


def _design(edges, S, T):
    X = np.zeros((len(edges), len(S) + len(T)))
    for e, (s, t) in enumerate(edges):
        X[e, S.index(s)] = 1; X[e, len(S) + T.index(t)] = 1
    return X


def hodge(C, S=None, T=None, sigma2=None, edges=None):
    edges = edges or sorted(C)
    S = S or sorted({s for s, _ in edges}); T = T or sorted({t for _, t in edges})
    X = _design(edges, S, T)
    b = np.array([C[e][0] if e in C else 0.0 for e in edges])
    n = np.array([C[e][2] if e in C else 0 for e in edges], float)
    if sigma2 is None:
        num = sum((C[e][2] - 1) * C[e][1] for e in edges if e in C); den = sum(C[e][2] - 1 for e in edges if e in C)
        sigma2 = max(num / max(den, 1), 1e-12)
    w = n / sigma2
    L0 = X.T @ (w[:, None] * X)
    beta = np.linalg.pinv(L0) @ (X.T @ (w * b))
    h = b - X @ beta
    ev = np.sort(np.linalg.eigvalsh(L0))
    b0 = int((ev < 1e-9 * max(ev.max(), 1e-300)).sum())
    beta1 = int((n > 0).sum()) - X.shape[1] + b0
    Q = float((w * h ** 2).sum())
    return dict(edges=edges, S=S, T=T, X=X, b=b, w=w, h=h, L0=L0, eig=ev, beta0=b0, beta1=beta1, Q=Q,
                p=float(stats.chi2.sf(Q, beta1)) if beta1 > 0 else 1.0, sigma2=sigma2,
                fiedler=float(ev[1]) if len(ev) > 1 else 0.0)


def _resample(units, rng):
    by = collections.defaultdict(list)
    for u in units:
        by[u[3]].append(u)
    keys = list(by)
    return [u for i in rng.integers(len(keys), size=len(keys)) for u in by[keys[i]]]


def certificate(units, min_n=5, B=300, n_refine=50, seed=0):
    """units: list of (source, topic, y, cluster)."""
    rng = np.random.default_rng(seed)
    src, top, y, _ = zip(*units)
    C = cells(src, top, y, min_n)
    H = hodge(C)
    dL, lam2b, dh, cos = [], [], [], []
    for _ in range(B):
        U = _resample(units, rng); s_, t_, y_, _ = zip(*U)
        Cb = cells(s_, t_, y_, 1)
        Hb = hodge(Cb, H["S"], H["T"], sigma2=H["sigma2"], edges=H["edges"])
        dL.append(np.linalg.norm(Hb["L0"] - H["L0"], 2)); lam2b.append(Hb["fiedler"])
        m = Hb["w"] > 0
        d = Hb["h"][m] - H["h"][m]; dh.append(float((H["w"][m] * d ** 2).sum()))
        a, c = H["h"][m], Hb["h"][m]; ww = H["w"][m]
        cos.append(float((ww * a * c).sum() / np.sqrt((ww * a * a).sum() * (ww * c * c).sum() + 1e-300)))
    q95 = float(np.quantile(dL, 0.95))
    out = dict(n_units=len(units), sources=H["S"], topics=H["T"], edges=len(H["edges"]), beta0=H["beta0"], beta1=H["beta1"],
               Q=H["Q"], p=H["p"], fiedler=H["fiedler"], dL_q95=q95, margin=H["fiedler"] / q95 if q95 > 0 else np.inf,
               frac_boot_connected=float(np.mean(np.array(lam2b) > 1e-9 * H["eig"].max())),
               snr=float(np.sqrt(H["Q"] / np.mean(dh))), median_cos=float(np.median(cos)))
    # refinement: split each topic's clusters into two random halves
    ps = []
    for _ in range(n_refine):
        lab = {}
        for t in set(top):
            cl = sorted({u[3] for u in units if u[1] == t}, key=str); rng.shuffle(cl)
            for i, c in enumerate(cl):
                lab[(t, c)] = f"{t}|{i % 2}"
        R = cells(src, [lab[(u[1], u[3])] for u in units], y, min_n)
        ps.append(hodge(R)["p"])
    out.update(refine_median_p=float(np.median(ps)), refine_frac_sig=float(np.mean(np.array(ps) < 0.05)))
    # leave one source out
    loo = {}
    for s in H["S"]:
        C2 = {k: v for k, v in C.items() if k[0] != s}
        if len({k[0] for k in C2}) >= 2 and len({k[1] for k in C2}) >= 2:
            H2 = hodge(C2); loo[str(s)] = dict(Q=H2["Q"], df=H2["beta1"], p=H2["p"])
    out["leave_one_source_out"] = loo
    return out
