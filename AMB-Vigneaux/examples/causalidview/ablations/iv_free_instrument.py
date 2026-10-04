import os, sys, time, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
os.environ.setdefault("CAUSALIDVIEW_DIR", "/tmp/ds/CausalIDView")
import run as R
sig = R._sig


def irls(Z, y, lam, free, iters=60):
    p = Z.shape[1]; b = np.zeros(p); P = np.eye(p) * lam
    for i in free: P[i, i] = 0
    for _ in range(iters):
        mu = sig(Z @ b); W = mu * (1 - mu) + 1e-9
        st = np.linalg.solve((Z * W[:, None]).T @ Z + P + 1e-8 * np.eye(p), Z.T @ (y - mu) - P @ b); b += st
        if np.abs(st).max() < 1e-8: break
    return b


def cvlog(Z, y, free, lams=(1, 3, 10, 30, 100, 300, 1000, 3000), k=5):
    f = np.random.default_rng(0).integers(0, k, len(y)); best = None
    for lam in lams:
        ll = 0.0
        for i in range(k):
            b = irls(Z[f != i], y[f != i], lam, free); p = np.clip(sig(Z[f == i] @ b), 1e-6, 1 - 1e-6)
            ll -= float((y[f == i] * np.log(p) + (1 - y[f == i]) * np.log(1 - p)).sum())
        if best is None or ll < best[0]: best = (ll, lam)
    return irls(Z, y, best[1], free)


def iv_cells(W, free_I, ydrop_I=False):
    c, qy = W["ctx"], W["qry"]; X = W["X"]; T = W["T"][c].astype(float); Y = W["Y"][c].astype(float); I = W["I"][c].astype(float); nq = len(qy)
    one = lambda n: np.ones((n, 1)); Zc = np.hstack([one(len(c)), X[c], I[:, None]]); Zq = [np.hstack([one(nq), X[qy], np.full((nq, 1), j)]) for j in (0, 1)]
    fr = [0, Zc.shape[1] - 1] if free_I else [0]
    bT = cvlog(Zc, T, fr); bY = {t: cvlog(Zc[T == t], Y[T == t], fr) for t in (0, 1)}
    pj = np.zeros((nq, 2, 2, 2))
    for j in (0, 1):
        e = sig(Zq[j] @ bT)
        for t in (0, 1):
            pt = e if t else 1 - e; my = sig(Zq[j] @ bY[t]); pj[:, j, t, 1] = pt * my; pj[:, j, t, 0] = pt * (1 - my)
    return np.array([R.iv_bounds(pj[i]) for i in range(nq)])


res = {}
for seed in [int(s) for s in os.environ.get("SEEDS", "0 1 2 3 4 5 6 7").split()]:
    W = R.world(seed); ivo = np.array([R.iv_bounds(W["q"][i]) for i in range(len(W["qry"]))]); o = {}
    for name, fi in {"penalised I": False, "free I": True}.items():
        e = iv_cells(W, fi); o[name] = (R.ep_rmse(e[:, 0], e[:, 1], ivo[:, 0], ivo[:, 1]), float(np.mean(e[:, 0] - ivo[:, 0])), float(np.mean(e[:, 1] - ivo[:, 1])))
    for k, v in o.items(): res.setdefault(k, []).append(v)
    print(seed, {k: tuple(round(x, 4) for x in v) for k, v in o.items()}, flush=True)
print("mean (rmse, biasL, biasU):", {k: tuple(np.round(np.mean(v, 0), 4)) for k, v in res.items()})
