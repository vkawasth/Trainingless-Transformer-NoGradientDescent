import os, sys, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
os.environ.setdefault("CAUSALIDVIEW_DIR", "/tmp/ds/CausalIDView")
import run as R
LAMS = np.logspace(-1, 4, 21)


def krr_multi(K, Ymat, Kq):
    """KRR for several targets sharing one kernel; ridge chosen by the closed-form LOO summed over targets."""
    m = Ymat.mean(0); Yc = Ymat - m; ev, V = np.linalg.eigh(K); ev = np.maximum(ev, 0); Yt = V.T @ Yc; best = None
    for lam in LAMS:
        d = ev / (ev + lam); H = (V ** 2) @ d; loo = (Yc - V @ (d[:, None] * Yt)) / (1 - H)[:, None]; e = float(np.mean(loo ** 2))
        if best is None or e < best[0]: best = (e, lam)
    A = V @ (Yt / (ev + best[1])[:, None]); return best[0], m + Kq @ A


def ivkrr(W, ws=(0.0, 1.0, 4.0), wi=(0.0, 0.2)):
    c, qy = W["ctx"], W["qry"]; X = W["X"]; T = W["T"][c]; Y = W["Y"][c]; I = W["I"][c].astype(float); nq = len(qy)
    Yc = np.eye(4)[2 * T + Y]; G = X[c] @ X[c].T; Gq = X[qy] @ X[c].T; same = (I[:, None] == I[None, :]).astype(float)
    best = None
    for w in ws:
        for v in wi:   # v: weight of an arm-indicator (intercept-shift) kernel, keeps arm means unpenalised-ish
            K = G * (1 + w * same) + v * len(c) * same
            e, _ = krr_multi(K, Yc, np.zeros((1, len(c))))
            if best is None or e < best[0]: best = (e, w, v)
    _, w, v = best; K = G * (1 + w * same) + v * len(c) * same; pj = np.zeros((nq, 2, 2, 2))
    for j in (0, 1):
        sq = (I[None, :] == j).astype(float); Kq = Gq * (1 + w * sq) + v * len(c) * sq
        _, P = krr_multi(K, Yc, Kq); P = np.clip(P, 1e-4, None); P /= P.sum(1, keepdims=True); pj[:, j] = P.reshape(-1, 2, 2)
    return np.array([R.iv_bounds(pj[i]) for i in range(nq)]), (w, v)


if __name__ == "__main__":
    res = []
    for seed in [int(s) for s in os.environ.get("SEEDS", "0 1 2 3 4 5 6 7").split()]:
        W = R.world(seed); ivo = np.array([R.iv_bounds(W["q"][i]) for i in range(len(W["qry"]))])
        e, wv = ivkrr(W); r = (R.ep_rmse(e[:, 0], e[:, 1], ivo[:, 0], ivo[:, 1]), float(np.mean(e[:, 0] - ivo[:, 0])), float(np.mean(e[:, 1] - ivo[:, 1])), float(np.mean(e[:, 2] > 1e-7)))
        res.append(r); print(seed, "w,v", wv, "rmse,biasL,biasU,infeas", np.round(r, 4), flush=True)
    print("mean", np.round(np.mean(res, 0), 4))
