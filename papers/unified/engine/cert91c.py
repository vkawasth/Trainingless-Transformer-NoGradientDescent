"""Study 91C (PREREG91C.md): certify the hypotheses of prop:disjoint at every Study 91 collision. Usage: python3 cert91c.py <path> <L>"""
import os, sys, pickle, json, time
import numpy as np, lib91 as Q
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "study91c_cert")
p, L = int(sys.argv[1]), int(sys.argv[2]); pk = pickle.load(open(os.path.join(D, "study91_3x4", "walks", f"walk_9102_{p}_L{L}.pkl"), "rb"))
S = Q.Sys(3, 4); W = Q.Walker(S, pk["Fw"], pk["Gw"], pk["L"]); W.T, W.X, W.V = pk["T"], pk["X"], pk["V"]; T = np.asarray(W.T)
ta = [k + t for k in range(13) for t, _, _ in Q.ties_on(pk["Fw"][k], pk["Gw"][k], pk["Fw"][k + 1], pk["Gw"][k + 1])]
def cond_eq(x):
    J = S.J(x[None].astype(Q.LD))[0]; J = J / np.abs(J).max(1, keepdims=True); sv = np.linalg.svd(J.astype(complex), compute_uv=False); return float(sv.min() / sv.max())
rows = []; t0 = time.time()
for e in pk["ev"]:
    if e["kind"] != "collision": continue
    t = float(e["t"]); k = int(np.searchsorted(T, t, side="right") - 1); a, b = e["pair"]
    y, ok = S.track(W.X[k][[a, b]], W.seg(T[k], t)); al, be = W.prm(t); y, res = S.newton(y, al, be)
    v = S.value(y); s = 1 + float(np.abs(v).max()); xs = 1 + float(np.abs(y).max())
    c1 = float(abs(v[0] - v[1])) / s; c2 = float(np.abs(y[0] - y[1]).max()) / xs; rs = [float(r) for r in res]; cn = [cond_eq(y[0]), cond_eq(y[1])]
    h = 1e-7; yp, okp = S.track(y, W.seg(t, t + h)); ym, okm = S.track(y, W.seg(t, t - h)); vp, vm = S.value(yp), S.value(ym)
    der = float(abs((vp[0] - vp[1]) - (vm[0] - vm[1])) / (2 * h)) / s
    cert = bool(ok.all()) and c1 <= 1e-8 and c2 >= 1e-6 and max(rs) <= 1e-12 and min(cn) >= 1e-14; simple = cert and bool(okp.all() and okm.all()) and der >= 1e-8
    tie = min([abs(t - u) for u in ta] + [9.0]) < 1e-6
    rows.append(dict(t=t, pair=[int(a), int(b)], tie=bool(tie), c1=c1, c2=c2, res=rs, cond=cn, der=der, tracked=bool(ok.all()), certified=cert, simple=simple))
json.dump(dict(path=p, L=L, rows=rows, seconds=time.time() - t0), open(os.path.join(OUT, f"cert_{p}_L{L}.json"), "w"))
n = len(rows); print(p, L, "collisions", n, "certified", sum(r["certified"] for r in rows), "simple", sum(r["simple"] for r in rows), f"{time.time() - t0:.0f}s")
