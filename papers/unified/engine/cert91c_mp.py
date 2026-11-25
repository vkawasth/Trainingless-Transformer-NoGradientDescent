"""Study 91C AMEND1: 50-digit recheck of the events not certified in run 1. Usage: python3 cert91c_mp.py"""
import os, glob, json, pickle, time
import numpy as np, mpmath as mp, lib91 as Q
D = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(D, "study91c_cert"); S = Q.Sys(3, 4); M = Q.MPStrand(S); out = []
for f in sorted(glob.glob(os.path.join(C, "cert_*_L*.json"))):
    R = json.load(open(f)); p, L = R["path"], R["L"]; pk = pickle.load(open(os.path.join(D, "study91_3x4", "walks", f"walk_9102_{p}_L{L}.pkl"), "rb"))
    W = Q.Walker(S, pk["Fw"], pk["Gw"], pk["L"]); W.T, W.X, W.V = pk["T"], pk["X"], pk["V"]; T = np.asarray(W.T)
    for r in R["rows"]:
        if r["certified"]: continue
        t = r["t"]; k = int(np.searchsorted(T, t, side="right") - 1); a, b = r["pair"]
        y, ok = S.track(W.X[k][[a, b]], W.seg(T[k], t)); al, be = W.prm(t); y, _ = S.newton(y, al, be)
        with mp.workdps(Q.MP_DPS):
            A = [mp.mpc(z.real, z.imag) for z in np.asarray(al, complex)]; B = [mp.mpc(z.real, z.imag) for z in np.asarray(be, complex)]
            pts, res, cnd = [], [], []
            for q in range(2):
                x = Q._mp_vec(y[q])
                for _ in range(12): x = x - mp.lu_solve(M.J(x, A, B), M.F(x, A, B))
                res.append(float(max(abs(z) for z in M.F(x, A, B)))); J = M.J(x, A, B)
                for i in range(J.rows):
                    m = max(abs(J[i, j]) for j in range(J.cols))
                    for j in range(J.cols): J[i, j] /= m
                sv = mp.svd_c(J, compute_uv=False); cnd.append(float(min(sv) / max(sv))); pts.append(x)
            val = lambda x: S.c * sum(x[i] for i in range(S.r)) + S.r * sum(x[S.r + j] for j in range(S.c - 1))
            va, vb = val(pts[0]), val(pts[1]); s = 1 + float(max(abs(va), abs(vb))); xs = 1 + float(max(max(abs(z) for z in pts[0]), max(abs(z) for z in pts[1])))
            c1 = float(abs(va - vb)) / s; c2 = float(max(abs(pts[0][i] - pts[1][i]) for i in range(S.n))) / xs
        cert = bool(ok.all()) and c1 <= 1e-8 and c2 >= 1e-6 and max(res) <= 1e-12 and min(cnd) >= 1e-14
        simple = cert and r["der"] >= 1e-8
        out.append(dict(path=p, L=L, t=t, pair=r["pair"], tie=r["tie"], c1=c1, c2=c2, res=res, cond=cnd, der=r["der"], certified=cert, simple=simple))
        print(p, L, round(t, 6), cert, simple, "res %.1e cond %.1e" % (max(res), min(cnd)), flush=True)
json.dump(out, open(os.path.join(C, "cert_mp_amend1.json"), "w"))
print("rechecked", len(out), "certified", sum(o["certified"] for o in out), "simple", sum(o["simple"] for o in out))
