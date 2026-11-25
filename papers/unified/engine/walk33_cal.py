"""CALIBRATION (Study 92 Part F): 3x3 walker on seed 9201, path 0, L = 20. Exploratory; not a check."""
import os, sys, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np, lib91 as Q, lib92 as G
S = Q.Sys(3, 3); p = int(sys.argv[1]) if len(sys.argv) > 1 else 0; L = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
Fw, Gw = G.gen33(9201)[p]; t0 = time.time()
x0, r0, n0, ok0 = G.start_solutions33(S, Fw[0], Gw[0], L); print("start", len(x0), "max res %.1e" % np.max(r0), "ok", ok0.all(), f"{time.time()-t0:.0f}s", flush=True)
W = Q.Walker(S, Fw, Gw, L).run(x0); print("walk done", f"{time.time()-t0:.0f}s", "detours", sum(W.det), flush=True)
ev = G.analyse_fine(W); ta = [k + t for k in range(13) for t, _, _ in Q.ties_on(Fw[k], Gw[k], Fw[k + 1], Gw[k + 1])]
kinds = {}; [kinds.__setitem__(e["kind"], kinds.get(e["kind"], 0) + 1) for e in ev]
col = [e for e in ev if e["kind"] == "collision"]; ontie = [e for e in col if min([abs(e["t"] - u) for u in ta] + [9]) < 1e-6]
print("events", kinds, "ties", len(ta), "collisions on ties", len(ontie), "off", len(col) - len(ontie))
print("per tie:", [sum(abs(e["t"] - u) < 1e-6 for e in col) for u in sorted(ta)])
print("real count range", int(Q.isreal(W.V).sum(1).min()), int(Q.isreal(W.V).sum(1).max()), f"{time.time()-t0:.0f}s")
pickle.dump(dict(T=W.T, X=W.X, V=W.V, det=W.det, ev=ev, Fw=Fw, Gw=Gw, L=L), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"walk33_cal_{p}_L{int(L)}.pkl"), "wb"))
