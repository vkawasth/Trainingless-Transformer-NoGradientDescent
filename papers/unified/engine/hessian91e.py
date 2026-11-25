"""Study 91E (PREREG91E.md): intrinsic nondegeneracy (log-coordinate Hessian of W) at the 19 conditioning events of 91C/91D."""
import os, json, pickle
import numpy as np, mpmath as mp, lib91 as Q
D = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(D, "study91e_hessian"); S = Q.Sys(3, 4); MS = Q.MPStrand(S)
r, c = 3, 4; B = [(i, j) for i in range(r - 1) for j in range(c - 1)]; m = {}
for (i, j) in B: e = np.zeros(6, int); e[B.index((i, j))] = 1; m[(i, j)] = e
for i in range(r - 1): m[(i, c - 1)] = -sum(m[(i, j)] for j in range(c - 1))
for j in range(c): m[(r - 1, j)] = -sum(m[(i, j)] for i in range(r - 1))
WK = {}
def walker(p, L):
    if (p, L) not in WK:
        pk = pickle.load(open(os.path.join(D, "study91_3x4", "walks", f"walk_9102_{p}_L{L}.pkl"), "rb"))
        W = Q.Walker(S, pk["Fw"], pk["Gw"], pk["L"]); W.T, W.X, W.V = pk["T"], pk["X"], pk["V"]; WK[(p, L)] = W
    return WK[(p, L)]
def evaluate(e):
    W = walker(e["path"], e["L"]); T = np.asarray(W.T); t = e["t"]; k = int(np.searchsorted(T, t, side="right") - 1)
    y, ok = S.track(W.X[k][list(e["pair"])], W.seg(T[k], t)); al, be = W.prm(t); y, _ = S.newton(y, al, be); kH, res = [], []
    with mp.workdps(Q.MP_DPS):
        A = [mp.mpc(z.real, z.imag) for z in np.asarray(al, complex)]; Bb = [mp.mpc(z.real, z.imag) for z in np.asarray(be, complex)]
        for q in range(2):
            x = Q._mp_vec(y[q])
            for _ in range(12): x = x - mp.lu_solve(MS.J(x, A, Bb), MS.F(x, A, Bb))
            res.append(float(max(abs(z) for z in MS.F(x, A, Bb)))); u = [x[i] for i in range(r)]; v = [x[r + j] for j in range(c - 1)] + [mp.mpf(0)]
            H = mp.matrix(6, 6)
            for i in range(r):
                for j in range(c):
                    pij = u[i] + v[j]; mm = m[(i, j)]
                    for a in range(6):
                        for b in range(6): H[a, b] += pij * int(mm[a]) * int(mm[b])
            sv = mp.svd_c(H, compute_uv=False); kH.append(float(min(sv) / max(sv)))
    return dict(path=e["path"], L=e["L"], t=t, kappaH=kH, res=res, c1=e["c1"], c2=e["c2"],
                ok=bool(ok.all()) and e["c1"] <= 1e-8 and e["c2"] >= 1e-6 and max(res) <= 1e-12 and min(kH) >= 1e-14)
R = json.load(open(os.path.join(D, "study91d_rescale", "rescale91d.json")))
out = {k: [evaluate(e) for e in R[k]] for k in ("cond", "other", "ref")}
json.dump(out, open(os.path.join(O, "hessian91e.json"), "w"))
L1 = [("PASS" if len(out["cond"]) == 19 and all(e["ok"] for e in out["cond"]) else "FAIL") + f" LH1 intrinsic nondegeneracy at the 19 conditioning events: {sum(e['ok'] for e in out['cond'])}/19 with kappaH >= 1e-14"]
L1 += [f"  t={e['t']:.6f} kappaH min {min(e['kappaH']):.1e} res {max(e['res']):.1e} {'ok' if e['ok'] else 'no'}" for e in out["cond"]]
L1 += ["REC  LH2 reference kappaH min: " + ", ".join(f"{min(e['kappaH']):.1e}" for e in out["ref"]), "REC  LH2 other events kappaH min: " + ", ".join(f"t={e['t']:.6f} {min(e['kappaH']):.1e}" for e in out["other"])]
open(os.path.join(O, "check91e_output.txt"), "w").write("==== Study 91E (PREREG91E), 2026-11-24\n" + "\n".join(L1) + "\n"); print("\n".join(L1))
