"""Study 91D (PREREG91D.md): two-sided (Ruiz) scaled conditioning at the 21 events uncertified in 91C, plus 10 reference events."""
import os, json, pickle
import numpy as np, mpmath as mp, lib91 as Q
D = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(D, "study91c_cert"); O = os.path.join(D, "study91d_rescale")
S = Q.Sys(3, 4); M = Q.MPStrand(S); WK = {}
def walker(p, L):
    if (p, L) not in WK:
        pk = pickle.load(open(os.path.join(D, "study91_3x4", "walks", f"walk_9102_{p}_L{L}.pkl"), "rb"))
        W = Q.Walker(S, pk["Fw"], pk["Gw"], pk["L"]); W.T, W.X, W.V = pk["T"], pk["X"], pk["V"]; WK[(p, L)] = W
    return WK[(p, L)]
def ruiz(J):
    for _ in range(30):
        for i in range(J.rows):
            m = mp.sqrt(max(abs(J[i, j]) for j in range(J.cols)))
            for j in range(J.cols): J[i, j] /= m
        for j in range(J.cols):
            m = mp.sqrt(max(abs(J[i, j]) for i in range(J.rows)))
            for i in range(J.rows): J[i, j] /= m
    return J
def evaluate(p, L, t, pair):
    W = walker(p, L); T = np.asarray(W.T); k = int(np.searchsorted(T, t, side="right") - 1); a, b = pair
    y, ok = S.track(W.X[k][[a, b]], W.seg(T[k], t)); al, be = W.prm(t); y, _ = S.newton(y, al, be)
    with mp.workdps(Q.MP_DPS):
        A = [mp.mpc(z.real, z.imag) for z in np.asarray(al, complex)]; B = [mp.mpc(z.real, z.imag) for z in np.asarray(be, complex)]
        pts, res, k2 = [], [], []
        for q in range(2):
            x = Q._mp_vec(y[q])
            for _ in range(12): x = x - mp.lu_solve(M.J(x, A, B), M.F(x, A, B))
            res.append(float(max(abs(z) for z in M.F(x, A, B)))); sv = mp.svd_c(ruiz(M.J(x, A, B)), compute_uv=False); k2.append(float(min(sv) / max(sv))); pts.append(x)
        val = lambda x: S.c * sum(x[i] for i in range(S.r)) + S.r * sum(x[S.r + j] for j in range(S.c - 1))
        va, vb = val(pts[0]), val(pts[1]); s = 1 + float(max(abs(va), abs(vb))); xs = 1 + float(max(max(abs(z) for z in pts[0]), max(abs(z) for z in pts[1])))
        c1 = float(abs(va - vb)) / s; c2 = float(max(abs(pts[0][i] - pts[1][i]) for i in range(S.n))) / xs
    return dict(path=p, L=L, t=t, pair=list(pair), c1=c1, c2=c2, res=res, kappa2=k2, tracked=bool(ok.all()),
                ok=bool(ok.all()) and c1 <= 1e-8 and c2 >= 1e-6 and max(res) <= 1e-12 and min(k2) >= 1e-14)
M1 = json.load(open(os.path.join(C, "cert_mp_amend1.json"))); left = [m for m in M1 if not m["certified"]]
cond = [m for m in left if m["c2"] >= 1e-6 and max(m["res"]) <= 1e-12]; other = [m for m in left if m not in cond]
out = dict(cond=[evaluate(m["path"], m["L"], m["t"], m["pair"]) for m in cond], other=[evaluate(m["path"], m["L"], m["t"], m["pair"]) for m in other])
R = json.load(open(os.path.join(C, "cert_0_L30.json")))["rows"]; ref = [r for r in R if r["certified"]][:10]
out["ref"] = [evaluate(0, 30, r["t"], r["pair"]) for r in ref]
json.dump(out, open(os.path.join(O, "rescale91d.json"), "w"))
lines = [("PASS" if len(out["cond"]) == 19 and all(e["ok"] for e in out["cond"]) else "FAIL") + f" RQ1 two-sided scaled conditioning at the {len(out['cond'])} conditioning events: {sum(e['ok'] for e in out['cond'])}/{len(out['cond'])} meet kappa2 >= 1e-14"]
lines += [f"  t={e['t']:.6f} kappa2 min {min(e['kappa2']):.1e} res {max(e['res']):.1e} c2 {e['c2']:.1e} {'ok' if e['ok'] else 'no'}" for e in out["cond"]]
lines += ["REC  RQ2 other events: " + "; ".join(f"t={e['t']:.6f} kappa2 {min(e['kappa2']):.1e} c2 {e['c2']:.1e} res {max(e['res']):.1e}" for e in out["other"])]
lines += ["REC  RQ2 reference (first 10 certified, path 0 L30): kappa2 min " + ", ".join(f"{min(e['kappa2']):.1e}" for e in out["ref"])]
open(os.path.join(O, "check91d_output.txt"), "w").write("==== Study 91D (PREREG91D), 2026-11-24\n" + "\n".join(lines) + "\n"); print("\n".join(lines))
