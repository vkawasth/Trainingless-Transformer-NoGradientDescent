"""Study 91 AMEND3: resume the check after QF4. Usage: python3 check91_resume.py qf5 <path> <caustic|off>  |  python3 check91_resume.py final
QF5/QF6/R7 code copied verbatim from check91.py; QF5 split into six jobs (AMEND3)."""
import os, sys, glob, pickle, json, time
import numpy as np, lib91 as Q
from scipy.optimize import linprog
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "study91_3x4"); J5 = os.path.join(D, "qf5"); os.makedirs(J5, exist_ok=True)
S = Q.Sys(3, 4)
WK = {(pk["path"], pk["L"]): pk for pk in (pickle.load(open(f, "rb")) for f in sorted(glob.glob(os.path.join(D, "walks", "walk_9102_*.pkl"))))}
def walker(pk):
    W = Q.Walker(S, pk["Fw"], pk["Gw"], pk["L"]); W.T, W.X, W.V = pk["T"], pk["X"], pk["V"]; return W
def ties_all(pk): return [(k + t, nm, p) for k in range(13) for t, nm, p in Q.ties_on(pk["Fw"][k], pk["Gw"][k], pk["Fw"][k + 1], pk["Gw"][k + 1])]
if sys.argv[1] == "qf5":
    p, kind = int(sys.argv[2]), sys.argv[3]; t0 = time.time()
    pk = WK.get((p, 20.0)); W = walker(pk); ta = ties_all(pk); evs = pk["ev"]; ts = sorted(e["t"] for e in evs)
    iso = lambda e: sum(abs(u - e["t"]) < 2e-3 for u in ts) == 1
    cau = [e for e in evs if e["kind"] == "caustic" and iso(e)][:5]
    off = [e for e in evs if e["kind"] == "collision" and iso(e) and min([abs(e["t"] - t) for t, _, _ in ta] + [9]) > 1e-3][:5]
    out = []; ok = True
    if kind == "caustic":
        for e in cau:
            try: w, perm = Q.loop_caustic(W, e["t"], e["pair"]); a_, b_ = e["pair"]; good = abs(abs(w) - 3 * np.pi) < 0.05 * np.pi and perm[a_] == b_ and perm[b_] == a_ and all(perm[q] == q for q in range(96) if q not in (a_, b_))
            except RuntimeError: w, good = float("nan"), False
            out.append(float(round(float(w) / np.pi, 3))); ok &= bool(good); print(e["t"], out[-1], good, flush=True)
        n = len(cau)
    else:
        for e in off:
            try: w, perm = Q.loop_caustic(W, e["t"], e["pair"]); good = abs(abs(w) - 2 * np.pi) < 0.05 * np.pi and all(perm[q] == q for q in range(96))
            except RuntimeError: w, good = float("nan"), False
            out.append(float(round(float(w) / np.pi, 3))); ok &= bool(good); print(e["t"], out[-1], good, flush=True)
        n = len(off)
    json.dump(dict(path=p, kind=kind, w=out, ok=bool(ok), n=n, seconds=time.time() - t0), open(os.path.join(J5, f"qf5_{p}_{kind}.json"), "w"))
    sys.exit(0)
# ---------------------------------------------------------------- final: combine QF5, then QF6 and R7 (verbatim)
OUT = open(os.path.join(D, "check91_output.txt"), "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
def check(name, cond, info=""): say(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else "")); return bool(cond)
say(f"==== Study 91 check resumed under AMEND3 {time.strftime('%Y-%m-%d %H:%M:%S')} (QF1-QF4 above stand from invocation 1)")
ok5 = True; info5 = []
for p in range(3):
    jc, jo = (json.load(open(os.path.join(J5, f"qf5_{p}_{k}.json"))) for k in ("caustic", "off"))
    ok5 &= jc["ok"] and jo["ok"] and jc["n"] >= 1 and jo["n"] >= 1; info5.append(f"path {p}: caustic windings/pi {jc['w']}; off-tie collision windings/pi {jo['w']}")
check("QF5 local braids: caustic loops exchange the pair with winding 3 pi (others fixed); off-tie collision loops are pure with winding 2 pi", ok5, "; ".join(info5))
def counts(p, n=120):
    x = np.asarray(p) * n; f = np.floor(x).astype(int); k = n - f.sum(); f[np.argsort(-(x - f))[:k]] += 1; return f
ok6 = True; nweeks = 0
for p in range(3):
    Fw, Gw = Q.gen(9102)[p]
    for k in range(14):
        F, G = counts(Fw[k]), counts(Gw[k]); A = np.zeros((7, 12)); b = np.concatenate([F, G])
        for i in range(3):
            for j in range(4): A[i, 4 * i + j] = 1; A[3 + j, 4 * i + j] = 1
        cobj = np.zeros(12); cobj[3] = 1; lo = linprog(cobj, A_eq=A, b_eq=b, bounds=(0, None), method="highs-ds"); hi = linprog(-cobj, A_eq=A, b_eq=b, bounds=(0, None), method="highs-ds")
        fl, fh = max(0, F[0] + G[3] - 120), min(F[0], G[3]); integ = np.allclose(lo.x, np.round(lo.x)) and np.allclose(hi.x, np.round(hi.x))
        ok6 &= abs(lo.fun - fl) < 1e-9 and abs(-hi.fun - fh) < 1e-9 and integ; nweeks += 1
check("QF6 decision band: LP band = Frechet band, integral optimal vertices, every integer week of the held-out paths", ok6, f"{nweeks} weeks")
for (p, L), pk in sorted(WK.items()):
    W = walker(pk); s0 = 2 * np.abs(pk["V"]).max() * np.exp(0.3j); st = Q.stokes(W, s0)
    say(f"R7 RECORD path {p} L{L:.0f}: collisions {sum(e['kind'] == 'collision' for e in pk['ev'])}, caustics {sum(e['kind'] == 'caustic' for e in pk['ev'])}, "
        f"other {sum(e['kind'] not in ('collision', 'caustic') for e in pk['ev'])}, Stokes events {len(st)}, detours {sum(pk['det'])}, {pk['seconds']:.0f}s")
say("==== done")
