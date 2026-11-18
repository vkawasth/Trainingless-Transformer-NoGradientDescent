"""Study 45V: valuations of the six critical points of the Study 45 example over the Novikov field (T = e^{-L}); checks VL1-VL4.
Pre-registered in study45_example/valuation/PREREG_valuation.md, frozen by SHA-256 before this code ran. Uses the certified machinery of
check_example45.py (imported, not copied). Checkpointed per (t, L-segment): study45_example/valuation/ckpt_<t>_<L0>_<L1>.json; a rerun
resumes at the first missing segment. Output: check_valuation45_output.txt.
"""
import os, sys, json, runpy, time
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DV = os.path.join(HERE, "study45_example", "valuation")
_argv = sys.argv; sys.argv = ["check_example45.py", "--lib"]; G = runpy.run_path(os.path.join(HERE, "check_example45.py")); sys.argv = _argv
arb, acb, Q, des, ser, newton, cert_point, contains_box, tball = G["arb"], G["acb"], G["Q"], G["des"], G["ser"], G["newton"], G["cert_point"], G["contains_box"], G["tball"]
V, x0b, s_enc, dwdL, CK = G["V"], G["x0b"], G["s_enc"], G["dwdL"], G["CK"]
OUT = open(os.path.join(HERE, "check_valuation45_output.txt"), "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
def check(name, cond, info=""):
    say(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else "")); return bool(cond)
MARKS = [20, 40, 80, 160, 320]; TS = {"TA": Fr(-1, 10), "T0": Fr(0), "TB": Fr(1, 10)}
USTAR = (Fr(1, 200), Fr(7, 200))

def record(B, K, tq, L):
    """Enclosures of u = -Re(w)/L, the cells x(u), and val(s) = -log|s|/L for one branch at one L (B: tight box, K: Krawczyk image)."""
    Lb = arb(L); u = [-(K[m].real) / Lb for m in range(2)]
    cells = [x0b(k, Q(tq)) + V[k][0] * u[0] + V[k][1] * u[1] for k in range(6)]
    s = s_enc(K, Q(tq), Lb); vs = -abs(s).log() / Lb
    inside = all(c > 0 for c in cells); outside = any(c < 0 for c in cells)
    return dict(u=[ser(x) for x in u], cells=[ser(c) for c in cells], s=ser(s), val_s=ser(vs), inside=bool(inside), outside=bool(outside))

def segment(name, tq, L0, L1, ws, Bprev):
    p = os.path.join(DV, f"ckpt_{name}_{L0}_{L1}.json")
    if os.path.exists(p):
        d = json.load(open(p)); return d, [np.array([complex(*w[0]), complex(*w[1])]) for w in d["w_end"]], [[des(z) for z in b] for b in d["B_end"]]
    tm = float(tq); tbq = Q(tq); Lc = float(L0); h = 0.05; nst = 0; t0 = time.time()
    while Lc < L1:
        Lb_ = min(Lc + h, float(L1)); Lm = (Lc + Lb_) / 2; LL = tball(Lc, Lb_); out = []; fail = False
        for i, w in enumerate(ws):
            wm = newton(w + dwdL(w, tm, Lc) * (Lm - Lc), tm, Lm)
            if wm is None: fail = True; break
            sp = np.abs(dwdL(wm, tm, Lm)).max() * (Lb_ - Lc) / 2; sc = 1 + np.abs(wm).max()
            c = cert_point(wm, tbq, tm, L=LL, Lmid=Lm, rads=tuple(r * sc for r in (1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2) if r * sc > 1.5 * sp)[:4])
            if c is None or not contains_box(c[1], Bprev[i]): fail = True; break
            we = newton(wm, tm, Lb_); ce = None if we is None else cert_point(we, tbq, tm, L=arb(Lb_), Lmid=Lb_, rads=(1e-12 * sc, 1e-10 * sc, 1e-8 * sc))
            if ce is None or not contains_box(c[1], ce[1]): fail = True; break
            out.append((we, ce))
        if not fail:                                                            # distinct points of the torus on the whole step
            fail = not all(G["distinct_mod"](out[i][1][1], out[j][1][1]) for i in range(6) for j in range(i + 1, 6))
        if fail:
            h /= 2
            if h < 1e-9: raise RuntimeError(f"L continuation failed at t = {tq}, L = {Lc}")
            continue
        ws = [o[0] for o in out]; Bprev = [o[1][1] for o in out]; Lc = Lb_; h = min(h * 1.5, 2.0); nst += 1
    rec = [record(o[1][1], o[1][2], tq, L1) for o in out]
    d = dict(t=str(tq), L0=L0, L1=L1, steps=nst, seconds=round(time.time() - t0), rec=rec,
             w_end=[[[w[0].real, w[0].imag], [w[1].real, w[1].imag]] for w in ws], B_end=[[ser(z) for z in b] for b in Bprev])
    json.dump(d, open(p, "w")); say(f"  [{name} L {L0}->{L1}] {nst} steps, {d['seconds']}s, inside {[r['inside'] for r in rec]}, outside {[r['outside'] for r in rec]}")
    return d, ws, Bprev

def run_t(name):
    s3 = json.load(open(os.path.join(CK, "S3_points.json")))[name]; tq = TS[name]
    ws = [np.array([complex(*w[0]), complex(*w[1])]) for w in s3["w"]]; K = [[des(k) for k in KK] for KK in s3["K"]]
    B = [[G["box"](complex(*w[0]), r), G["box"](complex(*w[1]), r)] for w, r in zip(s3["w"], s3["rad"])]
    recs = {20: [record(B[i], K[i], tq, 20) for i in range(6)]}
    marks = MARKS if name != "TB" else [20, 40, 80]                            # see NOTES_valuation.md (t = 1/10 stopped at L = 80)
    for L0, L1 in zip(marks, marks[1:]):
        d, ws, B = segment(name, tq, L0, L1, ws, B); recs[L1] = d["rec"]
    return recs

if __name__ == "__main__":
    say(f"==== run {time.strftime('%Y-%m-%d %H:%M:%S')}")
    R = {n: run_t(n) for n in ("T0", "TA", "TB")}
    lead, far = [0, 1, 2, 3], [4, 5]; r0 = R["T0"]
    vl1 = all(r0[L][i]["inside"] for L in MARKS if L >= 40 for i in lead)
    dist = [max(float((des(r0[320][i]["u"][m]) - Q(USTAR[m])).abs_upper()) for m in range(2)) for i in lead]
    check("VL1 HYPOTHESIS t = 0: leading-group valuations inside the fibre for L >= 40, within 0.01 of the leximin point at L = 320",
          vl1 and max(dist) < 0.01, f"inside by L: {[(L, [r0[L][i]['inside'] for i in lead]) for L in MARKS]}; distance to u* at L = 320: {[round(d, 5) for d in dist]}")
    check("VL2 HYPOTHESIS t = 0: far valuations outside the fibre at every recorded L", all(r0[L][i]["outside"] for L in MARKS for i in far),
          f"outside by L: {[(L, [r0[L][i]['outside'] for i in far]) for L in MARKS]}; u at L = 320: {[[round(float(des(x).mid()), 4) for x in r0[320][i]['u']] for i in far]}")
    vsl = [float(des(r0[320][i]["val_s"]).mid()) for i in lead]; vsf = [float(des(r0[320][i]["val_s"]).upper()) for i in far]
    check("VL3 HYPOTHESIS t = 0: val(s) -> 1/10 for the leading group (within 0.01 at L = 320), val(s) < 0 for the far values",
          all(abs(v - 0.1) < 0.01 and float((des(r0[320][i]['val_s']) - arb(1) / 10).abs_upper()) < 0.01 for v, i in zip(vsl, lead)) and all(v < 0 for v in vsf),
          f"leading {[round(v, 5) for v in vsl]}; far (upper) {[round(v, 4) for v in vsf]}")
    info = []
    for n in ("TA", "TB"):
        r = R[n]; st = lambda d: "in" if d["inside"] else ("out" if d["outside"] else "?")
        Lmax = max(r)
        info.append(f"{n} (L up to {Lmax}): " + "; ".join(f"branch {i}: {[st(r[L][i]) for L in sorted(r)]}, u({Lmax}) {[round(float(des(x).mid()), 4) for x in r[Lmax][i]['u']]}, val(s) {round(float(des(r[Lmax][i]['val_s']).mid()), 4)}" for i in range(6))
                    + f"; inside at L = {Lmax}: {sum(r[Lmax][i]['inside'] for i in range(6))}")
    check("VL4 RECORD t = -1/10 and 1/10: valuations by branch", True, " || ".join(info))
    json.dump({n: {str(L): v for L, v in R[n].items()} for n in R}, open(os.path.join(DV, "valuations.json"), "w"), indent=1)
