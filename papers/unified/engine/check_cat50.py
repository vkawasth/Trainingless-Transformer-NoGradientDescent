"""Study 50: the directed Fukaya-Seidel category of the 2x2 potential W = A w + B/w, computed from curves (KQ1-KQ7).
Pre-registered in study50_cat22/PREREG_study50.md, frozen by SHA-256 before this code ran.
Output: check_cat50_output.txt, study50_cat22/result.json. Double precision; tolerances as pre-registered.
"""
import os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "study50_cat22")
OUT = open(os.path.join(HERE, "check_cat50_output.txt"), "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
def check(name, cond, info=""):
    say(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else "")); return bool(cond)

C1 = 0.3
def AB(r1, L, c1=C1):
    c2 = 1 - c1
    return 1 + np.exp(L * (r1 - c2)), np.exp(-L * r1) + np.exp(-L * c1)
def crit(A, B):
    v = 2 * np.sqrt(A * B); return [v, -v], [np.sqrt(B / A), -np.sqrt(B / A)]

def cont_sqrt(z):
    """continuous branch of sqrt along a sampled path z (starting at the principal branch)"""
    r = np.sqrt(z.astype(complex)); out = r.copy()
    for k in range(1, len(r)):
        if abs(out[k] - out[k - 1]) > abs(-r[k] - out[k - 1]) and abs(r[k] - out[k - 1]) > abs(-r[k] - out[k - 1]): out[k] = -r[k]
        else: out[k] = r[k] if abs(r[k] - out[k - 1]) <= abs(-r[k] - out[k - 1]) else -r[k]
    return out

def arc(v, s0, A, B, N=40001):
    """thimble over the segment s0 -> v, parameterised by u in [-1,1], s = v + u^2 (s0 - v)"""
    d = s0 - v; u = np.linspace(-1, 1, N)
    tg = np.linspace(0, 1, (N + 1) // 2)                       # t = u^2 grid
    hg = cont_sqrt(d * (2 * v + tg * d))
    t = u ** 2; h = np.interp(t, tg, hg.real) + 1j * np.interp(t, tg, hg.imag)
    # exact h at the sample points, aligned with the continuous branch
    hx = np.sqrt((d * (2 * v + t * d)).astype(complex)); h = np.where(abs(hx - h) <= abs(-hx - h), hx, -hx)
    s = v + t * d
    w = (s + u * h) / (2 * A)
    dw = (2 * u * d + h + u * (d * d / (2 * h)) * 2 * u) / (2 * A)
    return u, s, w, dw

def phase(z): return np.unwrap(np.angle(z)) / np.pi

def thimbles(r1, L, R=None):
    A, B = AB(r1, L); vals, pts = crit(A, B)
    if R is None: R = 4 * max(abs(x) for x in vals)
    s0 = R * np.exp(0.3j)
    order = sorted(range(2), key=lambda i: np.angle((vals[i] - s0) * np.exp(-1j * np.angle(-s0))))   # angular order seen from s0
    S = np.sqrt(s0 * s0 - 4 * A * B); p, q = (s0 + S) / (2 * A), (s0 - S) / (2 * A)
    arcs = []
    for i in order:
        u, s, w, dw = arc(vals[i], s0, A, B)
        if abs(w[0] - q) < abs(w[0] - p): w, dw, s = w[::-1], -dw[::-1], s[::-1]      # orient p -> q
        arcs.append(dict(v=vals[i], cp=pts[i], w=w, dw=dw, s=s))
    return dict(A=A, B=B, vals=vals, pts=pts, R=R, s0=s0, p=p, q=q, order=order, arcs=arcs)

def degrees(T, omega="dw/w"):
    al = []
    for a in T["arcs"]:
        z = a["dw"] / a["w"] if omega == "dw/w" else a["dw"]
        al.append(phase(z))
    dp = al[1][0] - al[0][0]; dq = al[1][-1] - al[0][-1]
    return dp, dq, int(np.floor(dq)) - int(np.floor(dp))

if __name__ == "__main__":
    say(f"==== run {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rec = {}
    ok1 = True; ok2 = True; ok3 = True; ok4 = True; info1 = []; info2 = []; info3 = []; info4 = []
    for L in (10, 20):
        T = thimbles(0.4, L)
        A, B = T["A"], T["B"]
        for v, c in zip(T["vals"], T["pts"]):
            W2 = 2 * B / c ** 3; W0 = A * c + B / c
            ok1 &= abs(W2) > 1e-9 and abs(W0 - v) < 1e-9 * max(1, abs(v)) and abs(A - B / c ** 2) < 1e-9
        ok1 &= abs(T["vals"][0] - T["vals"][1]) > 1e-6
        info1.append(f"L={L}: values {T['vals'][0]:.6g}, {T['vals'][1]:.6g}")
        e2 = []
        for a in T["arcs"]:
            res = np.max(abs(A * a["w"] + B / a["w"] - a["s"])) / max(1, abs(a["v"]))
            ends = max(abs(a["w"][0] - T["p"]), abs(a["w"][-1] - T["q"]))
            thr = np.min(abs(a["w"] - a["cp"]))
            e2.append((res, ends, thr)); ok2 &= res < 1e-8 and ends < 1e-9 and thr < 1e-6
        a1, a2 = T["arcs"]; m = len(a1["w"]); cut = m // 50
        far = np.min(abs(a1["w"][cut:-cut:10, None] - a2["w"][None, cut:-cut:10]))
        dp, dq, dd = degrees(T)
        trans = min(abs(np.sin(np.pi * dp)), abs(np.sin(np.pi * dq)))
        ok2 &= trans > 1e-6 and far > 1e-6
        info2.append(f"L={L}: residual/ends/through-crit {[(f'{x:.1e}', f'{y:.1e}', f'{z:.1e}') for x, y, z in e2]}; interior separation {far:.3g}; min |sin| {trans:.3g}")
        ok3 &= dd == 0
        info3.append(f"L={L}: delta(p) = {dp:.6f}, delta(q) = {dq:.6f}, deg(q) - deg(p) = {dd}")
        # KQ4: local intersection signs of the oriented arcs (arc1, arc2) at p and q, versus (-1)^deg
        sg = []
        for k in (0, -1):
            t1, t2 = a1["dw"][k], a2["dw"][k]; sg.append(int(np.sign((np.conj(t1) * t2).imag)))
        deg = [int(np.floor(dp)), int(np.floor(dq))]
        euler = sum((-1) ** x for x in deg); signed = sum(sg)
        cons = all(sg[i] == (-1) ** deg[i] * sg[0] * (-1) ** deg[0] for i in range(2))
        ok4 &= cons and abs(euler) == 2 and abs(signed) == 2
        info4.append(f"L={L}: degrees {deg}, signs {sg}, Floer Euler char {euler}, signed intersection {signed}; symmetric form [[2,{abs(signed)}],[{abs(signed)},2]] up to sign")
        dpw, dqw, ddw = degrees(T, "dw")
        rec[f"L={L}"] = dict(R=T["R"], s0=str(T["s0"]), p=str(T["p"]), q=str(T["q"]), order=T["order"], delta_p=dp, delta_q=dq,
                             degdiff=dd, delta_p_dw=dpw, delta_q_dw=dqw, degdiff_dw=ddw, signs=sg, euler=euler)
    check("KQ1 IDENTITY two nondegenerate critical points with values +-2 sqrt(AB), distinct", ok1, "; ".join(info1))
    check("KQ2 IDENTITY thimble arcs join the base-fibre points through their critical points, meet only at {p,q}, transversely", ok2, "; ".join(info2))
    check("KQ3 HYPOTHESIS with Omega = dw/w both generators in one degree: Kronecker quiver, derived category D^b(Coh P^1)", ok3, "; ".join(info3))
    check("KQ4 IDENTITY Floer Euler characteristic = signed intersection = +-2 = chi(O,O(1)) up to sign; consistent signs", ok4, "; ".join(info4))
    # KQ5: complex loops of r1
    ok5 = True; info5 = []
    for name, (x0, x1) in (("A-zero", (0.6, 0.8)), ("B-zero", (0.2, 0.4))):
        for L in (10, 20):
            H = 2 * np.pi / L; n = 20000
            corners = [x0, x1, x1 + 1j * H, x0 + 1j * H, x0]
            path = np.concatenate([np.linspace(corners[k], corners[k + 1], n, endpoint=False) for k in range(4)] + [[corners[4]]])
            A, B = AB(path, L); prod = A * B
            v1 = 2 * cont_sqrt(prod)
            if v1[0].real < 0: v1 = -v1
            turn = (np.unwrap(np.angle(v1))[-1] - np.unwrap(np.angle(v1))[0]) / np.pi
            wind = (np.unwrap(np.angle(prod))[-1] - np.unwrap(np.angle(prod))[0]) / (2 * np.pi)
            R = 4 * np.max(abs(v1)); s0 = R * np.exp(0.3j)
            ang = lambda v: np.angle((v - s0) * np.exp(-1j * np.angle(-s0)))
            first0 = ang(v1[0]) < ang(-v1[0]); first1 = ang(v1[-1]) < ang(-v1[-1])
            ok5 &= abs(turn - 1) < 1e-6 and np.min(abs(prod)) > 1e-8
            info5.append(f"{name} L={L}: AB winds {wind:+.6f}, v1 turns {turn:+.6f} pi, min|AB| {np.min(abs(prod)):.3g}, angular order of the tracked value swapped: {first0 != first1}")
            rec[f"loop {name} L={L}"] = dict(winding_AB=wind, half_turns=turn, swapped=bool(first0 != first1), R=R)
    check("KQ5 HYPOTHESIS each counterclockwise loop around a zero of A or B exchanges the critical values by sigma^{+1}", ok5, "; ".join(info5))
    # KQ6: real wall crossing
    ok6 = True; info6 = []
    for L in (10, 20):
        r = np.linspace(0.6, 0.8, 4001); A, B = AB(r, L); v = 2 * np.sqrt(A * B)
        R = 4 * np.max(abs(v)); s0 = R * np.exp(0.3j)
        ang = lambda x: np.angle((x - s0) * np.exp(-1j * np.angle(-s0)))
        orders = set(bool(x) for x in ang(v) < ang(-v))
        dds = [degrees(thimbles(x, L, R))[2] for x in (0.6, 0.8)]
        ok6 &= len(orders) == 1 and dds[0] == dds[1] and np.min(abs(v)) > 1e-8
        info6.append(f"L={L}: angular orders seen {orders}, deg(q)-deg(p) at r1=0.6, 0.8: {dds}")
        rec[f"real L={L}"] = dict(orders=list(orders), degdiffs=dds, R=R)
    check("KQ6 IDENTITY the real wall r1 = c2 causes no Hurwitz move and no change of degrees", ok6, "; ".join(info6))
    check("KQ7 RECORD deltas, Omega = dw degree differences (predicted +-2), labels, R", True,
          "; ".join(f"{k}: deg diff dw/w {v['degdiff']}, dw {v['degdiff_dw']} (delta_dw p {v['delta_p_dw']:.4f}, q {v['delta_q_dw']:.4f}), R {v['R']:.4g}" for k, v in rec.items() if k.startswith("L=")))
    json.dump(rec, open(os.path.join(D, "result.json"), "w"), indent=1, default=str)
