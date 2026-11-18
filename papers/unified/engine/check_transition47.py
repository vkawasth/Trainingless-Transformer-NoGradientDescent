"""Study 47: below L = 30 on the Study 45/46 wall (LT1-LT6). Pre-registered in study47_transition/PREREG_study47.md, frozen by SHA-256
before this code ran. Reuses the families, tracker, braid reading and Artin comparison of check_leadbraid46.py.
Checkpointed per item in study47_transition/ckpt/ (rerun resumes). Output: check_transition47_output.txt.
"""
import os, sys, json, runpy, time, math, itertools
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D47 = os.path.join(HERE, "study47_transition"); CK = os.path.join(D47, "ckpt")
os.makedirs(CK, exist_ok=True)
_argv = sys.argv; sys.argv = ["check_leadbraid46.py"]; M = runpy.run_path(os.path.join(HERE, "check_leadbraid46.py")); sys.argv = _argv
arb, acb, Family, track, cert, cert_start, model_family, model_start, W_family, W_curve = (M[k] for k in
    ("arb", "acb", "Family", "track", "cert", "cert_start", "model_family", "model_start", "W_family", "W_curve"))
read_word, live_view, refine_steps, zrot, zang, artin, ser, des, box, Q, tball, sgn = (M[k] for k in
    ("read_word", "live_view", "refine_steps", "zrot", "zang", "artin", "ser", "des", "box", "Q", "tball", "sgn"))
OUT = open(os.path.join(HERE, "check_transition47_output.txt"), "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
def check(name, cond, info=""):
    say(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else "")); return bool(cond)
def ckpt(name, fn):
    p = os.path.join(CK, name + ".json")
    if os.path.exists(p): return json.load(open(p))
    t0 = time.time(); r = fn(); r["seconds"] = round(time.time() - t0); json.dump(r, open(p, "w")); say(f"  [{name}] {({k: v for k, v in r.items() if k not in ('steps',)})}"); return r

# ---------------------------------------------------------------- Part A: certificate with (tau, L)-dependent coefficient intervals ---------
def family_LI(side, L1, L2):
    """F_{tau,c} with c11 = e^{-0.15 L + k tau}, c12 = e^{-0.25 L + k tau}, L in [L1, L2] (ball), tau the step ball."""
    LI = arb((L1 + L2) / 2, (L2 - L1) / 2 * (1 + 1e-12)); Lm = (L1 + L2) / 2; k = 2.0 / 3 if side == "row" else 0.5; kb = arb(2) / 3 if side == "row" else arb(1) / 2
    one = lambda P: acb(1) if isinstance(P, (arb, acb)) else 1.0
    def cf(P):
        if isinstance(P, (arb, acb)):
            P = acb(P); e = lambda x: x.exp()
            c11 = (-arb(15) / 100 * LI + kb * P).exp(); c12 = (-arb(25) / 100 * LI + kb * P).exp()
            return [acb(1), acb(1), acb(1), (2 * P / 3).exp(), c11, c12] if side == "row" else [acb(1), (-P / 2).exp(), (-P / 2).exp(), acb(1), c11, c12]
        c11 = math.exp(-0.15 * Lm + k * P); c12 = math.exp(-0.25 * Lm + k * P)
        return [1.0, 1.0, 1.0, math.exp(2 * P / 3), c11, c12] if side == "row" else [1.0, math.exp(-P / 2), math.exp(-P / 2), 1.0, c11, c12]
    def dcf(P):
        if isinstance(P, (arb, acb)):
            P = acb(P); c11 = (-arb(15) / 100 * LI + kb * P).exp(); c12 = (-arb(25) / 100 * LI + kb * P).exp()
            return [acb(0), acb(0), acb(0), 2 * (2 * P / 3).exp() / 3, kb * c11, kb * c12] if side == "row" else [acb(0), -(-P / 2).exp() / 2, -(-P / 2).exp() / 2, acb(0), kb * c11, kb * c12]
        c11 = math.exp(-0.15 * Lm + k * P); c12 = math.exp(-0.25 * Lm + k * P)
        return [0.0, 0.0, 0.0, 2 * math.exp(2 * P / 3) / 3, k * c11, k * c12] if side == "row" else [0.0, -math.exp(-P / 2) / 2, -math.exp(-P / 2) / 2, 0.0, k * c11, k * c12]
    return Family(cf, cf, dcf, dcf)
def certify_interval(L1, L2, Ns=(1, 4, 16)):
    """The L-interval is split into N equal pieces if needed (the pieces cover it)."""
    for N in Ns:
        parts = []
        for j in range(N):
            r = certify_piece(L1 + (L2 - L1) * j / N, L1 + (L2 - L1) * (j + 1) / N); parts.append(r)
            if not r["ok"]: break
        if len(parts) == N and all(r["ok"] for r in parts):
            return dict(ok=True, pieces=N, min_separation_lower=min(r["min_separation_lower"] for r in parts), n_steps=sum(r["n_steps"] for r in parts))
    return dict(ok=False, pieces_tried=list(Ns), reason=parts[-1].get("reason", "value enclosures overlap"), last=parts[-1])
def certify_piece(L1, L2):
    fr, fc = family_LI("row", L1, L2), family_LI("col", L1, L2); ER = tuple(1e-7 * 1.25 ** k for k in range(70))
    ws = [fr.newton(w, -2.0) for w in model_start(model_family("row"))]
    if any(w is None for w in ws): return dict(ok=False, reason="newton at start")
    c0 = cert_start(fr, ws, -2.0, rads=ER)
    if c0 is None: return dict(ok=False, reason="start not certified")
    try:
        s1, B1, w1 = track(fr, ws, [c[1] for c in c0], -2.0, 0.0, h0=0.005, hmax=0.02, hmin=1e-8, erads=ER, rads_all=ER, nr=14)
        s2, B2, w2 = track(fc, w1, B1, 0.0, 2.0, h0=0.005, hmax=0.02, hmin=1e-8, erads=ER, rads_all=ER, nr=14)
    except RuntimeError as e: return dict(ok=False, reason=str(e))
    steps = s1 + s2
    pairs = [(st, i, j) for st in steps for i, j in itertools.combinations(range(4), 2)]
    disj = all(not st["br"][i]["s"].overlaps(st["br"][j]["s"]) for st, i, j in pairs)
    minsep = min(float(abs(st["br"][i]["s"] - st["br"][j]["s"]).lower()) for st, i, j in pairs)
    return dict(ok=bool(disj), n_steps=len(steps), min_separation_lower=minsep, max_value_radius=max(float(o["s"].rad()) for st in steps for o in st["br"]))
def diagnose(L):
    """All critical points of F at L (c at their exact values) along the window: distance from the leading four to the other two, and
    the minimal separation of the leading values (float, for the LT2 diagnosis only)."""
    out = []
    for tau in np.linspace(-2, 2, 81):
        side = "row" if tau <= 0 else "col"; fam = family_LI(side, L, L); rng = np.random.default_rng(1); found = []
        for _ in range(800):
            w = fam.newton(rng.normal(0, 3, 2) + 1j * rng.uniform(-np.pi, np.pi, 2), tau)
            if w is None: continue
            w = w.real + 1j * ((w.imag + np.pi) % (2 * np.pi) - np.pi)
            if not any(np.abs(np.exp(w) - np.exp(q)).max() < 1e-7 * (1 + np.abs(np.exp(w)).max()) for q in found): found.append(w)
        vals = sorted([fam.s(w, tau) for w in found], key=abs)
        out.append(dict(tau=float(tau), n=len(found), moduli=[round(abs(v), 4) for v in vals]))
    return out

# ---------------------------------------------------------------- Part B: W directly ---------------------------------------------------------
def start_six(L):
    p = os.path.join(CK, f"start_L{L}.json")
    if os.path.exists(p):
        d = json.load(open(p)); return [np.array([complex(*w[0]), complex(*w[1])]) for w in d["w"]], [[des(z) for z in b] for b in d["B"]]
    s3 = json.load(open(os.path.join(HERE, "study45_example", "ckpt", "S3_points.json")))["TA"]
    ws = [np.array([complex(*w[0]), complex(*w[1])]) for w in s3["w"]]; B = [[box(complex(*w[0]), r), box(complex(*w[1]), r)] for w, r in zip(s3["w"], s3["rad"])]
    if L > 20:
        steps, B, ws = track(W_curve(), ws, B, 20.0, float(L), h0=0.05, hmax=1.0, rel=True)
    json.dump(dict(w=[[[w[0].real, w[0].imag], [w[1].real, w[1].imag]] for w in ws], B=[[ser(z) for z in b] for b in B]), open(p, "w"))
    return ws, B
def word_ref(fam, steps, zmap, sel, n):
    for _ in range(10):
        view = [dict(pa=v["pa"], pb=v["pb"], s=[v["s"][i] for i in sel], sa=[v["sa"][i] for i in sel], sb=[v["sb"][i] for i in sel]) for v in live_view(steps)]
        word, perm, bad = read_word(view, zmap, n)
        if not bad: return word, perm, steps
        steps = refine_steps(fam, steps, set(bad), 6)
    return None, None, steps
def hurw(word, sel):
    return None if word is None else [dict(near=sel[l["left"] if l["e"] == 1 else l["right"]], far=sel[l["right"] if l["e"] == 1 else l["left"]],
                                           eps=(-1 if l["e"] == 1 else 1), tau=l["p"]) for l in word]
S0FIX = acb(9800) * acb(0, arb(3) / 10).exp()
def W_run(L):
    fam = W_family(float(L)); ws, B = start_six(L)
    s1, B1, w1 = track(fam, ws, B, -2.0, 0.0, h0=0.005, hmax=0.02, rel=True); s2, B2, w2 = track(fam, w1, B1, 0.0, 2.0, h0=0.005, hmax=0.02, rel=True)
    steps = s1 + s2
    fr = model_family("row"); mvals = [fr.s(w, -2.0) for w in model_start(fr)]; sc0 = math.exp(L * (0.1 - 2 / (3 * L)))
    lead = [int(np.argmin([abs(complex(float(steps[0]["br"][i]["sa"].real.mid()), float(steps[0]["br"][i]["sa"].imag.mid())) * sc0 - mv) for i in range(6)])) for mv in mvals]
    six = list(range(6))
    w4, p4, steps = word_ref(fam, steps, zrot, lead, 4)
    w6, p6, steps = word_ref(fam, steps, zrot, six, 6)
    mx = max(float(o["s"].abs_upper()) for st in steps for o in st["br"]); R = 3 * mx; e = math.floor(math.log10(R)) - 1; Rq = Fr(math.ceil(R / 10 ** e)) * Fr(10) ** e
    s0 = acb(Q(Rq)) * acb(0, arb(3) / 10).exp()
    hr, _, steps = word_ref(fam, steps, zang(s0), six, 6)
    hf, _, steps = word_ref(fam, steps, zang(S0FIX), six, 6)
    ends = [sorted(range(6), key=lambda k, o=o: float(zrot(o[k]).real.mid())) for o in ([x["sa"] for x in steps[0]["br"]], [x["sb"] for x in steps[-1]["br"]])]
    return dict(L=L, lead=lead, word4=w4, word6=w6, R=str(Rq), max_modulus=mx, fixed_base_ok=bool(3 * mx < 9800),
                hurwitz_rule=hurw(hr, six), hurwitz_fixed=hurw(hf, six), end_orders=ends, n_steps=len(steps))

if __name__ == "__main__":
    say(f"==== run {time.strftime('%Y-%m-%d %H:%M:%S')}")
    A = {}
    for L1 in (28, 26, 24, 22, 20):
        A[L1] = ckpt(f"A_{L1}", lambda L1=L1: certify_interval(float(L1), float(L1 + 2)))
    lt1 = all(A[L1]["ok"] for L1 in A)
    below = {}
    if lt1:
        for L1 in (18, 16, 14, 12, 10, 8, 6, 4):
            below[L1] = ckpt(f"A_{L1}", lambda L1=L1: certify_interval(float(L1), float(L1 + 2)))
            if not below[L1]["ok"]: break
    first_fail = next((L1 for L1 in list(A) + list(below) if not (A.get(L1) or below.get(L1))["ok"]), None)
    diag = ckpt(f"diag_{first_fail}", lambda: dict(L=first_fail, rows=diagnose(float(first_fail)))) if first_fail is not None else None
    Bres = {L: ckpt(f"W_L{L}", lambda L=L: W_run(L)) for L in list(range(20, 31)) + [40]}
    check("LT1 HYPOTHESIS leading-braid certificate on every L-interval [L1, L1+2], L1 = 28, ..., 20 (so the leading braid holds for all L >= 20)",
          lt1, "; ".join(f"[{L1},{L1 + 2}]: {'ok' if A[L1]['ok'] else 'failed (' + A[L1].get('reason', 'overlap') + ')'}" for L1 in A))
    check("LT2 RECORD below L = 20: first failure and diagnosis", True,
          ("; ".join(f"[{L1},{L1 + 2}]: {'ok, min separation ' + format(below[L1]['min_separation_lower'], '.3g') if below[L1]['ok'] else 'failed (' + below[L1].get('reason', 'value enclosures overlap') + ')'}" for L1 in below)
           + (f"; diagnosis at L = {first_fail}: " + "; ".join(f"tau {r['tau']:+.2f}: {r['n']} critical points, moduli {r['moduli']}" for r in diag["rows"][::10]) if diag else "")))
    A4 = artin([dict(k=1, e=1), dict(k=2, e=-1)])
    def w6s(w): return None if w is None else " ".join(f"s{l['k']}{'' if l['e'] == 1 else '^-1'}" for l in w)
    check("LT3 RECORD direct W, L = 20..30, 40: leading and six-value braids, Hurwitz words (rule and fixed base point), end orders", True, " || ".join(
        f"L = {L}: B4 {w6s(r['word4'])}; B6 {w6s(r['word6'])}; rule R = {r['R']}: {[(h['near'], h['far'], h['eps']) for h in (r['hurwitz_rule'] or [])]}; fixed: {[(h['near'], h['far'], h['eps']) for h in (r['hurwitz_fixed'] or [])]}; ends {r['end_orders']}"
        for L, r in Bres.items()))
    check("LT4 HYPOTHESIS the leading B_4 element is sigma_1 sigma_2^{-1} at every L in {20, ..., 30, 40}",
          all(r["word4"] is not None and artin(r["word4"]) == A4 for r in Bres.values()), f"{[(L, w6s(r['word4'])) for L, r in Bres.items()]}")
    # Part C: the third letter
    three = {L: [h for h in (r["hurwitz_rule"] or []) if not (h["near"] in r["lead"] and h["far"] in r["lead"])] for L, r in Bres.items()}
    present = [L for L in Bres if three[L]]; Lmax = max(present) if present else None
    E = (Lmax is not None) and any(abs(abs(h["tau"][1]) - 2) < 0.05 or abs(abs(h["tau"][0]) - 2) < 0.05 for h in three[Lmax])
    Bt = Bres[20]["fixed_base_ok"] and not [h for h in (Bres[20]["hurwitz_fixed"] or []) if not (h["near"] in Bres[20]["lead"] and h["far"] in Bres[20]["lead"])]
    E6 = {L: artin(r["word6"], 6) if r["word6"] is not None else None for L, r in Bres.items()}
    Ls = sorted(Bres); changes = [(a, b) for a, b in zip(Ls, Ls[1:]) if E6[a] != E6[b]]
    check("LT5 RECORD classification of the extra letter: (E) exit through the window end, (B) base-point artifact, (C) collision", True,
          f"present at L = {present}; positions {[(L, [[round(x, 4) for x in h['tau']] for h in three[L]]) for L in present]}; (E) {E}; (B) {Bt}; (C) B_6 changes between {changes}")
    check("LT6 RECORD collisions between consecutive L (if any)", True, "none: the six-value braid element is the same at all L tested" if not changes else f"B_6 changes between {changes} (not located further)")
