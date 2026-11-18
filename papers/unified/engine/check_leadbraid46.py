"""Study 46: the leading-group braid, predicted from the two quartic families before W is examined, then confirmed on W (LB1-LB6).
Pre-registered in study46_leadbraid/PREREG_study46.md, frozen by SHA-256 before this code ran. Interval primitives (Arb balls, the Krawczyk
operator) are imported from check_example45.py; the families, the tracker and the braid reading are written here.
Checkpointed by stage in study46_leadbraid/ckpt/ (rerun resumes at the first missing stage). Output: check_leadbraid46_output.txt.

Families. A family is a Laurent polynomial sum_k c_k(p) Y^{v_k} on (C*)^2 in log coordinates w (Y = e^w) with coefficients depending on a
real path parameter p. Its critical points are continued in p with chained Krawczyk boxes; the critical values s = sum_k c_k e^{<v_k, w>}
and their p-derivatives (= sum_k c_k'(p) e^{<v_k, w>} at a critical point) are enclosed on every step.
Braid reading (convexity rule, as in Study 45 note 7). In a coordinate z(s), on each step every pair keeps either the sign of Re(z_i - z_j)
or the sign of Im(z_i - z_j); these open linear conditions cut out a convex set containing the whole motion, so the step's braid is that
of the straight-line motion between the end configurations, read off from the end enclosures. Steps that fail are subdivided.
  braid word: z = e^{-0.7 i} s; a crossing of neighbours k, k+1 is sigma_k^{+1} if the point moving right passes above, else sigma_k^{-1}.
  Hurwitz word: z = -i log((s - s0) / nu), nu = -s0 / |s0|, so Re z is the angle seen from s0 and -Im z = log |s - s0|; a crossing is an
  angular swap; the nearer value is the one with larger Im z.
"""
import os, sys, json, runpy, time, math, itertools
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D46 = os.path.join(HERE, "study46_leadbraid"); CK = os.path.join(D46, "ckpt")
os.makedirs(CK, exist_ok=True)
_argv = sys.argv; sys.argv = ["check_example45.py", "--lib"]; G = runpy.run_path(os.path.join(HERE, "check_example45.py")); sys.argv = _argv
arb, acb, ctx = G["arb"], G["acb"], G["ctx"]
kraw, box, pt, mat_inv_acb, distinct_mod, tball, contains_box, ser, des, sgn, Q = (G[k] for k in
    ("kraw", "box", "pt", "mat_inv_acb", "distinct_mod", "tball", "contains_box", "ser", "des", "sgn", "Q"))
V = G["V"]; Vf = np.array(V, float); CELL = G["CELL"]; x0f = G["x0f"]; DXf = G["DX"]; x0b = G["x0b"]; FS, GA = G["FS"], G["GA"]
OUT = open(os.path.join(HERE, "check_leadbraid46_output.txt"), "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
def check(name, cond, info=""):
    say(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else "")); return bool(cond)
def stage(name):
    def deco(fn):
        def run(*a):
            p = os.path.join(CK, name + ".json")
            if os.path.exists(p): say(f"[{name}] loaded from checkpoint"); return json.load(open(p))
            t0 = time.time(); say(f"[{name}] computing"); r = fn(*a); json.dump(r, open(p, "w")); say(f"[{name}] done in {time.time() - t0:.0f}s"); return r
        return run
    return deco
ROTc = np.exp(-0.7j); ROTb = acb(0, -arb(7) / 10).exp()

# ---------------------------------------------------------------- families ------------------------------------------------------------------
class Family:
    """coef(p) -> list of 6 coefficients (floats complex for p float, acb for p ball); dcoef(p) -> their p-derivatives."""
    def __init__(self, coef_f, coef_b, dcoef_f, dcoef_b): self.cf, self.cb, self.dcf, self.dcb = coef_f, coef_b, dcoef_f, dcoef_b
    def E(self, w, p): return np.array(self.cf(p)) * np.exp(Vf @ w)
    def F(self, w, p): return Vf.T @ self.E(w, p)
    def J(self, w, p): E = self.E(w, p); return (Vf.T * E) @ Vf
    def s(self, w, p): return self.E(w, p).sum()
    def dwdp(self, w, p): return -np.linalg.solve(self.J(w, p), Vf.T @ (np.array(self.dcf(p)) * np.exp(Vf @ w)))
    def newton(self, w, p, it=60):
        w = np.array(w, complex)
        for _ in range(it):
            try: st = np.linalg.solve(self.J(w, p), self.F(w, p))
            except np.linalg.LinAlgError: return None
            w = w - st
            if not np.isfinite(w).all(): return None
            if np.abs(st).max() < 1e-15 * (1 + np.abs(w).max()): break
        E = self.E(w, p); return w if np.abs(Vf.T @ E).max() < 1e-11 * np.abs(E).max() else None
    def Eb(self, x, P): c = self.cb(P); return [c[k] * (V[k][0] * x[0] + V[k][1] * x[1]).exp() for k in range(6)]
    def Fb(self, P): return lambda x: (lambda e: [sum(V[k][m] * e[k] for k in range(6)) for m in range(2)])(self.Eb(x, P))
    def Jb(self, P): return lambda x: (lambda e: [[sum(V[k][a] * V[k][b] * e[k] for k in range(6)) for b in range(2)] for a in range(2)])(self.Eb(x, P))
    def sb(self, x, P): return sum(self.Eb(x, P))
    def dsb(self, x, P): d = self.dcb(P); return sum(d[k] * (V[k][0] * x[0] + V[k][1] * x[1]).exp() for k in range(6))
def cert(fam, w, P, pm, rads):
    Y = mat_inv_acb(fam.J(w, pm)); xm = [pt(w[0]), pt(w[1])]
    for r in rads:
        X = [box(w[0], r), box(w[1], r)]; ok, K = kraw(fam.Fb(P), fam.Jb(P), xm, X, Y)
        if ok: return r, X, K
    return None
def one(x): return acb(1) if isinstance(x, (arb, acb)) else 1.0
def ex(x): return x.exp() if isinstance(x, (arb, acb)) else np.exp(x)
def model_family(side, c11=None, c12=None):
    """Scaled family F_{tau, c} on one side of the wall; c11, c12: None (model, c = 0) or ball boxes (uniform certificate) or floats."""
    z = lambda P: (acb(0) if isinstance(P, (arb, acb)) else 0.0)
    def cc(P, c):
        if c is None: return z(P)
        return acb(c) if isinstance(P, (arb, acb)) else (float(c.mid()) if isinstance(c, arb) else c)
    if side == "row":
        cf = lambda P: [one(P), one(P), one(P), ex(2 * P / 3), cc(P, c11), cc(P, c12)]
        dcf = lambda P: [z(P), z(P), z(P), 2 * ex(2 * P / 3) / 3, z(P), z(P)]
    else:
        cf = lambda P: [one(P), ex(-P / 2), ex(-P / 2), one(P), cc(P, c11), cc(P, c12)]
        dcf = lambda P: [z(P), -ex(-P / 2) / 2, -ex(-P / 2) / 2, z(P), z(P), z(P)]
    return Family(cf, lambda P: [acb(x) for x in cf(acb(P) if isinstance(P, arb) else P)], dcf, lambda P: [acb(x) for x in dcf(acb(P) if isinstance(P, arb) else P)])
def W_family(L):
    """W itself at fixed L along t = tau / L (p = tau); coefficients e^{-L x0_k(tau / L)}; unscaled values."""
    Lq = L
    def cf(P):
        if isinstance(P, (arb, acb)): t = acb(P) / Lq; return [(-Lq * x0b(k, t)).exp() for k in range(6)]
        return list(np.exp(-Lq * x0f(P / Lq)))
    def dcf(P):
        if isinstance(P, (arb, acb)): t = acb(P) / Lq; return [-(FS[CELL[k][0]] * GA[CELL[k][1]]) * (-Lq * x0b(k, t)).exp() for k in range(6)]
        return list(-DXf * np.exp(-Lq * x0f(P / Lq)))
    return Family(cf, cf, dcf, dcf)
def h1_of(tau, L):                                                              # first level on the window (ball or float)
    t = tau / L
    return (arb(1) / 10 + t / 3) if (isinstance(tau, (arb, acb)) and float(tau.mid() + tau.rad()) <= 0) else (arb(1) / 10 if isinstance(tau, (arb, acb)) else (0.1 + t / 3 if tau <= 0 else 0.1))

# ---------------------------------------------------------------- tracker -------------------------------------------------------------------
def track(fam, ws, Bprev, pa, pb, h0=0.01, hmax=0.05, hmin=1e-10, rel=False, erads=(1e-13, 1e-11, 1e-9, 1e-7), rads_all=(1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2), nr=4):
    """Chained certified continuation of all branches over p in [pa, pb]. Returns steps (with live balls) and end boxes."""
    p = pa; ws = [np.array(w) for w in ws]; h = h0; steps = []
    while p < pb:
        q = min(p + h, pb); pm = (p + q) / 2; P = tball(p, q); out = []; fail = False
        for i, w in enumerate(ws):
            wm = fam.newton(w + fam.dwdp(w, p) * (pm - p), pm)
            if wm is None: fail = True; break
            sp = np.abs(fam.dwdp(wm, pm)).max() * (q - p) / 2; sc = (1 + np.abs(wm).max()) if rel else 1.0
            ends = []                                                           # tight end boxes first; the step box must contain them
            for pe in (p, q):
                we = fam.newton(wm, pe); ce = None if we is None else cert(fam, we, arb(pe), pe, tuple(x * sc for x in erads))
                if ce is None: fail = True; break
                ends.append(ce)
            if fail: break
            rmin = max([1.5 * sp] + [1.3 * e[0] for e in ends] + ([1.3 * float(Bprev[i][0].real.rad())] if Bprev is not None else []))
            c = cert(fam, wm, P, pm, tuple(r * sc for r in rads_all if r * sc > rmin)[:nr])
            if c is None: fail = True; break
            r, X, K = c
            if not all(contains_box(X, e[1]) for e in ends) or (Bprev is not None and not contains_box(X, Bprev[i])): fail = True; break
            out.append(dict(wm=wm, X=X, Ba=ends[0][1], Bb=ends[1][1], s=fam.sb(X, P), ds=fam.dsb(X, P), sa=fam.sb(ends[0][2], arb(p)), sb=fam.sb(ends[1][2], arb(q))))
        if not fail:
            Xs = [o["X"] for o in out]; fail = not all(distinct_mod(Xs[i], Xs[j]) for i in range(len(Xs)) for j in range(i + 1, len(Xs)))
        if fail:
            h /= 2
            if h < hmin: raise RuntimeError(f"tracking failed at p = {p}")
            continue
        steps.append(dict(pa=p, pb=q, br=out)); Bprev = [o["Bb"] for o in out]; ws = [fam.newton(o["wm"], q) for o in out]; p = q; h = min(h * 1.5, hmax)
    return steps, Bprev, ws
def jsteps(steps, idx=None):
    idx = range(len(steps[0]["br"])) if idx is None else idx
    return [dict(pa=st["pa"], pb=st["pb"], wm=[[[st["br"][i]["wm"][m].real, st["br"][i]["wm"][m].imag] for m in range(2)] for i in idx],
                 s=[ser(st["br"][i]["s"]) for i in idx], sa=[ser(st["br"][i]["sa"]) for i in idx], sb=[ser(st["br"][i]["sb"]) for i in idx],
                 ds=[ser(st["br"][i]["ds"]) for i in idx]) for st in steps]

# ---------------------------------------------------------------- braid reading ---------------------------------------------------------------
def read_word(steps, zmap, n):
    """steps: list of dict(pa, pb, s=[...], sa=[...], sb=[...]) (acb); zmap: s -> coordinate. Returns (word, perm, failed_step_indices)."""
    cur = None; word = []; bad = []
    for si, st in enumerate(steps):
        Z = [zmap(x) for x in st["s"]]; Za = [zmap(x) for x in st["sa"]]; Zb = [zmap(x) for x in st["sb"]]
        if cur is None:
            if any(sgn((Za[i] - Za[j]).real) == 0 for i, j in itertools.combinations(range(n), 2)): bad.append(si); return word, None, bad
            cur = sorted(range(n), key=lambda k: float(Za[k].real.mid()))
        evs = []; ok = True
        for i, j in itertools.combinations(range(n), 2):
            d = Z[i] - Z[j]
            if not d.real.contains(0): continue
            if d.imag.contains(0): ok = False; break
            ra, rb = (Za[i] - Za[j]).real, (Zb[i] - Zb[j]).real
            if sgn(ra) == 0 or sgn(rb) == 0: ok = False; break
            if sgn(ra) != sgn(rb): evs.append((ra / (ra - rb), i, j))
        if ok and any(a[0].overlaps(b[0]) for a, b in itertools.combinations(evs, 2)): ok = False
        if not ok: bad.append(si); continue
        for lam, i, j in sorted(evs, key=lambda e: float(e[0].mid())):
            pi_, qi_ = cur.index(i), cur.index(j)
            if abs(pi_ - qi_) != 1: bad.append(si); break
            k = min(pi_, qi_); p_, q_ = cur[k], cur[k + 1]
            above = (Z[p_] - Z[q_]).imag > 0
            word.append(dict(k=k + 1, e=1 if above else -1, left=p_, right=q_, step=si, p=[st["pa"], st["pb"]])); cur[k], cur[k + 1] = q_, p_
    return word, cur, bad
def refine_steps(fam, steps_live, idx_bad, n):
    """Replace the steps with the given indices by certified sub-steps (chained inside the parent boxes)."""
    out = []
    for si, st in enumerate(steps_live):
        if si not in idx_bad: out.append(st); continue
        ws = [fam.newton(o["wm"], st["pa"]) for o in st["br"]]
        sub, _, _ = track(fam, ws, [o["Ba"] for o in st["br"]], st["pa"], st["pb"], h0=(st["pb"] - st["pa"]) / 8, hmax=(st["pb"] - st["pa"]) / 8)
        out += sub
    return out
def live_view(steps): return [dict(pa=s["pa"], pb=s["pb"], s=[o["s"] for o in s["br"]], sa=[o["sa"] for o in s["br"]], sb=[o["sb"] for o in s["br"]]) for s in steps]
def word_with_refinement(fam, steps, zmap, n, sel=None, rounds=8):
    for _ in range(rounds):
        view = live_view(steps)
        if sel is not None: view = [dict(pa=v["pa"], pb=v["pb"], s=[v["s"][i] for i in sel], sa=[v["sa"][i] for i in sel], sb=[v["sb"][i] for i in sel]) for v in view]
        word, perm, bad = read_word(view, zmap, n)
        if not bad: return word, perm, steps
        steps = refine_steps(fam, steps, set(bad), n)
    raise RuntimeError("braid reading undecided after refinement")
def zrot(s): return ROTb * s
def zang(s0):
    nu = -s0 / abs(s0)
    return lambda s: acb(0, -1) * ((s - s0) / nu).log()

# ---------------------------------------------------------------- B_4 comparison (Artin action on F_4) --------------------------------------
def free_reduce(w):
    out = []
    for x in w:
        if out and out[-1] == -x: out.pop()
        else: out.append(x)
    return out
def artin(word, n=4):
    """Images of the free generators 1..n under the braid (sigma_k: x_k -> x_k x_{k+1} x_k^-1, x_{k+1} -> x_k; inverse accordingly)."""
    img = {i: [i] for i in range(1, n + 1)}
    def inv(w): return [-x for x in reversed(w)]
    for let in word:                                                            # apply letters left to right as automorphisms acting on images
        k, e = let["k"], let["e"]; new = dict(img)
        if e == 1: sub = {k: [k, k + 1, -k], k + 1: [k]}
        else: sub = {k: [k + 1], k + 1: [-(k + 1), k, k + 1]}
        def apply(w):
            r = []
            for x in w:
                y = sub.get(abs(x), [abs(x)]); r += y if x > 0 else inv(y)
            return free_reduce(r)
        img = {i: apply(img[i]) for i in img}
    return {i: img[i] for i in img}

# ================================================================ stages =====================================================================
def model_start(fam_row, tau0=-2.0):
    b = math.exp(2 * tau0 / 3); Ys = np.roots([1, 0, 0, -1, -b]); out = []
    for Y in Ys:
        w = fam_row.newton(np.array([np.log(complex(Y)), np.log(complex(Y))]), tau0); out.append(w)
    return out
def track_window(fam_r, fam_c, ws, B0, **kw):  # noqa
    s1, B1, w1 = track(fam_r, ws, B0, -2.0, 0.0, **kw); s2, B2, w2 = track(fam_c, w1, B1, 0.0, 2.0, **kw)
    return s1 + s2, B2, w2
def cert_start(fam, ws, p, rel=False, rads=(1e-13, 1e-11, 1e-9, 1e-7)):
    out = []
    for w in ws:
        sc = (1 + np.abs(w).max()) if rel else 1.0
        c = cert(fam, w, arb(p), p, tuple(x * sc for x in rads))
        if c is None: return None
        out.append(c)
    Xs = [c[1] for c in out]
    return out if all(distinct_mod(Xs[i], Xs[j]) for i in range(len(Xs)) for j in range(i + 1, len(Xs))) else None

@stage("S1_model")
def S1():
    fr, fc = model_family("row"), model_family("col"); ws = model_start(fr); c0 = cert_start(fr, ws, -2.0)
    steps, Bend, wend = track_window(fr, fc, ws, [c[1] for c in c0])
    # braid word (refinement needs the family of the step: row for p <= 0, col for p >= 0)
    for _ in range(8):
        view = live_view(steps); word, perm, bad = read_word(view, zrot, 4)
        if not bad: break
        steps = [st for st in steps]; new = []
        for si, st in enumerate(steps):
            if si not in bad: new.append(st); continue
            fam = fr if st["pb"] <= 0 else fc; wsx = [fam.newton(o["wm"], st["pa"]) for o in st["br"]]
            sub, _, _ = track(fam, wsx, [o["Ba"] for o in st["br"]], st["pa"], st["pb"], h0=(st["pb"] - st["pa"]) / 8, hmax=(st["pb"] - st["pa"]) / 8); new += sub
        steps = new
    else: raise RuntimeError("model braid undecided")
    mx = max(float(o["s"].abs_upper()) for st in steps for o in st["br"]); R = 3 * mx; e = math.floor(math.log10(R)) - 1; Rq = Fr(math.ceil(R / 10 ** e)) * Fr(10) ** e
    s0 = acb(Q(Rq)) * acb(0, arb(3) / 10).exp()
    for _ in range(8):
        view = live_view(steps); hw, hperm, bad = read_word(view, zang(s0), 4)
        if not bad: break
        new = []
        for si, st in enumerate(steps):
            if si not in bad: new.append(st); continue
            fam = fr if st["pb"] <= 0 else fc; wsx = [fam.newton(o["wm"], st["pa"]) for o in st["br"]]
            sub, _, _ = track(fam, wsx, [o["Ba"] for o in st["br"]], st["pa"], st["pb"], h0=(st["pb"] - st["pa"]) / 8, hmax=(st["pb"] - st["pa"]) / 8); new += sub
        steps = new
    else: raise RuntimeError("model Hurwitz word undecided")
    hurwitz = [dict(near=(l["left"] if l["e"] == 1 else l["right"]), far=(l["right"] if l["e"] == 1 else l["left"]), eps=(-1 if l["e"] == 1 else 1), p=l["p"]) for l in hw]
    a0 = 2 * 3 ** 0.25 / 3; tau0 = -2 * math.log(a0)
    sw = next(st for st in steps if st["pa"] <= tau0 <= st["pb"]); zero_branch = [i for i, o in enumerate(sw["br"]) if o["s"].real.contains(0)]
    return dict(n_steps=len(steps), start_order=sorted(range(4), key=lambda k: float(zrot(steps[0]["br"][k]["sa"]).real.mid())), word=word, perm=perm,
                R=str(Rq), hurwitz=hurwitz, zero_crossing_tau_exact="-2 ln(2 3^(1/4) / 3)", zero_crossing_tau=tau0, zero_branch=zero_branch,
                steps=jsteps(steps), start_w=[[[w[0].real, w[0].imag], [w[1].real, w[1].imag]] for w in ws],
                values_at_wall=[ser(o["sb"]) for o in next(st for st in steps if st["pb"] == 0.0)["br"]])

def cbox(L0):
    e11, e12 = math.exp(1 - 0.15 * L0), math.exp(1 - 0.25 * L0)
    return arb(e11 / 2, e11 / 2 * (1 + 1e-12)), arb(e12 / 2, e12 / 2 * (1 + 1e-12)), e11, e12
def uniform_sub(L0, c11, c12):
    fr, fc = model_family("row", c11, c12), model_family("col", c11, c12)
    ws = model_start(model_family("row")); ER = tuple(1e-6 * 1.25 ** k for k in range(60))
    c0 = cert_start(fr, [fr.newton(w, -2.0) for w in ws], -2.0, rads=ER)
    if c0 is None: return dict(ok=False, reason="start not certified")
    try:
        steps, Bend, wend = track_window(fr, fc, [fr.newton(w, -2.0) for w in ws], [c[1] for c in c0], h0=0.005, hmax=0.02, hmin=1e-7, erads=ER, rads_all=ER, nr=12)
    except RuntimeError as e: return dict(ok=False, reason=str(e))
    disj = all(not st["br"][i]["s"].overlaps(st["br"][j]["s"]) for st in steps for i, j in itertools.combinations(range(4), 2))
    minsep = min(float(abs(st["br"][i]["s"] - st["br"][j]["s"]).lower()) for st in steps for i, j in itertools.combinations(range(4), 2))
    maxw = max(float(o["s"].rad()) for st in steps for o in st["br"])
    return dict(ok=bool(disj), n_steps=len(steps), min_separation_lower=minsep, max_value_radius=maxw)
def uniform(L0, Ns=(1, 4, 16)):
    """LB2/LB3: certify the four leading critical points of F_{tau, c} for every c in the box [0, e11] x [0, e12] (c11 split into N equal
    pieces if needed; the union of the pieces is the box), with disjoint value enclosures on every step."""
    e11, e12 = math.exp(1 - 0.15 * L0), math.exp(1 - 0.25 * L0); c12 = arb(e12 / 2, e12 / 2 * (1 + 1e-12))
    for N in Ns:
        parts = []
        for j in range(N):
            lo, hi = e11 * j / N, e11 * (j + 1) / N; c11 = arb((lo + hi) / 2, (hi - lo) / 2 * (1 + 1e-12) + 1e-300)
            r = uniform_sub(L0, c11, c12); parts.append(r)
            if not r["ok"]: break
        if all(r["ok"] for r in parts) and len(parts) == N:
            return dict(L0=L0, ok=True, eps=[e11, e12], pieces=N, min_separation_lower=min(r["min_separation_lower"] for r in parts),
                        max_value_radius=max(r["max_value_radius"] for r in parts), steps=sum(r["n_steps"] for r in parts))
    return dict(L0=L0, ok=False, eps=[e11, e12], pieces_tried=list(Ns), last=parts[-1])
@stage("S2_uniform")
def S2():
    r = {"40": uniform(40)}
    for L0 in (35, 30, 25, 20):
        r[str(L0)] = uniform(L0)
        if not r[str(L0)]["ok"]: break
    return r
def W_curve():
    """W along t = -2/L as L varies (p = L): coefficients e^{-L x0_k(0) + 2 dx_k}."""
    x00 = x0f(0.0); x00b = [x0b(k, arb(0)) for k in range(6)]; dxb = [FS[CELL[k][0]] * GA[CELL[k][1]] for k in range(6)]
    def cf(P):
        if isinstance(P, (arb, acb)): return [(-acb(P) * x00b[k] + 2 * dxb[k]).exp() for k in range(6)]
        return list(np.exp(-P * x00 + 2 * DXf))
    def dcf(P):
        if isinstance(P, (arb, acb)): return [-x00b[k] * (-acb(P) * x00b[k] + 2 * dxb[k]).exp() for k in range(6)]
        return list(-x00 * np.exp(-P * x00 + 2 * DXf))
    return Family(cf, cf, dcf, dcf)
def ustar(tau, L):
    t = tau / L
    if tau <= 0: h = 0.1 + t / 3; return np.array([(0.3 + t) * 0.35 - h, (0.3 + t) * 0.45 - h])
    return np.array([(0.3 + t) * 0.35 - 0.1 - t / 2, (0.3 + t) * 0.45 - 0.1 - t / 2])
def h1b(P, L, side): return (arb(1) / 10 + acb(P) / (3 * L)) if side == "row" else acb(arb(1) / 10)
def start_six(L):
    p = os.path.join(CK, f"start_L{L}.json")
    if os.path.exists(p):
        d = json.load(open(p)); return [np.array([complex(*w[0]), complex(*w[1])]) for w in d["w"]], [[des(z) for z in b] for b in d["B"]]
    s3 = json.load(open(os.path.join(HERE, "study45_example", "ckpt", "S3_points.json")))["TA"]
    ws = [np.array([complex(*w[0]), complex(*w[1])]) for w in s3["w"]]; B = [[box(complex(*w[0]), r), box(complex(*w[1]), r)] for w, r in zip(s3["w"], s3["rad"])]
    if L > 20:
        fam = W_curve(); steps, B, ws = track(fam, ws, B, 20.0, float(L), h0=0.05, hmax=2.0, rel=True)
    json.dump(dict(w=[[[w[0].real, w[0].imag], [w[1].real, w[1].imag]] for w in ws], B=[[ser(z) for z in b] for b in B]), open(p, "w"))
    return ws, B
def W_at(L):
    p = os.path.join(CK, f"S3_W_L{L}.json")
    if os.path.exists(p): say(f"[S3_W_L{L}] loaded from checkpoint"); return json.load(open(p))
    t0 = time.time(); say(f"[S3_W_L{L}] computing")
    fam = W_family(float(L)); ws, B = start_six(L)
    s1, B1, w1 = track(fam, ws, B, -2.0, 0.0, h0=0.005, hmax=0.02, rel=True); s2, B2, w2 = track(fam, w1, B1, 0.0, 2.0, h0=0.005, hmax=0.02, rel=True)
    steps = s1 + s2
    fr, fc = model_family("row"), model_family("col")
    # leading group: the four branches whose scaled value at tau = -2 is close to a model value
    us = ustar(-2.0, L); wsF = [fr.newton(w + L * us, -2.0) for w in ws]
    model0 = model_start(fr); mvals = [fr.s(w, -2.0) for w in model0]
    sc0 = math.exp(L * (0.1 - 2 / (3 * L)))
    lead = []
    for mv in mvals:
        d = [abs(steps[0]["br"][i]["wm"] is not None and (float(steps[0]["br"][i]["sa"].real.mid()) + 1j * float(steps[0]["br"][i]["sa"].imag.mid())) * sc0 - mv) for i in range(6)]
        lead.append(int(np.argmin(d)))
    # closeness on every step: |scaled W - model| < half the minimal model separation (model certified on the same step)
    worst = 0.0; ok = len(set(lead)) == 4; ratio_max = 0.0; zero = []
    for st in steps:
        side = "row" if st["pb"] <= 0 else "col"; mf = fr if side == "row" else fc; P = tball(st["pa"], st["pb"]); pm = (st["pa"] + st["pb"]) / 2
        Ms = []
        for i in lead:
            wF = mf.newton(fam.newton(st["br"][i]["wm"], pm) + L * ustar(pm, L), pm)
            c = None if wF is None else cert(mf, wF, P, pm, tuple(1e-12 * 1.25 ** k for k in range(120)))
            if c is None: ok = False; break
            Ms.append(mf.sb(c[1], P))
        if not ok: break
        scale = (L * h1b(P, L, side)).exp(); Sw = [st["br"][i]["s"] * scale for i in lead]
        sep = min(float(abs(Ms[a] - Ms[b]).lower()) for a, b in itertools.combinations(range(4), 2))
        dist = max(float(abs(Sw[a] - Ms[a]).upper()) for a in range(4)); worst = max(worst, dist); ratio_max = max(ratio_max, dist / (sep / 2))
        if dist >= sep / 2: ok = False
        for a in range(4):
            sa_, sb_ = st["br"][lead[a]]["sa"].real, st["br"][lead[a]]["sb"].real
            if sgn(sa_) * sgn(sb_) < 0: zero.append(dict(branch=lead[a], tau=[st["pa"], st["pb"]], im_upper=float(abs(st["br"][lead[a]]["s"].imag).upper()) if False else float(st["br"][lead[a]]["s"].imag.abs_upper()) if hasattr(st["br"][lead[a]]["s"].imag, "abs_upper") else None))
    # braid word of the leading four (projection reading on W values; positive scaling does not change the reading)
    for _ in range(8):
        view = [dict(pa=v["pa"], pb=v["pb"], s=[v["s"][i] for i in lead], sa=[v["sa"][i] for i in lead], sb=[v["sb"][i] for i in lead]) for v in live_view(steps)]
        word, perm, bad = read_word(view, zrot, 4)
        if not bad: break
        steps = refine_steps(fam, steps, set(bad), 6)
    else: word = None
    # full six-value Hurwitz word from the Study 45 base-point rule
    mx = max(float(o["s"].abs_upper()) for st in steps for o in st["br"]); R = 3 * mx; e = math.floor(math.log10(R)) - 1; Rq = Fr(math.ceil(R / 10 ** e)) * Fr(10) ** e
    s0 = acb(Q(Rq)) * acb(0, arb(3) / 10).exp()
    for _ in range(8):
        hw, hperm, bad = read_word(live_view(steps), zang(s0), 6)
        if not bad: break
        steps = refine_steps(fam, steps, set(bad), 6)
    else: hw = None
    hurw = None if hw is None else [dict(near=(l["left"] if l["e"] == 1 else l["right"]), far=(l["right"] if l["e"] == 1 else l["left"]), eps=(-1 if l["e"] == 1 else 1), p=l["p"],
                                         leading_only=bool((l["left"] in lead) and (l["right"] in lead))) for l in hw]
    r = dict(L=L, lead=lead, close_ok=bool(ok), max_dist=worst, max_dist_over_halfsep=ratio_max, word=word, perm=perm, R=str(Rq), hurwitz=hurw,
             zero_crossings=zero, n_steps=len(steps), seconds=round(time.time() - t0))
    json.dump(r, open(p, "w")); say(f"[S3_W_L{L}] done in {r['seconds']}s"); return r
if __name__ == "__main__" and "--s3" in sys.argv:
    for L in [int(x) for x in sys.argv[sys.argv.index("--s3") + 1].split(",")]:
        r = W_at(L); say({k: v for k, v in r.items() if k != "hurwitz"}); say("hurwitz", r["hurwitz"])
if __name__ == "__main__" and "--s2" in sys.argv:
    r = S2(); say(r)
def evaluate():
    s1 = json.load(open(os.path.join(CK, "S1_model.json"))); s2 = json.load(open(os.path.join(CK, "S2_uniform.json")))
    Ws = {L: json.load(open(os.path.join(CK, f"S3_W_L{L}.json"))) for L in (20, 40, 80, 160)}
    wl = "".join(f"s{l['k']}{'' if l['e'] == 1 else '^-1'} " for l in s1["word"]).strip()
    check("LB1 IDENTITY model braid of the leading families, certified, continuous at the wall, no collision", s1["perm"] is not None and len(s1["word"]) > 0,
          f"beta_lead = {wl} (projection reading, {s1['n_steps']} steps), permutation {s1['perm']}; leading-sector Hurwitz word from R' = {s1['R']}: "
          + ", ".join(f"near {h['near']} far {h['far']} eps {h['eps']:+d} at tau in [{h['p'][0]:.4f}, {h['p'][1]:.4f}]" for h in s1["hurwitz"])
          + f"; zero crossing at tau = {s1['zero_crossing_tau']:.5f} (exact {s1['zero_crossing_tau_exact']}), branch {s1['zero_branch']}")
    check("LB2 HYPOTHESIS uniform certificate for L0 = 40: leading critical values of F_{tau,c} disjoint on every step for all c in the box",
          s2["40"]["ok"], f"{s2['40']}")
    okL = [int(k) for k, v in s2.items() if v["ok"]]
    check("LB3 RECORD smallest L0 in {20, 25, 30, 35, 40} with the uniform certificate", True, f"smallest certified L0 = {min(okL) if okL else None}; tried {sorted(int(k) for k in s2)}; "
          + "; ".join(f"L0 = {k}: {'ok, ' + str(v['pieces']) + ' pieces' if v['ok'] else 'failed: ' + v['last'].get('reason', '')}" for k, v in s2.items()))
    A = artin(s1["word"]); res = {L: (W["close_ok"], W["word"] is not None and artin(W["word"]) == A) for L, W in Ws.items()}
    check("LB4 HYPOTHESIS direct confirmation on W at L = 20, 40, 80, 160: uniformly within half the model separation, and the same element of B_4",
          all(a and b for a, b in res.values()), "; ".join(f"L = {L}: close {res[L][0]} (max dist / half separation {Ws[L]['max_dist_over_halfsep']:.3g}), braid equal {res[L][1]}" for L in Ws))
    check("LB5 RECORD distances and the full six-value Hurwitz words", True, " || ".join(
        f"L = {L}: max |scaled W - model| = {W['max_dist']:.3g}; R = {W['R']}; word " + ", ".join(f"({h['near']},{h['far']},{h['eps']:+d}{',lead' if h['leading_only'] else ''})" for h in (W["hurwitz"] or []))
        for L, W in Ws.items()))
    check("LB6 RECORD zero crossing of a leading value: model and W", True, f"model tau = {s1['zero_crossing_tau']:.5f}; " + "; ".join(
        f"L = {L}: " + ", ".join(f"branch {z['branch']} tau in [{z['tau'][0]:.4f}, {z['tau'][1]:.4f}]" for z in W["zero_crossings"]) for L, W in Ws.items()))
if __name__ == "__main__" and "--eval" in sys.argv:
    say(f"==== evaluation {time.strftime('%Y-%m-%d %H:%M:%S')}"); evaluate()
