"""Study 45: one fully worked 2x3 example with every link exact or interval-certified (CE1-CE12). Pre-registered in
study45_example/PREREG_study45.md, frozen by SHA-256 before this code ran. Self-contained: it does not import the earlier study code, so it
also serves as an independent re-implementation of the Study 41/44 machinery.

Checkpointed by stage: every stage writes study45_example/ckpt/<stage>.json and is skipped on a rerun if that file exists
(delete the file, or pass --redo <stage>, to recompute it). Output: check_example45_output.txt (append the per-run lines).

Certification: Arb ball arithmetic (python-flint), Krawczyk operator K(X) = m - Y F(m, P) + (I - Y J(X, P)) (X - m) over complex (or real)
boxes, with the parameter P (t, L or a path parameter) an interval; K(X) in the interior of X proves a unique zero in X for every parameter
in P, and K(X) encloses it. Consecutive boxes are chained by requiring the enclosure at the shared parameter value to lie in the next box.
"""
import os, sys, json, math, itertools, time
from fractions import Fraction as Fr
import numpy as np
from flint import arb, acb, ctx
ctx.prec = 160
HERE = os.path.dirname(os.path.abspath(__file__)); D45 = os.path.join(HERE, "study45_example"); CK = os.path.join(D45, "ckpt")
os.makedirs(CK, exist_ok=True)
OUT = open(os.path.join(HERE, "check_example45_output.txt"), "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
LIB = "--lib" in sys.argv or "--s9worker" in sys.argv
REDO = set(sys.argv[sys.argv.index("--redo") + 1].split(",")) if "--redo" in sys.argv else set()
def stage(name):
    def deco(fn):
        def run():
            p = os.path.join(CK, name + ".json")
            if os.path.exists(p) and name not in REDO:
                say(f"[{name}] loaded from checkpoint"); return json.load(open(p))
            t0 = time.time(); say(f"[{name}] computing"); r = fn()
            json.dump(r, open(p, "w"), indent=1); say(f"[{name}] done in {time.time() - t0:.0f}s"); return r
        return run
    return deco
def check(name, cond, info=""):
    say(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))
    return bool(cond)

# ---------------------------------------------------------------- the example -------------------------------------------------------------
V = [(1, 1), (-1, 0), (0, -1), (-1, -1), (1, 0), (0, 1)]                       # exponents of cells 00, 01, 02, 10, 11, 12
CELL = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
GQ = [Fr(1, 5), Fr(7, 20), Fr(9, 20)]; FQ0 = [Fr(3, 10), Fr(7, 10)]; FS = [1, -1]   # f(t) = (3/10 + t, 7/10 - t)
TA, TB = Fr(-1, 10), Fr(1, 10); L0 = 20; H1 = Fr(1, 10)
Vf = np.array(V, float); gf = np.array([float(g) for g in GQ])
DX = np.array([FS[i] * gf[j] for i, j in CELL])                                 # d x0 / dt
def x0f(t): return np.array([(float(FQ0[i]) + FS[i] * t) * gf[j] for i, j in CELL])
def Q(fr): return arb(fr.numerator) / fr.denominator
GA = [Q(g) for g in GQ]; FA = [Q(f) for f in FQ0]; PI = arb.pi()
def x0b(k, t):                                                                  # ball version; t arb or acb
    i, j = CELL[k]; return (FA[i] + FS[i] * t) * GA[j]

# float model (approximations only; never used for a certified statement)
def Ef(w, t, L=L0): return np.exp(-L * x0f(t) + Vf @ w)
def Ff(w, t, L=L0): return Vf.T @ Ef(w, t, L)
def Jf(w, t, L=L0): E = Ef(w, t, L); return (Vf.T * E) @ Vf
def sf(w, t, L=L0): return Ef(w, t, L).sum()
def newton(w, t, L=L0, it=60):
    w = np.array(w, complex)
    for _ in range(it):
        try: st = np.linalg.solve(Jf(w, t, L), Ff(w, t, L))
        except np.linalg.LinAlgError: return None
        w = w - st
        if np.abs(st).max() < 1e-15 * (1 + np.abs(w).max()): break
    E = Ef(w, t, L)
    return w if np.abs(Vf.T @ E).max() < 1e-11 * np.abs(E).max() else None
def wnorm(w): w = np.array(w, complex); return w.real + 1j * ((w.imag + np.pi) % (2 * np.pi) - np.pi)
def all_crit(t, L=L0, seed=4501, starts=3000):
    rng = np.random.default_rng(seed); found = []
    for _ in range(starts):
        w = newton(rng.normal(0, 8.0, 2) + 1j * rng.uniform(-np.pi, np.pi, 2), t, L)
        if w is None: continue
        w = wnorm(w)
        if not any(np.abs(np.exp(w) - np.exp(q)).max() < 1e-8 * (1 + np.abs(np.exp(w)).max()) for q in found): found.append(w)
    return found

# ---------------------------------------------------------------- Krawczyk ------------------------------------------------------------------
def box(z, rad): return acb(arb(float(z.real), rad), arb(float(z.imag), rad))
def pt(z): return acb(float(z.real), float(z.imag))
def kraw(Ffun, Jfun, xm, X, Yinv):
    n = len(xm); Fm = Ffun(xm); JX = Jfun(X); K = []
    for i in range(n):
        k = xm[i] - sum(Yinv[i][j] * Fm[j] for j in range(n))
        for j in range(n):
            c = (1 if i == j else 0) - sum(Yinv[i][l] * JX[l][j] for l in range(n)); k = k + c * (X[j] - xm[j])
        K.append(k)
    return all(X[i].contains_interior(K[i]) for i in range(n)), K
def mat_inv_acb(M): Mi = np.linalg.inv(np.array(M, complex)); return [[acb(complex(Mi[i, j]).real, complex(Mi[i, j]).imag) for j in range(len(M))] for i in range(len(M))]
def terms(x, t, L):
    return [(-L * x0b(k, t) + V[k][0] * x[0] + V[k][1] * x[1]).exp() for k in range(6)]
def Fb(t, L):
    def F(x): E = terms(x, t, L); return [sum(V[k][m] * E[k] for k in range(6)) for m in range(2)]
    return F
def Jb(t, L):
    def J(x): E = terms(x, t, L); return [[sum(V[k][a] * V[k][b] * E[k] for k in range(6)) for b in range(2)] for a in range(2)]
    return J
def s_enc(X, t, L): return sum(terms(X, t, L))
def dsdt_enc(X, t, L): E = terms(X, t, L); return sum(-L * (FS[CELL[k][0]] * GA[CELL[k][1]]) * E[k] for k in range(6))
def cert_point(w, tball, tmid, L=L0, Lmid=None, rads=(1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3)):
    """Certify a unique critical point near the approximation w for every t in tball (and L in L if a ball). Returns (rad, K) or None."""
    Lm = L0 if Lmid is None else Lmid
    Y = mat_inv_acb(Jf(w, tmid, Lm)); xm = [pt(w[0]), pt(w[1])]
    for r in rads:
        X = [box(w[0], r), box(w[1], r)]
        ok, K = kraw(Fb(tball, L), Jb(tball, L), xm, X, Y)
        if ok: return r, X, K
    return None
def distinct_mod(Xa, Xb):
    """True if the boxes are disjoint modulo 2 pi i Z^2 (distinct points of the torus)."""
    for m in range(2):
        d = Xa[m] - Xb[m]
        if not d.real.contains(0): return True
        if not (d.imag / (2 * PI)).contains_integer(): return True
    return False
def contains_box(X, K): return all(X[m].contains(K[m]) for m in range(len(X)))
def tball(a, b): a, b = float(a), float(b); return arb((a + b) / 2, (b - a) / 2 * (1 + 1e-12) + 1e-30)
def ser(z):
    if isinstance(z, acb): return [z.real.str(40, radius=True), z.imag.str(40, radius=True)]
    return z.str(40, radius=True)
def des(o): return acb(arb(o[0]), arb(o[1])) if isinstance(o, list) else arb(o)
def cfl(z): return complex(float(z.real.mid()), float(z.imag.mid()))

# ================================================================ S1: the wall, exactly =====================================================
def peel_exact(f, g):
    fr = list(f); gc = list(g); R = [0, 1]; C = [0, 1, 2]; steps = []
    while len(R) > 1 and len(C) > 1:
        cand = sorted([(fr[i] / len(C), "r", i) for i in R] + [(gc[j] / len(R), "c", j) for j in C]); steps.append(cand); v, kind, idx = cand[0]
        if kind == "r":
            for j in C: gc[j] -= v
            R.remove(idx)
        else:
            for i in R: fr[i] -= v
            C.remove(idx)
    return steps
def lin_peel(ts):
    """Peeling with entries as linear forms (a, b) = a + b t, with the order decided at the sample point ts."""
    ev = lambda l: l[0] + l[1] * ts
    fr = [(FQ0[0], Fr(1)), (FQ0[1], Fr(-1))]; gc = [(g, Fr(0)) for g in GQ]; R = [0, 1]; C = [0, 1, 2]; forms = []
    while len(R) > 1 and len(C) > 1:
        cand = [((fr[i][0] / len(C), fr[i][1] / len(C)), "r", i) for i in R] + [((gc[j][0] / len(R), gc[j][1] / len(R)), "c", j) for j in C]
        forms.append(cand); m = min(cand, key=lambda c: ev(c[0])); (va, vb), kind, idx = m
        if kind == "r":
            for j in C: gc[j] = (gc[j][0] - va, gc[j][1] - vb)
            R.remove(idx)
        else:
            for i in R: fr[i] = (fr[i][0] - va, fr[i][1] - vb)
            C.remove(idx)
    return forms
def eq_points(cand, lo, hi):
    out = set()
    for (l1, _, _), (l2, _, _) in itertools.combinations(cand, 2):
        if l1[1] != l2[1]:
            t = (l2[0] - l1[0]) / (l1[1] - l2[1])
            if lo <= t <= hi: out.add(t)
    return out
@stage("S1_wall")
def S1():
    bps = {TA, TB} | eq_points(lin_peel(TA)[0], TA, TB)
    changed = True
    while changed:
        changed = False; pts = sorted(bps)
        for a, b in zip(pts, pts[1:]):
            for cand in lin_peel((a + b) / 2):
                new = eq_points(cand, a, b) - bps
                if new: bps |= new; changed = True
    pts = sorted(bps); probe = sorted(set(pts) | {(a + b) / 2 for a, b in zip(pts, pts[1:])}); ties = []
    for t in probe:
        st = peel_exact([FQ0[0] + t, FQ0[1] - t], GQ)
        for lev, cand in enumerate(st):
            mins = [c for c in cand if c[0] == cand[0][0]]
            if len(mins) > 1: ties.append(dict(t=str(t), step=lev + 1, tied=[f"{k}{i}" for _, k, i in mins], value=str(cand[0][0])))
    return dict(breakpoints=[str(p) for p in pts], ties=ties)

# ================================================================ S2: the tied-cell polynomial, exactly =====================================
@stage("S2_leading")
def S2():
    import sympy as sp
    Y, b, a, s = sp.symbols("Y b a s")
    # row side f_b = Y1Y2 + 1/Y1 + 1/Y2 + b/(Y1Y2); its critical points have Y1 = Y2 = Y with Y^4 - Y - b = 0, value s = 2Y^2 + 1/Y
    Prow = sp.factor(sp.resultant(Y**4 - Y - b, s * Y - 2 * Y**3 - 1, Y))
    # column side g_a = Y1Y2 + 1/(Y1Y2) + a(1/Y1 + 1/Y2): Y1 = Y2 = Y with Y^4 - aY - 1 = 0, value s = Y^2 + Y^-2 + 2a/Y = 2Y^2 + a/Y
    Pcol = sp.factor(sp.resultant(Y**4 - a * Y - 1, s * Y - 2 * Y**3 - a, Y))
    # sanity: the critical equations of the two-variable forms reduce to these (checked symbolically)
    Y1, Y2 = sp.symbols("Y1 Y2")
    fb = Y1 * Y2 + 1 / Y1 + 1 / Y2 + b / (Y1 * Y2); ga = Y1 * Y2 + 1 / (Y1 * Y2) + a * (1 / Y1 + 1 / Y2)
    r1 = sp.simplify((Y1 * sp.diff(fb, Y1) - Y2 * sp.diff(fb, Y2)) * Y1 * Y2 - (Y1 - Y2))
    r2 = sp.simplify((Y1 * sp.diff(ga, Y1) - Y2 * sp.diff(ga, Y2)) * Y1 * Y2 - a * (Y1 - Y2))
    wall = sp.Poly(sp.expand(Prow.subs(b, 1) * 1), s); wall_c = sp.Poly(sp.expand(Pcol.subs(a, 1)), s)
    from flint import fmpz_poly
    coeffs = [int(c) for c in reversed(wall.all_coeffs())]; roots = fmpz_poly(coeffs).complex_roots()
    vals = []
    for z, mult in roots:
        r_ = abs(cfl(z)); ph = math.atan2(cfl(z).imag, cfl(z).real) / math.pi; vals.append(dict(ball=ser(z), modulus=round(r_, 4), phase_over_pi=round(ph, 4), mult=mult))
    zero_row = sp.solve(sp.Eq(Prow.subs(s, 0), 0), b); zero_col = sp.solve(sp.Eq(Pcol.subs(s, 0), 0), a)
    disc_row = sp.factor(sp.discriminant(Y**4 - Y - b, Y)); disc_col = sp.factor(sp.discriminant(Y**4 - a * Y - 1, Y))
    return dict(P_row=str(Prow), P_col=str(Pcol), reduction_ok=bool(r1 == 0 and r2 == 0), wall_poly=str(wall.as_expr()),
                wall_equal_from_both_sides=bool(sp.expand(wall.as_expr() - wall_c.as_expr()) == 0), wall_values=vals,
                zero_row=[str(z) for z in zero_row], zero_col=[str(z) for z in zero_col], disc_row=str(disc_row), disc_col=str(disc_col))

# ================================================================ S3: certified critical points at three parameters =========================
def certify_all(t_exact_ball, tmid, L=L0, Lmid=None, seed=4501):
    ws = all_crit(tmid, L0 if Lmid is None else Lmid, seed)
    out = []
    for w in ws:
        c = cert_point(w, t_exact_ball, tmid, L, Lmid)
        if c is None: return ws, None
        out.append((w, c))
    return ws, out
@stage("S3_points")
def S3():
    res = {}
    for name, tq in (("TA", TA), ("T0", Fr(0)), ("TB", TB)):
        ws, out = certify_all(Q(tq), float(tq))
        if out is None: res[name] = dict(n_found=len(ws), certified=False); continue
        Xs = [X for _, (r, X, K) in out]
        disj = all(distinct_mod(Xs[i], Xs[j]) for i in range(len(Xs)) for j in range(i + 1, len(Xs)))
        svals = [s_enc(K, Q(tq), L0) for _, (r, X, K) in out]
        sdis = all(not svals[i].overlaps(svals[j]) for i in range(len(svals)) for j in range(i + 1, len(svals)))
        res[name] = dict(n_found=len(ws), certified=True, disjoint=disj, values_disjoint=sdis, w=[[[w[0].real, w[0].imag], [w[1].real, w[1].imag]] for w, _ in out],
                         rad=[c[0] for _, c in out], K=[[ser(k) for k in c[2]] for _, c in out], s=[ser(v) for v in svals])
    return res

# ================================================================ S4: certified continuation along the segment ==============================
def dwdt(w, t, L=L0): E = Ef(w, t, L); return -np.linalg.solve(Jf(w, t, L), Vf.T @ (E * (-L * DX)))
def cert_step(ws, ta, tb, L=L0):
    """Certify all branches on [ta, tb] (floats). ws: approximations near ta. Returns step dict or None."""
    tm = (ta + tb) / 2; T = tball(ta, tb); h = tb - ta; out = []
    for w in ws:
        wm = newton(w + dwdt(w, ta, L) * (tm - ta), tm, L)
        if wm is None: return None
        sp = np.abs(dwdt(wm, tm, L)).max() * h / 2
        rads = [r for r in (1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2) if r > 1.5 * sp]
        c = cert_point(wm, T, tm, rads=tuple(rads[:4]))
        if c is None: return None
        r, X, K = c
        ends = []
        for te in (ta, tb):
            we = newton(wm, te)
            ce = None if we is None else cert_point(we, arb(te), te, rads=(1e-12, 1e-10, 1e-8))
            if ce is None or not contains_box(X, ce[1]): return None
            ends.append(ce)
        (_, Ba, Ka), (_, Bb, Kb) = ends
        out.append(dict(wm=wm, r=r, X=X, Ba=Ba, Bb=Bb, s=s_enc(X, T, L0), ds=dsdt_enc(X, T, L0), sa=s_enc(Ka, arb(ta), L0), sb=s_enc(Kb, arb(tb), L0)))
    Xs = [o["X"] for o in out]
    if not all(distinct_mod(Xs[i], Xs[j]) for i in range(len(Xs)) for j in range(i + 1, len(Xs))): return None
    return dict(ta=ta, tb=tb, br=out)
def step_json(st):
    return dict(ta=st["ta"], tb=st["tb"], wm=[[[o["wm"][m].real, o["wm"][m].imag] for m in range(2)] for o in st["br"]], r=[o["r"] for o in st["br"]],
                s=[ser(o["s"]) for o in st["br"]], ds=[ser(o["ds"]) for o in st["br"]], sa=[ser(o["sa"]) for o in st["br"]], sb=[ser(o["sb"]) for o in st["br"]])
def track(ws0, Kstart, a, b, h0=1e-3, hmax=5e-3, hmin=1e-9, Lb=L0):
    """Chained certified continuation of all branches over [a, b]. Kstart: enclosures of the roots at a (must lie in the first boxes)."""
    t = a; ws = [np.array(w) for w in ws0]; Kprev = Kstart; Bprev = None; h = h0; steps = []
    while t < b:
        tb_ = min(t + h, b); st = cert_step(ws, t, tb_)
        if st is None or not all(contains_box(o["X"], Kprev[i]) and contains_box(o["X"], o["Ba"]) and (Bprev is None or contains_box(o["X"], Bprev[i])) for i, o in enumerate(st["br"])):
            h /= 2
            if h < hmin: raise RuntimeError(f"tracking failed at t = {t}")
            continue
        steps.append(st); Kprev = [o["Bb"] for o in st["br"]]; Bprev = Kprev; ws = [newton(o["wm"] + dwdt(o["wm"], (t + tb_) / 2) * (tb_ - t) / 2, tb_) for o in st["br"]]
        t = tb_; h = min(h * 1.5, hmax)
    return steps, Kprev
@stage("S4_track")
def S4():
    s3 = json.load(open(os.path.join(CK, "S3_points.json")))
    wsA = [np.array([complex(*w[0]), complex(*w[1])]) for w in s3["TA"]["w"]]; KA = [[des(k) for k in K] for K in s3["TA"]["K"]]
    a, b = float(TA), float(TB)                                                 # the first/last balls cover the exact endpoints (tball slack)
    steps, Kend = track(wsA, KA, a, b)
    KB = [[des(k) for k in K] for K in s3["TB"]["K"]]; Xlast = [o["X"] for o in steps[-1]["br"]]
    perm = []                                                                   # branch i ends at the S3 point j at t = 1/10 (exact)
    for i in range(6):
        hits = [j for j in range(6) if contains_box(Xlast[i], KB[j])]; perm.append(hits[0] if len(hits) == 1 else None)
    return dict(n_steps=len(steps), perm_to_TB=perm, steps=[step_json(s) for s in steps])

# ================================================================ S5: the complete braid word as seen from s0 ===============================
def live_step(sj):
    """Re-certify a stored step (for refinement), returning live enclosures."""
    ws = [np.array([complex(*m[0]), complex(*m[1])]) for m in sj["wm"]]
    return ws
def phi(zi, zj): return (zi * zj.conjugate()).imag
def dphi(zi, zj, dzi, dzj): return (dzi * zj.conjugate() + zi * dzj.conjugate()).imag
def sgn(x):
    if x > 0: return 1
    if x < 0: return -1
    return 0
def base_point():
    s4 = json.load(open(os.path.join(CK, "S4_track.json")))
    mx = max(float(des(z).abs_upper()) for st in s4["steps"] for z in st["s"])
    R = 3 * mx; e = math.floor(math.log10(R)) - 1; Rq = Fr(math.ceil(R / 10 ** e)) * Fr(10) ** e
    return Rq, acb(Q(Rq)) * acb(0, arb(3) / 10).exp()
def analyse_interval(ta, tb, ws, s0, parentX=None, depth=0, maxdepth=14):
    """Return list of (status, data) for the pairs on [ta, tb]; refine by bisection when undecided."""
    st = cert_step(ws, ta, tb)
    if st is None or (parentX is not None and not all(contains_box(parentX[i], o["Ba"]) for i, o in enumerate(st["br"]))):
        raise RuntimeError(f"refinement certification failed on [{ta}, {tb}]")
    z = [o["s"] - s0 for o in st["br"]]; za = [o["sa"] - s0 for o in st["br"]]; zb = [o["sb"] - s0 for o in st["br"]]; dz = [o["ds"] for o in st["br"]]
    undecided = False; events = []
    for i, j in itertools.combinations(range(6), 2):
        f = phi(z[i], z[j])
        if not f.contains(0): continue
        fa, fb = sgn(phi(za[i], za[j])), sgn(phi(zb[i], zb[j])); fd = dphi(z[i], z[j], dz[i], dz[j])
        if fa and fb and not fd.contains(0):
            if fa == fb: continue                                               # monotone, same sign at both ends: no crossing
            others_ok = all((not phi(z[i], z[k]).contains(0)) and (not phi(z[j], z[k]).contains(0)) and sgn(phi(za[i], za[k])) == sgn(phi(za[j], za[k]))
                            for k in range(6) if k not in (i, j))
            di = (z[i] * z[i].conjugate()).real; dj = (z[j] * z[j].conjugate()).real
            if others_ok and not di.overlaps(dj):
                near, far = (i, j) if di < dj else (j, i)
                eps = sgn(dphi(z[far], z[near], dz[far], dz[near]))
                events.append(dict(ta=ta, tb=tb, near=near, far=far, eps=eps)); continue
        undecided = True; break
    if not undecided: return events
    if depth >= maxdepth: raise RuntimeError(f"undecided pair on [{ta}, {tb}]")
    tm = (ta + tb) / 2; X = [o["X"] for o in st["br"]]
    wsm = [newton(o["wm"], tm) for o in st["br"]]
    return analyse_interval(ta, tm, [newton(o["wm"], ta) for o in st["br"]], s0, X, depth + 1) + analyse_interval(tm, tb, wsm, s0, X, depth + 1)
@stage("S5_braid")
def S5():
    s4 = json.load(open(os.path.join(CK, "S4_track.json"))); Rq, s0 = base_point(); nu = -s0
    cone = all(((des(z) - s0) * nu.conjugate()).real > 0 for st in s4["steps"] for z in st["s"])
    za = [des(z) - s0 for z in s4["steps"][0]["sa"]]
    order = sorted(range(6), key=lambda i: sum(1 for k in range(6) if k != i and phi(za[i], za[k]) > 0))   # theta increasing
    order_ok = all(not phi(za[i], za[k]).contains(0) for i, k in itertools.combinations(range(6), 2))
    events = []
    for st in s4["steps"]:
        ws = [np.array([complex(*m[0]), complex(*m[1])]) for m in st["wm"]]
        ws = [newton(w, st["ta"]) for w in ws]
        z = [des(x) - s0 for x in st["s"]]
        if all(not phi(z[i], z[j]).contains(0) for i, j in itertools.combinations(range(6), 2)): continue
        events += analyse_interval(st["ta"], st["tb"], ws, s0)
    word = []; cur = list(order); ok_adj = True
    for e in sorted(events, key=lambda e: e["ta"]):
        p, q = cur.index(e["near"]), cur.index(e["far"])
        if abs(p - q) != 1: ok_adj = False
        k = min(p, q); cur[p], cur[q] = cur[q], cur[p]; word.append(dict(e, pos=k))
    zb = [des(z) - s0 for z in s4["steps"][-1]["sb"]]
    order_end = sorted(range(6), key=lambda i: sum(1 for k in range(6) if k != i and phi(zb[i], zb[k]) > 0))
    return dict(R=str(Rq), cone=bool(cone), order_start=order, order_start_certified=bool(order_ok), events=word, order_after_word=cur,
                order_end=order_end, adjacent=bool(ok_adj))

# ================================================================ S6: real critical values crossing 0 =======================================
def real_funcs(m, t, L=L0):
    sg = [(-1) ** ((V[k][0] * m[0] + V[k][1] * m[1]) % 2) for k in range(6)]
    def E(x): return [sg[k] * (-L * x0b(k, t) + V[k][0] * x[0] + V[k][1] * x[1]).exp() for k in range(6)]
    def F(x): e = E(x); return [sum(V[k][a] * e[k] for k in range(6)) for a in range(2)]
    def J(x): e = E(x); return [[sum(V[k][a] * V[k][b] * e[k] for k in range(6)) for b in range(2)] for a in range(2)]
    def S(x): return sum(E(x))
    def dS(x): e = E(x); return sum(-L * (FS[CELL[k][0]] * GA[CELL[k][1]]) * e[k] for k in range(6))
    return F, J, S, dS
def cert_real(x, m, T, tm, rads):
    w = np.array([x[0] + 1j * np.pi * m[0], x[1] + 1j * np.pi * m[1]]); Jm = Jf(w, tm).real; Yi = np.linalg.inv(Jm)
    Y = [[arb(float(Yi[a, b])) for b in range(2)] for a in range(2)]; F, J, S, dS = real_funcs(m, T); xm = [arb(float(x[0])), arb(float(x[1]))]
    for r in rads:
        X = [arb(float(x[0]), r), arb(float(x[1]), r)]; ok, K = kraw(F, J, xm, X, Y)
        if ok: return r, X, K, S, dS
    return None
def newton_real(x, m, t): w = newton(np.array([x[0] + 1j * np.pi * m[0], x[1] + 1j * np.pi * m[1]]), t); return None if w is None else w.real
def real_crossings(x0_, m, a, b, Bstart, h0=1e-3, hmax=5e-3):
    """Certified real continuation of one real branch on [a, b]; returns the number of certified simple zeros of s and their intervals."""
    t = a; x = np.array(x0_); h = h0; Bprev = Bstart; zeros = []; nsteps = 0
    def step(x, ta, tb):
        tm = (ta + tb) / 2; xm_ = newton_real(x, m, tm)
        if xm_ is None: return None
        sp = np.abs(dwdt(np.array([xm_[0] + 1j * np.pi * m[0], xm_[1] + 1j * np.pi * m[1]]), tm)).max() * (tb - ta) / 2
        c = cert_real(xm_, m, tball(ta, tb), tm, tuple(r for r in (1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2) if r > 1.5 * sp)[:4])
        if c is None: return None
        r, X, K, S, dS = c; ends = []
        for te in (ta, tb):
            xe = newton_real(xm_, m, te); ce = None if xe is None else cert_real(xe, m, arb(te), te, (1e-12, 1e-10, 1e-8))
            if ce is None or not all(X[k].contains(ce[1][k]) for k in range(2)): return None
            ends.append(ce)
        Sa = ends[0][3](ends[0][2]); Sb = ends[1][3](ends[1][2])
        return dict(xm=xm_, X=X, Ba=ends[0][1], Bb=ends[1][1], s=S(X), ds=dS(X), sa=Sa, sb=Sb)
    def decide(x, ta, tb, X_parent, depth):
        st = step(x, ta, tb)
        if st is None or (X_parent is not None and not all(X_parent[k].contains(st["Ba"][k]) for k in range(2))): raise RuntimeError("real refinement failed")
        if not st["s"].contains(0): return []
        fa, fb = sgn(st["sa"]), sgn(st["sb"])
        if fa and fb and not st["ds"].contains(0): return [] if fa == fb else [(ta, tb, fa, fb)]
        if depth > 14: raise RuntimeError("real zero undecided")
        tm = (ta + tb) / 2
        return decide(newton_real(st["xm"], m, ta), ta, tm, st["X"], depth + 1) + decide(newton_real(st["xm"], m, tm), tm, tb, st["X"], depth + 1)
    first = None
    while t < b:
        tb_ = min(t + h, b); st = step(x, t, tb_)
        if st is None or (Bprev is not None and not all(st["X"][k].contains(Bprev[k]) for k in range(2))):
            h /= 2
            if h < 1e-9: raise RuntimeError(f"real tracking failed at {t}")
            continue
        if first is None: first = st
        zeros += decide(newton_real(st["xm"], m, t), t, tb_, None, 0) if st["s"].contains(0) else []
        Bprev = st["Bb"]; x = newton_real(st["xm"], m, tb_); t = tb_; h = min(h * 1.5, hmax); nsteps += 1
    return zeros, nsteps, first
@stage("S6_real")
def S6():
    s3 = json.load(open(os.path.join(CK, "S3_points.json"))); out = []
    for lab, w in enumerate(s3["TA"]["w"]):
        im = [w[0][1], w[1][1]]
        if not all(min(abs(v), abs(abs(v) - np.pi)) < 1e-9 for v in im): out.append(dict(label=lab, real=False)); continue
        m = [0 if abs(v) < 1e-9 else 1 for v in im]
        cs = cert_real(np.array([w[0][0], w[1][0]]), m, Q(TA), float(TA), (1e-12, 1e-10, 1e-8))      # tight real box at the exact t = -1/10
        X3 = [box(complex(*w[0]), s3["TA"]["rad"][lab]), box(complex(*w[1]), s3["TA"]["rad"][lab])]
        lifted_in = cs is not None and all(X3[c].real.contains(cs[1][c]) and any(X3[c].imag.contains(PI * m[c] + 2 * PI * kk) for kk in (-1, 0, 1)) for c in range(2))
        zeros, n, first = real_crossings([w[0][0], w[1][0]], m, float(TA), float(TB), cs[1])
        out.append(dict(label=lab, real=True, lifted_in_complex_box=bool(lifted_in), signs=m, steps=n, zeros=[dict(ta=z[0], tb=z[1], sign_before=z[2], sign_after=z[3]) for z in zeros]))
    return dict(branches=out)

# ================================================================ S7: the fold (certified A2 point) =========================================
CK_ = -20.0 * DX                                                                # d/dt of the exponents (L = 20)
def Gf(z):
    w, t = z[:2], z[2]; E = Ef(w, t); J = (Vf.T * E) @ Vf
    return np.array([*(Vf.T @ E), J[0, 0] * J[1, 1] - J[0, 1] ** 2])
def DGf(z):
    w, t = z[:2], z[2]; E = Ef(w, t); J = (Vf.T * E) @ Vf; M = np.zeros((3, 3), complex)
    M[:2, :2] = J; M[:2, 2] = Vf.T @ (E * CK_)
    def dJ(weights): return (Vf.T * (E * weights)) @ Vf
    for c, wts in enumerate([Vf[:, 0], Vf[:, 1], CK_]):
        d = dJ(wts); M[2, c] = d[0, 0] * J[1, 1] + J[0, 0] * d[1, 1] - 2 * J[0, 1] * d[0, 1]
    return M
def Gb(z):
    w, t = z[:2], z[2]; E = terms(w, t, L0); J = [[sum(V[k][a] * V[k][b] * E[k] for k in range(6)) for b in range(2)] for a in range(2)]
    return [sum(V[k][0] * E[k] for k in range(6)), sum(V[k][1] * E[k] for k in range(6)), J[0][0] * J[1][1] - J[0][1] ** 2]
def DGb(z):
    w, t = z[:2], z[2]; E = terms(w, t, L0); ck = [-L0 * (FS[CELL[k][0]] * GA[CELL[k][1]]) for k in range(6)]
    J = [[sum(V[k][a] * V[k][b] * E[k] for k in range(6)) for b in range(2)] for a in range(2)]
    M = [[J[0][0], J[0][1], sum(V[k][0] * ck[k] * E[k] for k in range(6))], [J[1][0], J[1][1], sum(V[k][1] * ck[k] * E[k] for k in range(6))], [None] * 3]
    for c in range(3):
        wt = [V[k][c] if c < 2 else ck[k] for k in range(6)]
        d = [[sum(V[k][a] * V[k][b] * wt[k] * E[k] for k in range(6)) for b in range(2)] for a in range(2)]
        M[2][c] = d[0][0] * J[1][1] + J[0][0] * d[1][1] - 2 * J[0][1] * d[0][1]
    return M
@stage("S7_fold")
def S7():
    sols = []
    for re_ in (-0.1, -0.05, 0.0, 0.05, 0.1):
        for im_ in (0.01, 0.03, 0.06, 0.1, 0.15, 0.2):
            t0 = complex(re_, im_)
            for w in all_crit(re_, starts=1500):
                z = np.array([w[0], w[1], t0], complex)
                for _ in range(80):
                    try: st = np.linalg.solve(DGf(z), Gf(z))
                    except np.linalg.LinAlgError: break
                    z = z - st
                    if not np.isfinite(z).all(): break
                    if np.abs(st).max() < 1e-14 * (1 + np.abs(z).max()): break
                if not np.isfinite(z).all() or np.abs(Gf(z)).max() > 1e-10 * (1 + np.abs(Ef(z[:2], z[2])).max() ** 2): continue
                t = z[2]
                if not (0 < t.imag <= 0.2 + 1e-9 and abs(t.real) <= 0.1 + 1e-9): continue
                if not any(abs(t - q[2]) < 1e-8 for q in sols): sols.append(z)
    sols.sort(key=lambda z: abs(z[2])); zs = sols[0]
    Yi = np.linalg.inv(DGf(zs)); Y = [[acb(complex(Yi[a, b]).real, complex(Yi[a, b]).imag) for b in range(3)] for a in range(3)]
    xm = [pt(v) for v in zs]; cert = None
    for r in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        X = [box(v, r) for v in zs]; ok, K = kraw(Gb, DGb, xm, X, Y)
        if ok: cert = (r, X, K); break
    res = dict(found=[[[z[0].real, z[0].imag], [z[1].real, z[1].imag], [z[2].real, z[2].imag]] for z in sols], certified=cert is not None)
    if cert is None: return res
    r, X, K = cert; w, t = K[:2], K[2]
    E = terms(w, t, L0); J = [[sum(V[k][a] * V[k][b] * E[k] for k in range(6)) for b in range(2)] for a in range(2)]
    rank1 = not J[0][0].contains(0); kv = [-J[0][1], J[0][0]]
    cubic = sum(E[k] * (V[k][0] * kv[0] + V[k][1] * kv[1]) ** 3 for k in range(6))
    res.update(rad=r, K=[ser(k) for k in K], J11=ser(J[0][0]), cubic=ser(cubic), rank_one=bool(rank1), cubic_nonzero=bool(not cubic.contains(0)),
               s_star=ser(s_enc(K[:2], t, L0)), t_star=[float(t.real.mid()), float(t.imag.mid())])
    return res

# ================================================================ S8: the loop around the fold =============================================
def track_path(ws0, Bstart, pf, pb, dpf, h0=2e-3, hmax=1e-2):
    """Chained certified continuation of all branches along t = pf(theta), theta in [0, 1]; pb gives the ball version. Returns steps."""
    th = 0.0; ws = [np.array(w) for w in ws0]; Bprev = Bstart; h = h0; steps = []
    while th < 1.0:
        tb_ = min(th + h, 1.0); tm = (th + tb_) / 2; P = pb(tball(th, tb_)); out = []; fail = False
        for i, w in enumerate(ws):
            wm = newton(w + dwdt(w, pf(th)) * dpf(th) * (tm - th), pf(tm))
            if wm is None: fail = True; break
            sp = np.abs(dwdt(wm, pf(tm)) * dpf(tm)).max() * h / 2
            c = cert_point(wm, P, pf(tm), rads=tuple(r for r in (1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2) if r > 1.5 * sp)[:4])
            if c is None: fail = True; break
            r, X, K = c; ends = []
            for te in (th, tb_):
                we = newton(wm, pf(te)); ce = None if we is None else cert_point(we, pb(arb(te)), pf(te), rads=(1e-12, 1e-10, 1e-8))
                if ce is None or not contains_box(X, ce[1]): fail = True; break
                ends.append(ce)
            if fail or not contains_box(X, Bprev[i]): fail = True; break
            out.append(dict(wm=wm, X=X, Ba=ends[0][1], Bb=ends[1][1], s=s_enc(X, P, L0), sa=s_enc(ends[0][2], pb(arb(th)), L0), sb=s_enc(ends[1][2], pb(arb(tb_)), L0)))
        if not fail:
            Xs = [o["X"] for o in out]; fail = not all(distinct_mod(Xs[i], Xs[j]) for i in range(6) for j in range(i + 1, 6))
        if fail:
            h /= 2
            if h < 1e-10: raise RuntimeError(f"path tracking failed at theta = {th}")
            continue
        steps.append(dict(ta=th, tb=tb_, br=out)); Bprev = [o["Bb"] for o in out]; ws = [newton(o["wm"], pf(tb_)) for o in out]
        th = tb_; h = min(h * 1.5, hmax)
    return steps
@stage("S8_loop")
def S8():
    s7 = json.load(open(os.path.join(CK, "S7_fold.json"))); ts_ = complex(*s7["t_star"]); rho = 0.3 * ts_.imag
    tsb = acb(ts_.real, ts_.imag); rhob = arb(rho)
    pf = lambda th: ts_ + rho * np.exp(2j * np.pi * th)
    pb = lambda TH: tsb + rhob * (acb(0, 2) * acb(arb.pi()) * acb(TH)).exp()
    dpf = lambda th: 2j * np.pi * rho * np.exp(2j * np.pi * th)
    ws = all_crit(pf(0.0)); start = [cert_point(w, pb(arb(0)), pf(0.0), rads=(1e-12, 1e-10, 1e-8)) for w in ws]
    if len(ws) != 6 or any(c is None for c in start): return dict(ok=False, n=len(ws))
    B0 = [c[1] for c in start]
    steps = track_path(ws, B0, pf, pb, dpf)
    Bend = [o["Bb"] for o in steps[-1]["br"]]
    # identify modulo 2 pi i: the end box of branch i and the start box of branch j hold the same torus point iff they are not distinct_mod
    perm = [[j for j in range(6) if not distinct_mod(B0[j], Bend[i])] for i in range(6)]
    moved = [i for i in range(6) if perm[i] != [i]]
    res = dict(ok=True, rho=rho, n_steps=len(steps), perm=perm, moved=moved)
    if len(moved) == 2:
        a, b = moved; total = arb(0); ok_w = True
        for st in steps:
            za = st["br"][a]["sa"] - st["br"][b]["sa"]; zb = st["br"][a]["sb"] - st["br"][b]["sb"]; Z = st["br"][a]["s"] - st["br"][b]["s"]
            if not ((Z * za.conjugate()).real > 0): ok_w = False; break
            total = total + (zb / za).arg()
        res.update(winding_ok=ok_w, total_turn_over_pi=ser(total / PI) if ok_w else None)
    return res

# ================================================================ S9: vanishing classes at both ends ========================================
ROTc = np.exp(-0.7j); ROTb = acb(0, -arb(7) / 10).exp()
TAm = np.array([[1, 1], [0, 1]]); TBm = np.array([[1, 0], [-1, 1]]); GEN = {1: TAm, 2: TBm, 3: TAm}
def inv2(M): return np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]])
def coef_at(tq):                                                                # a_k at an exact rational t, ball and float
    ab = [(-L0 * x0b(k, Q(tq))).exp() for k in range(6)]; return ab, np.array([float(x.mid()) for x in ab])
def Dpoly_f(af, s):                                                             # coefficients (highest first) of D(y; s), float
    Bq = np.array([af[4], -s, af[1]], complex); A = np.array([af[0], af[5], 0], complex); C = np.array([af[2], af[3]], complex)
    return np.polysub(np.polymul(Bq, Bq), 4 * np.polymul(A, C))
def D_b(ab, y, s): Bq = ab[4] * y * y - s * y + ab[1]; return Bq * Bq - 4 * (ab[0] * y * y + ab[5] * y) * (ab[2] * y + ab[3])
def dDy_b(ab, y, s): Bq = ab[4] * y * y - s * y + ab[1]; return 2 * Bq * (2 * ab[4] * y - s) - 4 * ((2 * ab[0] * y + ab[5]) * (ab[2] * y + ab[3]) + (ab[0] * y * y + ab[5] * y) * ab[2])
def dDs_b(ab, y, s): return -2 * (ab[4] * y * y - s * y + ab[1]) * y
def roots_f(af, s): return np.roots(Dpoly_f(af, s))
def cert_root(ab, af, y, S, sm, rads):
    der = np.polyval(np.polyder(Dpoly_f(af, sm)), y); Y = acb(complex(1 / der).real, complex(1 / der).imag); ym = pt(y)
    for r in rads:
        X = box(y, r); K = ym - Y * D_b(ab, ym, S) + (1 - Y * dDy_b(ab, X, S)) * (X - ym)
        if X.contains_interior(K): return r, X, K
    return None
def newton_root(af, y, s):
    p = Dpoly_f(af, s); dp = np.polyder(p)
    for _ in range(50):
        st = np.polyval(p, y) / np.polyval(dp, y); y = y - st
        if abs(st) < 1e-15 * (1 + abs(y)): break
    return y
def braid_cert(ab, af, sf_, sb_, dsf, dsb, corners=(), h0=2e-3, hmax=2e-2):
    """Certified tracking of the 4 branch points along s = sf_(tau), tau in [0, 1]; returns the SL2(Z) matrix of the braid."""
    tau = 0.0; ys = list(roots_f(af, sf_(0.0))); M = np.eye(2, dtype=int); h = h0; Bprev = None; nst = 0; nev = 0; cur = None
    while tau < 1.0:
        tb_ = min([tau + h, 1.0] + [c for c in corners if c > tau + 1e-15]); tm = (tau + tb_) / 2; TT = tball(tau, tb_); S = sb_(TT); dS = dsb(TT); fail = False; out = []
        for i, y in enumerate(ys):
            yv = newton_root(af, y, sf_(tm)); p = Dpoly_f(af, sf_(tm))
            dyds = -(-2 * (af[4] * yv ** 2 - sf_(tm) * yv + af[1]) * yv) / np.polyval(np.polyder(p), yv)
            sp = abs(dyds * dsf(tm)) * h / 2
            sc = abs(yv)                                                        # radii relative to the size of the root
            c = cert_root(ab, af, yv, S, sf_(tm), tuple(r * sc for r in (1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1) if r * sc > 1.5 * sp)[:4])
            if c is None: fail = True; break
            r, X, K = c; ends = []
            for te in (tau, tb_):
                ce = cert_root(ab, af, newton_root(af, yv, sf_(te)), sb_(arb(te)), sf_(te), (1e-13 * sc, 1e-11 * sc, 1e-9 * sc))
                if ce is None or not X.contains(ce[1]): fail = True; break
                ends.append(ce)
            if fail or (Bprev is not None and not X.contains(Bprev[i])): fail = True; break
            dyS = -dDs_b(ab, X, S) / dDy_b(ab, X, S) * dS                      # dy/dtau over the step
            out.append(dict(y=yv, X=X, dy=dyS, Ba=ends[0][1], Bb=ends[1][1], Ka=ends[0][2], Kb=ends[1][2]))
        if not fail:
            Xs = [o["X"] for o in out]; fail = any(Xs[i].overlaps(Xs[j]) for i in range(4) for j in range(i + 1, 4))
        if not fail:
            # Every pair must, on the whole step, keep either the sign of Re ROT (y_i - y_j) (projection order) or the sign of Im ROT (y_i - y_j).
            # These sign conditions are open linear inequalities on the configuration, so they cut out a convex set that contains the whole
            # motion on the step; the step's braid is therefore that of the straight-line motion between the end configurations, in which each
            # pair's projection difference is linear and crosses 0 at most once. Crossing times are computed from the end enclosures and must
            # be pairwise disjoint; at each crossing the sign of Im decides the half-twist.
            pa_ = [ROTb * o["Ka"] for o in out]; pb_ = [ROTb * o["Kb"] for o in out]; Xr = [ROTb * o["X"] for o in out]; evs = []
            for i, j in itertools.combinations(range(4), 2):
                d = Xr[i] - Xr[j]
                if not d.real.contains(0): continue
                if d.imag.contains(0): fail = True; break
                ra, rb = (pa_[i] - pa_[j]).real, (pb_[i] - pb_[j]).real
                if sgn(ra) == 0 or sgn(rb) == 0: fail = True; break
                if sgn(ra) != sgn(rb): evs.append((ra / (ra - rb), i, j))
            if not fail and any(e1[0].overlaps(e2[0]) for e1, e2 in itertools.combinations(evs, 2)): fail = True
            if not fail and cur is None:
                if any(sgn((pa_[i] - pa_[j]).real) == 0 for i, j in itertools.combinations(range(4), 2)): raise RuntimeError("start order undecided")
                cur = sorted(range(4), key=lambda k: float(pa_[k].real.mid()))
        if fail:
            h /= 2
            if h < 1e-13: raise RuntimeError(f"branch-point tracking failed at tau = {tau}")
            continue
        for lam, i, j in sorted(evs, key=lambda e: float(e[0].mid())):
            pi_, qi_ = cur.index(i), cur.index(j)
            if abs(pi_ - qi_) != 1: raise RuntimeError("non-adjacent crossing")
            k = min(pi_, qi_); p_, q_ = cur[k], cur[k + 1]                     # p_ is left before the swap and moves right
            above = ((Xr[p_] - Xr[q_]).imag > 0)
            Sg = GEN[k + 1] if above else inv2(GEN[k + 1]); M = Sg @ M; nev += 1; cur[k], cur[k + 1] = cur[k + 1], cur[k]
        if sorted(range(4), key=lambda k: float(pb_[k].real.mid())) != cur: raise RuntimeError("order bookkeeping mismatch")
        Bprev = [o["Bb"] for o in out]; ys = [newton_root(af, o["y"], sf_(tb_)) for o in out]; tau = tb_; h = min(h * 1.5, hmax); nst += 1
    return M, nst, nev
def loop_fns(s0c, scc, r):
    d = scc - s0c; u = d / abs(d); pa_ = scc - r * u
    s0b, pab, scb, ub = acb(s0c.real, s0c.imag), acb(pa_.real, pa_.imag), acb(scc.real, scc.imag), acb(u.real, u.imag)
    def f(t):
        if t <= 0.4: return s0c + (pa_ - s0c) * (t / 0.4)
        if t <= 0.6: return scc - r * u * np.exp(1j * (t - 0.4) / 0.2 * 2 * np.pi)
        return pa_ + (s0c - pa_) * ((t - 0.6) / 0.4)
    def fb(T):
        lo, hi = float(T.mid() - T.rad()), float(T.mid() + T.rad())
        if hi <= 0.4 + 1e-12: return s0b + (pab - s0b) * (acb(T) / acb(arb(4) / 10))
        if lo >= 0.4 - 1e-12 and hi <= 0.6 + 1e-12: return scb - acb(arb(r)) * ub * (acb(0, 1) * (acb(T) - acb(arb(4) / 10)) / acb(arb(2) / 10) * 2 * acb(arb.pi())).exp()
        if lo >= 0.6 - 1e-12: return pab + (s0b - pab) * ((acb(T) - acb(arb(6) / 10)) / acb(arb(4) / 10))
        raise ValueError("step straddles a path corner")
    def dfb(T):
        lo, hi = float(T.mid() - T.rad()), float(T.mid() + T.rad())
        if hi <= 0.4 + 1e-12: return (pab - s0b) / acb(arb(4) / 10)
        if lo >= 0.4 - 1e-12 and hi <= 0.6 + 1e-12:
            return -acb(arb(r)) * ub * acb(0, 1) * acb(10 * arb.pi()) * (acb(0, 1) * (acb(T) - acb(arb(4) / 10)) / acb(arb(2) / 10) * 2 * acb(arb.pi())).exp()
        if lo >= 0.6 - 1e-12: return (s0b - pab) / acb(arb(4) / 10)
        raise ValueError("step straddles a path corner")
    def df(t):
        if t < 0.4: return (pa_ - s0c) / 0.4
        if t < 0.6: return -r * u * 1j * 2 * np.pi / 0.2 * np.exp(1j * (t - 0.4) / 0.2 * 2 * np.pi)
        return (s0c - pa_) / 0.4
    return f, fb, df, dfb
def prim(N):
    for c in range(2):
        v = N[:, c]
        if np.any(v != 0): g = math.gcd(abs(int(v[0])), abs(int(v[1]))); return [int(v[0] // g), int(v[1] // g)]
    return None
def Tmat(v): return np.array([[1 - v[0] * v[1], v[0] * v[0]], [-v[1] * v[1], 1 + v[0] * v[1]]])   # T_v(x) = x + det(v, x) v
def S9_part(name, i):
    """One loop (i = 0..5) or the big circle (i = 'big') at one end; saved to ckpt/S9_parts so a restart resumes mid-stage."""
    os.makedirs(os.path.join(CK, "S9_parts"), exist_ok=True); p = os.path.join(CK, "S9_parts", f"{name}_{i}.json")
    if os.path.exists(p): return json.load(open(p))
    s3 = load("S3_points"); s5 = load("S5_braid"); R = float(Fr(s5["R"])); s0c = R * np.exp(0.3j)
    ab, af = coef_at(TA if name == "TA" else TB); sv = [des(z) for z in s3[name]["s"]]; svc = [cfl(z) for z in sv]; t0 = time.time()
    if i == "big":
        s0b = acb(s0c.real, s0c.imag)
        M, nst, nev = braid_cert(ab, af, lambda t: s0c * np.exp(2j * np.pi * t), lambda T: s0b * (acb(0, 2) * acb(arb.pi()) * acb(T)).exp(),
                                 lambda t: 2j * np.pi * s0c * np.exp(2j * np.pi * t), lambda T: s0b * acb(0, 2) * acb(arb.pi()) * (acb(0, 2) * acb(arb.pi()) * acb(T)).exp())
        r_ = dict(M=M.tolist(), steps=nst, events=nev)
    else:
        r = 0.25 * min(abs(svc[i] - svc[j]) for j in range(6) if j != i)
        encl = bool((sv[i] - acb(svc[i].real, svc[i].imag)).abs_upper() < r and all((sv[j] - acb(svc[i].real, svc[i].imag)).abs_lower() > r for j in range(6) if j != i))
        f, fb, df, dfb = loop_fns(s0c, svc[i], r)
        M, nst, nev = braid_cert(ab, af, f, fb, df, dfb, corners=(0.4, 0.6))
        v = prim(M - np.eye(2, dtype=int)); kap = 1 if v is not None and np.array_equal(M, Tmat(v)) else (-1 if v is not None and np.array_equal(M, inv2(Tmat(v))) else 0)
        r_ = dict(M=M.tolist(), v=v, kappa=kap, steps=nst, events=nev, encloses_only_this_value=encl)
    r_["seconds"] = round(time.time() - t0); json.dump(r_, open(p, "w")); say(f"  [S9 part {name} {i}] {r_}"); return r_
S9_JOBS = [(n, i) for n in ("TA", "TB") for i in list(range(6)) + ["big"]]
@stage("S9_classes")
def S9():
    s5 = load("S5_braid"); res = {}
    for name, order in (("TA", s5["order_start"]), ("TB", s5["order_end"])):
        loops = {i: S9_part(name, i) for i in range(6)}; big = S9_part(name, "big"); Mbig = np.array(big["M"])
        P = np.eye(2, dtype=int)
        for i in order: P = np.array(loops[i]["M"]) @ P
        g6 = math.gcd(*[abs(int(x)) for x in (Mbig - np.eye(2, dtype=int)).flatten()])
        res[name] = dict(order=order, loops=loops, Mbig=Mbig.tolist(), product_equals_big=bool(np.array_equal(P, Mbig)), trace_big=int(np.trace(Mbig)), gcd_big=g6)
    return res

# ================================================================ S8b: which segment branches meet at the fold ==============================
@stage("S8b_foldlabels")
def S8b():
    s4 = json.load(open(os.path.join(CK, "S4_track.json"))); s7 = json.load(open(os.path.join(CK, "S7_fold.json"))); s8 = json.load(open(os.path.join(CK, "S8_loop.json")))
    ts_ = complex(*s7["t_star"]); rho = s8["rho"]; tr = ts_.real + rho
    st = next(x for x in s4["steps"] if x["ta"] <= tr <= x["tb"])
    Xst = [[box(complex(*m[0]), r), box(complex(*m[1]), r)] for m, r in zip(st["wm"], st["r"])]
    ws = [newton(np.array([complex(*m[0]), complex(*m[1])]), tr) for m in st["wm"]]
    B0 = [cert_point(w, acb(tr), tr, rads=(1e-12, 1e-10, 1e-8)) for w in ws]
    inside = all(b is not None and contains_box(Xst[i], b[1]) for i, b in enumerate(B0))
    tsb = acb(ts_.real, ts_.imag)
    pf = lambda th: tr + 1j * th * ts_.imag; pb = lambda TH: acb(tr) + acb(0, 1) * acb(TH) * acb(ts_.imag); dpf = lambda th: 1j * ts_.imag
    steps = track_path(ws, [b[1] for b in B0], pf, pb, dpf)
    Bend = [o["Bb"] for o in steps[-1]["br"]]
    p8 = lambda th: ts_ + rho * np.exp(2j * np.pi * th); p8b = lambda TH: tsb + arb(rho) * (acb(0, 2) * acb(arb.pi()) * acb(TH)).exp()
    w8 = all_crit(p8(0.0)); S8B = [cert_point(w, p8b(arb(0)), p8(0.0), rads=(1e-12, 1e-10, 1e-8))[1] for w in w8]
    lab = [[j for j in range(6) if not distinct_mod(S8B[j], Bend[i])] for i in range(6)]          # segment label i -> S8 label
    pair = [i for i in range(6) if lab[i] and lab[i][0] in s8["moved"]]
    return dict(t_real=tr, start_inside_segment_boxes=bool(inside), segment_to_loop=lab, fold_pair_segment_labels=pair, steps=len(steps))

# ================================================================ S10: the closing identity ================================================
def Tpow(v, e): return Tmat(v) if e == 1 else inv2(Tmat(v))
def gram(vs): return [[int(a[0] * b[1] - a[1] * b[0]) for b in vs] for a in vs]
def equal_up_to_signs(G, H):
    n = len(G)
    for signs in itertools.product((1, -1), repeat=n - 1):
        d = (1,) + signs
        if all(G[i][j] == d[i] * d[j] * H[i][j] for i in range(n) for j in range(n)): return True, d
    return False, None
@stage("S10_closing")
def S10():
    s9 = json.load(open(os.path.join(CK, "S9_classes.json"))); s5 = json.load(open(os.path.join(CK, "S5_braid.json")))
    vA = {int(k): np.array(v["v"]) for k, v in s9["TA"]["loops"].items()}; vB = {int(k): np.array(v["v"]) for k, v in s9["TB"]["loops"].items()}
    oA, oB = s5["order_start"], s5["order_end"]
    GA_, GB_ = gram([vA[i] for i in oA]), gram([vB[i] for i in oB]); out = dict(G_start=GA_, G_end=GB_, results={})
    for c in (1, -1):
        cl = {k: v.copy() for k, v in vA.items()}
        for e in s5["events"]:
            cl[e["far"]] = Tpow(cl[e["near"]], c * e["eps"]) @ cl[e["far"]]
        Gp = gram([cl[i] for i in oB]); ok, d = equal_up_to_signs(GB_, Gp)
        out["results"][str(c)] = dict(G_pred=Gp, equal_up_to_signs=ok, signs=d)
    out["unchanged_without_moves"] = equal_up_to_signs(GB_, gram([vA[i] for i in oB]))[0]
    return out

# ================================================================ S11: transfer to the leading form (continuation in L) =====================
def dwdL(w, t, L): E = Ef(w, t, L); return -np.linalg.solve(Jf(w, t, L), Vf.T @ (E * (-x0f(t))))
@stage("S11_transfer")
def S11():
    s3 = json.load(open(os.path.join(CK, "S3_points.json"))); s2 = json.load(open(os.path.join(CK, "S2_leading.json")))
    ws = [np.array([complex(*w[0]), complex(*w[1])]) for w in s3["T0"]["w"]]; Bprev = [[des(k) for k in K] for K in s3["T0"]["K"]]
    t0b = arb(0); Lc = 20.0; h = 0.05; Lend = 160.0; recs = {}; marks = [40.0, 80.0, 160.0]
    while Lc < Lend:
        Lb_ = min(Lc + h, Lend, *[m for m in marks if m > Lc]); Lm = (Lc + Lb_) / 2; LL = tball(Lc, Lb_); out = []; fail = False
        for i, w in enumerate(ws):
            wm = newton(w + dwdL(w, 0.0, Lc) * (Lm - Lc), 0.0, Lm)
            if wm is None: fail = True; break
            sp = np.abs(dwdL(wm, 0.0, Lm)).max() * (Lb_ - Lc) / 2; sc = 1 + np.abs(wm).max()
            c = cert_point(wm, t0b, 0.0, L=LL, Lmid=Lm, rads=tuple(r * sc for r in (1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2) if r * sc > 1.5 * sp)[:4])
            if c is None or not contains_box(c[1], Bprev[i]): fail = True; break
            we = newton(wm, 0.0, Lb_); ce = None if we is None else cert_point(we, t0b, 0.0, L=arb(Lb_), Lmid=Lb_, rads=(1e-12 * sc, 1e-10 * sc, 1e-8 * sc))
            if ce is None or not contains_box(c[1], ce[1]): fail = True; break
            out.append((we, ce))
        if fail:
            h /= 2
            if h < 1e-9: raise RuntimeError(f"L continuation failed at L = {Lc}")
            continue
        ws = [o[0] for o in out]; Bprev = [o[1][1] for o in out]; Lc = Lb_; h = min(h * 1.5, 2.0)
        if any(abs(Lc - m) < 1e-12 for m in marks):
            vals = [s_enc(o[1][2], t0b, arb(Lc)) * (arb(Lc) / 10).exp() for o in out]; recs[str(int(Lc))] = [ser(v) for v in vals]
    wall = [des(v["ball"]) for v in s2["wall_values"]]
    final = [des(v) for v in recs["160"]]
    match = {}
    for i, v in enumerate(final):
        d = [float((v - wv).abs_upper()) for wv in wall]; j = int(np.argmin(d)); match[i] = dict(nearest_wall_value=j, dist_upper=d[j], second=sorted(d)[1])
    return dict(scaled_values=recs, match_at_160=match, wall_values=[ser(w) for w in wall])

# ================================================================ S12: comparison with the independent re-derivation (CE12) ================
def iv_to_acb(d): 
    re = [arb(x) for x in d["re"]]; im = [arb(x) for x in d["im"]]
    return acb(re[0].union(re[1]), im[0].union(im[1]))
@stage("S12_independent")
def S12():
    p = os.path.join(D45, "independent", "result.json")
    if not os.path.exists(p): return dict(available=False)
    ind = json.load(open(p)); s3 = load("S3_points"); s7 = load("S7_fold"); out = dict(available=True)
    for mine, theirs in (("TA", "t=-1/10"), ("TB", "t=1/10")):
        A = [des(z) for z in s3[mine]["s"]]; B = [iv_to_acb(q["critical_value"]) for q in ind["taskA"][theirs]["points"]]
        match = [[j for j in range(len(B)) if A[i].overlaps(B[j])] for i in range(len(A))]
        out[mine] = dict(their_count=ind["taskA"][theirs]["count"], their_all_certified=ind["taskA"][theirs]["all_certified"], matching=match,
                         one_to_one=all(len(m) == 1 for m in match) and len({m[0] for m in match if m}) == len(A))
    tb_ = ind["taskB"]; ts_mine = des(s7["K"][2]); sv_mine = des(s7["s_star"])
    out["fold"] = dict(their_certified=tb_["certified_regular_zero"], their_rank_one=tb_["rank_exactly_1"], their_cubic_nonzero=tb_["cubic_nonzero"],
                       t_star_overlap=bool(ts_mine.overlaps(iv_to_acb(tb_["t_star"]))), value_overlap=bool(sv_mine.overlaps(iv_to_acb(tb_["critical_value"]))))
    return out

# ================================================================ evaluation of CE1-CE11 (from the checkpoints) ============================
def load(n): return json.load(open(os.path.join(CK, n + ".json")))
def evaluate():
    r = []
    s1 = load("S1_wall"); r.append(check("CE1 IDENTITY exact wall: the only tie on [-1/10, 1/10] is the first-level row0/col0 tie at t = 0",
        len(s1["ties"]) == 1 and s1["ties"][0]["t"] == "0" and s1["ties"][0]["step"] == 1 and sorted(s1["ties"][0]["tied"]) == ["c0", "r0"], f"ties {s1['ties']}"))
    s2 = load("S2_leading"); want = sorted([(0.3305, 1.0), (3.7996, 0.0), (2.9597, 0.7724), (2.9597, -0.7724)])
    got = sorted((v["modulus"], v["phase_over_pi"]) for v in s2["wall_values"])
    r.append(check("CE2 IDENTITY exact leading form: wall polynomial, the four wall values, exact zero and collision parameters",
        s2["reduction_ok"] and s2["wall_equal_from_both_sides"] and got == want and s2["zero_row"] and s2["zero_col"],
        f"{s2['wall_poly']} = 0; values {got}; zero at b in {s2['zero_row']} (row), a in {s2['zero_col']} (column); collisions {s2['disc_row']} = 0, {s2['disc_col']} = 0"))
    s3 = load("S3_points"); r.append(check("CE3 IDENTITY six certified, disjoint critical points with disjoint values at t = -1/10, 0, 1/10",
        all(s3[k]["n_found"] == 6 and s3[k]["certified"] and s3[k]["disjoint"] and s3[k]["values_disjoint"] for k in ("TA", "T0", "TB")),
        "; ".join(f"{k}: {s3[k]['n_found']} found" for k in ("TA", "T0", "TB"))))
    s4 = load("S4_track"); r.append(check("CE4 IDENTITY certified continuation over [-1/10, 1/10], boxes pairwise disjoint on every step",
        s4["n_steps"] > 0 and sorted(s4["perm_to_TB"]) == list(range(6)), f"{s4['n_steps']} steps; branch -> point at t = 1/10: {s4['perm_to_TB']}"))
    s5 = load("S5_braid"); r.append(check("CE5 IDENTITY complete braid word as seen from s0, every swap certified simple and adjacent",
        s5["cone"] and s5["order_start_certified"] and s5["adjacent"] and s5["order_after_word"] == s5["order_end"],
        f"R = {s5['R']}; start order {s5['order_start']}; word " + ", ".join(f"t in [{e['ta']:.5f}, {e['tb']:.5f}]: near {e['near']}, far {e['far']}, eps {e['eps']:+d}, positions {e['pos']},{e['pos'] + 1}" for e in s5["events"])))
    s6 = load("S6_real"); nz = sum(len(b.get("zeros", [])) for b in s6["branches"])
    nonreal_ok = True
    for b in s6["branches"]:
        if b["real"]: continue
        w = s3["TA"]["w"][b["label"]]; rr = s3["TA"]["rad"][b["label"]]; X = [box(complex(*w[0]), rr), box(complex(*w[1]), rr)]
        nonreal_ok &= distinct_mod(X, [x.conjugate() for x in X])
    r.append(check("CE6 HYPOTHESIS exactly one real critical value changes sign along the segment", nz == 1 and nonreal_ok and all(b.get("lifted_in_complex_box", True) for b in s6["branches"]),
        "; ".join(f"branch {b['label']}: " + ("non-real" if not b["real"] else f"signs {b['signs']}, zeros {[(round(z['ta'], 5), round(z['tb'], 5), z['sign_before'], z['sign_after']) for z in b['zeros']]}") for b in s6["branches"])))
    s7 = load("S7_fold"); r.append(check("CE7 IDENTITY certified fold: regular zero of (grad W, det Hess), Hessian rank 1, cubic term nonzero (A2)",
        s7["certified"] and s7["rank_one"] and s7["cubic_nonzero"], f"t* = {s7.get('t_star')}; s* = {s7.get('s_star')}; candidates found {len(s7['found'])}"))
    s8 = load("S8_loop"); s8b = load("S8b_foldlabels")
    tot = des(s8["total_turn_over_pi"]) if s8.get("total_turn_over_pi") else None
    r.append(check("CE8 HYPOTHESIS loop around t* exchanges exactly the colliding pair; their value difference turns by +-3 pi",
        s8["ok"] and len(s8["moved"]) == 2 and s8.get("winding_ok") and tot is not None and (tot.contains(3) or tot.contains(-3)) and tot.rad() < 0.5,
        f"rho = {s8['rho']:.5f}; permutation {s8['perm']}; turn/pi {s8.get('total_turn_over_pi')}; the pair is segment branches {s8b['fold_pair_segment_labels']}"))
    s9 = load("S9_classes"); ok9 = True; info = []
    kap = {lp["kappa"] for k in ("TA", "TB") for lp in s9[k]["loops"].values()}
    for k in ("TA", "TB"):
        e = s9[k]; ok9 &= e["product_equals_big"] and e["trace_big"] == 2 and e["gcd_big"] == 6 and all(lp["encloses_only_this_value"] for lp in e["loops"].values())
        info.append(f"{k}: classes (angular order) {[e['loops'][str(i)]['v'] for i in e['order']]}, M_big {e['Mbig']}")
    r.append(check("CE9 IDENTITY Picard-Lefschetz at both ends: transvections with one kappa, ordered product = monodromy at infinity (trace 2, gcd 6)",
        ok9 and len(kap) == 1 and 0 not in kap, f"kappa {sorted(kap)}; " + "; ".join(info)))
    s10 = load("S10_closing"); cs = [c for c, v in s10["results"].items() if v["equal_up_to_signs"]]
    r.append(check("CE10 IDENTITY closing identity: Gram matrix at t = 1/10 = Hurwitz image of t = -1/10 under the braid word, one global sign c",
        len(cs) >= 1, f"c that works: {cs}; G_start {s10['G_start']}; G_end {s10['G_end']}; G_end equals G_start without the moves: {s10['unchanged_without_moves']}"))
    s11 = load("S11_transfer")
    r.append(check("CE11 RECORD transfer: scaled critical values at t = 0, L = 160, against the wall-form values", True,
        "; ".join(f"branch {i}: wall value {m['nearest_wall_value']} within {m['dist_upper']:.1e} (next {m['second']:.2f})" for i, m in s11["match_at_160"].items())))
    s12 = load("S12_independent")
    r.append(check("CE12 RECORD independent re-derivation (separate agent, mpmath.iv, own code): CE3 at t = +-1/10 and CE7", True,
        "not available" if not s12["available"] else f"t = -1/10: {s12['TA']['their_count']} certified, values one-to-one with ours {s12['TA']['one_to_one']}; "
        f"t = 1/10: {s12['TB']['their_count']} certified, one-to-one {s12['TB']['one_to_one']}; fold: {s12['fold']}"))
    say(f"{sum(r)}/{len(r)} passed")
STAGES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S8b", "S9", "S10", "S11", "S12"]
if __name__ == "__main__" and "--s9worker" in sys.argv:                     # parallel helper: python3 check_example45.py --s9worker k n
    k, n = int(sys.argv[-2]), int(sys.argv[-1])
    for job in S9_JOBS[k::n]: S9_part(*job)
elif __name__ == "__main__" and not LIB:
    say(f"==== run {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for nm in STAGES: globals()[nm]()
    evaluate()
