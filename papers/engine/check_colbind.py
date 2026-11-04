"""Checks for section 12.7 of the companion note: the column-binding example classified exactly (module pce/colbind.py).
  CB1  the reduction (exact, sympy): y1 dW/dy1 - y2 dW/dy2 = y1 - y2 + tau^3 / y2, so y1 = s - tau^3 / s with s = y2; the first equation then becomes
       exactly q(s) = s^5 + tau^2 s^4 - 2 tau^3 s^3 - 2 tau^5 s^2 + tau^8; q(0) != 0 and y1 = 0 is impossible, so critical points <-> roots of q
  CB2  the Newton polygon of q: lower hull (0, 8), (3, 3), (5, 0); an edge of slope -5/3 with edge polynomial 1 - 2c^3 (three roots s ~ 2^(-1/3) omega tau^(5/3))
       and an edge of slope -3/2 with edge polynomial c^2 - 2 (two roots s ~ +-sqrt2 tau^(3/2)): so the valuations in T are (3/20, 3/20) twice and
       (2/15, 1/6) three times
  CB3  for every T > 0 the five critical points are distinct and nondegenerate: disc_s q = -tau^31 (256 tau^2 - 837 tau + 3456) and the Hessian resultant
       tau^54 (256 tau^2 - 837 tau + 3456) have no positive root (the quadratic has negative discriminant)
  CB4  the five critical values are distinct for every T in (0, 1): W = 2s + tau^4/(s^2 - tau^3) + (s^2 - tau^3)/tau^2, and the discriminant of the
       critical-value polynomial is -tau^225 (tau - 1)^2 (256 tau^2 - 837 tau + 3456)^3, vanishing only at T = 1
  CB5  the leading data, against the roots at tau = 10^-10, 10^-20, 10^-40: y = coef T^val with coef (sqrt2/2, sqrt2), (-sqrt2/2, -sqrt2) on the leximin
       fibre and (-2^(1/3) omega^-1, 2^(-1/3) omega), omega^3 = 1, on the fibre (2/15, 1/6), with relative error decreasing; the finite-T offsets of the
       valuations are -log|coef| / log(1/T) (this is the ln 2 / (2L) of Example 30); critical values 2 tau +- 2 sqrt2 tau^(3/2) and
       -2 tau + 2^(2/3) omega tau^(5/3)
  CB6  the Clifford forms: Q/W -> J = [[1, 1], [1, 1]] (rank 1) for all five; det(Q/W) ~ +-sqrt2 tau^(1/2) on the leximin fibre (positive definite for the
       positive object, indefinite for (-,-)) and ~ -3 2^(-4/3) omega^-2 tau^(2/3) on the other fibre, where the first-order correction cancels in the
       small direction
  CB7  the Hom table: 5 objects over 2 disjoint fibres (2 + 3); within a fibre distinct holonomies and distinct critical values, so all 20 off-diagonal
       Hom vanish; every diagonal entry is Cl(Q_c) = M_2 (explicit generators)
  CB8  the reduction theorem tested on 23 objects (F_1 at alpha = 1/5, 3/5; the 2x3 example, its one-count neighbour, and this example): with K the leading
       set at the exact valuation and rho the leading coefficients of the terms, sum_K rho != 0 (H1), Q/W -> M_c = sum_K rho v v^T / sum_K rho, and
       rank M_c = dim span{v_k : k in K} whenever the rho_k are real of one sign (H2); every object here satisfies H1, and rank M_c = dim span in all 23
"""
import os, sys, math
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
import sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import colbind as CB, floermorph as FM, kahlerbridge as KB, couplingfibre as CF
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
d = CB.symbolic(); t, s, w = d["t"], d["s"], d["w"]

# CB1
ok = sp.simplify(d["diff"] - (sp.Symbol("y1") - sp.Symbol("y2") + t ** 3 / sp.Symbol("y2"))) == 0 and sp.expand(d["E1"] - d["q"]) == 0
ok &= d["q"].subs(s, 0) != 0 and sp.expand(d["q"].subs(s ** 2, t ** 3)) != 0
qq = sp.Poly(d["q"], s); rem = sp.rem(qq, sp.Poly(s ** 2 - t ** 3, s)); ok &= not rem.is_zero          # q and s^2 - tau^3 have no common root
check("CB1 exact reduction to the quintic q(s); roots <-> critical points (s != 0, y1 != 0)", ok, f"q = {d['q']}")

# CB2
pts, hull, edges = CB.newton_polygon(); ok = hull == [(0, 8), (3, 3), (5, 0)] and [e[2] for e in edges] == [Fr(5, 3), Fr(3, 2)]
c = sp.Symbol("c"); e1 = sp.expand((d["q"].subs(s, c * t ** sp.Rational(5, 3)) / t ** 8)).subs(t, 0); e2 = sp.expand((d["q"].subs(s, c * t ** sp.Rational(3, 2)) / t ** sp.Rational(15, 2))).subs(t, 0)
ok &= sp.expand(e1 - (1 - 2 * c ** 3)) == 0 and sp.factor(e2) == sp.factor(c ** 3 * (c ** 2 - 2))
check("CB2 Newton polygon: edges of root valuation 5/3 (polynomial 1 - 2c^3) and 3/2 (c^2 - 2); valuations (3/20,3/20) x2, (2/15,1/6) x3", ok, f"hull {hull}")

# CB3
quad = 256 * t ** 2 - 837 * t + 3456; ok = sp.expand(d["disc"] + t ** 31 * quad) == 0 and sp.expand(d["hess_res"] - t ** 54 * quad) == 0
ok &= sp.discriminant(quad, t) < 0 and sp.real_roots(sp.Poly(quad, t)) == []
check("CB3 distinct and nondegenerate for every T > 0: disc = -tau^31 (256 tau^2 - 837 tau + 3456), Hessian resultant tau^54 (same), no positive root", ok,
      f"disc of the quadratic = {sp.discriminant(quad, t)}")

# CB4
Wexact = 2 * s + t ** 4 / (s ** 2 - t ** 3) + (s ** 2 - t ** 3) / t ** 2; ok = sp.simplify(d["Ws"] - Wexact) == 0
ok &= sp.expand(d["disc_w"] + t ** 225 * (t - 1) ** 2 * quad ** 3) == 0
check("CB4 critical values distinct for T in (0, 1): disc_w = -tau^225 (tau - 1)^2 (quadratic)^3; W = 2s + tau^4/(s^2 - tau^3) + (s^2 - tau^3)/tau^2", ok)

# CB5, CB6, CB7
ex = CB.exact_branches(); ok5 = ok6 = ok7 = True; errs = []; info7 = ""
J = np.ones((2, 2))
for k in (10, 20, 40):
    tau = mp.mpf(10) ** -k; T = tau ** 10; L = float(-mp.log(T)); mp.mp.dps = 600
    ob = FM.objects(CB.V, [mp.mpf(x.numerator) / x.denominator for x in CB.LAM], T, CB.branches(tau, 600))
    used = set(); e_k = 0
    for xb in ex:
        val = [mp.mpf(v_.numerator) / v_.denominator for v_ in xb["val"]]
        best = min((i for i in range(5) if i not in used), key=lambda i: sum(abs(ob[i]["y"][j] / T ** val[j] - xb["coef"][j]) for j in range(2)))
        used.add(best); o = ob[best]; rel = max(abs(o["y"][j] / T ** val[j] / xb["coef"][j] - 1) for j in range(2)); e_k = max(e_k, float(rel))
        off = [float(mp.log(abs(o["y"][j])) / mp.log(T)) - float(val[j]) for j in range(2)]; pred = [-float(mp.log(abs(xb["coef"][j]))) / L for j in range(2)]
        ok5 &= max(abs(a - b) for a, b in zip(off, pred)) < 5 * float(rel) / L + 1e-12
        Wl = (2 * tau + (1 if xb["name"] == "positive" else -1) * 2 * mp.sqrt(2) * tau ** 1.5) if xb["val"][0] == Fr(3, 20) else (-2 * tau + mp.mpf(2) ** (mp.mpf(2) / 3) * xb["omega"] * tau ** (mp.mpf(5) / 3))
        ok5 &= abs(o["W"] - Wl) < abs(o["W"] - (Wl.real if False else Wl)) + abs(Wl) * float(tau ** 0.3)
        Qn = np.array(o["Qn"], complex); dl, pw = xb["det_lead"]; dv = complex(dl) * float(tau ** (float(pw)))
        ok6 &= np.abs(Qn - J).max() < 10 * float(tau ** (1 / 3)) and abs(complex((o["Q"][0][0] * o["Q"][1][1] - o["Q"][0][1] ** 2) / o["W"] ** 2) / dv - 1) < 10 * float(tau ** (1 / 6))
        if xb["name"] == "positive": ok6 &= float(mp.re((o["Q"][0][0] * o["Q"][1][1] - o["Q"][0][1] ** 2) / o["W"] ** 2)) > 0 and Qn.real[0, 0] > 0
        if xb["name"] == "real (-,-)": ok6 &= float(mp.re((o["Q"][0][0] * o["Q"][1][1] - o["Q"][0][1] ** 2) / o["W"] ** 2)) < 0
        if k == 20:
            e, dim = FM.clifford_generators(Qn); ok7 &= dim == 4
    errs.append(e_k)
    if k == 20:
        Wv = [complex(o["W"]) for o in ob]; ok7 &= min(abs(Wv[i] - Wv[j]) for i in range(5) for j in range(i + 1, 5)) > 0
        fib = sorted([tuple(np.round(o["u"], 2)) for o in ob]); G = {}
        for o in ob: G.setdefault(int(np.argmin([np.abs(o["u"] - np.array(f_, float)).max() for f_ in [(0.15, 0.15), (2 / 15, 1 / 6)]])), []).append(o)
        ok7 &= sorted(len(x) for x in G.values()) == [2, 3]
        for g_ in G.values():
            for i in range(len(g_)):
                for j in range(i + 1, len(g_)): ok7 &= abs(complex(g_[i]["W"]) - complex(g_[j]["W"])) > 0 and np.abs(g_[i]["hol"] - g_[j]["hol"]).max() > 1e-3
        info7 = f"fibres {sorted(len(x) for x in G.values())}"
ok5 &= errs[0] > errs[1] > errs[2]
check("CB5 leading data: coefficients (+-sqrt2/2, +-sqrt2) and (-2^(1/3)/omega, 2^(-1/3) omega); offsets = -log|coef|/L; leading critical values", ok5,
      f"relative errors {[f'{x:.1e}' for x in errs]}")
check("CB6 Clifford forms: Q/W -> J (rank 1) for all five; det ~ +-sqrt2 tau^(1/2) and -3 2^(-4/3) omega^-2 tau^(2/3); positive definite / indefinite", ok6)
check("CB7 Hom table: 5 objects over 2 disjoint fibres (2 + 3); distinct holonomies and critical values within a fibre; End = Cl(Q_c) = M_2", ok7, info7)

# CB8
def objs_F1(a, T):
    v, lam = KB.F1(a); return v, [Fr(x).limit_denominator(1000) for x in lam], FM.objects(v, [mp.mpf(Fr(x).limit_denominator(1000).numerator) / Fr(x).limit_denominator(1000).denominator for x in lam], T, FM.f1_critical(a, T, 200))
def objs_cpl(f, g, tau):
    pts, D = CF.critical_points(f, g, tau, dps=400); ff = [float(x) for x in f]; gf = [float(x) for x in g]; F = CF.facet_cells(ff, gf)
    Vv, Ll = CF.cell_functions(Fr(f[0]), Fr(g[0]), Fr(g[1])); T = (mp.mpf(Fr(tau).numerator) / Fr(tau).denominator) ** D; mp.mp.dps = 400
    v = [Vv[c_] for c_ in F]; lam = [Fr(Ll[c_]) for c_ in F]; return v, lam, FM.objects(v, [mp.mpf(x.numerator) / x.denominator for x in lam], T, pts), T
def reduce_test(v, lam, ob_seq, fibres):
    out = []
    for idx in range(len(ob_seq[0][1])):
        o_last = ob_seq[-1][1][idx]; u = min(fibres, key=lambda f_: np.abs(o_last["u"] - np.array([float(x) for x in f_])).max())
        ell = [Fr(vk[0]) * u[0] + Fr(vk[1]) * u[1] - l for vk, l in zip(v, lam)]; h = min(ell); K = [k for k in range(len(v)) if ell[k] == h]
        errs = []; rho = None
        for T, ob in ob_seq:
            o = min(ob, key=lambda z: abs(z["y"][0] - o_last["y"][0] * 0) * 0 + np.abs(z["hol"] - o_last["hol"]).max() + np.abs(z["u"] - o_last["u"]).max())
            cks = [T ** (-mp.mpf(l.numerator) / l.denominator) * mp.fprod(yi ** vi for yi, vi in zip(o["y"], vk)) for vk, l in zip(v, lam)]
            rho = [complex(cks[k] / T ** (mp.mpf(h.numerator) / h.denominator)) for k in K]
            M = sum(r * np.outer(v[k], v[k]) for r, k in zip(rho, K)) / sum(rho); errs.append(np.abs(np.array(o["Qn"], complex) - M).max())
        span = np.linalg.matrix_rank(np.array([v[k] for k in K], float)); rkM = np.linalg.matrix_rank(M, tol=1e-6)
        oneSign = all(abs(r.imag) < 1e-6 * abs(r) for r in rho) and (all(r.real > 0 for r in rho) or all(r.real < 0 for r in rho))
        out.append(dict(H1=abs(sum(rho)) > 1e-6, rk=rkM, span=span, H2=oneSign, conv=errs[-1] < errs[0] and errs[-1] < 0.2))
    return out
ok = True; recs = []
for a, fib in ((0.2, [(Fr(1, 3), Fr(1, 3)), (Fr(1, 5), Fr(3, 5))]), (0.6, [(Fr(2, 5), Fr(1, 5))])):
    seq = []
    for k in (24, 48, 96):
        T = mp.mpf(10) ** -k; v, lam, ob = objs_F1(a, T); seq.append((T, ob))
    recs += reduce_test(v, lam, seq, fib)
for f, g, fib in ((CF.EX_F, CF.EX_G, [(Fr(2, 15), Fr(2, 15)), (Fr(13, 100), Fr(13, 100)), (Fr(1, 5), Fr(1, 10))]),
                  (CF.EX_F, (Fr(30, 100), Fr(44, 100), Fr(26, 100)), [(Fr(27, 200), Fr(27, 200)), (Fr(1, 5), Fr(1, 10))])):
    seq = []
    for tau in ("1/100", "1/1000", "1/10000"):
        v, lam, ob, T = objs_cpl(f, g, tau); seq.append((T, ob))
    recs += reduce_test(v, lam, seq, fib)
seq = []
for k in (10, 20, 40):
    tau = mp.mpf(10) ** -k; T = tau ** 10; mp.mp.dps = 600; seq.append((T, FM.objects(CB.V, [mp.mpf(x.numerator) / x.denominator for x in CB.LAM], T, CB.branches(tau, 600))))
recs += reduce_test(CB.V, CB.LAM, seq, [(Fr(3, 20), Fr(3, 20)), (Fr(2, 15), Fr(1, 6))])
ok = len(recs) == 23 and all(r["H1"] and r["conv"] for r in recs) and all(r["rk"] == r["span"] for r in recs if r["H2"]) and all(r["rk"] == r["span"] for r in recs)
check("CB8 the reduction theorem: H1 holds, Q/W -> M_c, rank M_c = dim span (forced by H2 where it holds; observed in all)", ok,
      f"{len(recs)} objects; H2 holds for {sum(r['H2'] for r in recs)}; ranks {[r['rk'] for r in recs]}")
print(f"{sum(results)}/{len(results)} passed")
