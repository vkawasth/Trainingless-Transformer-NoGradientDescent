"""Study 30: a structure-aware certified drift bound (sec:structbound). Pre-registered in study30_structure/PREREG_study30.md, frozen by
SHA-256 (study30_structure/FROZEN_PREREG_study30.txt) before any code for it ran. Responds to the TK6 failure of Study 29, which stays a
failure. New script; reuses the helpers of check_tolerance.py (its code before the families block) and pce/credalledger.py; modifies
nothing.
  SW1 IDENTITY   interaction coordinates (null space, product-contrast basis, exact reconstruction, programme in interaction coordinates)
  SW2 IDENTITY   U0, U1, Ud and their combination dominate the drift; lower bounds lie below it
  SW3 IDENTITY   screened top-20 equals exhaustive top-20
  SW4 IDENTITY   context order: two evidence weights in either order; C = 0 then B in {0,1} versus the joint event
  SW5 IDENTITY   support boundary handling
  SW6 HYPOTHESIS P332: the screen evaluates <= 50% of defined tables
  SW7 HYPOTHESIS P332: Spearman(min(U0, U1), D) >= 0.7
  SW8 HYPOTHESIS P332tol, P442: the screen evaluates no larger share than the band-only search
  SW9 RECORD     solves vs evaluations, U1 time, Spearman per bound, active-set stability, centre/width, interaction vs tolerance parts
First run (the only run): 8/9, about 45 minutes. SW7 FAILED: Spearman(min(U0, U1), D) = 0.53 in P332 (0.56 P332tol, 0.72 P442), below
0.7. SW6 and SW8 passed by wide margins (P332: 32 of 10000 tables evaluated, against 8142 for the band-only search). Threshold unchanged.
Post-hoc diagnosis (not a check; rerun of P332 only): with the 20th largest drift 0.1456, the band-only bound alone excludes 18.6% of
tables, the block relaxation U1 alone 70.9%, their a-priori combination with library vertices 86.4%, and the final bound after the dual
certificates of the 32 evaluated tables 99.8%. Ranking quality (SW7) and certification power (SW6) are different properties: the
certificate only has to be tight near the top. RuntimeWarnings from logit at band ends numerically outside [0, 1] are benign: those
entries are replaced by np.where or ignored by fmax/fmin, and SW2 confirms no bound became NaN.
Output: check_structbound_output.txt, study30_structure/structbound.json (aggregates only; synthetic data).
"""
import os, sys, json, time
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
exec(open(os.path.join(HERE, "check_tolerance.py")).read().split("# ---------------------------------------------------------------- families")[0])
results.clear()
E = Fr(1, 100)
def pair_rows(a, b, c):
    cells = [(i, j, k) for i in range(a) for j in range(b) for k in range(c)]; rows = []
    for i in range(a):
        for j in range(b): rows.append([1 if (cl[0], cl[1]) == (i, j) else 0 for cl in cells])
    for j in range(b):
        for k in range(c): rows.append([1 if (cl[1], cl[2]) == (j, k) else 0 for cl in cells])
    for i in range(a):
        for k in range(c): rows.append([1 if (cl[0], cl[2]) == (i, k) else 0 for cl in cells])
    return cells, np.array(rows, int)
def joints(ncell, den, seed, T, zero=None):
    rng = np.random.default_rng(seed); out = []
    for _ in range(T):
        w = rng.dirichlet([1] * ncell)
        if zero is not None: w[zero] = 0; w = w / w.sum()
        v = np.floor(w * den).astype(int); v[int(np.argmax(v))] += den - v.sum(); out.append([Fr(int(x), den) for x in v])
    return out
def family_pairs(a, b, c, den, seed, tol=None, T=10000, zero_block=False):
    cells, RA = pair_rows(a, b, c); blk = [k for k, cl in enumerate(cells) if cl[1] in (0, 1) and cl[2] == 0]
    J = joints(len(cells), den, seed, T, zero=blk if zero_block else None)
    vals = [[sum(Fr(int(r[k])) * j[k] for k in range(len(cells))) for r in RA] for j in J]
    if tol is None:
        A, _, keys = CL.pairwise_form(a, b, c); bF = [CL.b_pair(j, cells, keys) for j in J]; lo = [[float(v) for v in row] for row in vals]; hi = lo
    else:
        nx = len(cells); R = len(RA); M = nx + 2 * R; rows = []; s = nx
        for r in range(R):
            for sign in (+1, -1):
                row = list(RA[r]) + [0] * (M - nx); row[s] = sign; s += 1; rows.append(row)
        rows.append([1] * nx + [0] * (M - nx)); A = np.array(rows, int)
        bF = [[x for v in row for x in (v + tol, v - tol)] + [Fr(1)] for row in vals]
        lo = [[max(0.0, float(v - tol)) for v in row] for row in vals]; hi = [[float(v + tol) for v in row] for row in vals]
    Bm = np.array([[float(x) for x in bb] for bb in bF]).T
    num = [k for k, cl in enumerate(cells) if cl[0] == 0 and cl[1] in (0, 1) and cl[2] == 0]
    L = [Fr(1) + Fr((x + 2 * y + z) % 3, 2) for (x, y, z) in cells]
    cov = np.array([float(sum(j[k] for k in blk)) for j in J])
    return dict(A=A, M=A.shape[1], nx=len(cells), bF=bF, Bm=Bm, num=num, den=blk, L=L, cov=cov, pop=None, cells=cells, RA=RA,
                rlo=np.array(lo), rhi=np.array(hi), shape=(a, b, c), joints=J)
sig = lambda x: 1 / (1 + np.exp(-x))
def logit(p):
    with np.errstate(divide="ignore"): return np.log(p) - np.log1p(-p)
def dpdm(F):
    lL = np.log([float(x) for x in F["L"]]); rest = [k for k in F["den"] if k not in F["num"]]
    return max(lL[F["num"]].max() - lL[rest].min(), 0), max(lL[rest].max() - lL[F["num"]].min(), 0)
def relax_band(F, t, dmin, n1, d1):
    """U1: after-evidence band of the block relaxation of table t (Charnes-Cooper over the block cells)."""
    blk = F["den"]; nb = len(blk); RA = F["RA"][:, blk]; full = F["RA"].sum(1); inside = RA.sum(1) == full
    lo, hi = F["rlo"][t], F["rhi"][t]; Aub = []; bub = []; Aeq = []; beq = []
    for r in range(len(RA)):
        if RA[r].sum() == 0: continue
        Aub.append(list(RA[r]) + [-hi[r]]); bub.append(0.0)
        if inside[r]: Aub.append(list(-RA[r]) + [lo[r]]); bub.append(0.0)
    Aub.append([1.0] * nb + [-1.0]); bub.append(0.0); Aub.append([-1.0] * nb + [dmin]); bub.append(0.0)
    nn = n1[blk]; dd = d1[blk]; Aeq = [list(dd) + [0.0]]; beq = [1.0]; out = []
    for sense in (+1, -1):
        r = linprog(np.concatenate([-sense * nn, [0.0]]), A_ub=np.array(Aub), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq, bounds=[(0, None)] * (nb + 1), method="highs")
        out.append(float(nn @ r.x[:nb]) if r.status == 0 else (1.0 if sense > 0 else 0.0))
    return out[0] + 1e-9, out[1] - 1e-9
def dual_bound(L, k, Bd, n, d, sense, tol=1e-12):
    """weak-duality bound from basis k for every table: upper bound on the max ratio (sense +1) or lower bound on the min (sense -1)."""
    B = list(L.bases[k]); Nn = [j for j in range(L.M) if j not in set(B)]; inv = L.inv[k]; Mx = inv @ L.A[:, Nn]
    rn = n[Nn] - n[B] @ Mx; rd = d[Nn] - d[B] @ Mx; lo, hi = 0.0, 1.0
    s = 1 if sense > 0 else -1                                   # need s * (rn - lam rd) <= 0
    for a_, b_ in zip(s * rn, s * rd):
        if abs(b_) <= tol:
            if a_ > 1e-10: return None
        elif b_ > 0: lo = max(lo, a_ / b_)
        else: hi = min(hi, a_ / b_)
    if lo > hi + 1e-12: return None
    X = inv @ Bd; P = n[B] @ X; Q = d[B] @ X
    if sense > 0:
        lam = np.where(Q > tol, np.maximum(P / np.where(Q > tol, Q, 1), lo), lo); ok = np.where(Q > tol, lam <= hi, P - lo * Q <= 1e-12)
        return np.where(ok, np.minimum(lam + 1e-9, 1.0), 1.0)
    lam = np.where(Q > tol, np.minimum(P / np.where(Q > tol, Q, 1), hi), hi); ok = np.where(Q > tol, lam >= lo, P - hi * Q >= -1e-12)
    return np.where(ok, np.maximum(lam - 1e-9, 0.0), 0.0)
def vertex_values(L, k, Bd, n, d):
    X = L.inv[k] @ Bd; B = list(L.bases[k]); feas = (X >= -1e-10).all(0); den = d[B] @ X
    v = np.where(feas & (den > 1e-12), (n[B] @ X) / np.where(den > 1e-12, den, 1), np.nan); return v
def screen(F, R, k=20):
    Di = R["Di"]; Bd = F["Bm"][:, Di]; TD = len(Di); hi0, lo0 = R["vals"]["hi0"], R["vals"]["lo0"]; dp, dm = dpdm(F)
    n1, d1 = R["obj"]["hi1"][0], R["obj"]["hi1"][1]
    hu = sig(logit(hi0) + dp); hl = sig(logit(hi0) - dm); lu = sig(logit(lo0) + dp); ll = sig(logit(lo0) - dm)
    hu = np.where(hi0 >= 1, 1.0, hu); hl = np.where(hi0 <= 0, 0.0, hl); lu = np.where(lo0 >= 1, 1.0, lu); ll = np.where(lo0 <= 0, 0.0, ll)
    vmax = np.full(TD, -np.inf); vmin = np.full(TD, np.inf)
    for key in ("hi0", "lo0"):
        for kk in range(len(R["libs"][key].bases)):
            v = vertex_values(R["libs"][key], kk, Bd, n1, d1); vmax = np.fmax(vmax, v); vmin = np.fmin(vmin, v)
    t0 = time.perf_counter(); r_hi = np.empty(TD); r_lo = np.empty(TD)
    for q in range(TD): r_hi[q], r_lo[q] = relax_band(F, Di[q], R["dmin"][q], n1, d1)
    t_u1 = time.perf_counter() - t0
    hl_v = np.fmax(hl, vmax); lu_v = np.fmin(lu, vmin)
    U1 = np.maximum.reduce([r_hi - hi0, hi0 - np.fmax(vmax, 0 * vmax), np.fmin(vmin, 1 + 0 * vmin) - lo0, lo0 - r_lo, np.zeros(TD)])
    U1 = np.where(np.isnan(U1), 1.0, U1)
    Ud_hi = np.ones(TD); Ud_lo = np.zeros(TD); Ls = {"hi1": CL.Library(F["A"]), "lo1": CL.Library(F["A"])}; done = {"hi1": 0, "lo1": 0}
    def bounds():
        HU = np.minimum.reduce([hu, r_hi, Ud_hi]); LL = np.maximum.reduce([ll, r_lo, Ud_lo])
        up = np.maximum.reduce([HU - hi0, hi0 - hl_v, lu_v - lo0, lo0 - LL, np.zeros(TD)])
        low = np.maximum.reduce([hl_v - hi0, hi0 - HU, LL - lo0, lo0 - lu_v, np.zeros(TD)])
        return up, low
    up0, low0 = bounds(); evaluated = np.zeros(TD, bool); exact = np.full(TD, np.nan); solves = 0; rng = np.random.default_rng(3)
    while True:
        up, low = bounds(); ex = exact[evaluated]
        if evaluated.sum() >= k:
            kth = np.sort(ex)[-k]
            if not (up[~evaluated] >= kth).any(): break
        cand = np.where(~evaluated, up, -np.inf); q = int(np.argmax(cand))
        if cand[q] == -np.inf: break
        vv = {}
        for key in ("hi1", "lo1"):
            n, d, s, _, _ = R["obj"][key]; kk, v = CL.certify_one(Ls[key], Bd[:, q], n, d, s)
            if kk < 0:
                _, val, sv, _ = CL.build(Ls[key], Bd[:, [q]], n, d, s, [0], rng); v = float(val[0]); solves += sv
            vv[key] = v
        for key in ("hi1", "lo1"):
            n, d, s, _, _ = R["obj"][key]
            for kk in range(done[key], len(Ls[key].bases)):
                b_ = dual_bound(Ls[key], kk, Bd, n, d, s)
                if b_ is not None:
                    if s > 0: Ud_hi = np.minimum(Ud_hi, b_)
                    else: Ud_lo = np.maximum(Ud_lo, b_)
                v = vertex_values(Ls[key], kk, Bd, n1, d1); hl_v = np.fmax(hl_v, v); lu_v = np.fmin(lu_v, v)
            done[key] = len(Ls[key].bases)
        exact[q] = max(abs(vv["hi1"] - hi0[q]), abs(vv["lo1"] - lo0[q])); evaluated[q] = True
    top = np.sort(exact[evaluated])[-k:]
    return dict(top=top, evaluated=int(evaluated.sum()), share=float(evaluated.mean()), solves=solves, U1=U1, up0=up0, low0=low0, t_u1=t_u1,
                up_final=bounds()[0], low_final=bounds()[1], Ud_hi=Ud_hi, Ud_lo=Ud_lo, r_hi=r_hi, r_lo=r_lo)
def run(name, F):
    R = run_family(name, F, np.random.default_rng(0)); M = F["M"]; dlin = np.zeros(M); dlin[F["den"]] = 1
    Ld = CL.Library(F["A"]); _, vd, _, _ = CL.build(Ld, F["Bm"][:, R["Di"]], dlin, np.zeros(M), -1, np.random.default_rng(1).permutation(len(R["Di"])))
    R["dmin"] = vd; return R
FAM = {"P332": family_pairs(3, 3, 2, 600, 20261129), "P332tol": family_pairs(3, 3, 2, 600, 20261129, tol=E), "P442": family_pairs(4, 4, 2, 800, 20261130)}
# the frozen pair332 population of Study 29 is reproduced exactly (same seed and generator): checked inside SW1 below
RES = {}; SCR = {}; BAND = {}
for name, F in FAM.items():
    t0 = time.perf_counter(); RES[name] = run(name, F); R = RES[name]; R["U0"] = drift_bound(F, R["vals"]["hi0"], R["vals"]["lo0"])
    vb, cb, sb = topk(F, R); BAND[name] = dict(evaluated=int(cb), share=cb / len(R["Di"]), solves=int(sb), top=vb)
    SCR[name] = screen(F, R)
    print(f"  {name}: defined {len(R['Di'])}/{R['T']}; exhaustive solves {sum(R['solves'].values())}; band-only search {BAND[name]['share']:.3f} "
          f"({BAND[name]['solves']} LPs); structure-aware screen {SCR[name]['share']:.3f} ({SCR[name]['solves']} LPs); U1 time {SCR[name]['t_u1']:.1f}s; "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
# SW1 interaction coordinates
ok1 = True; info1 = []
J29 = family_pair(); same = all(np.allclose(FAM["P332"]["Bm"][:, t], J29["Bm"][:, t]) for t in range(0, 10000, 97)); ok1 &= same
for name in ("P332", "P442"):
    F = FAM[name]; R = RES[name]; a, b, c = F["shape"]; cells = F["cells"]; H = []
    for i in range(a - 1):
        for j in range(b - 1):
            for k_ in range(c - 1):
                H.append([(1 if x == i else -1 if x == a - 1 else 0) * (1 if y == j else -1 if y == b - 1 else 0) * (1 if z == k_ else -1 if z == c - 1 else 0) for (x, y, z) in cells])
    H = np.array(H).T; dimH = (a - 1) * (b - 1) * (c - 1); nulldim = F["nx"] - np.linalg.matrix_rank(F["RA"].astype(float))
    ok1 &= (F["RA"] @ H == 0).all() and np.linalg.matrix_rank(H.astype(float)) == dimH == nulldim
    rng = np.random.default_rng(13); bad_rec = 0; bad_lp = 0; HF = [[Fr(int(x)) for x in row] for row in H]
    for q in rng.choice(len(R["Di"]), 200, replace=False):
        xs = []
        for key in ("hi0", "lo0"):
            k_ = R["asg"][key][q]; B = list(R["libs"][key].bases[k_]); AF = [[Fr(int(x)) for x in row] for row in F["A"]]
            xb = CL.gsolve([[AF[i][j] for j in B] for i in range(len(AF))], F["bF"][R["Di"][q]]); x = [Fr(0)] * F["M"]
            for i, j in enumerate(B): x[j] = xb[i]
            xs.append(x[:F["nx"]])
        dlt = [u - v for u, v in zip(*xs)]; HtH = [[sum(HF[r][p] * HF[r][s] for r in range(len(HF))) for s in range(dimH)] for p in range(dimH)]
        th = CL.gsolve(HtH, [sum(HF[r][p] * dlt[r] for r in range(len(HF))) for p in range(dimH)])
        bad_rec += any(sum(HF[r][p] * th[p] for p in range(dimH)) != dlt[r] for r in range(len(HF)))
        base = np.array([float(x) for x in xs[1]]); n1, d1 = R["obj"]["hi1"][0][:F["nx"]], R["obj"]["hi1"][1][:F["nx"]]
        c_ = np.concatenate([-(n1 @ H), [-(n1 @ base)]]); Aub = np.hstack([-H, -base[:, None]]); Aeq = np.concatenate([d1 @ H, [d1 @ base]])[None, :]
        r = linprog(c_, A_ub=Aub, b_ub=np.zeros(F["nx"]), A_eq=Aeq, b_eq=[1.0], bounds=[(None, None)] * dimH + [(0, None)], method="highs")
        bad_lp += abs(-r.fun - R["vals"]["hi1"][q]) > 1e-7
    ok1 &= bad_rec == 0 and bad_lp == 0; info1.append(f"{name}: null-space dimension {nulldim} = (a-1)(b-1)(c-1) = {dimH}; reconstruction failures {bad_rec}/200; programme-in-theta mismatches {bad_lp}/200")
check("SW1 IDENTITY interaction coordinates", ok1, f"P332 is the Study 29 population: {same}; " + "; ".join(info1))
# SW2 validity
ok2 = True; info2 = []
for name in FAM:
    R = RES[name]; S = SCR[name]; D = R["D"]; e_ = 1e-9
    v = dict(U0=(D <= R["U0"] + e_).all(), U1=(D <= S["U1"] + e_).all(), combined0=(D <= S["up0"] + e_).all(), combinedF=(D <= S["up_final"] + e_).all(),
             lower0=(S["low0"] <= D + e_).all(), lowerF=(S["low_final"] <= D + e_).all(),
             relax=((R["vals"]["hi1"] <= S["r_hi"] + e_).all() and (R["vals"]["lo1"] >= S["r_lo"] - e_).all()),
             dual=((R["vals"]["hi1"] <= S["Ud_hi"] + e_).all() and (R["vals"]["lo1"] >= S["Ud_lo"] - e_).all()))
    ok2 &= all(v.values()); info2.append(f"{name}: " + ", ".join(f"{k_} {bool(x)}" for k_, x in v.items()))
check("SW2 IDENTITY every bound dominates the drift; lower bounds lie below it", ok2, "; ".join(info2))
ok3 = all(np.allclose(np.sort(SCR[nm]["top"]), np.sort(RES[nm]["D"])[-20:], atol=1e-9) and np.allclose(np.sort(BAND[nm]["top"]), np.sort(RES[nm]["D"])[-20:], atol=1e-9) for nm in FAM)
check("SW3 IDENTITY screened top-20 equals exhaustive top-20", ok3)
# SW4 context order
bad4 = 0; n4 = 0
for name in FAM:
    F = FAM[name]; R = RES[name]; M = F["M"]; cells = F["cells"]; rng = np.random.default_rng(17)
    L1 = np.ones(M); L1[:F["nx"]] = [float(x) for x in F["L"]]; L2 = np.ones(M); L2[:F["nx"]] = [1 + ((2 * x + y + z) % 3) / 2 for (x, y, z) in cells]
    n0, d0 = R["obj"]["hi0"][0], R["obj"]["hi0"][1]; cC = np.zeros(M); cC[[k_ for k_, cl in enumerate(cells) if cl[2] == 0]] = 1
    for q in rng.choice(len(R["Di"]), 200, replace=False):
        b = F["Bm"][:, R["Di"][q]]
        for s in (+1, -1):
            z1 = CL.solve_ratio(F["A"], b, n0 * L1 * L2, d0 * L1 * L2, s); z2 = CL.solve_ratio(F["A"], b, n0 * L2 * L1, d0 * L2 * L1, s)
            v1 = (n0 * L1 * L2) @ z1 / ((d0 * L1 * L2) @ z1); v2 = (n0 * L2 * L1) @ z2 / ((d0 * L2 * L1) @ z2)
            m, Mm = F["A"].shape; Aeq = np.vstack([np.hstack([F["A"], -b[:, None], np.zeros((m, 1))]), np.concatenate([cC, [0.0, -1.0]])[None, :], np.concatenate([d0, [0.0, 0.0]])[None, :]])
            r = linprog(np.concatenate([-s * n0, [0.0, 0.0]]), A_eq=Aeq, b_eq=np.concatenate([np.zeros(m + 1), [1.0]]), bounds=[(0, None)] * (Mm + 2), method="highs")
            seq = float(n0 @ r.x[:Mm]); direct = R["vals"]["hi0" if s > 0 else "lo0"][q]
            bad4 += abs(v1 - v2) > 1e-9 or abs(seq - direct) > 1e-7; n4 += 1
check("SW4 IDENTITY context order: evidence in either order, and C = 0 then B in {0,1}, give the same bands", bad4 == 0, f"{n4} band ends; mismatches {bad4}")
# SW5 support boundary
Z = family_pairs(3, 3, 2, 600, 20261131, T=100, zero_block=True); M = Z["M"]; dl = np.zeros(M); dl[Z["den"]] = 1; Lz = CL.Library(Z["A"])
_, vz, _, _ = CL.build(Lz, Z["Bm"], dl, np.zeros(M), -1, list(range(100))); zero_ok = bool((vz <= 1e-12).all())
fin = all(np.isfinite(RES[nm]["vals"][k_]).all() for nm in FAM for k_ in RES[nm]["vals"])
Rt = RES["P332tol"]; und = np.where(~Rt["defined"])[0]; F = FAM["P332tol"]; dl = np.zeros(F["M"]); dl[F["den"]] = 1; bad5 = 0
for t in und[:200]:
    z = CL.solve_ratio(F["A"], F["Bm"][:, t], dl, np.zeros(F["M"]), -1); bad5 += (z is not None) and (dl @ z > 1e-9)
check("SW5 IDENTITY support boundary: empty blocks undefined, no non-finite band ends, vanishing tolerance blocks undefined", zero_ok and fin and bad5 == 0,
      f"constructed empty-block tables flagged {int((vz <= 1e-12).sum())}/100; all defined band ends finite {fin}; P332tol undefined {len(und)}, re-checked {min(200, len(und))}, wrong {bad5}")
check("SW6 HYPOTHESIS P332: the screen evaluates <= 50% of defined tables", SCR["P332"]["share"] <= 0.5, f"structure-aware {SCR['P332']['share']:.3f} (band-only {BAND['P332']['share']:.3f})")
rho = {nm: dict(U0=spearmanr(RES[nm]["U0"], RES[nm]["D"])[0], U1=spearmanr(SCR[nm]["U1"], RES[nm]["D"])[0], minU0U1=spearmanr(np.minimum(RES[nm]["U0"], SCR[nm]["U1"]), RES[nm]["D"])[0],
                combined=spearmanr(SCR[nm]["up0"], RES[nm]["D"])[0]) for nm in FAM}
check("SW7 HYPOTHESIS P332: Spearman(min(U0, U1), D) >= 0.7", rho["P332"]["minU0U1"] >= 0.7, "; ".join(f"{nm}: " + ", ".join(f"{k_} {v:.2f}" for k_, v in rho[nm].items()) for nm in FAM))
check("SW8 HYPOTHESIS P332tol and P442: the screen evaluates no larger share than the band-only search", all(SCR[nm]["share"] <= BAND[nm]["share"] for nm in ("P332tol", "P442")),
      "; ".join(f"{nm} {SCR[nm]['share']:.3f} vs {BAND[nm]['share']:.3f}" for nm in ("P332tol", "P442")))
# SW9 records
rec = {}
for nm in FAM:
    R = RES[nm]; F = FAM[nm]; Bd = F["Bm"][:, R["Di"]].T; tree = cKDTree(Bd); _, nb = tree.query(Bd, k=2); nb = nb[:, 1]; a = R["asg"]["hi1"]
    m_ = (a >= 0) & (a[nb] >= 0); stab = float((a[m_] == a[nb][m_]).mean())
    c0 = (R["vals"]["hi0"] + R["vals"]["lo0"]) / 2; c1 = (R["vals"]["hi1"] + R["vals"]["lo1"]) / 2; w0 = R["vals"]["hi0"] - R["vals"]["lo0"]; w1 = R["vals"]["hi1"] - R["vals"]["lo1"]
    rec[nm] = dict(defined=len(R["Di"]), exhaustive_solves=int(sum(R["solves"].values())), band_only=dict(share=BAND[nm]["share"], evaluated=BAND[nm]["evaluated"], solves=BAND[nm]["solves"]),
                   screen=dict(share=SCR[nm]["share"], evaluated=SCR[nm]["evaluated"], solves=SCR[nm]["solves"], t_u1=SCR[nm]["t_u1"]), spearman=rho[nm], active_set_stability=stab,
                   centre_shift=float(np.abs(c1 - c0).mean()), width_change=float((w1 - w0).mean()), width_dominates=float((np.abs(w1 - w0) > np.abs(c1 - c0)).mean()),
                   mean_width0=float(w0.mean()), mean_drift=float(R["D"].mean()), kth=float(np.sort(R["D"])[-20]),
                   U0_median=float(np.median(R["U0"])), U1_median=float(np.median(SCR[nm]["U1"])), up0_median=float(np.median(SCR[nm]["up0"])))
A0, At = RES["P332"], RES["P332tol"]; both = A0["defined"] & At["defined"]; p0 = {t: q for q, t in enumerate(A0["Di"])}; pt = {t: q for q, t in enumerate(At["Di"])}
ids = np.where(both)[0]; D0 = np.array([A0["D"][p0[t]] for t in ids]); Dt = np.array([At["D"][pt[t]] for t in ids])
W0 = np.array([A0["vals"]["hi0"][p0[t]] - A0["vals"]["lo0"][p0[t]] for t in ids]); Wt = np.array([At["vals"]["hi0"][pt[t]] - At["vals"]["lo0"][pt[t]] for t in ids])
rec["interaction_vs_tolerance"] = dict(tables=int(len(ids)), interaction_width=float(W0.mean()), tolerance_width=float((Wt - W0).mean()), interaction_drift=float(D0.mean()),
                                       tolerance_drift=float((Dt - D0).mean()), share_drift_falls=float((Dt < D0 - 1e-12).mean()), share_width_grows=float((Wt > W0 + 1e-12).mean()))
check("SW9 RECORD searches, bounds, stability, centre/width, interaction vs tolerance", True,
      "; ".join(f"{nm}: screen {rec[nm]['screen']['evaluated']} evals/{rec[nm]['screen']['solves']} LPs vs band-only {rec[nm]['band_only']['evaluated']}/{rec[nm]['band_only']['solves']}, "
                f"stability {rec[nm]['active_set_stability']:.2f}, centre {rec[nm]['centre_shift']:.4f}, width change {rec[nm]['width_change']:+.4f}" for nm in FAM)
      + "; interaction vs tolerance " + json.dumps({k_: round(v, 4) for k_, v in rec["interaction_vs_tolerance"].items()}))
json.dump(rec, open(os.path.join(HERE, "study30_structure", "structbound.json"), "w"), indent=1, default=float)
print(f"{sum(results)}/{len(results)} passed")
