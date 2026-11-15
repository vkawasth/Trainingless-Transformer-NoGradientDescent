"""Credal geometry, mixing and flow (section sec:credalgeom). New script; it modifies nothing; uses pce/credalgeom.py, pce/sleepworld.py.
Declared before the first run (the central experiment, GM8-GM9):
  credal set K: joint sleep-world laws whose margins on the 14 source contexts (lifestyle, diary, wearable, psg) are within 0.01 of the
  honest law's, and whose (P, C, T, O) margin is within a factor e^(+-0.5) of the honest one in every cell (relative-error lab anchors;
  without them every (P, C, T, O) cell can be 0). The flow acts on the 24 cells of (P, C, T, O): mixing L flips the circadian window C
  at rate mu and the vasodilation reading T at rate nu; evidence tilts toward T = 1 at rate beta. Configurations: A tilt only
  (beta 1, mu = nu = 0); B mixing only (beta 0, mu = nu = 0.3); C both (beta 1, mu = nu = 0.1). Times 0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6.
  Primary band: P(O = 2 | P = 0, C = 1), log-odds width. Hypothesis H: in configuration C there are consecutive times at which the
  Hilbert diameter of K_t strictly falls while the primary band strictly widens. Reported either way.
  GM1 IDENTITY  Hilbert diameter by linear-fractional programmes equals the brute-force maximum over vertex pairs (300 random polytopes);
                structural zeros are dropped; touching zeros give an infinite diameter; the eps-profile is non-increasing in eps
  GM2 IDENTITY  conditioning commutes with mixing for sets: the band of P(a | C) over a polytope is attained at conditioned vertices; the
                conditioned mixture is the mixture of conditioned vertices with weights lam_k(C) = lam_k V_k(C) / sum lam_j V_j(C)
  GM3 IDENTITY  the ceiling: the log-odds band of any P(a | C) has width at most the Hilbert diameter (2000 cases); the coin example
                attains it (width = diameter = log 81)
  GM4 IDENTITY  the flow p -> p M_t / <p M_t, 1>, M_t = exp(t (L + diag a)), solves the replicator-mutator equation (ODE, 1e-8); the
                diameter never increases along it; with L = 0 it is constant (isometry)
  GM5 IDENTITY  sleep world, boundaries: from the sources alone every (P, C, T, O) cell can be 0, so the diameter is infinite (reported
                with the touching cells); the eps-profile is finite and non-increasing; with the anchors the diameter is finite
  GM6 IDENTITY  sleep world, band ladder for O = 2 under T = 1, T = 1 and P = 0, T = 1 and P = 0 and C = 1 (and complements), from the
                sources alone and with anchors: every band satisfies the total-probability constraint against its two children;
                undefined bands are flagged; dilation (a child wider than its parent) is flagged
  GM7 RECORD    mixing over the unreported population Z: P(O = 2 | C) against the prior-weight mixture of the populations' conditionals
                (the point-level conditioning-mixing discrepancy) and the condition-dependent weights w_z(C)
  GM8 RECORD    the central experiment (configurations A, B, C): diameter and bands over time; verdict on H
  GM9 IDENTITY  the ceiling holds at every time of every configuration (band widths in log-odds <= diameter)
"""
import os, sys, math, json, itertools
import numpy as np
from scipy.integrate import solve_ivp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import credalgeom as CG, sleepworld as SW
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(20261121)
def vspec(V):
    """mixing weights lam >= 0, sum 1, as base variables; cells y = V^T lam."""
    k = len(V); return dict(A_ub=None, b_ub=None, A_eq=np.ones((1, k)), b_eq=np.array([1.0])), np.array(V).T
# GM1
bad = 0; inf_ok = 0; mono_bad = 0
for it in range(300):
    m = int(rng.integers(3, 7)); k = int(rng.integers(2, 6)); V = [rng.dirichlet(np.ones(m) * 2) for _ in range(k)]
    if it % 10 == 0: V = [np.concatenate([v[:-1], [0.0]]) / v[:-1].sum() for v in V]          # a structural zero
    spec, Y = vspec(V); d = CG.diam_lp(spec, Y); bf = CG.diam_vertices(V)
    bad += abs(d["diam"] - bf) > 1e-7 * max(1, bf)
    if it % 10 == 5:                                                                          # touching zero: one vertex on the face
        V2 = V[:-1] + [np.concatenate([V[-1][:-1], [0.0]]) / V[-1][:-1].sum()]; s2, Y2 = vspec(V2); d2 = CG.diam_lp(s2, Y2)
        inf_ok += d2["diam"] == float("inf") and (m - 1) in d2["touching"]
        prof = CG.eps_profile(s2, Y2, [0.002, 0.01, 0.03]); vals = [x for _, x in prof if x is not None]
        mono_bad += any(vals[i + 1] > vals[i] + 1e-9 for i in range(len(vals) - 1))
check("GM1 IDENTITY Hilbert diameter by LP equals the vertex brute force; structural zeros dropped; touching zeros infinite; eps-profile monotone",
      bad == 0 and inf_ok == 30 and mono_bad == 0, f"300 polytopes; mismatches {bad}; touching detected {inf_ok}/30; profile violations {mono_bad}")
# GM2
bad = 0
for _ in range(300):
    m = int(rng.integers(3, 7)); k = int(rng.integers(2, 6)); V = [rng.dirichlet(np.ones(m) * 2) for _ in range(k)]; spec, Y = vspec(V)
    C = (rng.random(m) < 0.6).astype(float); C[int(rng.integers(m))] = 1; aC = C * (rng.random(m) < 0.5)
    b = CG.band(spec, Y.T @ aC if False else aC @ Y, C @ Y)
    cv = [float(aC @ v) / float(C @ v) for v in V]; bad += abs(b["lo"] - min(cv)) > 1e-9 or abs(b["hi"] - max(cv)) > 1e-9
    lam = rng.dirichlet(np.ones(k)); x = sum(l * v for l, v in zip(lam, V)); w = CG.vertex_weights(lam, V, C)
    xc = x * C / float(C @ x); mixc = sum(wi * (v * C / float(C @ v)) for wi, v in zip(w, V)); bad += np.abs(xc - mixc).max() > 1e-12
check("GM2 IDENTITY sets: bands sit at conditioned vertices; the conditioned mixture uses weights lam_k V_k(C) / sum lam_j V_j(C)", bad == 0, f"300 cases; violations {bad}")
# GM3
bad = 0; slack = []
for _ in range(2000):
    m = int(rng.integers(3, 7)); k = int(rng.integers(2, 6)); V = [rng.dirichlet(np.ones(m) * 2) for _ in range(k)]; spec, Y = vspec(V)
    C = (rng.random(m) < 0.7).astype(float); a = (rng.random(m) < 0.5).astype(float)
    if (a * C).sum() == 0 or ((1 - a) * C).sum() == 0: continue
    lb = CG.logodds_band(spec, (a * C) @ Y, ((1 - a) * C) @ Y); D = CG.diam_vertices(V)
    bad += lb[1] - lb[0] > D + 1e-9; slack.append((lb[1] - lb[0]) / D if D > 0 else 1)
def coin(rho): q = 0.25 * (1 + rho); r = 0.25 * (1 - rho); return np.array([q, r, r, q])
Vc = [coin(-0.8), coin(0.8)]; sc, Yc = vspec(Vc); aC = np.array([0, 0, 0, 1.0]) @ Yc; nC = np.array([0, 1.0, 0, 0]) @ Yc   # X=1,Y=1 vs X=0,Y=1
lbc = CG.logodds_band(sc, aC, nC); Dc = CG.diam_vertices(Vc)
check("GM3 IDENTITY the ceiling: every conditional log-odds band is at most the Hilbert diameter; the coin example attains it", bad == 0 and abs((lbc[1] - lbc[0]) - Dc) < 1e-9 and abs(Dc - math.log(81)) < 1e-9,
      f"{len(slack)} cases; violations {bad}; median width/diameter {np.median(slack):.2f}; coin width {lbc[1] - lbc[0]:.4f} = diameter {Dc:.4f}")
# GM4
bad_ode = 0; bad_mono = 0; iso = 0.0
for _ in range(60):
    m = int(rng.integers(3, 6)); L = rng.uniform(0, 0.5, size=(m, m)); np.fill_diagonal(L, 0); L -= np.diag(L.sum(1)); a = rng.normal(size=m)
    p0 = rng.dirichlet(np.ones(m)); sol = solve_ivp(lambda t, p: p @ L + p * (a - p @ a), (0, 2.0), p0, rtol=1e-11, atol=1e-13)
    M = CG.flow_matrix(L, a, 2.0); pM = p0 @ M; pM /= pM.sum(); bad_ode += np.abs(sol.y[:, -1] - pM).max() > 1e-8
    V = [rng.dirichlet(np.ones(m) * 2) for _ in range(4)]; prev = CG.diam_vertices(V)
    for t in (0.3, 0.8, 1.5, 3.0):
        Mt = CG.flow_matrix(L, a, t); Vt = [(v @ Mt) / (v @ Mt).sum() for v in V]; d = CG.diam_vertices(Vt); bad_mono += d > prev + 1e-9; prev = d
    Vt0 = [(v @ CG.flow_matrix(0 * L, a, 1.7)) for v in V]; Vt0 = [v / v.sum() for v in Vt0]; iso = max(iso, abs(CG.diam_vertices(Vt0) - CG.diam_vertices(V)))
check("GM4 IDENTITY the flow map solves the replicator-mutator equation; the diameter never increases; tilt alone is an isometry", bad_ode == 0 and bad_mono == 0 and iso < 1e-9,
      f"60 generators; ODE mismatches {bad_ode}; increases {bad_mono}; tilt-only change {iso:.1e}")
# sleep world credal sets
W = SW.W; p = SW.law(); NS = W.NS; rows = []; b = []
src = sorted({tuple(sorted(ctx, key=W.vars.index)) for fam, role, items in SW.FAMILIES if fam not in ("vendor", "lab") for ctx, r, n in items})
for c in src:
    A = W.A(c); f = A @ p; rows += [A, -A]; b += [f + 0.01, -(f - 0.01)]
A_ub = np.vstack(rows); b_ub = np.concatenate(b); EQ = dict(A_eq=np.ones((1, NS)), b_eq=np.array([1.0]))
spec_src = dict(A_ub=A_ub, b_ub=b_ub, **EQ)
VC = ("P", "C", "T", "O"); R = W.A(VC); m0 = R @ p; RHO = 0.5
spec_lab = dict(A_ub=np.vstack([A_ub, R, -R]), b_ub=np.concatenate([b_ub, m0 * np.exp(RHO), -m0 * np.exp(-RHO)]), **EQ)
cells = list(itertools.product(range(2), range(2), range(2), range(3)))                 # (P, C, T, O) in the order of W.A
def ind(f_): return np.array([1.0 if f_(*c) else 0.0 for c in cells])
d_src = CG.diam_lp(spec_src, R); prof = CG.eps_profile(spec_src, R, [0.002, 0.005, 0.01]); d_lab = CG.diam_lp(spec_lab, R)
pv = [x for _, x in prof if x is not None]
check("GM5 IDENTITY sleep world: from the sources alone the (P,C,T,O) diameter is infinite (all cells touch 0); eps-profile finite and monotone; anchors make it finite",
      d_src["diam"] == float("inf") and len(d_src["touching"]) == 24 and all(pv[i + 1] <= pv[i] + 1e-9 for i in range(len(pv) - 1)) and np.isfinite(d_lab["diam"]),
      f"touching {len(d_src['touching'])}/24; eps-profile " + ", ".join(f"{e}: {x:.2f}" if x is not None else f"{e}: empty" for e, x in prof) + f"; with anchors {d_lab['diam']:.3f}")
# GM6 band ladder
conds = {"all": lambda P_, C, T, O: True, "T=1": lambda P_, C, T, O: T == 1, "T=0": lambda P_, C, T, O: T == 0,
         "T=1,P=0": lambda P_, C, T, O: T == 1 and P_ == 0, "T=1,P=1": lambda P_, C, T, O: T == 1 and P_ == 1,
         "T=1,P=0,C=1": lambda P_, C, T, O: T == 1 and P_ == 0 and C == 1, "T=1,P=0,C=0": lambda P_, C, T, O: T == 1 and P_ == 0 and C == 0,
         "P=0,C=1": lambda P_, C, T, O: P_ == 0 and C == 1}
kids = {"all": ("T=1", "T=0"), "T=1": ("T=1,P=0", "T=1,P=1"), "T=1,P=0": ("T=1,P=0,C=1", "T=1,P=0,C=0")}
table = {}; tp_bad = 0; undefined = []; dil = []
for nm_s, spec in (("sources", spec_src), ("anchors", spec_lab)):
    tb = {}
    for nm, f_ in conds.items():
        den = ind(f_) @ R; num = ind(lambda P_, C, T, O, f_=f_: f_(P_, C, T, O) and O == 2) @ R; bd = CG.band(spec, num, den)
        tb[nm] = bd; undefined += [(nm_s, nm)] if bd["undefined"] else []
    for par, (k1, k2) in kids.items():
        if tb[par]["undefined"] or tb[k1]["undefined"] or tb[k2]["undefined"]: continue
        lo = min(tb[k1]["lo"], tb[k2]["lo"]); hi = max(tb[k1]["hi"], tb[k2]["hi"]); tp_bad += tb[par]["lo"] < lo - 1e-9 or tb[par]["hi"] > hi + 1e-9
        for k_ in (k1, k2):
            if tb[k_]["hi"] - tb[k_]["lo"] > tb[par]["hi"] - tb[par]["lo"] + 1e-9: dil.append((nm_s, par, k_))
    table[nm_s] = tb
fmt = lambda d: "undefined" if d["undefined"] else f"[{d['lo']:.3f}, {d['hi']:.3f}]"
check("GM6 IDENTITY band ladder for O = 2: total-probability constraint holds; undefined bands and dilation flagged", tp_bad == 0,
      "; ".join(f"{s}: " + ", ".join(f"{k} {fmt(v)}" for k, v in table[s].items() if k in ("all", "T=1", "T=1,P=0", "T=1,P=0,C=1", "P=0,C=1")) for s in table)
      + f"; undefined {undefined}; dilation {dil}; violations {tp_bad}")
# GM7 population mixture
Zc = SW.col("Z"); pz = [p * (Zc == z) for z in (0, 1)]; wz = [float(x.sum()) for x in pz]
rowsZ = []
for nm in ("T=1", "T=1,P=0,C=1", "P=0,C=1"):
    f_ = conds[nm]; den = ind(f_) @ R; num = ind(lambda P_, C, T, O, f_=f_: f_(P_, C, T, O) and O == 2) @ R
    true = float(num @ p) / float(den @ p); comp = [float(num @ x) / float(den @ x) for x in pz]
    naive = sum(w * c for w, c in zip(wz, comp)); wc = [w * float(den @ x) / wz_ for w, x, wz_ in zip(wz, pz, wz)]; wc = np.array(wc) / sum(wc)
    rowsZ.append(f"{nm}: true {true:.3f}, prior-weight mixture {naive:.3f}, weights given the condition ({wc[0]:.2f}, {wc[1]:.2f}) vs prior ({wz[0]:.2f}, {wz[1]:.2f})")
check("GM7 RECORD conditioning reweights the unreported populations: the point-level conditioning-mixing discrepancy", True, "; ".join(rowsZ))
# GM8-GM9 the central experiment
def gen(mu, nu):
    L = np.zeros((24, 24))
    for i, (P_, C, T, O) in enumerate(cells):
        for j, (P2, C2, T2, O2) in enumerate(cells):
            if (P2, T2, O2) == (P_, T, O) and C2 != C: L[i, j] = mu
            if (P2, C2, O2) == (P_, C, O) and T2 != T: L[i, j] = nu
    return L - np.diag(L.sum(1))
CONF = dict(A=(1.0, 0.0, 0.0), B=(0.0, 0.3, 0.3), C=(1.0, 0.1, 0.1)); TIMES = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6]
prim = ("P=0,C=1",); sec = ("T=1,P=0,C=1",); out = {}; ceil_bad = 0
for nm, (beta, mu, nu) in CONF.items():
    L = gen(mu, nu); a = beta * np.array([1.0 if c[2] == 1 else 0.0 for c in cells]); rows_t = []
    for t in TIMES:
        M = CG.flow_matrix(L, a, t); Rt = (R.T @ M).T                                          # cells at time t: y = p R^T M
        D = CG.diam_lp(spec_lab, Rt)["diam"]; rec = dict(t=t, diam=D)
        for key in prim + sec:
            f_ = conds[key]; aC = ind(lambda P_, C, T, O, f_=f_: f_(P_, C, T, O) and O == 2) @ Rt; nC = ind(lambda P_, C, T, O, f_=f_: f_(P_, C, T, O) and O != 2) @ Rt
            lo_, hi_ = CG.logodds_band(spec_lab, aC, nC); bd = CG.band(spec_lab, aC, aC + nC)
            rec[key] = dict(lo=bd["lo"], hi=bd["hi"], lo_lo=lo_, hi_lo=hi_); ceil_bad += (hi_ - lo_) > D + 1e-9
        rows_t.append(rec); print(nm, t, f"diam {D:.4f}", {k: round(rec[k]['hi_lo'] - rec[k]['lo_lo'], 4) for k in prim + sec}, flush=True)
    out[nm] = rows_t
# vertices for the figure: random-objective LP solutions of K (joint laws), then their (P, C, T, O) margins
from scipy.optimize import linprog
verts = []
for _ in range(80):
    r = linprog(rng.normal(size=NS), A_ub=spec_lab["A_ub"], b_ub=spec_lab["b_ub"], A_eq=EQ["A_eq"], b_eq=EQ["b_eq"], bounds=[(0, None)] * NS, method="highs")
    if r.status == 0: verts.append((R @ r.x).tolist())
json.dump(dict(conf=CONF, times=TIMES, out=out, verts=verts, cells=cells, ladder={s: {k: v for k, v in tb.items()} for s, tb in table.items()},
               diam_src=str(d_src["diam"]), eps_profile=prof, diam_lab=d_lab["diam"]), open(os.path.join(HERE, "credalgeom_results.json"), "w"), indent=1, default=str)
w = lambda r_: r_[prim[0]]["hi_lo"] - r_[prim[0]]["lo_lo"]; Cr = out["C"]
H = any(Cr[i + 1]["diam"] < Cr[i]["diam"] - 1e-9 and w(Cr[i + 1]) > w(Cr[i]) + 1e-9 for i in range(len(Cr) - 1))
check("GM8 RECORD central experiment: in configuration C the diameter falls while the P(O=2 | P=0, C=1) band widens (H)", True,
      f"H {'held' if H else 'did not hold'}; " + "; ".join(f"{nm}: diam {out[nm][0]['diam']:.3f} -> {out[nm][-1]['diam']:.3f}, band {w(out[nm][0]):.3f} -> {max(w(r_) for r_ in out[nm]):.3f} (max) -> {w(out[nm][-1]):.3f}" for nm in out))
check("GM9 IDENTITY the ceiling holds at every time of every configuration", ceil_bad == 0, f"{3 * len(TIMES) * 2} bands; violations {ceil_bad}")
print(f"{sum(results)}/{len(results)} passed")
