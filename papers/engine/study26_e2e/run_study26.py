"""Study 26: one end-to-end example (analysis). Follows PREREG_study26.md (frozen with world26.py; hashes in FROZEN_PREREG_study26.txt).
Writes results_study26.json and prints the XE checks."""
import os, sys, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import warnings; warnings.filterwarnings("ignore")
from scipy.optimize import linprog, minimize
from scipy.stats import chi2
import world26 as W
from pce import refitcommute as RC, selectorlink as SLK

results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

# ---------------------------------------------------------------------------------------------------------------- basic objects
S, V = W.STATES, W.VARS; NS = 32
def ctxmat(ctx, states=S, names=V):
    idx = [names.index(v) for v in ctx]; A = np.zeros((2 ** len(ctx), len(states)))
    for i, s in enumerate(states): A[int("".join(str(s[j]) for j in idx), 2), i] = 1
    return A
def ev(pred): return np.array([1.0 if pred(dict(zip(V, s))) else 0.0 for s in S])
def labels(ctx): return np.array([int("".join(str(s[V.index(v)]) for v in ctx), 2) for s in S])
Pstar = W.truth()
dev, hold = W.generate()

def lp(c, Aub=None, bub=None, Aeq=None, beq=None, n=NS, bounds=None):
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds or [(0, None)] * n, method="highs"); return r

# ---------------------------------------------------------------------------------------------------------------- flagging and pooling
def flag(sources):
    by = {}
    for s in sources: by.setdefault(s["ctx"], []).append(s)
    flagged = set()
    for ctx, ss in by.items():
        for s in ss:
            others = sum(o["counts"] for o in ss if o is not s); fr = others / others.sum(); e = s["n"] * fr; m = e > 1e-9
            stat = float((((s["counts"] - e) ** 2)[m] / e[m]).sum()); p = chi2.sf(stat, m.sum() - 1)
            if p < 0.001: flagged.add(s["id"])
    return flagged
def pool(sources, exclude=()):
    out = {}
    for s in sources:
        if s["id"] in exclude: continue
        out.setdefault(s["ctx"], np.zeros(len(s["counts"]))); out[s["ctx"]] = out[s["ctx"]] + s["counts"]
    return out
core = [s for s in dev if s["ctx"] != ("L3",)]
F = flag(core); kinds = {s["id"]: s["kind"] for s in dev}
fl_shift = sum(kinds[i] == "shift" for i in F); fl_coal = sum(kinds[i] == "coalition" for i in F); fl_other = sum(kinds[i] == "honest" for i in F)
check("XE2 PREDICTION the flagging rule catches the other population, not the coherent coalition", fl_shift >= 8 and fl_other <= 3 and fl_coal < 7.5,
      f"flagged {len(F)}: other-population {fl_shift}/10, coalition {fl_coal}/15, honest {fl_other}/{sum(k == 'honest' for k in kinds.values()) - 30}")
OUT["flag"] = dict(n=len(F), shift=fl_shift, coalition=fl_coal, honest=fl_other)

# ---------------------------------------------------------------------------------------------------------------- fibres
def robust(pooled, z=3.0, states=S, names=V):
    Aub, bub = [], []
    for ctx, cnt in pooled.items():
        A = ctxmat(ctx, states, names); N = cnt.sum(); p = cnt / N; tol = z * np.sqrt(p * (1 - p) / N) + 1e-6
        Aub += [A, -A]; bub += [p + tol, -p + tol]
    return np.vstack(Aub), np.concatenate(bub)
ONE = np.ones((1, NS))
def feasible_ub(Aub, bub, n=NS): return lp(np.zeros(n), Aub, bub, np.ones((1, n)), [1.0], n=n).status == 0
DEVCTX = [c for c, _, _ in W.CORE_ALLOC]
def oracle(ctxs):
    A = np.vstack([ctxmat(c) for c in ctxs]); return A, A @ Pstar
def dim_of(ctxs): A = np.vstack([ctxmat(c) for c in ctxs]); return NS - np.linalg.matrix_rank(A)
d9, d6 = dim_of(DEVCTX), dim_of(DEVCTX + [("M", "C", "R", "Y")])
A0, m0 = oracle(DEVCTX); nonempty = lp(np.zeros(NS), Aeq=A0, beq=m0).status == 0
check("XE1 IDENTITY the oracle fibre is non-empty, dim 9; with {M,C,R,Y} dim 6 (Thm 64(2))", nonempty and d9 == 9 and d6 == 6, f"dim {d9}, then {d6}")
OUT["dims"] = [int(d9), int(d6)]

poolA = pool(core); poolF = pool(core, F)
fa = feasible_ub(*robust(poolA)); ff = feasible_ub(*robust(poolF))
check("XE3 PREDICTION population mixing shows as incompatibility: J_z empty with flagged sources, non-empty without", (not fa) and ff,
      f"with flagged: {'non-empty' if fa else 'empty'}; without: {'non-empty' if ff else 'empty'}")
check("XE4 PREDICTION the coherent coalition passes the compatibility test", ff, "J_z (flagged removed, coalition included) non-empty" if ff else "empty")
# ---- Deviation D1 (DEVIATIONS_study26.md): elastic attribution, z*, and J_z^- when one context carries the slack
def elastic(pooled, z=3.0):
    ctxs = list(pooled); n = NS; k = len(ctxs); Aub, bub = [], []
    for j, ctx in enumerate(ctxs):
        A = ctxmat(ctx); cnt = pooled[ctx]; N = cnt.sum(); p = cnt / N; tol = z * np.sqrt(p * (1 - p) / N) + 1e-6
        col = np.zeros((len(p), k)); col[:, j] = -tol
        Aub += [np.hstack([A, col]), np.hstack([-A, col])]; bub += [p + tol, -p + tol]
    r = linprog(np.r_[np.zeros(n), np.ones(k)], A_ub=np.vstack(Aub), b_ub=np.concatenate(bub), A_eq=np.r_[np.ones(n), np.zeros(k)][None], b_eq=[1.0],
                bounds=[(0, None)] * (n + k), method="highs")
    return {ctxs[j]: float(r.x[n + j]) for j in range(k)}
def zstar(pooled):
    lo, hi = 3.0, 3.0
    while not feasible_ub(*robust(pooled, hi)): hi *= 2
    for _ in range(40):
        mid = (lo + hi) / 2
        if feasible_ub(*robust(pooled, mid)): hi = mid
        else: lo = mid
    return hi
el = elastic(poolF); zs = zstar(poolF); carriers = [c for c, v in el.items() if v > 1e-9]
OUT["deviation_D1"] = dict(elastic={"".join(c): v for c, v in el.items()}, zstar=zs, carriers=["".join(c) for c in carriers])
print("D1 elastic slack by context:", {"".join(c): round(v, 3) for c, v in el.items()}, f"; z* = {zs:.2f}")
poolR = {c: v for c, v in poolF.items() if not (len(carriers) == 1 and c == carriers[0])}
fm = feasible_ub(*robust(poolR))
check("XE4b DEVIATION with the one slack-carrying context removed, J_z^- is non-empty (it removed {M,C,Y}, the coalition context: see D3)", len(carriers) == 1 and fm,
      f"slack carried by {['/'.join(c) for c in carriers]}; J_z^- {'non-empty' if fm else 'empty'}")
# ---- Deviation D3 (post hoc diagnostic): XE4b removed {M,C,Y}, the coalition's own context, so it cannot answer XE4's question.
# Remove {M,C} instead (keeping the coalition's {M,C,Y}) and ask again. Reported, not used downstream.
poolMC = {c: v for c, v in poolF.items() if c != ("M", "C")}; f_mc = feasible_ub(*robust(poolMC))
print(f"D3 diagnostic: J_z with {{M,C}} removed and {{M,C,Y}} kept (coalition included): {'non-empty' if f_mc else 'empty'}", flush=True)
OUT["deviation_D3"] = dict(mc_removed_nonempty=bool(f_mc))
RUB, RBB = robust(poolR)

# ---------------------------------------------------------------------------------------------------------------- hypothesis, paths, ladder
def orange(c, A=A0, m=m0):
    lo = lp(c, Aeq=A, beq=m).fun; hi = -lp(-c, Aeq=A, beq=m).fun; return lo, hi
def frac_range(a, b, Aub=RUB, bub=RBB):
    """range of (a.P)/(b.P) over {Aub P <= bub, sum P = 1, P >= 0} (Charnes-Cooper)"""
    n = NS; Aub2 = np.hstack([Aub, -bub[:, None]]); Aeq = np.vstack([np.r_[b, 0.0], np.r_[np.ones(n), -1.0]]); beq = [1.0, 0.0]
    out = []
    for sg in (1, -1):
        r = linprog(np.r_[sg * a, 0.0], A_ub=Aub2, b_ub=np.zeros(len(bub)), A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (n + 1), method="highs")
        out.append(sg * r.fun)
    return out[0], out[1]
M1 = ev(lambda v: v["M"] == 1); M0 = ev(lambda v: v["M"] == 0); Y1 = ev(lambda v: v["Y"] == 1)
paths = {"pi1 circadian": ev(lambda v: v["M"] and v["R"] and v["Y"]), "pi2 caffeine-free": ev(lambda v: v["M"] and not v["C"] and v["Y"]),
         "pi3 naive": ev(lambda v: v["M"] and not v["H"] and v["Y"])}
E = ev(lambda v: v["M"] and v["Y"] and (v["R"] or not v["C"] or not v["H"]))
LAD = [("none", lambda v: True), ("C=0", lambda v: v["C"] == 0), ("C=0,R=1", lambda v: v["C"] == 0 and v["R"] == 1),
       ("C=0,R=1,H=0", lambda v: v["C"] == 0 and v["R"] == 1 and v["H"] == 0)]
den1 = M1 @ Pstar
pathrng = {k: [x / den1 for x in orange(e)] for k, e in paths.items()}; pathrng["E (union)"] = [x / den1 for x in orange(E)]
pathrob = {k: list(frac_range(e, M1)) for k, e in paths.items()}; pathrob["E (union)"] = list(frac_range(E, M1))
ladder = []
for name, K in LAD:
    k = ev(K); d1, d0 = (M1 * k) @ Pstar, (M0 * k) @ Pstar
    dvec = Y1 * M1 * k / d1 - Y1 * M0 * k / d0; Evec = E * k / d1
    lo, hi = orange(dvec); elo, ehi = orange(Evec)
    r1 = frac_range(Y1 * M1 * k, M1 * k); r0 = frac_range(Y1 * M0 * k, M0 * k); er = frac_range(E * k, M1 * k)
    ladder.append(dict(K=name, delta_oracle=[lo, hi], delta_true=float(dvec @ Pstar), delta_robust=[r1[0] - r0[1], r1[1] - r0[0]],
                       E_oracle=[elo, ehi], E_robust=list(er), E_true=float(Evec @ Pstar)))
OUT["paths_oracle"] = pathrng; OUT["paths_robust"] = pathrob; OUT["ladder"] = ladder
pt = lambda r: r[1] - r[0] < 1e-9
ok = pt(pathrng["pi2 caffeine-free"]) and not pt(pathrng["pi1 circadian"]) and not pt(pathrng["pi3 naive"]) and pt(ladder[0]["delta_oracle"]) \
    and pt(ladder[1]["delta_oracle"]) and not pt(ladder[2]["delta_oracle"]) and not pt(ladder[3]["delta_oracle"])
check("XE5 IDENTITY what the tables identify: pi2 and the first two rungs are points; pi1, pi3 and the last two rungs are open", ok,
      "; ".join(f"{r['K']}: [{r['delta_oracle'][0]:+.3f}, {r['delta_oracle'][1]:+.3f}]" for r in ladder))
tau = 0.10; tg = ladder[3]; certified = tg["delta_oracle"][0] >= tau or tg["delta_oracle"][1] < tau
check("XE6 IDENTITY the target decision is identified iff its range excludes tau (Thm 148(i))", True,
      f"range [{tg['delta_oracle'][0]:+.3f}, {tg['delta_oracle'][1]:+.3f}] vs tau = {tau}: {'certified' if certified else 'open'}; truth {tg['delta_true']:+.3f}")
OUT["target"] = dict(range=tg["delta_oracle"], certified=bool(certified), truth=tg["delta_true"])

# ---------------------------------------------------------------------------------------------------------------- Fisher and VOI
def ipf(ctxs, target=Pstar, Q=None, sweeps=20000):
    P = np.ones(NS) / NS if Q is None else Q.copy(); As = [ctxmat(c) for c in ctxs]; ms = [A @ target for A in As]
    for _ in range(sweeps):
        for A, m in zip(As, ms):
            cur = A @ P; P = P * (A.T @ np.where(cur > 0, m / np.where(cur > 0, cur, 1), 0))
        if max(np.abs(A @ P - m).max() for A, m in zip(As, ms)) < 1e-13: break
    return P
def kerdim(ctxs):
    P = ipf(ctxs); I = sum(ctxmat(c).T @ np.diag(1 / (ctxmat(c) @ P)) @ ctxmat(c) for c in ctxs)
    T = np.linalg.svd(np.eye(NS) - 1 / NS)[0][:, :NS - 1]; ev_ = np.linalg.eigvalsh(T.T @ I @ T); return int((ev_ < 1e-9 * ev_.max()).sum())
k9, k6 = kerdim(DEVCTX), kerdim(DEVCTX + [("M", "C", "R", "Y")])
check("XE7 IDENTITY Fisher: dim ker I_obs equals the fibre dimension (Prop 95)", k9 == d9 and k6 == d6, f"{k9} and {k6}")
urec = M1 * (Y1 - 0.55); uabs = np.zeros(NS); U = [urec, uabs]; Z = [ev(lambda v: v["R"] == 0), ev(lambda v: v["R"] == 1)]
RULES = list(itertools.product(range(2), repeat=2))
def voi_env(A, m):
    Vr = {r: sum(U[r[z]] * Z[z] for z in range(2)) for r in RULES}
    hi = -np.inf
    for r in RULES:   # max V_r.P - t, t >= U_a.P
        c = np.r_[-Vr[r], 1.0]; Aub = np.array([np.r_[U[a], -1.0] for a in range(2)])
        res = linprog(c, A_ub=Aub, b_ub=np.zeros(2), A_eq=np.hstack([A, np.zeros((len(m), 1))]), b_eq=m, bounds=[(0, None)] * NS + [(None, None)], method="highs")
        hi = max(hi, -res.fun)
    lo = np.inf
    for a0 in range(2):  # min t - U_a0.P, t >= V_r.P, U_a0 >= U_b
        c = np.r_[-U[a0], 1.0]; Aub = [np.r_[Vr[r], -1.0] for r in RULES] + [np.r_[U[b] - U[a0], 0.0] for b in range(2) if b != a0]
        res = linprog(c, A_ub=np.array(Aub), b_ub=np.zeros(len(Aub)), A_eq=np.hstack([A, np.zeros((len(m), 1))]), b_eq=m, bounds=[(0, None)] * NS + [(None, None)], method="highs")
        if res.status == 0: lo = min(lo, res.fun)
    full, crit = [], True
    for a0 in range(2):
        for r in RULES:   # max s: a0 and r strictly optimal
            Aub = [np.r_[U[b] - U[a0], 1.0] for b in range(2) if b != a0] + [np.r_[(U[b] - U[r[z]]) * Z[z], 1.0] for z in range(2) for b in range(2) if b != r[z]]
            res = linprog(np.r_[np.zeros(NS), -1.0], A_ub=np.array(Aub), b_ub=np.zeros(len(Aub)), A_eq=np.hstack([A, np.zeros((len(m), 1))]), b_eq=m,
                          bounds=[(0, None)] * NS + [(None, 1.0)], method="highs")
            if res.status == 0 and -res.fun > 1e-9:
                g = sum(Z[z] * (U[r[z]] - U[a0]) for z in range(2)); resid = RC.residual(g, A.T); full.append((a0, r, resid)); crit &= resid < 1e-9
    return lo, hi, full, crit
lo9, hi9, full9, crit9 = voi_env(A0, m0); A6, m6 = oracle(DEVCTX + [("M", "C", "R", "Y")]); lo6, hi6, full6, crit6 = voi_env(A6, m6)
ok = ((hi9 - lo9 < 1e-9) == crit9) and ((hi6 - lo6 < 1e-9) == crit6) and hi6 - lo6 < 1e-9
check("XE8 IDENTITY VOI of R: constant iff the full-dimensional cells' value functionals lie in the data span (Thm 92); constant after {M,C,R,Y}", ok,
      f"development: VOI in [{lo9:.4f}, {hi9:.4f}], {len(full9)} full cells, criterion {crit9}; with {{M,C,R,Y}}: [{lo6:.4f}, {hi6:.4f}], criterion {crit6}")
OUT["voi"] = dict(dev=[lo9, hi9], holdout=[lo6, hi6], full_cells=len(full9))

# ---------------------------------------------------------------------------------------------------------------- selection (coupling M x L3)
def lexmaxmin(Aeq, beq, n, tol=1e-9):
    fixed, free, levels = {}, set(range(n)), []
    while free:
        def cons(t=None):
            Ae = [Aeq] + [np.eye(n)[k][None] for k in fixed]; be = [beq] + [[fixed[k]] for k in fixed]
            return np.vstack(Ae), np.concatenate(be)
        Ae, be = cons(); Aub = np.array([np.r_[-np.eye(n)[k], 1.0] for k in sorted(free)])
        r = linprog(np.r_[np.zeros(n), -1.0], A_ub=Aub, b_ub=np.zeros(len(free)), A_eq=np.hstack([Ae, np.zeros((len(be), 1))]), b_eq=be,
                    bounds=[(0, None)] * n + [(None, None)], method="highs"); t = -r.fun; Sx = []
        for k in sorted(free):
            Aub2 = np.array([-np.eye(n)[j] for j in sorted(free)]); r2 = linprog(-np.eye(n)[k], A_ub=Aub2, b_ub=-np.full(len(free), t) + 1e-12, A_eq=Ae, b_eq=be,
                                                                             bounds=[(0, None)] * n, method="highs")
            if r2.status == 0 and -r2.fun <= t + 1e-7: Sx.append(k)
        if not Sx: Sx = sorted(free)
        for k in Sx: fixed[k] = t; free.discard(k)
        levels.append((round(t, 9), tuple(Sx)))
    return np.array([fixed[k] for k in range(n)]), levels
def coupling_cons(f, g):
    nf, ng = len(f), len(g); A = []; b = []
    for i in range(nf): A.append([1.0 if k // ng == i else 0.0 for k in range(nf * ng)]); b.append(f[i])
    for j in range(ng): A.append([1.0 if k % ng == j else 0.0 for k in range(nf * ng)]); b.append(g[j])
    return np.array(A), np.array(b)
mcount = np.zeros(2)
for ctx, cnt in poolF.items():
    if "M" in ctx: mcount += ctxmat(("M",), [tuple(int(x) for x in format(i, f"0{len(ctx)}b")) for i in range(len(cnt))], ctx) @ cnt
f = mcount / mcount.sum(); lc = sum(s["counts"] for s in dev if s["ctx"] == ("L3",)); g = lc / lc.sum()
xlex, lev = lexmaxmin(*coupling_cons(f, g), 6)
ul = {L: SLK.toric(tuple(f), tuple(g), L) for L in (1e3, 1e4)}; dist = {L: np.linalg.norm(ul[L] - xlex[[0, 1]]) for L in ul}
rel = abs(1e4 * dist[1e4] - 1e3 * dist[1e3]) / max(1e4 * dist[1e4], 1e-300)
check("XE9 IDENTITY the toric point tends to the leximin coupling at rate 1/L (Thm 27)", dist[1e4] < 1e-3 and rel < 0.01,
      f"f = {np.round(f, 4)}, g = {np.round(g, 4)}; leximin levels {[(round(t, 4), s) for t, s in lev]}; L*dist {1e3 * dist[1e3]:.4f}, {1e4 * dist[1e4]:.4f}")
walls, dec, prev, prevd = [], [], None, None
for k in range(-40, 41):
    c = lc + np.array([-k, 0, k]); gk = c / c.sum(); x, lv = lexmaxmin(*coupling_cons(f, gk), 6); struct = tuple(s for _, s in lv); d = x[5] >= 0.10
    if prev is not None and struct != prev: walls.append(k)
    if prevd is not None and d != prevd: dec.append(k)
    prev, prevd = struct, d
me_adv = (f[1] * g[2]) >= 0.10; lex_adv = xlex[5] >= 0.10
check("XE10 PREDICTION selection walls and the decision threshold are crossed at different counts", len(walls) >= 1 and len(dec) >= 1 and not set(dec) & set(walls),
      f"walls at k = {walls}; advisory switches at k = {dec}; at k = 0 leximin P(M=1, bright) = {xlex[5]:.4f} ({'issue' if lex_adv else 'no'}), maximum entropy {f[1] * g[2]:.4f} ({'issue' if me_adv else 'no'})")
OUT["selection"] = dict(f=f.tolist(), g=g.tolist(), lex=xlex.tolist(), levels=[(t, list(s)) for t, s in lev], walls=walls, decision_switch=dec,
                        lex_P_M1_bright=float(xlex[5]), maxent_P_M1_bright=float(f[1] * g[2]))

# ---------------------------------------------------------------------------------------------------------------- composition and system independence
yc = sum(cnt for ctx, cnt in poolF.items() if ctx == ("R", "H", "Y")); h = np.array([yc[0::2].sum(), yc[1::2].sum()]); h = h / h.sum()
def compose(Pxy, Pyz, gy): return Pxy.reshape(len(Pxy) // len(gy), len(gy)) @ (Pyz.reshape(len(gy), -1) / gy[:, None])
me_direct = np.outer(f, h); me_comp = compose(np.outer(f, g).ravel(), np.outer(g, h).ravel(), g)
lx_direct = lexmaxmin(*coupling_cons(f, h), 4)[0].reshape(2, 2); lx_comp = compose(lexmaxmin(*coupling_cons(f, g), 6)[0], lexmaxmin(*coupling_cons(g, h), 6)[0], g)
d_me, d_lx = np.abs(me_direct - me_comp).max(), np.abs(lx_direct - lx_comp).max()
check("XE11 IDENTITY/PREDICTION maximum entropy commutes with composition; leximin does not", d_me < 1e-12 and d_lx > 1e-6,
      f"max |difference|: maximum entropy {d_me:.1e}, leximin {d_lx:.4f}")
cyc = [s for s in hold if s["family"].startswith("cycle")]
def pairpool(cond): return {s["ctx"]: sum(o["counts"] for o in cyc if o["ctx"] == s["ctx"] and o["family"] == "cycle-" + cond) for s in cyc if s["family"] == "cycle-" + cond}
pc = pairpool("consistent"); ab = pc[("A", "B")]; a = np.array([ab[0] + ab[1], ab[2] + ab[3]]) / ab.sum(); b = np.array([ab[0] + ab[2], ab[1] + ab[3]]) / ab.sum()
singles = [f, g, a, b]; dims = [2, 3, 2, 2]; n24 = 24; cells = list(itertools.product(*[range(k) for k in dims]))
A24 = []; b24 = []
for v, (marg, k) in enumerate(zip(singles, dims)):
    for x in range(k): A24.append([1.0 if c[v] == x else 0.0 for c in cells]); b24.append(marg[x])
lx24 = lexmaxmin(np.array(A24), np.array(b24), n24)[0]
prod_lex = np.einsum("ij,kl->ijkl", lexmaxmin(*coupling_cons(f, g), 6)[0].reshape(2, 3), lexmaxmin(*coupling_cons(a, b), 4)[0].reshape(2, 2)).ravel()
me24 = np.einsum("i,j,k,l->ijkl", f, g, a, b).ravel(); prod_me = np.einsum("ij,kl->ijkl", np.outer(f, g), np.outer(a, b)).ravel()
si_me, si_lx = np.abs(me24 - prod_me).max(), np.abs(lx24 - prod_lex).max()
check("XE12 IDENTITY/PREDICTION maximum entropy is system independent; leximin is not (Prop 71, Thm 65)", si_me < 1e-12 and si_lx > 1e-6,
      f"max |section - product of sections|: maximum entropy {si_me:.1e}, leximin {si_lx:.4f}")
OUT["composition"] = dict(maxent=float(d_me), leximin=float(d_lx)); OUT["system_independence"] = dict(maxent=float(si_me), leximin=float(si_lx))

# ---------------------------------------------------------------------------------------------------------------- contextuality: fibre, sheaf, inequality
S4 = list(itertools.product((0, 1), repeat=4)); N4 = ("A", "B", "D", "E")
def cyc_report(pool_, drop=None):
    pool_ = {k: v for k, v in pool_.items() if k != drop}
    Aub, bub = robust(pool_, 3.0, S4, N4); feas = feasible_ub(Aub, bub, 16)
    Es, ses = {}, {}
    for (x, y), cnt in pool_.items():
        p = cnt / cnt.sum(); e = p[0] - p[1] - p[2] + p[3]; Es[(x, y)] = e; ses[(x, y)] = np.sqrt(max(1 - e * e, 1e-12) / cnt.sum())
    best = None
    if drop is None:
        for gam in itertools.product((1, -1), repeat=4):
            if np.prod(gam) != -1: continue
            val = sum(gm * Es[e] for gm, e in zip(gam, W.CYCLE_EDGES)); se = np.sqrt(sum(ses[e] ** 2 for e in W.CYCLE_EDGES))
            if best is None or val > best[0]: best = (val, se, gam)
    supp = {k: (v > 0) for k, v in pool_.items()}; glob = [x for x in S4 if all(supp[(p, q)][2 * x[N4.index(p)] + x[N4.index(q)]] for (p, q) in pool_)]
    logical = any(not any(x[N4.index(p)] == i // 2 and x[N4.index(q)] == i % 2 for x in glob) for (p, q), sp in supp.items() for i in range(4) if sp[i])
    Aub2 = np.vstack([ctxmat(k, S4, N4) for k in pool_]); bub2 = np.concatenate([v / v.sum() for v in pool_.values()])
    ncf = -lp(-np.ones(16), Aub2, bub2, n=16).fun
    A_eq = np.vstack([ctxmat(k, S4, N4) for k in pool_]); dimf = 16 - np.linalg.matrix_rank(A_eq)
    return dict(feasible=bool(feas), best=None if best is None else [float(best[0]), float(best[1]), list(best[2])], full_support=bool(all(v.all() for v in supp.values())),
                global_sections=len(glob), logical=bool(logical), ncf=float(ncf), fibre_dim=int(dimf))
cc, ct, ca = cyc_report(pairpool("consistent")), cyc_report(pairpool("twisted")), cyc_report(pairpool("twisted"), drop=("E", "A"))
ok = cc["feasible"] and cc["best"][0] <= 2 and (not ct["feasible"]) and ct["best"][0] - 2 > 3 * ct["best"][1] and ca["feasible"]
check("XE13 PREDICTION the twisted cycle violates a noncontextuality inequality and has no global law; the consistent and acyclic designs glue", ok,
      f"consistent: largest left side {cc['best'][0]:.3f}, glue {cc['feasible']}; twisted: {ct['best'][0]:.3f} (SE {ct['best'][1]:.3f}, pattern {ct['best'][2]}), glue {ct['feasible']}; acyclic: glue {ca['feasible']}")
ok = cc["full_support"] and ct["full_support"] and not cc["logical"] and not ct["logical"] and ct["ncf"] < 0.95 and cc["ncf"] > 0.98
check("XE14 PREDICTION sheaf layer: full supports, global sections and no logical contextuality; the noncontextual fraction separates the conditions", ok,
      f"global deterministic sections {cc['global_sections']} and {ct['global_sections']}; logical contextuality {cc['logical']}, {ct['logical']}; NCF consistent {cc['ncf']:.4f}, twisted {ct['ncf']:.4f}")
check("XE15 IDENTITY imprecision without contextuality (Prop 137): the consistent and acyclic fibres glue and are not points", cc["feasible"] and ca["feasible"] and cc["fibre_dim"] > 0 and ca["fibre_dim"] > 0,
      f"fibre dimensions {cc['fibre_dim']} (cycle) and {ca['fibre_dim']} (path)")
OUT["cycle"] = dict(consistent=cc, twisted=ct, acyclic=ca)

# ---------------------------------------------------------------------------------------------------------------- history
Bc = {c: RC.span_labels([labels(c)], NS) for c in DEVCTX + [("M", "R", "Y"), ("M", "H", "Y")]}
Ball = RC.span_labels([labels(c) for c in DEVCTX], NS)
def refit(Q, ctxs, target=Pstar):
    B = RC.span_labels([labels(c) for c in ctxs], NS); return RC.iproj(Q, B, B.T @ target)
def inj_law():
    th = dict(W.THETA_TRIPLE); th[("M", "R", "Y")] += 1.5; logp = np.zeros(NS)
    for i, s in enumerate(S):
        v = dict(zip(V, s)); logp[i] = sum(t * v[x] for x, t in W.THETA_MAIN.items()) + sum(t * v[x] * v[y] for (x, y), t in W.THETA_PAIR.items()) \
            + sum(t * v[x] * v[y] * v[z] for (x, y, z), t in th.items())
    p = np.exp(logp); return p / p.sum()
rng = np.random.default_rng(26); o1, o2 = rng.permutation(len(DEVCTX)), rng.permutation(len(DEVCTX)); U0 = np.ones(NS) / NS
def history(order, inject=False):
    P = U0.copy()
    for step, i in enumerate(order):
        P = refit(P, [DEVCTX[i]])
        if inject and step == 2: P = refit(P, [("M", "R", "Y")], inj_law())
    return refit(P, DEVCTX)
RA1, RA2, RB = history(o1), history(o2), history(o1, True); R0 = refit(U0, DEVCTX)
tv = lambda p, q: 0.5 * np.abs(p - q).sum(); trace = np.log(RB) - np.log(RA1)
rMRY = RC.residual(trace, np.column_stack([Ball, Bc[("M", "R", "Y")]])); rMHY = RC.residual(trace, np.column_stack([Ball, Bc[("M", "H", "Y")]]))
ok = tv(RA1, RA2) < 1e-9 and tv(RA1, R0) < 1e-9 and tv(RB, RA1) > 1e-4 and rMRY < 1e-6 and rMHY > 1e-4
check("XE16 IDENTITY history: path independence on the data's contexts; a retracted foreign table leaves a trace attributable to {M,R,Y} (Thm 146)", ok,
      f"TV(order 1, order 2) {tv(RA1, RA2):.1e}; TV(B, A) {tv(RB, RA1):.4f}; trace residual from H_D+F_MRY {rMRY:.1e}, from H_D+F_MHY {rMHY:.3f}")
k = ev(LAD[3][1]); dvec = Y1 * M1 * k / ((M1 * k) @ Pstar) - Y1 * M0 * k / ((M0 * k) @ Pstar)
FMRY = RC.span_labels([labels(("M", "R", "Y"))], NS)
def slice_val(t):   # Deviation D2: trace coordinates bounded by 12 tanh(t/12); a non-converged projection counts as infeasible (nan)
    w = FMRY @ (12 * np.tanh(np.asarray(t) / 12)); Q = R0 * np.exp(w - w.max()); Q = Q / Q.sum()
    try: return dvec @ refit(Q, DEVCTX)
    except RuntimeError: return np.nan
smin, smax = np.inf, -np.inf; rng2 = np.random.default_rng(261)
for _ in range(20):
    t0 = rng2.normal(0, 2, FMRY.shape[1])
    for sg in (1, -1):
        obj = lambda t: (lambda v: 1e6 if np.isnan(v) else sg * v)(slice_val(t))
        r = minimize(obj, t0, method="Nelder-Mead", options=dict(maxiter=600, xatol=1e-6, fatol=1e-9))
        v = slice_val(r.x)
        if not np.isnan(v): smin, smax = min(smin, v), max(smax, v)
flo, fhi = tg["delta_oracle"]; hon = dvec @ R0; honest_act = hon >= tau
rev = (smin < tau) if honest_act else (smax >= tau)
check("XE17 IDENTITY the {M,R,Y} slice lies inside the fibre range; reversal by such histories reported (Thm 148(iii))", flo - 1e-9 <= smin and smax <= fhi + 1e-9,
      f"honest Delta {hon:+.3f} ({'intervene' if honest_act else 'abstain'}); history B {dvec @ RB:+.3f}; slice [{smin:+.3f}, {smax:+.3f}] inside [{flo:+.3f}, {fhi:+.3f}]; reversal {'possible' if rev else 'not possible'}")
OUT["history"] = dict(tvB=float(tv(RB, RA1)), honest=float(hon), histB=float(dvec @ RB), slice=[float(smin), float(smax)], fibre=[float(flo), float(fhi)], reversal=bool(rev),
                      resid_MRY=float(rMRY), resid_MHY=float(rMHY))

# ---------------------------------------------------------------------------------------------------------------- protection
Mq = ctxmat(("M", "Y")); gvec = Y1 * M1 / (M1 @ Pstar) - Y1 * M0 / (M0 @ Pstar); gstar = gvec @ Pstar; tau2 = gstar + 0.04
hctx = [c for c in DEVCTX if c != ("M", "C", "Y")]; Ah, mh = oracle(hctx); qstar = Mq @ Pstar
def l1min(target, Aeq, beq, extra_ub=None, extra_b=None):
    n = NS; k = len(target); c = np.r_[np.zeros(n), np.ones(k)]
    Aub = [np.hstack([Mq, -np.eye(k)]), np.hstack([-Mq, -np.eye(k)])]; bub = [target, -target]
    if extra_ub is not None: Aub.append(np.hstack([extra_ub, np.zeros((len(extra_ub), k))])); bub.append(extra_b)
    Aeq2 = None if Aeq is None else np.hstack([Aeq, np.zeros((len(Aeq), k))])
    r = linprog(c, A_ub=np.vstack(Aub), b_ub=np.concatenate(bub), A_eq=Aeq2, b_eq=beq, bounds=[(0, None)] * (n + k), method="highs"); return r.fun if r.status == 0 else np.inf
rho = l1min(qstar, Ah, mh, -gvec[None], np.array([-tau2]))
qc = sum(s["counts"] for s in hold if s["ctx"] == ("M", "Y")); qobs = qc / qc.sum(); eta = 3 * np.sqrt(qobs * (1 - qobs) / qc.sum()).sum(); kappa = eta
Aall, mall = oracle(DEVCTX); dq_honest = l1min(qobs, Aall, mall)
mcy = poolF[("M", "C", "Y")]; pm = mcy / mcy.sum(); Am = ctxmat(("M", "C", "Y")); tolm = 3 * np.sqrt(pm * (1 - pm) / mcy.sum()) + 1e-6
dq_coal = l1min(qobs, Ah, mh, np.vstack([Am, -Am]), np.r_[pm + tolm, -pm + tolm])
pmy = (ctxmat(("M", "Y"), [tuple(int(x) for x in format(i, "03b")) for i in range(8)], ("M", "C", "Y")) @ mcy) / mcy.sum()
g_pooled = pmy[3] / (pmy[2] + pmy[3]) - pmy[1] / (pmy[0] + pmy[1]); far = g_pooled >= tau2
ok = (rho > 2 * eta) and dq_honest <= kappa and dq_coal > kappa and far
check("XE18 IDENTITY/PREDICTION protection: rho_q > 2 eta, honest reports pass, the coalition's pooled configuration is flagged (Thm 122(iii))", ok,
      f"g(P*) = {gstar:.3f}, tau2 = {tau2:.3f}, pooled g = {g_pooled:.3f} ({'far side' if far else 'near side'}); rho_q = {rho:.4f}, eta = {eta:.4f}; Delta_q honest {dq_honest:.4f}, coalition {dq_coal:.4f}")
OUT["protection"] = dict(gstar=float(gstar), tau2=float(tau2), g_pooled=float(g_pooled), rho=float(rho), eta=float(eta), dq_honest=float(dq_honest), dq_coalition=float(dq_coal))

# ---------------------------------------------------------------------------------------------------------------- holdout calibration
cov, tot, rows = 0, 0, []
for ctx in (("M", "R", "Y"), ("M", "H", "Y")):
    hc = sum(s["counts"] for s in hold if s["ctx"] == ctx); ph = hc / hc.sum(); se = np.sqrt(ph * (1 - ph) / hc.sum()); A = ctxmat(ctx)
    for i in range(len(ph)):
        lo = lp(A[i], RUB, RBB, ONE, [1.0]).fun; hi = -lp(-A[i], RUB, RBB, ONE, [1.0]).fun
        c_ = lo - 3 * se[i] <= ph[i] <= hi + 3 * se[i]; cov += c_; tot += 1; rows.append((ctx, i, lo, hi, float(ph[i])))
check("XE19 PREDICTION holdout calibration: development ranges cover the holdout cells", cov / tot >= 0.9, f"{cov} of {tot} cells covered; median width {np.median([r[3] - r[2] for r in rows]):.3f}")
OUT["calibration"] = dict(covered=int(cov), total=int(tot), median_width=float(np.median([r[3] - r[2] for r in rows])))
json.dump(OUT, open(os.path.join(HERE, "results_study26.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(f"{sum(results)}/{len(results)} passed")
