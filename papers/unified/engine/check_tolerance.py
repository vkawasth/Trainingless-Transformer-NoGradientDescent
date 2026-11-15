"""Study 29: credal sets beyond exact margins; certified top-k and a stratified probability sample (sec:ledger). Pre-registered in
study29_tolerance/PREREG_study29.md, frozen by SHA-256 (study29_tolerance/FROZEN_PREREG_study29.txt) before any code for it ran.
New script and module (pce/credalledger.py); modifies nothing.
  TK1 IDENTITY   tol3 at e = 0 reproduces the exact-margin bands of stratledger.json (all defined tables, four ends, 1e-9)
  TK2 IDENTITY   bands nest in e (every tol3 table, every end, 1e-9)
  TK3 IDENTITY   values agree with direct LPs (200 tables per family and end, 1e-7); exact re-check of 100 certified tables per family/end
  TK4 IDENTITY   U >= D everywhere; certified top-20 values equal the exhaustive top-20 values
  TK5 HYPOTHESIS LP solves (four ends + defined test) <= 0.2 T, families with e > 0 and pair332
  TK6 HYPOTHESIS certified top-20 evaluates the after-evidence ends of <= 50% of defined tables, every family
  TK7 HYPOTHESIS stratified 95% intervals cover the truth in >= 93% of 500 replicates, both targets, every family
  TK8 HYPOTHESIS stratified variance <= simple-random-sampling variance for the mean drift, every family
  TK9 RECORD     widths, drifts, defined share, solves, fallbacks, top-k evaluations, half-widths, cost of the sample
First run (the only run): 8/9. TK6 FAILED for pair332 only: certified top-20 had to evaluate 81% of the tables (tol families 27-32%).
Post-hoc diagnosis (not a check): the logistic bound depends on a table only through hi0 and lo0 and peaks at about 0.172 when an end is
near 1/2; in pair332 the bands are narrow and interior (mean width 0.22), so U sits near that peak for most tables (median 0.169), while
the 20th largest drift is 0.146; 81% of tables have U >= 0.146 and cannot be excluded. Spearman(U, D) = 0.31 in pair332 and tol3-1/100.
Threshold unchanged.
Output: check_tolerance_output.txt, study29_tolerance/tolerance.json (aggregates only; synthetic data).
"""
import os, sys, json, time, itertools
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import credalledger as CL
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def nondeg(f, g):
    fs = {sum(c) for k in range(1, len(f)) for c in itertools.combinations(f, k)}; gs = {sum(c) for k in range(1, len(g)) for c in itertools.combinations(g, k)}
    return not (fs & gs)
def margins(rng, k, den):
    v = rng.dirichlet([2] * k) * den; x = np.maximum(1, np.floor(v)).astype(int); x[int(np.argmax(x))] += den - x.sum()
    return [Fr(int(y), den) for y in x] if x.min() >= 1 else None
def population(r, c, den, seed, n=10000):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        f, g = margins(rng, r, den), margins(rng, c, den)
        if f and g and nondeg(f, g): out.append((f, g))
    return out
def family_tol(r, c, den, seed, e):
    A, cells = CL.tolerance_form(r, c); pop = population(r, c, den, seed); M = A.shape[1]; ix = {cl: k for k, cl in enumerate(cells)}
    bF = [CL.b_tol(f, g, e) for f, g in pop]; Bm = np.array([[float(x) for x in b] for b in bF]).T
    num = [ix[(0, 0)]]; den_ = [ix[cl] for cl in ((0, 0), (0, 1), (1, 0), (1, 1))]; L = [Fr(1) + Fr((i + 2 * j) % 3, 2) for (i, j) in cells]
    cov = np.array([float(f[0] + f[1] + g[0] + g[1]) for f, g in pop])
    return dict(A=A, M=M, nx=len(cells), bF=bF, Bm=Bm, num=num, den=den_, L=L, cov=cov, pop=pop)
def family_pair(seed=20261129, T=10000):
    A, cells, keys = CL.pairwise_form(3, 3, 2); rng = np.random.default_rng(seed); bF = []; cov = []
    blk = [k for k, cl in enumerate(cells) if cl[1] in (0, 1) and cl[2] == 0]
    for _ in range(T):
        v = np.floor(rng.dirichlet([1] * len(cells)) * 600).astype(int); v[int(np.argmax(v))] += 600 - v.sum(); joint = [Fr(int(x), 600) for x in v]
        bF.append(CL.b_pair(joint, cells, keys)); cov.append(float(sum(joint[k] for k in blk)))
    Bm = np.array([[float(x) for x in b] for b in bF]).T
    num = [k for k, cl in enumerate(cells) if cl[0] == 0 and cl[1] in (0, 1) and cl[2] == 0]
    L = [Fr(1) + Fr((a + 2 * b + c) % 3, 2) for (a, b, c) in cells]
    return dict(A=A, M=A.shape[1], nx=len(cells), bF=bF, Bm=Bm, num=num, den=blk, L=L, cov=np.array(cov), pop=None)
def objectives(F):
    M = F["M"]; n0 = np.zeros(M); d0 = np.zeros(M); n0[F["num"]] = 1; d0[F["den"]] = 1; Lf = np.ones(M); Lf[:F["nx"]] = [float(x) for x in F["L"]]
    n1, d1 = n0 * Lf, d0 * Lf
    nF0 = [Fr(int(x)) for x in n0]; dF0 = [Fr(int(x)) for x in d0]; LF = list(F["L"]) + [Fr(1)] * (M - F["nx"])
    nF1 = [a * b for a, b in zip(nF0, LF)]; dF1 = [a * b for a, b in zip(dF0, LF)]
    return dict(hi0=(n0, d0, +1, nF0, dF0), lo0=(n0, d0, -1, nF0, dF0), hi1=(n1, d1, +1, nF1, dF1), lo1=(n1, d1, -1, nF1, dF1))
def drift_bound(F, hi0, lo0):
    lL = np.log([float(x) for x in F["L"]]); rest = [k for k in F["den"] if k not in F["num"]]
    dp = max(lL[F["num"]].max() - lL[rest].min(), 0); dm = max(lL[rest].max() - lL[F["num"]].min(), 0); sig = lambda x: 1 / (1 + np.exp(-x))
    def mv(p):
        with np.errstate(divide="ignore", invalid="ignore"):
            l = np.log(p) - np.log1p(-p)
            return np.where((p > 0) & (p < 1), np.maximum(sig(l + dp) - p, p - sig(l - dm)), 0.0)
    return np.maximum(mv(hi0), mv(lo0))
def run_family(name, F, rng):
    T = F["Bm"].shape[1]; Bm = F["Bm"]; M = F["M"]; t0 = time.perf_counter(); order = np.random.default_rng(1).permutation(T)
    # defined test: min d.x > 0 (linear objective, certified)
    dlin = np.zeros(M); dlin[F["den"]] = 1; Ld = CL.Library(F["A"]); ad, vd, sd, fd = CL.build(Ld, Bm, dlin, np.zeros(M), -1, order, rng)
    defined = vd > 1e-12; Di = np.where(defined)[0]; Bd = Bm[:, Di]; ordD = np.random.default_rng(1).permutation(len(Di))
    obj = objectives(F); vals = {}; asg = {}; libs = {}; solves = {"defined": sd}; fb = {"defined": fd}
    for key, (n, d, s, _, _) in obj.items():
        L = CL.Library(F["A"]); a, v, sv, fv = CL.build(L, Bd, n, d, s, ordD, rng); vals[key] = v; asg[key] = a; libs[key] = L; solves[key] = sv; fb[key] = fv
    t_ex = time.perf_counter() - t0
    D = np.maximum(np.abs(vals["hi1"] - vals["hi0"]), np.abs(vals["lo1"] - vals["lo0"])); U = drift_bound(F, vals["hi0"], vals["lo0"])
    return dict(T=T, defined=defined, Di=Di, vals=vals, asg=asg, libs=libs, solves=solves, fallback=fb, D=D, U=U, t_ex=t_ex, obj=obj)
def topk(F, R, k=20):
    Di = R["Di"]; Bd = F["Bm"][:, Di]; U = R["U"]; ordU = np.argsort(-U); Ls = {key: CL.Library(F["A"]) for key in ("hi1", "lo1")}
    got = []; solves = 0; rng = np.random.default_rng(3)
    for cnt, q in enumerate(ordU, 1):
        vv = {}
        for key in ("hi1", "lo1"):
            n, d, s, _, _ = R["obj"][key]; kk, v = CL.certify_one(Ls[key], Bd[:, q], n, d, s)
            if kk < 0:
                order = [0]; a, val, sv, fv = CL.build(Ls[key], Bd[:, [q]], n, d, s, order, rng); v = float(val[0]); solves += sv
            vv[key] = v
        got.append(max(abs(vv["hi1"] - R["vals"]["hi0"][q]), abs(vv["lo1"] - R["vals"]["lo0"][q])))
        if len(got) >= k:
            kth = np.sort(got)[-k]; nxt = U[ordU[cnt]] if cnt < len(ordU) else -1
            if nxt < kth: break
    return np.sort(got)[-k:], cnt, solves
def strata(cov, H=4):
    q = np.quantile(cov, [0.25, 0.5, 0.75]); return np.searchsorted(q, cov, side="right")
def sample_study(F, R, n=400, reps=500, seed=29):
    T = R["T"]; Dfull = np.zeros(T); Dfull[R["Di"]] = R["D"]; z = R["defined"].astype(float); y = Dfull * z; w = ((Dfull > 0.05) & R["defined"]).astype(float)
    truth_mean = y.sum() / z.sum(); truth_share = w.mean(); h = strata(F["cov"]); Hs = sorted(set(h)); Nh = np.array([(h == s).sum() for s in Hs])
    nh = np.floor(n * Nh / T).astype(int); rem = n - nh.sum(); nh[np.argsort(-(n * Nh / T - nh))[:rem]] += 1
    rng = np.random.default_rng(seed); cover = dict(mean_str=0, share_str=0, mean_srs=0, share_srs=0); est = dict(str=[], srs=[]); hw = dict(mean=[], share=[])
    idx_h = [np.where(h == s)[0] for s in Hs]
    for _ in range(reps):
        # stratified
        Yh = []; Zh = []; Wh = []; sel = []
        for g, s in enumerate(Hs):
            S = rng.choice(idx_h[g], nh[g], replace=False); sel.append(S)
        Yt = sum(Nh[g] * y[S].mean() for g, S in enumerate(sel)); Zt = sum(Nh[g] * z[S].mean() for g, S in enumerate(sel)); Rh = Yt / Zt if Zt > 0 else 0.0
        V = sum(Nh[g] ** 2 * (1 - nh[g] / Nh[g]) * np.var(y[S] - Rh * z[S], ddof=1) / nh[g] for g, S in enumerate(sel)) / Zt ** 2
        Ws = sum(Nh[g] * w[S].mean() for g, S in enumerate(sel)) / T; Vw = sum((Nh[g] / T) ** 2 * (1 - nh[g] / Nh[g]) * np.var(w[S], ddof=1) / nh[g] for g, S in enumerate(sel))
        cover["mean_str"] += abs(Rh - truth_mean) <= 1.96 * np.sqrt(V); cover["share_str"] += abs(Ws - truth_share) <= 1.96 * np.sqrt(Vw)
        est["str"].append(Rh); hw["mean"].append(1.96 * np.sqrt(V)); hw["share"].append(1.96 * np.sqrt(Vw))
        # simple random sampling
        S = rng.choice(T, n, replace=False); R2 = y[S].sum() / z[S].sum(); fpc = 1 - n / T
        V2 = fpc * np.var(y[S] - R2 * z[S], ddof=1) / n / z[S].mean() ** 2; W2 = w[S].mean(); Vw2 = fpc * np.var(w[S], ddof=1) / n
        cover["mean_srs"] += abs(R2 - truth_mean) <= 1.96 * np.sqrt(V2); cover["share_srs"] += abs(W2 - truth_share) <= 1.96 * np.sqrt(Vw2); est["srs"].append(R2)
    return dict(truth_mean=float(truth_mean), truth_share=float(truth_share), coverage={k: v / reps for k, v in cover.items()},
                var_str=float(np.var(est["str"])), var_srs=float(np.var(est["srs"])), halfwidth_mean=float(np.mean(hw["mean"])),
                halfwidth_share=float(np.mean(hw["share"])), allocation=nh.tolist())
# ---------------------------------------------------------------- families
FAM = {}
for e in (Fr(0), Fr(1, 200), Fr(1, 100), Fr(1, 50)): FAM[f"tol3-{e}"] = family_tol(3, 3, 120, 20261126, e)
FAM["tol4-1/100"] = family_tol(4, 4, 240, 20261127, Fr(1, 100)); FAM["pair332"] = family_pair()
RES = {}; rng = np.random.default_rng(0)
for name, F in FAM.items():
    RES[name] = run_family(name, F, rng); R = RES[name]
    print(f"  {name}: defined {len(R['Di'])}/{R['T']}; solves {R['solves']}; fallbacks {R['fallback']}; libraries {[len(R['libs'][k].bases) for k in R['libs']]}; "
          f"mean width {np.mean(R['vals']['hi0'] - R['vals']['lo0']):.4f}; mean drift {R['D'].mean():.4f}; {R['t_ex']:.1f}s", flush=True)
# TK1
S = json.load(open(os.path.join(HERE, "stratledger.json")))["3x3"]; R0 = RES["tol3-0"]; pop = FAM["tol3-0"]["pop"]
key_of = {(tuple(str(x) for x in f), tuple(str(x) for x in g)): t for t, (f, g) in enumerate(pop)}; pos = {t: q for q, t in enumerate(R0["Di"])}
bad1 = 0; n1 = 0; miss = 0
for tb in S["tables"]:
    t = key_of[(tuple(tb["f"]), tuple(tb["g"]))]
    if t not in pos: miss += 1; continue
    for k in ("hi0", "lo0", "hi1", "lo1"): n1 += 1; bad1 += abs(R0["vals"][k][pos[t]] - tb[k]) > 1e-9
undef_ok = len(R0["Di"]) == len(S["tables"])
check("TK1 IDENTITY tolerance form at e = 0 reproduces the exact-margin bands of the ledger", bad1 == 0 and miss == 0 and undef_ok,
      f"{n1} band ends compared; mismatches {bad1}; ledger tables not defined here {miss}; defined {len(R0['Di'])} vs ledger {len(S['tables'])}")
# TK2
names = ["tol3-0", "tol3-1/200", "tol3-1/100", "tol3-1/50"]; bad2 = 0; cmp2 = 0
full = {}
for nm in names:
    R = RES[nm]; T = R["T"]; full[nm] = {k: np.full(T, np.nan) for k in R["vals"]}
    for k in R["vals"]: full[nm][k][R["Di"]] = R["vals"][k]
for a, b in zip(names, names[1:]):
    both = ~np.isnan(full[a]["hi0"]) & ~np.isnan(full[b]["hi0"])
    for k in ("hi0", "hi1"): bad2 += int((full[b][k][both] < full[a][k][both] - 1e-9).sum()); cmp2 += int(both.sum())
    for k in ("lo0", "lo1"): bad2 += int((full[b][k][both] > full[a][k][both] + 1e-9).sum()); cmp2 += int(both.sum())
    bad2 += int((~np.isnan(full[a]["hi0"]) & np.isnan(full[b]["hi0"])).sum())       # defined at smaller e must stay defined? (no: block can vanish) counted below
defined_lost = sum(int((~np.isnan(full[a]["hi0"]) & np.isnan(full[b]["hi0"])).sum()) for a, b in zip(names, names[1:]))
check("TK2 IDENTITY bands nest as the tolerance grows", bad2 - defined_lost == 0, f"{cmp2} comparisons; violations {bad2 - defined_lost}; tables that become undefined as e grows {defined_lost}")
# TK3
bad3 = 0; bad3x = 0; info3 = []
for name, F in FAM.items():
    R = RES[name]; Bd = F["Bm"][:, R["Di"]]; r3 = np.random.default_rng(11); lpbad = 0; exbad = 0
    for key, (n, d, s, nF, dF) in R["obj"].items():
        for q in r3.choice(len(R["Di"]), min(200, len(R["Di"])), replace=False):
            z = CL.solve_ratio(F["A"], Bd[:, q], n, d, s); v = (n @ z) / (d @ z); lpbad += abs(v - R["vals"][key][q]) > 1e-7
        cert = np.where(R["asg"][key] >= 0)[0]
        for q in r3.choice(cert, min(100, len(cert)), replace=False):
            ok, v = CL.exact_check(F["A"], R["libs"][key].bases[R["asg"][key][q]], F["bF"][R["Di"][q]], nF, dF, s); exbad += (not ok) or abs(float(v) - R["vals"][key][q]) > 1e-9
    bad3 += lpbad; bad3x += exbad; info3.append(f"{name} {lpbad}/{exbad}")
check("TK3 IDENTITY values agree with direct LPs; exact rational re-check of certified tables", bad3 == 0 and bad3x == 0, "LP / exact failures: " + "; ".join(info3))
# TK4 + TK6
ok4 = True; info4 = []; TOPK = {}
for name, F in FAM.items():
    R = RES[name]; Uok = bool((R["D"] <= R["U"] + 1e-12).all()); vals_k, cnt, sv = topk(F, R); true_k = np.sort(R["D"])[-20:]
    same = np.allclose(np.sort(vals_k), true_k, atol=1e-9); ok4 &= Uok and same; TOPK[name] = dict(evaluated=int(cnt), share=cnt / len(R["Di"]), solves=int(sv))
    info4.append(f"{name}: U>=D {Uok}, top-20 equal {same}, evaluated {cnt}/{len(R['Di'])}, solves {sv}")
check("TK4 IDENTITY drift bound valid; certified top-20 equals exhaustive top-20", ok4, "; ".join(info4))
TEST = ["tol3-1/200", "tol3-1/100", "tol3-1/50", "tol4-1/100", "pair332"]
tot = {nm: sum(RES[nm]["solves"].values()) for nm in FAM}
check("TK5 HYPOTHESIS compression persists: LP solves <= 0.2 T", all(tot[nm] <= 0.2 * RES[nm]["T"] for nm in TEST), "; ".join(f"{nm} {tot[nm]}" for nm in FAM))
check("TK6 HYPOTHESIS certified top-20 evaluates <= 50% of defined tables", all(TOPK[nm]["share"] <= 0.5 for nm in FAM), "; ".join(f"{nm} {TOPK[nm]['share']:.2f}" for nm in FAM))
SAMP = {nm: sample_study(FAM[nm], RES[nm]) for nm in TEST}
check("TK7 HYPOTHESIS stratified 95% intervals cover in >= 93% of replicates, both targets", all(SAMP[nm]["coverage"]["mean_str"] >= 0.93 and SAMP[nm]["coverage"]["share_str"] >= 0.93 for nm in TEST),
      "; ".join(f"{nm} mean {SAMP[nm]['coverage']['mean_str']:.3f} share {SAMP[nm]['coverage']['share_str']:.3f} (SRS {SAMP[nm]['coverage']['mean_srs']:.3f}/{SAMP[nm]['coverage']['share_srs']:.3f})" for nm in TEST))
check("TK8 HYPOTHESIS stratification reduces the variance of the mean-drift estimate", all(SAMP[nm]["var_str"] <= SAMP[nm]["var_srs"] for nm in TEST),
      "; ".join(f"{nm} {SAMP[nm]['var_str'] / SAMP[nm]['var_srs']:.2f}" for nm in TEST))
check("TK9 RECORD widths, drifts, defined share, top-k, half-widths, sample cost", True,
      "; ".join(f"{nm}: defined {len(RES[nm]['Di']) / RES[nm]['T']:.2f}, width {np.mean(RES[nm]['vals']['hi0'] - RES[nm]['vals']['lo0']):.3f}, drift {RES[nm]['D'].mean():.4f}, "
                f"fallbacks {sum(RES[nm]['fallback'].values())}" + (f", truth {SAMP[nm]['truth_mean']:.4f}+-{SAMP[nm]['halfwidth_mean']:.4f}, share {SAMP[nm]['truth_share']:.3f}+-{SAMP[nm]['halfwidth_share']:.3f}" if nm in SAMP else "") for nm in FAM))
OUT = {nm: dict(T=RES[nm]["T"], defined=len(RES[nm]["Di"]), solves=RES[nm]["solves"], fallback=RES[nm]["fallback"], libraries={k: len(v.bases) for k, v in RES[nm]["libs"].items()},
                mean_width=float(np.mean(RES[nm]["vals"]["hi0"] - RES[nm]["vals"]["lo0"])), mean_drift=float(RES[nm]["D"].mean()), max_drift=float(RES[nm]["D"].max()),
                share_moved=float((RES[nm]["D"] > 1e-9).mean()), time_exhaustive=RES[nm]["t_ex"], topk=TOPK[nm], sample=SAMP.get(nm),
                drift_hist=np.histogram(RES[nm]["D"], bins=20, range=(0, max(RES[nm]["D"].max(), 1e-9)))[0].tolist()) for nm in FAM}
json.dump(OUT, open(os.path.join(HERE, "study29_tolerance", "tolerance.json"), "w"), indent=1, default=float)
print(f"{sum(results)}/{len(results)} passed")
