"""Study 27c runner (specification study27c/SPEC_study27c.md, frozen before this script was run). Calibration on seed 27201, test on
seed 27202, run once. Writes study27c/results_study27c.json. World, generator, scenarios and reporters: Study 27b's code, unchanged.
"""
import os, sys, json, math, itertools, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
D = os.path.join(HERE, "study27c")
h_ = hashlib.sha256(open(os.path.join(D, "SPEC_study27c.md"), "rb").read()).hexdigest()
assert h_ == open(os.path.join(D, "FROZEN_SPEC_study27c.txt")).read().split()[0], "27c specification changed after the freeze"
SMOKE = os.environ.get("SMOKE27C") == "1"
src = open(os.path.join(HERE, "run_study27b.py")).read()
_file = __file__; __file__ = os.path.join(HERE, "run_study27b.py"); os.environ["SMOKE27B"] = "0"
exec(src[:src.index("# features")])
a = src.index("REPORTERS = ["); b = src.index("def feasible("); exec(src[a:b])
a = src.index("def laws(sc, t, a3, ad):"); b = src.index("for sc in SCEN:\n    log(\"scenario\""); exec(src[a:b])
__file__ = _file
STR = ["S0", "S1", "S2", "S3", "S4", "S5", "MRY", "MHY", "I120k"]; SI = {s: i for i, s in enumerate(STR)}
CONF = {"A": ["MRY", "MHY"], "B": ["MRY", "MHY", "I120k"]}
SEX = {s: (SEG[s] if s in SEG else 0.0) for s in STR}
GBLIND = ["S0", "S1", "S2", "S4", "S5"]
def yfun(name, tab):
    ctx = REP_CTX[name]; key = "M" if "M" in ctx else ("Y" if "Y" in ctx else "C"); st = list(itertools.product((0, 1), repeat=len(ctx)))
    return sum(tab[i] for i, s in enumerate(st) if dict(zip(ctx, s))[key] == 1)
def xfun(name, tab): return g_from(name, tab) if ("M" in REP_CTX[name] and "Y" in REP_CTX[name]) else 0.0
NC = 30 if SMOKE else 1000
def simulate(seed, scen):
    rng = np.random.default_rng(seed + (500 if SMOKE else 0)); runs = []
    for sc in scen:
        for _ in range(NC):
            chain = sc in ("relay_fixed", "cascade_timely", "cascade_lagging")
            h, r0 = gen_history(rng, chain); act = act_rounds(h); a3 = act.get(3); ad = int(rng.integers(3, 9)) if sc == "drift" else None
            X = np.zeros((T + 1, 9)); Y = np.zeros((T + 1, 9)); mc = np.full(9, np.inf)
            for t in range(T + 1):
                L, truth = laws(sc, t, a3, ad)
                for name, ctx, n in REPORTERS:
                    if name not in SI: continue
                    pr = CM[name] @ L[name]; pr = pr / pr.sum(); tb = rng.multinomial(n, pr) / n; i = SI[name]
                    X[t, i] = xfun(name, tb); Y[t, i] = yfun(name, tb); mc[i] = min(mc[i], n * tb.min())
            runs.append(dict(sc=sc, X=X, Y=Y, mc=mc, act={f"S{k}": v for k, v in act.items()}))
    return runs
def pair_count(run, k, A, z, braid):
    i, j = SI[k], SI[A]; kap = z * math.sqrt(2) * math.sqrt(SEX[k] ** 2 + SEX[A] ** 2)
    dx = run["X"] - run["X"][0]; s = dx[:, i] - dx[:, j] - kap
    if braid: return int(s[-1] > 0), s
    bky, bAy = i * 1e-3, j * 1e-3; dy = run["Y"] - run["Y"][0]; c = 0
    for t in range(T):
        s0, s1 = s[t], s[t + 1]
        if s0 == 0: s0 = -1e-15
        if s1 == 0: s1 = -1e-15
        if s0 * s1 < 0:
            lam = s0 / (s0 - s1); yk = bky + dy[t, i] + lam * (dy[t + 1, i] - dy[t, i]); yA = bAy + dy[t, j] + lam * (dy[t + 1, j] - dy[t, j])
            c += (1 if s0 < 0 else -1) * (1 if yk > yA else -1)
    return abs(c), s
def stat(run, conf, z, comp):
    anchors = CONF[conf]; cnt = {}; S = {}
    for A in anchors:
        for k in STR[:6]: cnt[(k, A)], S[(k, A)] = pair_count(run, k, A, z, comp["braid"])
    excluded = set(); departed = []; guard = 0
    if comp["contextual"]:
        for A in anchors:
            if any(cnt[(k, A)] >= 1 for k in GBLIND): departed.append(A); excluded |= {(k, A) for k in STR[:6]}
    if comp["geometric"]:
        bad = {s for s in STR if run["mc"][SI[s]] < 30}
        for key in cnt:
            if key[0] in bad or key[1] in bad: excluded.add(key); guard += 1
    Phi = sum(v for key, v in cnt.items() if key not in excluded)
    E = sorted({k for (k, A), v in cnt.items() if v >= 1 and (k, A) not in excluded})
    last = {}
    for k in E:
        rs = [t for (kk, A), s in S.items() if kk == k and (kk, A) not in excluded for t in range(1, T + 1) if s[t - 1] <= 0 < s[t]]
        last[k] = max(rs) if rs else None
    expl = {k: (k in run["act"] and last[k] is not None and run["act"][k] <= last[k]) for k in E}
    return Phi >= 1, dict(E=E, departed=departed, guard=guard, explained=expl)
FULL = dict(braid=True, contextual=True, geometric=True)
def rate(runs, sc, conf, z, comp): return float(np.mean([stat(r, conf, z, comp)[0] for r in runs if r["sc"] == sc]))
def loss(runs, conf, z, comp):
    miss = 1 - rate(runs, "relay_fixed", conf, z, comp)
    fa = np.mean([rate(runs, sc, conf, z, comp) for sc in ("honest", "shock", "cascade_timely")])
    return 0.5 * miss + 0.5 * 0.2 * fa
R = {"spec_sha256": h_}
cal = simulate(27201, ["honest", "shock", "cascade_timely", "relay_fixed"])
ZG = [round(2.0 + 0.05 * i, 2) for i in range(41)]
calib = {}
for conf in CONF:
    z = next((z for z in ZG if rate(cal, "honest", conf, z, FULL) <= 0.05), ZG[-1])
    comp = dict(FULL); steps = []
    for c in ("geometric", "contextual", "braid"):
        l_on = loss(cal, conf, z, comp); trial = dict(comp); trial[c] = False; l_off = loss(cal, conf, z, trial)
        steps.append(dict(component=c, loss_on=l_on, loss_off=l_off, switched_off=bool(l_off <= l_on)))
        if l_off <= l_on: comp = trial
    calib[conf] = dict(z=z, components=comp, voi_steps=steps, honest_rate_at_z=rate(cal, "honest", conf, z, FULL))
R["calibration"] = calib
test = simulate(27202, ["honest", "shock", "relay_fixed", "cascade_timely", "cascade_lagging", "drift"])
TS = {}
for conf in CONF:
    z, comp = calib[conf]["z"], calib[conf]["components"]
    rates = {sc: rate(test, sc, conf, z, comp) for sc in ("honest", "shock", "relay_fixed", "cascade_timely", "cascade_lagging", "drift")}
    bal = 0.5 * (rates["relay_fixed"] + 1 - rates["cascade_timely"])
    abl = {}
    for c in ("braid", "contextual", "geometric"):
        cc = dict(comp); cc[c] = not cc[c]
        rr = {sc: rate(test, sc, conf, z, cc) for sc in ("honest", "shock", "relay_fixed", "cascade_timely")}
        abl[f"{c}={'on' if cc[c] else 'off'}"] = dict(rates=rr, balanced=0.5 * (rr["relay_fixed"] + 1 - rr["cascade_timely"]))
    emap = {}; guard = 0
    for sc in rates:
        st = [stat(r, conf, z, comp)[1] for r in test if r["sc"] == sc]; guard += sum(x["guard"] for x in st)
        Es = {}
        for x in st: Es[",".join(x["E"]) or "none"] = Es.get(",".join(x["E"]) or "none", 0) + 1
        ne = sum(len(x["explained"]) for x in st)
        planted = ["S3"] if sc in ("relay_fixed", "drift") else []
        emap[sc] = dict(E_sets=dict(sorted(Es.items(), key=lambda q: -q[1])[:5]),
                        changes_after_activation=(sum(sum(x["explained"].values()) for x in st) / ne) if ne else None,
                        anchor_departures=int(sum(len(x["departed"]) for x in st)), agree_planted=float(np.mean([x["E"] == planted for x in st])))
    TS[conf] = dict(z=z, components=comp, rates=rates, balanced_accuracy=bal, ablation=abl, explanation_map=emap, guard_activations=guard)
P = {}
P["P1"] = all(TS[c]["rates"]["honest"] <= 0.065 for c in CONF)
P["P2"] = all(TS[c]["rates"]["shock"] <= 0.10 for c in CONF)
P["P3"] = TS["B"]["balanced_accuracy"] >= 0.90
P["P4"] = TS["B"]["balanced_accuracy"] >= TS["A"]["balanced_accuracy"] - 0.01
P["P5"] = all(TS[c]["rates"]["cascade_lagging"] >= 0.90 for c in CONF)
P["P6"] = all(TS[c]["guard_activations"] == 0 for c in CONF)
R["test"] = TS; R["predictions"] = {k: bool(v) for k, v in P.items()}
json.dump(R, open(os.path.join(D, "results_study27c.json") if not SMOKE else "/dev/null", "w"), indent=1, default=str)
print("written", flush=True)
