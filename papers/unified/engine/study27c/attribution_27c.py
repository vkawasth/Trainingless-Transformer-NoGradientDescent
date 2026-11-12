"""Study 27c, step 1: attribution of the witness-braid noise to its sources (EXPLORATORY diagnosis; no hypothesis, no verdict of 27b changes).
Same world, generator and witness rules as Study 27b (code reused from run_study27b.py, unchanged); fresh seed 27101.
Five switchable noise sources, evaluated on common random numbers (each run samples once; every switch setting reuses the samples):
  N1 source sampling   off: source reports are the exact marginals of their laws
  N2 anchor sampling   off: anchor and inspection reports exact
  N3 baseline noise    off: positions are measured from the exact round-0 value instead of the noisy round-0 report
  N4 repeated looks    off: only rounds 0 and 12 are compared (one look)
  N5 y noise           off: the second coordinate is exact (noise-free), so a crossing and its return always cancel
Metric v(S) for each subset S of switched-on sources: relay-like rate (Phi >= 1) per scenario and configuration. Exact Shapley values
split v(all on) - v(all off) among N1-N5. Separately, N6 degeneration: delta-method SE of g against the empirical SD, the second-order
bias, the Fisher condition number and the smallest cell probability of each strand's context table at the states visited.
Also writes the region/context overlap table (OVERLAP_TABLE.md).
"""
import os, sys, json, math, itertools
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
src = open(os.path.join(HERE, "run_study27b.py")).read()
__file__ = os.path.join(HERE, "run_study27b.py"); os.environ["SMOKE27B"] = "0"
exec(src[:src.index("# features")])                                        # constants, histories (unchanged code)
a = src.index("REPORTERS = ["); b = src.index("def feasible(")
exec(src[a:b])                                                              # reporters, SEs, g_from
a = src.index("def laws(sc, t, a3, ad):"); b = src.index("for sc in SCEN:\n    log(\"scenario\"")
exec(src[a:b])
OUT = os.path.join(HERE, "study27c")
SRC_ = [f"S{k}" for k in range(6)]; CONF = {"A": ["MRY", "MHY"], "B": ["MRY", "MHY", "I120k"]}
def kappa(nm): return 2 * math.sqrt(SEG["S3"] ** 2 + SEG[nm] ** 2)
def yfun(name, tab):
    ctx = REP_CTX[name]; key = "M" if "M" in ctx else ("Y" if "Y" in ctx else "C"); st = list(itertools.product((0, 1), repeat=len(ctx)))
    return sum(tab[i] for i, s in enumerate(st) if dict(zip(ctx, s))[key] == 1)
def xfun(name, tab): return g_from(name, tab) if ("M" in REP_CTX[name] and "Y" in REP_CTX[name]) else 0.0

def Phi_vec(X, Y, base, anchors):
    """X, Y: arrays (rounds, strands) of displacements; strands = SRC_ + anchors. Net source-anchor crossing exponents, vectorised."""
    S = SRC_ + anchors; bx = np.array([base[s][0] for s in S]); by = np.array([base[s][1] for s in S])
    PX = X + bx; PY = Y + by; tot = 0
    for ai, A in enumerate(anchors):
        j = 6 + ai
        d = PX[:, :6] - PX[:, [j]]                                           # (R, 6)
        d = np.where(d == 0, -1e-15, d)
        d0, d1 = d[:-1], d[1:]; cr = d0 * d1 < 0
        lam = np.where(cr, d0 / np.where(cr, d0 - d1, 1), 0)
        yk = PY[:-1, :6] + lam * (PY[1:, :6] - PY[:-1, :6]); yA = PY[:-1, [j]] + lam * (PY[1:, [j]] - PY[:-1, [j]])
        dirk = np.where(d0 < 0, 1, -1); front = np.where(yk > yA, 1, -1)
        c = np.sum(np.where(cr, dirk * front, 0), axis=0); tot += np.abs(c).sum()
    return tot

SCN = ["honest", "shock", "relay_fixed", "cascade_timely", "cascade_lagging", "drift"]
NRUN = 600
rng = np.random.default_rng(27101)
combos = list(itertools.product((0, 1), repeat=5))
V = {c: {sc: {k: [] for k in combos} for sc in SCN} for c in CONF}
DEG = {nm: dict(g_hat=[], g_true=[], mincell=[], cond=[]) for nm in ("S3", "MRY", "MHY", "I120k")}
for sc in SCN:
    for run in range(NRUN):
        chain = sc in ("relay_fixed", "cascade_timely", "cascade_lagging")
        h, r0 = gen_history(rng, chain); a3 = act_rounds(h).get(3); ad = int(rng.integers(3, 9)) if sc == "drift" else None
        exact, noisy = [], []
        for t in range(T + 1):
            L, truth = laws(sc, t, a3, ad); ex, no = {}, {}
            for name, ctx, n in REPORTERS:
                pr = CM[name] @ L[name]; pr = pr / pr.sum(); ex[name] = pr; no[name] = rng.multinomial(n, pr) / n
                if name in DEG and t in (0, T):
                    DEG[name]["g_hat"].append(xfun(name, no[name])); DEG[name]["g_true"].append(xfun(name, pr))
                    DEG[name]["mincell"].append(float(pr.min())); DEG[name]["cond"].append(float(pr.max() / pr.min()))
            exact.append(ex); noisy.append(no)
        for c, anchors in CONF.items():
            S = SRC_ + anchors
            base = {f"S{k}": (k * 1e-4, k * 1e-3) for k in range(6)}
            for j, nm in enumerate(anchors): base[nm] = (kappa(nm) + (6 + j) * 1e-4, (6 + j) * 1e-3)
            xe = np.array([[xfun(s_, exact[t][s_]) for s_ in S] for t in range(T + 1)]); xn = np.array([[xfun(s_, noisy[t][s_]) for s_ in S] for t in range(T + 1)])
            ye = np.array([[yfun(s_, exact[t][s_]) for s_ in S] for t in range(T + 1)]); yn = np.array([[yfun(s_, noisy[t][s_]) for s_ in S] for t in range(T + 1)])
            issrc = np.array([s_.startswith("S") for s_ in S])
            for cb in combos:
                n1, n2, n3, n4, n5 = cb
                use = np.where(issrc, n1, n2).astype(bool)                 # which strands are noisy
                xs = np.where(use, xn, xe); ys_noisy = np.where(use, yn, ye)
                rounds = list(range(T + 1)) if n4 else [0, T]
                x0 = xs[0] if n3 else xe[0]
                yy = ys_noisy if n5 else ye; y0 = yy[0] if n3 else ye[0]
                X = xs[rounds] - x0; Y = yy[rounds] - y0
                V[c][sc][cb].append(Phi_vec(X, Y, base, anchors) >= 1)
    print("done", sc, flush=True)
names = ["N1 source sampling", "N2 anchor sampling", "N3 baseline noise", "N4 repeated looks", "N5 y noise"]
def shapley(v):
    n = 5; phi = []
    for i in range(n):
        s = 0.0
        for cb in combos:
            if cb[i] == 1: continue
            k = sum(cb); w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            cb1 = tuple(1 if j == i else cb[j] for j in range(n)); s += w * (v[cb1] - v[cb])
        phi.append(s)
    return phi
RES = {"note": "exploratory; seed 27101; 600 runs per scenario", "shapley": {}, "rates": {}}
for c in CONF:
    for sc in SCN:
        v = {cb: float(np.mean(V[c][sc][cb])) for cb in combos}
        RES["rates"][f"{c}/{sc}"] = {"all_on": v[(1, 1, 1, 1, 1)], "all_off": v[(0, 0, 0, 0, 0)],
                                    **{f"only_{names[i].split()[0]}": v[tuple(1 if j == i else 0 for j in range(5))] for i in range(5)},
                                    **{f"all_but_{names[i].split()[0]}": v[tuple(0 if j == i else 1 for j in range(5))] for i in range(5)}}
        RES["shapley"][f"{c}/{sc}"] = dict(zip(names, shapley(v)))
deg = {}
for nm, d in DEG.items():
    gh, gt = np.array(d["g_hat"]), np.array(d["g_true"]); n_ = dict((x[0], x[2]) for x in REPORTERS)[nm]
    err = gh - gt
    deg[nm] = dict(n=n_, empirical_sd=float(err.std()), delta_se_at_P=float(se_g(n_)), ratio=float(err.std() / se_g(n_)),
                   bias=float(err.mean()), bias_over_se=float(err.mean() / se_g(n_)), min_cell_prob=float(min(d["mincell"])),
                   min_expected_cell_count=float(min(d["mincell"]) * n_), max_cond=float(max(d["cond"])))
RES["degeneration"] = deg
json.dump(RES, open(os.path.join(OUT, "attribution_27c.json"), "w"), indent=1)

# region / context overlap table
ALL = REPORTERS
def region(pr, n):
    m = pr.min() * n
    return "near face" if m < 5 else ("boundary layer" if m < 30 else "interior")
lines = ["# Region / context overlap table (Study 27b world; maintained for 27b/27c)", "",
         "Strands of the witness braid: their contexts, which variables they share, whether they see the decision quantity g (contain M",
         "and Y), whether they are hexagon neighbours (co-revision edges), and the region of the moment polytope their table occupies",
         "(smallest expected cell count: interior >= 30, boundary layer 5-30, near face < 5) under the truth P and the far-side law Q.", ""]
hdr = "| strand | context | n | sees g | hexagon neighbours | min cell P | min cell Q | region (P) | region (Q) |"
lines += [hdr, "|" + "---|" * 9]
nb = {k: sorted({j for e in E6 for j in e if k in e and j != k}) for k in range(6)}
for name, ctx, n in ALL:
    pP = CM[name] @ P; pQ = CM[name] @ Q
    neigh = ", ".join(f"S{j}" for j in nb[int(name[1])]) if name.startswith("S") else "-"
    lines.append(f"| {name} | {{{','.join(ctx)}}} | {n} | {'yes' if ('M' in ctx and 'Y' in ctx) else 'no'} | {neigh} | {pP.min():.4f} | {pQ.min():.4f} | {region(pP, n)} | {region(pQ, n)} |")
lines += ["", "Shared variables between contexts (overlap = the restriction maps of the sheaf):", "",
          "| | " + " | ".join(nm for nm, _, _ in ALL) + " |", "|" + "---|" * (len(ALL) + 1)]
for n1_, c1, _ in ALL:
    lines.append(f"| {n1_} | " + " | ".join("".join(v for v in c1 if v in c2) or "-" for _, c2, _ in ALL) + " |")
open(os.path.join(OUT, "OVERLAP_TABLE.md"), "w").write("\n".join(lines) + "\n")
print(json.dumps(RES["degeneration"], indent=1))
