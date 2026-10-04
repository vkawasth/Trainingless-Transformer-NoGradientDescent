"""Blackwell-graded region search: reward-free exact pruning, higher-order paths, and obstruction intervals.

(A) Exact pruning. Theta ~ Bernoulli(1/2) and n binary measurements:
      X1..X3 fair coins; X4 = Theta xor X1 xor X2 xor X3, flipped w.p. 0.1 (a parity block: any 3 of X1..X4 say
      nothing about Theta, all 4 say a lot); X5, X6 = Theta flipped w.p. 0.35, 0.40 (visible pairwise);
      X7 = X5 flipped w.p. 0.1 (a noisy copy); X8 = Theta flipped w.p. 0.25 (good but costly);
      for n = 10, 12: further weak direct signals and noisy copies.
    Costs 1 per measurement except X7 (0.5) and X8 (3). Every region's standard measure is computed; the cost-Blackwell
    frontier is found once, with no reward. Then for three convex rewards (accuracy; asymmetric loss with false
    negatives x3; three actions with an abstain option worth 0.7) and every budget / cost price, the optimum over ALL
    regions is checked to lie on the frontier.
(B) Higher-order paths. Under a budget of 4, compare the region found by
      greedy-MI      add the measurement with the largest single-variable MI with Theta
      greedy-value   add the measurement with the largest immediate gain in accuracy (myopic)
      pairwise-MI    best region by the sum of pairwise MIs I(Theta; X_i, X_j) over its pairs
      frontier       best region on the Blackwell frontier
    and show the value along the growth path (the parity block is flat until its last step).
(C) Obstructions. Contexts {Theta, X1}, {X1, X2}, {X2, Theta} sampled separately, from the family
    e_lam = lam * (contradictory cycle: Theta = X1, X1 = X2, X2 != Theta) + (1 - lam) * uniform noise.
    Region: decide Theta from (X1, X2). Naive maximum-entropy gluing (IPF) reports a value; the honest value is the
    interval over every consistent joint, and none exists once the family is contextual.
"""
import os, sys, json, time, itertools
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
from amb_vigneaux import blackwell as B

REWARDS = {"accuracy": np.array([[1.0, 0.0], [0.0, 1.0]]),
           "asymmetric (FN x3)": np.array([[1.0, -2.0], [0.0, 1.0]]),
           "abstain option": np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])}


def joint(n, seed=0):
    rng = np.random.default_rng(seed); X = B.cells(n + 1); p = np.zeros(len(X))
    flips = {5: 0.35, 6: 0.40, 8: 0.25}; extra = {}
    for i in range(9, n + 1):
        extra[i] = ("copy", 5 + (i % 2), 0.15) if i % 3 == 0 else ("direct", None, float(rng.uniform(0.3, 0.45)))
    for k, x in enumerate(X):
        th = x[0]; v = x[1:]
        pr = 0.5 * 0.125                                               # theta, X1..X3
        par = th ^ v[0] ^ v[1] ^ v[2]; pr *= 0.9 if v[3] == par else 0.1
        for i, e in flips.items():
            if i <= n:
                pr *= (1 - e) if v[i - 1] == th else e
        if n >= 7:
            pr *= 0.9 if v[6] == v[4] else 0.1
        for i, (kind, src, e) in extra.items():
            ref = th if kind == "direct" else v[src - 1]
            pr *= (1 - e) if v[i - 1] == ref else e
        p[k] = pr
    cost = np.ones(n); cost[6] = 0.5; cost[7] = 3.0
    return p / p.sum(), cost


def part_a(n):
    p, cost = joint(n); regs = [U for r in range(n + 1) for U in itertools.combinations(range(n), r)]
    t0 = time.time(); M = [B.standard_measure(p, U, n) for U in regs]; c = np.array([cost[list(U)].sum() for U in regs])
    mi = [B.mutual_information(m, 0.5) for m in M]
    F, checks = B.frontier(M, c, mi); tF = time.time() - t0
    Fs = set(F); tests = ok = 0
    for name, R in REWARDS.items():
        vals = np.array([B.value(m, R) for m in M])
        for price in np.linspace(0, 0.2, 21):
            best = int(np.argmax(vals - price * c)); tests += 1; ok += best in Fs or any(abs((vals - price * c)[f] - (vals - price * c)[best]) < 1e-12 for f in F)
        for budget in np.arange(0, cost.sum() + 0.5, 0.5):
            feas = c <= budget + 1e-12; best = int(np.argmax(np.where(feas, vals, -np.inf))); tests += 1
            ok += best in Fs or any(feas[f] and abs(vals[f] - vals[best]) < 1e-12 for f in F)
    print(f"(A) n={n}: {len(regs)} regions, frontier {len(F)} ({100 * len(F) / len(regs):.1f}%), {checks} dominance checks"
          f" (vs {len(regs) * (len(regs) - 1)} pairwise), {tF:.1f}s; optimum on the frontier in {ok}/{tests} (reward, budget/price) cases")
    return dict(n=n, regions=len(regs), frontier=len(F), checks=checks, seconds=tF, ok=ok, tests=tests,
                frontier_points=[(float(c[i]), float(mi[i])) for i in F], all_points=[(float(ci), float(m)) for ci, m in zip(c, mi)])


def part_b(n=8, budget=4.0):
    p, cost = joint(n); acc = REWARDS["accuracy"]
    meas = lambda U: B.standard_measure(p, U, n); val = lambda U: B.value(meas(U), acc); mi = lambda U: B.mutual_information(meas(U), 0.5)
    def greedy(score):
        U = []; path = [0.5]
        while True:
            cand = [i for i in range(n) if i not in U and cost[U + [i]].sum() <= budget + 1e-12]
            if not cand:
                return U, path
            i = max(cand, key=lambda i: score(U, i)); U = U + [i]; path.append(val(U))
    out = {}
    out["greedy-MI"] = greedy(lambda U, i: mi([i]))
    out["greedy-value"] = greedy(lambda U, i: val(U + [i]) + 1e-9 * mi([i]))
    regs = [list(U) for r in range(n + 1) for U in itertools.combinations(range(n), r) if cost[list(U)].sum() <= budget + 1e-12]
    pw = lambda U: sum(mi([i, j]) for i, j in itertools.combinations(U, 2)) + (mi(U) if len(U) == 1 else 0)
    Upw = max(regs, key=pw); out["pairwise-MI"] = (Upw, [val(Upw[:k]) for k in range(len(Upw) + 1)])
    M = [meas(U) for U in regs]; c = np.array([cost[U].sum() for U in regs]); F, _ = B.frontier(M, c, [B.mutual_information(m, 0.5) for m in M])
    Ubest = max((regs[i] for i in F), key=val); out["frontier"] = (Ubest, [val(Ubest[:k]) for k in range(len(Ubest) + 1)])
    print(f"(B) budget {budget}:")
    for k, (U, path) in out.items():
        print(f"   {k:12s} region {['X%d' % (i + 1) for i in U]}  accuracy {val(U):.3f}  value along the growth path {np.round(path, 3).tolist()}")
    return {k: dict(region=[int(i) for i in U], value=val(U), path=[float(x) for x in path]) for k, (U, path) in out.items()}


def part_c():
    ctx = [(0, 1), (1, 2), (2, 0)]                                     # variables: 0 = Theta, 1 = X1, 2 = X2
    def fam(lam):
        eq = np.array([0.5, 0, 0, 0.5]); ne = np.array([0, 0.5, 0.5, 0]); u = np.full(4, 0.25)
        return [lam * eq + (1 - lam) * u, lam * eq + (1 - lam) * u, lam * ne + (1 - lam) * u]
    acc = REWARDS["accuracy"]; rows = []
    for lam in np.linspace(0, 1, 11):
        m = fam(lam); q = B.ipf_glue(3, ctx, m)
        naive = B.value(B.standard_measure(q, (0, 1), 2), acc) if q is not None else float("nan")
        rv = B.robust_value(3, ctx, m, 0, (1, 2), acc)
        rows.append(dict(lam=float(lam), naive=naive, lo=None if rv is None else rv[0], hi=None if rv is None else rv[1]))
        print(f"(C) lam {lam:.1f}: naive glued accuracy {naive:.3f}; " + ("NO consistent joint (contextual): region has no value" if rv is None else f"honest interval [{rv[0]:.3f}, {rv[1]:.3f}]"))
    return rows


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, green, grey, red = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#b8b6ae", "#d6402a"
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7)
    A = OUT["A"][-1]; P = np.array(A["all_points"]); Fp = np.array(A["frontier_points"])
    ax[0].scatter(P[:, 0] + np.random.default_rng(0).uniform(-0.12, 0.12, len(P)), P[:, 1], s=4, color=grey, label=f"all {A['regions']} regions")
    ax[0].scatter(Fp[:, 0], Fp[:, 1], s=16, color=blue, label=f"Blackwell frontier ({A['frontier']})")
    ax[0].set_xlabel("cost of the region", fontsize=7.5, color=muted); ax[0].set_ylabel("mutual information with Θ (nats)", fontsize=7.5, color=muted)
    ax[0].legend(fontsize=6.5, frameon=False); ax[0].set_title(f"(a) reward-free pruning (n = {A['n']}): every optimum\nlies on the frontier ({A['ok']}/{A['tests']} cases)", fontsize=8, loc="left")
    Bp = OUT["B"]; col = {"greedy-MI": orange, "greedy-value": green, "pairwise-MI": grey, "frontier": blue}
    for k, d in Bp.items():
        ax[1].plot(range(len(d["path"])), d["path"], "-o", ms=4, color=col[k], label=f"{k}: {d['value']:.2f}")
    ax[1].set_xlabel("gluing step (measurements added)", fontsize=7.5, color=muted); ax[1].set_ylabel("accuracy of the region", fontsize=7.5, color=muted)
    ax[1].legend(fontsize=6.5, frameon=False); ax[1].set_title("(b) the parity path is flat until its last step:\ninvisible to pairwise and myopic grading", fontsize=8, loc="left")
    C = OUT["C"]; lam = [r["lam"] for r in C]
    ax[2].plot(lam, [r["naive"] for r in C], "-o", ms=4, color=red, label="naive max-ent gluing")
    lo = [r["lo"] if r["lo"] is not None else np.nan for r in C]; hi = [r["hi"] if r["hi"] is not None else np.nan for r in C]
    ax[2].fill_between(lam, lo, hi, color=blue, alpha=0.25, label="honest interval over gluings")
    ax[2].plot(lam, lo, color=blue, lw=1.5)
    bad = [l for l, r in zip(lam, C) if r["lo"] is None]
    if bad:
        ax[2].axvspan(min(bad) - 0.05, 1.0, color=grey, alpha=0.2, label="contextual: no value")
    ax[2].set_xlabel("strength of the contradictory cycle λ", fontsize=7.5, color=muted); ax[2].set_ylabel("accuracy for Θ from (X1, X2)", fontsize=7.5, color=muted)
    ax[2].legend(fontsize=6.5, frameon=False); ax[2].set_title("(c) across an obstruction the value is an\ninterval, then undefined", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(here, "blackwell_chart.pdf")); fig.savefig(os.path.join(here, "blackwell_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    OUT = dict(A=[part_a(n) for n in (8, 10, 12)], B=part_b(), C=part_c())
    json.dump(json.loads(json.dumps(OUT, default=float)), open(os.path.join(here, "results.json"), "w")); plot(OUT)
