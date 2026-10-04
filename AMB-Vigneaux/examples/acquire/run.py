"""Identification-aware acquisition: cost to a correct decision, against the strongest existing policy.

Covers and costs:
  triangle   n = 3; pairs AB, BC, CA cost 1; full ABC cost 4.
  4-cycle    n = 4; cycle pairs AB, BC, CD, DA cost 1; diagonals AC, BD cost 2; triples cost 3; full ABCD cost 8.
Targets f (decide <f, p> >= t at delta = 0.05):
  triangle  unanimity P(A = B = C)   (identified by the pairs)       all-ones P(A = B = C = 1)   (not identified)
  4-cycle   P(A = C)                 (identified by the diagonal AC) all-ones P(ABCD = 1111)       (not identified)
Thresholds, relative to the identified set [L, H] of the cheap cover at the truth:
  outside   t = L - 0.03 or H + 0.03: the cheap data can decide
  inside    t = truth +- 0.03 inside [L, H]: only identifying data can decide
Truths p ~ Dirichlet(2), 30 per scenario; cost cap 60000; every policy uses the same anytime-valid robust-bound stop.
"""
import os, sys, json, time
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
from amb_vigneaux import acquire as A

rng = np.random.default_rng(31)
POL = ("bounds-first", "c-opt", "cheap-only", "full-only")


def covers():
    X3 = A.cells(3); X4 = A.cells(4)
    tri = dict(n=3, contexts=[(0, 1), (1, 2), (0, 2), (0, 1, 2)], costs=[1, 1, 1, 4], cheap=[0, 1, 2],
               targets={"unanimity (identified)": (X3.min(1) == X3.max(1)).astype(float), "all-ones (not identified)": X3.prod(1).astype(float)})
    cyc = dict(n=4, contexts=[(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3), (0, 1, 2), (1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2, 3)],
               costs=[1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 8], cheap=[0, 1, 2, 3],
               targets={"P(A = C) (diagonal identifies)": (X4[:, 0] == X4[:, 2]).astype(float), "all-ones (not identified)": X4.prod(1).astype(float)})
    return {"triangle": tri, "4-cycle": cyc}


def ident_set(cv, f, p):
    Rs = [A.restriction(cv["n"], cv["contexts"][i]) for i in cv["cheap"]]
    return A.robust_bounds(cv["n"], Rs, [R @ p for R in Rs], [0.0] * len(Rs), f)


def scenario(cv, f, regime, reps=30, gap=0.03):
    rows = {p: [] for p in POL}; made = 0; tries = 0
    while made < reps and tries < 2000:
        tries += 1; p = rng.dirichlet(np.full(2 ** cv["n"], 2.0)); truth = float(f @ p); L, H = ident_set(cv, f, p)
        if regime == "outside":
            t = L - gap if rng.random() < 0.5 else H + gap
            if t <= 0.0 or t >= 1.0:
                continue
        else:
            if H - L < 0.08 and H - L > 1e-9:
                continue
            t = truth + (0.03 if rng.random() < 0.5 else -0.03)
            if H - L > 1e-9 and not (L + 0.01 < t < H - 0.01):
                continue
        made += 1
        prob = A.Problem(cv["n"], cv["contexts"], cv["costs"], cv["cheap"], f, t)
        for pol in POL:
            r = A.run(prob, p, pol, rng=np.random.default_rng(rng.integers(1 << 31)))
            r["correct"] = (r["decision"] == (truth >= t)) if r["stopped"] else None
            rows[pol].append(r)
    out = {}
    for pol, R in rows.items():
        cost = np.array([r["cost"] for r in R]); st = np.array([r["stopped"] for r in R])
        out[pol] = dict(stop=float(st.mean()), error=float(np.mean([not r["correct"] for r in R if r["stopped"]])) if st.any() else np.nan,
                        cost_median=float(np.median(cost)), cost_mean=float(cost.mean()), costs=cost.tolist(),
                        switched=float(np.mean([r["switched_at"] is not None for r in R])))
    base = np.array(out["c-opt"]["costs"])
    for pol in POL:
        out[pol]["ratio_to_copt_median"] = float(np.median(np.array(out[pol]["costs"]) / base))
    return out


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, green, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#b8b6ae"
    col = {"bounds-first": blue, "c-opt": orange, "cheap-only": green, "full-only": grey}
    keys = [k for k in OUT if not k.startswith("_")]; fig, ax = plt.subplots(1, 3, figsize=(16, 3.9), gridspec_kw=dict(width_ratios=[1.3, 1.3, 0.8]))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7)
    x = np.arange(len(keys)); w = 0.2
    for j, pol in enumerate(POL):
        ax[0].bar(x + (j - 1.5) * w, [OUT[k][pol]["cost_median"] for k in keys], w, color=col[pol], label=pol)
        ax[1].bar(x + (j - 1.5) * w, [OUT[k][pol]["stop"] for k in keys], w, color=col[pol], label=pol)
    for a in ax:
        short = lambda k: k.replace("triangle", "tri").replace("4-cycle", "cyc").replace(" (identified)", " (id)").replace(" (not identified)", " (not id)").replace(" (diagonal identifies)", " (diag id)").replace(" | ", "\n")
        a.set_xticks(x); a.set_xticklabels([short(k) for k in keys], fontsize=6.2)
    ax[0].set_yscale("log"); ax[0].set_ylabel("median cost to a decision (cap 60000)", fontsize=7.5, color=muted); ax[0].legend(fontsize=6.5, frameon=False)
    ax[0].set_title("(a) cost: lower is better", fontsize=8.5, loc="left")
    ax[1].set_ylabel("share of runs that decide within the cap", fontsize=7.5, color=muted); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("(b) stopping (every decision made was correct)", fontsize=8.5, loc="left")
    X = OUT.get("_crossover")
    if X:
        costs = (4, 16, 64); gaps = (0.03, 0.1, 0.2)
        best_other = lambda d: min(d[k]["median"] for k in ("c-opt", "cheap-only", "full-only") if np.isfinite(d[k]["median"]))
        M = np.array([[X[f"full cost {c}, gap {g}"]["bounds-first"]["median"] / best_other(X[f"full cost {c}, gap {g}"]) for g in gaps] for c in costs])
        im = ax[2].imshow(np.log2(M), cmap="RdBu_r", vmin=-2, vmax=2, origin="lower")
        for i in range(3):
            for j in range(3):
                ax[2].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=ink)
        ax[2].set_xticks(range(3)); ax[2].set_xticklabels([str(g) for g in gaps], fontsize=7); ax[2].set_yticks(range(3)); ax[2].set_yticklabels([str(c) for c in costs], fontsize=7)
        ax[2].set_xlabel("distance of t from the cheap identified set", fontsize=7.5, color=muted); ax[2].set_ylabel("cost of one full-joint sample (pairs cost 1)", fontsize=7.5, color=muted)
        W = OUT.get("_minimax", {})
        ax[2].set_title("(c) cost of bounds-first / best other policy\n(triangle, all-ones). Worst case over the grid:\n" +
                        ", ".join(f"{k} {v:.1f}x" for k, v in W.items()), fontsize=7.5, loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(here, "acquire_chart.pdf")); fig.savefig(os.path.join(here, "acquire_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    OUT = {}; reps = int(os.environ.get("REPS", "30"))
    for cname, cv in covers().items():
        for tname, f in cv["targets"].items():
            regimes = ("inside",) if "identifies" in tname or "(identified)" in tname else ("outside", "inside")
            for reg in regimes:
                t0 = time.time(); key = f"{cname} | {tname} | t {reg}"; OUT[key] = scenario(cv, f, reg, reps)
                print(f"{key}  ({time.time() - t0:.0f}s)")
                for pol in POL:
                    o = OUT[key][pol]
                    print(f"   {pol:12s} stop {o['stop']:.2f}  error {o['error']:.2f}  median cost {o['cost_median']:7.0f}  mean {o['cost_mean']:7.0f}"
                          f"  median ratio to c-opt {o['ratio_to_copt_median']:.2f}" + (f"  switched {o['switched']:.2f}" if pol == "bounds-first" else ""), flush=True)
    OUT = json.loads(json.dumps(OUT, default=float)); json.dump(OUT, open(os.path.join(here, "results.json"), "w")); plot(OUT)


def crossover(reps=8, cap=200000):
    """where can cheap partial identification beat optimal design? triangle, all-ones, t outside the identified set"""
    X3 = A.cells(3); f = X3.prod(1).astype(float); out = {}
    for c_full in (4, 16, 64):
        for gap in (0.03, 0.1, 0.2):
            res = {p: [] for p in ("bounds-first", "c-opt", "cheap-only", "full-only")}; made = 0
            while made < reps:
                p = rng.dirichlet(np.full(8, 2.0)); Rs = [A.restriction(3, c) for c in ((0, 1), (1, 2), (0, 2))]
                L, H = A.robust_bounds(3, Rs, [R @ p for R in Rs], [0.0] * 3, f)
                t = L - gap if rng.random() < 0.5 else H + gap
                if not (0 < t < 1):
                    continue
                made += 1; prob = A.Problem(3, [(0, 1), (1, 2), (0, 2), (0, 1, 2)], [1, 1, 1, c_full], [0, 1, 2], f, t)
                for pol in res:
                    r = A.run(prob, p, pol, cap=cap, rng=np.random.default_rng(rng.integers(1 << 31))); res[pol].append(r["cost"] if r["stopped"] else np.nan)
            key = f"full cost {c_full}, gap {gap}"
            out[key] = {k: dict(median=float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan, stop=float(np.mean(np.isfinite(v)))) for k, v in
                        ((k, np.array(v, float)) for k, v in res.items())}
            print(f"  crossover {key:24s} " + "  ".join(f"{k} {v['median']:8.0f} (stop {v['stop']:.2f})" for k, v in out[key].items()), flush=True)
    return out


if __name__ == "__main__" and os.environ.get("CROSSOVER"):
    X = crossover(); R = json.load(open(os.path.join(here, "results.json"))) if os.path.exists(os.path.join(here, "results.json")) else {}
    R["_crossover"] = json.loads(json.dumps(X, default=float)); json.dump(R, open(os.path.join(here, "results.json"), "w"))
