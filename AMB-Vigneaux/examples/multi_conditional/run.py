"""Several conditionals at once: which combinations of what-if conditions are jointly possible, and what they imply.

Proposal under test: build the prediction space by an adjunction, prune by local restrictions, and mask contradictory
combinations of conditions by the cup product of their cocycles (compatible iff the product class vanishes).

(0) Cup products on our covers: Betti numbers of the Cech nerve. Degree-1 classes multiply into H^2; if b_2 = 0 the
    mask is identically 'compatible'.
(1) AllSides triangle (pair tables from pair-only stories). Two conditions on the never-observed triple:
      s1 = P(R adv | L, C adv),   s2 = P(L adv | C, R adv).
    Each alone: its feasible range. Jointly: for each s1, the range of s2 over all global laws satisfying s1
    (grey_bounds with s2 as a conditional target). Count of independent conditions = rank on ker R (condition_rank).
    Then the outcome P(majority adversarial) along the jointly feasible set.
(2) MBIC, two grey filters on annotators: A = age >= 55, F = female. Data: {I, Y} from one half of the annotators,
    the population table {I, A, F} from the other (I = ideology L/C/R, Y = 'biased'). Conditions s1 = P(Y | A),
    s2 = P(Y | F). Joint feasible region in (s1, s2); for points in it, bounds on P(Y | A, F) (older women) and
    on P(Y | not A, not F). Hidden truth for comparison.
(3) Kleisli (monad) composition of conditionals: the chain P(Y | A) = sum_x P(Y | x) P(x | A) is the Kleisli
    composite of the kernels A -> X -> Y. It is exact only under conditional independence (the max-ent gluing);
    the bounds of (2) are the range over all composites consistent with the data.
Env: MBIC_XLSX, ALLSIDES_DIR.
"""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../grey"), os.path.join(here, "../bias3"), os.path.join(here, "../allsides")]
import numpy as np, pandas as pd
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.bounds import grey_bounds, event, hypothesis, functional, condition_rank, loop_space, nerve_betti
import run as G                      # examples/grey/run.py helpers

rng = np.random.default_rng(0)


def triangle_model(parts):
    sc = Scenario({"L": (0, 1), "C": (0, 1), "R": (0, 1)}, (("L", "C"), ("C", "R"), ("L", "R")))
    e = EmpiricalModel(sc, {C: (G.tab2(parts[C], *C) + 0.5) / (G.tab2(parts[C], *C) + 0.5).sum() for C in sc.contexts})
    return sc, e


def part1(parts, T3):
    sc, e = triangle_model(parts)
    c1 = (lambda g: g["R"] == 1, lambda g: g["L"] == 1 and g["C"] == 1)
    c2 = (lambda g: g["L"] == 1, lambda g: g["C"] == 1 and g["R"] == 1)
    alone = [grey_bounds(e, event(sc, lambda g, E=E, Gv=Gv: E(g) and Gv(g)), given=event(sc, Gv)) for E, Gv in (c1, c2)]
    maj = functional(sc, lambda g: float(g["L"] + g["C"] + g["R"] >= 2)); rows = []
    for s1 in np.linspace(alone[0]["lo"], alone[0]["hi"], 41):
        h = hypothesis(sc, c1[0], c1[1], s1, s1)
        b2 = grey_bounds(e, event(sc, lambda g: c2[0](g) and c2[1](g)), given=event(sc, c2[1]), rows=h)
        bm = grey_bounds(e, maj, rows=h)
        rows.append(dict(s1=float(s1), s2_lo=b2["lo"], s2_hi=b2["hi"], maj_lo=bm["lo"], maj_hi=bm["hi"]))
    mid = rows[len(rows) // 2]
    r = condition_rank(sc, [c1, c2], [mid["s1"], mid["s2_lo"]])
    # a pair that each condition allows alone but that is jointly impossible
    s1x, s2x = alone[0]["lo"] + 0.25 * (alone[0]["hi"] - alone[0]["lo"]), alone[1]["lo"] + 0.75 * (alone[1]["hi"] - alone[1]["lo"])
    both = grey_bounds(e, maj, rows=hypothesis(sc, *c1, s1x, s1x) + hypothesis(sc, *c2, s2x, s2x))
    obs = dict(s1=float(T3[(T3[:, 0] == 1) & (T3[:, 1] == 1), 2].mean()), s2=float(T3[(T3[:, 1] == 1) & (T3[:, 2] == 1), 0].mean()))
    print(f"(1) AllSides triangle: dim ker R = {loop_space(sc).shape[1]}; nerve Betti {nerve_betti(sc.contexts)}")
    print(f"    alone: s1 in [{alone[0]['lo']:.3f}, {alone[0]['hi']:.3f}], s2 in [{alone[1]['lo']:.3f}, {alone[1]['hi']:.3f}];"
          f" independent conditions among (s1, s2): {r}")
    print(f"    jointly: s2 is a function of s1 (max width of s2-range {max(x['s2_hi'] - x['s2_lo'] for x in rows):.1e});"
          f" e.g. s1 {rows[10]['s1']:.3f} -> s2 {rows[10]['s2_lo']:.3f};  s1 {rows[30]['s1']:.3f} -> s2 {rows[30]['s2_lo']:.3f}")
    print(f"    the pair (s1, s2) = ({s1x:.3f}, {s2x:.3f}) is allowed by each condition alone but jointly "
          f"{'feasible' if both['feasible'] else 'ruled out, L1 slack %.4f' % both['slack']}; cup-product mask says: compatible (H^2 = 0)")
    print(f"    triple-covered stories: s1 {obs['s1']:.3f}, s2 {obs['s2']:.3f}")
    return dict(alone=[[a["lo"], a["hi"]] for a in alone], rows=rows, rank=r, counterexample=dict(s1=s1x, s2=s2x, feasible=both["feasible"], slack=both["slack"]),
                observed=obs)


def part2(d, n_grid=31):
    ann = d["mturk_id"].unique(); a = rng.permutation(ann); half = set(a[: len(a) // 2])
    h1 = d[d["mturk_id"].isin(half)]; h2 = d[~d["mturk_id"].isin(half)]
    levels = ("C", "L", "R")
    sc = Scenario({"X": levels, "A": (0, 1), "F": (0, 1), "Y": (0, 1)}, (("X", "Y"), ("X", "A", "F")))
    def tab_xy(h):
        t = np.zeros(6)
        for x, y in zip(h["I"], h["Y"]):
            t[2 * levels.index(x) + int(y)] += 1
        return (t + .01) / (t + .01).sum()
    def tab_xaf(h):
        t = np.zeros(12); A = (h["age"] >= 55).astype(int); F = h["fem"]
        for x, aa, ff in zip(h["I"], A, F):
            t[4 * levels.index(x) + 2 * aa + ff] += 1
        return (t + .01) / (t + .01).sum()
    e = EmpiricalModel(sc, {("X", "Y"): tab_xy(h1), ("X", "A", "F"): tab_xaf(h2)})
    cA = (lambda g: g["Y"] == 1, lambda g: g["A"] == 1); cF = (lambda g: g["Y"] == 1, lambda g: g["F"] == 1)
    YA = event(sc, lambda g: g["Y"] == 1 and g["A"] == 1); YF = event(sc, lambda g: g["Y"] == 1 and g["F"] == 1)
    aloneA = grey_bounds(e, YA, given=event(sc, cA[1])); aloneF = grey_bounds(e, YF, given=event(sc, cF[1]))
    YAF = event(sc, lambda g: g["Y"] == 1 and g["A"] == 1 and g["F"] == 1); AF = event(sc, lambda g: g["A"] == 1 and g["F"] == 1)
    Ynn = event(sc, lambda g: g["Y"] == 1 and g["A"] == 0 and g["F"] == 0); nn = event(sc, lambda g: g["A"] == 0 and g["F"] == 0)
    region = []
    for s1 in np.linspace(0, 1, n_grid):
        h = hypothesis(sc, *cA, s1, s1)
        b = grey_bounds(e, YF, given=event(sc, cF[1]), rows=h)
        region.append(dict(s1=float(s1), s2_lo=b["lo"], s2_hi=b["hi"], feasible=b["feasible"]))
    A1 = (h1["age"] >= 55); F1 = h1["fem"] == 1
    truth = dict(s1=float(h1.loc[A1, "Y"].mean()), s2=float(h1.loc[F1, "Y"].mean()), af=float(h1.loc[A1 & F1, "Y"].mean()),
                 nn=float(h1.loc[~A1 & ~F1, "Y"].mean()))
    grid = []
    for s1 in np.linspace(0, 1, 21):
        for s2 in np.linspace(0, 1, 21):
            h = hypothesis(sc, *cA, s1, s1) + hypothesis(sc, *cF, s2, s2)
            b = grey_bounds(e, YAF, given=AF, rows=h); bn = grey_bounds(e, Ynn, given=nn, rows=h)
            grid.append(dict(s1=float(s1), s2=float(s2), feasible=b["feasible"], af_lo=b["lo"], af_hi=b["hi"], nn_lo=bn["lo"], nn_hi=bn["hi"]))
    h = hypothesis(sc, *cA, truth["s1"], truth["s1"]) + hypothesis(sc, *cF, truth["s2"], truth["s2"])
    at_truth = dict(af=grey_bounds(e, YAF, given=AF, rows=h), nn=grey_bounds(e, Ynn, given=nn, rows=h))
    free = dict(af=grey_bounds(e, YAF, given=AF), nn=grey_bounds(e, Ynn, given=nn))
    r = condition_rank(sc, [cA, cF], [truth["s1"], truth["s2"]])
    feas = [g for g in grid if g["feasible"]]
    print(f"\n(2) MBIC, filters 55+ (A) and female (F): dim ker R = {loop_space(sc).shape[1]}, nerve Betti {nerve_betti(sc.contexts)};"
          f" independent conditions among (s1, s2): {r}")
    print(f"    alone: P(Y|A) in [{aloneA['lo']:.3f}, {aloneA['hi']:.3f}], P(Y|F) in [{aloneF['lo']:.3f}, {aloneF['hi']:.3f}];"
          f" jointly feasible share of the 21x21 grid {len(feas) / len(grid):.2f}")
    print(f"    P(Y | A, F): no conditions [{free['af']['lo']:.3f}, {free['af']['hi']:.3f}];  at the true (s1, s2) = ({truth['s1']:.3f}, {truth['s2']:.3f}):"
          f" [{at_truth['af']['lo']:.3f}, {at_truth['af']['hi']:.3f}];  hidden truth {truth['af']:.3f}")
    print(f"    P(Y | not A, not F): no conditions [{free['nn']['lo']:.3f}, {free['nn']['hi']:.3f}];  at the true (s1, s2):"
          f" [{at_truth['nn']['lo']:.3f}, {at_truth['nn']['hi']:.3f}];  hidden truth {truth['nn']:.3f}")
    # (3) Kleisli composite (CIA) for P(Y | A, F): sum_x P(Y|x) P(x | A, F)
    py = h1.groupby("I")["Y"].mean(); A2 = h2["age"] >= 55; F2 = h2["fem"] == 1
    px = h2.loc[A2 & F2, "I"].value_counts(normalize=True); kl = float(sum(px[x] * py[x] for x in px.index))
    print(f"(3) Kleisli composite A,F -> X -> Y (conditional independence): P(Y | A, F) = {kl:.3f}; inside the no-condition bounds:"
          f" {free['af']['lo'] - 1e-9 <= kl <= free['af']['hi'] + 1e-9}")
    return dict(alone=dict(A=[aloneA["lo"], aloneA["hi"]], F=[aloneF["lo"], aloneF["hi"]]), region=region, grid=grid, truth=truth, rank=r,
                at_truth={k: [v["lo"], v["hi"]] for k, v in at_truth.items()}, free={k: [v["lo"], v["hi"]] for k, v in free.items()}, kleisli=kl)


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#b8b6ae"
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7.5)
    P1 = OUT["allsides"]; (a1, b1), (a2, b2) = P1["alone"]
    ax[0].add_patch(plt.Rectangle((a1, a2), b1 - a1, b2 - a2, color=grey, alpha=0.3, lw=0, label="allowed by each condition alone"))
    ax[0].plot([r["s1"] for r in P1["rows"]], [r["s2_lo"] for r in P1["rows"]], color=blue, lw=2, label="jointly possible")
    ax[0].scatter([P1["counterexample"]["s1"]], [P1["counterexample"]["s2"]], color=orange, marker="x", s=30, zorder=4, label="each alone OK, jointly ruled out")
    ax[0].scatter([P1["observed"]["s1"]], [P1["observed"]["s2"]], color=ink, s=14, zorder=4, label="triple-covered stories")
    ax[0].set_xlabel("s1 = P(R adv | L, C adv)", fontsize=8, color=muted); ax[0].set_ylabel("s2 = P(L adv | C, R adv)", fontsize=8, color=muted)
    ax[0].set_title("(a) AllSides: two conditions, one free direction", fontsize=8.5, loc="left", color=ink); ax[0].legend(fontsize=6.3, frameon=False, loc="upper left")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    P2 = OUT["mbic"]; R = [r for r in P2["region"] if r["feasible"]]
    ax[1].fill_between([r["s1"] for r in R], [r["s2_lo"] for r in R], [r["s2_hi"] for r in R], color=blue, alpha=0.3, lw=0, label="jointly possible")
    ax[1].scatter([P2["truth"]["s1"]], [P2["truth"]["s2"]], color=ink, s=14, zorder=4, label="hidden truth")
    ax[1].set_xlabel("s1 = P(biased | 55+)", fontsize=8, color=muted); ax[1].set_ylabel("s2 = P(biased | female)", fontsize=8, color=muted)
    ax[1].set_title("(b) MBIC: two filters, a 2-D region", fontsize=8.5, loc="left", color=ink); ax[1].legend(fontsize=6.3, frameon=False, loc="upper left")
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
    g = [x for x in P2["grid"] if x["feasible"]]
    sc_ = ax[2].scatter([x["s1"] for x in g], [x["s2"] for x in g], c=[x["af_hi"] - x["af_lo"] for x in g], cmap="Blues_r", s=22, marker="s", vmin=0, vmax=1)
    cb = fig.colorbar(sc_, ax=ax[2], fraction=0.046); cb.ax.tick_params(labelsize=7); cb.set_label("width of bounds on P(biased | 55+, female)", fontsize=7, color=muted)
    ax[2].scatter([P2["truth"]["s1"]], [P2["truth"]["s2"]], color=orange, s=18, zorder=4)
    ax[2].set_xlabel("s1", fontsize=8, color=muted); ax[2].set_ylabel("s2", fontsize=8, color=muted)
    ax[2].set_title("(c) what the two conditions pin down", fontsize=8.5, loc="left", color=ink); ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(here, "multi_conditional.pdf")); fig.savefig(os.path.join(here, "multi_conditional.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    print("(0) cup products: nerve Betti numbers  triangle", nerve_betti([("L", "C"), ("C", "R"), ("L", "R")]),
          " CHSH", nerve_betti([("a0", "b0"), ("a0", "b1"), ("a1", "b0"), ("a1", "b1")]),
          " star {X,Y},{X,A,F}", nerve_betti([("X", "Y"), ("X", "A", "F")]), " -> no H^2: every cup product of degree-1 classes is 0\n")
    parts, T3 = G.allsides_parts(); d = G.mbic_frame()
    OUT = dict(allsides=part1(parts, T3), mbic=part2(d))
    json.dump(OUT, open(os.path.join(here, "results.json"), "w"), indent=1, default=float)
    plot(OUT)
