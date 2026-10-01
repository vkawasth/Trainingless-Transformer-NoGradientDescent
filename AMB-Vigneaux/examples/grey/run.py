"""Grey (what-if) contexts: stitching a filter that is never observed together with the outcome.

(1) MBIC validation. Units are annotations: Y = 'biased' label, A = an annotator filter (age >= 55, age >= 45, female,
    graduate work, no bachelor's degree), X = shared covariates seen in both sources (ideology group L/C/R; then
    ideology x gender; then ideology x gender x bachelor's+; the filter's own variable is never in X).
    The joint (A, Y) is HIDDEN. The engine sees two tables from DIFFERENT annotators (a random half each, as if the
    filter's distribution came from a census): {X, Y} from half 1 and {X, A} from half 2. It bounds P(Y=1 | A=1) over
    all global laws (grey_bounds, Charnes--Cooper), and gives the conditional-independence (max-ent) point
    sum_x P(Y=1 | x) P(x | A=1). Truth: P(Y=1 | A=1) in half 1. 50 random annotator splits.
(2) MBIC what-if: hypothesis P(Y=1 | age >= 55) = s, swept; the data rule out s outside the bounds (L1 slack > 0);
    inside, bounds on P(Y=1 | age >= 55, right-leaning).
(3) AllSides what-if on the triangle (pair tables from stories covered by exactly that pair): hypothesis
    P(R adversarial | L and C adversarial) = s, swept; bounds on P(majority of sides adversarial), and the slack.
(4) Population test for the bounds (paper Section bounds): for each pair, the pair's table from pair-only stories vs
    the same pair's table from triple-covered stories: TV distance and chi^2 test.
Env: MBIC_XLSX, ALLSIDES_DIR.
"""
import os, sys, json, collections
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../bias3"), os.path.join(here, "../allsides"), os.path.join(here, "../bounds")]
import numpy as np, pandas as pd
from scipy import stats
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.bounds import grey_bounds, event, hypothesis, functional

rng = np.random.default_rng(0)


def mbic_frame():
    import mbic
    d = mbic.load()
    d["I"] = pd.cut(d["political_ideology"], [-11, -4, 3, 11], labels=["L", "C", "R"]).astype(str)
    d["fem"] = (d["gender"] == "Female").astype(int)
    d["ba"] = d["education"].isin(["Bachelor’s degree", "Graduate work"]).astype(int)
    d["Y"] = d["b"].astype(int)
    return d


FILTERS = {"age >= 55": lambda d: (d["age"] >= 55).astype(int), "age >= 45": lambda d: (d["age"] >= 45).astype(int),
           "female": lambda d: d["fem"], "graduate work": lambda d: (d["education"] == "Graduate work").astype(int),
           "no bachelor's": lambda d: 1 - d["ba"]}
STRATA = {"ideology": ["I"], "ideology x gender": ["I", "fem"], "ideology x gender x BA+": ["I", "fem", "ba"]}


def scenario_tables(h1, h2, xcols, A1, A2):
    x1 = h1[xcols].astype(str).agg("|".join, axis=1); x2 = h2[xcols].astype(str).agg("|".join, axis=1)
    levels = tuple(sorted(set(x1) | set(x2)))
    sc = Scenario({"X": levels, "A": (0, 1), "Y": (0, 1)}, (("X", "Y"), ("X", "A")))
    def tab(x, v):
        t = np.zeros(2 * len(levels))
        for xi, vi in zip(x, v):
            t[2 * levels.index(xi) + int(vi)] += 1
        return (t + 0.01) / (t + 0.01).sum()
    return sc, EmpiricalModel(sc, {("X", "Y"): tab(x1, h1["Y"]), ("X", "A"): tab(x2, A2)}), x1, x2


def cia_point(h1, h2, x1, x2, A2):
    py = h1.groupby(x1.values)["Y"].mean()
    w = pd.Series(np.asarray(A2)).groupby(x2.values).sum(); w = w / w.sum()
    return float(sum(w[k] * py.get(k, h1["Y"].mean()) for k in w.index))


def validation(d, n_rep=50):
    ann = d["mturk_id"].unique(); res = collections.defaultdict(list)
    for _ in range(n_rep):
        a = rng.permutation(ann); half = set(a[: len(a) // 2])
        h1 = d[d["mturk_id"].isin(half)]; h2 = d[~d["mturk_id"].isin(half)]
        for fname, fa in FILTERS.items():
            A1, A2 = fa(h1), fa(h2); truth = float(h1.loc[A1 == 1, "Y"].mean())
            for sname, xcols in STRATA.items():
                xc = [c for c in xcols if not (fname == "female" and c == "fem") and not (fname in ("graduate work", "no bachelor's") and c == "ba")]
                if xc != xcols and sname != "ideology":
                    continue
                sc, e, x1, x2 = scenario_tables(h1, h2, xc, A1, A2)
                b = grey_bounds(e, event(sc, lambda g: g["Y"] == 1 and g["A"] == 1), given=event(sc, lambda g: g["A"] == 1))
                res[(fname, sname)].append((b["lo"], b["hi"], cia_point(h1, h2, x1, x2, A2), truth))
    out = []
    print("(1) MBIC: bound P(biased | filter) from {X,Y} (half 1) and {X,filter} (half 2); truth hidden in half 1; 50 splits")
    for (f, s), v in res.items():
        v = np.array(v); inside = np.mean((v[:, 0] - 1e-9 <= v[:, 3]) & (v[:, 3] <= v[:, 1] + 1e-9))
        r = dict(filter=f, strata=s, lo=float(np.median(v[:, 0])), hi=float(np.median(v[:, 1])), width=float(np.median(v[:, 1] - v[:, 0])),
                 cia=float(np.median(v[:, 2])), truth=float(np.median(v[:, 3])), coverage=float(inside), cia_err=float(np.mean(np.abs(v[:, 2] - v[:, 3]))))
        out.append(r)
        print(f"   {f:14s} | X = {s:24s} bounds [{r['lo']:.3f}, {r['hi']:.3f}] (width {r['width']:.3f})  CIA point {r['cia']:.3f}"
              f"  truth {r['truth']:.3f}  coverage {inside:.2f}  CIA error {r['cia_err']:.3f}")
    return out


def whatif_mbic(d):
    A = (d["age"] >= 55).astype(int); sc, e, _, _ = scenario_tables(d, d, ["I"], A, A)
    Y1A1R = event(sc, lambda g: g["Y"] == 1 and g["A"] == 1 and g["X"] == "R"); A1R = event(sc, lambda g: g["A"] == 1 and g["X"] == "R")
    base = grey_bounds(e, event(sc, lambda g: g["Y"] == 1 and g["A"] == 1), given=event(sc, lambda g: g["A"] == 1))
    rows = []
    for s in np.round(np.linspace(0, 1, 41), 3):
        h = hypothesis(sc, lambda g: g["Y"] == 1, lambda g: g["A"] == 1, s, s)
        b = grey_bounds(e, Y1A1R, given=A1R, rows=h)
        rows.append(dict(s=float(s), lo=b["lo"], hi=b["hi"], feasible=b["feasible"], slack=b["slack"]))
    feas = [r["s"] for r in rows if r["feasible"]]
    print(f"\n(2) MBIC what-if P(biased | age >= 55) = s: data allow s in [{base['lo']:.3f}, {base['hi']:.3f}]; sweep feasible for s in [{min(feas):.3f}, {max(feas):.3f}];"
          f" observed (hidden) {d.loc[A == 1, 'Y'].mean():.3f}")
    for r in rows[::5]:
        print(f"    s {r['s']:.3f}: " + (f"P(biased | 55+, right) in [{r['lo']:.3f}, {r['hi']:.3f}]" if r["feasible"] else f"ruled out, L1 slack {r['slack']:.3f}"))
    return dict(rows=rows, allowed=[base["lo"], base["hi"]], observed=float(d.loc[A == 1, "Y"].mean()),
                observed_right=float(d.loc[(A == 1) & (d["I"] == "R"), "Y"].mean()))


def allsides_parts():
    import load
    st = collections.defaultdict(dict)
    for r in load.load():
        k = (r["date"], r["topic"]); s = r["side"][0].upper(); st[k][s] = max(st[k].get(s, 0), r["outcome"])
    pair_only = collections.defaultdict(list); triple = []
    for v in st.values():
        if len(v) == 3:
            triple.append([v["L"], v["C"], v["R"]])
        elif len(v) == 2:
            C = tuple(s for s in "LCR" if s in v); pair_only[C].append([v.get("L", 0), v.get("C", 0), v.get("R", 0)])
    return {C: np.array(v) for C, v in pair_only.items()}, np.array(triple)


def tab2(T, a, b):
    I = {"L": 0, "C": 1, "R": 2}; t = np.zeros(4)
    for x in T:
        t[2 * x[I[a]] + x[I[b]]] += 1
    return t


def whatif_allsides(parts, T3):
    sc = Scenario({"L": (0, 1), "C": (0, 1), "R": (0, 1)}, (("L", "C"), ("C", "R"), ("L", "R")))
    e = EmpiricalModel(sc, {C: (tab2(parts[C], *C) + 0.5) / (tab2(parts[C], *C) + 0.5).sum() for C in sc.contexts})
    maj = functional(sc, lambda g: float(g["L"] + g["C"] + g["R"] >= 2)); rows = []
    for s in np.round(np.linspace(0, 1, 41), 3):
        h = hypothesis(sc, lambda g: g["R"] == 1, lambda g: g["L"] == 1 and g["C"] == 1, s, s)
        b = grey_bounds(e, maj, rows=h)
        rows.append(dict(s=float(s), lo=b["lo"], hi=b["hi"], feasible=b["feasible"], slack=b["slack"]))
    feas = [r["s"] for r in rows if r["feasible"]]
    obs = float(T3[(T3[:, 0] == 1) & (T3[:, 1] == 1), 2].mean())
    print(f"\n(3) AllSides what-if P(R adv | L and C adv) = s: feasible for s in [{min(feas):.3f}, {max(feas):.3f}] (triple stories show {obs:.3f})")
    for r in rows[::5]:
        print(f"    s {r['s']:.3f}: " + (f"P(majority adversarial) in [{r['lo']:.3f}, {r['hi']:.3f}]" if r["feasible"] else f"ruled out, L1 slack {r['slack']:.3f}"))
    return dict(rows=rows, observed_s=obs, observed_majority=float((T3.sum(1) >= 2).mean()))


def population_test(parts, T3):
    print("\n(4) Population test: each pair's table from pair-only stories vs from triple-covered stories")
    out = {}
    for C, T in parts.items():
        a, b = tab2(T, *C), tab2(T3, *C)
        tv = 0.5 * np.abs(a / a.sum() - b / b.sum()).sum()
        chi2, p, _, _ = stats.chi2_contingency(np.vstack([a, b]) + 0.5)
        out["-".join(C)] = dict(tv=float(tv), p=float(p), n_pair=int(a.sum()), n_triple=int(b.sum()),
                                adv_pair=[float(T[:, {"L": 0, "C": 1, "R": 2}[x]].mean()) for x in C],
                                adv_triple=[float(T3[:, {"L": 0, "C": 1, "R": 2}[x]].mean()) for x in C])
        o = out["-".join(C)]
        print(f"   {'-'.join(C)}: TV {tv:.3f}, chi2 p {p:.2g}  (P(adv) pair-only {o['adv_pair'][0]:.3f}/{o['adv_pair'][1]:.3f}"
              f" vs triple {o['adv_triple'][0]:.3f}/{o['adv_triple'][1]:.3f})")
    return out


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#b8b6ae"
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7.5)
    V = [r for r in OUT["validation"]]; y = np.arange(len(V))[::-1]
    for yi, r in zip(y, V):
        ax[0].plot([r["lo"], r["hi"]], [yi, yi], color=blue, lw=4, alpha=0.35, solid_capstyle="butt")
        ax[0].scatter([r["truth"]], [yi], color=ink, s=14, zorder=3); ax[0].scatter([r["cia"]], [yi], color=orange, s=14, marker="x", zorder=3)
    ax[0].set_yticks(y); ax[0].set_yticklabels([f"{r['filter']} | {r['strata']}" for r in V], fontsize=6.3)
    ax[0].scatter([], [], color=ink, s=14, label="hidden truth"); ax[0].scatter([], [], color=orange, marker="x", s=14, label="independence (max-ent) point")
    ax[0].plot([], [], color=blue, lw=4, alpha=0.35, label="bounds over all stitchings"); ax[0].legend(fontsize=6.3, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3)
    ax[0].set_xlabel("P(biased | filter)", fontsize=8, color=muted); ax[0].set_xlim(0, 1)
    ax[0].set_title("(a) MBIC: stitching a filter never seen with the label", fontsize=8.5, loc="left", color=ink)
    for a, W, ttl, xl, yl in ((ax[1], OUT["whatif_mbic"], "(b) What if P(biased | 55+) = s?", "assumed s", "P(biased | 55+, right-leaning)"),
                              (ax[2], OUT["whatif_allsides"], "(c) What if P(R adv | L, C adv) = s?", "assumed s", "P(majority of sides adversarial)")):
        R = W["rows"]; S = [r["s"] for r in R]; F = [r for r in R if r["feasible"]]
        if max(r["hi"] - r["lo"] for r in F) > 1e-6:
            a.fill_between([r["s"] for r in F], [r["lo"] for r in F], [r["hi"] for r in F], color=blue, alpha=0.25, lw=0, label="bounds under the assumption")
        else:
            a.plot([r["s"] for r in F], [r["lo"] for r in F], color=blue, lw=2, label="identified value under the assumption")
        for r0, r1 in zip(R[:-1], R[1:]):
            if not r0["feasible"]:
                a.axvspan(r0["s"], r1["s"], color=grey, alpha=0.25, lw=0)
        if any(not r["feasible"] for r in R):
            a.text(0.98, 0.5, "ruled out\nby the data", transform=a.transAxes, fontsize=6.5, color=muted, ha="right")
        a.set_xlabel(xl, fontsize=8, color=muted); a.set_ylabel(yl, fontsize=8, color=muted); a.set_title(ttl, fontsize=8.5, loc="left", color=ink)
        a.set_xlim(0, 1)
    ax[1].axvline(OUT["whatif_mbic"]["observed"], color=ink, lw=0.9, ls="--"); ax[1].axhline(OUT["whatif_mbic"]["observed_right"], color=ink, lw=0.9, ls=":")
    ax[1].text(OUT["whatif_mbic"]["observed"] + 0.01, 0.80, "hidden truth", fontsize=6.5, color=ink, rotation=90)
    ax[2].axvline(OUT["whatif_allsides"]["observed_s"], color=ink, lw=0.9, ls="--")
    ax[2].text(OUT["whatif_allsides"]["observed_s"] + 0.01, 0.02, "s on triple stories", fontsize=6.5, color=ink, rotation=90)
    ax[2].axhline(OUT["whatif_allsides"]["observed_majority"], color=ink, lw=0.9, ls=":")
    ax[2].text(0.02, OUT["whatif_allsides"]["observed_majority"] + 0.002, "majority on triple stories", fontsize=6.5, color=ink)
    ax[1].text(0.02, OUT["whatif_mbic"]["observed_right"] + 0.015, "hidden truth (55+, right)", fontsize=6.5, color=ink)
    ax[1].set_ylim(0, 1); ax[2].set_ylim(0, 0.14)

    fig.tight_layout(); fig.savefig(os.path.join(here, "grey_chart.pdf")); fig.savefig(os.path.join(here, "grey_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    d = mbic_frame(); parts, T3 = allsides_parts()
    OUT = dict(validation=validation(d), whatif_mbic=whatif_mbic(d), whatif_allsides=whatif_allsides(parts, T3), population=population_test(parts, T3))
    json.dump(OUT, open(os.path.join(here, "results.json"), "w"), indent=1, default=float)
    plot(OUT)
