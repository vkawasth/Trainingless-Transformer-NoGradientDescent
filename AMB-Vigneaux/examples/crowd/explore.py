"""Exploring the open problems on crowdsourced entity resolution (CrowdER product data; paper Section sec:crowd).

Data: CROWD_DIR = datasets/d_jn-product of github.com/zhydhkcws/crowd_truth_infer. 8,315 product pairs, 176 workers,
exactly three workers per item, binary answer, gold truth for every item. Not redistributed.

(A) Second order: all tetrahedral covers (worker quadruples whose four triples each labelled >= MIN items; the four
    never labelled an item together). Raw and gold-stratified (gold = 0 items, the majority class): CF of the four triple
    tables against a sampling floor (multinomial redraws, same counts, from the nearest consistent law); population test
    (do the triples' items differ in gold rate? chi^2); Benjamini-Hochberg over quadruples. Bounds on
    P(>= 3 of 4 say 'match').
(B) Interaction order inside a gold class: Dawid-Skene assumes workers independent given the truth. Observed vs expected
    numbers of items on which two, or all three, workers err (expectation from each worker's error rate in that class;
    item bootstrap). Excess co-error = item difficulty, a latent variable beyond the truth.
(C) Holonomy / loop content on worker x difficulty: difficulty of an item for worker i = number of the OTHER two workers
    who erred (0, 1, 2). Cells (worker, difficulty) -> (errors, n). Additive logit model (natural part: worker skill +
    difficulty) vs loop residual (chi^2): does difficulty shift every worker alike (e-flat) or differently (curvature)?
(D) Outcomes: majority vote. Split type (3-0 vs 2-1) and its error rate; pivotal items (a single flip changes the
    majority); one-coin Dawid-Skene EM (no gradient steps) vs majority vote, accuracy against gold.
(E) Singular statistics: latent classes for the item (truth only, K = 2; truth x difficulty, K = 3, 4) in the
    three-answer table of the busiest worker triples: BIC vs held-out likelihood.
Env: CROWD_DIR.
"""
import os, sys, json, itertools, collections
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np, pandas as pd
from scipy import stats
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.outcome import contextual_fraction
from amb_vigneaux.bounds import functional, outcome_bounds
from amb_vigneaux.deficits import nearest_consistent
from amb_vigneaux.lattice_path import fit_region
from amb_vigneaux import singular as SG

rng = np.random.default_rng(0)
D = os.environ.get("CROWD_DIR", "data/crowd_truth_infer/datasets/d_jn-product")
M4 = ("a", "b", "c", "d"); SC = Scenario({x: (0, 1) for x in M4}, tuple(itertools.combinations(M4, 3)))


def load():
    a = pd.read_csv(os.path.join(D, "answer.csv")); gold = pd.read_csv(os.path.join(D, "truth.csv")).set_index("question")["truth"]
    per = {q: dict(zip(g.worker, g.answer)) for q, g in a.groupby("question")}
    return a, gold, per


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def part_a(a, gold, per, MIN=12, top_k=60, B=200):
    out = {}
    for stratum in ("all", "gold0"):
        tri = collections.defaultdict(list)
        for q, v in per.items():
            if stratum == "gold0" and gold[q] != 0:
                continue
            for c in itertools.combinations(sorted(v), 3):
                tri[c].append(q)
        top = [w for w, _ in collections.Counter(a.worker).most_common(top_k)]
        rows = []
        for q4 in itertools.combinations(sorted(top), 4):
            trips = list(itertools.combinations(q4, 3))
            if min(len(tri.get(c, [])) for c in trips) < MIN:
                continue
            counts = {}
            for C, trip in zip(SC.contexts, trips):
                t = np.zeros(8)
                for q in tri[trip]:
                    v = per[q]; t[4 * v[trip[0]] + 2 * v[trip[1]] + v[trip[2]]] += 1
                counts[C] = t
            e = EmpiricalModel(SC, {C: (t + .5) / (t + .5).sum() for C, t in counts.items()})
            cf = contextual_fraction(e).value
            qq = nearest_consistent(e, weights={C: counts[C].sum() for C in SC.contexts})
            null = np.array([contextual_fraction(EmpiricalModel(SC, {C: (lambda v: (v + .5) / (v + .5).sum())(
                rng.multinomial(int(counts[C].sum()), SC.restriction_matrix(SC.measurements, C) @ qq)) for C in SC.contexts})).value for _ in range(B)])
            g = np.array([[gold[tri[t]].sum(), len(tri[t]) - gold[tri[t]].sum()] for t in trips]) + .5
            pop_p = float(stats.chi2_contingency(g)[1]) if stratum == "all" else np.nan
            bd = outcome_bounds(e, functional(SC, lambda x: float(sum(x.values()) >= 3)))
            rows.append(dict(workers=list(q4), n=[int(counts[C].sum()) for C in SC.contexts], CF=cf, floor_med=float(np.median(null)),
                             p=float((null >= cf).mean() * B / (B + 1) + 1 / (B + 1)), pop_p=pop_p, bounds=[bd["lo"], bd["hi"]]))
        q = bh([r["p"] for r in rows]) if rows else []
        for r, qq_ in zip(rows, q):
            r["q_bh"] = float(qq_)
        sig = [r for r in rows if r["q_bh"] < 0.05]
        out[stratum] = dict(rows=rows, n_covers=len(rows), n_sig_bh=len(sig), n_raw_p05=int(sum(r["p"] < 0.05 for r in rows)),
                            sig_with_pop_shift=int(sum(1 for r in sig if r.get("pop_p", 1) < 0.05)) if stratum == "all" else None)
        print(f"(A) {stratum:5s}: {len(rows)} tetrahedral covers (all triples >= {MIN} items); CF above floor at p<0.05: {out[stratum]['n_raw_p05']};"
              f" after Benjamini-Hochberg: {len(sig)}" + (f" (of which with a significant gold-rate difference between triples: {out[stratum]['sig_with_pop_shift']})" if stratum == "all" else "")
              + f"; median CF {np.median([r['CF'] for r in rows]):.3f} vs median floor {np.median([r['floor_med'] for r in rows]):.3f}")
    return out


def part_b(a, gold, per, B=500):
    """co-error excess: observed vs expected (workers independent given the truth) counts of items where two / all three
    workers err, within gold = 0 and gold = 1; expectation from each worker's error rate in that gold class; item bootstrap"""
    out = {}
    for g in (0, 1):
        Q = [q for q in per if gold[q] == g]
        err = collections.defaultdict(lambda: [0, 0])
        for q in Q:
            for w, x in per[q].items():
                err[w][0] += int(x != g); err[w][1] += 1
        e = {w: (k + 0.5) / (n + 1) for w, (k, n) in err.items()}
        rows = []
        for q in Q:
            ws = sorted(per[q]); wr = [int(per[q][w] != g) for w in ws]; ew = [e[w] for w in ws]
            o2 = sum(wr[i] * wr[j] for i, j in itertools.combinations(range(3), 2)); e2 = sum(ew[i] * ew[j] for i, j in itertools.combinations(range(3), 2))
            rows.append((o2, e2, int(all(wr)), float(np.prod(ew))))
        R = np.array(rows); o2, e2, o3, e3 = R.sum(0)
        bs2, bs3 = [], []
        for _ in range(B):
            idx = rng.integers(len(R), size=len(R)); S = R[idx].sum(0); bs2.append(S[0] / S[1]); bs3.append(S[2] / S[3])
        out[f"gold{g}"] = dict(items=len(Q), pair_OE=float(o2 / e2), pair_ci=[float(np.quantile(bs2, .025)), float(np.quantile(bs2, .975))],
                               triple_OE=float(o3 / e3), triple_ci=[float(np.quantile(bs3, .025)), float(np.quantile(bs3, .975))],
                               pair_obs=int(o2), pair_exp=float(e2), triple_obs=int(o3), triple_exp=float(e3))
        o = out[f"gold{g}"]
        print(f"(B) gold = {g} ({len(Q)} items): co-errors observed / expected under conditional independence -- pairs {o['pair_OE']:.2f} "
              f"[{o['pair_ci'][0]:.2f}, {o['pair_ci'][1]:.2f}] ({int(o2)} vs {e2:.0f}); all three {o['triple_OE']:.2f} [{o['triple_ci'][0]:.2f}, {o['triple_ci'][1]:.2f}] ({int(o3)} vs {e3:.1f})")
    return out


def part_c(a, gold, per, min_n=8):
    cells = collections.defaultdict(lambda: [0, 0])
    for q, v in per.items():
        wrong = {w: int(x != gold[q]) for w, x in v.items()}
        for w in v:
            d = sum(wrong[o] for o in v if o != w)
            c = cells[(w, f"d{d}")]; c[0] += wrong[w]; c[1] += 1
    cells = {k: tuple(v) for k, v in cells.items() if v[1] >= min_n}
    workers = sorted({w for w, _ in cells}); diffs = sorted({d for _, d in cells})
    fit = fit_region(cells, workers, set(diffs))
    T = fit["T"]; S = fit["S"]; beta = {t: float(fit["coef"][len(S) + T.index(t)]) for t in T}
    base = beta[T[0]]; beta = {t: b - base for t, b in beta.items()}
    out = dict(workers=len(S), cells=len(cells), Q=fit["Q"], df=fit["df"], p=fit["p"], difficulty_effect=beta,
               error_rate={d: float(sum(cells[k][0] for k in cells if k[1] == d) / sum(cells[k][1] for k in cells if k[1] == d)) for d in diffs})
    print(f"(C) worker x difficulty ({len(S)} workers, {len(cells)} cells >= {min_n}): additive (natural) logit model; loop residual Q {fit['Q']:.1f} on {fit['df']} df, p {fit['p']:.2g};"
          f" difficulty effect on the logit of an error (vs d0): " + ", ".join(f"{t} {b:+.2f}" for t, b in beta.items())
          + "; pooled error rate " + ", ".join(f"{d} {r:.3f}" for d, r in out["error_rate"].items()))
    return out


def one_coin_ds(per, iters=200):
    W = sorted({w for v in per.values() for w in v}); idx = {w: i for i, w in enumerate(W)}
    acc = np.full(len(W), 0.8); prior = 0.2; Q = list(per)
    for _ in range(iters):
        post = {}
        for q in Q:
            l1, l0 = np.log(prior), np.log(1 - prior)
            for w, x in per[q].items():
                s = acc[idx[w]]; l1 += np.log(s if x == 1 else 1 - s); l0 += np.log(s if x == 0 else 1 - s)
            post[q] = 1 / (1 + np.exp(l0 - l1))
        num = np.zeros(len(W)); den = np.zeros(len(W))
        for q in Q:
            for w, x in per[q].items():
                num[idx[w]] += post[q] if x == 1 else 1 - post[q]; den[idx[w]] += 1
        acc = np.clip((num + 1) / (den + 2), 0.01, 0.99); prior = float(np.mean(list(post.values())))
    return post


def part_d(gold, per):
    split = {}; err = collections.Counter(); cnt = collections.Counter()
    for q, v in per.items():
        s = sum(v.values()); kind = "3-0" if s in (0, 3) else "2-1"; mv = int(s >= 2)
        cnt[kind] += 1; err[kind] += mv != gold[q]
    post = one_coin_ds(per)
    ds_acc = float(np.mean([(post[q] > 0.5) == gold[q] for q in per])); mv_acc = float(np.mean([(sum(v.values()) >= 2) == gold[q] for q, v in per.items()]))
    out = dict(frac_2_1=cnt["2-1"] / len(per), err_3_0=err["3-0"] / cnt["3-0"], err_2_1=err["2-1"] / cnt["2-1"], mv_acc=mv_acc, ds_acc=ds_acc,
               share_of_errors_in_2_1=err["2-1"] / (err["2-1"] + err["3-0"]))
    print(f"(D) outcomes: {out['frac_2_1']:.0%} of items are 2-1 splits (pivotal: one flip changes the majority); majority-vote error 3-0 {out['err_3_0']:.3f},"
          f" 2-1 {out['err_2_1']:.3f}; {out['share_of_errors_in_2_1']:.0%} of all errors are on pivotal items; accuracy: majority {mv_acc:.3f}, one-coin Dawid-Skene {ds_acc:.3f}")
    return out


def part_e(per, top=6, reps=10):
    trip = collections.defaultdict(list)
    for q, v in per.items():
        trip[tuple(sorted(v))].append(tuple(v[w] for w in sorted(v)))
    busiest = sorted(trip, key=lambda k: -len(trip[k]))[:top]
    res = []
    for k in busiest:
        X = np.array(trip[k]); n = len(X); rows = {}
        for K in (1, 2, 3):
            c = np.bincount(X @ np.array([4, 2, 1]), minlength=8); em = SG.lca_em(c, K, 3, starts=10, rng=rng)
            d = (K - 1) + 3 * K; rows[K] = dict(bic=float(-em["loglik"] + d / 2 * np.log(n)), d=d)
        ho = {K: [] for K in (1, 2, 3)}
        for _ in range(reps):
            p = rng.permutation(n); tr, te = X[p[: n // 2]], X[p[n // 2:]]
            ctr = np.bincount(tr @ np.array([4, 2, 1]), minlength=8); cte = np.bincount(te @ np.array([4, 2, 1]), minlength=8)
            for K in (1, 2, 3):
                pr = SG.lca_em(ctr, K, 3, starts=5, rng=rng)["probs"]; ho[K].append(float(cte @ np.log(np.maximum(pr, 1e-9)) / cte.sum()))
        res.append(dict(n=n, bic_pick=min(rows, key=lambda K: rows[K]["bic"]), heldout_pick=max(ho, key=lambda K: np.mean(ho[K])),
                        heldout={K: float(np.mean(v)) for K, v in ho.items()}))
    print(f"(E) latent classes on the {top} busiest worker triples (n = {[r['n'] for r in res]}): BIC picks K = {[r['bic_pick'] for r in res]},"
          f" held-out likelihood picks K = {[r['heldout_pick'] for r in res]} (K = 2 is truth only; K = 3 needs no more than 7 free parameters on 8 cells)")
    return res


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#b8b6ae"
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    for a_ in ax:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.tick_params(colors=muted, labelsize=7.5)
    for st, col, lab in (("all", grey, "all items"), ("gold0", blue, "gold = 0 only")):
        R = OUT["A"][st]["rows"]
        ax[0].scatter([r["floor_med"] for r in R], [r["CF"] for r in R], s=[30 if r["q_bh"] < .05 else 12 for r in R], color=col, alpha=0.7, label=f"{lab} ({len(R)})")
    lim = [0, 1]; ax[0].plot(lim, lim, color=muted, lw=0.8, ls=":")
    ax[0].set_xlabel("sampling floor (median CF under consistency)", fontsize=8, color=muted); ax[0].set_ylabel("observed CF", fontsize=8, color=muted)
    ax[0].legend(fontsize=6.5, frameon=False, loc="upper left"); ax[0].set_title("(a) tetrahedral covers: CF vs floor\n(large dots: BH-significant)", fontsize=8, loc="left", color=ink)
    B = OUT["B"]; xpos = np.arange(2)
    for k, (g, col) in enumerate((("gold0", blue), ("gold1", orange))):
        vals = [B[g]["pair_OE"], B[g]["triple_OE"]]; lo = [B[g]["pair_ci"][0], B[g]["triple_ci"][0]]; hi = [B[g]["pair_ci"][1], B[g]["triple_ci"][1]]
        ax[1].bar(xpos + (k - 0.5) * 0.36, vals, width=0.34, color=col, alpha=0.75, yerr=[np.array(vals) - lo, np.array(hi) - vals], capsize=3,
                  label=f"gold = {g[-1]} ({B[g]['items']} items)")
    ax[1].axhline(1, color=ink, lw=0.8, ls=":"); ax[1].set_xticks(xpos); ax[1].set_xticklabels(["two workers err", "all three err"], fontsize=7.5)
    ax[1].set_ylabel("observed / expected (independent given truth)", fontsize=7.5, color=muted); ax[1].legend(fontsize=6.5, frameon=False, loc="upper left")
    ax[1].set_title("(b) co-errors exceed conditional independence", fontsize=8, loc="left", color=ink)
    C = OUT["C"]; ds = list(C["error_rate"])
    ax[2].bar(range(len(ds)), [C["error_rate"][d] for d in ds], color=blue, alpha=0.7)
    ax[2].set_xticks(range(len(ds))); ax[2].set_xticklabels([d.replace("d", "") for d in ds], fontsize=7.5); ax[2].set_xlabel("how many of the other two workers erred", fontsize=8, color=muted)
    ax[2].set_ylabel("error rate", fontsize=8, color=muted)
    ax[2].set_title(f"(c) worker × difficulty: loop test p = {C['p']:.2g}", fontsize=8, loc="left", color=ink)
    Dd = OUT["D"]
    ax[3].bar([0, 1], [Dd["err_3_0"], Dd["err_2_1"]], color=[grey, orange]); ax[3].set_xticks([0, 1]); ax[3].set_xticklabels(["3–0 items", "2–1 (pivotal)"], fontsize=7.5)
    ax[3].set_ylabel("majority-vote error", fontsize=8, color=muted)
    ax[3].set_title(f"(d) outcomes: MV {Dd['mv_acc']:.3f}, Dawid–Skene {Dd['ds_acc']:.3f}", fontsize=8, loc="left", color=ink)
    fig.tight_layout(); fig.savefig(os.path.join(here, "crowd_chart.pdf")); fig.savefig(os.path.join(here, "crowd_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    a, gold, per = load()
    print(f"{len(per)} items, {a.worker.nunique()} workers, gold positive rate {gold.mean():.3f}")
    OUT = dict(A=part_a(a, gold, per), B=part_b(a, gold, per), C=part_c(a, gold, per), D=part_d(gold, per), E=part_e(per))
    js = json.loads(json.dumps(OUT, default=lambda o: o.tolist() if hasattr(o, "tolist") else (float(o) if isinstance(o, (np.floating, np.integer)) else str(o))))
    json.dump(js, open(os.path.join(here, "results.json"), "w"), indent=1)
    plot(js)
