"""Bounds on outcomes over all gluings (amb_vigneaux/bounds.py): synthetic check, validation on real triangles, and
the AllSides stream.

Triangle scenario: measurements L, C, R in {0, 1}; contexts {L,C}, {C,R}, {L,R}; each context's table comes from
units where that pair was observed. Outcome functionals (none needs the triple to be observed jointly):
  pair_LR     P(L=1, R=1), bounded from {L,C} and {C,R} only (the {L,R} context dropped): a held-out context
  all         P(L=C=R=1), from the three pair tables
  majority    P(at least two of L, C, R = 1)
  unanimous   P(L=C=R)
(0) CHSH, lambda PR + noise, closing context dropped: bound on the closing correlator vs its truth.
(1) MBIC annotator groups (L / C / R by political_ideology; group outcome = group majority 'biased' on a sentence).
    1,600 sentences are rated by all three groups. 200 random splits into four parts: {L,C} from part 1, {C,R} from
    part 2, {L,R} from part 3, truth of every functional from part 4. Plug-in bounds (mode exact or projected), and
    bounds widened by a bootstrap of each part (B = 40; 2.5% of lo, 97.5% of hi). Coverage of the part-4 truth, width,
    and the error of the maximum-entropy point.
(2) AllSides stories (outcome of a side on a story = any adversarial article): (a) the same split design on the 736
    stories covered by all three sides; (b) the natural design: pair tables from the stories covered by exactly that
    pair (disjoint by construction), truth from the triple stories.
(3) AllSides stream: 26-week windows every 2 weeks; pair tables from all stories with that pair; bounds on 'all' and
    'majority' with the observed triple value (stories covered by all three) and the max-ent point. Figure.
Env: MBIC_XLSX, ALLSIDES_DIR.
"""
import os, sys, json, collections, datetime
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../bias3"), os.path.join(here, "../allsides")]
import numpy as np
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.models import pr_box, white_noise
from amb_vigneaux.outcome import contextual_fraction
from amb_vigneaux.bounds import functional, outcome_bounds

rng = np.random.default_rng(0)
SC = Scenario({"L": (0, 1), "C": (0, 1), "R": (0, 1)}, (("L", "C"), ("C", "R"), ("L", "R")))
SC2 = Scenario({"L": (0, 1), "C": (0, 1), "R": (0, 1)}, (("L", "C"), ("C", "R")))
F = {"all": (SC, functional(SC, lambda g: g["L"] * g["C"] * g["R"])),
     "majority": (SC, functional(SC, lambda g: float(g["L"] + g["C"] + g["R"] >= 2))),
     "unanimous": (SC, functional(SC, lambda g: float(g["L"] == g["C"] == g["R"]))),
     "pair_LR": (SC2, functional(SC2, lambda g: g["L"] * g["R"]))}
TRUTH = {"all": lambda T: (T[:, 0] * T[:, 1] * T[:, 2]).mean(), "majority": lambda T: (T.sum(1) >= 2).mean(),
         "unanimous": lambda T: ((T[:, 0] == T[:, 1]) & (T[:, 1] == T[:, 2])).mean(), "pair_LR": lambda T: (T[:, 0] * T[:, 2]).mean()}
IDX = {"L": 0, "C": 1, "R": 2}


def tab(T, a, b, smooth=0.0):
    t = np.zeros(4)
    for x in T:
        t[2 * int(x[IDX[a]]) + int(x[IDX[b]])] += 1
    t = t + smooth
    return t / t.sum(), int(len(T))


def model(parts, sc):
    tables, w = {}, {}
    for C in sc.contexts:
        tables[C], w[C] = tab(parts[C], *C, smooth=0.5)
    return EmpiricalModel(sc, tables), w


def bound(parts, name):
    sc, f = F[name]; m, w = model(parts, sc)
    return outcome_bounds(m, f, weights=w)


def split_validation(T, n_rep=200, B=40, label=""):
    res = {k: dict(cov=0, cov_boot=0, cov_pred=0, width=[], width_boot=[], err_point=[], projected=0) for k in F}
    for _ in range(n_rep):
        idx = rng.permutation(len(T)); P = np.array_split(idx, 4)
        parts = {("L", "C"): T[P[0]], ("C", "R"): T[P[1]], ("L", "R"): T[P[2]]}; truth_units = T[P[3]]
        for k in F:
            b = bound(parts, k); tr = TRUTH[k](truth_units); r = res[k]
            r["cov"] += b["lo"] - 1e-9 <= tr <= b["hi"] + 1e-9; r["width"].append(b["hi"] - b["lo"])
            r["err_point"].append(abs(b["point"] - tr)); r["projected"] += b["mode"] == "projected"
            los, his = [], []
            for _ in range(B):
                bp = {C: v[rng.integers(len(v), size=len(v))] for C, v in parts.items()}
                bb = bound(bp, k); los.append(bb["lo"]); his.append(bb["hi"])
            lo, hi = np.quantile(los, 0.025), np.quantile(his, 0.975)
            r["cov_boot"] += lo - 1e-9 <= tr <= hi + 1e-9; r["width_boot"].append(hi - lo)
            z = 1.96 * np.sqrt(max(tr * (1 - tr), 1e-12) / len(truth_units))      # the part-4 truth is itself a sample
            r["cov_pred"] += lo - z - 1e-9 <= tr <= hi + z + 1e-9
    out = {}
    print(f"\n({label}) {len(T)} units, {n_rep} random 4-way splits")
    for k, r in res.items():
        out[k] = dict(coverage=r["cov"] / n_rep, coverage_boot=r["cov_boot"] / n_rep, coverage_pred=r["cov_pred"] / n_rep, width=float(np.median(r["width"])),
                      width_boot=float(np.median(r["width_boot"])), maxent_err=float(np.mean(r["err_point"])), projected=r["projected"] / n_rep)
        o = out[k]
        print(f"   {k:10s} plug-in coverage {o['coverage']:.2f} (width {o['width']:.3f}, projected {o['projected']:.2f})"
              f" | bootstrap-widened coverage {o['coverage_boot']:.2f} (width {o['width_boot']:.3f}), + truth-sample noise {o['coverage_pred']:.2f} | max-ent point error {o['maxent_err']:.3f}")
    return out


def allsides_units():
    import load
    st = collections.defaultdict(dict)
    for r in load.load():
        k = (r["date"], r["topic"]); s = r["side"][0].upper()
        st[k][s] = max(st[k].get(s, 0), r["outcome"])
    return st


def plot(OUT):
    rows, stream = OUT["chsh"], OUT["stream"]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
    ink, muted, blue, orange = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw=dict(width_ratios=[1, 1.3, 1.3]))
    for a_ in ax:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.tick_params(colors=muted, labelsize=7.5)
    L = [r["lam"] for r in rows]
    ax[0].fill_between(L, [r["lo"] for r in rows], [r["hi"] for r in rows], color=blue, alpha=0.18, lw=0, label="bounds from the other 3")
    ax[0].plot(L, [r["truth"] for r in rows], color=ink, lw=1.6, label="truth (−λ)"); ax[0].plot(L, [r["point"] for r in rows], color=blue, lw=1.2, ls="--", label="max-ent (λ³)")
    ax[0].axvline(0.5, color=muted, lw=0.8, ls=":"); ax[0].text(0.52, -0.95, "CF > 0", fontsize=7, color=muted)
    ax[0].set_xlabel("λ (weight of PR box)", fontsize=8, color=muted); ax[0].set_ylabel("correlator of the dropped context", fontsize=8, color=muted)
    ax[0].set_title("(a) CHSH: truth leaves the bounds iff CF > 0", fontsize=8.5, loc="left", color=ink); ax[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    x = [datetime.date.fromisoformat(r["mid"][:10]) for r in stream]
    for a_, k, col, ttl in ((ax[1], "all", blue, "(b) P(all three sides adversarial)"), (ax[2], "majority", orange, "(c) P(majority of sides adversarial)")):
        a_.fill_between(x, [r[k]["lo"] for r in stream], [r[k]["hi"] for r in stream], color=col, alpha=0.2, lw=0, label="bounds over all gluings")
        a_.plot(x, [r[k]["point"] for r in stream], color=col, lw=1.2, ls="--", label="max-ent point")
        a_.plot(x, [r[k]["truth"] for r in stream], color=ink, lw=1.4, label="observed on triple-covered stories")
        a_.set_title(ttl + ", AllSides", fontsize=8.5, loc="left", color=ink); a_.legend(fontsize=6.5, frameon=False, loc="upper left")
        a_.xaxis.set_major_locator(mdates.YearLocator()); a_.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        a_.set_ylim(0, 0.34)
    fig.tight_layout(); fig.savefig(os.path.join(here, "bounds_chart.pdf")); fig.savefig(os.path.join(here, "bounds_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    OUT = {}
    # identification: the fibre of the triangle is a segment along the parity chi = (-1)^(L+C+R) (ker R is 1-dim);
    # an outcome is identified (width 0) iff <f, chi> = 0
    chi = functional(SC, lambda g: (-1.0) ** (g["L"] + g["C"] + g["R"]))
    Rm = np.vstack([SC.restriction_matrix(SC.measurements, C) for C in SC.contexts])
    print(f"(id) triangle: dim ker R = {Rm.shape[1] - np.linalg.matrix_rank(Rm)}; R chi = 0: {np.allclose(Rm @ chi, 0)};  <f, chi>: "
          + ", ".join(f"{k} {F[k][1] @ chi:+.0f}" for k in ("all", "majority", "unanimous")))
    # (0)
    PR = pr_box(); sc = PR.scenario; WN = white_noise(sc); sub = Scenario(dict(sc.outcomes), sc.contexts[:3]); a, b = sc.contexts[3]
    f = functional(sub, lambda g: 1.0 if g[a] == g[b] else -1.0); rows = []
    print("(0) CHSH: bound on the dropped context's correlator from the other three")
    for lam in np.round(np.linspace(0, 1, 11), 2):
        e = {C: lam * PR.tables[C] + (1 - lam) * WN.tables[C] for C in sc.contexts}
        bd = outcome_bounds(EmpiricalModel(sub, {C: e[C] for C in sub.contexts}), f); truth = -lam
        cf = contextual_fraction(EmpiricalModel(sc, e)).value
        rows.append(dict(lam=float(lam), lo=bd["lo"], hi=bd["hi"], point=bd["point"], truth=truth, CF=cf, inside=bool(bd["lo"] - 1e-9 <= truth <= bd["hi"] + 1e-9)))
        print(f"    lambda {lam:.1f}: [{bd['lo']:+.3f}, {bd['hi']:+.3f}]  max-ent {bd['point']:+.3f}  truth {truth:+.3f}  CF {cf:.2f}  inside {rows[-1]['inside']}")
    OUT["chsh"] = rows
    # (1)
    import mbic, pandas as pd
    d = mbic.load(); g = pd.cut(d["political_ideology"], [-11, -4, 3, 11], labels=["L", "C", "R"])
    S = d.assign(g=g).groupby(["sentence_id", "g"], observed=True)["b"].mean().unstack().dropna()
    T_mbic = (S[["L", "C", "R"]].to_numpy() >= 0.5).astype(int)
    OUT["mbic_split"] = split_validation(T_mbic, label="1: MBIC annotator groups")
    # (2)
    st = allsides_units()
    T3 = np.array([[v["L"], v["C"], v["R"]] for v in st.values() if len(v) == 3])
    OUT["allsides_split"] = split_validation(T3, label="2a: AllSides triple stories")
    pairs = {("L", "C"): [], ("C", "R"): [], ("L", "R"): []}
    for v in st.values():
        if len(v) == 2:
            C = tuple(s for s in "LCR" if s in v); row = [v.get("L", 0), v.get("C", 0), v.get("R", 0)]; pairs[C].append(row)
    parts = {C: np.array(v) for C, v in pairs.items()}
    print(f"\n(2b) AllSides natural design: pair-only stories {[len(v) for v in parts.values()]}, truth from {len(T3)} triple stories")
    nat = {}
    for k in F:
        bd = bound(parts, k); tr = TRUTH[k](T3)
        nat[k] = dict(lo=bd["lo"], hi=bd["hi"], point=bd["point"], truth=float(tr), mode=bd["mode"], defect=bd["defect"])
        print(f"   {k:10s} [{bd['lo']:.3f}, {bd['hi']:.3f}] ({bd['mode']}, defect {bd['defect']:.3f})  max-ent {bd['point']:.3f}  triple-story truth {tr:.3f}")
    OUT["allsides_natural"] = nat
    # (3)
    import load
    R = load.load(); W = datetime.timedelta(weeks=26); step = datetime.timedelta(weeks=2)
    t = min(r["date"] for r in R); end = max(r["date"] for r in R); stream = []
    while t + W <= end + datetime.timedelta(days=1):
        stw = collections.defaultdict(dict)
        for r in R:
            if t <= r["date"] < t + W:
                k = (r["date"], r["topic"]); s = r["side"][0].upper(); stw[k][s] = max(stw[k].get(s, 0), r["outcome"])
        parts = {C: np.array([[v.get("L", 0), v.get("C", 0), v.get("R", 0)] for v in stw.values() if C[0] in v and C[1] in v]) for C in SC.contexts}
        T3w = np.array([[v["L"], v["C"], v["R"]] for v in stw.values() if len(v) == 3])
        if min(len(v) for v in parts.values()) >= 30 and len(T3w) >= 30:
            row = dict(mid=str(t + W / 2))
            for k in ("all", "majority"):
                bd = bound(parts, k); row[k] = dict(lo=bd["lo"], hi=bd["hi"], point=bd["point"], truth=float(TRUTH[k](T3w)), defect=bd["defect"])
            stream.append(row)
        t += step
    inside = {k: np.mean([r[k]["lo"] - 1e-9 <= r[k]["truth"] <= r[k]["hi"] + 1e-9 for r in stream]) for k in ("all", "majority")}
    print(f"\n(3) AllSides stream: {len(stream)} windows; observed triple value inside the bounds: all {inside['all']:.2f}, majority {inside['majority']:.2f};"
          f" median width all {np.median([r['all']['hi'] - r['all']['lo'] for r in stream]):.3f}, majority {np.median([r['majority']['hi'] - r['majority']['lo'] for r in stream]):.3f}")
    OUT["stream"] = stream
    json.dump(OUT, open(os.path.join(here, "results.json"), "w"), indent=1, default=float)

    plot(OUT)
