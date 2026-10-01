"""One chart, all layers: outcome series, probabilities, perturbations (natural vs non-natural), contextuality (CF,
gamma, gluing defect) and gluability, on the AllSides stream with the approval / 2016-margin outcome.

Windows: 26 weeks, stepped every 2 weeks (as stream_outcome.py). Per window:
  outcome      weekly net approval (2017-20) and Trump-Clinton margin (2016), from outcome_series.py
  probability  P_L, P_C, P_R = share of each side's Trump articles with adversarial framing
  perturbation Delta b = b(window) - b(window 26 weeks earlier) on the side x topic table of rates, split
               (dynamics.natural_split) into its natural part (re-calibrations u_s + v_t) and its non-natural part
               (loops); RMS of each over cells
  contextuality the TRIANGLE scenario: measurements L, C, R in {adversarial, not}; contexts {L,C}, {C,R}, {L,R};
               the 2x2 table of a context counts stories (AllSides story groups) covered by both sides (outcome of a
               side on a story = any adversarial article). Different stories feed different contexts, so the family
               can fail to glue. CF = contextual fraction (LP; includes signalling), signalling = mean total variation
               between the two marginals of each side, gluing defect = min_q sum_C KL(e_C || q_C) over global laws
               (nearest consistent law), gamma = AMB class on the support.
  gluability   parametric-bootstrap p of the loop-free model on side x topic x outcome (stream_outcome.glue_p)
Env: APPROVAL_CSV, POLLS2016_CSV, ALLSIDES_DIR.
"""
import os, sys, json, datetime, collections
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../allsides"), here]
import numpy as np
import load, stream_outcome as SO, outcome_series as OS
from amb_vigneaux.dynamics import natural_split
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.outcome import contextual_fraction, analyse_outcomes
from amb_vigneaux.deficits import nearest_consistent

SIDES = ("left", "center", "right"); G = SO.GROUPS
SC = Scenario({"L": (0, 1), "C": (0, 1), "R": (0, 1)}, [("L", "C"), ("C", "R"), ("L", "R")])
KEY = {"left": "L", "center": "C", "right": "R"}


def rate_table(rows):
    T = np.full((3, len(G)), np.nan)
    for i, s in enumerate(SIDES):
        for j, g in enumerate(G):
            x = [r["outcome"] for r in rows if r["side"] == s and r["group"] == g]
            if len(x) >= 5:
                T[i, j] = np.mean(x)
    return T


def triangle(rows):
    st = collections.defaultdict(dict)
    for r in rows:
        k = (r["date"], r["topic"]); st[k][r["side"]] = max(st[k].get(r["side"], 0), r["outcome"])
    tabs, ns = {}, {}
    for C in SC.contexts:
        a, b = [s for s in SIDES if KEY[s] == C[0]][0], [s for s in SIDES if KEY[s] == C[1]][0]
        t = np.zeros(4)
        for v in st.values():
            if a in v and b in v:
                t[2 * v[a] + v[b]] += 1
        ns[C] = int(t.sum()); tabs[C] = t + 0.5                     # Jeffreys-type smoothing for empty cells
    e = EmpiricalModel(SC, tabs)
    cf = contextual_fraction(e).value
    marg = {m: [] for m in "LCR"}
    for C in SC.contexts:
        t = e.tables[C].reshape(2, 2); marg[C[0]].append(t.sum(1)[1]); marg[C[1]].append(t.sum(0)[1])
    sig = float(np.mean([abs(v[0] - v[1]) for v in marg.values()]))
    q = nearest_consistent(e, weights={C: ns[C] for C in SC.contexts})
    R = {C: SC.restriction_matrix(SC.measurements, C) for C in SC.contexts}
    kl = float(sum(ns[C] / sum(ns.values()) * np.sum(e.tables[C] * np.log(e.tables[C] / np.maximum(R[C] @ q, 1e-300)))
                   for C in SC.contexts))
    raw = EmpiricalModel(SC, {C: tabs[C] - 0.5 + 1e-12 for C in SC.contexts})
    gam = analyse_outcomes(raw, eps=1e-9, with_cf=False).gamma_h1_nonzero
    return dict(CF=cf, signalling=sig, glue_defect=kl, gamma=bool(gam), n_pairs={"-".join(C): v for C, v in ns.items()})


if __name__ == "__main__":
    R = [r for r in load.load() if r["group"] in G]
    app, m16, _ = OS.outcome_weekly()
    W = datetime.timedelta(weeks=26); step = datetime.timedelta(weeks=2)
    t = datetime.date(2015, 12, 14) + W; end = max(r["date"] for r in R)
    rows_out = []
    while t + W <= end + datetime.timedelta(days=1):
        cur = [r for r in R if t <= r["date"] < t + W]; prev = [r for r in R if t - W <= r["date"] < t]
        P = {s: float(np.mean([r["outcome"] for r in cur if r["side"] == s])) for s in SIDES}
        D = rate_table(cur) - rate_table(prev); ok = ~np.isnan(D).any(0)
        sp = natural_split(D[:, ok]) if ok.sum() >= 2 else None
        tri = triangle(cur); gp, _ = SO.glue_p(SO.table(cur, "outcome"), B=60)
        mid = t + W / 2
        rows_out.append(dict(mid=str(mid), P=P, natural=float(np.sqrt(np.mean(sp["natural"] ** 2))) if sp else np.nan,
                             non_natural=float(np.sqrt(np.mean(sp["non_natural"] ** 2))) if sp else np.nan, glue_p=gp, **tri))
        t += step
    print(f"{len(rows_out)} windows; gamma != 0 in {sum(r['gamma'] for r in rows_out)};  CF range {min(r['CF'] for r in rows_out):.3f}-{max(r['CF'] for r in rows_out):.3f};"
          f"  signalling range {min(r['signalling'] for r in rows_out):.3f}-{max(r['signalling'] for r in rows_out):.3f};"
          f"  corr(CF, signalling) {np.corrcoef([r['CF'] for r in rows_out], [r['signalling'] for r in rows_out])[0, 1]:.2f}")
    nat = np.array([r["natural"] for r in rows_out]); non = np.array([r["non_natural"] for r in rows_out])
    print(f"perturbation RMS: natural median {np.nanmedian(nat):.4f}, non-natural median {np.nanmedian(non):.4f}")
    # window-level association of each layer with the outcome (mean net approval in the window), 2017 on
    aw = []
    for r in rows_out:
        m = datetime.date.fromisoformat(r["mid"][:10]); w0 = (m - OS.W0).days // 7
        v = app[(app.index >= w0 - 13) & (app.index < w0 + 13)]
        aw.append(v.mean() if len(v) >= 10 else np.nan)
    aw = np.array(aw); ok = ~np.isnan(aw)
    for k in ("CF", "signalling", "glue_defect", "glue_p", "natural", "non_natural"):
        v = np.array([r[k] for r in rows_out], float); o = ok & ~np.isnan(v)
        print(f"   corr(window {k}, window mean net approval) = {np.corrcoef(v[o], aw[o])[0, 1]:+.2f}  (n {o.sum()} overlapping windows)")
    json.dump(rows_out, open(os.path.join(here, "layers_chart_results.json"), "w"), indent=1, default=float)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
    ink, muted = "#0b0b0b", "#52514e"; c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"
    x = [datetime.date.fromisoformat(r["mid"][:10]) for r in rows_out]
    wdate = lambda w: OS.W0 + datetime.timedelta(weeks=int(w))
    EV = {"Access Hollywood": "2016-10-07", "election": "2016-11-08", "Charlottesville": "2017-08-12",
          "Mueller": "2019-04-18", "impeachment": "2019-09-24", "COVID": "2020-03-13"}
    fig, ax = plt.subplots(6, 1, figsize=(7.4, 10.5), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1.2, 1, 1, 1, 0.9]))
    for a in ax:
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
        a.tick_params(colors=muted, labelsize=7.5); a.grid(axis="y", color="#e6e5e1", lw=0.6)
        for d in EV.values():
            a.axvline(datetime.date.fromisoformat(d), color=muted, lw=0.7, ls=":")
    ax[0].plot([wdate(w) for w in app.index], app.values, color=c1, lw=1.6, label="net approval (2017–20)")
    ax[0].plot([wdate(w) for w in m16.index], m16.values, color=c2, lw=1.6, label="Trump − Clinton margin (2016)")
    ax[0].axhline(0, color=ink, lw=0.7); ax[0].axhline(-10, color=muted, lw=0.7, ls="--")
    ax[0].legend(fontsize=7, frameon=False, loc="lower left"); ax[0].set_ylabel("points", fontsize=8, color=muted)
    ax[0].set_title("Outcome (dashed: approval toggle at −10)", fontsize=8.5, loc="left", color=ink)
    for d, nm in ((EV[k], k) for k in EV):
        dd = datetime.date.fromisoformat(d) + (datetime.timedelta(days=-20) if nm == "Access Hollywood" else datetime.timedelta(days=12) if nm == "election" else datetime.timedelta(0))
        ax[0].text(dd, ax[0].get_ylim()[1], nm, fontsize=6.5, color=muted, rotation=90, va="top", ha="right" if nm == "Access Hollywood" else "left")
    for s, c, nm in (("left", c1, "left"), ("center", c3, "centre"), ("right", c2, "right")):
        ax[1].plot(x, [r["P"][s] for r in rows_out], color=c, lw=1.6, label=nm)
    ax[1].legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    ax[1].set_title("Probability layer: P(adversarial framing of Trump) by side", fontsize=8.5, loc="left", color=ink)
    ax[2].plot(x, nat, color=c1, lw=1.6, label="natural (re-calibration)"); ax[2].plot(x, non, color=c2, lw=1.6, label="non-natural (loops)")
    ax[2].legend(fontsize=7, frameon=False, loc="upper right")
    ax[2].set_title("Perturbation: RMS change of the side×topic table vs 26 weeks earlier", fontsize=8.5, loc="left", color=ink)
    ax[3].plot(x, [r["CF"] for r in rows_out], color=c1, lw=1.6, label="CF (LP)")
    ax[3].plot(x, [r["signalling"] for r in rows_out], color=c2, lw=1.6, label="signalling (TV)")
    ax[3].legend(fontsize=7, frameon=False, loc="upper right")
    ax[3].set_title("Contextuality, triangle L–C–R on shared stories (γ = 0 in every window: full support)", fontsize=8.5, loc="left", color=ink)
    ax[4].plot(x, [r["glue_defect"] for r in rows_out], color=c1, lw=1.6)
    ax[4].set_title("Gluing defect: KL to the nearest consistent global law (nats)", fontsize=8.5, loc="left", color=ink)
    ax[5].semilogy(x, [r["glue_p"] for r in rows_out], color=c1, lw=1.6); ax[5].axhline(0.05, color=ink, lw=0.7)
    ax[5].set_title("Gluability p (loop-free model, side×topic×outcome)", fontsize=8.5, loc="left", color=ink)
    ax[5].xaxis.set_major_locator(mdates.YearLocator()); ax[5].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(); fig.savefig(os.path.join(here, "layers_chart.pdf")); fig.savefig(os.path.join(here, "layers_chart.png"), dpi=150)
