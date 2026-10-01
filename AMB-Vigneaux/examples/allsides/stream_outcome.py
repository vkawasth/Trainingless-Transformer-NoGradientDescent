"""Gluability -> outcome map on streaming corpora, and what would have flipped the outcome.

Stream: AllSides Trump coverage, sliding 26-week windows every 2 weeks, June 2015 - July 2020 (corpus 1 = window
before an event, corpus 2 = window after).
Binary outcome (probability layer):  O = [f >= 0],  f = 2 P_C - P_L - P_R,  P_s = share of side s's Trump articles
with a DIRECTED fact-check sentence ("Trump falsely said ..."). O = yes: the centre fact-checks Trump at least as much
as the midpoint of left and right (it sits with the left); no: it sits with the right.
Gluability of the window: does the side x topic x outcome table fit the glued (no-three-way, loop-free) model?
Parametric-bootstrap p of G^2 (sparse cells), 200 draws.
Counterfactual ("what more could have been done"): the smallest number of centre articles that would flip O,
  relabel:  k_rel = ceil(|f| n_C / 2)  existing centre articles change label;
  add:      k_add = extra directed centre articles needed, (x_C + k)/(n_C + k) >= m, m = (P_L + P_R)/2.
AMB layer: the possibilistic outcome 'a directed fact-check of Trump is possible for the centre on topic t'
(count > 0) before / after each event, and the exact evidence needed to make it impossible under the glued model
(co-facial set, strata.py) versus the saturated model (the cell alone).
Data via ALLSIDES_DIR.
"""
import os, sys, json, math, datetime
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), here]
import numpy as np
from scipy import stats
import load
from amb_vigneaux.strata import ipf, design, exact_toggle

SIDES = ("left", "center", "right")
GROUPS = ["elections", "politics", "foreign", "justice", "immigration", "media", "economy", "healthcare"]
MARG = [(0, 1), (0, 2), (1, 2)]
EVENTS = {"election 2016": "2016-11-08", "inauguration": "2017-01-20", "COVID emergency": "2020-03-13"}
rng = np.random.default_rng(0)


def table(rows, key="outcome_dir"):
    N = np.zeros((3, len(GROUPS), 2))
    for r in rows:
        if r["group"] in GROUPS:
            N[SIDES.index(r["side"]), GROUPS.index(r["group"]), 0 if r[key] else 1] += 1
    return N


def glue_p(N, B=200):
    live = N.sum(2) > 0
    M = ipf(N + 1e-9 * live[..., None], MARG)
    def g2(A, F):
        m = (A > 0) & (F > 0)
        return 2 * float((A[m] * np.log(A[m] / F[m])).sum())
    g0 = g2(N, M); n = int(N.sum()); p = (M / M.sum()).ravel(); null = []
    for _ in range(B):
        S = rng.multinomial(n, p).reshape(N.shape).astype(float)
        null.append(g2(S, ipf(S + 1e-9 * (S.sum(2) > 0)[..., None], MARG)))
    return float((1 + (np.array(null) >= g0).sum()) / (B + 1)), g0


def outcome(rows, key="outcome_dir"):
    x = {s: sum(r[key] for r in rows if r["side"] == s) for s in SIDES}
    n = {s: sum(1 for r in rows if r["side"] == s) for s in SIDES}
    P = {s: x[s] / n[s] for s in SIDES}
    f = 2 * P["center"] - P["left"] - P["right"]
    se = math.sqrt(4 * P["center"] * (1 - P["center"]) / n["center"] + sum(P[s] * (1 - P[s]) / n[s] for s in ("left", "right")))
    m = (P["left"] + P["right"]) / 2
    k_rel = math.ceil(abs(f) * n["center"] / 2)
    if f < 0:
        k_add = math.ceil(max(0.0, (m * n["center"] - x["center"]) / (1 - m)))
    else:                                              # to flip yes -> no, add non-directed centre articles
        k_add = math.ceil(max(0.0, x["center"] / m - n["center"])) if m > 0 else 0
    return dict(P=P, n=n, x=x, f=f, se=se, yes=bool(f >= 0), p_yes=float(stats.norm.cdf(f / se)) if se > 0 else float(f >= 0),
                k_relabel=k_rel, k_add=k_add)


def decompose(pre, post, key="outcome_dir"):
    """Delta f = RATE part (topic-specific rates changed, topic mix held at the average) + MIX part (topic mix changed,
    rates held at the average); Kitagawa decomposition per side, combined with the weights (-1, 2, -1)."""
    coef = {"left": -1.0, "center": 2.0, "right": -1.0}; rate = mix = 0.0
    for s in SIDES:
        def cell(rows):
            n = np.array([sum(1 for r in rows if r["side"] == s and r["group"] == g) for g in GROUPS], float)
            x = np.array([sum(r[key] for r in rows if r["side"] == s and r["group"] == g) for g in GROUPS], float)
            return n / n.sum(), np.where(n > 0, x / np.maximum(n, 1), 0.0)
        w0, p0 = cell(pre); w1, p1 = cell(post)
        rate += coef[s] * float(((w0 + w1) / 2 * (p1 - p0)).sum()); mix += coef[s] * float(((p0 + p1) / 2 * (w1 - w0)).sum())
    return dict(rate=rate, mix=mix)


def amb_centre(rows):
    N = table(rows); cen = {}
    live = N.sum(2) > 0
    M = ipf(N + 1e-9 * live[..., None], MARG); p = (M / M.sum()).ravel(); A, cells = design(N.shape, MARG)
    for j, g in enumerate(GROUPS):
        possible = N[1, j, 0] > 0
        entry = dict(possible=bool(possible), count=int(N[1, j, 0]), n=int(N[1, j].sum()))
        if possible:
            e = exact_toggle(A, p, cells.index((1, j, 0)))
            entry.update(saturated=float(N.sum() * p[cells.index((1, j, 0))]), glued=float(N.sum() * e["mass"]),
                         glued_set=len(e["forced"]))
        cen[g] = entry
    return cen


if __name__ == "__main__":
    R = [r for r in load.load() if r["group"] in GROUPS]
    W = datetime.timedelta(weeks=26); step = datetime.timedelta(weeks=2)
    t = datetime.date(2015, 12, 14); end = max(r["date"] for r in R)
    stream = []
    while t + W <= end + datetime.timedelta(days=1):
        rows = [r for r in R if t <= r["date"] < t + W]
        o = outcome(rows); gp, g2 = glue_p(table(rows), B=100)
        stream.append(dict(start=str(t), mid=str(t + W / 2), glue_p=gp, G2=g2, **{k: v for k, v in o.items() if k not in ("P", "n", "x")},
                           P=o["P"], n=o["n"]))
        t += step
    fs = np.array([s["f"] for s in stream]); d_f = np.diff(fs[::13]) if len(fs) > 13 else np.diff(fs)
    print(f"stream: {len(stream)} windows of 26 weeks, step 2 weeks;  outcome yes in {np.mean([s['yes'] for s in stream]):.2f} of windows;"
          f"  glued (p > 0.05) in {np.mean([s['glue_p'] > 0.05 for s in stream]):.2f}")
    print(f"sd of f between non-overlapping windows: {d_f.std():.4f};  windows whose outcome is decided (|f| > 1.96 se):"
          f" {np.mean([abs(s['f']) > 1.96 * s['se'] for s in stream]):.2f}")
    ev = {}
    for name, ds in EVENTS.items():
        tau = datetime.date.fromisoformat(ds)
        pre = [r for r in R if tau - W <= r["date"] < tau]; post = [r for r in R if tau <= r["date"] < min(tau + W, end + datetime.timedelta(days=1))]
        ob, oa = outcome(pre), outcome(post)
        gb, ga = glue_p(table(pre))[0], glue_p(table(post))[0]
        cb, ca = amb_centre(pre), amb_centre(post)
        z = (oa["f"] - ob["f"]) / math.sqrt(ob["se"] ** 2 + oa["se"] ** 2)
        dec = decompose(pre, post)
        ev[name] = dict(before=dict(ob, glue_p=gb, amb=cb), after=dict(oa, glue_p=ga, amb=ca), z_change=z, decomposition=dec)
        print(f"\n== {name} ({ds})")
        for lab, o, g in (("before", ob, gb), ("after", oa, ga)):
            print(f"   {lab:6s}: P(directed) L/C/R " + "/".join(f"{o['P'][s]:.3f}" for s in SIDES) + f"  f {o['f']:+.4f} (se {o['se']:.4f})"
                  f"  outcome {'YES' if o['yes'] else 'no '} (P(yes) {o['p_yes']:.2f})  gluability p {g:.2f}"
                  f"  | to flip: relabel {o['k_relabel']} centre articles, or add {o['k_add']} (of n_C = {o['n']['center']})")
        print(f"   change of f: {oa['f'] - ob['f']:+.4f} (z {z:+.2f});  outcome {'changed' if ob['yes'] != oa['yes'] else 'did not change'};"
              f"  of which topic-rate change {dec['rate']:+.4f}, topic-mix change {dec['mix']:+.4f}")
        flips = [g for g in GROUPS if cb[g]["possible"] != ca[g]["possible"]]
        print(f"   AMB (centre: directed fact-check possible on topic): before {[g for g in GROUPS if cb[g]['possible']]}"
              f"  after {[g for g in GROUPS if ca[g]['possible']]}  -> toggled: {flips or 'none'}")
        near = sorted([(v["glued"], g) for g, v in ca.items() if v["possible"]])[:3]
        print("   nearest AMB toggles after (articles that must vanish; glued model vs the cell alone): " +
              "  ".join(f"{g} {v:.0f} vs {ca[g]['saturated']:.1f}" for v, g in near))
    json.dump(dict(stream=stream, events=ev), open(os.path.join(here, "stream_outcome_results.json"), "w"), indent=1, default=float)

    # figure: three aligned panels (no dual axis)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
    x = [datetime.date.fromisoformat(s["mid"][:10]) for s in stream]
    f = np.array([s["f"] for s in stream]); se = np.array([s["se"] for s in stream])
    ink, muted, blue, orange = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=8); a.grid(axis="y", color="#e6e5e1", lw=0.6)
        for name, ds in EVENTS.items():
            a.axvline(datetime.date.fromisoformat(ds), color=muted, lw=0.8, ls=":")
    ax[0].fill_between(x, f - 1.96 * se, f + 1.96 * se, color=blue, alpha=0.18, lw=0)
    ax[0].plot(x, f, color=blue, lw=2)
    ax[0].axhline(0, color=ink, lw=0.8)
    yes = f >= 0
    for i in range(len(x) - 1):
        if yes[i]:
            ax[0].axvspan(x[i], x[i + 1], color=orange, alpha=0.10, lw=0)
    for name, ds in EVENTS.items():
        ax[0].text(datetime.date.fromisoformat(ds), ax[0].get_ylim()[1], " " + name, fontsize=7, color=muted, va="top", rotation=90)
    ax[0].set_ylabel("f = 2P_C − P_L − P_R\n(directed fact-checks)", fontsize=8, color=muted)
    ax[0].set_title("Centre alignment on fact-checking Trump (outcome yes = shaded, f ≥ 0)", fontsize=9, color=ink, loc="left")
    ax[1].plot(x, [s["glue_p"] for s in stream], color=blue, lw=2); ax[1].axhline(0.05, color=ink, lw=0.8)
    ax[1].set_yscale("log"); ax[1].set_ylabel("gluability p\n(loop-free model)", fontsize=8, color=muted)
    ax[2].plot(x, [s["k_add"] if not s["yes"] else 0 for s in stream], color=blue, lw=2)
    ax[2].set_ylabel("centre articles to add\nto flip to yes", fontsize=8, color=muted)
    ax[2].xaxis.set_major_locator(mdates.YearLocator()); ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(); fig.savefig(os.path.join(here, "stream_outcome.pdf")); fig.savefig(os.path.join(here, "stream_outcome.png"), dpi=150)
