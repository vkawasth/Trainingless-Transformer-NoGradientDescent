"""Do probability-layer geometry quantities predict outcome flips beyond the toggle radius?

A stream is a count array C[fine bin, side (3), y (2)]. Sliding windows of W fine bins, step 1, are summarised by
  O_t        the binary outcome of the window (dataset-specific functional f >= 0)
  r_t        toggle radius in probability units: |f| / se (distance to the outcome wall)
  gamma2_t   curvature (Efron, e-connection, bootstrap bias-corrected) of the path of the 6-cell law (side x y) over 6
             sub-bins of the window, at its middle
  KL_t       holonomy information of flat transport on the (sub-bin x side) grid with cells y, i.e. side-specific
             change of the outcome rate within the window; reported as its z against resampled flat grids
  prolate_t  |z| of a Slepian (prolate) step test at the window midpoint on the fine-bin series of f
Target: flip_t = [O_{t+W} != O_t] (the next non-overlapping window). Model: logistic regression (IRLS, closed-form
steps) of flip on rank-standardised features; base = r only, full = r + gamma2 + KL + prolate. The likelihood-ratio
gain of the full model is calibrated by block permutation: the extra features are circularly shifted, jointly, by
a random offset >= W within each stream unit (keeps their autocorrelation and mutual correlation).
Datasets:
  planted   AllSides design (real article counts per week and side), simulated outcomes at base rate 0.15:
            (null) the centre wanders slowly near the midpoint, flips are noise; (ramp) a persistent state +-0.06 off
            the wall, moved to the other side by rare events (1/40 weeks) linearly over 12 weeks; (step) the same
            events as jumps. Four replicate streams each.
  AllSides  directed fact-checks of Trump, f = 2 P_C - P_L - P_R (Section sec:stream); 26-week windows.
  polls     2016 state polls (FL, NC, OH, NV, IA, AZ, GA and national), sides = mode (internet / live phone /
            IVR-automated), y = Clinton vs Trump among two-party respondents, f = share - 1/2; 12-week windows.
Env: ALLSIDES_DIR, POLLS2016_CSV.
"""
import os, sys, json, datetime
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../allsides")]
import numpy as np
import pandas as pd
from scipy import stats
from amb_vigneaux import infogeo2 as G
from amb_vigneaux.spectral import slepian_step_test

rng = np.random.default_rng(17)


# ------------------------------------------------------------------------------------------- outcome functionals
def f_allsides(T):
    """T[side, y] with y = 0 directed, 1 not; f = 2 P_C - P_L - P_R"""
    n = T.sum(1); P = T[:, 0] / np.maximum(n, 1)
    f = 2 * P[1] - P[0] - P[2]
    se = np.sqrt(4 * P[1] * (1 - P[1]) / max(n[1], 1) + sum(P[s] * (1 - P[s]) / max(n[s], 1) for s in (0, 2)))
    return f, max(se, 1e-6)


def f_polls(T):
    n = T.sum(); s = T[:, 0].sum() / max(n, 1)
    return s - 0.5, max(np.sqrt(s * (1 - s) / max(n, 1)) * 2.0, 1e-6)            # design effect 2 (house effects)


# ------------------------------------------------------------------------------------------- window features
def features(C, W, fn, B_hol=60, B_curv=30):
    nT = C.shape[0]; rows = []
    for t in range(nT - W + 1):
        X = C[t:t + W]; T = X.sum(0); f, se = fn(T)
        sub = np.array([s.sum(0) for s in np.array_split(X, 6)])               # 6 x side x y
        tab = sub.reshape(6, -1); svec = np.arange(6.0)
        g = [G.local_curvatures(tab, svec, i, half=2, B=B_curv, rng=rng)["gamma2_e"] for i in (2, 3)] if tab.sum(1).min() > 0 else [np.nan]
        h = G.holonomy_info(sub, B=B_hol, rng=rng)
        fs = np.array([fn(X[i])[0] if X[i].sum() > 0 else np.nan for i in range(W)])
        fs = np.where(np.isnan(fs), np.nanmean(fs) if np.isfinite(np.nanmean(fs)) else 0.0, fs)
        try:
            pz = abs(slepian_step_test(fs, W // 2, NW=1)["z"])
        except Exception:
            pz = 0.0
        g = [v for v in g if np.isfinite(v)] or [0.0]
        rows.append(dict(t=t, f=float(f), se=float(se), O=int(f >= 0), r=float(abs(f) / se), gamma2=float(np.mean(g)),
                         KL=float(h["z"]) if np.isfinite(h["z"]) else 0.0, prolate=float(pz if np.isfinite(pz) else 0.0), n=int(T.sum())))
    for i, r in enumerate(rows):
        r["flip"] = int(rows[i + W]["O"] != r["O"]) if i + W < len(rows) else None
    return rows


# ------------------------------------------------------------------------------------------- logistic + permutation
def rankz(x):
    r = stats.rankdata(x); return (r - r.mean()) / r.std() if r.std() > 0 else r * 0


def logit_irls(X, y, ridge=1e-3, iters=100):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); w = np.maximum(p * (1 - p), 1e-9)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(len(b)); g = X.T @ (y - p) - ridge * b
        step = np.linalg.solve(H, g); b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ b))
    return b, float(np.sum(y * np.log(np.maximum(p, 1e-12)) + (1 - y) * np.log(np.maximum(1 - p, 1e-12))))


def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float((stats.rankdata(np.concatenate([pos, neg]))[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


EXTRA = ("gamma2", "KL", "prolate")


def analyse(units, W, B=500):
    """units: list of row lists (one per stream unit). Pools units; block permutation within units."""
    D = [[r for r in rows if r["flip"] is not None] for rows in units]
    y = np.array([r["flip"] for rows in D for r in rows], float)
    def design(rowsets, extra=True):
        cols = [np.concatenate([rankz([-r["r"] for r in rows]) for rows in rowsets])]          # small r -> large
        if extra:
            cols += [np.concatenate([rankz([r[k] for r in rows]) for rows in rowsets]) for k in EXTRA]
        return np.column_stack([np.ones(len(y))] + cols)
    b0, l0 = logit_irls(design(D, False), y); b1, l1 = logit_irls(design(D), y); lr = 2 * (l1 - l0)
    null = []
    for _ in range(B):
        perm = []
        for rows in D:
            n = len(rows); k = int(rng.integers(min(W, n - 1), max(n - min(W, n - 1), min(W, n - 1) + 1))) if n > 2 * W else int(rng.integers(1, max(n, 2)))
            sh = [dict(r) for r in rows]
            for key in EXTRA:
                vals = np.roll([r[key] for r in rows], k)
                for s, v in zip(sh, vals):
                    s[key] = v
            perm.append(sh)
        null.append(2 * (logit_irls(design(perm), y)[1] - l0))
    null = np.array(null)
    au = {k: auc(np.concatenate([np.array([(-1 if k == "r" else 1) * r[k] for r in rows]) for rows in D]), y) for k in ("r",) + EXTRA}
    return dict(n=int(len(y)), flips=int(y.sum()), lr_extra=float(lr), p_perm=float((1 + (null >= lr).sum()) / (B + 1)), null_mean=float(null.mean()),
                coef_full=dict(zip(("const", "r") + EXTRA, b1.tolist())), auc=au)


# ------------------------------------------------------------------------------------------- data
def allsides_counts():
    import load
    GROUPS = ["elections", "politics", "foreign", "justice", "immigration", "media", "economy", "healthcare"]
    R = [r for r in load.load() if r["group"] in GROUPS]
    t0 = datetime.date(2015, 12, 14); end = max(r["date"] for r in R); nW = (end - t0).days // 7 + 1
    C = np.zeros((nW, 3, 2)); S = ("left", "center", "right")
    for r in R:
        w = (r["date"] - t0).days // 7
        if 0 <= w < nW:
            C[w, S.index(r["side"]), 0 if r["outcome_dir"] else 1] += 1
    return C


def planted_counts(Creal, kind, base=0.15, scale=1):
    """centre rate = midpoint + d_t / 2, article counts = real AllSides counts x scale.
    null : d wanders slowly near the wall (flips are noise).
    ramp : a persistent state d = +-0.06; rare events (1 per 52 weeks) move it to the other side linearly over 24
           weeks, so a flip's cause is visible in the window before it.
    step : the same events as jumps (the cause is not visible in advance)."""
    n = Creal.sum(2) * scale; nT = len(n); d = np.zeros(nT); x = 0.0; core = 0.06 * rng.choice([-1, 1]); slope = 0.0; left = 0
    for t in range(nT):
        x = 0.9 * x + rng.normal(0, 0.004)
        if kind == "null":
            core = 0.98 * core + rng.normal(0, 0.006)
        elif left == 0 and rng.random() < 1 / 52:
            target = -np.sign(core) * 0.06
            if kind == "step":
                core = target
            else:
                slope = (target - core) / 24; left = 24
        if left > 0:
            core += slope; left -= 1
        d[t] = core + x
    PL, PR = base, base * 1.1
    PC = np.clip((PL + PR) / 2 + d / 2, 0.001, 0.9)
    C = np.zeros_like(Creal, dtype=float)
    for t in range(nT):
        for s_, P in enumerate((PL, PC[t], PR)):
            k = rng.binomial(int(n[t, s_]), P); C[t, s_] = (k, n[t, s_] - k)
    return C


def polls_units(W=12):
    path = os.environ.get("POLLS2016_CSV", "/tmp/ds/econ/data/all_polls.csv")
    d = pd.read_csv(path); d = d[d.population.isin(["Likely Voters", "Registered Voters", "Adults"])].copy()
    mode = {"Internet": 0, "Live Phone": 1, "IVR/Online": 2, "Automated Phone": 2, "IVR/Live Phone": 2}
    d = d[d["mode"].isin(mode)]; d["side"] = d["mode"].map(mode)
    d["end"] = pd.to_datetime(d["end.date"]); d = d[d.end >= "2016-01-01"]
    d["n"] = d["number.of.observations"].fillna(d["number.of.observations"].median())
    t0 = pd.Timestamp("2016-01-04"); nW = (pd.Timestamp("2016-11-08") - t0).days // 7 + 1; units = {}
    for st in ("FL", "NC", "OH", "NV", "IA", "AZ", "GA", "--"):
        s = d[d.state == st]; C = np.zeros((nW, 3, 2))
        for _, r in s.iterrows():
            w = (r.end - t0).days // 7; tot = r.clinton + r.trump
            if 0 <= w < nW and tot > 0:
                m = r.n * tot / 100; C[w, int(r.side)] += (m * r.clinton / tot, m * r.trump / tot)
        units[st] = C
    return units


# ------------------------------------------------------------------------------------------- main
def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, green, grey, red = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#b8b6ae", "#d6402a"
    names = list(OUT["analysis"]); fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7)
    x = np.arange(len(names)); w = 0.2
    for j, (k, col) in enumerate((("r", ink), ("gamma2", blue), ("KL", orange), ("prolate", green))):
        ax[0].bar(x + (j - 1.5) * w, [OUT["analysis"][n]["auc"][k] for n in names], w, color=col, label={"r": "small toggle radius", "gamma2": "curvature γ²", "KL": "holonomy (KL z)", "prolate": "prolate step |z|"}[k])
    ax[0].axhline(0.5, color=grey, lw=0.8); ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=7); ax[0].set_ylim(0.2, 1.0)
    ax[0].set_ylabel("AUC for the next-window flip", fontsize=7.5, color=muted); ax[0].legend(fontsize=6, frameon=False, ncol=2)
    ax[0].set_title("(a) what predicts a flip, one feature at a time", fontsize=8, loc="left")
    ax[1].bar(x, [OUT["analysis"][n]["lr_extra"] for n in names], 0.5, color=blue, label="LR gain of γ², KL, prolate over r")
    ax[1].scatter(x, [OUT["analysis"][n]["null_mean"] for n in names], color=ink, zorder=3, s=14, label="block-permutation null mean")
    for i, n in enumerate(names):
        ax[1].text(i, OUT["analysis"][n]["lr_extra"], f"p={OUT['analysis'][n]['p_perm']:.3f}", ha="center", va="bottom", fontsize=6.5, color=muted)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=7); ax[1].legend(fontsize=6, frameon=False)
    ax[1].set_title("(b) does geometry add to the toggle radius?", fontsize=8, loc="left")
    S = OUT["series"]["AllSides"]; t = np.arange(len(S))
    a = ax[2]; a.plot(t, [s["f"] / s["se"] for s in S], color=blue, lw=1.4, label="z = f / se")
    a.plot(t, [s["KL"] for s in S], color=orange, lw=1, label="holonomy z"); a.axhline(0, color=ink, lw=0.6)
    fl = [i for i, s in enumerate(S) if s["flip"] == 1]
    a.scatter(fl, [0] * len(fl), marker="|", color=red, s=60, label="flip in next window")
    a.set_xlabel("window (2-week steps from Dec 2015)", fontsize=7.5, color=muted); a.legend(fontsize=6, frameon=False)
    a.set_title("(c) AllSides: the outcome lives on its wall", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(here, "flips_chart.pdf")); fig.savefig(os.path.join(here, "flips_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    OUT = dict(analysis={}, series={})
    Cw = allsides_counts()
    C2 = np.array([Cw[i:i + 2].sum(0) for i in range(0, len(Cw) - 1, 2)])            # 2-week fine bins; window 13 = 26 weeks
    for kind, scale in (("null", 10), ("ramp", 1), ("ramp", 10), ("step", 10)):
        rows_all = []
        for rep in range(4):
            Cp = planted_counts(Cw, kind, scale=scale); Cp2 = np.array([Cp[i:i + 2].sum(0) for i in range(0, len(Cp) - 1, 2)])
            rows_all.append(features(Cp2, 13, f_allsides))
        res = analyse(rows_all, 13); OUT["analysis"][f"{kind} x{scale}"] = res
        print(f"planted {kind:5s} x{scale:<2d}: windows {res['n']} flips {res['flips']} | AUC " + " ".join(f"{k} {v:.2f}" for k, v in res["auc"].items())
              + f" | LR gain {res['lr_extra']:.1f} (null mean {res['null_mean']:.1f}, p {res['p_perm']:.3f})", flush=True)
    rows = features(C2, 13, f_allsides); res = analyse([rows], 13); OUT["analysis"]["AllSides"] = res; OUT["series"]["AllSides"] = rows
    print(f"AllSides     : windows {res['n']} flips {res['flips']} | AUC " + " ".join(f"{k} {v:.2f}" for k, v in res["auc"].items())
          + f" | LR gain {res['lr_extra']:.1f} (null mean {res['null_mean']:.1f}, p {res['p_perm']:.3f})", flush=True)
    U = polls_units(); prow = {st: features(C, 12, f_polls) for st, C in U.items()}
    res = analyse(list(prow.values()), 12); OUT["analysis"]["2016 polls"] = res; OUT["series"]["polls"] = prow
    print(f"2016 polls   : windows {res['n']} flips {res['flips']} | AUC " + " ".join(f"{k} {v:.2f}" for k, v in res["auc"].items())
          + f" | LR gain {res['lr_extra']:.1f} (null mean {res['null_mean']:.1f}, p {res['p_perm']:.3f})")
    OUT = json.loads(json.dumps(OUT, default=float)); json.dump(OUT, open(os.path.join(here, "results.json"), "w")); plot(OUT)
