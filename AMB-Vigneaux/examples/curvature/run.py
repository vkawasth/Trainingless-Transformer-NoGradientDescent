"""Component [C]: curvature of the sponsorship bend on the 2016 polls (amb_vigneaux/curvature.py).

Data: POLLS2016_CSV (HuffPost Pollster export, all_polls.csv): Trump and Clinton shares, sample size, state (or
national), dates, mode, population, and the sponsor fields partisan (Nonpartisan / Pollster / Sponsor) and
affiliation (Dem / Rep). Kept: 2016 polls of Likely Voters, Registered Voters or Adults (party subgroups dropped),
affiliation Dem, Rep or none.

Baseline (closed-form WLS on the logit of the two-party Trump share): race (state or national) + two-week period +
mode + population; weights 1 / (1/(n p (1-p)) + tau^2), tau^2 the over-dispersion estimated from nonpartisan polls.
  [C1] velocity: sponsor shifts delta_D, delta_R (e-flat), in logit and in two-party margin points; pollster-cluster
       bootstrap SEs; split by 'Pollster' vs 'Sponsor' affiliation.
  [C2] curvature: (a) slope kappa of sponsored residuals on the race baseline (0 under the shift); (b) the pull model
       p -> (1-w) p + w q with q estimated (a closed-form grid over q, WLS for w), its fit against the shift; (c)
       Efron's gamma^2 of the fitted pull family at w = 0 over the sponsored polls (0 for the shift).
  [C3] singular point: among nonpartisan polls, EM for a hidden mixture of honest and bent (shifts delta_D, delta_R);
       LR against pi = 0, calibrated by parametric bootstrap under pi = 0.
Env: POLLS2016_CSV.
"""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np, pandas as pd
from amb_vigneaux.curvature import wls, efron_gamma2, pull_jets, mixture_lr

rng = np.random.default_rng(0)
PATH = os.environ.get("POLLS2016_CSV", "data/all_polls.csv")
logit = lambda p: np.log(p / (1 - p)); expit = lambda e: 1 / (1 + np.exp(-e))


def load():
    d = pd.read_csv(PATH)
    d["end"] = pd.to_datetime(d["end.date"])
    d = d[(d["end"] >= "2016-01-01") & d["population"].isin(["Likely Voters", "Registered Voters", "Adults"])]
    d = d[d["affiliation"].isna() | d["affiliation"].isin(["Dem", "Rep"])].copy()
    d["n"] = d["number.of.observations"].fillna(d["number.of.observations"].median()).clip(upper=5000)
    d["p"] = d["trump"] / (d["trump"] + d["clinton"])
    d["neff"] = d["n"] * (d["trump"] + d["clinton"]) / 100
    d["y"] = logit((d["p"] * d["neff"] + 0.5) / (d["neff"] + 1))
    d["race"] = d["state"].replace("--", "US")
    d["period"] = ((d["end"] - pd.Timestamp("2016-01-01")).dt.days // 14).astype(int)
    d["D"] = (d["affiliation"] == "Dem").astype(float); d["R"] = (d["affiliation"] == "Rep").astype(float)
    d["kind"] = d["partisan"]
    return d.reset_index(drop=True)


def design(d, sponsor=True, extra=None):
    cols = [pd.get_dummies(d["race"], prefix="r", dtype=float), pd.get_dummies(d["period"], prefix="t", drop_first=True, dtype=float),
            pd.get_dummies(d["mode"], prefix="m", drop_first=True, dtype=float), pd.get_dummies(d["population"], prefix="pop", drop_first=True, dtype=float)]
    X = pd.concat(cols, axis=1)
    if sponsor:
        X["D"] = d["D"]; X["R"] = d["R"]
    if extra is not None:
        for k, v in extra.items():
            X[k] = v
    return X


def weights(d, tau2):
    s = d["neff"] * d["p"] * (1 - d["p"])
    return 1.0 / (1.0 / s + tau2)


def fit_shift(d, tau2):
    X = design(d); w = weights(d, tau2).to_numpy()
    b, cov = wls(X.to_numpy(), d["y"].to_numpy(), w)
    return pd.Series(b, index=X.columns), X, w


def tau2_mom(d):
    """over-dispersion from nonpartisan polls: E[r^2] = 1/(n p q) + tau^2 under the baseline"""
    nd = d[(d.D == 0) & (d.R == 0)]
    tau2 = 0.0
    for _ in range(5):
        X = design(nd, sponsor=False); w = weights(nd, tau2).to_numpy()
        b, _ = wls(X.to_numpy(), nd["y"].to_numpy(), w)
        r = nd["y"].to_numpy() - X.to_numpy() @ b
        tau2 = max(0.0, float(np.mean(r ** 2 - 1.0 / (nd["neff"] * nd["p"] * (1 - nd["p"])).to_numpy())))
    return tau2


def cluster_boot(d, tau2, fn, B=200):
    pol = d["pollster"].unique(); out = []
    for _ in range(B):
        pick = rng.choice(pol, size=len(pol), replace=True)
        db = pd.concat([d[d.pollster == p] for p in pick], ignore_index=True)
        try:
            out.append(fn(db, tau2))
        except Exception:
            pass
    return np.array(out)


def c1(d, tau2):
    b, X, w = fit_shift(d, tau2)
    base = d["y"].to_numpy() - X[["D", "R"]].to_numpy() @ b[["D", "R"]].to_numpy()
    pbar = expit(np.median(base))
    def pts(delta):                               # logit shift -> two-party margin points at a 50-50 race and at the median race
        return 100 * (2 * expit(delta) - 1), 100 * 2 * (expit(logit(pbar) + delta) - pbar)
    bs = cluster_boot(d, tau2, lambda db, t: fit_shift(db, t)[0][["D", "R"]].to_numpy())
    se = bs.std(0)
    by_kind = {}
    for kind in ("Pollster", "Sponsor"):
        dk = d[(d.kind == "Nonpartisan") | (d.kind == kind)]
        bk, _, _ = fit_shift(dk, tau2); by_kind[kind] = dict(D=float(bk["D"]), R=float(bk["R"]), nD=int(dk.D.sum()), nR=int(dk.R.sum()))
    out = dict(delta_D=float(b["D"]), delta_R=float(b["R"]), se_D=float(se[0]), se_R=float(se[1]),
               margin_pts_D=pts(b["D"])[0], margin_pts_R=pts(b["R"])[0], n_D=int(d.D.sum()), n_R=int(d.R.sum()), by_kind=by_kind, tau2=tau2)
    print(f"[C1] velocity (first-order house effect; tau^2 {tau2:.4f}; {len(d)} polls, {d.pollster.nunique()} pollsters)")
    print(f"    Dem-affiliated ({out['n_D']}): delta {b['D']:+.4f} logit (cluster SE {se[0]:.4f}) = {out['margin_pts_D']:+.2f} margin pts")
    print(f"    Rep-affiliated ({out['n_R']}): delta {b['R']:+.4f} logit (cluster SE {se[1]:.4f}) = {out['margin_pts_R']:+.2f} margin pts")
    for k, v in by_kind.items():
        print(f"    {k}-affiliated only: delta_D {v['D']:+.4f} (n {v['nD']}), delta_R {v['R']:+.4f} (n {v['nR']})")
    return out, b, X, w


def c2(d, tau2, b, X, w):
    base_eta = X.drop(columns=["D", "R"]).to_numpy() @ b.drop(["D", "R"]).to_numpy()   # each poll's fitted baseline
    res = d["y"].to_numpy() - base_eta; out = {}
    print("\n[C2] curvature")
    for s in ("D", "R"):
        m = d[s].to_numpy() == 1; e0 = base_eta[m]; r = res[m]; ww = w[m]
        Z = np.column_stack([np.ones(m.sum()), e0 - np.average(e0, weights=ww)])
        beta, cov = wls(Z, r, ww)
        # the pull model: grid over the target q, closed-form w for each q
        best = None; p0 = expit(e0)
        for q in np.linspace(0.0, 1.0, 101):
            d1, _ = pull_jets(p0, q)
            if np.allclose(d1, 0):
                continue
            # first-order: r ~ w d1 ; exact: r = logit((1-w) p0 + w q) - e0, solve w by 1-D grid refinement
            wg = np.linspace(-0.5, 0.5, 401); sse = []
            for wv in wg:
                pm = np.clip((1 - wv) * p0 + wv * q, 1e-6, 1 - 1e-6); sse.append(np.sum(ww * (r - (logit(pm) - e0)) ** 2))
            k = int(np.argmin(sse))
            if best is None or sse[k] < best[0]:
                best = (sse[k], q, wg[k])
        sse_pull, q, wv = best
        delta = np.average(r, weights=ww); sse_shift = float(np.sum(ww * (r - delta) ** 2)); sse_null = float(np.sum(ww * r ** 2))
        d1, d2 = pull_jets(p0, q); I = d["neff"].to_numpy()[m] * p0 * (1 - p0)
        g2 = efron_gamma2(wv * d1, wv ** 2 * d2, I) if wv != 0 else np.nan
        g2_unit = efron_gamma2(d1, d2, I)
        bs = cluster_boot(d, tau2, lambda db, t, s=s: _kappa(db, t, s), B=200)
        out[s] = dict(kappa=float(beta[1]), kappa_se_model=float(np.sqrt(cov[1, 1])), kappa_se_cluster=float(np.nanstd(bs)),
                      q=float(q), w=float(wv), sse_null=sse_null, sse_shift=sse_shift, sse_pull=float(sse_pull),
                      gamma2=float(efron_gamma2(d1, d2, I)), n=int(m.sum()), baseline_range=[float(p0.min()), float(p0.max())])
        o = out[s]
        print(f"  {'Dem' if s == 'D' else 'Rep'}-affiliated (n {o['n']}, baseline Trump share {o['baseline_range'][0]:.2f}-{o['baseline_range'][1]:.2f}):")
        print(f"    kappa (slope of residual on baseline logit) {o['kappa']:+.3f} (model SE {o['kappa_se_model']:.3f}, cluster SE {o['kappa_se_cluster']:.3f})")
        print(f"    weighted SSE: none {sse_null:.1f} | shift (e-flat) {sse_shift:.1f} | pull to q={q:.2f}, w={wv:+.3f} (m-flat) {sse_pull:.1f}")
        print(f"    Efron curvature of the pull family at w=0: gamma^2 = {o['gamma2']:.3g} (shift: 0; 'large' if >= 1/8)")
    return out


def _kappa(db, tau2, s):
    b, X, w = fit_shift(db, tau2)
    e = X.drop(columns=["D", "R"]).to_numpy() @ b.drop(["D", "R"]).to_numpy(); r = db["y"].to_numpy() - e
    m = db[s].to_numpy() == 1
    if m.sum() < 10:
        return np.nan
    Z = np.column_stack([np.ones(m.sum()), e[m] - np.average(e[m], weights=w[m])])
    return wls(Z, r[m], w[m])[0][1]


def c3(d, tau2, b, X, w, B=300):
    base_eta = X.drop(columns=["D", "R"]).to_numpy() @ b.drop(["D", "R"]).to_numpy()
    m = (d.D == 0).to_numpy() & (d.R == 0).to_numpy()
    r = (d["y"].to_numpy() - base_eta)[m]; s2 = 1.0 / w[m]
    shifts = [float(b["D"]), float(b["R"])]
    pi, lr = mixture_lr(r, s2, shifts)
    null = []
    for _ in range(B):
        rs = rng.normal(0, np.sqrt(s2)); null.append(mixture_lr(rs, s2, shifts)[1])
    null = np.array(null); p = float((null >= lr).mean())
    out = dict(pi_D=float(pi[0]), pi_R=float(pi[1]), lr=lr, p_boot=p, null_frac_zero=float((null < 1e-6).mean()),
               null_q95=float(np.quantile(null, 0.95)), n=int(m.sum()))
    print(f"\n[C3] singular point: hidden sponsorship among {out['n']} nonpartisan polls (shifts fixed at delta_D, delta_R)")
    print(f"    EM: pi_D {pi[0]:.3f}, pi_R {pi[1]:.3f}; LR vs pi = 0: {lr:.2f}; parametric-bootstrap p {p:.3f}"
          f" (null: LR = 0 in {out['null_frac_zero']:.2f} of draws -- the boundary mass of a chi-bar-squared; 95% point {out['null_q95']:.2f})")
    return out


def plot(OUT, d=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#b8b6ae"
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7.5)
    C1 = OUT["C1"]; ks = [("D", "Dem-affiliated", blue), ("R", "Rep-affiliated", orange)]
    for i, (k, lab, col) in enumerate(ks):
        dl, se = C1[f"delta_{k}"], C1[f"se_{k}"]
        ax[0].errorbar([dl], [i], xerr=[[1.96 * se], [1.96 * se]], fmt="o", color=col, capsize=3)
        for j, kind in enumerate(("Pollster", "Sponsor")):
            ax[0].scatter([C1["by_kind"][kind][k]], [i + 0.2 * (j + 1)], color=col, marker="^v"[j], s=18, alpha=0.7)
    ax[0].axvline(0, color=muted, lw=0.8, ls=":"); ax[0].set_yticks([0, 1]); ax[0].set_yticklabels([k[1] for k in ks], fontsize=7.5)
    ax[0].set_xlabel("house effect on logit Trump share (95% cluster CI)", fontsize=8, color=muted)
    ax[0].set_title("(a) [C1] velocity: the first-order bend", fontsize=8.5, loc="left", color=ink)
    ax[0].text(0.02, 0.96, "▲ pollster-affiliated  ▼ sponsor-commissioned", transform=ax[0].transAxes, fontsize=6.5, color=muted)
    if "scatter" in OUT:
        for k, lab, col in ks:
            S = OUT["scatter"][k]; ax[1].scatter(S["p0"], S["r"], s=6, color=col, alpha=0.35, lw=0, label=lab)
            xs = np.linspace(min(S["p0"]), max(S["p0"]), 50); C2 = OUT["C2"][k]
            ax[1].plot(xs, np.full_like(xs, OUT["C1"][f"delta_{k}"]), color=col, lw=1.2, ls="--")
            pm = np.clip((1 - C2["w"]) * xs + C2["w"] * C2["q"], 1e-6, 1 - 1e-6); ax[1].plot(xs, logit(pm) - logit(xs), color=col, lw=1.8)
        ax[1].axhline(0, color=muted, lw=0.6, ls=":"); ax[1].set_ylim(-0.4, 0.4)
        ax[1].set_xlabel("race baseline (Trump two-party share)", fontsize=8, color=muted); ax[1].set_ylabel("sponsored residual (logit)", fontsize=8, color=muted)
        ax[1].set_title("(b) [C2] shift (dashed) vs pull (solid)", fontsize=8.5, loc="left", color=ink); ax[1].legend(fontsize=6.5, frameon=False, loc="upper right")
    C3 = OUT["C3"]
    if "null" in C3:
        ax[2].hist(C3["null"], bins=40, color=grey, alpha=0.8, label="LR under π = 0 (bootstrap)")
        ax[2].axvline(C3["lr"], color=orange, lw=2, label=f"observed LR {C3['lr']:.2f}")
        ax[2].legend(fontsize=6.5, frameon=False, loc="upper right")
    ax[2].set_xlabel("likelihood ratio, hidden-sponsor mixture vs none", fontsize=8, color=muted)
    ax[2].set_title("(c) [C3] the singular point π = 0", fontsize=8.5, loc="left", color=ink)
    fig.tight_layout(); fig.savefig(os.path.join(here, "curvature_chart.pdf")); fig.savefig(os.path.join(here, "curvature_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    d = load(); tau2 = tau2_mom(d)
    o1, b, X, w = c1(d, tau2)
    o2 = c2(d, tau2, b, X, w)
    o3 = c3(d, tau2, b, X, w)
    base_eta = X.drop(columns=["D", "R"]).to_numpy() @ b.drop(["D", "R"]).to_numpy(); res = d["y"].to_numpy() - base_eta
    scatter = {s: dict(p0=expit(base_eta[d[s] == 1]).tolist(), r=res[d[s] == 1].tolist()) for s in ("D", "R")}
    m = (d.D == 0).to_numpy() & (d.R == 0).to_numpy(); s2 = 1.0 / w[m]
    o3["null"] = [mixture_lr(rng.normal(0, np.sqrt(s2)), s2, [o1["delta_D"], o1["delta_R"]])[1] for _ in range(300)]
    OUT = dict(C1=o1, C2=o2, C3=o3, scatter=scatter)
    json.dump(OUT, open(os.path.join(here, "results.json"), "w"), indent=1, default=float)
    plot(OUT)
