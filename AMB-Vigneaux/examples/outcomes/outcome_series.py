"""An outcome series for the AllSides coverage stream: Trump approval (2017-2020) and the Trump-Clinton poll margin
(2015-2016), with a first lead-lag test against coverage.

Sources (GitHub-reachable mirrors, not redistributed):
  * FiveThirtyEight presidential approval poll list (approval_polllist.csv, model date 2020-08-31), mirrored in
    https://github.com/kiranrangaraj/Trump-Tweets-and-Approval-Rating-ETL (Resources/approval_polllist.csv); CC BY 4.0.
  * 2016 national general-election polls (HuffPost Pollster), in https://github.com/TheEconomist/us-potus-model
    (data/all_polls.csv).
Weekly series: net approval = mean of 538-adjusted (approve - disapprove) over 'All polls' polls ending that week,
weighted by the 538 poll weight; 2016 margin = mean (trump - clinton) over national likely/registered-voter polls.
Coverage series (AllSides, weekly): centre alignment f = 2P_C - P_L - P_R on the broad adversarial marker, and the
left-right adversarial gap. Lead-lag: correlation of week-to-week CHANGES at lags -8..+8 weeks; null by circular
shifts of the coverage series (preserves its autocorrelation). Event windows: outcome before (26 wk) vs after.
Env: APPROVAL_CSV, POLLS2016_CSV, ALLSIDES_DIR.
"""
import os, sys, json, datetime
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../allsides")]
import numpy as np, pandas as pd
import load

APP = os.environ.get("APPROVAL_CSV", "data/approval_polllist.csv")
P16 = os.environ.get("POLLS2016_CSV", "data/all_polls.csv")
W0 = datetime.date(2015, 6, 15)
wk = lambda d: (d - W0).days // 7
rng = np.random.default_rng(0)


def outcome_weekly():
    a = pd.read_csv(APP)
    a = a[(a.president == "Donald Trump") & (a.subgroup == "All polls")].copy()
    a["end"] = pd.to_datetime(a.enddate).dt.date; a["net"] = a.adjusted_approve - a.adjusted_disapprove
    a["w"] = a["weight"].fillna(1.0); a["week"] = a.end.map(wk)
    app = a.groupby("week").apply(lambda g: np.average(g.net, weights=g.w)).rename("net_approval")
    p = pd.read_csv(P16); p = p[(p.state == "--") & p.population.isin(["Likely Voters", "Registered Voters"])].copy()
    p["end"] = pd.to_datetime(p["end.date"]).dt.date; p["week"] = p.end.map(wk); p["margin"] = p.trump - p.clinton
    m16 = p.groupby("week").margin.mean().rename("margin_2016")
    return app, m16, dict(approval_polls=int(len(a)), approval_range=(str(a.end.min()), str(a.end.max())),
                          polls_2016=int(len(p)), polls_range=(str(p.end.min()), str(p.end.max())))


def coverage_weekly():
    R = load.load(); n = max(wk(r["date"]) for r in R) + 1
    num = np.zeros((3, n)); den = np.zeros((3, n)); S = ("left", "center", "right")
    for r in R:
        if r["date"] >= W0:
            i = S.index(r["side"]); w_ = wk(r["date"]); num[i, w_] += r["outcome"]; den[i, w_] += 1
    P = num / np.maximum(den, 1); ok = (den > 0).all(0)
    f = np.where(ok, 2 * P[1] - P[0] - P[2], np.nan); gap = np.where(ok, P[0] - P[2], np.nan)
    return pd.Series(f, name="f_centre"), pd.Series(gap, name="gap_LR")


def leadlag(cov, out, maxlag=8, B=2000):
    df = pd.concat([cov, out], axis=1).dropna()
    df = df.reindex(range(df.index.min(), df.index.max() + 1)).interpolate(limit=2).dropna()
    c, o = df.iloc[:, 0].diff().dropna(), df.iloc[:, 1].diff().dropna()
    idx = c.index.intersection(o.index); c, o = c[idx].values, o[idx].values
    def xc(x, y, k):          # k > 0: coverage leads outcome by k weeks
        return np.corrcoef(x[:len(x) - k], y[k:])[0, 1] if k >= 0 else np.corrcoef(x[-k:], y[:len(y) + k])[0, 1]
    lags = list(range(-maxlag, maxlag + 1)); r = np.array([xc(c, o, k) for k in lags])
    null = np.array([np.max(np.abs([xc(np.roll(c, int(rng.integers(12, len(c) - 12))), o, k) for k in lags])) for _ in range(B)])
    i = int(np.argmax(np.abs(r)))
    return dict(n_weeks=int(len(c)), best_lag=lags[i], r=float(r[i]), p_max=float((null >= abs(r[i])).mean()),
                r_by_lag=dict(zip(lags, r.round(3).tolist())))


if __name__ == "__main__":
    app, m16, info = outcome_weekly(); f, gap = coverage_weekly(); out = dict(sources=info)
    print(f"approval: {info['approval_polls']} 'All polls' polls, {info['approval_range'][0]} .. {info['approval_range'][1]}; "
          f"weekly points {app.notna().sum()}")
    print(f"2016 margin: {info['polls_2016']} national LV/RV polls, {info['polls_range'][0]} .. {info['polls_range'][1]}; weekly points {m16.notna().sum()}")
    for cname, cov in (("centre alignment f", f), ("left-right adversarial gap", gap)):
        for oname, o in (("net approval 2017-20", app), ("Trump-Clinton margin 2015-16", m16)):
            ll = leadlag(cov, o); out[f"{cname} vs {oname}"] = ll
            print(f"   {cname:28s} vs {oname:30s}: n {ll['n_weeks']:3d} weeks; strongest lag {ll['best_lag']:+d} wk"
                  f" (+ = coverage leads), r {ll['r']:+.3f}, max-over-lags p {ll['p_max']:.3f}")
    for name, ds, s in (("inauguration", "2017-01-20", app), ("Charlottesville", "2017-08-12", app), ("Mueller report", "2019-04-18", app),
                        ("impeachment inquiry", "2019-09-24", app), ("COVID emergency", "2020-03-13", app),
                        ("Access Hollywood tape", "2016-10-07", m16), ("Comey letter", "2016-10-28", m16)):
        t = wk(datetime.date.fromisoformat(ds))
        b = s[(s.index >= t - 26) & (s.index < t)]; a_ = s[(s.index >= t) & (s.index < t + (26 if name != "COVID emergency" else 19))]
        if name in ("Access Hollywood tape", "Comey letter"):
            b = s[(s.index >= t - 4) & (s.index < t)]; a_ = s[(s.index >= t) & (s.index < t + 2)]
        d = a_.mean() - b.mean(); se = np.sqrt(b.var(ddof=1) / len(b) + a_.var(ddof=1) / len(a_)) if len(b) > 1 and len(a_) > 1 else np.nan
        out[f"event {name}"] = dict(before=float(b.mean()), after=float(a_.mean()), delta=float(d), z=float(d / se) if se == se else None)
        print(f"   {name:22s}: outcome {b.mean():+.1f} -> {a_.mean():+.1f}  (change {d:+.1f} points, z {d / se:+.1f})")
    json.dump(out, open(os.path.join(here, "outcome_series_results.json"), "w"), indent=1, default=float)
