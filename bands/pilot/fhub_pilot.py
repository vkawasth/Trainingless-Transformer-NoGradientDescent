"""Forecast Hub pilot: probability bands that are revised as information arrives, walls, tear time, unreliable data.

Data: COVIDhub-ensemble weekly state-level incident-death quantile forecasts, forecast dates 2021-06-07 .. 2022-04-04,
horizons 1-4 weeks; truth-file vintages of daily state deaths (JHU via the hub) as they stood on each forecast date.
Final truth = the last state-level vintage in the repository (2022-05-09). Targets ending after 2022-03-26 are dropped
so that every target has at least six weeks of later revisions.
"""
import json
import numpy as np, pandas as pd

fc = pd.read_csv("data/fc_state_incdeath.csv", dtype={"location": str}, parse_dates=["forecast_date", "target_end_date"])
fc["h"] = fc.target.str.extract(r"^(\d)").astype(int)
tv = pd.read_csv("data/truth_vintages_incdeath.csv", dtype={"location": str}, parse_dates=["date"])
tv = tv[tv.location != "US"]

def weekly(v):
    """Daily values -> epiweeks ending Saturday."""
    v = v.copy(); v["wk"] = v.date + pd.to_timedelta((5 - v.date.dt.weekday) % 7, unit="D")
    return v.groupby(["location", "wk"]).value.sum()

final_v = tv.vintage.max(); final = weekly(tv[tv.vintage == final_v]).rename("final")
LAST_TARGET = pd.Timestamp("2022-03-26")
fc = fc[fc.target_end_date <= LAST_TARGET]
fc = fc[fc.location.isin(final.index.get_level_values(0).unique())]

q = fc.pivot_table(index=["location", "target_end_date", "h", "forecast_date"], columns="quantile", values="value").reset_index()
q = q.rename(columns={0.1: "lo80", 0.9: "hi80", 0.25: "lo50", 0.75: "hi50", 0.5: "med"})
q = q.join(final, on=["location", "target_end_date"])
q = q.dropna(subset=["final"])
out = {"n_forecasts": int(len(q)), "states": int(q.location.nunique()), "target_weeks": int(q.target_end_date.nunique()),
       "final_vintage": str(final_v)}

# ---------------------------------------------------------------- 1. coverage of the bands at each horizon
q["in80"] = (q.final >= q.lo80) & (q.final <= q.hi80); q["in50"] = (q.final >= q.lo50) & (q.final <= q.hi50)
out["coverage80_by_h"] = q.groupby("h").in80.mean().round(3).to_dict(); out["coverage50_by_h"] = q.groupby("h").in50.mean().round(3).to_dict()

# ---------------------------------------------------------------- 2. band dynamics for a fixed target: h = 4 -> 3 -> 2 -> 1
q = q.sort_values(["location", "target_end_date", "h"], ascending=[True, True, False])
g = q.groupby(["location", "target_end_date"])
q["width80"] = q.hi80 - q.lo80
q["prev_width"] = g.width80.shift(1); q["prev_med"] = g.med.shift(1)
q["prev_lo"] = g.lo80.shift(1); q["prev_hi"] = g.hi80.shift(1)
step = q.dropna(subset=["prev_width"]).copy()
step = step[step.prev_width > 0]
step["width_ratio"] = step.width80 / step.prev_width
step["dilation"] = step.width_ratio > 1.0
# genuine dilation in the strict sense: the new band is not contained in the old one and is wider
step["escapes_old_band"] = (step.lo80 < step.prev_lo) | (step.hi80 > step.prev_hi)
out["revision_steps"] = int(len(step))
out["fraction_bands_wider_after_update"] = round(float(step.dilation.mean()), 3)
out["fraction_new_band_not_inside_old"] = round(float(step.escapes_old_band.mean()), 3)
out["median_width_ratio_by_h"] = step.groupby("h").width_ratio.median().round(3).to_dict()

# ---------------------------------------------------------------- 3. walls, margins, speeds and tear time
# wall for each state: its median final weekly deaths over the study period ("above typical" vs "below typical")
per = final.reset_index(); per = per[(per.wk >= "2021-06-12") & (per.wk <= LAST_TARGET)]
wall = per.groupby("location").final.median().rename("wall")
q = q.join(wall, on="location")
def side(lo, hi, w): return np.where(lo > w, "above", np.where(hi < w, "below", "straddles"))
q["side"] = side(q.lo80, q.hi80, q.wall)
q["margin"] = np.where(q.side == "above", q.lo80 - q.wall, np.where(q.side == "below", q.wall - q.hi80, 0.0))
q["prev_side"] = q.groupby(["location", "target_end_date"]).side.shift(1)
step = q.dropna(subset=["prev_med"]).copy()
step["speed"] = (step.med - step.prev_med).abs()                         # deaths per week of horizon
# tear time computed from the PREVIOUS forecast and the speed observed up to it: use the previous step's speed
q["speed"] = (q.med - q.groupby(["location", "target_end_date"]).med.shift(1)).abs()
q["tau"] = np.where(q.speed > 0, q.margin / q.speed, np.inf)            # weeks of horizon until the band reaches the wall
q["next_side"] = q.groupby(["location", "target_end_date"]).side.shift(-1)
dec = q[(q.side != "straddles") & q.next_side.notna() & np.isfinite(q.tau)].copy()
dec["tears_next"] = dec.next_side != dec.side                            # leaves its side at the next update
dec["flips_next"] = ((dec.side == "above") & (dec.next_side == "below")) | ((dec.side == "below") & (dec.next_side == "above"))
bins = pd.cut(dec.tau, [0, 0.5, 1, 2, 4, np.inf], right=False)
tab = dec.groupby(bins, observed=True).agg(n=("tears_next", "size"), tear_rate=("tears_next", "mean"), flip_rate=("flips_next", "mean")).round(3)
tab.index = tab.index.astype(str); out["tear_rate_by_tau"] = tab.to_dict("index")
# a scale-free comparison: tau against the margin alone as a predictor of tearing (rank AUC)
def auc(score, y):
    s = pd.Series(score).rank(); y = np.asarray(y, bool); n1, n0 = y.sum(), (~y).sum()
    return float((s[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
rel_margin = dec.margin / dec.wall.clip(lower=1)
out["auc_tear_from_small_tau"] = round(auc(-dec.tau, dec.tears_next), 3)
out["auc_tear_from_small_relative_margin"] = round(auc(-rel_margin, dec.tears_next), 3)
out["decisive_bands"] = int(len(dec)); out["tear_rate_overall"] = round(float(dec.tears_next.mean()), 3)
# outcome: when the band is decisive at horizon 1, is the final value on that side of the wall?
h1 = q[(q.h == 1) & (q.side != "straddles")]
out["h1_decisive_correct_side"] = round(float(((h1.side == "above") == (h1.final > h1.wall)).mean()), 3)
h4 = q[(q.h == 4) & (q.side != "straddles")]
out["h4_decisive_correct_side"] = round(float(((h4.side == "above") == (h4.final > h4.wall)).mean()), 3)

# ---------------------------------------------------------------- 4. unreliable information: backfill of the truth itself
first = []
for v, sub in tv.groupby("vintage"):
    w = weekly(sub).reset_index(); vdate = pd.Timestamp(v)
    last_full = w[w.wk <= vdate - pd.Timedelta(days=2)].groupby("location").wk.max()   # most recent complete week
    w = w.join(last_full.rename("last_wk"), on="location"); w = w[w.wk == w.last_wk]
    w["vintage"] = vdate; first.append(w[["location", "wk", "value", "vintage"]])
first = pd.concat(first).join(final, on=["location", "wk"]).dropna()
first = first[first.wk <= LAST_TARGET]
first["rel_rev"] = (first.final - first.value) / first.final.clip(lower=1)
out["backfill"] = {"state_weeks": int(len(first)),
                   "frac_revised_more_than_10pct": round(float((first.rel_rev.abs() > 0.10).mean()), 3),
                   "frac_revised_more_than_25pct": round(float((first.rel_rev.abs() > 0.25).mean()), 3),
                   "median_abs_rel_revision": round(float(first.rel_rev.abs().median()), 3),
                   "frac_revised_up": round(float((first.rel_rev > 0).mean()), 3)}
# do forecasts made on badly revised data miss more? join the revision of the last observed week to 1-week-ahead forecasts
q1 = q[q.h == 1].copy(); q1["vintage"] = q1.forecast_date
q1 = q1.merge(first[["location", "vintage", "rel_rev"]], on=["location", "vintage"], how="left").dropna(subset=["rel_rev"])
q1["bad_data"] = q1.rel_rev.abs() > 0.25
out["coverage80_h1_by_data_reliability"] = {"revised_le_25pct": round(float(q1[~q1.bad_data].in80.mean()), 3),
                                            "revised_gt_25pct": round(float(q1[q1.bad_data].in80.mean()), 3),
                                            "n_bad": int(q1.bad_data.sum()), "n_good": int((~q1.bad_data).sum())}
from math import sqrt
def wilson_ci(k, n, z=1.96):
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; h = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]
out["coverage80_h1_by_data_reliability"]["ci_bad"] = wilson_ci(int(q1[q1.bad_data].in80.sum()), int(q1.bad_data.sum()))
out["forecast_dates_used"] = [str(q.forecast_date.min().date()), str(q.forecast_date.max().date())]
alld = q[(q.side != "straddles") & q.next_side.notna()]
out["decisive_with_next"] = int(len(alld)); out["jumps_across_wall"] = int((((alld.side == "above") & (alld.next_side == "below")) | ((alld.side == "below") & (alld.next_side == "above"))).sum())
out["tau_sample_excluded_h4"] = int((alld.h == 4).sum()); out["tau_sample_excluded_zero_speed"] = int(((alld.h < 4) & ~np.isfinite(alld.tau)).sum())
out["n_locations"] = int(q.location.nunique())
q.to_csv("out/fhub_bands.csv", index=False); first.to_csv("out/fhub_backfill.csv", index=False)
json.dump(out, open("out/fhub_summary.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
