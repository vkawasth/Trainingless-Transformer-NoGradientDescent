"""Two Trump forecasts on BASIL.

(a) HPO favourability toward Trump. Target: is a directly aimed span at Trump positive? Rolling origin: for each
    test year Y in 2017..2019, fit on years < Y and forecast HPO's Trump spans in Y. Forecasters:
      global       positive rate of all spans (all sources, all targets) before Y
      source       HPO's positive rate over all its targets before Y
      source x target  HPO's positive rate at Trump before Y (Jeffreys Beta(1/2,1/2) prior)
    Score: log loss on the HPO Trump spans of year Y; plus P(0 positives | forecast) as a calibration check.
(b) Fox criticising Trump. Unit: a Fox article. y = 1 if it has >= 1 directly aimed NEGATIVE span at Trump.
    Features: trump_event (Trump among any article's main entities for this event), year >= 2016,
    other outlets' criticism on the SAME event (number of negative Trump spans in NYT and HPO; cross-source coupling),
    event focal party. Leave-one-event-out CV (and a rolling-origin variant). Log loss, Brier, AUC.
"""
import json, glob, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import beta as beta_dist

import bias3
from event_labels import triplet_labels

BASIL = "/tmp/claude-0/BASIL"
T = {"Donald_Trump", "Trump"}
TL = triplet_labels()


def load():
    arts = []
    for a in glob.glob(f"{BASIL}/annotations/*/*_ann.json"):
        A = json.load(open(a)); d = json.load(open(a.replace("/annotations/", "/articles/").replace("_ann.json", ".json")))
        sp = [(p["target"], p["polarity"]) for p in A["phrase-level-annotations"] if p["aim"] == "dir"]
        arts.append(dict(event=d["triplet-uuid"], source=d["source"].lower(), year=int(d["date"][:4]),
                         ents=" ".join(d["main-entities"]), spans=sp))
    return arts


def part_a(arts):
    rows = [(a["source"], a["year"], t in T, pol == "pos") for a in arts for t, pol in a["spans"]]
    out = {}
    print("(a) HPO favourability toward Trump, rolling origin")
    for Y in (2017, 2018, 2019):
        past = [r for r in rows if r[1] < Y]
        test = [r for r in rows if r[1] == Y and r[0] == "hpo" and r[2]]
        n, k = len(test), sum(r[3] for r in test)
        g = np.mean([r[3] for r in past])
        s = np.mean([r[3] for r in past if r[0] == "hpo"])
        hp = [r[3] for r in past if r[0] == "hpo" and r[2]]
        st = (sum(hp) + 0.5) / (len(hp) + 1.0)
        res = {}
        for name, p in (("global", g), ("source", s), ("source x target", st)):
            ll = -(k * np.log(p) + (n - k) * np.log(1 - p)) / n
            res[name] = dict(p=float(p), logloss=float(ll), p_zero=float((1 - p) ** n))
        out[Y] = dict(n=n, positives=int(k), models=res, hpo_trump_history=f"{int(sum(hp))}/{len(hp)}")
        print(f"   {Y}: observed {k}/{n} positive   (HPO-Trump history before {Y}: {int(sum(hp))}/{len(hp)})")
        for name, v in res.items():
            print(f"        {name:16s} forecast p = {v['p']:.3f}  log loss {v['logloss']:.3f}  P(0 of {n}) = {v['p_zero']:.3f}")
    return out


def part_b(arts):
    by_ev = collections.defaultdict(dict)
    for a in arts:
        by_ev[a["event"]][a["source"]] = a
    X, y, years, evs = [], [], [], []
    for e, d in by_ev.items():
        if "fox" not in d:
            continue
        fox = d["fox"]
        trump_event = any("Trump" in d[s]["ents"] for s in d)
        neg = {s: sum(1 for t, pol in d[s]["spans"] if t in T and pol == "neg") for s in d}
        camp = TL[e][1]
        X.append([float(trump_event), float(fox["year"] >= 2016), np.log1p(neg.get("nyt", 0)), np.log1p(neg.get("hpo", 0)),
                  float(camp == "R"), float(camp == "D")])
        y.append(int(neg["fox"] > 0)); years.append(fox["year"]); evs.append(e)
    X, y, years = np.array(X), np.array(y), np.array(years)
    feats = {"M0 base rate": [], "M1 trump in event": [0], "M2 + era": [0, 1],
             "M3 + NYT/HPO criticism (same event)": [0, 1, 2, 3], "M4 + focal party": [0, 1, 2, 3, 4, 5]}
    out = {"n_articles": int(len(y)), "positives": int(y.sum())}
    print(f"\n(b) Fox article criticises Trump: {y.sum()} of {len(y)} Fox articles")
    for split in ("leave-one-event-out", "rolling origin 2016-2019"):
        print(f"   {split}")
        out[split] = {}
        if split.startswith("leave"):
            folds = [(np.arange(len(y)) != i, np.arange(len(y)) == i) for i in range(len(y))]
        else:
            folds = [(years < Y, years == Y) for Y in range(2016, 2020)]
        P = {m: np.full(len(y), np.nan) for m in feats}
        for tr, te in folds:
            for m, cols in feats.items():
                if not cols or y[tr].min() == y[tr].max():
                    P[m][te] = (y[tr].sum() + 0.5) / (tr.sum() + 1.0)
                else:
                    clf = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr][:, cols], y[tr])
                    P[m][te] = clf.predict_proba(X[te][:, cols])[:, 1]
        mask = ~np.isnan(P["M0 base rate"])
        for m in feats:
            p = np.clip(P[m][mask], 1e-4, 1 - 1e-4); t = y[mask]
            ll = float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()); br = float(((p - t) ** 2).mean())
            auc = float(roc_auc_score(t, p)) if 0 < t.sum() < len(t) else float("nan")
            out[split][m] = dict(logloss=ll, brier=br, auc=auc, n=int(mask.sum()))
            print(f"      {m:38s} log loss {ll:.3f}  Brier {br:.3f}  AUC {auc:.3f}  (n={int(mask.sum())})")
    # coefficients of M3 on all data (direction of the cross-source coupling)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(X[:, [0, 1, 2, 3]], y)
    out["M3_coefficients"] = dict(zip(["trump_event", "era>=2016", "log1p NYT neg", "log1p HPO neg"], clf.coef_[0].round(3).tolist()))
    print("   M3 coefficients (all data):", out["M3_coefficients"])
    # conditional rates for the coupling
    te = X[:, 0] == 1
    for lab, m in (("NYT criticises Trump", X[:, 2] > 0), ("NYT does not", X[:, 2] == 0)):
        sel = te & m
        print(f"   among Trump events, {lab}: Fox criticises in {y[sel].sum()}/{sel.sum()}")
    return out


if __name__ == "__main__":
    arts = load()
    R = {"a": part_a(arts), "b": part_b(arts)}
    json.dump(R, open("trump_predict_results.json", "w"), indent=1, default=str)
