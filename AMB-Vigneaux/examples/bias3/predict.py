"""Prediction from the three-level decomposition: does the loop term (selective bias) help forecast
where gluing failed, and not where it held?

Models (nested; logistic / multinomial logistic with the same L2 penalty, one-hot features):
  M0 base       intercept only (per-source for BASIL is M1)
  M1 lean       source
  M2 lean+topic source + topic                         (additive: bias is a coboundary)
  M3 +loop      source + topic + source x topic         (adds the loop residual h)
  M4 +trend     M3 + source x year                      (BASIL only: drifting lean)
Scoring: out-of-sample log loss (nats per prediction) and Brier score; lower is better. Differences are
reported with a paired bootstrap over test units.

(A) BASIL stance, ROLLING ORIGIN: train on all years < Y, predict year Y (Y = 2013..2019).
    Target: article relative stance (5 classes). Topic = focal party of the event.
(B) BASIL span favourability, ROLLING ORIGIN: target = span positive?; topic = party of the target.
(C) MBIC outlet x topic, 10-fold CV over SENTENCES: target = one annotator's label (biased?).
"""
import json, glob, collections
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

import bias3, bias3_spans

rng = np.random.default_rng(0)
C_REG = 1.0


def design(df, spec):
    cols = []
    for term in spec:
        if isinstance(term, tuple):
            cols.append(df[list(term)].astype(str).agg("|".join, axis=1).rename("x".join(term)))
        else:
            cols.append(df[term].astype(str))
    return pd.concat(cols, axis=1) if cols else None


def fit_predict(train, test, spec, target, numeric=()):
    ytr = train[target].to_numpy()
    classes = np.unique(ytr)
    if not spec and not numeric:
        freq = np.array([(ytr == c).mean() for c in classes])
        P = np.tile(freq, (len(test), 1)); return classes, P
    parts_tr, parts_te = [], []
    if spec:
        enc = OneHotEncoder(handle_unknown="ignore")
        parts_tr.append(enc.fit_transform(design(train, spec)).toarray())
        parts_te.append(enc.transform(design(test, spec)).toarray())
    for f, by in numeric:                          # source-specific slope on a centred numeric variable
        mu = train[f].mean()
        for lev in sorted(train[by].unique()):
            parts_tr.append(((train[f] - mu) * (train[by] == lev)).to_numpy()[:, None] / 5.0)
            parts_te.append(((test[f] - mu) * (test[by] == lev)).to_numpy()[:, None] / 5.0)
    Xtr, Xte = np.hstack(parts_tr), np.hstack(parts_te)
    m = LogisticRegression(C=C_REG, max_iter=2000)
    m.fit(Xtr, ytr)
    return m.classes_, m.predict_proba(Xte)


def score(classes, P, y):
    idx = np.array([list(classes).index(v) if v in classes else -1 for v in y])
    p = np.where(idx >= 0, P[np.arange(len(y)), np.maximum(idx, 0)], 1e-6)
    ll = -np.log(np.clip(p, 1e-6, 1))
    onehot = np.zeros_like(P); ok = idx >= 0; onehot[np.arange(len(y))[ok], idx[ok]] = 1
    brier = ((P - onehot) ** 2).sum(1)
    return ll, brier


def paired_boot(a, b, n=5000):
    d = a - b; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run(name, df, target, splits, models):
    per = {m: [] for m in models}; brier = {m: [] for m in models}
    for tr, te in splits:
        for mname, (spec, num) in models.items():
            cl, P = fit_predict(df.loc[tr], df.loc[te], spec, target, num)
            ll, br = score(cl, P, df.loc[te, target].to_numpy())
            per[mname].append(ll); brier[mname].append(br)
    per = {m: np.concatenate(v) for m, v in per.items()}; brier = {m: np.concatenate(v) for m, v in brier.items()}
    names = list(models)
    out = {"n_test": int(len(per[names[0]])), "logloss": {m: float(v.mean()) for m, v in per.items()},
           "brier": {m: float(v.mean()) for m, v in brier.items()}, "gain": {}}
    print(f"\n== {name}  ({out['n_test']} test predictions)")
    for m in names:
        print(f"   {m:14s} log loss {out['logloss'][m]:.4f}   Brier {out['brier'][m]:.4f}")
    for a, b in zip(names[:-1], names[1:]):
        g = paired_boot(per[a], per[b])            # positive = model b better
        out["gain"][f"{a} -> {b}"] = g
        print(f"   gain {a:>12s} -> {b:<12s}: {g[0]:+.4f} nats/pred  95% CI [{g[1]:+.4f}, {g[2]:+.4f}]")
    return out


if __name__ == "__main__":
    R = {}
    # (A) BASIL stance
    E = bias3.load()
    rows = []
    for e in E:
        for s in ("fox", "nyt", "hpo"):
            rows.append(dict(event=e["id"], year=e["year"], source=s, party=e["camp"], theme=e["theme"], y=e[s]))
    A = pd.DataFrame(rows)
    splits = [(A.index[A.year < Y], A.index[A.year == Y]) for Y in range(2013, 2020)]
    models = {"M1 lean": (["source"], ()), "M2 +topic": (["source", "party"], ()),
              "M3 +loop": (["source", "party", ("source", "party")], ()),
              "M4 +trend": (["source", "party", ("source", "party")], (("year", "source"),))}
    R["A_basil_stance"] = run("(A) BASIL stance, rolling origin 2013-2019, topic = focal party", A, "y", splits, models)
    # (B) BASIL spans
    sp = bias3_spans.load_spans()
    yr = {}
    for f in glob.glob(f"{bias3.BASIL}/articles/*/*.json"):
        d = json.load(open(f)); yr[d["triplet-uuid"]] = int(d["date"][:4])
    B = pd.DataFrame([dict(event=e, year=yr[e], source=s, party=c, theme=th, y=int(pol == "pos"))
                      for e, s, w, th, c, pol in sp])
    splits = [(B.index[B.year < Y], B.index[B.year == Y]) for Y in range(2013, 2020)]
    R["B_basil_spans"] = run("(B) BASIL span favourability, rolling origin 2013-2019, topic = target party",
                             B, "y", splits, models)
    # (C) MBIC
    import mbic
    d = mbic.load()
    Cdf = pd.DataFrame(dict(sent=d["sentence_id"], source=d["outlet"], topic=d["topic"], y=d["b"].astype(int)))
    sents = Cdf["sent"].unique(); rng.shuffle(sents); folds = np.array_split(sents, 10)
    splits = [(Cdf.index[~Cdf.sent.isin(f)], Cdf.index[Cdf.sent.isin(f)]) for f in folds]
    modelsC = {"M0 base": ([], ()), "M1 lean": (["source"], ()), "M2 +topic": (["source", "topic"], ()),
               "M3 +loop": (["source", "topic", ("source", "topic")], ())}
    R["C_mbic"] = run("(C) MBIC outlet x topic, 10-fold CV over sentences", Cdf, "y", splits, modelsC)
    json.dump(R, open("predict_results.json", "w"), indent=1)
