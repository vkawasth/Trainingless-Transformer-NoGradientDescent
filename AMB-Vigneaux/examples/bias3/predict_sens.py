"""Sensitivity of the loop-term gain to regularisation (C) and to the split (rolling origin vs random CV)."""
import json, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import predict, bias3, bias3_spans, mbic
rng = np.random.default_rng(1)

def gain(df, target, splits, specA, specB):
    la, lb = [], []
    for tr, te in splits:
        ca, Pa = predict.fit_predict(df.loc[tr], df.loc[te], specA, target)
        cb, Pb = predict.fit_predict(df.loc[tr], df.loc[te], specB, target)
        y = df.loc[te, target].to_numpy()
        la.append(predict.score(ca, Pa, y)[0]); lb.append(predict.score(cb, Pb, y)[0])
    return predict.paired_boot(np.concatenate(la), np.concatenate(lb), n=2000)

def cluster_folds(df, key, k=10):
    u = df[key].unique().copy(); rng.shuffle(u); f = np.array_split(u, k)
    return [(df.index[~df[key].isin(x)], df.index[df[key].isin(x)]) for x in f]

sp = bias3_spans.load_spans(); yr = {}
for f in glob.glob(f"{bias3.BASIL}/articles/*/*.json"):
    d = json.load(open(f)); yr[d["triplet-uuid"]] = int(d["date"][:4])
B = pd.DataFrame([dict(event=e, year=yr[e], source=s, party=c, y=int(pol == "pos")) for e, s, w, th, c, pol in sp])
B2 = B[B.source.isin(["fox", "hpo"])].reset_index(drop=True)          # the pair whose loop failed to glue
d = mbic.load()
Cdf = pd.DataFrame(dict(sent=d["sentence_id"], source=d["outlet"], topic=d["topic"], y=d["b"].astype(int)))
add = ["source", "party"]; loop = ["source", "party", ("source", "party")]
addC = ["source", "topic"]; loopC = ["source", "topic", ("source", "topic")]
out = {}
for C in (0.1, 0.3, 1.0, 3.0, 10.0):
    predict.C_REG = C
    r = {}
    r["spans all, rolling"] = gain(B, "y", [(B.index[B.year < Y], B.index[B.year == Y]) for Y in range(2013, 2020)], add, loop)
    r["spans all, event-CV"] = gain(B, "y", cluster_folds(B, "event"), add, loop)
    r["spans Fox-HPO, event-CV"] = gain(B2, "y", cluster_folds(B2, "event"), add, loop)
    r["MBIC, sentence-CV"] = gain(Cdf, "y", cluster_folds(Cdf, "sent"), addC, loopC)
    out[C] = r
    print(f"C={C}: " + " | ".join(f"{k}: {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]" for k, v in r.items()), flush=True)
json.dump({str(k): v for k, v in out.items()}, open("predict_sens_results.json", "w"), indent=1)
