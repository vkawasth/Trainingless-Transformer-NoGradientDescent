"""Does tree-structured pooling of the loop term help prediction? (the usable part of the ultrametric idea)
Instead of a free outlet x topic loop (few observations per cell), pool it one level up the outlet tree:
outlet -> outlet type (left / center / right). M3t = source + topic + type x topic."""
import numpy as np, pandas as pd, warnings, glob, json
warnings.filterwarnings("ignore")
import predict, mbic, bias3, bias3_spans
rng = np.random.default_rng(2)

def cv(df, key, k=10):
    u = df[key].unique().copy(); rng.shuffle(u); f = np.array_split(u, k)
    return [(df.index[~df[key].isin(x)], df.index[df[key].isin(x)]) for x in f]

def compare(df, target, splits, models, base):
    L = {m: [] for m in models}
    for tr, te in splits:
        y = df.loc[te, target].to_numpy()
        for m, spec in models.items():
            c, P = predict.fit_predict(df.loc[tr], df.loc[te], spec, target)
            L[m].append(predict.score(c, P, y)[0])
    L = {m: np.concatenate(v) for m, v in L.items()}
    for m in models:
        g = predict.paired_boot(L[base], L[m], n=2000)
        print(f"   {m:18s} log loss {L[m].mean():.4f}   gain vs {base}: {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]")

d = mbic.load()
C = pd.DataFrame(dict(sent=d.sentence_id, source=d.outlet, type=d["type"], topic=d.topic, y=d.b.astype(int)))
print("MBIC, 10-fold CV by sentence")
compare(C, "y", cv(C, "sent"), {
    "M2 additive": ["source", "topic"],
    "M3 free loop": ["source", "topic", ("source", "topic")],
    "M3t tree loop": ["source", "topic", ("type", "topic")],
}, "M2 additive")
