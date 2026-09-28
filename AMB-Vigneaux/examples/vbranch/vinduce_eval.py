"""Build an explicit production grammar from the induced chunks (root collapsed to one symbol,
per the O2 root-collapse result), score it exactly, and refine the rule probabilities by EM."""
import json, sys, time, collections, numpy as np
from multiprocessing import Pool
from vtree import VGrammar, chart_io, em_sweep, probs_of
from vchunks import induce, score_level1

NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
D = json.load(open("cV/corpus.json")); g_true = VGrammar.from_json(D["grammar"])
tr, va = D["train"][:NTR], D["val"][:500]
t0 = time.time()
lv = induce(tr)
print("level-1 score:", score_level1(lv, g_true), f"[{time.time()-t0:.0f}s]")

# production grammar from the induced structure
V = 8
rules = {}
for l in (1, 2, 3):
    segs, lab = lv[l - 1]
    cnt = collections.Counter()
    for sg in segs:
        for ch in sg:
            b = lab[ch] if l < 3 else 0                 # one root symbol
            cnt[(b, ch)] += 1
    tot = collections.Counter()
    for (b, ch), v in cnt.items():
        tot[b] += v
    rules[l] = [(b, ch, v / tot[b]) for (b, ch), v in sorted(cnt.items())]
prior = np.zeros(V); prior[0] = 1.0
g = VGrammar(3, V, 12, rules, prior)
print("induced productions per level:", {l: len(r) for l, r in rules.items()},
      " true:", {l: len(r) for l, r in g_true.rules.items()})

def val_ll(gr, probs=None):
    return float(np.mean([chart_io(gr, s, probs)[0] for s in va]))

res = {"truth": val_ll(g_true), "induced_counts": val_ll(g)}
print(f"truth {res['truth']:.3f} | induced grammar (Viterbi counts) {res['induced_counts']:.3f}  "
      f"unparsable val sentences: {sum(np.isinf(chart_io(g, s)[0]) for s in va)}")
probs = probs_of(g)
with Pool(2) as pool:
    for it in range(8):
        ll, probs = em_sweep(g, tr, probs, pool=pool)
        v = val_ll(g, probs)
        print(f"  EM sweep {it+1}: train {ll:.3f}  val {v:.3f}  [{time.time()-t0:.0f}s]", flush=True)
res["induced_em"] = v
# how the root inventory compares with the truth (up to relabelling of level-2 symbols)
json.dump(res, open(f"cV/induced_N{NTR}.json", "w"), indent=1)
