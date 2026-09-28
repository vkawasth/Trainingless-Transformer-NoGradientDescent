"""Corpus V: head + right-chain learner.  Reference points and EM runs.
usage: python vrun.py NTRAIN SWEEPS SEEDS [markov|pos]"""
import json, sys, time, numpy as np
from vtree import VGrammar, sample, chart_io
from hchain import *

NTR, SWEEPS, SEEDS = int(sys.argv[1]), int(sys.argv[2]), [int(s) for s in sys.argv[3].split(",")]
CLS = sys.argv[4] if len(sys.argv) > 4 else "markov"
D = json.load(open("cV/corpus.json"))
g = VGrammar.from_json(D["grammar"])
_, trees = sample(g, 20000, np.random.default_rng(1), return_trees=True)   # same draw as vcorpus.py
tr, va = D["train"][:NTR], D["val"][:500]
assert trees and len(trees[0]) == 2
L, V, A = 3, 8, 12
lr = LearnerPos(L, V, A) if CLS == "pos" else Learner(L, V, A)
gold = gold_counts_pos if CLS == "pos" else gold_counts
init = init_params_pos if CLS == "pos" else init_params
Btr, Bva = buckets(tr), buckets(va)
log = {"truth_val": float(np.mean([chart_io(g, s)[0] for s in va]))}
toks = np.concatenate([np.array(s) for s in tr])
uni = np.bincount(toks, minlength=A) / len(toks)
log["unigram_val"] = float(np.mean([np.log(uni[np.array(s)]).sum() for s in va]))
big = np.ones((A + 1, A)) * 0.5
for s in tr:
    prev = A
    for a in s:
        big[prev, a] += 1; prev = a
big /= big.sum(1, keepdims=True)
log["bigram_val"] = float(np.mean([sum(np.log(big[p, a]) for p, a in zip([A] + s[:-1], s)) for s in va]))
t0 = time.time()
Pg = gold(trees[:NTR], L, V, A)
log["gold_supervised_val"] = lr.loglik(Bva, Pg)
print(f"truth {log['truth_val']:.3f} | unigram {log['unigram_val']:.3f} | bigram {log['bigram_val']:.3f} | "
      f"head+chain from gold trees {log['gold_supervised_val']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
# EM started at the gold-supervised model: is it a fixed point?
P, traj = Pg, []
for it in range(3):
    ll, P = lr.em_step(Btr, P)
    traj.append((ll, lr.loglik(Bva, P)))
    print(f"  EM from gold, sweep {it+1}: train {ll:.3f}  val {traj[-1][1]:.3f}  [{time.time()-t0:.0f}s]", flush=True)
log["em_from_gold"] = traj
for seed in SEEDS:
    P = init(L, V, A, np.random.default_rng(seed))
    traj = []
    for it in range(SWEEPS):
        ll, P = lr.em_step(Btr, P)
        vl = lr.loglik(Bva, P) if (it + 1) % 5 == 0 or it == SWEEPS - 1 else None
        traj.append((ll, vl))
        print(f"  EM seed {seed}, sweep {it+1}: train {ll:.3f}" + (f"  val {vl:.3f}" if vl else "") + f"  [{time.time()-t0:.0f}s]", flush=True)
    log[f"em_seed{seed}"] = traj
    np.savez(f"cV/hchain_{CLS}_seed{seed}.npz", **P)
    json.dump(log, open(f"cV/vrun_{CLS}_N{NTR}.json", "w"), indent=1)
json.dump(log, open(f"cV/vrun_{CLS}_N{NTR}.json", "w"), indent=1)
