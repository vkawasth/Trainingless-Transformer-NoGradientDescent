#!/usr/bin/env python3
"""BUILD_CORPUS3 -- LLM-LIKE: NON-UNIFORM WITHIN-SUPPORT DISTRIBUTIONS.

    python3 build_corpus3.py --out /tmp --report
    python3 build_corpus3.py --out /tmp

WHY THIS EXISTS
---------------
build_corpus2 samples successors UNIFORMLY over b(x), so the target conditional
is p*(y|x) = 1/b(x) on the support. That made two things true that are false of
natural text:

  1. the in-support "shape" deficit KL(u_S || p_S) is a meaningful error, and
  2. a loss term pushing p_S toward uniform is pushing toward the truth.

On real text the within-support distribution is heavily skewed -- "the" follows
far more often than "aardvark" -- so a uniformity penalty would push AWAY from
the target. The measured 96% shape share at step 700 may therefore be an
artifact of comparing against a uniform reference rather than a real deficiency.

This corpus keeps everything else and breaks only that assumption: each context
x has b(x) successors whose conditional probabilities follow a Zipf law rather
than being uniform, with the exponent itself varying across contexts so that
some are near-deterministic and some near-flat.

    p*(y_i | x) proportional to (i + 1)^(-s(x)),   i = 0 .. b(x)-1
    s(x) ~ Uniform[0.2, 2.0]      drawn independently of b(x) and freq(x)

Three quantities are now independent by construction: frequency, branching, and
within-support skew. That lets the three be separated as explanatory variables,
which neither earlier corpus allowed.

WHAT SHOULD CHANGE IF THE SHAPE FINDING IS AN ARTIFACT
------------------------------------------------------
The true in-support entropy is now H*(x) = -sum_i p*_i log p*_i < log b(x). A
model that correctly learns the skew will show:

    KL(p* || p_theta)          small       -- the real error
    KL(u_S || p_S)             LARGE       -- and correctly so, since the
                                              target is not uniform

So on this corpus the uniformity-based "shape" number should stay high even for
a good model, which would confirm it was measuring the reference and not the
model. The honest shape metric here is KL(p*_S || p_S), the in-support
divergence against the TRUE conditional, and that is what the meta file exports
so downstream scripts can compute it.

Train and validation are disjoint samples from the same process.
"""
import json, argparse, os, math, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp")
ap.add_argument("--train-tokens", type=int, default=368280)
ap.add_argument("--val-tokens", type=int, default=40920)
ap.add_argument("--vocab", type=int, default=1017)
ap.add_argument("--zipf", type=float, default=0.6, help="marginal frequency exponent")
ap.add_argument("--bmax", type=int, default=24)
ap.add_argument("--smin", type=float, default=0.2, help="min within-support skew")
ap.add_argument("--smax", type=float, default=2.0, help="max within-support skew")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--report", action="store_true")
a = ap.parse_args()

V = a.vocab
rng = np.random.default_rng(a.seed)

# marginal frequency: Zipf over a random permutation, as in build_corpus2
rank = rng.permutation(V)
freq_w = 1.0 / np.power(np.arange(1, V + 1), a.zipf)
freq_w = freq_w[np.argsort(rank)]
freq_w /= freq_w.sum()

# branching, independent of frequency
branch = rng.integers(1, a.bmax + 1, size=V)

# within-support skew, independent of BOTH
skew = rng.uniform(a.smin, a.smax, size=V)

succ, cond = [], []
for t in range(V):
    b = int(branch[t])
    cand = rng.choice(V, size=min(b * 4, V), replace=False, p=freq_w)[:b]
    order = rng.permutation(b)            # which successor gets the top mass
    w = np.power(np.arange(1, b + 1), -skew[t])
    w = w[order] / w.sum()
    succ.append(np.asarray(cand, dtype=np.int64))
    cond.append(w)

def generate(n, seed):
    r = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.int64)
    cur = int(r.choice(V, p=freq_w))
    for i in range(n):
        out[i] = cur
        s = succ[cur]
        cur = (int(r.choice(s, p=cond[cur])) if len(s)
               else int(r.choice(V, p=freq_w)))
    return out

train = generate(a.train_tokens, a.seed)
val = generate(a.val_tokens, a.seed + 99991)

cnt = collections.Counter(train.tolist())
f = np.array([cnt.get(t, 0) for t in range(V)], float)
m = f > 0
# true conditional entropy per context, and the floor
Hstar = np.array([float(-(c * np.log(c)).sum()) for c in cond])
P = f / f.sum()
floor = float((P * Hstar).sum())
uni_floor = float((P * np.log(np.maximum(branch, 1))).sum())

r_fb = float(np.corrcoef(np.log1p(f[m]), np.log(branch[m]))[0, 1])
r_fs = float(np.corrcoef(np.log1p(f[m]), skew[m])[0, 1])
r_bs = float(np.corrcoef(np.log(branch[m]), skew[m])[0, 1])

print(f"  vocab {V}   train {len(train):,}   val {len(val):,}")
print(f"  corr(log freq, log branch) = {r_fb:+.4f}")
print(f"  corr(log freq, skew)       = {r_fs:+.4f}")
print(f"  corr(log branch, skew)     = {r_bs:+.4f}   <- all three independent")
print(f"  skew s(x) range {a.smin}-{a.smax}, branching {branch.min()}-{branch.max()}")
print(f"  TRUE conditional floor  E[H*(x)]      = {floor:.4f} nats")
print(f"  uniform-support floor   E[ln b(x)]    = {uni_floor:.4f} nats")
print(f"  gap (what a uniformity penalty would wrongly demand) = "
      f"{uni_floor - floor:.4f} nats")
if max(abs(r_fb), abs(r_fs), abs(r_bs)) > 0.15:
    print(f"  WARNING: a pair is correlated above 0.15")

if a.report:
    raise SystemExit(0)

os.makedirs(a.out, exist_ok=True)
json.dump([f"tok{i}" for i in range(V)], open(f"{a.out}/vocab.json", "w"))
json.dump(train.tolist(), open(f"{a.out}/train_ids.json", "w"))
json.dump(val.tolist(), open(f"{a.out}/val_ids.json", "w"))
json.dump({"mode": "llm_like",
           "corr_freq_branch_generative": r_fb,
           "corr_freq_skew": r_fs, "corr_branch_skew": r_bs,
           "H_true": floor, "H_uniform_support": uni_floor,
           "branch": branch.tolist(), "skew": skew.tolist(),
           "succ": [s.tolist() for s in succ],
           "cond": [c.tolist() for c in cond]},
          open(f"{a.out}/corpus_meta.json", "w"))
print(f"  wrote {a.out}/{{vocab,train_ids,val_ids,corpus_meta}}.json")
print(f"  corpus_meta carries succ/cond so downstream can compute the HONEST")
print(f"  in-support divergence KL(p*_S || p_S) instead of KL(u_S || p_S).")
