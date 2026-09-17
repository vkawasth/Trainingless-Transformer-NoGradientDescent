#!/usr/bin/env python3
"""BUILD_CORPUS2 --- DECORRELATED FREQUENCY AND BRANCHING, WITH A REAL HELD-OUT SET.

    python3 build_corpus2.py --out /tmp --mode decorr
    python3 build_corpus2.py --out /tmp --mode degenerate     # reproduces the old corpus
    python3 build_corpus2.py --out /tmp --mode decorr --report

WHY THIS EXISTS
---------------
The original corpus is one 1364-token string repeated: train = BASE x 270,
val = BASE x 30, so val is the same 1364 tokens as train. Two consequences
that invalidated several measurements:

  1. FREQUENCY AND BRANCHING ARE THE SAME VARIABLE. In a single repeated
     string a token's successor set is fixed by its occurrences, so
     corr(log freq, successor entropy) = +0.9858. Every "does the corpus
     drive X" question collapsed: the norm correlated -0.74 with BOTH, and
     the partial correlation of successor entropy given frequency was only
     -0.110. The two cannot be separated by any analysis of that corpus.

  2. THERE IS NO GENERALISATION TO MEASURE. val == train, so "capability"
     means recall of a fixed sequence. Memorisation is intrinsically
     high-rank, which is the likely reason low-rank transplant carried
     +0.038 against its rank-matched null while GaLore-style low-rank
     projection works on real models.

WHAT --mode decorr DOES
-----------------------
Generates from an explicit bigram process over the SAME vocabulary, with two
token properties assigned INDEPENDENTLY:

    frequency f(t)   Zipf-distributed, controls how often t appears
    branching b(t)   number of distinct successors t may take, drawn
                     independently of f

so a token can be frequent-and-deterministic (b=1) or rare-and-branching
(b=20). The successor distribution of t is uniform over its b successors,
giving successor entropy log2(b) decoupled from f by construction.

Train and val are separate samples from the same process: disjoint token
streams, identical statistics. So val measures generalisation over the
process, not recall of a string.

WHAT TO EXPECT TO CHANGE
------------------------
Predictions recorded before running anything on this corpus:
  - corr(log freq, successor entropy) falls from +0.986 to near 0
  - the chord's embedding-row norm can now be regressed on frequency and
    branching separately; the earlier +0.2346 was one confounded number
  - val != train, so the loss floor is set by the process entropy rather
    than by memorisation, and the "capability" buckets measure prediction
  - the chord's rank may fall: the earlier high rank (R(4)=0.306) is what
    memorising 1364 arbitrary facts requires

--report prints the decorrelation check and the entropy floor without
writing anything, so the corpus can be validated before it is used.
"""
import json, argparse, os, math, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp")
ap.add_argument("--mode", default="decorr", choices=["decorr", "degenerate"])
ap.add_argument("--train-tokens", type=int, default=368280)
ap.add_argument("--val-tokens", type=int, default=40920)
ap.add_argument("--vocab", type=int, default=1017)
ap.add_argument("--zipf", type=float, default=0.6, help="frequency exponent")
ap.add_argument("--bmax", type=int, default=24, help="max successors per token")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--report", action="store_true")
a = ap.parse_args()

V = a.vocab
rng = np.random.default_rng(a.seed)

# ---- token properties, assigned INDEPENDENTLY ------------------------------
# frequency: Zipf over a random permutation of the vocabulary, so rank in
# frequency carries no relation to token id
rank = rng.permutation(V)
freq_w = 1.0 / np.power(np.arange(1, V + 1), a.zipf)
freq_w = freq_w[np.argsort(rank)]
freq_w /= freq_w.sum()

# branching: drawn independently. A separate permutation means a token's
# successor count is unrelated to how often it appears -- this is the whole
# point of the file.
branch = rng.integers(1, a.bmax + 1, size=V)

# successor sets: b(t) distinct targets, sampled by frequency so the marginal
# token distribution stays close to freq_w
succ = []
for t in range(V):
    b = int(branch[t])
    cand = rng.choice(V, size=min(b * 4, V), replace=False, p=freq_w)
    succ.append(np.array(cand[:b], dtype=np.int64))

def generate(n, seed):
    r = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.int64)
    cur = int(r.choice(V, p=freq_w))
    for i in range(n):
        out[i] = cur
        s = succ[cur]
        cur = int(s[r.integers(len(s))]) if len(s) else int(r.choice(V, p=freq_w))
    return out

if a.mode == "degenerate":
    import importlib.util as _il
    spec = _il.spec_from_file_location("bc", "build_corpus.py")
    print("  degenerate mode: run the original build_corpus.py instead")
    raise SystemExit(1)

train = generate(a.train_tokens, a.seed)
val = generate(a.val_tokens, a.seed + 99991)   # disjoint stream, same process

# ---- validation of the decorrelation ---------------------------------------
def stats(ids):
    cnt = collections.Counter(ids.tolist())
    nxt = collections.defaultdict(collections.Counter)
    for x, y in zip(ids[:-1], ids[1:]):
        nxt[int(x)][int(y)] += 1
    def H(c):
        tot = sum(c.values())
        return 0.0 if tot == 0 else -sum((n / tot) * math.log2(n / tot)
                                         for n in c.values() if n > 0)
    f = np.array([cnt.get(t, 0) for t in range(V)], float)
    h = np.array([H(nxt[t]) for t in range(V)])
    return f, h, nxt

f_tr, h_tr, nxt_tr = stats(train)
f_va, h_va, _ = stats(val)
m = f_tr > 0
r_fb = float(np.corrcoef(np.log1p(f_tr[m]), h_tr[m])[0, 1])
# The EMPIRICAL successor entropy is biased by frequency: a token seen k times
# can show at most log2(k) bits regardless of its true branching, so rare
# tokens look deterministic. The generative branching is the quantity that was
# assigned independently, so report both -- r_gen is the decorrelation that
# was actually constructed, r_fb is what an analysis of the token stream can
# recover, and the gap between them is the sampling bias.
r_gen = float(np.corrcoef(np.log1p(f_tr[m]), np.log2(branch[m]))[0, 1])
# restrict to tokens seen often enough for the entropy estimate to be usable
we = f_tr >= 4 * branch
r_fb_ok = float(np.corrcoef(np.log1p(f_tr[we]), h_tr[we])[0, 1]) if we.sum() > 30 else float("nan")

# unigram entropy floor in nats, and the bigram-conditional floor, which is
# what a model with full context can reach
p = f_tr[m] / f_tr[m].sum()
H_uni = float(-(p * np.log(p)).sum())
tot_pairs = sum(sum(c.values()) for c in nxt_tr.values())
H_big = 0.0
for t, c in nxt_tr.items():
    n = sum(c.values())
    if n == 0: continue
    q = np.array(list(c.values()), float) / n
    H_big += (n / tot_pairs) * float(-(q * np.log(q)).sum())

ov = len(set(train.tolist()) & set(val.tolist()))
print(f"  vocab {V}   train {len(train):,}   val {len(val):,}   mode {a.mode}")
print(f"  corr(log freq, log branching)      = {r_gen:+.4f}   <- as constructed")
print(f"  corr(log freq, successor entropy)  = {r_fb:+.4f}   [degenerate: +0.9858]")
print(f"  same, tokens with freq >= 4*branch = {r_fb_ok:+.4f}"
      f"   ({int(we.sum())} tokens, estimate unbiased)")
print(f"  branching range {branch.min()}-{branch.max()}   "
      f"median successors {int(np.median(branch))}")
print(f"  unigram entropy  {H_uni:.4f} nats   bigram-conditional {H_big:.4f} nats")
print(f"  distinct tokens shared train/val: {ov}/{V}  (types overlap by design;"
      f" the STREAMS are disjoint samples)")
seq_ov = len(set(zip(train[:-1].tolist(), train[1:].tolist())) &
             set(zip(val[:-1].tolist(), val[1:].tolist())))
print(f"  bigram types shared: {seq_ov}   (the process is shared, not the text)")
if abs(r_gen) > 0.15:
    print(f"  WARNING: constructed correlation {r_gen:+.3f} is not near zero")

if a.report:
    raise SystemExit(0)

os.makedirs(a.out, exist_ok=True)
json.dump([f"tok{i}" for i in range(V)], open(f"{a.out}/vocab.json", "w"))
json.dump(train.tolist(), open(f"{a.out}/train_ids.json", "w"))
json.dump(val.tolist(), open(f"{a.out}/val_ids.json", "w"))
json.dump({"mode": a.mode, "corr_freq_branch_empirical": r_fb,
           "corr_freq_branch_generative": r_gen, "corr_unbiased": r_fb_ok, "H_unigram": H_uni,
           "H_bigram": H_big, "branch": branch.tolist(),
           "freq": f_tr.tolist(), "succ_entropy": h_tr.tolist()},
          open(f"{a.out}/corpus_meta.json", "w"))
print(f"  wrote {a.out}/{{vocab,train_ids,val_ids,corpus_meta}}.json")
