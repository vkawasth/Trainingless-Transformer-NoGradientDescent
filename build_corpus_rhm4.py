#!/usr/bin/env python3
"""BUILD_CORPUS_RHM -- Random Hierarchy Model with an explicit symmetric /
asymmetric production split.

    python3 build_corpus_rhm.py --report
    python3 build_corpus_rhm.py --out /tmp

WHAT THIS IS FOR
----------------
A probabilistic context-free grammar: each level-l symbol expands into an
ORDERED pair of level-(l-1) symbols, recursively, down to observed tokens.
Groups of groups of groups, with the grouping known by construction. This is
the ground truth corpus2/corpus3 never had: they gave one known number (the
entropy floor), so the only available measurement was loss. Here the latent
symbol at every node is known, so "did the model recover the structure" is a
direct measurement rather than an inference from loss.

WHAT IS DESIGNED IN, AND SAID OUT LOUD
--------------------------------------
Standard RHM production rules are ordered tuples, so order is load-bearing
EVERYWHERE and the unordered control class would be empty. The symmetric rule
set here is therefore inserted deliberately:

  ASYM parent:  exactly one of (a,b) / (b,a) is a valid production of it
  SYM  parent:  BOTH (x,y) and (y,x) are productions of the same parent,
                so the two orders are semantically identical

This is designing the corpus to contain the distinction being tested for. That
is acceptable because the claim is about MECHANISM (does a 2-layer model encode
order-dependence where the grammar has it, and not where it doesn't), not about
language. No claim about natural language may be made from this file.

WHAT IT EXPORTS FOR THE HOLONOMY TEST
-------------------------------------
  C_ord    pairs (a,b) valid under an ASYM parent; (b,a) is not that parent's
  C_unord  pairs (x,y) where BOTH orders expand the same SYM parent
  C_perm   frequency-matched random pairs that are not productions at all
Each is a list of leaf-token pairs, to be read at a [PROBE] position.

C_unord is the POSITIONAL control: (x,y) and (y,x) differ in position
embedding regardless of grammar, so a purely positional order-sensitivity
appears in it identically. C_perm is the vocabulary/co-occurrence control.
They control different things and must be reported separately, never pooled.

TWO-COMPONENT LOSS READOUT
--------------------------
Every position is labelled by the context depth needed to predict it:
  local   second child of a level-1 pair: determined by its sibling
  higher  first child of a subtree: needs level>=2 context
Report CE separately on the two. A structural objective should move `higher`
and leave `local` alone; the global mean would hide that.
"""
import json, argparse, os, math, random, collections, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp")
ap.add_argument("--depth", type=int, default=4, help="L: levels of hierarchy")
ap.add_argument("--nsym", type=int, default=32, help="v: symbols per level")
ap.add_argument("--nrules", type=int, default=4, help="m: productions per symbol")
ap.add_argument("--nleaf", type=int, default=32, help="leaf token vocabulary")
ap.add_argument("--sym-frac", type=float, default=0.5,
                help="fraction of symbols at each level with SYMMETRIC rules")
ap.add_argument("--n-train", type=int, default=20000, help="training sequences")
ap.add_argument("--n-val", type=int, default=2000, help="held-out sequences")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--reverse-valid", action="store_true",
                help="every asymmetric production (x,y) under P1 has (y,x) as a "
                     "production of a DIFFERENT asymmetric parent P2")
ap.add_argument("--poset", action="store_true",
                help="leaf pairs are ordered by a genuine PARTIAL ORDER: "
                     "comparable pairs get asymmetric parents, incomparable "
                     "pairs get symmetric ones. Makes transitivity testable.")
ap.add_argument("--poset-density", type=float, default=0.12,
                help="edge probability of the DAG whose closure is the order")
ap.add_argument("--report", action="store_true")
a = ap.parse_args()

rng = random.Random(a.seed)
L, V, M = a.depth, a.nsym, a.nrules
SEQ = 2 ** L                      # leaf sequence length

# ---------------------------------------------------------------- grammar
# rules[l][i] = list of ordered child pairs for symbol i at level l.
# Level 1 children are LEAF tokens; levels 2..L children are level-(l-1) symbols.
# ------------------------------------------------------- partial order
# A random DAG on the LEAF vocabulary, closed under reachability. Its
# reachability relation is a genuine partial order: reflexive (trivially),
# antisymmetric (a DAG has no cycles) and TRANSITIVE (closure). Transitivity
# is the property the previous grammar could not have, because production
# rules were assigned independently -- (x,y) and (y,z) being rules implied
# nothing about (x,z). Here it does, and that is what makes the relation
# testable rather than merely "some pairs are ordered".
LEAF = a.nleaf
PREC = np.zeros((LEAF, LEAF), dtype=bool)
if a.poset:
    topo = list(range(LEAF)); rng.shuffle(topo)
    for ii in range(LEAF):
        for jj in range(ii + 1, LEAF):
            if rng.random() < a.poset_density:
                PREC[topo[ii], topo[jj]] = True
    for k in range(LEAF):                       # Warshall closure
        PREC |= np.outer(PREC[:, k], PREC[k, :])
    np.fill_diagonal(PREC, False)
    COMPARABLE = [(x, y) for x in range(LEAF) for y in range(LEAF)
                  if PREC[x, y]]                # x precedes y
    INCOMP = [(x, y) for x in range(LEAF) for y in range(x + 1, LEAF)
              if not PREC[x, y] and not PREC[y, x]]
    rng.shuffle(COMPARABLE); rng.shuffle(INCOMP)

rules, is_sym = {}, {}
for l in range(1, L + 1):
    child_n = a.nleaf if l == 1 else V
    rules[l], is_sym[l] = {}, {}
    n_sym_here = int(round(V * a.sym_frac))
    sym_ids = set(rng.sample(range(V), n_sym_here))
    asym_ids = [i for i in range(V) if i not in sym_ids]
    for i in range(V):
        is_sym[l][i] = (i in sym_ids)
        rules[l][i] = []

    # --- symmetric parents: M/2 base pairs, BOTH orders under the same parent
    #     with --poset at level 1 these are INCOMPARABLE pairs, so "order does
    #     not matter" is a fact about the relation, not a coin flip
    use_poset = a.poset and l == 1
    for i in sym_ids:
        for _ in range(max(1, M // 2)):
            if use_poset and INCOMP:
                x, y = INCOMP[rng.randrange(len(INCOMP))]
            else:
                x, y = rng.sample(range(child_n), 2)
            rules[l][i] += [(x, y), (y, x)]

    if a.reverse_valid:
        # --- asymmetric parents in COUPLES: (x,y) goes to P1, (y,x) to P2.
        # Both orders are then grammatical, just under different parents, so
        # the model cannot satisfy the ordered condition with a validity
        # detector -- it has to represent WHICH parent, i.e. the order.
        rng.shuffle(asym_ids)
        for k in range(0, len(asym_ids) - 1, 2):
            p1, p2 = asym_ids[k], asym_ids[k + 1]
            seen = set()
            guard = 0
            while len(rules[l][p1]) < M and guard < 10000:
                guard += 1
                if use_poset and COMPARABLE:
                    x, y = COMPARABLE[rng.randrange(len(COMPARABLE))]  # x prec y
                else:
                    x, y = rng.sample(range(child_n), 2)
                if (x, y) in seen or (y, x) in seen:
                    continue
                seen.add((x, y))
                rules[l][p1].append((x, y)); rules[l][p2].append((y, x))
        if len(asym_ids) % 2:                     # odd one out, unpaired
            i = asym_ids[-1]; seen = set()
            while len(rules[l][i]) < M:
                x, y = rng.sample(range(child_n), 2)
                if (x, y) in seen or (y, x) in seen:
                    continue
                seen.add((x, y)); rules[l][i].append((x, y))
    else:
        for i in asym_ids:
            seen = set()
            while len(rules[l][i]) < M:
                x, y = rng.sample(range(child_n), 2)
                if (y, x) in seen or (x, y) in seen:
                    continue
                seen.add((x, y)); rules[l][i].append((x, y))

def expand(sym, l):
    if l == 0:
        return [sym]
    x, y = rng.choice(rules[l][sym])
    return expand(x, l - 1) + expand(y, l - 1)

def sample_seq():
    root = rng.randrange(V)
    return expand(root, L), root

# ------------------------------------------------------- position labels
# leaf index t, 0-based. Odd t is the SECOND child of a level-1 pair, so it is
# determined by its sibling at t-1 plus the parent's rule choice -> `local`.
# Even t opens a new level-1 subtree and needs level>=2 context -> `higher`.
pos_kind = ["higher" if t % 2 == 0 else "local" for t in range(SEQ)]

# ------------------------------------------------------------- sampling
seen_train, train_seqs = set(), []
while len(train_seqs) < a.n_train:
    s, _ = sample_seq()
    train_seqs.append(s); seen_train.add(tuple(s))
val_seqs, tries = [], 0
while len(val_seqs) < a.n_val and tries < a.n_val * 200:
    tries += 1
    s, _ = sample_seq()
    if tuple(s) not in seen_train:          # held out: never seen in training
        val_seqs.append(s)
if len(val_seqs) < a.n_val:
    print(f"  WARNING only {len(val_seqs)} unseen val sequences available; "
          f"the grammar's support is small relative to --n-train. "
          f"Raise --nsym/--nrules/--depth or lower --n-train.")

train = np.array([t for s in train_seqs for t in s], dtype=np.int64)
val   = np.array([t for s in val_seqs   for t in s], dtype=np.int64)

# ------------------------------------------------- control pair extraction
# Pairs live at the LEAF level (level-1 productions), which is what a 2-layer
# model composes first.
prod_of = collections.defaultdict(set)        # (x,y) -> {parent symbols}
for i in range(V):
    for (x, y) in rules[1][i]:
        prod_of[(x, y)].add(i)

C_ord, C_unord = [], []
for i in range(V):
    for (x, y) in set(rules[1][i]):
        if is_sym[1][i]:
            if (y, x) in prod_of and i in prod_of[(y, x)] and (y, x) != (x, y):
                if [y, x] not in C_unord:
                    C_unord.append([x, y])
        else:
            rev_parents = prod_of.get((y, x), set())
            if i not in rev_parents:
                # in --reverse-valid mode rev_parents is non-empty (a DIFFERENT
                # asymmetric parent); otherwise the reverse is simply invalid
                C_ord.append([x, y])
# keep ONE direction per unordered set. Under --reverse-valid both directions
# are productions (of different parents) and both land in C_ord, so filtering
# "reverse also present" would empty the class; take a canonical orientation.
_seen_set, _keep = set(), []
for x, y in C_ord:
    k = (min(x, y), max(x, y))
    if k in _seen_set:
        continue
    _seen_set.add(k); _keep.append([x, y])
C_ord = _keep

leaf_cnt = collections.Counter(train.tolist())
hot = [t for t, _ in leaf_cnt.most_common()]
C_perm, guard = [], 0
while len(C_perm) < max(len(C_ord), 1) and guard < 100000:
    guard += 1
    x, y = rng.sample(hot[:max(4, len(hot))], 2)
    if (x, y) in prod_of or (y, x) in prod_of:         # must NOT be a rule
        continue
    if [x, y] in C_perm or [y, x] in C_perm:
        continue
    C_perm.append([x, y])

# -------------------------------------------------------------- measure
def cond_entropy_by_position(seqs):
    """Plug-in H(token at t | tokens before t), per position, estimated from
    the sampled corpus. Reported per position because the two position kinds
    have different floors by construction -- that is the point."""
    out = []
    for t in range(SEQ):
        ctx = collections.defaultdict(collections.Counter)
        for s in seqs:
            ctx[tuple(s[:t])][s[t]] += 1
        tot, H = 0, 0.0
        for c, cc in ctx.items():
            n = sum(cc.values()); tot += n
            p = np.array(list(cc.values()), float) / n
            H += n * float(-(p * np.log(p)).sum())
        out.append(H / max(tot, 1))
    return out

# EXACT floor from the generating process: the root is uniform over V symbols
# and each of the (2^L - 1) internal nodes picks uniformly among m rules. This
# is an UPPER bound on the true entropy, since distinct rule choices can in
# principle yield the same leaf string; it is exact when that map is injective.
H_seq_exact = math.log(V) + (2 ** L - 1) * math.log(M)
H_tok_exact = H_seq_exact / SEQ

Hpos = cond_entropy_by_position(train_seqs[:5000])
H_local  = float(np.mean([h for h, k in zip(Hpos, pos_kind) if k == "local"]))
H_higher = float(np.mean([h for h, k in zip(Hpos, pos_kind) if k == "higher"]))

bg = collections.defaultdict(collections.Counter)
for x, y in zip(train, train[1:]):
    bg[int(x)][int(y)] += 1
ALPHA, H_bg, n_bg = 0.1, 0.0, 0
NL = a.nleaf
for x, y in zip(val, val[1:]):
    c = bg.get(int(x)); tot = (sum(c.values()) if c else 0) + ALPHA * NL
    H_bg += -math.log(((c.get(int(y), 0) if c else 0) + ALPHA) / tot); n_bg += 1
H_bg /= max(n_bg, 1)

print(f"  RHM  depth L={L}  symbols/level v={V}  rules/symbol m={M}  "
      f"leaf vocab={a.nleaf}")
print(f"  sequence length {SEQ}   train {len(train_seqs):,} seqs "
      f"({len(train):,} tokens)   val {len(val_seqs):,} seqs (held-out, unseen)")
print(f"  distinct train sequences: {len(seen_train):,}")
print()
print(f"  RULE SPLIT (designed in, see docstring)")
for l in range(1, L + 1):
    ns = sum(is_sym[l].values())
    print(f"    level {l}: {ns}/{V} symmetric, {V-ns}/{V} asymmetric")
print()
print(f"  CONTROL CLASSES (leaf-level pairs)")
print(f"    C_ord   (asymmetric productions)      n={len(C_ord)}")
print(f"    C_unord (both orders, same parent)    n={len(C_unord)}   "
      f"<- POSITIONAL control")
print(f"    C_perm  (not productions)             n={len(C_perm)}   "
      f"<- vocabulary control")
if len(C_unord) < 8 or len(C_ord) < 8:
    print(f"    WARNING a control class is too small to average over. "
          f"Raise --nsym or --nrules before running the experiment.")
print()
def class_stats(pairs, name):
    """Unigram and pair marginals per control class. If these differ across
    classes the D comparison is confounded regardless of what the grammar
    says, since D scales with how much the model has seen of a pair."""
    if not pairs:
        return f"    {name:<8} n=0"
    uni = np.array([leaf_cnt.get(x, 0) + leaf_cnt.get(y, 0) for x, y in pairs],
                   float) / 2.0
    fwd = np.array([bg.get(x, {}).get(y, 0) for x, y in pairs], float)
    rev = np.array([bg.get(y, {}).get(x, 0) for x, y in pairs], float)
    revgram = sum(1 for x, y in pairs if (y, x) in prod_of)
    return (f"    {name:<8} n={len(pairs):<4} "
            f"unigram={uni.mean():8.1f}+/-{uni.std():7.1f}  "
            f"pair_fwd={fwd.mean():7.1f}  pair_rev={rev.mean():7.1f}  "
            f"reverse_is_a_production={revgram}/{len(pairs)}")

# ---------------------------------------------- transitivity test classes
# C_trans: a prec b prec c, with (a,b) and (b,c) BOTH appearing as level-1
# productions, but (a,c) appearing as NO production. Transitivity says a prec c
# anyway. If the model has learned the relation rather than memorising the
# rule table, an unseen comparable pair should look ORDERED; if it has only
# memorised rules, it should look like any other non-production.
#
# C_incomp: the matched negative -- pairs that are INCOMPARABLE and also not
# productions. Same "never seen as a rule" status, opposite order status.
C_trans, C_incomp, C_chain = [], [], []
if a.poset:
    prod_pairs = set(prod_of.keys())
    seen_t = set()
    for (x, y) in list(prod_pairs):
        if not PREC[x, y]:
            continue
        for z in range(LEAF):
            if not PREC[y, z] or (y, z) not in prod_pairs:
                continue
            if (x, z) in prod_pairs or (z, x) in prod_pairs:
                continue                      # must be UNSEEN as a rule
            if not PREC[x, z]:
                continue                      # closure guarantees this
            k = (x, z)
            if k in seen_t:
                continue
            seen_t.add(k); C_trans.append([x, z]); C_chain.append([x, y, z])
    _order = list(range(len(C_trans)))
    rng.shuffle(_order)
    C_trans = [C_trans[i] for i in _order][:64]
    C_chain = [C_chain[i] for i in _order][:64]
    for (x, y) in INCOMP:
        if (x, y) in prod_pairs or (y, x) in prod_pairs:
            continue
        C_incomp.append([x, y])
    rng.shuffle(C_incomp); C_incomp = C_incomp[:64]

print(f"  CLASS MARGINALS (confound check)")
print(class_stats(C_ord,   "C_ord"))
print(class_stats(C_unord, "C_unord"))
print(class_stats(C_perm,  "C_perm"))
if a.poset:
    print(class_stats(C_trans,  "C_trans"))
    print(class_stats(C_incomp, "C_incomp"))
    n_comp = int(PREC.sum())
    print(f"    PARTIAL ORDER: {n_comp} comparable ordered pairs of "
          f"{LEAF*(LEAF-1)} possible ({n_comp/(LEAF*(LEAF-1)):.1%}); "
          f"{len(INCOMP)} incomparable unordered pairs")
    print(f"    C_trans  = a<b<c with (a,b),(b,c) rules and (a,c) NOT a rule")
    print(f"    C_incomp = incomparable AND not a rule (matched negative)")
    if len(C_trans) < 8:
        print(f"    WARNING C_trans too small; raise --poset-density or --nrules")
n_sets_sym  = max(1, M // 2)
n_sets_asym = M
print(f"    RESIDUAL CONFOUND, not fixable by design: symmetric parents have")
print(f"    {n_sets_sym} distinct unordered child-sets, asymmetric ones have "
      f"{n_sets_asym}.")
print(f"    C_unord is necessarily a symmetric-parent population -- 'both")
print(f"    orders, same parent' IS the definition of a symmetric parent -- so")
print(f"    D_pos cannot be matched on parent type. Report it as such.")
print()
print(f"  ENTROPY (plug-in, nats)")
print(f"    EXACT from the grammar: {H_seq_exact:.4f} / sequence = "
      f"{H_tok_exact:.4f} per token   <- the floor, known by construction")
print(f"    (upper bound: exact iff rule choices -> leaf strings is injective)")
print()
print(f"    plug-in per position (BIASED LOW, full-prefix conditioning on a")
print(f"    finite sample reads ~0 once prefixes become unique -- do NOT use")
print(f"    the late positions as a floor):")
print(f"    per position: " + " ".join(f"{h:.3f}" for h in Hpos))
print(f"    kind:         " + " ".join(f"{k[:5]:>5}" for k in pos_kind))
print(f"    mean local  (sibling-determined)  {H_local:.4f}")
print(f"    mean higher (needs level>=2 ctx)  {H_higher:.4f}")
print(f"    bigram held-out (add-{ALPHA})      {H_bg:.4f}")
print()
print(f"  NOTE the two-component readout is CE-on-local vs CE-on-higher, not")
print(f"  the global mean. A structural objective should move `higher` and")
print(f"  leave `local` alone; the mean would hide that.")

if a.report:
    raise SystemExit(0)

os.makedirs(a.out, exist_ok=True)
json.dump([f"t{i}" for i in range(a.nleaf)], open(f"{a.out}/vocab.json", "w"))
json.dump(train.tolist(), open(f"{a.out}/train_ids.json", "w"))
json.dump(val.tolist(),   open(f"{a.out}/val_ids.json", "w"))
json.dump({"mode": "rhm", "reverse_valid": bool(a.reverse_valid), "depth": L, "nsym": V, "nrules": M,
           "nleaf": a.nleaf, "seq_len": SEQ, "seed": a.seed,
           "H_true": H_tok_exact, "H_bigram_heldout": H_bg,
           "H_seq_exact": H_seq_exact, "H_tok_exact": H_tok_exact,
           "H_local": H_local, "H_higher": H_higher,
           "H_by_position": Hpos, "pos_kind": pos_kind,
           "C_ord": C_ord, "C_unord": C_unord, "C_perm": C_perm,
           "C_trans": C_trans, "C_incomp": C_incomp, "C_chain": C_chain,
           "poset": bool(a.poset),
           "prec": (PREC.astype(int).tolist() if a.poset else []),
           "rules_level2": {str(i): rules[2][i] for i in range(V)} if L >= 2 else {},
           "is_sym_level2": {str(i): is_sym[2][i] for i in range(V)} if L >= 2 else {},
           "rules_level1": {str(i): rules[1][i] for i in range(V)},
           "is_sym_level1": {str(i): is_sym[1][i] for i in range(V)}},
          open(f"{a.out}/rhm_meta.json", "w"))
json.dump({"mode": "rhm", "H_true": H_tok_exact, "H_bigram_heldout": H_bg,
           "branch": [len(bg.get(t, {})) for t in range(a.nleaf)]},
          open(f"{a.out}/corpus_meta.json", "w"))
print(f"\n  wrote {a.out}/{{vocab,train_ids,val_ids,corpus_meta,rhm_meta}}.json")
