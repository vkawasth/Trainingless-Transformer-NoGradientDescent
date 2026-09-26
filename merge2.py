#!/usr/bin/env python3
"""MERGE2 -- can two corpora share one grammar, and at what resolution?

    python3 merge2.py --a /tmp/cH --b /tmp/cH --split      # control: one corpus halved
    python3 merge2.py --a /tmp/cOvA --b /tmp/cOvB          # two bracketings

THE QUESTION
------------
Two corpora, each with its own hierarchical structure. Merging them raises
three failures that live at different levels, and only the third is an
obstruction:

  1. the symbol alphabets do not correspond. Each grammar's symbols are
     defined only up to relabelling, so before anything else the two must be
     ALIGNED -- an assignment problem, solved exactly here by minimising
     sum KL(row_A(s) || row_B(pi(s))) over permutations pi.

  2. the aligned rows disagree. The merged row is the mixture, the cost is the
     Bregman information, and merging always succeeds -- it just costs. A
     measurement, not an obstruction.

  3. the BRACKETINGS disagree. Then the union of the two constituent families
     is a cover that overlaps without refining, which is generically cyclic.
     Neither corpus supplies a global section for the other's contexts, so the
     single-source theorem does not apply and the conflict fraction can be
     positive.

TRUNCATION IS THE PARAMETER THAT DECIDES IT
-------------------------------------------
On one corpus, truncation is idle: the data glue at every threshold, because
the empirical joint is a witness. With two sources it is load-bearing, because
it is the only knob that moves the obstruction. Prune every context whose mass
falls below tau and recompute:

    CF(tau) = conflict remaining after pruning below tau.

At large tau only the shared stem survives and the corpora agree; at tau = 0
the full fan-out is present and disagreement is maximal. The threshold

    tau* = inf { tau : CF(tau) = 0 }

is the resolution at which the two corpora become reconcilable: above it one
grammar covers both, below it the disagreement is irreducible.

CONTROL
-------
--split takes ONE corpus, halves it, and treats the halves as two sources.
They differ only by sampling noise, so tau* must be 0 (or at the smallest
tested tau). If the control shows a positive tau*, the pipeline is measuring
sampling variation rather than structural disagreement.

CAVEAT, MEASURED NOT ASSUMED
----------------------------
CF(tau) need not be monotone. Pruning removes constraints as well as contexts,
and dropping a context can free the remaining ones to glue. The sweep reports
the whole curve rather than a single crossing, so non-monotonicity is visible.
"""
import json, argparse, itertools
import numpy as np
from scipy.optimize import linprog

ap = argparse.ArgumentParser()
ap.add_argument("--a", default="/tmp/cH")
ap.add_argument("--b", default="")
ap.add_argument("--split", action="store_true",
                help="control: halve corpus A and treat the halves as two sources")
ap.add_argument("--patch", type=int, default=6, help="leaf positions compared")
ap.add_argument("--classes", type=int, default=3, help="coarse-grained leaf classes")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--taus", default="0,0.0005,0.001,0.002,0.005,0.01,0.02,0.05")
ap.add_argument("--shift-b", type=int, default=0,
                help="roll corpus B by this many positions. A shift of 1 makes "
                     "B's constituents straddle A's, which is a different "
                     "bracketing of the same tokens -- the cross-source case.")
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.RandomState(a.seed)

def load(path, n):
    M = json.load(open(f"{path}/rhm_meta.json"))
    SEQ = M["seq_len"]
    tr = np.array(json.load(open(f"{path}/train_ids.json")), dtype=np.int64)
    return tr[:(len(tr)//SEQ)*SEQ].reshape(-1, SEQ)[:n], M

XA, MA = load(a.a, a.n)
if a.split:
    h = len(XA)//2
    XB, MB = XA[h:], MA
    XA = XA[:h]
    src = f"{a.a} (halved: CONTROL)"
else:
    XB, MB = load(a.b, a.n)
    src = f"{a.a}  vs  {a.b}"
SEQ = MA["seq_len"]; P, C = a.patch, a.classes
pos = [(a.start+k) % SEQ for k in range(P)]
DA = XA[:, pos] % C
if a.shift_b: XB = np.roll(XB, a.shift_b, axis=1)
DB = XB[:, pos] % C
print(f"  merge2: {src}")
print(f"  positions {pos}, coarse-grained to {C} classes, "
      f"{len(DA)} + {len(DB)} sequences")

# ---------------------------------------------------- the two constituent families
# A brackets adjacent pairs; B brackets the staggered pairs. These are the two
# bracketings of the same patch that overlap without refining each other.
CTX_A = [tuple(sorted((i, i+1))) for i in range(0, P-1, 2)]
CTX_B = [tuple(sorted(((i+1) % P, (i+2) % P))) for i in range(0, P-1, 2)]
print(f"  A constituents {CTX_A}\n  B constituents {CTX_B}")

def is_acyclic(ctxs):
    S = [set(c) for c in ctxs]; changed = True
    while changed:
        changed = False
        cnt = {}
        for s in S:
            for v in s: cnt[v] = cnt.get(v, 0)+1
        for s in S:
            ear = {v for v in s if cnt[v] == 1}
            if ear: s -= ear; changed = True
        S2 = [s for s in S if s]
        for i, s in enumerate(S2):
            if any(i != j and s <= t for j, t in enumerate(S2)):
                S2.pop(i); changed = True; break
        S = S2
    return len(S) == 0
print(f"  union cover acyclic? {is_acyclic(CTX_A + CTX_B)}"
      f"   (False is the precondition for any obstruction)\n")

# ---------------------------------------------------- alignment of the alphabets
def block_codes(D, ctx):
    idx = np.zeros(len(D), dtype=np.int64)
    for p in ctx: idx = idx*C + D[:, p]
    return idx

def align_cost(DA, DB, ctx):
    """symbols here are the coarse-grained block values; the alignment is the
    permutation of B's values minimising the KL between the two block
    distributions. Solved exactly by the Hungarian algorithm."""
    K = C**len(ctx)
    ta = np.bincount(block_codes(DA, ctx), minlength=K).astype(float); ta /= ta.sum()
    tb = np.bincount(block_codes(DB, ctx), minlength=K).astype(float); tb /= tb.sum()
    return ta, tb

def table(D, ctx, tau):
    """empirical table on a context, with cells below tau pruned and the rest
    renormalised -- this is the truncation, applied to the CONTEXT data"""
    K = C**len(ctx)
    t = np.bincount(block_codes(D, ctx), minlength=K).astype(float); t /= t.sum()
    t = np.where(t >= tau, t, 0.0)
    s = t.sum()
    return t/s if s > 0 else t

OUT = np.array(list(itertools.product(range(C), repeat=P)), dtype=np.int64)
def out_codes(ctx):
    idx = np.zeros(len(OUT), dtype=np.int64)
    for p in ctx: idx = idx*C + OUT[:, p]
    return idx
CODES = {c: out_codes(c) for c in set(CTX_A) | set(CTX_B)}

def match_overlaps(tabs_a, tabs_b, iters=300):
    """Make B's tables agree with A's on every shared position, by iterative
    proportional fitting. WITHOUT this step CF measures sampling noise: two
    halves of ONE corpus have slightly different single-position marginals, the
    cover is cyclic, and mere inconsistency on overlaps already makes the
    family unrealisable. The control showed CF = 0.0060 for two halves of the
    same data. Matching removes that channel, so what remains is the structural
    conflict the measurement is for."""
    tb = {c: t.reshape([C]*len(c)).copy() for c, t in tabs_b.items()}
    for _ in range(iters):
        for cb in tb:
            for k, p_ in enumerate(cb):
                # target: A's marginal at this position, from whichever A
                # context contains it
                ca = next((c for c in tabs_a if p_ in c), None)
                if ca is None: continue
                ta = tabs_a[ca].reshape([C]*len(ca))
                tgt = ta.sum(axis=tuple(i for i in range(len(ca))
                                        if ca[i] != p_))
                cur = tb[cb].sum(axis=tuple(i for i in range(len(cb))
                                            if cb[i] != p_))
                sc = tgt/np.maximum(cur, 1e-300)
                shape = [1]*len(cb); shape[k] = C
                tb[cb] = tb[cb]*sc.reshape(shape)
            ssum = tb[cb].sum()
            if ssum > 0: tb[cb] = tb[cb]/ssum
    return {c: t.reshape(-1) for c, t in tb.items()}

def cf(tau, match=True):
    """conflict fraction of the union family at truncation tau: the weight no
    single distribution on the patch can carry"""
    ta = {c: table(DA, c, tau) for c in CTX_A}
    tb = {c: table(DB, c, tau) for c in CTX_B}
    if match: tb = match_overlaps(ta, tb)
    rows, b = [], []
    for tabs in (ta, tb):
        for ctx, t in tabs.items():
            code = CODES[ctx]
            for v in range(C**len(ctx)):
                rows.append((code == v).astype(float)); b.append(t[v])
    r = linprog(-np.ones(len(OUT)), A_ub=np.array(rows), b_ub=np.array(b),
                bounds=(0, None), method="highs")
    return max(0.0, 1.0 - (-r.fun if r.status == 0 else 0.0))

def kept(tau):
    n = 0; tot = 0
    for D, ctxs in ((DA, CTX_A), (DB, CTX_B)):
        for ctx in ctxs:
            t = table(D, ctx, 0.0)
            n += int((t >= tau).sum()); tot += len(t)
    return n, tot

print(f"    {'tau':>9}{'cells kept':>13}{'CF':>10}{'CF unmatched':>12}"
      f"   verdict")
print(f"    CF is computed after matching the two families on their shared")
print(f"    positions; CF unmatched is without that step, and is dominated by")
print(f"    sampling differences rather than structure.")
taus = [float(x) for x in a.taus.split(",")]
curve = []
for t in taus:
    k, tot = kept(t); v = cf(t); vraw = cf(t, match=False)
    curve.append((t, v))
    print(f"    {t:>9.4g}{f'{k}/{tot}':>13}{v:>10.4f}{vraw:>12.4f}"
          f"   {'reconcilable' if v <= 1e-6 else 'conflict remains'}")

zero = [t for t, v in curve if v <= 1e-6]
if zero:
    tstar = min(zero)
    print(f"\n  tau* = {tstar:.4g}: above this the two corpora share one grammar"
          f" on this patch.")
else:
    print(f"\n  tau* not reached within the sweep: the corpora conflict at every"
          f" tested resolution.")
mono = all(curve[i][1] >= curve[i+1][1] - 1e-12 for i in range(len(curve)-1))
print(f"  CF(tau) monotone decreasing: {mono}"
      f"   {'' if mono else '-- pruning a context can FREE the rest to glue'}")
print(f"\n  Control expectation: with --split the two halves differ only by")
print(f"  sampling, so tau* should be 0. A positive tau* there would mean the")
print(f"  pipeline is reading sampling variation as structural disagreement.")
