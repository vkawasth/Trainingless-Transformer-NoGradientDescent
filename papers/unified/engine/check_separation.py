"""Checks for the separation theorem (thm:separation, section sec:likval): the likelihood valuation and the structural invariants
(signed chain counts of the order watch, the event braid) are independent invariants of a history. New script; it modifies nothing;
uses pce/likval.py, pce/orderwatch.py, pce/dualgarside.py. World: six sources on E6 (the hexagon with chord 0-3; non-crossing chords,
so band generators of vertex-disjoint edges commute), events (edge, sign); term k carries a law P_k on the 14 events.
  SP1 IDENTITY  the valuation is a function of the counts: every permutation of a word gives the same shift lam - (n/N) H_hat (exact)
  SP2 IDENTITY  the structure is a trace invariant: swapping adjacent vertex-disjoint events, or inserting an adjacent cancelling pair,
                leaves every signed chain count (k = 0) and the event braid (dual left normal form) unchanged
  SP3 IDENTITY  same counts, different structure: the relay 01+, 12+, 23+ and its reversal have equal counts and equal valuation, but
                chain count 1 against 0 and different braids
  SP4 IDENTITY  same structure, different valuation: inserting a cancelling pair changes the counts and the shift (strictly, since every
                log-likelihood is negative), and moves the wall function by (Lambda_a(e+) + Lambda_a(e-)) / N, but changes no structural
                invariant
  SP5 RECORD    how often the structure varies within a count class: over random multisets of 3 to 6 events, the share of classes in which
                some permutation changes a chain count, and the share in which some permutation changes the braid
"""
import os, sys, math, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import likval as LV, orderwatch as OW, dualgarside as DG
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(20261115)
E6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]; ALPH = [(e, s) for e in E6 for s in (1, -1)]; IDX = {x: i for i, x in enumerate(ALPH)}
PATHS = [[(0, 1), (1, 2), (2, 3)], [(1, 2), (2, 3), (3, 4)], [(5, 0), (0, 3), (3, 4)], [(0, 1), (1, 2)], [(2, 3), (3, 4)]]
P = [rng.dirichlet(np.ones(len(ALPH)) * 2) for _ in range(4)]; a = np.array([2, -1, 2, -3], float); LA = LV.circuit_llr(a, P)
lam = rng.normal(size=4); N = 100.0
def counts(word): c = np.zeros(len(ALPH)); [c.__setitem__(IDX[(e, s)], c[IDX[(e, s)]] + 1) for e, s in word]; return c
def shift(word): return LV.lam_eff(lam, P, counts(word), N)
def tagged(word): return [(e, s, t) for t, (e, s) in enumerate(word)]
def chains(word): return tuple(OW.chain_count(tagged(word), p, k=0, signed=True) for p in PATHS)
def braid(word):
    inf, nf = DG.left_normal_form([((max(e) + 1, min(e) + 1), s) for e, s in word], 6); return (inf, tuple(nf))
def rword(k): return [ALPH[int(rng.integers(len(ALPH)))] for _ in range(k)]
worst = 0.0
for _ in range(500):
    w = rword(int(rng.integers(3, 12))); w2 = list(w); rng.shuffle(w2); worst = max(worst, float(np.abs(shift(w) - shift(w2)).max()))
check("SP1 IDENTITY the valuation shift depends only on the counts (every permutation gives the same shift)", worst < 1e-12, f"500 words; max deviation {worst:.1e}")
bad_c = bad_b = 0; nsw = nca = 0
for _ in range(500):
    w = rword(int(rng.integers(3, 10))); c0, b0 = chains(w), braid(w)
    idx = [i for i in range(len(w) - 1) if not set(w[i][0]) & set(w[i + 1][0])]
    if idx:
        i = idx[int(rng.integers(len(idx)))]; w2 = w[:i] + [w[i + 1], w[i]] + w[i + 2:]; nsw += 1
        bad_c += chains(w2) != c0; bad_b += braid(w2) != b0
    e = E6[int(rng.integers(7))]; s = int(rng.choice([1, -1])); j = int(rng.integers(len(w) + 1)); w3 = w[:j] + [(e, s), (e, -s)] + w[j:]; nca += 1
    bad_c += chains(w3) != c0; bad_b += braid(w3) != b0
check("SP2 IDENTITY signed chain counts and the event braid are invariant under independent swaps and cancelling pairs", bad_c == 0 and bad_b == 0 and nsw > 300,
      f"{nsw} swaps, {nca} cancelling pairs; chain-count changes {bad_c}, braid changes {bad_b}")
relay = [((0, 1), 1), ((1, 2), 1), ((2, 3), 1)]; rev = relay[::-1]
ok3 = np.array_equal(counts(relay), counts(rev)) and float(np.abs(shift(relay) - shift(rev)).max()) < 1e-12 and chains(relay)[0] == 1 and chains(rev)[0] == 0 and braid(relay) != braid(rev)
check("SP3 IDENTITY same counts and valuation, different structure: the relay and its reversal", ok3,
      f"chain counts {chains(relay)[0]} vs {chains(rev)[0]}; braids {braid(relay)} vs {braid(rev)}")
bad = 0; dmin = 1e9; wmove = []
for _ in range(300):
    w = rword(int(rng.integers(3, 9))); e = E6[int(rng.integers(7))]; j = int(rng.integers(len(w) + 1)); w3 = w[:j] + [(e, 1), (e, -1)] + w[j:]
    d = shift(w) - shift(w3); dmin = min(dmin, float(d.min()))
    pred = (LA[IDX[(e, 1)]] + LA[IDX[(e, -1)]]) / N; cw = -a @ shift(w); cw3 = -a @ shift(w3); wmove.append(abs((cw - cw3) - pred))
    bad += chains(w3) != chains(w) or braid(w3) != braid(w) or not (d > 0).all()
check("SP4 IDENTITY same structure, different valuation: a cancelling pair lowers every coordinate of the shift, moves the wall by its circuit LLR, changes no structure",
      bad == 0 and max(wmove) < 1e-12, f"300 insertions; least decrease {dmin:.3f}; max wall-move error {max(wmove):.1e}; violations {bad}")
cls_c = cls_b = tot = 0
for _ in range(300):
    w = rword(int(rng.integers(3, 7))); perms = set(itertools.permutations(w)); perms = list(perms)[:120]
    cs = {chains(list(p)) for p in perms}; bs = {braid(list(p)) for p in perms}; tot += 1; cls_c += len(cs) > 1; cls_b += len(bs) > 1
check("SP5 RECORD within a count class the structure varies: share of random classes where some reordering changes a chain count / the braid", True,
      f"{tot} classes of 3-6 events: chain counts vary in {cls_c / tot:.2f}, the braid varies in {cls_b / tot:.2f}")
print(f"{sum(results)}/{len(results)} passed")
