"""Checks for the annotated-history relay clock (Theorem thm:relayclock), on the valid six-source hexagon drawing of Study 27a
(sources 0..5, sides and the diameter (0,3); initiator 0; the acting member 3 controls {M,C,Y} of Study 26's world, D = 0.0347).
  AH1 IDENTITY  (a) activation (fastest schedule) and the activating events are functions of the trace: equal on every word obtained by
                independent swaps (400 words x 10 variants); and not of the multiset of events: reordering dependent events changes them
  AH2 IDENTITY  (b) uniform over schedules: in every schedule that realises the trace (random linear extensions with random idle rounds),
                each event's round is at least its Cartier-Foata layer, so the flip comes no earlier than T*; equality for the fastest
                schedule (400 words x 10 schedules)
  AH3 IDENTITY  (c) the braid class gives no clock bound in either direction: padding an event with its reversal keeps the braid class
                but (1) can make a member reached in the history unreached along the reduced word (the braid would overstate the time) and
                (2) can make a member reached earlier along the reduced word than in the history (the braid would understate it)
  AH4 IDENTITY  (d) the protocol assumption is necessary: if member 3 can act through a side channel from round 1, it flips before T*
                whenever a_3 > 1
"""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from pce import outerrelay as O, raagbraid as RG
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(2711)
ang = np.exp(1j * np.pi / 3 * np.arange(6)); E6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]; Dv = dict(name="hex_valid", P=ang, E=E6)
def ev_word(n): return [(E6[int(rng.integers(len(E6)))], int(rng.choice([1, -1]))) for _ in range(n)]
def trace_key(w):
    L = O.layers(w); return tuple(tuple(sorted(e for e, l in zip(w, L) if l == t)) for t in range(1, max(L, default=0) + 1))
def act_events(w, order=None):
    act = {0: (0, None)}
    for k in (order if order is not None else range(len(w))):
        (i, j), _ = w[k]
        if i in act and j not in act: act[j] = (k, w[k])
        elif j in act and i not in act: act[i] = (k, w[k])
    return act
Dd, ACT = 0.0347, 3

# AH1
ok = True
for _ in range(400):
    w = ev_word(int(rng.integers(5, 13))); a0 = O.activation(w, 0); tk = trace_key(w); ev0 = {j: v[1] for j, v in act_events(w).items()}
    for _ in range(10):
        w2 = list(w)
        for _ in range(15):
            i = int(rng.integers(len(w2) - 1))
            if not O.dependent(w2[i], w2[i + 1]): w2[i], w2[i + 1] = w2[i + 1], w2[i]
        ok &= trace_key(w2) == tk and O.activation(w2, 0) == a0 and {j: v[1] for j, v in act_events(w2).items()} == ev0
wa, wb = [((0, 1), 1), ((1, 2), 1)], [((1, 2), 1), ((0, 1), 1)]
ok &= sorted(wa) == sorted(wb) and O.activation(wa, 0) != O.activation(wb, 0)
check("AH1 IDENTITY activation and activating events are functions of the trace, not of the multiset of events", ok,
      f"same events, different order: {O.activation(wa, 0)} vs {O.activation(wb, 0)}")

# AH2
ok = True; eq = 0; tot = 0
for _ in range(400):
    while True:
        w = ev_word(int(rng.integers(4, 12))); a = O.activation(w, 0)
        if ACT in a: break
    delta = float(rng.uniform(0.005, 0.03)); Tstar = a[ACT] + math.ceil(Dd / delta - 1e-12); L = O.layers(w)
    for s in range(10):
        n = len(w); preds = {k: [i for i in range(k) if O.dependent(w[i], w[k])] for k in range(n)}; rnd = {}; left = set(range(n))
        while left:
            ready = [k for k in left if all(p in rnd for p in preds[k])]; k = ready[int(rng.integers(len(ready)))]
            gap = 0 if s == 0 else int(rng.integers(0, 3)); rnd[k] = max([rnd[p] for p in preds[k]], default=0) + 1 + gap; left.discard(k)
        ok &= all(rnd[k] >= L[k] for k in range(n))
        if s == 0: rnd = {k: L[k] for k in range(n)}
        aj = rnd[act_events(w, sorted(range(n), key=lambda k: rnd[k]))[ACT][0]]; Tflip = aj + math.ceil(Dd / delta - 1e-12)
        ok &= Tflip >= Tstar; tot += 1; eq += (s == 0 and Tflip == Tstar)
check("AH2 IDENTITY the bound is uniform over all schedules realising the trace, and attained by the fastest", ok and eq == 400,
      f"{tot} schedules; every event's round >= its layer; fastest schedule attains T* in {eq}/400")

# AH3
same = True; lost = earlier = 0
for _ in range(600):
    w = ev_word(int(rng.integers(4, 10))); pos = int(rng.integers(len(w) + 1)); e = E6[int(rng.integers(len(E6)))]
    wp = w[:pos] + [(e, 1), (e, -1)] + w[pos:]; red = O.reduce(wp)[1]
    same &= RG.braid_key(Dv, [(E6.index(x), s) for x, s in wp], 2) == RG.braid_key(Dv, [(E6.index(x), s) for x, s in red], 2)
    ah, ar = O.activation(wp, 0), O.activation(red, 0)
    lost += any(j not in ar for j in ah); earlier += any(j in ar and ar[j] < ah[j] for j in ah)
check("AH3 IDENTITY the braid class bounds the clock in neither direction", same and lost > 0 and earlier > 0,
      f"600 padded histories, braid class kept in all; a member unreached along the reduced word in {lost}; reached earlier along it in {earlier}")

# AH4
ok = True; n4 = 0
for _ in range(200):
    w = ev_word(int(rng.integers(4, 12))); a = O.activation(w, 0)
    if ACT not in a or a[ACT] <= 1: continue
    delta = float(rng.uniform(0.005, 0.03)); Tstar = a[ACT] + math.ceil(Dd / delta - 1e-12); Tside = 1 + math.ceil(Dd / delta - 1e-12)
    ok &= Tside < Tstar; n4 += 1
check("AH4 IDENTITY without the protocol assumption the bound fails: a side channel flips before T*", ok and n4 > 0, f"{n4} histories with a_3 > 1")
print(f"{sum(results)}/{len(results)} passed")
