"""The valuation profile of a history (section sec:histval): order and value together, with dissipation and aggregation. New module;
it modifies nothing; uses pce/orderwatch.py, pce/novorder.py and pce/hurwitz.py.
Events are (edge, sign, time, weight). At time t, under dissipation rate kappa, an event's weight is w e^(-kappa (t - time)) (events in
the future are absent). For a chain (an embedding of a path in the dependence order):
  grade       = the sum of its events' weights   (the Novikov valuation of its graded Magnus term: total evidence; aggregation adds)
  resolution  = the least of its events' weights (it is visible at level v iff every link is: the weakest link)
  death(v)    = the first time its resolution falls below v; under exponential dissipation, min over links of time + log(w / v) / kappa
  chains(word, path)                  all embeddings of a path (index tuples)
  profile(events, path, t, kappa)     the chains at time t with grade and resolution
  death_time(events, chain, v, kappa) closed form
  aggregate(events)                   merge events on one edge at one time: weights add with sign; the net sign is the sign of the sum
  band_word(events, t, kappa, v)      the braid of the events visible at level v at time t, one Birman-Ko-Lee band generator a_ij^(+-1)
                                      per event (strands = sources), as an Artin word
"""
from __future__ import annotations
import itertools, math
from . import orderwatch as OW, hurwitz as H
def _w(ev, t, kappa): return ev[3] * math.exp(-kappa * (t - ev[2])) if ev[2] <= t else 0.0
def chains(word, path):
    below = OW.poset([(e[0], e[1]) for e in word]); occ = [[i for i, e in enumerate(word) if OW._matches(e, ed)] for ed in path]
    return [c for c in itertools.product(*occ) if all(c[j] in below[c[j + 1]] for j in range(len(c) - 1))]
def profile(events, path, t, kappa):
    live = [e for e in events if e[2] <= t]; out = []
    for c in chains(live, path):
        ws = [_w(live[i], t, kappa) for i in c]; sg = 1
        for i in c: sg *= live[i][1]
        out.append(dict(chain=c, sign=sg, grade=sum(ws), resolution=min(ws)))
    return out
def death_time(events, chain, v, kappa):
    """the first time the chain's weakest link falls below v (exponential dissipation): min_i time_i + log(w_i / v) / kappa."""
    return min(events[i][2] + math.log(events[i][3] / v) / kappa for i in chain)
def aggregate(events):
    acc = {}
    for e, s, t, w in events:
        k = (tuple(sorted(e)), t); acc[k] = acc.get(k, 0.0) + s * w
    return [(k[0], 1 if v > 0 else -1, k[1], abs(v)) for k, v in sorted(acc.items(), key=lambda kv: kv[0][1]) if v != 0]
def band_word(events, t, kappa, v):
    w = []
    for e, s, tm, wt in sorted(events, key=lambda x: x[2]):
        if tm <= t and wt * math.exp(-kappa * (t - tm)) >= v:
            i, j = sorted(e); b = H.band(j + 1, i + 1)            # sources 0..n-1 -> strands 1..n
            w += b if s > 0 else H.inverse(b)
    return w
