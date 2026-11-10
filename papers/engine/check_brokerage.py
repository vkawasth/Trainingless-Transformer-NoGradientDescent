"""Checks for 'Brokerage and Milnor's invariant' (companion note; module pce/brokerage.py). Exact enumeration of all interleavings of the
two visitors' own histories (m intervals each: C(4m, 2m) histories), and an episode experiment.
  BR1 IDENTITY  Milnor's triple invariant of the closure equals the signed crossing number of the relationship intervals (m = 1, 2, 3)
  BR2 IDENTITY  held fixed across all interleavings: each source's own history (the broker never moves; each visitor's curve and order of
                events), event counts, co-revision counts, and all pairwise linking numbers (zero); the planar realisation gives the
                algebraic braid
  BR3 IDENTITY  the triple invariant is not a function of any one of: who opens first, who closes first, the open-precedence count, the
                close-precedence count, the unsigned precedence count (explicit pairs); but it is exactly the typed combination
                mu = 2 (P_oo + P_cc) - P_all, where P_oo counts (1 opens, 3 opens) pairs in that order, P_cc closing pairs, and P_all all pairs
                of a 1-event before a 3-event: a level-2 (pairwise, typed, ordered) statistic
  BR4 IDENTITY  in this family the braid, the triple invariant and the level-2 and level-3 signatures of the count path give the same classes
  BR5 IDENTITY  under random timing (all interleavings equally likely) the invariant has mean 0 and variance exactly m^2 / (4m - 1) for
                m = 1, ..., 5 (exact rational enumeration; a closed form we verified but did not prove)
  BR6 FINDING   episodes: a relay broker (1 opens, 3 opens, 1 closes, 3 closes) and a sequential broker (1 opens and closes, then 3) have the
                same opening and closing precedence; over 200 episodes with fidelity 0.6, the open- and close-precedence walks fire for both,
                the walk of the invariant fires only for the relay; a nested broker is separated by the close-precedence walk
"""
import os, sys, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import brokerage as BR, braidknot as BK, bundlebraid as BB
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
H = {m: BR.interleavings(m) for m in (1, 2, 3)}

# BR1
check("BR1 IDENTITY triple invariant = signed crossing number of the relationship intervals",
      all(BR.crossing_number(h) == BR.mu123(h) for m in H for h in H[m]), f"histories: { {m: len(v) for m, v in H.items()} }")

# BR2
own_ok = all([e for e in h if e[0] == "1"] == [("1", k) for k in BR.own(m)] and [e for e in h if e[0] == "3"] == [("3", k) for k in BR.own(m)] for m in H for h in H[m])
lk_ok = all(all(v == 0 for dd in BR.linking(h).values() for v in dd.values()) for m in (1, 2) for h in H[m])
geo = sum(BK.artin(BB.braid(BR.realise(h) + 1e-7 * np.exp(1j * np.arange(3))[None, :], 0.123)[0], 3) == BK.artin(BR.braid_word(h), 3) for h in H[2])
check("BR2 IDENTITY fixed own histories and counts, zero linking, geometric braid = algebraic", own_ok and lk_ok and geo == len(H[2]), f"geometric check on all {len(H[2])} histories with m = 2")

# BR3
S = {m: [BR.stats(h) for h in H[m]] for m in (1, 2)}; ex = {}
for key in ("first_open", "first_close", "open_prec", "close_prec", "unsigned_prec"):
    found = None
    for m in (1, 2):
        seen = {}
        for s, h in zip(S[m], H[m]):
            k = (s[key], s["counts"]); mu = BR.mu123(h)
            if k in seen and seen[k][1] != mu: found = (seen[k][0], h); break
            seen.setdefault(k, (h, mu))
        if found: break
    ex[key] = found
lin = all(BR.mu123(h) == 2 * (s["open_prec"] + s["close_prec"]) - s["unsigned_prec"] for m in (1, 2) for s, h in zip(S[m], H[m]))
lin3 = all(BR.crossing_number(h) == 2 * (BR.stats(h)["open_prec"] + BR.stats(h)["close_prec"]) - BR.stats(h)["unsigned_prec"] for h in H[3])
fmt = lambda h: " ".join(src + kind for src, kind in h)
check("BR3 IDENTITY no single precedence statistic determines the invariant; mu = 2 (P_oo + P_cc) - P_all exactly", all(v is not None for v in ex.values()) and lin and lin3,
      "; ".join(f"{k}: [{fmt(v[0])}] vs [{fmt(v[1])}]" for k, v in ex.items()))

# BR4
def ncls(m, f): return len({f(s, h) for s, h in zip(S[m], H[m])})
cls = {m: dict(mu=ncls(m, lambda s, h: BR.mu123(h)), braid=ncls(m, lambda s, h: s["braid"]), sig2=ncls(m, lambda s, h: s["sig2"]), sig3=ncls(m, lambda s, h: s["sig3"])) for m in (1, 2)}
check("BR4 IDENTITY braid, triple invariant and level-2/3 signatures give the same classes in this family", all(len(set(c.values())) == 1 for c in cls.values()), f"{cls}")

# BR5
var = {m: float(np.var([BR.crossing_number(h) for h in H[m]])) for m in H}; mean = {m: float(np.mean([BR.crossing_number(h) for h in H[m]])) for m in H}
from fractions import Fraction
def exact_var(m):
    tot = Fraction(0); tot2 = Fraction(0); cnt = 0
    for pos in itertools.combinations(range(4 * m), 2 * m):
        ps = set(pos); i1 = i3 = 0; I1 = []; I3 = []; o1 = o3 = None
        for t in range(4 * m):
            if t in ps:
                if i1 % 2 == 0: o1 = t
                else: I1.append((o1, t))
                i1 += 1
            else:
                if i3 % 2 == 0: o3 = t
                else: I3.append((o3, t))
                i3 += 1
        c = sum(1 for a, b in I1 for p, q in I3 if a < p < b < q) - sum(1 for a, b in I1 for p, q in I3 if p < a < q < b)
        tot += c; tot2 += c * c; cnt += 1
    return tot / cnt, tot2 / cnt
EX = {m: exact_var(m) for m in (1, 2, 3, 4, 5)}
check("BR5 IDENTITY random timing: mean 0 and variance m^2/(4m - 1) exactly for m = 1..5", all(mu_ == 0 and v == Fraction(m * m, 4 * m - 1) for m, (mu_, v) in EX.items()),
      f"variances { {m: str(v) for m, (_, v) in EX.items()} }")

# BR6
rng = np.random.default_rng(11); H1 = H[1]; E = 200; FID = 0.6
MECH = {"relay": (("1", "o"), ("3", "o"), ("1", "c"), ("3", "c")), "sequential": (("1", "o"), ("1", "c"), ("3", "o"), ("3", "c")), "nested": (("1", "o"), ("3", "o"), ("3", "c"), ("1", "c"))}
def incs(h):
    s = BR.stats(h); return np.array([BR.crossing_number(h) / np.sqrt(1 / 3), 1.0 if s["first_open"] == "1" else -1.0, 1.0 if s["first_close"] == "1" else -1.0])
INC = {h: incs(h) for h in H1}
def history(mech):
    out = []
    for _ in range(E):
        h = MECH[mech] if (mech != "null" and rng.random() < FID) else H1[rng.integers(len(H1))]; out.append(INC[h])
    return np.array(out)
def zmax(X): S_ = np.cumsum(X, axis=0) / np.sqrt(np.arange(1, E + 1))[:, None]; return np.abs(S_[9:]).max(axis=0)
Bnd = np.quantile(np.array([zmax(history("null")) for _ in range(2000)]), 0.95, axis=0)
rate = {mech: (np.array([zmax(history(mech)) for _ in range(400)]) >= Bnd).mean(axis=0).round(3).tolist() for mech in ("null", "relay", "sequential", "nested")}
ok = (max(rate["null"]) <= 0.08 and rate["relay"][0] >= 0.95 and rate["sequential"][0] <= 0.08 and rate["nested"][0] <= 0.08
      and min(rate["relay"][1:]) >= 0.95 and min(rate["sequential"][1:]) >= 0.95 and rate["nested"][1] >= 0.95 and rate["nested"][2] >= 0.95)
check("BR6 FINDING only the invariant's walk separates a relay broker from a sequential one", ok,
      f"alarm rates [invariant, open precedence, close precedence]: {rate}; boundaries {np.round(Bnd, 2).tolist()}")
json.dump(dict(classes=cls, variance=var, rates=rate), open(os.path.join(HERE, "brokerage_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
