"""Checks for Theorem 'co-revision over a graph' restated with the band generators of Birman, Ko and Lee (module pce/bklband.py).
Sources at n points in convex position (a concave arc), numbered along the arc; edges drawn as straight chords.
  KL1 IDENTITY  geometry = algebra: the clockwise exchange of two sources along their chord, read by the exact braid reader, is the band
                generator a_ts for every chord (n = 4, 5, 6); the co-revision twist of raagbraid (one source turns once around the other)
                is a_ts^2 for every chord
  KL2 IDENTITY  the two Birman-Ko-Lee relations hold: a_ts a_rq = a_rq a_ts for disjoint non-crossing chords; a_ts a_sr = a_tr a_ts = a_sr a_tr
                for t > s > r; and for crossing chords neither the band generators nor their squares commute (n = 4, 5, 6, all pairs)
  KL3 IDENTITY  delta = a_{n,n-1} ... a_21 rotates the chords: delta^{-1} a_ts delta = a_{t+1,s+1} (indices mod n, reordered); delta^n is
                the full twist and is central
  KL4 IDENTITY  the pure triangle relation, derived from the second BKL relation: for every triple r < s < t,
                tau_ts tau_sr tau_tr = tau_tr tau_ts tau_sr = tau_sr tau_tr tau_ts = full twist of {r, s, t}, the other three orders are not;
                so tau_ts commutes with tau_sr tau_tr. In convex position every triangle is empty, so this happens for every triangle
  KL5 FINDING   convex drawings of outerplanar graphs: the twist map is a homomorphism from A_G (BKL relation 1); single twists are
                faithful on balls for the triangle-free ones (path, 4-cycle, 5-cycle), and fail exactly by KL4 on a triangulated pentagon,
                where squared twists are faithful on the same ball
"""
import os, sys, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import bklband as K, braidknot as BK, raagbraid as RG
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
inv = BK._inv; same = BK.same_braid
def chords(n): return [(t, s) for s, t in itertools.combinations(range(1, n + 1), 2)]

# KL1
ok = True; cnt = 0
for n in (4, 5, 6):
    P = K.convex_points(n)
    for t, s in chords(n):
        ok &= same(K.geo_band(P, t, s), K.band(t, s), n) and same(K.geo_twist(P, t, s), K.band(t, s) * 2, n); cnt += 1
check("KL1 IDENTITY clockwise chord exchange = a_ts; co-revision twist = a_ts^2", ok, f"{cnt} chords")

# KL2
ok1 = ok2 = ok3 = True; c1 = c2 = c3 = 0
for n in (4, 5, 6):
    C = chords(n)
    for (t, s), (r, q) in itertools.combinations(C, 2):
        if {t, s} & {r, q}: continue
        a, b = K.band(t, s), K.band(r, q)
        if K.crossing(t, s, r, q):
            ok3 &= not same(a + b, b + a, n) and not same(a * 2 + b * 2, b * 2 + a * 2, n); c3 += 1
        else:
            ok1 &= same(a + b, b + a, n); c1 += 1
    for r, s, t in itertools.combinations(range(1, n + 1), 3):
        ts, sr, tr = K.band(t, s), K.band(s, r), K.band(t, r)
        ok2 &= same(ts + sr, tr + ts, n) and same(tr + ts, sr + tr, n); c2 += 1
check("KL2 IDENTITY BKL relations hold; crossing chords commute neither as half-twists nor as twists", ok1 and ok2 and ok3,
      f"non-crossing pairs {c1}, triples {c2}, crossing pairs {c3}")

# KL3
ok = True
for n in (4, 5, 6):
    d = K.delta(n)
    for t, s in chords(n):
        t2, s2 = t % n + 1, s % n + 1; t2, s2 = max(t2, s2), min(t2, s2)
        ok &= same(inv(d) + K.band(t, s) + d, K.band(t2, s2), n)
    F = d * n; ok &= same(F, K.full_twist(range(1, n + 1)), n) and all(same(F + [i], [i] + F, n) for i in range(1, n))
    ok &= same(F, [i for i in range(1, n)] * n, n)
check("KL3 IDENTITY delta rotates the chords; delta^n is the full twist and is central", ok)

# KL4
ok = True; cnt = 0
for n in (4, 5, 6):
    for r, s, t in itertools.combinations(range(1, n + 1), 3):
        tau = {"ts": K.band(t, s) * 2, "sr": K.band(s, r) * 2, "tr": K.band(t, r) * 2}; F = K.full_twist([r, s, t])
        good = {("ts", "sr", "tr"), ("tr", "ts", "sr"), ("sr", "tr", "ts")}
        for o in itertools.permutations(tau):
            ok &= same(sum((tau[x] for x in o), []), F, n) == (o in good)
        ok &= same(tau["ts"] + tau["sr"] + tau["tr"], tau["sr"] + tau["tr"] + tau["ts"], n); cnt += 1
check("KL4 IDENTITY pure triangle relation tau_ts tau_sr tau_tr = full twist (exactly the three cyclic orders), for every triple", ok, f"{cnt} triples")

# KL5
def convex(name, n, E): return dict(name="bkl_" + name, P=K.convex_points(n), E=E)
G = {"path4": convex("path4", 4, [(0, 1), (1, 2), (2, 3)]), "cycle4": convex("cycle4", 4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
     "cycle5": convex("cycle5", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]),
     "fan5": convex("fan5", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (0, 2), (0, 3)])}
L = {"path4": 5, "cycle4": 4, "cycle5": 4, "fan5": 3}
noncross = {n: all(not (RG.independent(D, a, b) and RG.segments_cross(D, a, b)) for a in range(len(D["E"])) for b in range(a + 1, len(D["E"]))) for n, D in G.items()}
res = {n: {N: RG.faithful_on_ball(D, N, L[n])[:2] for N in (1, 2)} for n, D in G.items()}
ok = all(noncross.values()) and all(res[n][1] == (True, True) for n in ("path4", "cycle4", "cycle5")) and res["fan5"][1] == (True, False) and res["fan5"][2] == (True, True)
check("KL5 FINDING outerplanar convex drawings: homomorphism; N=1 faithful on balls when triangle-free, fails on the triangulated pentagon; N=2 faithful",
      ok, f"(homomorphism, injective) by graph and power, ball radius {L}: {res}")
json.dump({n: {str(N): list(v) for N, v in r.items()} for n, r in res.items()}, open(os.path.join(HERE, "bklband_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
