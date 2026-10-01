"""Arity grading, p-adically: nested gluing radii by the arity of the units that make the claims (synthetic).

Sources s carry integer values x_s in Z_p (p = 3). A unit u covers k sources (its arity) and measures them as
y_{u,s} = x_s + e_{u,s}; each unit contributes the sum claims y_{u,i} + y_{u,j} for the pairs it covers.
Units of arity k measure to precision p^{a_k} (errors divisible by p^{a_k}), a_2 = 1, a_3 = 2, a_4 = 4:
wider units are more precise. For each grade threshold K we keep the claims of units with arity >= K and compute
the order profile e(m) of the obstruction (hull.obstruction_profile) and the gluing depth m*.
Predictions:  (1) the claims of ONE unit always glue (a filled cell is flat: x = y_u solves them), e = 0 at every m;
(2) the sub-system of arity >= K glues exactly to radius p^-m*_K with m*_K = the smallest error valuation among
disagreeing units of that grade, so the depths are nested: m*_2 <= m*_3 <= m*_4.
"""
import os, sys, json, itertools
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
from amb_vigneaux.hull import obstruction_profile
from amb_vigneaux.padic_loops import sum_claim_depth

p, K, n = 3, 8, 7
rng = np.random.default_rng(0)
A = {2: 1, 3: 2, 4: 4}
x = rng.integers(0, p ** K, size=n)
units = []
for k, cnt in ((2, 10), (3, 8), (4, 6)):
    for _ in range(cnt):
        S = sorted(rng.choice(n, size=k, replace=False).tolist())
        e = {s: int(rng.integers(1, 5)) * p ** A[k] for s in S}
        units.append((k, S, {s: int(x[s]) + e[s] for s in S}))

def claims(us):
    return [(i, j, y[i] + y[j]) for _, S, y in us for i, j in itertools.combinations(S, 2)]

out = {}
single = [sum_claim_depth(n, claims([u]), p, K) for u in units]
print(f"one unit alone: gluing depth m* = {K} (full precision, flat) for {sum(m == K for m in single)}/{len(units)} units")
for Kmin in (2, 3, 4):
    us = [u for u in units if u[0] >= Kmin]
    cl = claims(us); prof = obstruction_profile(n, cl, p, K); ms = sum_claim_depth(n, cl, p, K)
    out[Kmin] = dict(n_units=len(us), n_claims=len(cl), m_star=ms, profile=[o for _, o in prof])
    print(f"arity >= {Kmin}: {len(us)} units, {len(cl)} claims  -> glue to radius {p}^-{ms};  e(m) = {[o for _, o in prof]}")
json.dump(dict(single=single, graded=out), open(os.path.join(here, "padic_arity_results.json"), "w"), indent=1)
