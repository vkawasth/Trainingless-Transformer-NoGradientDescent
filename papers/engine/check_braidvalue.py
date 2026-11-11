"""Checks for the decision value of the braid (module pce/braidvalue.py). Brokerage histories (m = 2), honest and coordinated laws as in
check_credalbraid; lambda in [0.8, 1.25], beta in [0.5, 2], prior 1/2; actions flag / accept / inspect with costs (c_fp, c_fn, c_in).
  BV1 IDENTITY  ex post, shared nuisance: the posterior interval, the Gamma-maximin action and value, and the E-admissible set after
                observing the full history equal those after observing only its braid, history by history
  BV2 IDENTITY  ex ante (normal form): the Gamma-maximin value never decreases with information: V(full) >= V(typed) >= V(braid) >=
                V(counts) and V(full) >= V(untyped) >= V(counts), for shared and separate nuisances and three cost settings
  BV3 FINDING   separate nuisances, costs (1, 1, 10): ex ante the full record is worth more than the braid; ex post (averaged under the
                uniform honest law) the braid gives the higher Gamma-maximin value and a smaller indeterminate (E-admissibility) mass;
                with a shared nuisance the two coincide both ex ante and ex post
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import braidvalue as BV, credalbraid as CB, brokerage as BR
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
H = BR.interleavings(2); P = CB.projections(H); lams = np.linspace(0.8, 1.25, 7); betas = np.linspace(0.5, 2, 7)
LAWS = {sh: BV.joint_laws(H, lams, lams, betas, sh) for sh in (True, False)}
COSTS = [(1, 1, 10), (1, 3, 10), (1, 1, 0.45)]
mu = np.array([BR.mu123(h) for h in H])

# BV1
ok = True
for costs in COSTS:
    Pf, kf = P["full"]; Pb, kb = P["braid"]
    lf, hf = BV.posterior_interval(Pf, len(kf), LAWS[True]); lb, hb = BV.posterior_interval(Pb, len(kb), LAWS[True])
    af, vf, ef = BV.expost(lf, hf, costs); ab, vb, eb = BV.expost(lb, hb, costs)
    for i, h in enumerate(H):
        y = Pb[i]; ok &= abs(lf[Pf[i]] - lb[y]) < 1e-12 and abs(hf[Pf[i]] - hb[y]) < 1e-12 and af[Pf[i]] == ab[y] and abs(vf[Pf[i]] - vb[y]) < 1e-12 and ef[Pf[i]] == eb[y]
check("BV1 IDENTITY ex post, shared nuisance: full history and braid give the same interval, action, value and E-admissible set", ok, "70 histories, 3 cost settings")

# BV2
ok = True; table = {}
for sh in (True, False):
    for costs in COSTS:
        V = {k: BV.exante(P[k][0], len(P[k][1]), LAWS[sh], costs) for k in ("counts", "untyped", "braid", "typed", "full")}
        table[f"{'shared' if sh else 'separate'} {costs}"] = {k: round(v, 4) for k, v in V.items()}
        e = 1e-9; ok &= V["full"] >= V["typed"] - e >= V["braid"] - 2 * e >= V["counts"] - 3 * e and V["full"] >= V["untyped"] - e >= V["counts"] - 2 * e
check("BV2 IDENTITY ex ante the Gamma-maximin value is monotone in information", ok, f"{table}")

# BV3
res = {}
for sh in (True, False):
    res[sh] = {}
    for k in ("braid", "full"):
        Pi, keys = P[k]; m = len(keys); lo, hi = BV.posterior_interval(Pi, m, LAWS[sh]); acts, vals, adm = BV.expost(lo, hi, (1, 1, 10))
        ref = CB.push(CB.laws(H, 1.0, 0.0)[0], Pi, m)[0]
        res[sh][k] = dict(exante=round(BV.exante(Pi, m, LAWS[sh], (1, 1, 10)), 4), expost=round(float(np.sum(ref * vals)), 4),
                          indeterminate=round(float(sum(r for r, a in zip(ref, adm) if len(a) > 1)), 3))
S, D = res[True], res[False]
ok = D["full"]["exante"] > D["braid"]["exante"] and D["braid"]["expost"] > D["full"]["expost"] and D["braid"]["indeterminate"] < D["full"]["indeterminate"]
ok &= abs(S["full"]["exante"] - S["braid"]["exante"]) < 1e-6 and abs(S["full"]["expost"] - S["braid"]["expost"]) < 1e-9
check("BV3 FINDING separate nuisances: full record better ex ante, braid better ex post; shared nuisance: equal both ways", ok,
      f"shared {S}; separate {D}")
json.dump(dict(exante=table, costs_1_1_10={"shared": S, "separate": D}), open(os.path.join(HERE, "braidvalue_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
