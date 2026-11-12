"""Checks for 'A probabilistic theory of pseudo-links' (companion note; modules pce/pseudoprob.py, pce/tiebraid.py).
  PP1 IDENTITY  universality: under any circular law of the persistent biases (independent, equicorrelated, two sources correlated, and a
                non-Gaussian rotated law) the argument of every pairwise difference is uniform, so every pair's sector probability is its
                angle over 2 pi
  PP2 IDENTITY  the phase difference of two jointly circular Gaussian differences with complex correlation rho has the stated density
                (Kolmogorov distance to Monte Carlo below 0.01 for rho = 0.05, 0.5, 0.95)
  PP3 FINDING   on a tie history the pairwise linking laws do not depend on the correlation of the biases, but their joint law does
  PP4 IDENTITY  on the interleaved broker-visit history the triple invariant is 1 exactly when both visitors' differences fall in their
                enclosing sectors; its probability, computed from the phase-difference density, matches Monte Carlo for four correlation
                structures; for nested and sequential visits it is 0 always
  PP5 IDENTITY  independent circular jitter during a tie of T steps: winding variance (T - 1)/12 turns^2
  PP6 FINDING   random-walk perturbation: the winding divided by (1/2) log T moves towards the hyperbolic secant law of big windings (Pitman-Yor)
                and away from the Cauchy law of total windings (Spitzer), as T grows; convergence is slow
  PP7 IDENTITY  persistent bias: the link takes finitely many values, one per resolution (here at most 4 on the broker-visit history)
"""
import os, sys, json, collections, warnings
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import pseudoprob as PP, tiebraid as TB
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(20261110)
SIG = {"independent": np.eye(3), "visitors share (0.9)": np.array([[1, 0, .9], [0, 1, 0], [.9, 0, 1]]),
       "visitors opposed (-0.9)": np.array([[1, 0, -.9], [0, 1, 0], [-.9, 0, 1]]), "broker+visitor1 (0.9)": np.array([[1, .9, 0], [.9, 1, 0], [0, 0, 1]])}

# PP1
dev = {}
for nm, om in list((nm, PP.sample_circular_gaussian(S, 200000, rng)) for nm, S in SIG.items()) + [("rotated non-Gaussian", PP.sample_rotated(200000, 3, rng))]:
    dev[nm] = max(float(np.abs(np.histogram(np.angle(om[:, a] - om[:, b]), bins=12, range=(-np.pi, np.pi))[0] / len(om) - 1 / 12).max()) for a, b in ((0, 1), (0, 2), (1, 2)))
check("PP1 IDENTITY pairwise arguments are uniform under every circular law", max(dev.values()) < 0.004, f"largest deviation of a 12-bin histogram from 1/12: { {k: round(v, 4) for k, v in dev.items()} }")

# PP2
ks = {}
for S, (x, y, z) in ((SIG["independent"], (0, 1, 2)), (SIG["visitors share (0.9)"], (1, 0, 2)), (SIG["visitors opposed (-0.9)"], (1, 0, 2))):
    rho = PP.corr_of_differences(S, x, y, z); om = PP.sample_circular_gaussian(S, 200000, rng)
    psi = np.angle((om[:, x] - om[:, y]) * np.conj(om[:, x] - om[:, z]))
    g = np.linspace(-np.pi, np.pi, 40001); cdf = np.cumsum(PP.phase_diff_density(g, rho)) * (g[1] - g[0])
    ks[round(rho, 3)] = float(stats.kstest(psi, lambda v: np.interp(v, g, cdf)).statistic)
check("PP2 IDENTITY phase-difference density of correlated circular Gaussians", max(ks.values()) < 0.01, f"Kolmogorov distances by rho: { {k: round(v, 4) for k, v in ks.items()} }")

# PP3
Z = PP.broker_tie("interleaved"); marg = {}; joint = {}
for nm, S in SIG.items():
    om = PP.sample_circular_gaussian(S, 3000, rng); lks = [TB.linking(*TB.regularised(Z, o)[:2], 3) for o in om]
    marg[nm] = [float(np.mean([l[i] == -1 for l in lks])) for i in range(3)]; joint[nm] = float(np.mean([l == (-1, -1, -1) for l in lks]))
mspread = max(max(v[i] for v in marg.values()) - min(v[i] for v in marg.values()) for i in range(3))
check("PP3 FINDING tie history: pairwise linking laws do not depend on the bias correlation, the joint law does", mspread < 0.05 and max(joint.values()) - min(joint.values()) > 0.03,
      f"P(lk = -1) per pair (sector angle pi/2 gives 0.25): { {k: [round(x, 3) for x in v] for k, v in marg.items()} }; P(all three -1): { {k: round(v, 3) for k, v in joint.items()} }")

# PP4
Zv = PP.broker_visit("interleaved"); Wv = TB.walls(Zv); a01 = Wv[(0, 1)]; a12 = Wv[(1, 2)]
arcs = lambda a: {1: (a[0], a[1]), 0: (a[1], a[0] + 2 * np.pi)}
rows = {}; ok = True
for nm, S in SIG.items():
    om = PP.sample_circular_gaussian(S, 4000, rng); rec = collections.Counter()
    for o in om:
        w, init, gen, dm = TB.regularised(Zv, o); lk = TB.linking(w, init, 3); mu = TB.triple(w, init, 0, 1, 2) if all(v == 0 for v in lk) else None
        rec[(TB._sector(o[0] - o[1], a01), TB._sector(o[1] - o[2], a12), mu)] += 1
    enc = {(s1, s2) for (s1, s2, mu), c in rec.items() if mu == 1}
    cov = S[0, 1] - S[0, 2] - S[1, 1] + S[1, 2]; rho = cov / np.sqrt((S[0, 0] + S[1, 1] - 2 * S[0, 1]) * (S[1, 1] + S[2, 2] - 2 * S[1, 2]))
    exact = sum(PP.joint_sector_prob(arcs(a01)[s1], arcs(a12)[s2], rho, n=1500) for s1, s2 in enc)
    mc = sum(c for (s1, s2, mu), c in rec.items() if mu == 1) / len(om); se = np.sqrt(exact * (1 - exact) / len(om))
    pure = all((mu == 1) == ((s1, s2) in enc) for (s1, s2, mu) in rec)
    ok &= len(enc) == 1 and pure and abs(mc - exact) < 3 * se; rows[nm] = (round(float(rho), 3), round(exact, 4), round(mc, 4))
zero = all(TB.triple(*TB.regularised(PP.broker_visit(o_), PP.sample_circular_gaussian(np.eye(3), 1, rng)[0])[:2], 0, 1, 2) == 0 for o_ in ("nested", "sequential") for _ in range(200))
check("PP4 IDENTITY exact law of the triple invariant on interleaved visits (phase-difference density) matches Monte Carlo; nested and sequential give 0",
      ok and zero, f"(rho, exact P(mu = 1), Monte Carlo): {rows}")

# PP5
v = {T: float(np.var([PP.winding_turns(np.exp(1j * rng.uniform(0, 2 * np.pi, T)) * rng.exponential(1, T)) for _ in range(20000)])) for T in (5, 20, 80)}
check("PP5 IDENTITY independent jitter: winding variance (T - 1)/12", all(abs(v[T] / ((T - 1) / 12) - 1) < 0.05 for T in v), f"{ {T: (round(x, 3), round((T - 1) / 12, 3)) for T, x in v.items()} }")

# PP6
sech_cdf = lambda x: (2 / np.pi) * np.arctan(np.exp(np.pi * x / 2)); out = {}
for T in (100, 1000, 10000):
    W = []
    for _ in range(1500):
        p_ = 1 + np.cumsum(rng.standard_normal(T) + 1j * rng.standard_normal(T)); W.append(2 * np.pi * PP.winding_turns(np.r_[1, p_]))
    X = 2 * np.array(W) / np.log(T)
    out[T] = dict(median_abs=round(float(np.median(np.abs(X))), 3), ks_cauchy=round(float(stats.kstest(X, stats.cauchy.cdf).statistic), 3), ks_sech=round(float(stats.kstest(X, sech_cdf).statistic), 3))
check("PP6 FINDING random-walk winding moves towards the hyperbolic secant law (median |.| 0.561), away from Cauchy (median |.| 1)",
      out[10000]["ks_sech"] < out[10000]["ks_cauchy"] and out[10000]["median_abs"] < out[100]["median_abs"] < 1.0, f"{out}")

# PP7
links = set()
for o in PP.sample_circular_gaussian(np.eye(3), 2000, rng):
    w, init, _, _ = TB.regularised(Zv, o); links.add((TB.linking(w, init, 3), TB.triple(w, init, 0, 1, 2)))
check("PP7 IDENTITY persistent bias: finitely many links, one per resolution", len(links) <= 4, f"distinct (linking numbers, triple invariant) on the broker-visit history: {sorted(links, key=str)}")
json.dump(dict(universality=dev, ks=ks, tie_marginals=marg, tie_joint=joint, visit=rows, jitter=v, random_walk=out), open(os.path.join(HERE, "pseudoprob_summary.json"), "w"), indent=1, default=str)
print(f"{sum(results)}/{len(results)} passed")
