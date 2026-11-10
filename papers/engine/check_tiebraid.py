"""Checks for 'The bridge: ties, resolutions and probability' (companion note; module pce/tiebraid.py, on the histories of
pce/bundlebraid). IDENTITY = an exact instance of a statement proved in the text; FINDING = a computed fact about the toy.
  TB1 IDENTITY  coherent histories (drift, shock, matched drift, pump and retreat) and the static contextual world have no walls: one
                resolution, and the closed history is the trivial link for every regularising direction
  TB2 IDENTITY  stability: for every history and 25 random directions omega, the link of the regularised closed history (pairwise linking
                numbers) is the same for eps in {1e-4, 1e-5, 1e-6} and projection angles {0.123, 0.9, 2.2}; every regularised loop is
                collision-free and its projection generic
  TB3 IDENTITY  the link (linking numbers, and triple invariants where the triple's linking numbers vanish) is a function of the resolution
                (400 directions per history)
  TB4 FINDING   how much topology the data leave open: number of resolutions and of link types, and the entropy of the resolution under
                isotropic Gaussian directions; lockstep coalitions are determined, the liar leaves 3, framing and the cascade leave >= 150
  TB5 IDENTITY  the linking number of a pair depends only on that pair's sector (so it is a degree-1 functional of the pair's resolution)
  TB6 IDENTITY  under isotropic Gaussian directions each sector has probability equal to its angle over 2 pi (Monte Carlo, 4000 draws)
  TB7 FINDING   for every triple whose linking numbers vanish, the triple invariant is 0 in every sampled resolution of the liar, framing
                and cascade histories: the topology the data leave open is pairwise (degree 1 in the pairs' resolutions)
  TB8 FINDING   per-step noise (0.003) does not remove the ties, and it makes tied strands wind at random: in nine noisy closed histories
                every linking number outside the clean resolution set is on a pair tied in the clean data, and every pair the data separate
                keeps a linking number the clean resolutions allow
"""
import os, sys, itertools, math, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from pce import tiebraid as TB, bundlebraid as BB, watchtoy as WT
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
W = WT.setup(); g = W["gB"]; k = 7
NAMES = ["drift", "shock", "matched", "pump_retreat", "contextual", "redundant_liar", "framing", "cascade", "efficient", "stealthy"]
Z, NOISY = {}, {}
for nm in NAMES:
    S, V, _ = BB.features(BB.history(nm, W), g); Z[nm] = TB.closed(S + 1j * V)
for nm in ("redundant_liar", "framing", "cascade"):
    NOISY[nm] = []
    for sd in (1, 2, 3):
        S, V, _ = BB.features(BB.history(nm, W), g, noise=0.003, seed=sd); NOISY[nm].append(TB.closed(S + 1j * V))
Wl = {nm: TB.walls(Z[nm]) for nm in NAMES}
rng = np.random.default_rng(20261109)
def omega(): return rng.standard_normal(k) + 1j * rng.standard_normal(k)

# TB1
coh = ["drift", "shock", "matched", "pump_retreat", "contextual"]; ok = True; info = {}
for nm in coh:
    nw = sum(len(v) for v in Wl[nm].values()); lks = set()
    for _ in range(25):
        w, init, gen, dmin = TB.regularised(Z[nm], omega()); lks.add(TB.linking(w, init, k)); ok &= gen and dmin > 0
    ok &= nw == 0 and lks == {tuple([0] * 21)}; info[nm] = nw
check("TB1 IDENTITY coherent and static histories: no walls, the trivial link for every direction", ok, f"walls {info}")

# TB2
bad = 0; tested = 0
for nm in NAMES:
    for _ in range(25):
        om = omega(); ref = None
        for eps in (1e-4, 1e-5, 1e-6):
            for phi in (0.123, 0.9, 2.2):
                w, init, gen, dmin = TB.regularised(Z[nm], om, eps, phi); lk = TB.linking(w, init, k); tested += 1
                if not gen or dmin <= 0: bad += 1
                if ref is None: ref = lk
                elif lk != ref: bad += 1
check("TB2 IDENTITY the regularised link does not depend on eps or the projection, and every loop is generic and collision-free",
      bad == 0, f"{tested} regularised loops over {len(NAMES)} histories, {bad} disagreements")

# TB3, TB4, TB5
summary = {}; func_ok = True; pair_ok = True; samples = {}
for nm in NAMES:
    res_map = {}; pair_map = {}; recs = []
    for _ in range(400):
        om = omega(); w, init, gen, dmin = TB.regularised(Z[nm], om); lk = TB.linking(w, init, k)
        r = TB.resolution(Wl[nm], om); res_map.setdefault(r, set()).add(lk); recs.append((om, w, init, lk, r))
        for idx, (a, b) in enumerate(itertools.combinations(range(k), 2)):
            sec = TB._sector(om[a] - om[b], Wl[nm][(a, b)]); pair_map.setdefault((a, b, sec), set()).add(lk[idx])
    func_ok &= all(len(v) == 1 for v in res_map.values()); pair_ok &= all(len(v) == 1 for v in pair_map.values())
    cnt = {}
    for rec in recs: cnt[rec[4]] = cnt.get(rec[4], 0) + 1
    p = np.array(list(cnt.values())) / len(recs); H = float(-(p * np.log2(p)).sum())
    summary[nm] = dict(walls=sum(len(v) for v in Wl[nm].values()), resolutions=len(res_map), links=len(set().union(*res_map.values())), entropy_bits=round(H, 2))
    samples[nm] = recs
check("TB3 IDENTITY the link is a function of the resolution", func_ok, "linking numbers grouped by resolution vector, 400 directions per history")
check("TB4 FINDING the data determine the topology for coherent and lockstep histories and leave it open for the liar, framing and the cascade",
      all(summary[n]["resolutions"] == 1 for n in coh + ["efficient", "stealthy"]) and summary["redundant_liar"]["resolutions"] == 3
      and min(summary["framing"]["resolutions"], summary["cascade"]["resolutions"]) >= 150, f"{summary}")
check("TB5 IDENTITY each pair's linking number depends only on that pair's sector", pair_ok, "all pairs, all histories")
json.dump(summary, open(os.path.join(HERE, "tiebraid_summary.json"), "w"), indent=1)

# TB6
worst = 0.0; nsec = 0
for nm in ("redundant_liar", "framing", "cascade"):
    for (a, b), angs in Wl[nm].items():
        if len(angs) < 2: continue
        pr = TB.sector_probability(angs); cnt = np.zeros(len(pr))
        for _ in range(4000):
            om = omega(); cnt[TB._sector(om[a] - om[b], angs)] += 1
        worst = max(worst, float(np.abs(cnt / 4000 - pr).max())); nsec += len(pr)
check("TB6 IDENTITY sector probabilities are angles over 2 pi under isotropic Gaussian directions", worst < 0.03,
      f"{nsec} sectors; largest Monte Carlo deviation {worst:.4f} (4000 draws; 3 standard errors at p = 1/2 is 0.024)")

# TB7
pidx = {pr: i for i, pr in enumerate(itertools.combinations(range(k), 2))}; nz = 0; ntrip = 0
for nm in ("redundant_liar", "framing", "cascade"):
    for om, w, init, lk, r in samples[nm][:200]:
        for T3 in itertools.combinations(range(k), 3):
            if all(lk[pidx[p]] == 0 for p in itertools.combinations(T3, 2)):
                ntrip += 1; nz += TB.triple(w, init, *T3) != 0
check("TB7 FINDING triple invariants vanish wherever they are defined: the open topology is pairwise", nz == 0 and ntrip > 1000,
      f"{ntrip} (resolution, triple) cases with zero linking numbers; non-zero triple invariants: {nz}")

# TB8
pairs = list(itertools.combinations(range(k), 2)); ok = True; walls_noisy = []; nbad = []
for nm, Zs in NOISY.items():
    allowed = {}
    for om, w, init, lk, r in samples[nm]:
        for i, v in enumerate(lk): allowed.setdefault(i, set()).add(v)
    tied = {i for i, (a, b) in enumerate(pairs) if np.any(Z[nm][:, a] == Z[nm][:, b])}
    for Zn in Zs:
        walls_noisy.append(sum(len(v) for v in TB.walls(Zn).values()))
        w, init, gen, dmin = TB.regularised(Zn, omega(), 1e-9); lk = TB.linking(w, init, k)
        bad = {i for i, v in enumerate(lk) if v not in allowed[i]}; nbad.append(len(bad))
        ok &= bad <= tied and all(lk[i] in allowed[i] for i in range(len(pairs)) if i not in tied)
check("TB8 FINDING noise keeps the ties and winds tied strands at random; separated pairs keep their clean linking", ok and max(nbad) > 0 and min(walls_noisy) > 0,
      f"walls remaining in the nine noisy closed histories {walls_noisy}; pairs with linking outside the clean set {nbad}, all of them tied in the clean data")
print(f"{sum(results)}/{len(results)} passed")
