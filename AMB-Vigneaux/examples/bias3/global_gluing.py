"""Global gluing failure with local gluing: planted into real BASIL stance data, then on a ring of outlets.

Why regions must differ. If every pair of outlets is compared on the SAME events, the paired differences
telescope (d_AB + d_BC + d_CA = 0 for every event), so the loop always closes. A global failure needs the
regions to be different events, as when outlets cover overlapping but different stories.

(1) BASIL, disjoint regions. Split the 100 events at random into three thirds and compare
      Fox-NYT on third 1,  NYT-HPO on third 2,  HPO-Fox on third 3.
    Each region is a single edge, so it glues trivially: any lean fits one edge. The loop around the triangle is
      Phi = m(Fox-NYT) + m(NYT-HPO) + m(HPO-Fox),     m = mean paired difference in that region.
    With a consistent global scale, E[Phi] = 0 and Phi is pure noise. PLANT: in region 3 only, HPO's stance is
    read on a scale shifted by delta (clipped to [-2, 2]), i.e. HPO is locally consistent in every region but
    not on one global scale. Test: z = Phi / se(Phi), with se from the three regional standard errors.
    Calibrated by the false-positive rate at delta = 0 over many random splits.
(2) Ring of K outlets (no chords). Edges are compared on n events each, with per-event noise resampled from
    BASIL's real paired-difference residuals. Every proper arc of the ring is a tree, so it glues. Only the
    whole ring can fail. Planted holonomy h is spread evenly over the edges (h/K each), so no single edge
    looks unusual. Power of the loop test against h and n.
"""
import json, itertools
import numpy as np

import bias3

rng = np.random.default_rng(0)
E = bias3.load()


def loop_test(regions):
    """regions: list of arrays of paired differences along the oriented cycle"""
    m = np.array([r.mean() for r in regions]); se2 = np.array([r.var(ddof=1) / len(r) for r in regions])
    phi = m.sum(); z = phi / np.sqrt(se2.sum())
    return phi, z


def basil_split(delta, reps=2000):
    hits, phis = 0, []
    idx = np.arange(len(E))
    for _ in range(reps):
        p = rng.permutation(idx); t1, t2, t3 = p[:33], p[33:66], p[66:]
        d1 = np.array([E[i]["fox"] - E[i]["nyt"] for i in t1], float)
        d2 = np.array([E[i]["nyt"] - E[i]["hpo"] for i in t2], float)
        d3 = np.array([np.clip(E[i]["hpo"] + delta, -2, 2) - E[i]["fox"] for i in t3], float)
        phi, z = loop_test([d1, d2, d3])
        phis.append(phi); hits += abs(z) > 1.96
    return hits / reps, float(np.mean(phis)), float(np.std(phis))


# empirical noise: residuals of BASIL paired differences around each pair's mean
RES = np.concatenate([np.array([e[a] - e[b] for e in E], float) - np.mean([e[a] - e[b] for e in E])
                      for a, b in (("fox", "nyt"), ("fox", "hpo"), ("hpo", "nyt"))])


def ring(K, n, h, reps=1000):
    u = np.linspace(-1, 1, K)                                  # true house leans
    hits = 0
    for _ in range(reps):
        regs = []
        for i in range(K):
            j = (i + 1) % K
            regs.append((u[i] - u[j]) + h / K + rng.choice(RES, n))
        phi, z = loop_test(regs)
        hits += abs(z) > 1.96
    return hits / reps


if __name__ == "__main__":
    out = {"basil": {}, "ring": {}}
    print("(1) BASIL, three disjoint regions (Fox-NYT | NYT-HPO | HPO-Fox), planted HPO scale shift in region 3")
    print("    delta | detect (|z|>1.96) | mean Phi | sd Phi")
    for delta in (0.0, 0.5, 1.0, 1.5, 2.0):
        r, mphi, sphi = basil_split(delta)
        out["basil"][delta] = dict(detect=r, mean_phi=mphi, sd_phi=sphi)
        print(f"    {delta:4.1f}  |  {r:.3f}            | {mphi:+.2f}    | {sphi:.2f}")
    print("\n(2) ring of K outlets, n events per edge, BASIL noise, holonomy h spread over the ring")
    for K in (3, 6):
        for n in (10, 33, 100):
            row = {h: ring(K, n, h) for h in (0.0, 0.5, 1.0, 2.0, 3.0)}
            out["ring"][f"K={K},n={n}"] = row
            print(f"    K={K} n={n:3d}: " + "  ".join(f"h={h}: {v:.2f}" for h, v in row.items()))
    json.dump(out, open("global_gluing_results.json", "w"), indent=1)
