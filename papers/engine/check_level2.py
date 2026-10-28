"""check_level2.py -- partial gluing graded by the structure saved, and level two: latent types, singular fibres, second-order obstructions,
Puiseux branches and monodromy (note: level2/partial_gluing.pdf).

  LV1 graded explanations: E_alpha (laws accounting for a fraction alpha of every table) is non-empty iff alpha <= lam*, and the
      prediction interval of any event shrinks as alpha grows (nested) -- 60 random incoherent families
  LV2 the drug example with conflicting trials: lam* = 0.86; at every alpha the most probable (max-entropy) explanation lies in the interval;
      at alpha = lam* the interval contains the truth of both trial populations
  LV3 the latent-class map F(w, a): the exact second derivative, Taylor error O(eps^3)
  LV4 regular and singular points: dF has full rank 2m + 1 at generic points; where a type's weight vanishes its shape is a free direction
      (kernel, no obstruction); where two types coincide the second-order obstruction kappa is non-zero (l_2 != 0)
  LV5 the tangent cone at a coincidence is a union of branches: moving the weights, or splitting the two types along one variable (kappa = 0,
      and each is realised by an exact path in the fibre); splitting along two variables is obstructed (kappa != 0)
  LV6 a straight path of observed laws across the independence locus: the two types separate like sqrt(s) (Puiseux branches +/- sqrt(s),
      one per labelling); on the other side no two-type explanation exists (positive residual; the product of the three pairwise
      covariances of a two-type model is a square)
  LV7 monodromy: with three variables a loop that exchanges the two types' roles must pass through a singularity (dF loses rank where a gap
      reaches 0); with four variables a role-exchanging loop stays in the regular locus (labels swap, Z/2 monodromy); the full loop
      returns the labels; the observed law is single-valued
  LV8 latent structure sharpens prediction: the two-type interval for the triple-combination harm lies inside the level-one interval,
      contains the truth and is at least 3 times narrower
  LV9 graded gluing at level two: beyond the fold the best two-type explanation saves lam*_2 < 1 of the observed law (attained values,
      lower bounds), yet three types fit exactly (a new actor is born); for the conflicting trials lam*_2 reaches lam* = 0.86
Writes level2/level2_results.json.  Run: python check_level2.py
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import Window
from pce.integrate import gluable_fraction
from pce.level2 import (predict, most_probable, lcm_law, jacobian, second, kuranishi, kappa, fit, track, fiber_range, lam2)

rng = np.random.default_rng(20261023)
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))
OUT = {}
D3 = {"A": 2, "B": 2, "C": 2}; PAIRS = [("A", "B"), ("B", "C"), ("A", "C")]
harm = lambda x: x["A"] or x["B"] or x["C"]
def pair_tables(P): return [P.sum(axis=2), P.sum(axis=0), P.sum(axis=1)]

# LV1
ok = True; n_inc = 0
for _ in range(60):
    T = [rng.dirichlet(np.ones(4)).reshape(2, 2) for _ in range(3)]; W = [Window(p, t) for p, t in zip(PAIRS, T)]
    lam = gluable_fraction(D3, W).lam; n_inc += lam < 1 - 1e-7; prev = None
    for al in np.linspace(0.05, lam, 6):
        iv = predict(D3, W, harm, al); ok &= iv is not None
        if prev is not None: ok &= iv[0] >= prev[0] - 1e-9 and iv[1] <= prev[1] + 1e-9
        prev = iv
    ok &= predict(D3, W, harm, min(1.0, lam + 0.01)) is None or lam > 0.99
check("LV1 graded explanations exist iff alpha <= lam*, and prediction intervals shrink as more structure is saved", ok, f"{n_inc}/60 incoherent")

# LV2
aT = np.array([[0.5, 0.1, 0.4], [0.05, 0.5, 0.1]]); pop1 = lcm_law(np.array([0.6, 0.4]), aT); pop2 = lcm_law(np.array([0.25, 0.75]), aT)
W = [Window(("A", "B"), pop1.sum(axis=2)), Window(("B", "C"), pop2.sum(axis=0)), Window(("A", "C"), pop1.sum(axis=1))]
lam = gluable_fraction(D3, W).lam; rows = []; ok = abs(lam - 0.86) < 1e-6
for al in list(np.linspace(0.05, lam, 18)):
    iv = predict(D3, W, harm, al); q, atoms = most_probable(D3, W, al); mp = sum(v for a, v in zip(atoms, q) if any(a))
    ok &= iv[0] - 1e-6 <= mp <= iv[1] + 1e-6; rows.append([float(al), iv[0], iv[1], float(mp)])
truths = [1 - pop1[0, 0, 0], 1 - pop2[0, 0, 0]]; ok &= all(rows[-1][1] - 1e-9 <= x <= rows[-1][2] + 1e-9 for x in truths)
OUT["graded"] = {"lam": lam, "rows": rows, "truths": truths}
check("LV2 conflicting trials: lam* = 0.86; the most probable explanation stays in the interval; at lam* both populations' truths are inside",
      ok, f"interval at lam* [{rows[-1][1]:.3f}, {rows[-1][2]:.3f}], truths {truths[0]:.3f}, {truths[1]:.3f}")

# LV3
ok = True; worst = 0.0
for _ in range(50):
    m = int(rng.integers(3, 5)); w = rng.dirichlet(np.ones(2)); a = rng.uniform(0.05, 0.95, (2, m)); v = rng.normal(0, 1, 2 + 2 * m); v[:2] -= v[:2].mean()
    errs = []
    for eps in (1e-2, 1e-3):
        lhs = lcm_law(w + eps * v[:2], a + eps * v[2:].reshape(2, m)).ravel() - lcm_law(w, a).ravel()
        errs.append(np.abs(lhs - eps * jacobian(w, a) @ v - 0.5 * eps ** 2 * second(w, a, v)).max())
    ok &= errs[1] < errs[0] / 300 + 1e-15; worst = max(worst, errs[1])
check("LV3 exact second derivative of the latent-class map (Taylor error falls like eps^3)", ok, f"largest error at eps = 1e-3: {worst:.1e}")

# LV4
ok = True
for _ in range(40):
    m = 3; w = rng.dirichlet(np.ones(2)) * 0.8 + 0.1; a = rng.uniform(0.05, 0.95, (2, m))
    K, C, r = kuranishi(w, a); ok &= r == 2 * m + 1 and K.shape[1] == 0
w0 = np.array([1.0, 0.0]); a0 = rng.uniform(0.1, 0.9, (2, 3)); K, C, r = kuranishi(w0, a0)
ok &= r == 4 and K.shape[1] == 3 and all(np.linalg.norm(kappa(w0, a0, K[:, j], C)) < 1e-12 for j in range(3))
wc = np.array([0.6, 0.4]); ac = np.vstack([a0[0], a0[0]]); Kc, Cc, rc = kuranishi(wc, ac)
kc = [np.linalg.norm(kappa(wc, ac, Kc @ rng.normal(0, 1, Kc.shape[1]), Cc)) for _ in range(20)]
ok &= rc < 7 and Cc.shape[1] > 0 and min(kc) > 1e-6
check("LV4 generic points regular (rank 2m+1); vanishing weight: free shape, no obstruction; coinciding types: l_2 != 0", ok,
      f"ranks: generic 7, vanishing weight {r}, coincidence {rc}; |kappa| on random kernel vectors >= {min(kc):.3f}")

# LV5
a1 = np.array([0.3, 0.5, 0.7]); ac = np.vstack([a1, a1]); wc = np.array([0.6, 0.4]); Kc, Cc, rc = kuranishi(wc, ac); J = jacobian(wc, ac)
def split(e): return np.r_[0, 0, -(wc[1] / wc[0]) * e, e]
branches = {"weights": np.r_[1, -1, np.zeros(6)]}
for i in range(3): e = np.zeros(3); e[i] = 1; branches[f"split X{i + 1}"] = split(e)
ok = all(np.linalg.norm(J @ v) < 1e-12 and np.linalg.norm(kappa(wc, ac, v, Cc)) < 1e-12 for v in branches.values())
P0 = lcm_law(wc, ac)
for i in range(3):                                                         # exact paths: split along one variable keeping the mean
    for s in (0.05, 0.1):
        e = np.zeros(3); e[i] = s; ok &= np.abs(lcm_law(wc, np.vstack([a1 - (wc[1] / wc[0]) * e, a1 + e])) - P0).max() < 1e-12
for s in (0.05, 0.2): ok &= np.abs(lcm_law(np.array([0.6 + s, 0.4 - s]), ac) - P0).max() < 1e-12
mixed = []
for _ in range(20):
    e = np.zeros(3); idx = rng.choice(3, 2, replace=False); e[idx] = rng.normal(0, 1, 2); mixed.append(np.linalg.norm(kappa(wc, ac, split(e), Cc)))
ok &= min(mixed) > 1e-6
phis = np.linspace(0, np.pi, 181)
OUT["cone"] = {"phi": phis.tolist(), "kappa": [float(np.linalg.norm(kappa(wc, ac, split(np.array([np.cos(f), np.sin(f), 0.0])), Cc))) for f in phis]}
check("LV5 tangent cone at a coincidence = union of branches (weights; split along one variable), each an exact path; mixed splits obstructed",
      ok, f"|kappa| on two-variable splits >= {min(mixed):.3f}")

# LV6
c = np.array([0.4, 0.5, 0.6]); e = np.array([0.5, 0.4, 0.3]); wh = np.array([0.5, 0.5])
P0 = lcm_law(wh, np.vstack([c, c])); B = (lcm_law(wh, np.vstack([c + 0.1 * e, c - 0.1 * e])) - P0) / 0.01
ok = all(np.abs(lcm_law(wh, np.vstack([c + np.sqrt(s) * e, c - np.sqrt(s) * e])) - (P0 + s * B)).max() < 1e-13 for s in (0.0025, 0.01, 0.04))
ss = np.array([0.04, 0.02, 0.01, 0.005, 0.0025]); gaps = []
for s in ss:
    wf, af = fit(P0 + s * B, 2, (np.array([0.55, 0.45]), np.vstack([c + 0.3, c - 0.3])), bounded=True); gaps.append(float(np.abs(af[0] - af[1]).max()))
ok &= np.allclose(np.array(gaps) / np.sqrt(ss), 2 * e.max(), atol=1e-6)
neg = []
for s in (-0.005, -0.01, -0.02, -0.04):
    best = min(np.abs(lcm_law(*fit(P0 + s * B, 2, (np.array([0.5, 0.5]) + rng.normal(0, 0.1, 2) * [1, -1], rng.uniform(0.05, 0.95, (2, 3))), bounded=True)) - (P0 + s * B)).max() for _ in range(25))
    neg.append(float(best))
cov = lambda P: [P.sum(axis=2)[1, 1] - P.sum(axis=(1, 2))[1] * P.sum(axis=(0, 2))[1], P.sum(axis=0)[1, 1] - P.sum(axis=(0, 2))[1] * P.sum(axis=(0, 1))[1], P.sum(axis=1)[1, 1] - P.sum(axis=(1, 2))[1] * P.sum(axis=(0, 1))[1]]
ok &= min(neg) > 1e-5 and np.prod(cov(P0 - 0.01 * B)) < 0
OUT["puiseux"] = {"s": ss.tolist(), "gap": gaps, "neg_s": [-0.005, -0.01, -0.02, -0.04], "neg_res": neg}
check("LV6 crossing the independence locus: types separate like sqrt(s) (two Puiseux branches); on the other side no two-type explanation",
      ok, "gap/sqrt(s) = " + f"{gaps[0] / np.sqrt(ss[0]):.4f}" + "; residuals for s < 0: " + ", ".join(f"{x:.4f}" for x in neg))

# LV7
def loop(m, u, v, cc, T, n=500, r=0.22, rho=0.15):
    u, v, cc = map(lambda x: np.array(x, float), (u, v, cc))
    def par(th):
        d = r * (np.cos(th) * u + np.sin(th) * v); return np.array([0.5 + rho * np.cos(th), 0.5 - rho * np.cos(th)]), np.vstack([cc + d, cc - d])
    ths = np.linspace(0, T, n); path = [lcm_law(*par(t)) for t in ths]
    smin = [float(np.linalg.svd(np.vstack([jacobian(*par(t)), np.r_[np.ones(2), np.zeros(2 * m)]]), compute_uv=False)[-1]) for t in ths]
    tr = track(path, 2, par(0)); w0, a0 = par(0); wE, aE = tr[-1]
    return {"closes": float(np.abs(path[-1] - path[0]).max()), "smin": min(smin), "swap": bool(np.allclose(aE[0], a0[1], atol=1e-6) and np.allclose(wE[0], w0[1], atol=1e-6)),
            "identity": bool(np.allclose(aE[0], a0[0], atol=1e-6)), "trace": [x[1].tolist() for x in tr[::10]], "smin_path": smin[::5]}
L3 = loop(3, [1, 0.6, -0.4], [0.3, 1, 0.8], [0.5] * 3, np.pi, n=2001)
L4 = loop(4, [1, 0.6, -0.4, 0.2], [0.3, 1, 0.8, -0.9], [0.5] * 4, np.pi)
L4f = loop(4, [1, 0.6, -0.4, 0.2], [0.3, 1, 0.8, -0.9], [0.5] * 4, 2 * np.pi)
ok = L3["smin"] < 1e-3 and L4["smin"] > 5e-3 and L4["swap"] and L4f["identity"] and max(L3["closes"], L4["closes"], L4f["closes"]) < 1e-12
OUT["monodromy"] = {"m3": L3, "m4": L4}
check("LV7 role exchange: with 3 variables it must pass a singularity; with 4 a regular loop swaps the labels (Z/2 monodromy); full loop identity",
      ok, f"smallest singular value along the loop: m=3 {L3['smin']:.1e}, m=4 {L4['smin']:.4f}")

# LV8
PT = pop1; target = np.concatenate([t.ravel() for t in pair_tables(PT)])
lo1, hi1 = predict(D3, [Window(p, t) for p, t in zip(PAIRS, pair_tables(PT))], harm, 1.0)
lo2, hi2, sols = fiber_range(lambda P: np.concatenate([t.ravel() for t in pair_tables(P)]), target, 2, 3,
                             lambda w, a: 1 - lcm_law(w, a)[0, 0, 0], np.linspace(0.02, 0.98, 49), starts=12)
truth = 1 - PT[0, 0, 0]
# exact two-type interval: the fibre over the pair tables is the hiding segment, and the triple harm is affine along it (lab paper,
# Theorem A); its ends are at the box walls. The grid trace above (49 weights) is kept as a cross-check: it lies inside.
from pce.lab import moments as _mom, fit_v as _fv, segment as _seg, explanation as _expl
_mu, _M, _ = _mom(PT); _v = _fv(_mu, _M); _lo, _hi = _seg(_mu, _v)
_ends = sorted(1 - lcm_law(np.array([_w, 1 - _w]), np.vstack([_a1, _a2]))[0, 0, 0] for _w, _a1, _a2 in (_expl(_mu, _v, _lo), _expl(_mu, _v, _hi)))
lo2x, hi2x = float(_ends[0]), float(_ends[1])
ok = lo1 - 1e-9 <= lo2x <= truth + 1e-6 and truth - 1e-6 <= hi2x <= hi1 + 1e-9 and (hi1 - lo1) >= 3 * (hi2x - lo2x) and lo2x - 1e-9 <= lo2 and hi2 <= hi2x + 1e-9
OUT["sharpen"] = {"level1": [lo1, hi1], "level2": [lo2x, hi2x], "level2_grid": [lo2, hi2], "truth": truth}
check("LV8 latent structure sharpens prediction: the two-type interval (exact, ends at the box walls) sits inside the level-one interval, contains the truth, >= 3x narrower; the grid trace lies inside it",
      ok, f"level one [{lo1:.3f}, {hi1:.3f}], two types [{lo2x:.5f}, {hi2x:.5f}] (grid [{lo2:.4f}, {hi2:.4f}]), truth {truth:.3f}")

# LV9
flat = lambda P: P.ravel(); l2 = {}
for s in (-0.0025, -0.005, -0.01, -0.02, -0.04): l2[s] = lam2(flat, (P0 + s * B).ravel(), 2, 3, starts=60, seed=1)[0]
three = []
for s in (-0.01, -0.04):
    best = 9.0
    for _ in range(30):
        w3, a3 = fit(P0 + s * B, 3, (rng.dirichlet(np.ones(3)), rng.uniform(0.05, 0.95, (3, 3))), bounded=True)
        if w3.min() >= -1e-9: best = min(best, float(np.abs(lcm_law(w3, a3) - (P0 + s * B)).max()))
    three.append(best)
pt = lambda P: np.concatenate([t.ravel() for t in pair_tables(P)])
tgt = np.concatenate([pop1.sum(axis=2).ravel(), pop2.sum(axis=0).ravel(), pop1.sum(axis=1).ravel()])
l2drug = lam2(pt, tgt, 2, 3, starts=60, seed=2)[0]
ok = all(0.9 < v < 1 for v in l2.values()) and max(three) < 1e-10 and abs(l2drug - 0.86) < 1e-4
OUT["lam2"] = {"s": list(l2), "lam2": list(l2.values()), "three_type_residual": three, "drug": l2drug}
check("LV9 level-two graded gluing: beyond the fold two types save lam*_2 < 1, three types fit exactly; conflicting trials lam*_2 = lam*",
      ok, "lam*_2: " + ", ".join(f"s={s}: {v:.4f}" for s, v in l2.items()) + f"; three-type residual {max(three):.1e}; drug {l2drug:.4f}")

os.makedirs(os.path.join(HERE, "..", "level2"), exist_ok=True)
json.dump(OUT, open(os.path.join(HERE, "..", "level2", "level2_results.json"), "w"))
print(f"\n{sum(results)}/{len(results)} checks passed")
