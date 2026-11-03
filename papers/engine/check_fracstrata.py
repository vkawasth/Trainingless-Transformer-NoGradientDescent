"""Checks for Section "Fracture strata" of "Probability, Context and Differentiation: The Layers" (module pce/fracstrata.py).
  FS1  the fracture loci of the blueprint's flag world, over the closed quadrant (s, t) >= 0: definedness on the two axes, the act on the
       ray t = s/2, the value of information on the ray t = 4s/7 and the axis t = 0 (exact, 600 random points plus points on every ray)
  FS2  every invariant is eventually constant along arcs t = c s^w, and its eventual value is read from (w, c) through the fans: act a for
       w > 1, act b for w < 1, sign(1 - 7c/4 - c^3) on w = 1; value of information positive iff w > 1, or w = 1 and c < 4/7 (exact)
  FS3  the window world (theta, kappa): coherence changes exactly at theta*(kappa) and nowhere for kappa below the onset; b0 of the
       model fibre is 2 for every kappa > 0 (whatever theta) and 1 at kappa = 0; the two loci are disjoint
  FS4  the global fragility of the act: eps*(b), the least perturbation of the parameters that changes the act, equals the distance to the
       act's wall (brute force, 200 points), is 1-Lipschitz, vanishes exactly on the locus, and along arcs t = c s^w scales like s^min(1, w)
  FS5  reach is constant on each stratum: empty on the three open regions, {act} and {value of information} on the two rays,
       {definedness, value of information} on the axis t = 0, {definedness} on the axis s = 0, and all three at the corner
"""
import os, sys, random, math
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import fracstrata as FS
from pce import blueprint as BP
random.seed(20261102); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def rq(): return Fr(random.randint(1, 3000), 10000)
pts = [(rq(), rq()) for _ in range(600)]
s0 = [rq() for _ in range(40)]
on = {"act ray": [(s, s / 2) for s in s0], "VoI ray": [(s, Fr(4, 7) * s) for s in s0], "axis t=0": [(s, Fr(0)) for s in s0],
      "axis s=0": [(Fr(0), s) for s in s0], "corner": [(Fr(0), Fr(0))]}

# FS1
pred = {"definedness": lambda s, t: s == 0 or t == 0, "act": lambda s, t: (t == s / 2) or (s == 0 and t == 0),
        "value of information": lambda s, t: t == Fr(4, 7) * s or t == 0}
ok = True; bad = []
for b in pts + [q for L in on.values() for q in L]:
    for k in FS.KINDS:
        if FS.locus(k, b) != pred[k](*b): ok = False; bad.append((k, b))
check("FS1 fracture loci: definedness on the axes, act on t = s/2, value of information on t = 4s/7 and t = 0 (exact)", ok, f"{len(pts)} random points, {sum(len(v) for v in on.values())} on walls; mismatches {bad[:3]}")

# FS2
ok = True; n = 0
for w in (Fr(1, 2), Fr(3, 4), Fr(1), Fr(3, 2), Fr(2)):
    for c in (Fr(1, 4), Fr(1, 2), Fr(4, 7), Fr(3, 5), Fr(1), Fr(2)):
        if w.denominator > 1: ss = [Fr(1, 10 ** (2 * k)) for k in range(3, 7)]; tt = [c * Fr(1, 10 ** k) ** int(2 * w) for k in range(3, 7)]
        else: ss = [Fr(1, 10 ** k) for k in range(3, 7)]; tt = [c * x ** int(w) for x in ss]
        vals = [FS.invariants(s, t) for s, t in zip(ss, tt)]
        ok &= all(v == vals[-1] for v in vals)                                # eventually constant
        act = 1 if w > 1 else (-1 if w < 1 else (lambda x: (x > 0) - (x < 0))(1 - Fr(7, 4) * c - c ** 3))
        voi = 1 if (w > 1 or (w == 1 and c < Fr(4, 7))) else 0
        ok &= vals[-1]["act"] == act and vals[-1]["value of information"] == voi; n += 1
check("FS2 every invariant is eventually constant along arcs, with the value read from (w, c) through the fans", ok, f"{n} arcs")

# FS3
ok = True; kas = np.linspace(0.05, 1.65, 17); changes = []
for k in kas:
    th = np.linspace(0, 1, 201); g = [BP.boole(BP.window_tables(x, k))[0] <= 1e-12 for x in th]
    ch = [i for i in range(1, len(g)) if g[i] != g[i - 1]]; changes.append(len(ch)); w = BP.theta_wall(k)
    if w is None: ok &= not ch
    else: ok &= len(ch) == 1 and th[ch[0] - 1] <= w <= th[ch[0]] + 1e-12
rng = np.random.default_rng(3); b0 = []
for k in (0.2, 0.7, 1.3):
    _, cl = BP.fibre_points(BP.source1(k), 40, rng); b0.append(len(cl))
seg = all(np.allclose(BP.twotype_law(w_, np.full(3, .5), np.full(3, .5)), BP.source1(0.0)) for w_ in np.linspace(0, 1, 11))
onset = min(k for k in kas if BP.theta_wall(k) is not None)
ok &= b0 == [2, 2, 2] and seg and onset > 0.3
check("FS3 coherence changes exactly at theta*(kappa); b0 = 2 for kappa > 0 and 1 at kappa = 0; the loci are disjoint", ok,
      f"wall crossings per vertical line {changes}; onset kappa ~ {onset:.2f}; b0 {b0}")

# FS4
ok = True; errs = []
def brute(b, K=360):
    best = 1.0; s, t = map(float, b); a0 = FS.invariants(*b)["act"]
    for i in range(K):
        ang = 2 * math.pi * i / K; d = (math.cos(ang), math.sin(ang)); lo = 0.0
        hi = min([0.6] + [-(s if j == 0 else t) / d[j] for j in range(2) if d[j] < 0]) * (1 - 1e-12)   # stay in the quadrant
        def diff(r):
            q = (s + r * d[0], t + r * d[1])
            return q[0] >= 0 and q[1] >= 0 and FS.invariants(Fr(q[0]), Fr(q[1]))["act"] != a0
        if not diff(hi): continue
        for _ in range(40):
            m = (lo + hi) / 2
            if diff(m): hi = m
            else: lo = m
        best = min(best, hi)
    return best
sample = pts[:200]
for b in sample:
    e = FS.eps_act(b); bb = brute(b); errs.append(abs(e - bb)); ok &= abs(e - bb) < 2e-3 * max(1, e) + 1e-6
lip = all(abs(FS.eps_act(p) - FS.eps_act(q)) <= math.dist(map(float, p), map(float, q)) + 1e-15 for p, q in zip(pts[:300], pts[300:]))
zero = all(FS.eps_act(b) == 0 for b in on["act ray"]) and all(FS.eps_act(b) > 0 for b in pts)
exps = []
for w, c in ((0.5, 1.0), (0.75, 1.0), (1.0, 1.0), (1.0, 0.25), (1.5, 1.0), (2.0, 1.0)):
    s1, s2 = 1e-14, 1e-18; e1 = FS.eps_act((s1, c * s1 ** w)); e2 = FS.eps_act((s2, c * s2 ** w))
    exps.append(round((math.log(e1) - math.log(e2)) / (math.log(s1) - math.log(s2)), 3)); ok &= abs(exps[-1] - min(1, w)) < 1e-3
check("FS4 global fragility of the act: eps* = distance to the wall, 1-Lipschitz, zero exactly on the locus, exponent min(1, w) along arcs",
      ok and lip and zero, f"largest brute-force error {max(errs):.1e}; exponents {exps}")

# FS5
regions = {"t < s/2": lambda s: s * Fr(random.randint(1, 99), 200), "s/2 < t < 4s/7": lambda s: s * (Fr(1, 2) + Fr(random.randint(1, 99), 100) * Fr(1, 14)),
           "t > 4s/7": lambda s: s * (Fr(4, 7) + Fr(random.randint(1, 300), 100))}
expect = {"t < s/2": set(), "s/2 < t < 4s/7": set(), "t > 4s/7": set(), "act ray": {"act"}, "VoI ray": {"value of information"},
          "axis t=0": {"definedness", "value of information"}, "axis s=0": {"definedness"}, "corner": set(FS.KINDS)}
ok = True; seen = {}
for name, f in regions.items():
    R = {FS.reach((s, f(s))) for s in s0}; seen[name] = R; ok &= R == {frozenset(expect[name])}
for name, L in on.items():
    R = {FS.reach(b) for b in L}; seen[name] = R; ok &= R == {frozenset(expect[name])}
check("FS5 reach is constant on each of the eight strata, with all three kinds meeting at the corner", ok,
      "; ".join(f"{k}: {sorted(next(iter(v)))}" for k, v in seen.items()))
print(f"{sum(results)}/{len(results)} passed")
