"""check_strata.py -- walls and chambers (report Section 19, Theorem "Walls and chambers"; module pce/strata.py).

Rebuilt on 2026-10-29: the original file was overwritten by the covers Stage 4 check (now check_cover_strata.py) and no copy survived.
The eight checks below follow parts (a)-(g) of the report's theorem; the figure script report/fig_strata.py is unchanged.

  ST1  (a) inside a chamber the vertices of the fibre are affine in the tables (same bases; midpoint vertex = mean of end vertices)
  ST2  (a) the tangent complex: dim H0 - dim H1 = chi = |S| - rank A; on the Bell facet the support loses atoms and chi drops by as many
  ST3  (b) a subfamily is filled exactly when its tables lie in its marginal polytope: the disagreement triangle glues iff the four Bell
       / cycle inequalities hold (300 random points)
  ST4  (c) lam* is concave and piecewise affine, and the optimal witness y is a supergradient: lam*(t') <= lam*(t) + <y, t' - t>
  ST5  (d) the robust value R_g is convex in t and affine on each chamber (second differences vanish where the chamber type is constant)
  ST6  (e) the maximum-entropy outcome is continuous but not affine inside a chamber (non-zero second difference)
  ST7  (f) a context path meets finitely many walls; switching the cover (adding the triple window) changes rank A and chi
  ST8  (g) dequantization: if the tempered family t^beta (renormalised) glues, then delta_inf <= (log N + log n) / beta
"""
import os, sys, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import Window
from pce import strata as ST
from pce.integrate import gluable_fraction, maxent_merge, glues
from pce.places import tropical_defect

rng = np.random.default_rng(20261029); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
D = {"X0": 2, "X1": 2, "X2": 2}; PAIRS = [("X0", "X1"), ("X1", "X2"), ("X0", "X2")]
def marg(P, vs): return P.sum(axis=tuple(k for k in range(3) if f"X{k}" not in vs))
def fam(P): return [Window(p, marg(P, p)) for p in PAIRS]
def dis(d): return np.array([[(1 - d) / 2, d / 2], [d / 2, (1 - d) / 2]])
def tri(d1, d2, d3): return [Window(p, dis(d)) for p, d in zip(PAIRS, (d1, d2, d3))]
def mix(W0, W1, s): return [Window(a.vars, (1 - s) * a.table + s * b.table) for a, b in zip(W0, W1)]

# ST1
ok = True; tested = 0
for _ in range(40):
    P0, P1 = rng.dirichlet(np.ones(8)).reshape(2, 2, 2), rng.dirichlet(np.ones(8)).reshape(2, 2, 2)
    W0, W1 = fam(P0), fam(0.9 * P0 + 0.1 * P1); Wm = mix(W0, W1, 0.5)
    c0, c1, cm = ST.chamber_type(D, W0), ST.chamber_type(D, W1), ST.chamber_type(D, Wm)
    if not (c0 == c1 == cm): continue
    V0, V1, Vm = ({B: q for B, q in ST.vertices(D, W)} for W in (W0, W1, Wm))
    common = set(V0) & set(V1) & set(Vm)
    ok &= len(common) > 0 and all(np.allclose(Vm[B], 0.5 * (V0[B] + V1[B]), atol=1e-9) for B in common); tested += 1
check("ST1 inside a chamber the vertices of the fibre are affine in the tables", ok and tested >= 10, f"{tested} segments inside one chamber")

# ST2
ok = True
for _ in range(30):
    W = fam(rng.dirichlet(np.ones(8)).reshape(2, 2, 2)); tc = ST.tangent_complex(D, W)
    ok &= tc["H0"] - tc["H1"] == tc["chi"] == len(tc["S"]) - tc["rankA"]
inside = ST.tangent_complex(D, tri(0.3, 0.4, 0.5)); facet = ST.tangent_complex(D, tri(0.3, 0.4, 0.7))   # d3 = d1 + d2: a Bell facet
drop = len(inside["S"]) - len(facet["S"])
ok &= drop > 0 and inside["chi"] - facet["chi"] == drop and facet["H1"] > inside["H1"]
check("ST2 the tangent complex: dim H0 - dim H1 = chi = |S| - rank A; on a Bell facet the support loses atoms and chi drops by as many", ok,
      f"interior: |S| = {len(inside['S'])}, chi = {inside['chi']}; facet: |S| = {len(facet['S'])}, chi = {facet['chi']}, H1 = {facet['H1']}")

# ST3
ok = True; nin = 0
for _ in range(300):
    d = rng.uniform(0, 1, 3); bell = d[0] <= d[1] + d[2] and d[1] <= d[0] + d[2] and d[2] <= d[0] + d[1] and d.sum() <= 2
    if min(abs(d[0] - d[1] - d[2]), abs(d[1] - d[0] - d[2]), abs(d[2] - d[0] - d[1]), abs(d.sum() - 2)) < 1e-6: continue
    g = gluable_fraction(D, tri(*d)).lam > 1 - 1e-9; ok &= g == bell; nin += bell
check("ST3 a family is filled exactly when its tables lie in the marginal polytope (the Bell / cycle inequalities)", ok, f"{nin} of 300 inside")

# ST4
ok = True; kinks = 0
for _ in range(30):
    d0 = rng.uniform(0.05, 0.95, 3); W = tri(*d0); lam, y = ST.lam_gradient(D, W)
    t = np.concatenate([w.table.ravel() for w in W])
    for _ in range(5):
        d1 = np.clip(d0 + rng.normal(scale=0.2, size=3), 0.01, 0.99); W1 = tri(*d1); t1 = np.concatenate([w.table.ravel() for w in W1])
        lam1, _ = ST.lam_gradient(D, W1); ok &= lam1 <= lam + y @ (t1 - t) + 1e-9
W0, W1 = tri(0.2, 0.25, 0.3), tri(0.2, 0.25, 0.9); ss = np.linspace(0, 1, 201)
la = np.array([ST.lam_gradient(D, mix(W0, W1, s))[0] for s in ss]); d2 = la[2:] - 2 * la[1:-1] + la[:-2]
kinks = int(np.sum(np.abs(d2) > 1e-7)); ok &= np.all(d2 <= 1e-9) and 1 <= kinks <= 4
check("ST4 lam* is concave and piecewise affine in t; the optimal witness is a supergradient, and the kink is where it changes", ok,
      f"{kinks} kink(s) on a path that leaves the polytope")

# ST5, ST6
Ps = [rng.dirichlet(np.ones(8)).reshape(2, 2, 2) for _ in range(3)]
def slice_fam(s, t): return fam((1 - s - t) * Ps[0] + s * Ps[1] + t * Ps[2])
G = np.zeros(8); G[7] = 1.0; h = 0.01; ok5 = True; ok6 = False; n5 = 0
for _ in range(60):
    s0, t0 = rng.uniform(0.1, 0.4, 2); dirn = rng.normal(size=2); dirn /= np.linalg.norm(dirn)
    pts = [slice_fam(s0 + k * h * dirn[0], t0 + k * h * dirn[1]) for k in (-1, 0, 1)]
    types = [ST.chamber_type(D, W) for W in pts]; R = [ST.robust_value(D, W, G) for W in pts]
    ok5 &= R[1] <= 0.5 * (R[0] + R[2]) + 1e-10                                                     # convex
    if types[0] == types[1] == types[2]:
        n5 += 1; ok5 &= abs(R[0] - 2 * R[1] + R[2]) < 1e-9
        ME = [maxent_merge(D, W)[0][7] for W in pts]; ok6 |= abs(ME[0] - 2 * ME[1] + ME[2]) > 1e-7
check("ST5 the robust value is convex in t and affine on every chamber", ok5 and n5 >= 20, f"{n5} triples inside one chamber")
check("ST6 the maximum-entropy outcome is not affine inside a chamber (non-zero second difference)", ok6)

# ST7
P0 = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); W0 = fam(P0); W1 = tri(0.2, 0.25, 0.85)
f = lambda s: mix(W0, W1, s); key = lambda W: (gluable_fraction(D, W).lam > 1 - 1e-7, str(ST.chamber_type(D, W)))
ws = ST.walls_on_path(f, key, n=60)
tp = ST.tangent_complex(D, fam(P0)); tt = ST.tangent_complex(D, fam(P0) + [Window(("X0", "X1", "X2"), P0)])
ok = 0 < len(ws) < 20 and tt["rankA"] > tp["rankA"] and tt["chi"] < tp["chi"]
check("ST7 a context path meets finitely many walls; adding the triple window changes rank A and chi", ok,
      f"{len(ws)} walls; rank {tp['rankA']} -> {tt['rankA']}, chi {tp['chi']} -> {tt['chi']}")

# ST8
ok = True; glued = 0; tight = 0
for _ in range(200):
    W = tri(*rng.uniform(0.001, 0.999, 3))                  # symmetric tables keep uniform single margins under tempering
    dinf, N = tropical_defect(D, W); n = 4
    for beta in (1, 2, 4, 8):
        Wb = [Window(w.vars, w.table ** beta / (w.table ** beta).sum()) for w in W]
        if glues(D, Wb):
            glued += 1; ok &= dinf <= (np.log(N) + np.log(n)) / beta + 1e-9; tight += beta == 8 and dinf > 0
check("ST8 dequantization: a tempered family that glues has delta_inf <= (log N + log n) / beta, so the coherent region collapses onto"
      " the tropical set", ok and glued > 0, f"{glued} glued tempered families, {tight} of them with delta_inf > 0 at beta = 8")

print(f"{sum(results)}/{len(results)} passed")
