"""Checks for 'Transport along revisions, holonomy and laundering' (module pce/fieldholonomy.py). Variables R, S (pathways) and Y (outcome);
data are the tables of {R, Y} and {S, Y}. A recomputer's credal set is the fibre of the current data; an updater carries a law by Jeffrey
updates and refits the current data at the end.
  HL1 IDENTITY  likelihood edges commute and a likelihood followed by its reciprocal closes the loop exactly (100 random cases)
  HL2 IDENTITY  Jeffrey updates on {R,Y} and {S,Y} leave the (R,S) log odds ratios given Y unchanged (100 random sequences): edits on the
                data's own contexts cannot touch the interactions the data do not determine
  HL3 IDENTITY  an excursion of poisoned tables on the data's own contexts, followed by a refit, returns the honest updater's law exactly
                (zero holonomy), for 100 random excursions
  HL4 IDENTITY  an excursion on a context outside the data, published then retracted, leaves the updater's law consistent with every
                current table but moved (holonomy > 0); the pairwise field (determined by the data) does not move, the 'pathway alone'
                field does; a retracted full table makes every law of the fibre reachable, a retracted {R,S} table only a curve
  HL5 IDENTITY  laundering: in a world where the recomputer leaves the order of the 'pathway alone' field open (a pre-crossing), the
                honest updater reports one order and the laundered updater the other, both as points and both consistent with the data;
                in a world where the fibre decides the order, no retracted edit reverses it
  HL6 IDENTITY  field holonomy is bounded by law holonomy: |E_P[f|E] - E_Q[f|E]| <= 2 TV(P,Q) / max(P(E), Q(E)) for f in [0,1] (2000 pairs)
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from pce import fieldholonomy as FH
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(17); U = np.ones(8) / 8
def rtab(ctx): v = rng.dirichlet(np.ones(2 ** len(ctx))); keys = sorted({tuple(w[c] for c in ctx) for w in FH.OUT}); return dict(zip(keys, v))
diff = lambda Q: FH.joint_field(Q)[0] - FH.joint_field(Q)[1]

# HL1
ok = True
for _ in range(100):
    P = rng.dirichlet(np.ones(8)); L1, L2, L3 = (rng.uniform(0.2, 3, 8) for _ in range(3))
    a = FH.bayes(FH.bayes(FH.bayes(P, L1), L2), L3); b = FH.bayes(FH.bayes(FH.bayes(P, L3), L1), L2)
    ok &= FH.tv(a, b) < 1e-12 and FH.tv(FH.bayes(FH.bayes(P, L1), 1 / L1), P) < 1e-12
check("HL1 IDENTITY likelihood edges commute; L then 1/L closes", ok)

# HL2
ok = True
for _ in range(100):
    P = rng.dirichlet(np.ones(8)); i0 = FH.interaction(P); Q = P
    for _ in range(6): ctx = [(0, 2), (1, 2)][int(rng.integers(2))]; Q = FH.jeffrey(Q, ctx, rtab(ctx))
    ok &= np.allclose(FH.interaction(Q), i0, atol=1e-9)
check("HL2 IDENTITY Jeffrey updates on the data's contexts leave the (R,S) interactions unchanged", ok)

# world A (the order undecided by the fibre)
PA = np.array([0.08, 0.042, 0.087, 0.06, 0.225, 0.253, 0.124, 0.128]); PA = PA / PA.sum()
dataA = [((0, 2), FH.marg(PA, (0, 2))), ((1, 2), FH.marg(PA, (1, 2)))]
HA = FH.refit(U, dataA)

# HL3
ok = True; worst = 0.0
for _ in range(100):
    Q = U
    for _ in range(4): ctx = [(0, 2), (1, 2)][int(rng.integers(2))]; Q = FH.jeffrey(Q, ctx, rtab(ctx))
    Q = FH.refit(Q, dataA); worst = max(worst, FH.tv(Q, HA))
ok = worst < 1e-9
check("HL3 IDENTITY excursions on the data's own contexts are erased by the refit (zero holonomy)", ok, f"largest TV {worst:.1e}")

# HL4
cons = lambda Q, data: max(abs(FH.marg(Q, c)[k] - t[k]) for c, t in data for k in t)
ok = True; hol = []
for _ in range(50):
    Q = FH.refit(FH.jeffrey(U, (0, 1), rtab((0, 1))), dataA); hol.append(FH.tv(Q, HA))
    ok &= cons(Q, dataA) < 1e-9 and np.allclose(FH.pair_field(Q), FH.pair_field(HA), atol=1e-9)
V = np.array(FH.fibre_vertices(dataA)); reach_full = True
for _ in range(50):
    T = rng.dirichlet(np.ones(len(V))) @ V; Q = FH.refit(FH.jeffrey(U, (0, 1, 2), dict(zip(FH.OUT, T))), dataA); reach_full &= FH.tv(Q, T) < 1e-9
curve = []
for g in np.linspace(-6, 6, 61):
    tab = {(0, 0): np.exp(g), (0, 1): 1.0, (1, 0): 1.0, (1, 1): np.exp(g)}; z = sum(tab.values()); tab = {k: v / z for k, v in tab.items()}
    curve.append(diff(FH.refit(FH.jeffrey(U, (0, 1), tab), dataA)))
lo, hi = FH.fibre_range(dataA, diff, 20000)
ok &= min(hol) > 1e-3 and reach_full and (max(curve) - min(curve)) < 0.5 * (hi - lo)
check("HL4 IDENTITY a retracted edit on a new context leaves a consistent but moved law; full tables reach the whole fibre, {R,S} tables a curve", ok,
      f"holonomy (TV) from retracted {{R,S}} tables: {min(hol):.3f} to {max(hol):.3f}; 'alone' difference reachable by {{R,S}} tables [{min(curve):.3f}, {max(curve):.3f}] vs fibre [{lo:.3f}, {hi:.3f}]")

# HL5
VA = FH.fibre_vertices(dataA); vmin = min(VA, key=lambda Q: diff(Q) if np.isfinite(diff(Q)) else 9)
T = 0.85 * vmin + 0.15 * HA; LA = FH.refit(FH.jeffrey(U, (0, 1, 2), dict(zip(FH.OUT, T))), dataA)
okA = lo < 0 < hi and diff(HA) > 0 and diff(LA) < 0 and cons(LA, dataA) < 1e-9
PB = np.array([0.20, 0.06, 0.07, 0.07, 0.06, 0.10, 0.06, 0.38]); PB = PB / PB.sum(); dataB = [((0, 2), FH.marg(PB, (0, 2))), ((1, 2), FH.marg(PB, (1, 2)))]
loB, hiB = FH.fibre_range(dataB, diff, 20000); okB = loB > 0
for _ in range(200):
    ctx = [(0, 1), (0, 1, 2)][int(rng.integers(2))]; Q = FH.refit(FH.jeffrey(U, ctx, rtab(ctx)), dataB); okB &= diff(Q) > 0
check("HL5 IDENTITY laundering reverses an order the fibre leaves open, never one the fibre decides", okA and okB,
      f"world A: fibre [{lo:.3f}, {hi:.3f}], honest updater {diff(HA):+.3f}, laundered {diff(LA):+.3f} (TV {FH.tv(LA, HA):.3f}), true {diff(PA):+.3f}; "
      f"world B: fibre [{loB:.3f}, {hiB:.3f}], 200 retracted edits all keep the order")

# HL6
ok = True; worst = 0.0
for _ in range(2000):
    P, Q = rng.dirichlet(np.ones(8), 2); f = rng.uniform(0, 1, 8); E = rng.random(8) < 0.5
    if not E.any(): continue
    a = (P * f * E).sum() / (P * E).sum(); b = (Q * f * E).sum() / (Q * E).sum(); bound = 2 * FH.tv(P, Q) / max((P * E).sum(), (Q * E).sum())
    ok &= abs(a - b) <= bound + 1e-12; worst = max(worst, abs(a - b) / bound if bound > 0 else 0)
check("HL6 IDENTITY field holonomy <= 2 TV / max(P(E), Q(E))", ok, f"largest ratio {worst:.3f}")
json.dump(dict(worldA=dict(fibre=[lo, hi], honest=float(diff(HA)), laundered=float(diff(LA)), true=float(diff(PA)), tv=FH.tv(LA, HA)),
               worldB=dict(fibre=[loB, hiB]), curve=[min(curve), max(curve)]), open(os.path.join(HERE, "fieldholonomy_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
