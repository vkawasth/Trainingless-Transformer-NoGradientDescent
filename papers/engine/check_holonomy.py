"""Checks for history-indexed selection and holonomy (section 12.26, module pce/holonomy.py).
IDENTITY = an exact or numerical instance of a statement proved in the text; NUMERICAL = floating-point (linear and quadratic programs,
iterative proportional fitting), with tolerances stated.
  HY1  IDENTITY     KL transport is flat: along random histories whose context sets only grow, with data revised on the contexts already
                    seen, I-projection updating from the uniform law ends at the maximum-entropy law of the final snapshot
  HY2  IDENTITY     Euclidean nearest-point updating is flat while no positivity constraint binds (strictly positive laws, small steps), and
                    curved once one does; total-variation nearest-point updating is curved even in the interior
  HY3  IDENTITY     the bounds: holonomy <= TV diameter of the final fibre; the gap in w.x <= U - L; selected VOIs inside the exact envelope;
                    the Euclidean and TV rules agree on a rigid face (fibre a point)
  HY4  NUMERICAL    forgetting curves KL: when old contexts leave the window, I-projection (Jeffrey) updating depends on the path, as in
                    Diaconis and Zabell; the order gap agrees with pce/updates.py
  HY5  NUMERICAL    the three-snapshot example: fibre dimensions, intervals, selections and actions at tau = 0.49 for two histories; the
                    contextual variant at t3, where every three contexts glue and the four do not
  HY6  NUMERICAL    poison-then-remove loops on a fixed context set: KL and interior Euclidean updating return exactly; TV does not
"""
import os, sys, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import margsec as G
from pce import holonomy as H
from pce import poisonflip as P
from pce import voigeom as V
from pce import updates as UPD
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(20261107); M4 = np.array(G.M, float); C = G.CELLS
w = np.array([1.0 if c[0] == c[1] else 0.0 for c in C]); TAU = 0.49

def random_history(x_final, rng, revise=True):
    """a random growing history ending at the full snapshot of x_final: contexts arrive in a random order, in 2-4 steps; before the end the
    tables seen so far may be those of a different (revised) law"""
    order = list(rng.permutation(4)); cuts = sorted(rng.choice([1, 2, 3], size=rng.integers(1, 4), replace=False)); out = []
    Tf = H.tables(x_final)
    for c in cuts:
        T = H.tables(rng.dirichlet(np.ones(16))) if (revise and rng.random() < 0.5) else Tf
        out.append((order[:c], T))
    out.append((order, Tf)); return out

# HY1
dev = 0.0; n = 0
for s in range(30):
    x = rng.dirichlet(np.ones(16) * rng.uniform(0.5, 3)); me = G.maxent(M4 @ x)
    for _ in range(3):
        dev = max(dev, H.tv(H.run(H.kl_update, random_history(x, rng))[-1], me)); n += 1
check("HY1 IDENTITY I-projection updating along growing histories (with revisions) ends at the maximum-entropy law of the final snapshot",
      dev < 1e-9, f"{n} random histories over 30 final laws; largest TV distance to the maximum-entropy law {dev:.1e}")

# HY2
inner = 0.0; nin = 0; outer = []; tvin = []
for s in range(40):
    x = rng.dirichlet(np.ones(16) * 30)                                         # strictly positive, near uniform
    T = H.tables(x); Ha = [([0, 1], T), ([0, 1, 2], T), ([0, 1, 2, 3], T)]; Hb = [([3, 2], T), ([3, 2, 1], T), ([3, 2, 1, 0], T)]
    ra, rb = H.run(H.euclid_update, Ha), H.run(H.euclid_update, Hb)
    if min(min(v.min() for v in ra), min(v.min() for v in rb)) > 1e-9:
        inner = max(inner, H.tv(ra[-1], rb[-1])); nin += 1; tvin.append(H.holonomy(H.tv_update, Ha, Hb))
for s in range(40):
    x = rng.dirichlet(np.ones(16) * 0.5); T = H.tables(x)
    Ha = [([0, 1], T), ([0, 1, 2], T), ([0, 1, 2, 3], T)]; Hb = [([3, 2], T), ([3, 2, 1], T), ([3, 2, 1, 0], T)]
    outer.append(H.holonomy(H.euclid_update, Ha, Hb))
check("HY2 IDENTITY Euclidean updating is flat in the interior and curved at the boundary; TV updating is curved in the interior",
      nin >= 20 and inner < 1e-9 and max(outer) > 1e-3 and max(tvin) > 1e-3,
      f"interior: {nin} positive laws, every projection strictly positive, largest Euclidean holonomy {inner:.1e}, largest TV holonomy {max(tvin):.3f}; "
      f"boundary (Dirichlet 0.5): Euclidean holonomy nonzero in {sum(o > 1e-6 for o in outer)}/40, largest {max(outer):.3f}")

# HY3
ok = True; worst = [0.0, 0.0]; nn = 0; Uaa = V.act_abstain_utilities(TAU); zB0 = [c[2] for c in C]
for s in range(12):
    x = rng.dirichlet(np.ones(16) * 0.7); T = H.tables(x); d = M4 @ x
    diam, _ = H.tv_diameter([0, 1, 2, 3], T); L, U = H.interval([0, 1, 2, 3], T, w); lo, hi = V.voi_envelope(d, Uaa, zB0)
    hs = [random_history(x, rng) for _ in range(4)]
    for f in H.RULES.values():
        ends = [H.run(f, h)[-1] for h in hs]
        for a, b in itertools.combinations(ends, 2):
            ok &= H.tv(a, b) <= diam + 1e-9 and abs(w @ a - w @ b) <= U - L + 1e-9; nn += 1
            worst[0] = max(worst[0], H.tv(a, b) / diam); worst[1] = max(worst[1], abs(w @ a - w @ b) / (U - L))
        ok &= all(lo - 1e-9 <= V.voi_at(e, Uaa, zB0) <= hi + 1e-9 for e in ends)
F = G.facets_D(); rigid = [V_ for V_ in G.faces_D(F) if len(V_) == 8 and G.fibre_dim_face(V_) == 0][0]
xr = np.zeros(16); xr[sorted(rigid)] = 1 / 8; Tr = H.tables(xr); hr = [random_history(xr, rng, revise=True) for _ in range(4)]
RIG = (H.euclid_update, H.tv_update)                                       # KL needs a strictly positive law in the fibre; a rigid face has none
rig = max(H.tv(H.run(f, a)[-1], H.run(g, b)[-1]) for f in RIG for g in RIG for a in hr for b in hr)
check("HY3 IDENTITY holonomy <= TV diameter, decision gap <= U - L, VOI at every selection inside the envelope; Euclid and TV agree on a rigid face",
      ok and rig < 1e-7, f"{nn} pairs of histories (3 rules, 12 final laws); largest holonomy / diameter {worst[0]:.2f}, largest gap / (U - L) {worst[1]:.2f}; "
      f"rigid face: largest distance between the Euclidean and TV rules over four histories {rig:.1e}")

# HY4
x = rng.dirichlet(np.ones(16) * 2); T = H.tables(x); u = np.full(16, 1 / 16)
via = H.kl_update(H.kl_update(u, [0, 1], T), [2, 3], T); direct = H.kl_update(u, [2, 3], T)       # window forgets (A0,B0),(A0,B1)
# the same with pce/updates.py (Jeffrey on one context at a time): order gap of two single-context updates
xs = rng.dirichlet(np.ones(16)).reshape(2, 2, 2, 2); qa = rng.dirichlet(np.ones(4)).reshape(2, 2); qc = rng.dirichlet(np.ones(4)).reshape(2, 2)
r2 = UPD.two_orders(xs, (0, 2), qa, (0, 3), qc)                            # contexts (A0,B0) and (A0,B1): Jeffrey in both orders
x1 = H.kl_update(xs.ravel(), [0], [qa.ravel()] * 4); x12 = H.kl_update(x1, [1], [qc.ravel()] * 4)
hdev = float(np.nanmax(np.abs((r2["h"] - r2["h_pred"]) - np.nanmean(r2["h"] - r2["h_pred"]))))
check("HY4 NUMERICAL forgetting curves KL: a window that drops old contexts makes I-projection updating path dependent",
      H.tv(via, direct) > 1e-3 and r2["order_gap"] > 1e-3 and H.tv(x12, r2["C_after_A"].ravel()) < 1e-9 and hdev < 1e-9,
      f"window history (A0B0,A0B1) then (A1B0,A1B1) vs (A1B0,A1B1) alone: TV {H.tv(via, direct):.3f}, w-gap {abs(w @ via - w @ direct):.3f}; "
      f"single-context I-projections (Jeffrey) on (A0,B0) and (A0,B1) in the two orders differ by TV {r2['order_gap']:.3f}, and the log-ratio is "
      f"the cell function of pce/updates.py (deviation {hdev:.1e})")

# HY5
k = np.array([3, 0, 3, 0, 1, 0, 2, 1, 3, 4, 1, 0, 0, 2, 0, 0]); x5 = k / 20; T5 = H.tables(x5); d5 = M4 @ x5
HA = [([0, 1], T5), ([0, 1, 2], T5), ([0, 1, 2, 3], T5)]; HB = [([3, 2], T5), ([3, 2, 1], T5), ([3, 2, 1, 0], T5)]
dims = [[int(H.fibre_dim(ks, T5)) for ks, _ in h] for h in (HA, HB)]; ints = [[tuple(float(v) for v in np.round(H.interval(ks, T5, w), 6)) for ks, _ in h] for h in (HA, HB)]
sel = {nm: [round(float(w @ H.run(f, h)[-1]), 4) for h in (HA, HB)] for nm, f in H.RULES.items()}
act = {nm: [int(v >= TAU) for v in vals] for nm, vals in sel.items()}
hol = {nm: round(float(H.holonomy(f, HA, HB)), 4) for nm, f in H.RULES.items()}; diam5, _ = H.tv_diameter([0, 1, 2, 3], T5)
lo5, hi5 = V.voi_envelope(d5, Uaa, zB0); vois = {nm: [round(float(V.voi_at(H.run(f, h)[-1], Uaa, zB0)), 4) for h in (HA, HB)] for nm, f in H.RULES.items()}
T6 = list(T5); T6[3] = np.array([0.65, 0.05, 0.0, 0.30]); d6 = d5.copy(); d6[3] = 0.65
tri = all(H.glues(list(c), T6) for c in itertools.combinations(range(4), 3)); four = H.glues([0, 1, 2, 3], T6)
E4, _ = P.entries_chsh(); cf = P.contextual_fraction(d6, E4); chain = all(H.glues(ks, T6) for ks, _ in HA[:2]) and all(H.glues(ks, T6) for ks, _ in HB[:2])
ok = (dims == [[10, 8, 7], [10, 8, 7]] and ints[0][2] == ints[1][2] == (0.4, 0.6) and ints[0][1] == (0.2, 0.6) and ints[1][1] == (0.4, 0.8)
      and act["KL"] == [1, 1] and act["Euclid"] == [0, 1] and act["TV"] == [0, 1] and hol["KL"] < 1e-9 and hol["TV"] <= diam5
      and abs(sel["TV"][1] - sel["TV"][0]) <= 0.2 + 1e-9 and tri and not four and chain and cf > 0.05)
check("HY5 NUMERICAL three snapshots: one final snapshot, two histories; KL acts on both, Euclid and TV split; the contextual t3 fails to glue",
      ok, f"fibre dimensions {dims}; intervals of P(A0=A1) {ints}; w.x at t3 {sel}; actions at tau {TAU} {act}; holonomy {hol} (diameter {diam5:.2f}); "
      f"VOI of B0 at the selections {vois}, envelope [{lo5:.4f}, {hi5:.4f}]; contextual t3 (A1B1 table 0.65, 0.05, 0, 0.30): every triple glues {tri}, "
      f"all four {four}, contextual fraction {cf:.3f}")

# HY6
rows = []
for s in range(10):
    x = rng.dirichlet(np.ones(16) * 20); T0 = H.tables(x); xp = rng.dirichlet(np.ones(16)); T1 = H.tables(0.85 * x + 0.15 * xp)
    loop = [([0, 1, 2, 3], T0), ([0, 1, 2, 3], T1), ([0, 1, 2, 3], T0)]; stay = [([0, 1, 2, 3], T0)] * 3
    rows.append(tuple(H.holonomy(f, loop, stay) for f in (H.kl_update, H.euclid_update, H.tv_update)))
rows = np.array(rows)
check("HY6 NUMERICAL poison-then-remove loops: KL and interior Euclidean updating return to the start; TV nearest-point updating need not",
      rows[:, 0].max() < 1e-9 and rows[:, 1].max() < 1e-9 and rows[:, 2].max() > 1e-3,
      f"10 loops (15% poisoning, then removal), fixed contexts: largest holonomy KL {rows[:, 0].max():.1e}, Euclid {rows[:, 1].max():.1e}, "
      f"TV {rows[:, 2].max():.3f} (nonzero in {(rows[:, 2] > 1e-6).sum()}/10)")
print(f"{sum(results)}/{len(results)} passed")
