"""Checks for the determination theorem and the value-distance identity of section 12.23 (module pce/voigeom.py).
IDENTITY = an exact or numerical instance of a statement proved in the text; CERTIFICATE = an exact computation reported as a result;
NUMERICAL = floating-point (linear programs), with tolerances stated.
  VD1  NUMERICAL    the determination criterion (every full-dimensional policy cell's value functional is orthogonal to the fibre's
                    directions) holds exactly when the exact envelope of the classical VOI over the fibre is a point: random decision
                    problems (cover-additive or not), signals (one variable or a parity of two) and data (interior, fibre-1, fibre-2, rigid)
  VD2  NUMERICAL    the four cases: rigid face; a decision and signal inside one context (fibre dimension 7, VOI determined); the running
                    decision on a never-observed pair (not determined); an identified decision (R = 0) whose VOI is not determined
  VD3  NUMERICAL    when the value functional lives in one context, VOI is computed from that table alone and stays defined for contextual
                    data, where no global law exists (a PR box)
  VD4  NUMERICAL    value-based distance: the largest change of a single action's robust value over utilities in [-1, 1] equals 2 x the
                    Hausdorff distance (total variation) between the fibres, attained by the dual utility; random utilities and
                    multi-action robust values stay below it
"""
import os, sys, random, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import margsec as G
from pce import poisonflip as P
from pce import voigeom as V
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
C = G.CELLS; M4 = np.array(G.M, float); TAU = 0.45; rng = np.random.default_rng(20261107)
T = np.array([1.0 if c[0] == c[1] else 0.0 for c in C])
F = G.facets_D(); faces_ = sorted(G.faces_D(F), key=lambda V_: sorted(V_)); rr = random.Random(20261108)
fam = {nm: rr.sample([V_ for V_ in faces_ if len(V_) == 8 and G.face_dim(V_) == dm and G.fibre_dim_face(V_) == fb], 3) for nm, (dm, fb) in (("F0", (7, 0)), ("F1", (6, 1)), ("F2", (5, 2)))}
rr = random.Random(20261108); laws = {}
for nm in ("F0", "F1", "F2"):
    for j, V_ in enumerate(fam[nm]):
        wv = [rr.randint(5, 25) for _ in V_]; t = sum(wv); wv = [x * 240 // t for x in wv]; wv[0] += 240 - sum(wv); x = np.zeros(16)
        for k, v in zip(sorted(V_), wv): x[k] = v / 240
        laws[(nm, j)] = x

def additive(rg):
    """a cover-additive utility: sum over the four contexts of a random function of that context's two variables"""
    u = np.zeros(16)
    for (i, j) in G.CTX:
        f = rg.normal(size=(2, 2)); u += np.array([f[c[i]][c[j]] for c in C])
    return u

# VD1
agree = tot = det = 0; widths = []; projs = []
for s in range(80):
    kind = s % 4
    d = M4 @ (rng.dirichlet(np.ones(16)) if kind == 0 else laws[(("F1", "F2", "F0")[kind - 1], s % 3)])
    ut = rng.integers(3)
    U = [additive(rng) * 0.3 for _ in range(2)] if ut == 0 else [rng.normal(size=16) * 0.3 for _ in range(2)] if ut == 1 else [TAU - T, T - TAU]
    sv = rng.integers(4); z = [c[sv] for c in C] if rng.random() < 0.6 else [c[sv] ^ c[(sv + 1) % 4] for c in C]
    lo, hi = V.voi_envelope(d, U, z); crit, cells = V.determination_criterion(d, U, z)
    D = V.fibre_directions(d); pr = max([np.linalg.norm(D @ V.value_functional(U, z, p)) for p in cells] + [0.0]) if D.shape[0] else 0.0
    agree += crit == (hi - lo < 1e-8); tot += 1; det += crit; widths.append(hi - lo); projs.append(pr)
check("VD1 NUMERICAL the determination criterion holds exactly when the classical VOI is constant on the fibre (exact envelope by LPs)",
      agree == tot, f"{tot} random problems (20 each: interior, fibre-1 faces, fibre-2 faces, rigid faces); criterion and envelope agree in {agree}; "
      f"{det} determined; smallest envelope width among the undetermined {min([w for w, p in zip(widths, projs) if p > 1e-9] + [np.inf]):.2e}")

# VD2
d_int = M4 @ np.random.default_rng(1).dirichlet(np.ones(16)); L, U_ = P.lower_upper(d_int, T)
loc = np.array([1.0 if c[0] == c[2] else 0.0 for c in C]); zB0 = [c[2] for c in C]
cases = [("rigid face, running decision, signal B0", M4 @ laws[("F0", 0)], [TAU - T, T - TAU]),
         ("interior, decision and signal inside (A0, B0)", d_int, [TAU - loc, loc - TAU]),
         ("interior, running decision on (A0, A1), signal B0", d_int, [TAU - T, T - TAU]),
         ("interior, identified decision (tau = L - 0.02), signal B0", d_int, [(L - 0.02) - T, T - (L - 0.02)])]
rows = []; ok = True
for nm, d, U in cases:
    lo, hi = V.voi_envelope(d, U, zB0); crit, _ = V.determination_criterion(d, U, zB0); R = V.regret_value(d, U); fd = G.fibre_dim(d)
    rows.append((nm, fd, round(R, 4), round(lo, 4), round(hi, 4), crit))
ok = rows[0][5] and rows[0][1] == 0 and rows[1][5] and rows[1][1] == 7 and rows[1][3] > 0 and not rows[2][5] and rows[3][2] == 0 and not rows[3][5]
check("VD2 NUMERICAL four cases: the three determinations separate", ok,
      "; ".join(f"{nm}: fibre dim {fd}, R {R}, VOI envelope [{lo}, {hi}], determined {cr}" for nm, fd, R, lo, hi, cr in rows))

# VD3
a0 = np.array([1.0 if c[0] == 0 else 0.0 for c in C]); Ul = [TAU - a0, a0 - TAU]
def local_voi(tab):                    # tab = 2 x 2 table of (A0, B0): VOI from that table alone
    az = [tab[0][z] - TAU * (tab[0][z] + tab[1][z]) for z in (0, 1)]; return abs(az[0]) + abs(az[1]) - abs(sum(az))
ok = True; dev = 0.0
for s in range(20):
    x = rng.dirichlet(np.ones(16)); d = M4 @ x; lo, hi = V.voi_envelope(d, Ul, zB0)
    tab = [[sum(x[i] for i, c in enumerate(C) if c[0] == a and c[2] == b) for b in (0, 1)] for a in (0, 1)]
    dev = max(dev, abs(local_voi(tab) - lo), hi - lo); ok &= hi - lo < 1e-8
pr = P.pr_box((-1, -1, 1, -1)); E4, _ = P.entries_chsh(); cf = P.contextual_fraction(pr, E4); empty = P.lower_upper(pr, T) is None
tab = [[pr[0], 0.5 - pr[0]], [0.5 - pr[0], pr[0]]]
check("VD3 NUMERICAL a value functional inside one context gives a VOI computed from that table alone, defined even for contextual data",
      ok and dev < 1e-8 and empty and cf > 0.4, f"decision on A0 = 0 with signal B0: 20 random points, local formula = exact constant VOI (largest deviation {dev:.1e}); "
      f"PR box: contextual fraction {cf:.3f}, no global law (fibre empty), yet the local VOI is {local_voi(tab):.3f}")

# VD4
ok = True; rows4 = []
pairs = [(M4 @ rng.dirichlet(np.ones(16)), None) for _ in range(6)]
for i, (d1, _) in enumerate(pairs):
    d2 = d1 + rng.uniform(0.1, 0.5) * (M4 @ rng.dirichlet(np.ones(16)) - d1)
    h = V.hausdorff_tv(d1, d2); gap, u = V.worst_direction(d1, d2)
    rnd = max(abs(V.support_value(d1, uu) - V.support_value(d2, uu)) for uu in rng.uniform(-1, 1, (300, 16)))
    multi = max(abs(max(V.support_value(d1, ua) for ua in uu) - max(V.support_value(d2, ua) for ua in uu)) for uu in rng.uniform(-1, 1, (60, 3, 16)))
    ok &= abs(gap - 2 * h) < 1e-7 and np.abs(u).max() <= 1 + 1e-9 and rnd <= 2 * h + 1e-9 and multi <= 2 * h + 1e-9
    rows4.append((round(2 * h, 5), round(gap, 5), round(rnd, 4), round(multi, 4)))
check("VD4 NUMERICAL value-based distance: sup over utilities in [-1,1] of the change in robust value = 2 x Hausdorff distance (TV) between fibres",
      ok, "(2 dH, gap of the dual utility, largest of 300 random single-action gaps, largest of 60 random 3-action robust-value gaps): " + "; ".join(str(r_) for r_ in rows4))
print(f"{sum(results)}/{len(results)} passed")
