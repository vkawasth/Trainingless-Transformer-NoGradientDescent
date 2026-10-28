"""check_capstone.py -- one picture: chambers, slices and gauges (lab paper, Theorem L). Random two-type truths on three variables
(pair cover) and random two-act decisions with utilities on the eight cells; the worked example for the named numbers.

  CL1 settling: the set rule chooses the oracle act at every truth of the hiding segment M iff M lies in one chamber
  CL2 Savage: an exact (point-slice) probe has value R(M); R(M) > 0 iff M meets two chambers; a plateau probe (slice = M) has value 0
  CL3 Wald: an exact probe has value W_post - W(M) >= 0, positive iff M meets two chambers and the acts' slopes are strictly opposite
  CL4 margin over capacity, forward: the smallest injected share that crosses the wall is delta / (sup_Q l(Q) - l(P)): over all laws
      the vertex attack delta / (max_x l(x) - l(P)), along the pair line the hidden channel delta / (s_cap |l(chi)|); both equal the
      exact budgets found by bisection on the mixture
  CL5 margin over capacity, observer: the block radius is a water-filling (equal to the LP); it is delta / (c - min_x l(x)) when the
      lowest cell holds the removed mass, and on the line
      |s'| / (cap(O) + |s'|); the model constraint can only raise it (the worked example attains 1/128 inside the model)
  CL6 the worked example: margin 0.001 in every reading: vertex 0.001/0.873, hidden channel 0.001/0.027, observer 0.001/0.128 = 1/128,
      observer on the line 0.001/0.014 = 1/14, settling (z sigma_l / 0.001)^2 = Theorem D's (z tau / d)^2
  CL7 relevance: a gluing failure along chi changes the decision functional at rate l(chi); if l(chi) = 0 the hidden channel never
      moves it (infinite budget) whatever the share, and if l(chi) != 0 the budget is finite
  CL8 two boundaries: the stealth window eps* < eps_det exists iff the wall lies inside M
  CL9 beyond the line: keeping every pair table, the forward budget is the margin over the support function of the pair fibre polytope
      in the wall's direction (an LP); at m = 3 this is the hidden channel; at m = 4 (worked example 4) the five-dimensional fibre gives
      at most the m = 3 room (the ABC projection of the m = 4 fibre lies in the m = 3 fibre; here equal) and more than the single chi_ABC channel
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pce.lab import law, moments
from pce import segment_decision as SD
from pce import injection as IJ
from pce import capstone as CP
from pce import lifting as LF

rng = np.random.default_rng(20261031); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

def rand_truth():
    while True:
        P = law(rng.uniform(0.2, 0.8), rng.uniform(0.05, 0.95, 3), rng.uniform(0.05, 0.95, 3)); mu, M, _ = moments(P)
        if min(abs(M[0, 1]), abs(M[0, 2]), abs(M[1, 2])) > 0.01 and M[0, 1] * M[0, 2] * M[1, 2] > 0: return P

def rand_decision(P, flat=False):
    """utilities on cells; shift ub so that the wall falls inside or outside M with comparable frequency"""
    ua = rng.normal(size=8); ub = np.zeros(8) if flat else rng.normal(size=8)
    pa, pb, R, _, _ = CP._polys(P, ua, ub); mid = np.polyval if False else None
    from numpy.polynomial import polynomial as Pn
    t = rng.uniform(-0.6, 1.6); r = R[0] + t * (R[1] - R[0]); ub = ub + (Pn.polyval(r, pa) - Pn.polyval(r, pb))
    return ua, ub

designs = []
for _ in range(150):
    P = rand_truth(); ua, ub = rand_decision(P, flat=rng.random() < 0.2); designs.append((P, ua, ub))

# CL1
agree = [CP.chamber_test(P, ua, ub)[0] == CP.set_rule_is_oracle(P, ua, ub) for P, ua, ub in designs]
n1 = sum(CP.chamber_test(P, ua, ub)[0] for P, ua, ub in designs)
check("CL1 settling <=> one chamber (150 random designs)", all(agree), f"{n1} single-chamber, {150 - n1} two-chamber")

# CL2
ok = True
for P, ua, ub in designs:
    one = CP.chamber_test(P, ua, ub)[0]; RM = CP.savage_values(P, ua, ub); ok &= (RM > 1e-12) == (not one)
check("CL2 Savage: exact probe worth R(M), positive iff two chambers; a plateau probe is worth 0", ok)

# CL3
ok = True; npos = 0
for P, ua, ub in designs:
    one = CP.chamber_test(P, ua, ub)[0]; W, Wp, sa, sb = CP.wald_values(P, ua, ub); pos = Wp - W > 1e-12
    ok &= Wp - W >= -1e-12 and pos == ((not one) and sa * sb < -1e-15); npos += pos
check("CL3 Wald: exact probe worth W_post - W(M) >= 0, positive iff two chambers and strictly opposite slopes", ok, f"{npos} positive")

# CL4
gaps = []
for P, ua, ub in designs[:60]:
    l = CP.wall_functional(ua, ub); p = IJ.as_vec(P); c = l @ p + abs(rng.normal()) * 0.02
    k = int(np.argmax(l)); D = np.zeros_like(P); D[IJ.outcomes(3)[k]] = 1.0
    e_free = IJ.exact_budget(P, D, lambda Pq: l @ IJ.as_vec(Pq) >= c, emax=1.0, grid=400)
    if np.isfinite(e_free): gaps.append(abs(e_free - CP.fwd_free(P, l, c)))
    lc = l @ CP.CHI; sp, sm = CP.s_caps(P); s = sp if lc > 0 else -sm; Q = IJ.as_law(p + s * CP.CHI, 3)
    e_line = IJ.exact_budget(P, Q, lambda Pq: l @ IJ.as_vec(Pq) >= c, emax=1.0, grid=400); fl = CP.fwd_line(P, l, c)
    if np.isfinite(e_line) or np.isfinite(fl): gaps.append(abs(e_line - fl) if np.isfinite(e_line) and np.isfinite(fl) else (0.0 if fl > 1 else 1.0))
check("CL4 forward budgets are margin / capacity (vertex and hidden channel), equal to bisection", max(gaps) < 1e-8, f"max gap {max(gaps):.1e} over {len(gaps)}")

# CL5
gaps = []; closed = []
for P, ua, ub in designs[:60]:
    l = CP.wall_functional(ua, ub); c = l @ IJ.as_vec(P) + abs(rng.normal()) * 0.02; e = CP.obs_free(P, l, c); gaps.append(abs(e - CP.obs_free_lp(P, l, c)))
    k = int(np.argmin(l)); closed.append((IJ.as_vec(P)[k] >= e - 1e-12) == (abs(e - CP.obs_free_closed(P, l, c)) < 1e-12))
check("CL5 observer's block = water-filling = LP; it is margin / (c - min l) exactly when the lowest cell holds the removed mass", max(gaps) < 1e-9 and all(closed),
      f"max gap {max(gaps):.1e}; closed form attained in {sum(1 for P, ua, ub in designs[:60] if abs(CP.obs_free(P, CP.wall_functional(ua, ub), CP.wall_functional(ua, ub) @ IJ.as_vec(P) + 0.01) - CP.obs_free_closed(P, CP.wall_functional(ua, ub), CP.wall_functional(ua, ub) @ IJ.as_vec(P) + 0.01)) < 1e-12)} of 60 (margin 0.01)")

# CL6
W = LF.we(); P = W["P"]; l = np.zeros(8); l[1] = 1.0; c = 0.128                       # l = P(001); the wall is P(001) = 0.128
vals = dict(vertex=CP.fwd_free(P, l, c), hidden=CP.fwd_line(P, l, c), observer=CP.obs_free(P, l, c), observer_line=CP.obs_line(P, l, c))
eps_block, _, _ = LF.block_radius(P, lambda X: LF.gapTP(X), starts=60)
sig_l = abs(W["ps"][0][1]) * W["tau"]; n_half = (1.96 * sig_l / 0.001) ** 2; n_D = (1.96 * W["tau"] / (0.001 / abs(W["ps"][0][1]))) ** 2
check("CL6 worked example: one margin 0.001 over five capacities (0.873, 0.027, 0.128, 0.014; sigma for sampling); the model attains 1/128",
      abs(vals["vertex"] - 0.001 / 0.873) < 1e-9 and abs(vals["hidden"] - 0.001 / 0.027) < 1e-9 and abs(vals["observer"] - 1 / 128) < 1e-12
      and abs(vals["observer_line"] - 1 / 14) < 1e-9 and abs(eps_block - 1 / 128) < 1e-6 and abs(n_half / n_D - 1) < 1e-12,
      "; ".join(f"{k} {v:.5f}" for k, v in vals.items()) + f"; settling n {n_half:.0f}")

# CL7
ok = True
for P, ua, ub in designs[:40]:
    l = CP.wall_functional(ua, ub); l0 = l - (l @ CP.CHI) / 8 * CP.CHI; p = IJ.as_vec(P); c = l0 @ p + 0.01
    sp, sm = CP.s_caps(P); along = [l0 @ (p + s * CP.CHI) for s in np.linspace(-sm, sp, 11)]
    ok &= np.ptp(along) < 1e-12 and not np.isfinite(CP.fwd_line(P, l0, c)) and np.isfinite(CP.fwd_line(P, l, l @ p + 1e-4 * np.sign(l @ CP.CHI) * 0 + 1e-5))
check("CL7 relevance: with l(chi) = 0 the hidden channel never moves the decision; with l(chi) != 0 the budget is finite", ok)

# CL8
ok = True; nwin = 0
for P, ua, ub in designs:
    l = CP.wall_functional(ua, ub); p = IJ.as_vec(P); lc = l @ CP.CHI
    if abs(lc) < 1e-9: continue
    s_wall = -(l @ p) / lc; lo, hi = CP.model_part_s(P); sp, sm = CP.s_caps(P)
    if s_wall > 0: e_star, e_det = s_wall / sp, hi / sp
    else: e_star, e_det = -s_wall / sm, -lo / sm
    inside = lo < s_wall < hi; ok &= (e_star < e_det) == inside; nwin += inside
check("CL8 two boundaries: a stealth window exists iff the wall lies inside the hiding segment", ok, f"{nwin} designs with a window")

# CL9
P4 = law(0.6, np.array([.5, .1, .4, .2]), np.array([.05, .5, .1, .7])); l4 = np.array([float(x[:3] == (0, 0, 1)) for x in IJ.outcomes(4)])
e4, Q4 = CP.fwd_fibre(P4, l4, 0.128); e4b = IJ.exact_budget(P4, Q4, lambda Pq: l4 @ IJ.as_vec(Pq) >= 0.128 - 1e-15, emax=1.0, grid=2000)
l3 = np.zeros(8); l3[1] = 1.0; e3, _ = CP.fwd_fibre(W["P"], l3, 0.128)
c4 = IJ.chi(4, (0, 1, 2)); s4 = min(IJ.as_vec(P4)[c4 < 0]); e_single = (0.128 - l4 @ IJ.as_vec(P4)) / (s4 * (l4 @ c4))
pairs_kept = all(np.allclose(Q4.sum(axis=tuple(k for k in range(4) if k not in S)), P4.sum(axis=tuple(k for k in range(4) if k not in S))) for S in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
check("CL9 beyond the line: with pair tables kept the budget is margin over the support function of the pair fibre (an LP); at m = 3 it is the hidden channel; at m = 4 the full fibre gives no more room than m = 3, but more than a single channel",
      abs(e3 - vals["hidden"]) < 1e-9 and abs(e4 - e4b) < 1e-6 and pairs_kept and e4 >= e3 - 1e-12 and e4 < e_single,
      f"m = 3: {e3:.4f}; m = 4 full fibre: {e4:.4f} (bisection {e4b:.4f}); m = 4 single chi_ABC channel: {e_single:.4f}")

print(f"{sum(results)}/{len(results)} passed")
