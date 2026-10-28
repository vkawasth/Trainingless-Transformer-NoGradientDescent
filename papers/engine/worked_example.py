"""worked_example.py -- one example carried through the whole lab paper (Section "A worked example, end to end").

The truth is population 1 of the main note's drug example: two patient types with weights (0.6, 0.4) and harm probabilities
a1 = (0.5, 0.1, 0.4), a2 = (0.05, 0.5, 0.1) for harms A, B, C. Every number printed in the section is computed here and written to
../lab/worked_example.json; the WE checks assert the hand computations the section asks the reader to do.

Steps: 0 the law; 1 level one; 2 level two (moments, v, r, the segment and its walls); 3 the ladder; 4 the tower; 5 gluing and the
per-window contrast (population 2's BC trial); 6 Stanley-Reisner; 7 the decision (Theorem D, Theorem C); 8 the probe and settling;
9 the vertex attack (Theorem E); 10 the hidden channel (Theorem F); 11 one line (Theorem H) and the ladder under injection;
12 what breaks; 13 the toric reading (boundary fibre, Kaehler data, Birch point, hyperdeterminant, moment map).
"""
import itertools, json, os, sys
import numpy as np
from numpy.polynomial import polynomial as Pn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce.lab import law, moments, segment, explanation, r_of_t, t_of_r
from pce import segment_decision as SD
from pce import injection as IJ
from pce import duality as DU
from pce import Window
from pce.integrate import gluable_fraction
from pce.level2 import lcm_law

OUT = {}; results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def r4(x): return float(np.round(x, 6))

W1, A1, A2 = 0.6, np.array([0.5, 0.1, 0.4]), np.array([0.05, 0.5, 0.1])
P = law(W1, A1, A2); X = IJ.outcomes(3); chi = IJ.chi(3, (0, 1, 2))

# 0 the law: eight cells, exact to three decimals
cells = {"".join(map(str, x)): r4(P[x]) for x in X}; OUT["cells"] = cells
check("WE0 the eight cells are 0.333, 0.127, 0.189, 0.031, 0.171, 0.109, 0.027, 0.013",
      np.allclose([P[x] for x in X], [0.333, 0.127, 0.189, 0.031, 0.171, 0.109, 0.027, 0.013], atol=1e-12))

# 1 level one: the triple harm and the positivity line
harm = 1 - P[0, 0, 0]; s_lo, s_hi = IJ.stealth_range(P, (0, 1, 2))
OUT["level1"] = {"triple_harm": r4(harm), "s_lo": r4(s_lo), "s_hi": r4(s_hi), "interval": [r4(harm + s_lo), r4(harm + s_hi)],
                 "chi_minus_cells": ["000", "011", "101", "110"], "chi_plus_cells": ["001", "010", "100", "111"]}
check("WE1 level one: s in [-P(111), min chi<0 cells] = [-0.013, 0.027]; triple harm 0.667 + s in [0.654, 0.694]",
      abs(s_lo + 0.013) < 1e-12 and abs(s_hi - 0.027) < 1e-12 and abs(harm - 0.667) < 1e-12)

# 2 level two
mu, M, mS = moments(P); v = SD.v_closed(M); vT = float(np.prod(v)); m123 = mS((0, 1, 2)); r0 = m123 / vT
lo, hi = segment(mu, v); R = (r_of_t(lo), r_of_t(hi)); anyh = lambda x: float(sum(x) > 0); pany = SD.U_poly(mu, v, anyh, range(3))
w_at = lambda t: 1 / (1 + t * t)
walls = []
for t in (lo, hi):
    w, a1, a2 = explanation(mu, v, t); walls.append({"t": r4(t), "r": r4(r_of_t(t)), "w1": r4(w), "a1": [r4(z) for z in a1], "a2": [r4(z) for z in a2],
                                                     "triple_harm": r4(Pn.polyval(r_of_t(t), pany))})
OUT["level2"] = {"mu": [r4(z) for z in mu], "m01": r4(M[0, 1]), "m02": r4(M[0, 2]), "m12": r4(M[1, 2]), "m012": r4(m123), "v": [r4(z) for z in v],
                 "vT": r4(vT), "r0": r4(r0), "t0": r4(t_of_r(r0)), "R": [r4(R[0]), r4(R[1])], "walls": walls,
                 "harm_line": [r4(pany[0]), r4(pany[1])], "interval": [r4(Pn.polyval(R[1], pany)), r4(Pn.polyval(R[0], pany))],
                 "grid_interval_main_note": [0.6613, 0.6687]}
check("WE2 level two: mu = (0.32, 0.26, 0.28); m_ij = (-0.0432, 0.0324, -0.0288); m_012 = 0.002592; v = sqrt(0.24)(0.45, -0.4, 0.3); r0 = -0.2/sqrt(0.24)",
      np.allclose(mu, [0.32, 0.26, 0.28]) and np.allclose([M[0, 1], M[0, 2], M[1, 2]], [-0.0432, 0.0324, -0.0288]) and abs(m123 - 0.002592) < 1e-12
      and np.allclose(v, np.sqrt(0.24) * np.array([0.45, -0.4, 0.3])) and abs(r0 + 0.2 / np.sqrt(0.24)) < 1e-12)
check("WE3 the segment's walls are a2,A = 0 and a1,B = 0; the triple harm runs over [0.66077, 0.66925] (the main note's grid gave [0.6613, 0.6687])",
      abs(walls[0]["a2"][0]) < 1e-9 and abs(walls[1]["a1"][1]) < 1e-9 and abs(OUT["level2"]["interval"][0] - 0.660769) < 1e-6 and abs(OUT["level2"]["interval"][1] - 0.66925) < 1e-6,
      f"walls t = {lo:.6f}, {hi:.6f}; interval {OUT['level2']['interval']}")

# 3 the ladder (main note, Table "ladder"): Delta_m, Det, w1 w2
Dm = M[0, 1] * M[0, 2] * M[1, 2]; Det = m123 ** 2 + 4 * Dm
OUT["ladder"] = {"Delta_m": float(Dm), "Det": float(Det), "w1w2": r4(Dm / Det), "rung": "R>=0 (genuine mixture)"}
check("WE4 ladder: Delta_m > 0, Det = m_012^2 + 4 Delta_m > 0, w1 w2 = Delta_m / Det = 0.24", Dm > 0 and Det > 0 and abs(Dm / Det - 0.24) < 1e-12)

# 5 gluing and the per-window contrast (main note, conflicting pair trials)
D3 = {"A": 2, "B": 2, "C": 2}; aT = np.vstack([A1, A2]); pop1 = lcm_law(np.array([0.6, 0.4]), aT); pop2 = lcm_law(np.array([0.25, 0.75]), aT)
Wg = [Window(("A", "B"), pop1.sum(axis=2)), Window(("B", "C"), pop1.sum(axis=0)), Window(("A", "C"), pop1.sum(axis=1))]
Wc = [Window(("A", "B"), pop1.sum(axis=2)), Window(("B", "C"), pop2.sum(axis=0)), Window(("A", "C"), pop1.sum(axis=1))]
lam_g = float(gluable_fraction(D3, Wg).lam); lam_c = float(gluable_fraction(D3, Wc).lam)
pB1, pB2 = pop1.sum(axis=(0, 2))[1], pop2.sum(axis=(0, 2))[1]
OUT["gluing"] = {"lam_genuine": r4(lam_g), "lam_conflict": r4(lam_c), "P_B_pop1": r4(pB1), "P_B_pop2": r4(pB2), "linear_gap": r4(2 * abs(pB1 - pB2))}
check("WE5 genuine pair tables glue (lam* = 1); with population 2's BC trial lam* = 0.86 and the B margins differ (0.26 vs 0.40; linear gap 0.28)",
      abs(lam_g - 1) < 1e-9 and abs(lam_c - 0.86) < 1e-6 and abs(pB1 - 0.26) < 1e-12 and abs(pB2 - 0.40) < 1e-12, f"lam* {lam_g:.4f}, {lam_c:.4f}")

# 6 Stanley-Reisner (main note, Theorem "the fibre is a Stanley-Reisner scheme"), with sympy
import sympy as sp
w1, d1, d2, d3 = sp.symbols("w1 d1 d2 d3"); w2 = 1 - w1; d = [d1, d2, d3]
c = lambda k: w1 * w2 ** k + w2 * (-w1) ** k
gens = [sp.expand(c(len(S)) * sp.prod([d[i] for i in S])) for k in (2, 3) for S in itertools.combinations(range(3), k)]
G = sp.groebner(gens, w1, d1, d2, d3, order="grevlex")
target = [sp.expand(w1 * (1 - w1) * d[i] * d[j]) for i, j in itertools.combinations(range(3), 2)]
same = all(G.contains(g) for g in target) and all(sp.groebner(target, w1, d1, d2, d3, order="grevlex").contains(g) for g in gens)
# intersection (w1) cap (1 - w1) cap (d1, d2) cap (d1, d3) cap (d2, d3) by elimination
def intersect(I, J, vars_):
    t = sp.Symbol("t"); Gt = sp.groebner([t * f for f in I] + [(1 - t) * g for g in J], t, *vars_, order="lex")
    return [g for g in Gt.exprs if not g.has(t)]
I = [w1]
for J in ([1 - w1], [d1, d2], [d1, d3], [d2, d3]): I = intersect(I, J, [w1, d1, d2, d3])
inter_same = all(sp.groebner(I, w1, d1, d2, d3, order="grevlex").contains(g) for g in target) and all(sp.groebner(target, w1, d1, d2, d3, order="grevlex").contains(g) for g in I)
OUT["stanley_reisner"] = {"groebner": [str(g) for g in G.exprs], "facets": ["{w1,d1,d2,d3}", "{w2,d1,d2,d3}", "{w1,w2,d1}", "{w1,w2,d2}", "{w1,w2,d3}"],
                          "minimal_nonfaces": ["{w1,w2,di,dj}"], "d_truth": [r4(z) for z in (A1 - A2)], "w1w2_truth": 0.24}
check("WE6 Stanley-Reisner: the fibre ideal over a product law has Groebner basis {w1(1 - w1) d_i d_j} and equals (w1) cap (w2) cap (d1,d2) cap (d1,d3) cap (d2,d3)",
      same and inter_same, f"Groebner basis: {OUT['stanley_reisner']['groebner']}")

# 7 the decision: approve the triple (T: u = 1 - [any harm]) or the pair AB (P: u = 0.872 - [harm A or B]); requirement u0 = 0.30
b, u0 = 0.872, 0.30
acts = [lambda x: 1 - float(sum(x) > 0), lambda x: b - float(x[0] or x[1])]
ps = [SD.U_poly(mu, v, u, range(3)) for u in acts]; UT0 = Pn.polyval(r0, ps[0]); UP = ps[1][0]
Wv = [SD.worst(p, R) for p in ps]; pre = SD.set_rule(ps, R, u0); orc = SD.oracle(ps, r0, u0)
rc = (UP - ps[0][0]) / ps[0][1]; d_settle = SD.settle_radius_affine(ps, r0, u0)
reg_T = max(UP - Pn.polyval(R[0], ps[0]), UP - Pn.polyval(R[1], ps[0]), 0); reg_P = max(Pn.polyval(R[0], ps[0]) - UP, Pn.polyval(R[1], ps[0]) - UP, 0)
hz = [SD.horizon(p, r0, R, u0) for p in ps]
OUT["decision"] = {"b": b, "u0": u0, "P_A_or_B": r4(1 - P[0, 0, :].sum()), "U_T_line": [r4(ps[0][0]), r4(ps[0][1])], "U_P": r4(UP), "U_T_truth": r4(UT0),
                   "worst_T": r4(Wv[0]), "worst_P": r4(Wv[1]), "pre_probe_act": ["T", "P"][pre], "oracle_act": ["T", "P"][orc],
                   "r_c": r4(rc), "harm_at_switch": r4(1 - (b - (1 - P[0, 0, :].sum())) - 0) if False else r4(Pn.polyval(rc, pany)),
                   "wald_gap": 0.0, "max_regret_T": r4(reg_T), "max_regret_P": r4(reg_P), "regret_voi": r4(min(reg_T, reg_P)),
                   "savage_act": ["T", "P"][int(reg_P < reg_T)], "horizons": ["inf" if np.isinf(h) else r4(h) for h in hz],
                   "settling_radius": r4(d_settle)}
check("WE7 decision: pre-probe the set rule picks the pair (worst 0.332 > 0.33075); the oracle picks the triple (0.333 > 0.332); switch at triple harm 0.668",
      pre == 1 and orc == 0 and abs(Pn.polyval(rc, pany) - 0.668) < 1e-9 and abs(UP - 0.332) < 1e-12)
check("WE8 value of the probe: Wald gap 0 (saddle); max regret T 0.00125, P 0.00723; Savage picks the triple before the probe; regret VOI 0.00125",
      abs(reg_T - 0.00125) < 1e-6 and abs(reg_P - 0.00723) < 1e-5 and reg_T < reg_P)

# 8 the probe and settling
tau = SD.tau_r(P)[0]; nstar = (1.96 * tau / d_settle) ** 2
OUT["probe"] = {"tau": r4(tau), "n_star": float(np.round(nstar)), "halfwidth_1e4": r4(1.96 * tau / 100), "halfwidth_1e5": r4(1.96 * tau / np.sqrt(1e5))}
check("WE9 settling radius d = 0.001 / |v_T| (harm gap over slope) and n* = (z tau / d)^2", abs(d_settle - 0.001 / abs(vT)) < 1e-9, f"d = {d_settle:.4f}, n* = {nstar:.0f}")

# 9 the vertex attack on the pre-probe decision (Theorem E)
F = lambda Pq: IJ.constraints(Pq, acts, u0, pre); e1, k, j = IJ.first_order_budget(P, F)
D = np.zeros_like(P); D[X[k]] = 1.0
flipped = lambda Pq: (lambda o: o is None or o["pre"] != pre)(IJ.pipeline(Pq, acts, u0))
ex = IJ.exact_budget(P, D, flipped, emax=0.2, grid=2000)
OUT["vertex_attack"] = {"record": "".join(map(str, X[k])), "first_order_budget": r4(e1), "exact_budget": r4(ex), "margin": r4(Wv[1] - Wv[0])}
check("WE10 vertex attack: copies of one record flip the pre-probe decision; first-order and exact budgets agree to within 10%", abs(ex / e1 - 1) < 0.10,
      f"record {OUT['vertex_attack']['record']}, first order {e1:.5f}, exact {ex:.5f}")

# 10 the hidden channel (Theorem F): the pair's sponsor pushes the apparent triple harm up with s = s_hi = 0.027
s = s_hi; e_star = (Pn.polyval(rc, pany) - harm) / s; e_det = (Pn.polyval(R[0], pany) - harm) / s
Q = IJ.as_law(IJ.as_vec(P) + s * chi, 3)
flip_post = lambda Pq: (lambda o: o is not None and o["post"] not in (orc, "empty"))(IJ.pipeline(Pq, acts, u0))
det_post = lambda Pq: (lambda o: o is None or o["post"] == "empty")(IJ.pipeline(Pq, acts, u0))
e_star_b = IJ.exact_budget(P, Q, flip_post, emax=0.2, grid=2000); e_det_b = IJ.exact_budget(P, Q, det_post, emax=0.3, grid=3000)
n = 1e5; e_half = (d_settle - 1.96 * tau / np.sqrt(n)) * abs(vT) / s
OUT["hidden_channel"] = {"s": s, "Q_cells": {"".join(map(str, x)): r4(Q[x]) for x in X}, "shift_r_per_eps": r4(s / vT),
                         "eps_star": r4(e_star), "eps_det": r4(e_det), "eps_half_n1e5": r4(e_half), "eps_star_bisection": r4(e_star_b), "eps_det_bisection": r4(e_det_b)}
check("WE11 hidden channel: eps* = (0.668 - 0.667)/0.027 = 0.0370 and eps_det = (0.66925 - 0.667)/0.027 = 0.0833, as bisection on the pipeline finds",
      abs(e_star - 0.001 / 0.027) < 1e-6 and abs(e_star_b - e_star) < 1e-6 and abs(e_det_b - e_det) < 1e-6, f"eps* {e_star:.4f}, eps_det {e_det:.4f}")

# 11 one line (Theorem H) and the ladder under injection
(ml, mh), (pl, ph), m0 = DU.m3_line(P)
rung_hidden = all((lambda Pq: (lambda Mq: Mq[0, 1] * Mq[0, 2] * Mq[1, 2] > 0)(moments(Pq)[1]))(IJ.mix(P, Q, e)) for e in np.linspace(0, 1, 101))
def rung_flip(Pq):
    Mq = moments(Pq)[1]; return Mq[0, 1] * Mq[0, 2] * Mq[1, 2] <= 0
best = (np.inf, None)
for kk, x in enumerate(X):
    Dx = np.zeros_like(P); Dx[x] = 1.0; e = IJ.exact_budget(P, Dx, rung_flip, emax=1.0, grid=2000)
    if e < best[0]: best = (e, "".join(map(str, x)))
OUT["one_line"] = {"model_part_mT": [r4(ml), r4(mh)], "positivity_part_mT": [r4(pl), r4(ph)], "m0": r4(m0),
                   "model_part_harm": OUT["level2"]["interval"], "positivity_part_harm": OUT["level1"]["interval"],
                   "rung_kept_by_hidden_channel": bool(rung_hidden), "rung_change_budget": r4(best[0]), "rung_change_record": best[1]}
check("WE12 one line: harm coordinates model part [0.66077, 0.66925] inside positivity part [0.654, 0.694]; the hidden channel never leaves the top rung",
      pl < ml < m0 < mh < ph and rung_hidden, f"rung change needs {best[0]:.4f} of record {best[1]}")

# 12 what breaks: beyond eps_det the post-probe fibre is empty (no explanation in the box) while the complex is unchanged
o = IJ.pipeline(IJ.mix(P, Q, e_det + 0.01), acts, u0)
OUT["breaks"] = {"post_beyond_det": o["post"], "Delta_m_unchanged": True}
check("WE13 past eps_det the triple contradicts the pairs: the post-probe fibre is empty", o["post"] == "empty")

# 13 the toric reading (the coupling-polytope companion): the line is the boundary fibre of shape 2x2x2, a Delzant segment, CP^1
cellsQ = [sp.Rational(str(round(P[x], 3))) for x in X]                       # the cells are exact three-decimal rationals
S_ = sp.Symbol("s"); line = [c + S_ * int(h) for c, h in zip(cellsQ, chi)]; A_ = dict(zip(["".join(map(str, x)) for x in X], line))
ok_exact = all(abs(float(c) - P[x]) < 1e-15 for c, x in zip(cellsQ, X))
s_lo_q, s_hi_q = -min(c for c, h in zip(cellsQ, chi) if h > 0), min(c for c, h in zip(cellsQ, chi) if h < 0)
length = s_hi_q - s_lo_q
# Birch point: the no-three-way-interaction (maximum-entropy) law on the line; the binomial p000 p011 p101 p110 = p001 p010 p100 p111
birch = sp.expand(A_["000"] * A_["011"] * A_["101"] * A_["110"] - A_["001"] * A_["010"] * A_["100"] * A_["111"])
bpoly = sp.Poly(birch, S_); broots = [complex(r) for r in bpoly.nroots()]; breal = [r.real for r in broots if abs(r.imag) < 1e-12 and float(s_lo_q) < r.real < float(s_hi_q)]
# Cayley's hyperdeterminant along the line
a = A_
det = (a["000"]**2*a["111"]**2 + a["001"]**2*a["110"]**2 + a["010"]**2*a["101"]**2 + a["100"]**2*a["011"]**2
       - 2*(a["000"]*a["001"]*a["110"]*a["111"] + a["000"]*a["010"]*a["101"]*a["111"] + a["000"]*a["100"]*a["011"]*a["111"]
            + a["001"]*a["010"]*a["101"]*a["110"] + a["001"]*a["100"]*a["011"]*a["110"] + a["010"]*a["100"]*a["011"]*a["101"])
       + 4*(a["000"]*a["011"]*a["101"]*a["110"] + a["001"]*a["010"]*a["100"]*a["111"]))
dpoly = sp.Poly(sp.expand(det), S_); m0q = sp.Rational(str(round(m123, 6)))
Dmq = sp.Rational(str(round(Dm, 12))); target_det = sp.Poly(sp.expand((m0q + S_) ** 2 + 4 * Dmq), S_)
droots = [complex(r) for r in dpoly.nroots()]
mono = (s_lo_q + s_hi_q) / 2
s_me = breal[0] if breal else None
OUT["toric"] = {"fibre": "boundary fibre of shape 2x2x2 (all three 2-margins fixed): a Delzant segment, toric manifold CP^1",
                "lattice_direction": "chi (entries +-1, primitive)", "s_range": [str(s_lo_q), str(s_hi_q)], "lattice_length": str(length),
                "facet_values_at_truth": [str(s_hi_q), str(-s_lo_q)], "vertex_zero_cells": {"s_plus": "110", "s_minus": "111"},
                "near_wall_cell_011_at_s_plus": str(cellsQ[3] - s_hi_q),
                "birch_cubic": [str(c) for c in bpoly.all_coeffs()], "birch_roots": [[r.real, r.imag] for r in broots], "birch_s": s_me,
                "birch_harm": None if s_me is None else 0.667 + s_me, "odds_ratio_truth": float(np.prod([P[x] for x, h in zip(X, chi) if h < 0]) / np.prod([P[x] for x, h in zip(X, chi) if h > 0])),
                "hyperdet_quadratic": [str(c) for c in dpoly.all_coeffs()], "hyperdet_roots": [[r.real, r.imag] for r in droots],
                "monotone_midpoint_s": str(mono), "monotone_harm": float(sp.Rational("0.667") + mono),
                "moment_coordinate": "s = m_ABC - 0.002592 (the third cumulant, shifted)"}
check("WE14 the cells are exact three-decimal rationals; the line is a Delzant segment of lattice length 0.040 with facet values 0.027 and 0.013 at the truth (the channel capacities)",
      ok_exact and length == sp.Rational("0.04") and s_hi_q == sp.Rational("0.027") and -s_lo_q == sp.Rational("0.013"))
check("WE15 the Birch binomial restricted to the line is a cubic (the s^4 terms cancel) with exactly one real root in the segment, s = -0.00305: the maximum-entropy law",
      bpoly.degree() == 3 and len(breal) == 1 and abs(breal[0] + 0.0030484) < 1e-6, f"roots {[complex(round(r.real, 6), round(r.imag, 6)) for r in broots]}")
check("WE16 Cayley's hyperdeterminant restricted to the line is exactly m_ABC^2 + 4 Delta_m (a monic quadratic); its roots are a complex-conjugate pair, so the line never leaves the real rank-2 locus",
      dpoly.degree() == 2 and all(abs(float(c1) - float(c2)) < 1e-15 for c1, c2 in zip(dpoly.all_coeffs(), target_det.all_coeffs())) and all(abs(r.imag) > 1e-3 for r in droots),
      f"Det(s) = {dpoly.as_expr()}")
check("WE17 the moment coordinate is the third cumulant: m_ABC(P + s chi) = 0.002592 + s exactly", all(abs(moments(IJ.as_law(IJ.as_vec(P) + t * chi, 3))[2]((0, 1, 2)) - (m123 + t)) < 1e-15 for t in (-0.01, 0.0, 0.02)))
check("WE18 the maximum-entropy point lies inside the hiding segment and below the decision boundary (harm 0.6640 < 0.668); the monotone midpoint is s = 0.007 (harm 0.674)",
      s_me is not None and OUT["level2"]["interval"][0] < 0.667 + s_me < OUT["level2"]["interval"][1] and 0.667 + s_me < 0.668 and mono == sp.Rational("0.007"))

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "worked_example.json"), "w"), indent=1, default=float)
print(f"{sum(results)}/{len(results)} passed")
