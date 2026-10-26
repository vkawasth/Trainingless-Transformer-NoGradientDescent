"""check_timesheaf.py -- decisions on a time-varying sheaf of evidence, with covers chosen by robust satisficing (note: Section 17).

  TS1 covers: C subset of C' implies lam*(C) >= lam*(C'), and for alpha <= lam*(C') the event interval over E_alpha(C') lies inside the
      interval over E_alpha(C) (more evidence: less glueable, more informative)
  TS2 acyclic covers glue when their overlaps agree (Vorob'ev); with direct influence even a tree can fail (the worked example at t3)
  TS3 awareness growth (refining outcomes, no drift) never raises lam*, and coarsening maps E_alpha(after) into E_alpha(before)
  TS4 new windows never raise lam*; a pendant window (new variables attached at one old variable, with that variable's marginal equal to
      an existing window's) leaves lam* unchanged; a pendant with a different marginal can lower it
  TS5 drift: if every window moves by at most d in total variation, lam* moves by at most m d (m = number of windows)
  TS6 robust value and robustness: V(a; alpha) is attained by an explanation in E_alpha and bounds every other; the requirement holds on
      the whole robustness interval; frontier points are non-dominated
  TS7 the selection hazard: on structureless random tables, choosing the cover after seeing the data inflates the apparent glueability
      (lam* of the best proper sub-cover against the full cover), and in some systems makes an act look good enough that is not good
      enough on all the evidence
  TS8 the worked example: lam* over time (1, 0.86, 0.86, 0.874, 0.874), the decision sequence (triple, triple, pair, pair, pair), the
      expiry at t2 by awareness growth with lam* unchanged, and the support of the expert hypothesis "triple harm <= 0.60" at each
      time (full until t1, below lam* after awareness growth and drift: the hypothesis cuts the explanation polytope)
Writes ../level2/timesheaf_results.json.  Run: python check_timesheaf.py
"""
import itertools, json, os, sys, warnings
import numpy as np
from scipy.optimize import linprog
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import Window
from pce.timesheaf import (System, lam, interval, robust_value, robustness, covers, frontier, coarsen, coarsen_table, drift,
                           lam_under, _lp_setup, _vec)
from timesheaf_example import steps, ACTS, U0, any_harm, pair_harm, REFINE_C

rng = np.random.default_rng(20261026); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
S, labels = steps()
D3 = {"A": 2, "B": 2, "C": 2}; PAIRS = [("A", "B"), ("B", "C"), ("A", "C")]
def pairs_of(P): return {("A", "B"): P.sum(axis=2), ("B", "C"): P.sum(axis=0), ("A", "C"): P.sum(axis=1)}
def rand_sys(conflict=True):
    P1 = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); P2 = rng.dirichlet(np.ones(8)).reshape(2, 2, 2) if conflict else P1
    T = pairs_of(P1); T[("B", "C")] = pairs_of(P2)[("B", "C")]
    return System(D3, [Window(p, T[p]) for p in PAIRS], ["AB", "BC", "AC"])

# TS1
ok = True; npairs = 0
for s in S[:4] + [rand_sys() for _ in range(6)]:
    cs = covers(s)
    for c1 in cs:
        for c2 in cs:
            if set(c1) < set(c2):
                l1, l2 = lam(s, c1), lam(s, c2); ok &= l1 >= l2 - 1e-9; npairs += 1
                al = min(l2, 0.95) - 1e-6; i1 = interval(s, any_harm, al, c1); i2 = interval(s, any_harm, al, c2)
                ok &= i1[0] <= i2[0] + 1e-7 and i2[1] <= i1[1] + 1e-7
check("TS1 covers: lam* decreases and event intervals shrink as windows are added to a cover", ok, f"{npairs} nested pairs of covers")

# TS2
ok = True
for _ in range(20):
    P = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); T = pairs_of(P); s = System(D3, [Window(p, T[p]) for p in PAIRS])
    for c in ((0, 1), (0, 2), (1, 2)): ok &= lam(s, c) > 1 - 1e-9
tree_t3 = lam(S[3], (0, 1)); ok &= lam(S[0]) > 1 - 1e-9 and tree_t3 < 1 - 1e-3
check("TS2 acyclic covers glue when overlaps agree; with direct influence a tree can fail", ok,
      f"60 random trees glue; worked example t3: tree AB+AC has lam* = {tree_t3:.3f} (A's marginal differs between AB and AC)")

# TS3
def refine_window(T, vars_, var, share):
    """split outcome 1 of var into (1, 2) with severe share 'share'"""
    ax = vars_.index(var); shp = list(T.shape); shp[ax] = 3; R = np.zeros(shp)
    sl = lambda k: tuple(slice(None) if i != ax else k for i in range(T.ndim))
    R[sl(0)] = T[sl(0)]; R[sl(1)] = T[sl(1)] * (1 - share); R[sl(2)] = T[sl(1)] * share; return R
ok = True; worst = 0.0
d12 = drift(S[1], S[2], REFINE_C, [(0, 0), (1, 1), (2, 2)]); ok &= d12 < 1e-12 and lam(S[2]) <= lam(S[1]) + 1e-9
for _ in range(25):
    s = rand_sys(); Dn = {"A": 2, "B": 2, "C": 3}
    Wn = [Window(w.vars, refine_window(w.table, list(w.vars), "C", rng.uniform(0.1, 0.9)) if "C" in w.vars else w.table) for w in s.windows]
    sn = System(Dn, Wn); l0, l1 = lam(s), lam(sn); ok &= l1 <= l0 + 1e-9; worst = max(worst, l1 - l0)
# pushforward of E_alpha(t2) into E_alpha(t1) (random LP vertices of E_alpha(t2), coarsened)
A2, t2, at2, nm2 = _lp_setup(S[2], None); A1, t1, at1, nm1 = _lp_setup(S[1], None); al = lam(S[2]) - 1e-6; inside = True
for _ in range(20):
    r = linprog(rng.normal(size=len(at2)), A_ub=al * A2, b_ub=t2, A_eq=np.ones((1, len(at2))), b_eq=[1.0], bounds=(0, None), method="highs")
    q1 = coarsen(r.x, at2, nm2, at1, nm1, REFINE_C); inside &= np.all(al * A1 @ q1 <= t1 + 1e-7)
ok &= inside
check("TS3 awareness growth never raises lam*, and coarsening maps E_alpha(after) into E_alpha(before)", ok,
      f"worked example t1 -> t2: drift {d12:.0e}, lam* {lam(S[1]):.3f} -> {lam(S[2]):.3f}; 25 random refinements (largest change {worst:+.1e}); 20 vertices pushed forward")

# TS4
ok = lam(S[1]) <= lam(S[0]) + 1e-9 and abs(lam(S[4]) - lam(S[3])) < 1e-9; rnd_ok = True; lowered = 0
for _ in range(20):
    s = rand_sys(); base = lam(s)
    # a new random window on an existing pair can only lower lam*
    extra = System(D3, s.windows + [Window(("A", "B"), rng.dirichlet(np.ones(4)).reshape(2, 2))]); rnd_ok &= lam(extra) <= base + 1e-9
    # pendant on B with B's marginal from the AB window: unchanged; with a different B marginal: may drop
    pB = s.windows[0].table.sum(axis=0); k = rng.dirichlet(np.ones(2), 2)
    D4 = {"A": 2, "B": 2, "C": 2, "D": 2}
    same = System(D4, s.windows + [Window(("B", "D"), pB[:, None] * k)]); rnd_ok &= abs(lam(same) - base) < 1e-9
    pB2 = rng.dirichlet(np.ones(2)); other = System(D4, s.windows + [Window(("B", "D"), pB2[:, None] * k)])
    rnd_ok &= lam(other) <= base + 1e-9; lowered += lam(other) < base - 1e-6
ok &= rnd_ok and lowered > 0
check("TS4 new windows never raise lam*; a pendant with a matching marginal leaves it unchanged, a mismatched one can lower it", ok,
      f"worked example: t0 -> t1 {lam(S[0]):.3f} -> {lam(S[1]):.3f}, t3 -> t4 {lam(S[3]):.4f} -> {lam(S[4]):.4f}; 20 random systems, mismatched pendant lowered lam* in {lowered}")

# TS5
d23 = drift(S[2], S[3], {}, [(0, 0), (1, 1), (2, 2)]); ok = abs(lam(S[3]) - lam(S[2])) <= 3 * d23 + 1e-9; ratios = []
for _ in range(30):
    s = rand_sys(); d = rng.uniform(0.005, 0.08); Wn = []
    for w in s.windows:
        T = w.table.ravel(); v = rng.normal(size=T.size); v -= v.mean(); v *= d / (0.5 * np.abs(v).sum()); Tn = T + v
        if Tn.min() < 0: Tn = T                                      # skip infeasible moves
        Wn.append(Window(w.vars, Tn.reshape(w.table.shape)))
    sn = System(D3, Wn); dd = drift(s, sn, {}, [(i, i) for i in range(3)])
    if dd > 0: ratios.append(abs(lam(sn) - lam(s)) / (3 * dd)); ok &= abs(lam(sn) - lam(s)) <= 3 * dd + 1e-9
check("TS5 drift: |lam*(t') - lam*(t)| <= m d", ok, f"worked example t2 -> t3: d = {d23:.4f}, change {lam(S[3]) - lam(S[2]):+.4f} <= {3 * d23:.4f}; "
      f"30 random drifts, largest ratio {max(ratios):.2f}")

# TS6
ok = True
for s in S:
    for a, u in ACTS.items():
        al = lam(s) - 1e-6; V = robust_value(s, u, al); A, t, atoms, names = _lp_setup(s, None); U = _vec(u, atoms, names)
        for _ in range(10):
            r = linprog(rng.normal(size=len(atoms)), A_ub=al * A, b_ub=t, A_eq=np.ones((1, len(atoms))), b_eq=[1.0], bounds=(0, None), method="highs")
            ok &= V <= al * U @ r.x + (1 - al) * U.min() + 1e-9
        h = robustness(s, u, U0)
        if h > -np.inf:
            for e in np.linspace(1 - al, h, 5): ok &= robust_value(s, u, 1 - e) >= U0 - 1e-9
    pts = [(c, lam(s, c), np.diff(interval(s, any_harm, lam(s, c) - 1e-9, c))[0]) for c in covers(s)]
    fr = frontier(pts); ok &= all(not any(l2 >= l and w2 <= w and (l2 > l + 1e-12 or w2 < w - 1e-12) for _, l2, w2 in pts) for _, l, w in fr)
check("TS6 robust values are attained lower bounds; the requirement holds on the robustness interval; frontiers are non-dominated", ok)

# TS7
infl_l = []; created = 0; nsys = 40
for _ in range(nsys):
    s = rand_sys(); full = lam(s); best_sub = max(lam(s, c) for c in covers(s) if len(c) < 3); infl_l.append(best_sub - full)
    hf = max(robustness(s, u, U0) for u in ACTS.values()); hb = max(robustness(s, u, U0, c) for c in covers(s) for u in ACTS.values())
    created += (hf == -np.inf) and (hb > -np.inf)
OUT["selection"] = {"lam_inflation": infl_l, "satisficing_created_by_selection": created, "systems": nsys}
check("TS7 selection hazard: choosing the cover after seeing the data inflates apparent glueability, and can make an act look good enough",
      np.mean(infl_l) > 0.05 and created > 0,
      f"{nsys} structureless systems: the best proper sub-cover glues {np.mean(infl_l):.2f} more on average; in {created} of them no act is good "
      f"enough on all windows but one is on a sub-cover chosen afterwards")

# TS8
lams = [lam(s) for s in S]; choice = []; hyp = []
H = lambda x: any_harm(x)
for s in S:
    best = max(((c, a, robustness(s, u, U0, c)) for c in covers(s) for a, u in ACTS.items()), key=lambda r: r[2]); choice.append(best)
    hyp.append(lam_under(s, H, 0.60))
acts = [b[1] for b in choice]
ok = np.allclose(lams, [1, 0.86, 0.86, 0.874, 0.874], atol=1e-3) and acts == ["triple A+B+C", "triple A+B+C", "pair A+B", "pair A+B", "pair A+B"] \
     and abs(lams[2] - lams[1]) < 1e-9 and all(h <= l + 1e-9 for h, l in zip(hyp, lams)) and hyp[1] > lams[1] - 1e-6 and hyp[2] < lams[2] - 1e-3 and hyp[3] < lams[3] - 1e-3
OUT["example"] = {"labels": labels, "lam": lams, "choice": [["+".join(S[i].names[j] for j in b[0]), b[1], b[2]] for i, b in enumerate(choice)],
                  "hypothesis_support": hyp}
check("TS8 worked example: lam* over time, the decision sequence, expiry at t2 by awareness growth, hypothesis support", ok,
      "lam* " + ", ".join(f"{l:.3f}" for l in lams) + "; acts " + ", ".join(a.split()[0] for a in acts) + "; support of 'triple harm <= 0.60' "
      + ", ".join(f"{h:.3f}" for h in hyp))

json.dump(OUT, open(os.path.join(HERE, "..", "level2", "timesheaf_results.json"), "w"), default=float)
print(f"\n{sum(results)}/{len(results)} checks passed")
