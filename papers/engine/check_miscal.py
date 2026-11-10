"""Checks for 'Coherent miscalibration' (companion note; module pce/miscal.py).
IDENTITY = an exact instance of a statement proved in the text; FINDING = a computed fact about the toy, reported as such.
  MC1  IDENTITY  Prop A: snapshot coherence regions are convex, so mixtures of coherent configurations are coherent (200 random pairs);
                 along every path below the snapshot residual is 0 at every step
  MC2  IDENTITY  Prop B: a coalition's reach is the honest complement's interval and contains every member's budget (all 10 pairs);
                 for {S2, S3}: [0, 1], against S3 alone [0.365, 1] and S2 alone a point
  MC3  IDENTITY  Prop C: the anchor (X3, X5) gives the anchored interval [0.613, 0.930] against {S2, S3}, above the threshold (a
                 certificate); the anchor (X2, X4), which settled B against S3 alone, gives [0, 1] against the pair; along the coalition path
                 every configuration on the wrong side of the threshold is incoherent with the (X3, X5) anchor
  MC4  FINDING   matched paths from the same start over 20 steps: the coalition path A (S2, S3 move, B to 0.35) and the honest drift B0 of
                 the world to the coalition's own endpoint law produce identical reports, so every report-based statistic agrees
                 (boundary-directed movement, speed, revision set, snapshot residual); only the independent anchor separates them (residual
                 0.525 against 0); an honest coupling drift B1 to the same value of B has the same boundary-directed movement but moves all
                 five reports; a generic honest drift B2 of matched displacement barely moves toward the boundary
  MC5  FINDING   environment-coupled deformation: a coalition that advances only while the decision maker still accepts stops at the first
                 step where his decision flips; every state on the way is coherent; its reach lies inside the coalition interval
"""
import os, sys, itertools
import numpy as np
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import miscal as MC, bridgetoy as BT, tamper as T
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
n = 5; rng = np.random.default_rng(20261109)
J0 = np.full(5, 0.7); field = np.array([0.15, 0, 0, 0, 0]); P = MC.ising(J0, field); _, gA, gB = BT.world()
hon = MC.reports(P); anch = [hon["S1"], hon["S4"], hon["S5"]]; AC = (2, 4); TARGET = 0.35

# paths
pathA, p1 = MC.coalition_path(P, ["S2", "S3"], anch, gB, TARGET); lams = np.linspace(0, 1, 21)
lawsB0 = [(1 - l) * P + l * p1 for l in lams]; pathB0 = [MC.reports(p) for p in lawsB0]
j1 = brentq(lambda j: gB @ MC.ising(np.r_[0.7, 0.7, j, 0.7, 0.7], field) - TARGET, -3, 0.7)
pathB1, lawsB1 = MC.world_path(J0, np.r_[0.7, 0.7, j1, 0.7, 0.7], field)
def disp_one(j):
    pth, _ = MC.world_path(J0, np.r_[j, 0.7, 0.7, 0.7, 0.7], field, T_=1); return MC.stats(pth, [BT.table(P, AC)] * 2, gB, AC)["displacement"]
sA = MC.stats(pathA, [BT.table(P, AC)] * 21, gB, AC)
j2 = brentq(lambda j: disp_one(j) - sA["displacement"], -0.5, 0.0)
pathB2, lawsB2 = MC.world_path(J0, np.r_[j2, 0.7, 0.7, 0.7, 0.7], field)
S = {"A": sA, "B0": MC.stats(pathB0, [BT.table(p, AC) for p in lawsB0], gB, AC),
     "B1": MC.stats(pathB1, [BT.table(p, AC) for p in lawsB1], gB, AC), "B2": MC.stats(pathB2, [BT.table(p, AC) for p in lawsB2], gB, AC)}

# MC1
ok = True
for _ in range(200):
    q0, q1 = rng.dirichlet(np.ones(32)), rng.dirichlet(np.ones(32)); lam = rng.random()
    c0 = [(s, BT.CTX[s], BT.table(q0, BT.CTX[s])) for s in ("S1", "S4", "S5", "S2")]
    mix = [(c[0], c[1], {k: (1 - lam) * c[2][k] + lam * BT.table(q1, c[1])[k] for k in c[2]}) for c in c0]
    ok &= T.residual(n, mix) < 1e-8
check("MC1 IDENTITY coherence regions are convex; every path stays coherent at every step",
      ok and all(S[k]["max_res_no_anchor"] < 1e-8 for k in S), f"max snapshot residual along A, B0, B1, B2: { {k: round(S[k]['max_res_no_anchor'], 12) for k in S} }")

# MC2
ok = True; pairs = {}
for a, b in itertools.combinations(BT.LABELS, 2):
    rest = [hon[s] for s in BT.LABELS if s not in (a, b)]; Ic = T.interval(n, rest, gB)
    for m in (a, b):
        Im = T.interval(n, [hon[s] for s in BT.LABELS if s != m], gB); ok &= Ic[0] <= Im[0] + 1e-9 and Im[1] <= Ic[1] + 1e-9
    pairs[(a, b)] = (round(Ic[0], 3), round(Ic[1], 3))
I23 = T.interval(n, anch, gB)
check("MC2 IDENTITY coalition reach contains every member's budget; {S2, S3} reaches [0, 1]", ok and abs(I23[0]) < 1e-9 and abs(I23[1] - 1) < 1e-9,
      f"pair reach for B: {pairs}")

# MC3
I35 = T.interval(n, anch + [("anchor", (2, 4), BT.table(P, (2, 4)))], gB); I24 = T.interval(n, anch + [("anchor", (1, 3), BT.table(P, (1, 3)))], gB)
wrong_side_incoh = all(T.residual(n, list(pathA[t].values()) + [("anchor", AC, BT.table(P, AC))]) > 1e-6
                       for t in range(21) if MC.naive_value(list(pathA[t].values()), gB)[0] < 0.5)
check("MC3 IDENTITY anchor (X3,X5) certifies B against {S2,S3}; anchor (X2,X4) does not; crossing configurations violate the anchor",
      I35[0] > 0.5 and I24[0] < 0.5 and wrong_side_incoh, f"anchored intervals: (X3,X5) [{I35[0]:.3f}, {I35[1]:.3f}], (X2,X4) [{I24[0]:.3f}, {I24[1]:.3f}]")

# MC4
same = max(abs(pathA[t][s][2][k] - pathB0[t][s][2][k]) for t in range(21) for s in pathA[t] for k in pathA[t][s][2])
rnd = {k: {kk: (round(v, 3) if isinstance(v, (float, np.floating)) else v) for kk, v in S[k].items()} for k in S}
check("MC4 FINDING A and B0 give identical reports; only the anchor separates them; B1 matches A's direction but moves all reports; B2 barely moves toward the boundary",
      same < 1e-12 and abs(S["A"]["D_T"] - S["B0"]["D_T"]) < 1e-9 and S["A"]["moved"] == S["B0"]["moved"] == ["S2", "S3"]
      and S["A"]["max_res_anchor"] > 0.1 and S["B0"]["max_res_anchor"] < 1e-8 and S["B1"]["max_res_anchor"] < 1e-8
      and abs(S["B1"]["D_T"] - S["A"]["D_T"]) < 1e-6 and len(S["B1"]["moved"]) == 5 and S["B2"]["margin1"] > 0 and S["B2"]["D_T"] < 0.1,
      f"{rnd}")

# MC5
lam, step, states = 0.0, 0.05, []
while lam <= 1 + 1e-12:
    pl = (1 - lam) * P + lam * p1; rep = dict(hon); rep["S2"] = ("S2", BT.CTX["S2"], BT.table(pl, BT.CTX["S2"])); rep["S3"] = ("S3", BT.CTX["S3"], BT.table(pl, BT.CTX["S3"]))
    v, r = MC.naive_value(list(rep.values()), gB); states.append((lam, v, r))
    if v < 0.5: break
    lam += step
check("MC5 FINDING the feedback-driven coalition stops at the first flip; coherent throughout; reach inside the coalition interval",
      states[-1][1] < 0.5 and all(s[1] >= 0.5 for s in states[:-1]) and all(s[2] < 1e-8 for s in states) and I23[0] <= states[-1][1] <= I23[1],
      f"stopped at lambda = {states[-1][0]:.2f} with B-hat = {states[-1][1]:.3f} after {len(states) - 1} advances")
print(f"{sum(results)}/{len(results)} passed")
