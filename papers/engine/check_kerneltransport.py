"""Checks for 'Revisions that change the variables' (module pce/kerneltransport.py). Fine version: X in {a,b,c,d}, Y in {0,1};
coarse version: X' in {A = {a,b}, B = {c,d}}, Y. Fine data: tables of {X} and {X', Y} unless stated.
  KT1 IDENTITY  fibre transport: the merge kernel carries every law of the fine fibre into the coarse fibre of the pushed-forward data,
                and onto it (every coarse law of that fibre has a preimage); with target data that are not the pushforward, no fine law
                lands in the target fibre
  KT2 IDENTITY  refit then merge equals merge then refit when every data table is a function of the coarse variables (30 random cases,
                to 1e-12); with a fine margin among the data the two orders differ (a transport/refitting defect)
  KT3 IDENTITY  representation holonomy of two paths over the fibre is a maximum over its vertices: merge-then-uniform-split against
                doing nothing gives H > 0, never exceeded by 5000 sampled laws; relabelling inside the classes then merging against merging
                gives H = 0
  KT4 IDENTITY  audit, part one: two declared kernel paths applied to the same law differ by at most H (2000 random laws of the fibre)
  KT5 IDENTITY  audit, part two: after merging, splitting and refitting, the updater's law returns to the honest refit exactly when the
                split kernel's odds inside each class do not depend on the outcome (50 random kernels of each kind); an outcome-dependent
                split launders P(Y=1 | a) - P(Y=1 | b) to either sign, consistent with all data, while the recomputer's fibre leaves it open
  KT6 IDENTITY  if the fine table of {X, Y} is kept among the data, no split kernel can launder: the refit returns the data's law
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from pce import kerneltransport as KT
from scipy.optimize import linprog
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(23); F, C = KT.FINE, KT.COARSE; M = KT.merge(); U = np.ones(8) / 8
P0 = np.array([0.10, 0.06, 0.04, 0.12, 0.14, 0.09, 0.15, 0.30]); P0 = P0 / P0.sum()
data = [(KT.ctx_X, KT.table(P0, KT.ctx_X, F)), (KT.ctx_Xc_Y, KT.table(P0, KT.ctx_Xc_Y, F))]
V = KT.vertices(data, F)
def in_fibre(P, tables, outs): A, b = KT.constraints(tables, outs); return np.abs(A @ P - b).max() < 1e-9

# KT1
cdata = [(KT.ctx_c, KT.table(P0 @ M, KT.ctx_c, C)), (KT.ctx_cY, KT.table(P0 @ M, KT.ctx_cY, C))]
ok = all(in_fibre(w @ V @ M, cdata, C) for w in rng.dirichlet(np.ones(len(V)), 200))
Vc = KT.vertices(cdata, C); A, b = KT.constraints(data, F)
for w in rng.dirichlet(np.ones(len(Vc)), 50):
    target = w @ Vc; Aeq = np.vstack([A, M.T]); beq = np.concatenate([b, target])
    ok &= linprog(np.zeros(8), A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * 8, method="highs").status == 0
bad = 0
for _ in range(50):
    pert = {k: v for k, v in cdata[1][1].items()}; ks = list(pert); d = rng.uniform(0.01, 0.05); pert[ks[0]] += d; pert[ks[1]] -= d
    tgt = [(KT.ctx_cY, pert)]; At, bt = KT.constraints(tgt, C)
    r = linprog(np.zeros(8), A_eq=np.vstack([A, At @ M.T]), b_eq=np.concatenate([b, bt]), bounds=[(0, None)] * 8, method="highs"); bad += r.status == 0
check("KT1 IDENTITY the merge kernel maps the fine fibre onto the coarse fibre of the pushed data, and into no other", ok and bad == 0,
      f"200 laws mapped, 50 coarse laws lifted, 50 perturbed targets unreachable")

# KT2
cX = lambda w: KT.cls(w[0]); ok = True; defects = []
for _ in range(30):
    P = rng.dirichlet(np.ones(8) * 2); Q = rng.dirichlet(np.ones(8))
    cd = [(KT.ctx_c, KT.table(P @ M, KT.ctx_c, C)), (KT.ctx_Y, KT.table(P @ M, KT.ctx_Y, C))]
    b_ = KT.refit(Q @ M, cd, C)
    a1 = KT.refit(Q, [(cX, KT.table(P, cX, F)), (KT.ctx_Y, KT.table(P, KT.ctx_Y, F))], F) @ M
    a2 = KT.refit(Q, [(KT.ctx_X, KT.table(P, KT.ctx_X, F)), (KT.ctx_Y, KT.table(P, KT.ctx_Y, F))], F) @ M
    ok &= KT.tv(a1, b_) < 1e-12; defects.append(KT.tv(a2, b_))
ok &= np.mean(np.array(defects) > 1e-6) > 0.9
check("KT2 IDENTITY refit and merge commute for coarse-measurable data; a fine margin makes a defect", ok,
      f"defects with a fine X margin: median {np.median(defects):.4f}, max {max(defects):.4f}")

# KT3
Su = KT.split({(0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.5, (1, 1): 0.5}); I8 = np.eye(8)
H = KT.hol(M @ Su, I8, V); samp = max(KT.tv(P @ M @ Su, P) for P in rng.dirichlet(np.ones(len(V)) * 0.5, 5000) @ V)
perm = np.zeros((8, 8))
for i, (x, y) in enumerate(F): perm[i, F.index(({0: 1, 1: 0, 2: 3, 3: 2}[x], y))] = 1
H0 = KT.hol(perm @ M, M, V)
check("KT3 IDENTITY representation holonomy is a vertex maximum; within-class relabelling commutes with merging", H > 0 and samp <= H + 1e-12 and H0 < 1e-12,
      f"merge then uniform split vs nothing: H = {H:.4f} (sampled max {samp:.4f}); relabel then merge vs merge: {H0:.1e}")

# KT4
ok = all(KT.tv(P @ M @ Su, P) <= H + 1e-12 for P in rng.dirichlet(np.ones(len(V)), 2000) @ V)
check("KT4 IDENTITY two declared kernel paths differ by at most H on the fibre", ok)

# KT5
Hh = KT.refit(U, data, F); zero = []; pos = []
for _ in range(50):
    t = rng.uniform(0.05, 0.95, 2); th = {(0, 0): t[0], (0, 1): t[0], (1, 0): t[1], (1, 1): t[1]}
    zero.append(KT.tv(KT.refit(Hh @ M @ KT.split(th), data, F), Hh))
    t4 = rng.uniform(0.05, 0.95, 4); th = {(0, 0): t4[0], (0, 1): t4[1], (1, 0): t4[2], (1, 1): t4[3]}
    pos.append(KT.tv(KT.refit(Hh @ M @ KT.split(th), data, F), Hh))
la = {}
for name, th in (("up", {(0, 0): 0.2, (0, 1): 0.8, (1, 0): 0.5, (1, 1): 0.5}), ("down", {(0, 0): 0.8, (0, 1): 0.2, (1, 0): 0.5, (1, 1): 0.5})):
    Q = KT.refit(Hh @ M @ KT.split(th), data, F); la[name] = (KT.condY(Q, 0) - KT.condY(Q, 1), in_fibre(Q, data, F))
dv = [KT.condY(v, 0) - KT.condY(v, 1) for v in V if min(v[0] + v[1], v[2] + v[3]) > 1e-12]
ok = max(zero) < 1e-9 and min(pos) > 1e-6 and la["up"][0] > 0.1 and la["down"][0] < -0.1 and la["up"][1] and la["down"][1] and min(dv) < 0 < max(dv)
check("KT5 IDENTITY a split leaves a trace iff its within-class odds depend on the outcome; outcome-dependent splits launder either way", ok,
      f"trace: outcome-free splits max {max(zero):.1e}, outcome-dependent min {min(pos):.4f}; honest P(Y|a) - P(Y|b) = {KT.condY(Hh, 0) - KT.condY(Hh, 1):+.3f}; "
      f"laundered {la['up'][0]:+.3f} and {la['down'][0]:+.3f}; recomputer's range over the fibre [{min(dv):.3f}, {max(dv):.3f}]")

# KT6
dfine = data + [(KT.ctx_X_Y, KT.table(P0, KT.ctx_X_Y, F))]; ok = True
for _ in range(30):
    t4 = rng.uniform(0.05, 0.95, 4); th = {(0, 0): t4[0], (0, 1): t4[1], (1, 0): t4[2], (1, 1): t4[3]}
    ok &= KT.tv(KT.refit(Hh @ M @ KT.split(th), dfine, F), P0) < 1e-9
check("KT6 IDENTITY with the fine table of {X, Y} kept, no split kernel launders", ok)
json.dump(dict(H=H, defects=[float(np.median(defects)), float(max(defects))], laundered=[la["up"][0], la["down"][0]], fibre=[min(dv), max(dv)]),
          open(os.path.join(HERE, "kerneltransport_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
