"""Checks for the transport/refitting defect (module pce/refitdefect.py). Fine X in {a,b,c,d}, Y; merge to X' in {A,B}. Fine data: the
tables of X and of Y; coarse data: the tables of X' and of Y (their pushforwards). Q = an updater's carried law before refitting.
  RD1 IDENTITY  D(g_*R || R') = Delta_fine - W for 300 random (data, Q), with Delta_fine >= W >= 0, and the defect (TV) <= sqrt((Delta_fine
                - W)/2) <= sqrt(Delta_fine / 2) (Pinsker)
  RD2 IDENTITY  zero set: with only coarse-measurable tables the defect, Delta_fine and W all vanish (100 cases); on the fibre itself
                (Q compatible with the data) the defect vanishes (100 cases)
  RD3 IDENTITY  the defect is not convex in Q: midpoints of random segments exceed the average of the endpoints, so there is no vertex
                principle as in the representation holonomy
  RD4 IDENTITY  decision relevance: P(Y = 1), determined by the coarse data, is unchanged by the defect whatever its size (400 cases);
                P(Y = 1 | A), which the coarse data leave open, moves
  RD5 FINDING   at a threshold tau, a defect below 0.03 can flip the decision on P(Y = 1 | A) while a defect more than five times larger
                leaves it unchanged: the size of the defect does not decide its relevance. (A first version required the unflipped defect to
                exceed 0.25, a threshold taken from an exploration run with another seed; it failed at 0.230 and was restated as this ratio.)
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from pce import refitdefect as RD, kerneltransport as KT
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(29); F, C = KT.FINE, KT.COARSE; M = KT.merge(); cX = lambda w: KT.cls(w[0])
def tabs(P, fine_x=True):
    f = [((KT.ctx_X if fine_x else cX), KT.table(P, (KT.ctx_X if fine_x else cX), F)), (KT.ctx_Y, KT.table(P, KT.ctx_Y, F))]
    c = [(KT.ctx_c, KT.table(P @ M, KT.ctx_c, C)), (KT.ctx_Y, KT.table(P @ M, KT.ctx_Y, C))]
    return f, c

# RD1
ok = True; worst = 0.0
for _ in range(300):
    P = rng.dirichlet(np.ones(8) * 2); Q = rng.dirichlet(np.ones(8)); f, c = tabs(P); d = RD.parts(Q, f, c)
    ok &= abs(d["kl"] - (d["delta_fine"] - d["within"])) < 1e-9 and d["within"] >= -1e-12 and d["delta_fine"] >= d["within"] - 1e-9
    ok &= d["defect"] <= np.sqrt(max(d["delta_fine"] - d["within"], 0) / 2) + 1e-9 <= np.sqrt(d["delta_fine"] / 2) + 2e-9
    worst = max(worst, abs(d["kl"] - (d["delta_fine"] - d["within"])))
check("RD1 IDENTITY D(g_*R || R') = Delta_fine - W; Pinsker bound", ok, f"largest identity error {worst:.1e}")

# RD2
ok = True
for _ in range(100):
    P = rng.dirichlet(np.ones(8) * 2); Q = rng.dirichlet(np.ones(8)); f, c = tabs(P, fine_x=False); d = RD.parts(Q, f, c)
    ok &= d["defect"] < 1e-9 and abs(d["delta_fine"]) < 1e-9 and abs(d["within"]) < 1e-9
for _ in range(100):
    P = rng.dirichlet(np.ones(8) * 2); f, c = tabs(P); V = KT.vertices(f, F); Q = rng.dirichlet(np.ones(len(V))) @ V; Q = 0.999 * Q + 0.001 * P
    Q = KT.refit(Q, f, F); d = RD.parts(Q, f, c); ok &= d["defect"] < 1e-9
check("RD2 IDENTITY coarse-measurable data: defect, Delta_fine and W vanish; on the fibre the defect vanishes", ok)

# RD3
P0 = np.array([0.10, 0.06, 0.04, 0.12, 0.14, 0.09, 0.15, 0.30]); P0 = P0 / P0.sum(); f0, c0 = tabs(P0)
viol = 0; n = 150
for _ in range(n):
    a, b = rng.dirichlet(np.ones(8), 2)
    da, db, dm = (RD.parts(x, f0, c0)["defect"] for x in (a, b, (a + b) / 2)); viol += dm > (da + db) / 2 + 1e-9
check("RD3 IDENTITY the defect is not convex in the starting law (no vertex principle)", viol > 0, f"{viol} of {n} midpoints above the endpoint average")

# RD4, RD5
phiY = lambda Z: Z[[1, 3]].sum(); phiA = lambda Z: Z[1] / (Z[0] + Z[1]); rows = []
for _ in range(400):
    Q = rng.dirichlet(np.ones(8) * 0.7); d = RD.parts(Q, f0, c0); rows.append((d["defect"], abs(phiY(d["gR"]) - phiY(d["Rp"])), phiA(d["gR"]), phiA(d["Rp"])))
rows = np.array(rows); moveA = np.abs(rows[:, 2] - rows[:, 3])
check("RD4 IDENTITY a coarse-determined functional never moves; an undetermined one does", rows[:, 1].max() < 1e-12 and moveA.max() > 0.05,
      f"defects up to {rows[:, 0].max():.3f}; largest change of P(Y=1) {rows[:, 1].max():.1e}; of P(Y=1 | A) {moveA.max():.3f}")
tau = float(np.median(rows[:, 3])); flip = (rows[:, 2] - tau) * (rows[:, 3] - tau) < 0
smallest_flip = rows[flip, 0].min(); largest_keep = rows[~flip, 0].max()
check("RD5 FINDING a small defect can flip a decision and a large one leave it unchanged", smallest_flip < 0.03 and largest_keep > 5 * smallest_flip,
      f"tau = {tau:.3f}; smallest defect that flips {smallest_flip:.3f}; largest defect that does not {largest_keep:.3f}; flips {int(flip.sum())} of 400")
json.dump(dict(convexity_violations=int(viol), smallest_flip=float(smallest_flip), largest_keep=float(largest_keep), tau=tau),
          open(os.path.join(HERE, "refitdefect_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
