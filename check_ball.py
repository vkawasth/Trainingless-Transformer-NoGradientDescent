"""check_ball.py -- perturbing connexions, data and fibres (note: level2/partial_gluing.pdf, Section 14).

  BL1 with no budget and no data move, the support of the copy system (disagreement form) equals the classical fraction of the consistified
      system (cellwise form, 1 - CNTF) on Snow Queen and on random cyclic systems of rank 3-5
  BL2 cyclic systems: lam(r) = min(1, 1 - c + r) for a total connexion budget r (the CbD degree c is the radius of the smallest budget ball
      that makes the system noncontextual), on Snow Queen and random cyclic systems
  BL3 lam(r) is nondecreasing and concave (parametric LP), on a non-cyclic system (two A settings, three B settings, one odd context)
  BL4 grading: in that system the connexion lying on both contextual cycles (B3) recovers the full budget profile alone; every other connexion,
      relaxed alone, stalls below 1; the shadow price of the budget is 1 until saturation, then 0
  BL5 the data ball: c_min(rho) (smallest CbD degree among tables within TV rho of each observed window) equals c at rho = 0, is
      nonincreasing, and on cyclic systems max lam over the ball = 1 - c_min(rho)
  BL6 level two: along a path from a genuine two-type mixture out of the model, every rung change coincides with a sign change of Delta, of
      Det, or of a box coordinate (a weight or susceptibility leaving [0, 1]); between walls the fibre moves continuously
  BL7 level two: the data ball keeps every sampled table in the mixture's rung below the wall distance, and not above twice it
Writes ../level2/ball_results.json.  Run: python check_ball.py
"""
import itertools, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import Window
from pce.cbd import cbd
from pce.cbdfrac import lam_classical_cbd
from pce.ball import CopySystem, support, profile, min_contextuality, track_fibre, wall_distance, ball_rungs, _l2
from pce.level2 import lcm_law

rng = np.random.default_rng(20261025); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))

SQ = json.load(open(os.path.join(HERE, "study8_snowqueen", "data_study8.json"))); KEYS = ["11", "12", "21", "22"]
def sq_windows(ds):
    return [Window((f"A{k[0]}", f"B{k[1]}"), np.asarray(SQ[ds][k], float) / np.sum(SQ[ds][k])) for k in KEYS]
def cyclic(n):
    W = []
    for i in range(n):
        T = (np.array([[0, 1], [1, 0]]) if i == n - 1 else np.eye(2)) / 2
        T = 0.75 * T + 0.25 * rng.dirichlet(np.ones(4)).reshape(2, 2)
        a = rng.uniform(0, 0.08); T = (1 - a) * T + a * np.outer([1, 0], T.sum(axis=0)); W.append(Window((f"R{i}", f"R{(i+1)%n}"), T))
    return W
def table(E, ma, mb):
    pp = (1 + E) / 4 + (ma - 0.5) / 2 + (mb - 0.5) / 2
    return np.array([[pp, ma - pp], [mb - pp, 1 - ma - mb + pp]])
v = 0.7
E23 = {(1, 1): v, (1, 2): v, (1, 3): v, (2, 1): v, (2, 2): v, (2, 3): -v}
mA = {(1, 1): 0.55, (1, 2): 0.5, (1, 3): 0.45, (2, 1): 0.5, (2, 2): 0.52, (2, 3): 0.5}
mB = {(1, 1): 0.5, (1, 2): 0.5, (1, 3): 0.56, (2, 1): 0.5, (2, 2): 0.5, (2, 3): 0.5}
W23 = [Window((f"A{x}", f"B{y}"), table(E23[x, y], mA[x, y], mB[x, y])) for x in (1, 2) for y in (1, 2, 3)]

# BL1
cases = [("Snow Queen D2", sq_windows("D2")), ("Snow Queen D3", sq_windows("D3"))] + [(f"cyclic {n}", cyclic(n)) for n in (3, 4, 5) for _ in range(4)]
dev = max(abs(support(CopySystem(W)) - lam_classical_cbd(W)) for _, W in cases)
check("BL1 support at zero budget = lam_C,cbd (disagreement form = cellwise form)", dev < 1e-8, f"{len(cases)} systems, largest deviation {dev:.1e}")

# BL2
dev = 0.0; nctx = 0
for _, W in cases:
    S = CopySystem(W); c = cbd(W)["c"]; nctx += c > 1e-6
    rs = [0, c / 3, c / 2, c, c + 0.1]; dev = max(dev, max(abs(a - min(1, 1 - c + r)) for a, r in zip(profile(S, rs), rs)))
check("BL2 cyclic: lam(r) = min(1, 1 - c + r)", dev < 1e-8 and nctx >= 8, f"{len(cases)} systems ({nctx} contextual), largest deviation {dev:.1e}")

# BL3, BL4
S23 = CopySystem(W23); rs = np.linspace(0, 0.4, 21); tot = profile(S23, rs)
d1 = np.diff(tot); d2 = np.diff(tot, 2)
check("BL3 lam(r) nondecreasing and concave (non-cyclic system)", np.all(d1 >= -1e-9) and np.all(d2 <= 1e-9),
      f"lam(0) = {tot[0]:.3f}, saturates at r = {rs[np.argmax(tot > 1 - 1e-9)]:.2f}")
single = {S23.labels()[k]: profile(S23, rs, which=k) for k in range(len(S23.pairs))}
b3 = [k for k in single if k.startswith("B3")][0]
others_max = max(single[k][-1] for k in single if k != b3)
d = support(S23, r=0.05, duals=True); d_sat = support(S23, r=0.4, duals=True)
ok = np.allclose(single[b3], tot, atol=1e-8) and others_max < 1 - 0.1 and abs(d["price_r"] - 1) < 1e-8 and abs(d_sat["price_r"]) < 1e-8
OUT["noncyclic"] = {"r": rs.tolist(), "total": tot.tolist(), "single": {k: v.tolist() for k, v in single.items()}}
check("BL4 grading: the connexion on both contextual cycles (B3) carries the whole profile; the others stall; price 1 then 0", ok,
      f"B3 alone reaches {single[b3][-1]:.3f}; best other connexion alone {others_max:.3f}")

# BL5
S3 = CopySystem(sq_windows("D3")); c3 = cbd(sq_windows("D3"))["c"]; rhos = [0, 0.004, 0.008, 0.012, 0.016, 0.02]
cm = [min_contextuality(S3, r) for r in rhos]; lm = [support(S3, data_radius=r) for r in rhos]
ok = abs(cm[0] - c3) < 1e-8 and all(cm[i + 1] <= cm[i] + 1e-9 for i in range(len(cm) - 1)) and max(abs(a - (1 - b)) for a, b in zip(lm, cm)) < 1e-7
OUT["snowqueen_data_ball"] = {"rho": rhos, "c_min": cm, "lam_max": lm}
check("BL5 data ball: c_min(0) = c, nonincreasing; max lam over the ball = 1 - c_min (Snow Queen, all responses)", ok,
      "c_min: " + ", ".join(f"{x:.3f}" for x in cm))

# BL6
w0 = [0.4, 0.6]; a0 = np.array([[0.2, 0.3, 0.25], [0.8, 0.7, 0.75]]); P0 = lcm_law(w0, a0)
found = {}
while len(found) < 2:
    Q = rng.dirichlet(np.ones(8)); r = _l2(Q)["rung"]
    if r in ("R", "C") and r not in found: found[r] = Q
ok = True; info = []
for tgt, P1 in found.items():
    rows, walls = track_fibre(P0, P1, 801)
    for k in range(1, len(rows)):
        if rows[k]["code"] != rows[k - 1]["code"]:
            a_, b_ = rows[k - 1], rows[k]
            flip = np.sign(a_["Delta"]) != np.sign(b_["Delta"]) or np.sign(a_["Det"]) != np.sign(b_["Det"]) or {a_["code"], b_["code"]} == {0, 1}
            ok &= bool(flip)
        elif rows[k]["code"] in (0, 1, 2):
            ok &= abs(rows[k]["w1"] - rows[k - 1]["w1"]) < 0.05 or min(rows[k]["w1"], 1 - rows[k]["w1"]) < 0.06
    info.append(f"to {tgt}: walls at t = " + ", ".join(f"{w:.3f}" for w in walls)); OUT[f"path_{tgt}"] = {"rows": rows, "walls": walls}
OUT["P0"] = P0.ravel().tolist(); OUT["P1"] = {k: v.tolist() for k, v in found.items()}
check("BL6 level two: rung changes are sign changes of Delta, Det or a box coordinate; the fibre is continuous between walls", ok, "; ".join(info))

# BL7
wd = wall_distance(P0, 300); lo = ball_rungs(P0, 0.9 * wd, 800); hi = ball_rungs(P0, 2 * wd, 800)
ok = lo["R>=0"] == 1.0 and hi["R>=0"] < 1.0
prof = {float(r): ball_rungs(P0, r, 600, seed=1) for r in np.linspace(0.005, 0.15, 12)}
OUT["ball"] = {"wall_distance": wd, "profile": prof}
check("BL7 level two: data ball stays in the mixture rung below the wall distance, not above twice it", ok,
      f"wall distance {wd:.4f}; share R>=0 at 0.9x: {lo['R>=0']:.3f}, at 2x: {hi['R>=0']:.3f}")

json.dump(OUT, open(os.path.join(HERE, "..", "level2", "ball_results.json"), "w"), default=float)
print(f"\n{sum(results)}/{len(results)} checks passed")
