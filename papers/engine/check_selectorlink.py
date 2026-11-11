"""Checks for the two selectors on one fibre (module pce/selectorlink.py): the toric family u_L = argmin sum_k a_k exp(-L P_k(u)) of the
selection theorem and the maximum-entropy (I-projection) selector, on 2 x 3 couplings. Thresholds in SL2 were set after an exploration run
with seed 3 (closest approach 0.0014 on the column-binding fibre; median ratio 0.04 over 60 margins); this script uses seed 41.
  SL1 IDENTITY  stationarity: the maximum-entropy law has log P in H_D; the toric point has its term law w ~ a exp(-L P) in H_D (Birch at
                target zero); 50 margins x L in {0.5, 5, 50}
  SL2 FINDING   the toric curve passes near the maximum-entropy point but not through it: on the column-binding fibre the closest approach
                exceeds 1e-4; over 60 random margins the median closest approach is between 0.5% and 20% of the distance from the
                maximum-entropy point to the leximin point
  SL3 IDENTITY  small L: with all six cells (sum of normals zero) u_L tends to the least-squares point; with the facet terms only
                (column-binding fibre, cell 21 excluded, sum of normals (1, 0)) L u_L tends to the T = 1 critical point w_0 != 0, so the
                valuation escapes every bounded set
  SL4 FINDING   large L: for weights a in [1/2, 2]^6 the toric point tends to the leximin point at rate 1/L: L * distance converges
                (relative change between L = 1e3 and 1e4 below 1%, 50 weights). (A first version required L * distance at 1e4 to be at
                most twice its value at L = 1e2 plus 0.05; it failed at 1.111, because at L = 1e2 the distance can be smaller than its
                limiting 1/L profile. Restated as convergence of L * distance.)
  SL5 IDENTITY  history: the maximum-entropy point moves under every trace outside H_D and under none inside (100 each); the toric limit
                ignores bounded weights but follows weights of size e^(L s), tending to the leximin point of the shifted cells P_k - s_k
  SL6 FINDING   decisions: on the column-binding fibre the functional P11 - P12 - 0.04 is positive at the maximum-entropy point and negative
                at the leximin point; the toric curve crosses zero at a finite L*; the fibre leaves the sign open
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from scipy.optimize import minimize_scalar, brentq
from pce import selectorlink as SL
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(41); B = SL.span_D(); f0, g0 = (0.4, 0.6), (0.5, 0.3, 0.2)
res = lambda x: float(np.linalg.norm(B @ np.linalg.lstsq(B, x, rcond=None)[0] - x))
def margins(): return rng.dirichlet([3, 3]), rng.dirichlet([3, 3, 3])

# SL1
worst = 0.0
for _ in range(50):
    f, g = margins(); V, lam = SL.cells(f, g); worst = max(worst, res(np.log(SL.law(SL.maxent(f, g), f, g))))
    for L in (0.5, 5, 50):
        a = rng.uniform(0.5, 2, 6); u = SL.toric(f, g, L, a=a); w = a * np.exp(-L * (V @ u - lam)); w /= w.sum(); worst = max(worst, res(w))
check("SL1 IDENTITY maximum entropy: log P in H_D; toric: the term law in H_D", worst < 1e-9, f"largest residual {worst:.1e}")

# SL2
def closest(f, g):
    me = SL.maxent(f, g); ts = np.linspace(-3, 9, 121); dd = [np.linalg.norm(SL.toric(f, g, np.exp(t)) - me) for t in ts]; i = int(np.argmin(dd))
    r = minimize_scalar(lambda t: np.linalg.norm(SL.toric(f, g, np.exp(t)) - me), bounds=(ts[max(i - 1, 0)], ts[min(i + 1, 120)]), method="bounded",
                        options=dict(xatol=1e-12))
    return r.fun, float(np.exp(r.x)), float(np.linalg.norm(SL.lex(f, g) - me))
c0, L0, sep0 = closest(f0, g0); ratios = []
for _ in range(60):
    f, g = margins(); c, _, sep = closest(f, g); ratios.append(c / sep)
med = float(np.median(ratios))
check("SL2 FINDING the toric curve passes near the maximum-entropy point but not through it", c0 > 1e-4 and 0.005 < med < 0.2,
      f"column-binding fibre: closest approach {c0:.5f} at L = {L0:.2f} (separation {sep0:.4f}); 60 margins: median ratio {med:.3f}, smallest {min(ratios):.1e}")

# SL3
ls = SL.leastsq(f0, g0); d = [np.linalg.norm(SL.toric(f0, g0, L) - ls) for L in (1e-1, 1e-2, 1e-3)]
fac = [0, 1, 2, 4, 5]; V, lam = SL.cells(f0, g0); w0 = SL.softmin(V[fac], np.zeros(5), 1.0)
uL = SL.toric(f0, g0, 1e-3, idx=fac); esc = np.linalg.norm(1e-3 * uL - w0)
ok = d[2] < 1e-3 and d[0] > d[1] > d[2] and esc < 1e-2 and np.linalg.norm(uL) > 100 and np.linalg.norm(w0) > 0.1
check("SL3 IDENTITY small L: all cells give the least-squares point; facet terms with unbalanced normals escape as w_0 / L", ok,
      f"least-squares point {np.round(ls, 4)}; distances at L = 0.1, 0.01, 0.001: {d[0]:.1e}, {d[1]:.1e}, {d[2]:.1e}; facet case: w_0 = {np.round(w0, 4)}, |L u_L - w_0| = {esc:.1e}, |u_L| = {np.linalg.norm(uL):.0f}")

# SL4
lx = SL.lex(f0, g0); ok = True; worst = 0.0
for _ in range(50):
    a = rng.uniform(0.5, 2, 6); ds = {L: np.linalg.norm(SL.toric(f0, g0, L, a=a) - lx) for L in (1e3, 1e4)}
    rel = abs(1e4 * ds[1e4] - 1e3 * ds[1e3]) / (1e4 * ds[1e4]); ok &= ds[1e4] < 1e-3 and rel < 0.01; worst = max(worst, rel)
check("SL4 FINDING bounded weights: the toric point tends to the leximin point at rate 1/L (L * distance converges)", ok, f"largest relative change of L * distance from 1e3 to 1e4: {worst:.1e}")

# SL5
inside, outside = [], []
me0 = SL.maxent(f0, g0)
for _ in range(100):
    h = B @ rng.normal(0, 1, B.shape[1]); inside.append(np.linalg.norm(SL.maxent(f0, g0, tau=h) - me0))
    t = rng.normal(0, 1, 6); outside.append(np.linalg.norm(SL.maxent(f0, g0, tau=t) - me0) if res(t) > 1e-3 else 1.0)
shift_ok = True; moved = 0
for _ in range(30):
    s = rng.uniform(-0.03, 0.03, 6); u = SL.softmin(V, lam + s, 1e4); target = SL.lex(f0, g0, shift=s)
    shift_ok &= np.linalg.norm(u - target) < 1e-3; moved += np.linalg.norm(target - lx) > 0.01
ok = max(inside) < 1e-9 and min(outside) > 1e-6 and shift_ok and moved > 0
check("SL5 IDENTITY maximum entropy follows every trace class; the toric limit ignores bounded weights and follows weights of size e^(Ls)", ok,
      f"inside H_D: max move {max(inside):.1e}; outside: min move {min(outside):.1e}; shifted weights: limit equals shifted leximin in 30 cases, {moved} of them away from the unshifted point")

# SL6
gfun = lambda u: u[0] - u[1] - 0.04
Vx = np.array([[0.2, 0], [0.4, 0], [0.1, 0.3], [0, 0.3], [0, 0.2]]); rng_f = [gfun(v) for v in Vx]
Lstar = float(np.exp(brentq(lambda t: gfun(SL.toric(f0, g0, np.exp(t))), 0, 9)))
ok = gfun(me0) > 0 > gfun(lx) and min(rng_f) < 0 < max(rng_f)
check("SL6 FINDING maximum entropy and leximin choose different actions; the toric curve switches at a finite L*", ok,
      f"maximum entropy {gfun(me0):+.3f}; leximin {gfun(lx):+.3f}; L* = {Lstar:.2f}; fibre range [{min(rng_f):+.2f}, {max(rng_f):+.2f}]")
json.dump(dict(closest=c0, L_closest=L0, separation=sep0, median_ratio=med, w0=w0.tolist(), leastsq=ls.tolist(), Lstar=Lstar),
          open(os.path.join(HERE, "selectorlink_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
