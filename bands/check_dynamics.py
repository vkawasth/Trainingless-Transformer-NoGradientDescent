"""check_dynamics.py -- the families, shards, stability paths, second-order breaking and tower dynamics of "Bands", draft 2.

  D1  the family over banded margins P_B = {alpha : margins in F x G} is a polytope; its upper entropy is
      Hup(F) + Hup(G), attained at the product of the two water-fillings
  D2  regime effects of updates: conditioning breaks (zeroes the complement), a Jeffrey update with positive targets keeps
      the support, discounting heals (full support, min >= (1-alpha)/n), a kernel with no zero column keeps full support
  D3  compatibility is lost only through breaking: along paths of pairwise margins (three bits) from compatible to
      incompatible, the largest minimum cell over compatible joints tends to 0 at the transition
  D4  tolerance: if a compatible joint has minimum cell delta, every change of the margins (within their affine span) of
      sup-norm below delta / kappa, kappa the sup-norm of the pseudo-inverse, stays compatible
  D5  a 2 x 2 path f_1 in (0,1), g = (0.3, 0.7): the Floer-nontrivial point (Frechet midpoint) is piecewise linear with kinks
      exactly at the walls 0.3 and 0.7; the product coupling is smooth; they coincide in the middle chamber only at f_1 = 0.5;
      the segment length tends to 0 at the boundary
  D6  the three-bit path: shards (0,1/3) incompatible, 1/3 boundary, (1/3,1) positive, 1 boundary; the best minimum cell is
      min((3 eps - 1)/4, (1 - eps)/4), attained by the filler, linear in the distance to either end
  D7  no cascade through the middle margin: gluing stays continuous as a middle probability g_y -> 0, whatever happens to
      the undefined conditional row K_y
  D8  breaking shrinks the gluing algebra: dim (n-1)^2+1 -> (n-2)^2+1; the centre stays 2-dimensional except when the last
      circulation goes (n = 2 -> 1), where the Morita type changes from R x R to R
  D9  the tower under evidence: an outcome impossible under one model removes it for ever (Cromwell); discounting each model
      with reliability alpha keeps every weight positive, with likelihood ratios at most n/(1-alpha) per observation; the
      decomposition H(predictive) = E H + I holds at every step
"""
import itertools, math
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.linalg import null_space
R = np.random.default_rng(11); res = []
def check(n, c, info=""): res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))
def H(p): p = np.asarray(p, float).ravel(); p = p[p > 0]; return float(-(p * np.log(p)).sum())
def tighten(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.maximum(lo, 1 - (hi.sum() - hi)), np.minimum(hi, 1 - (lo.sum() - lo))
def waterfill(lo, hi):
    a, b = 0.0, 1.0
    for _ in range(200):
        c = (a + b) / 2
        if np.clip(c, lo, hi).sum() > 1: b = c
        else: a = c
    return np.clip((a + b) / 2, lo, hi)
def random_band(n):
    p = R.dirichlet(np.ones(n) * 2); w = R.uniform(0.02, 0.3, n)
    return tighten(np.clip(p - w * R.random(n), 0, 1), np.clip(p + w * R.random(n), 0, 1))

# ---------------------------------------------------------------- D1
ok = True
for _ in range(15):
    (lf, hf), (lg, hg) = random_band(2), random_band(3)
    pf, pg = waterfill(lf, hf), waterfill(lg, hg); target = H(pf) + H(pg)
    # maximise H(alpha) over alpha in R^{2x3}, row sums in [lf, hf], column sums in [lg, hg]
    cons = [{"type": "eq", "fun": lambda a: a.sum() - 1}]
    for i in range(2):
        cons += [{"type": "ineq", "fun": lambda a, i=i: a.reshape(2, 3)[i].sum() - lf[i]}, {"type": "ineq", "fun": lambda a, i=i: hf[i] - a.reshape(2, 3)[i].sum()}]
    for j in range(3):
        cons += [{"type": "ineq", "fun": lambda a, j=j: a.reshape(2, 3)[:, j].sum() - lg[j]}, {"type": "ineq", "fun": lambda a, j=j: hg[j] - a.reshape(2, 3)[:, j].sum()}]
    best = -1e9
    for _ in range(4):
        x0 = np.outer(R.dirichlet(np.ones(2)), R.dirichlet(np.ones(3))).ravel()
        r = minimize(lambda a: -H(np.clip(a, 1e-15, 1)), x0, bounds=[(1e-12, 1)] * 6, constraints=cons, method="SLSQP", options={"maxiter": 500})
        if r.success: best = max(best, -r.fun)
    ok &= best <= target + 1e-6 and best >= target - 1e-4
check("D1 upper entropy of the coupling family over banded margins = Hup(F) + Hup(G), at the product of the water-fillings", ok)

# ---------------------------------------------------------------- D2 regime effects of updates
ok = True
for _ in range(100):
    n = 5; p = R.dirichlet(np.ones(n)); E = R.random(n) < 0.5
    if not E.any() or E.all(): continue
    pc = p * E / (p * E).sum(); ok &= np.all(pc[~E] == 0) and np.all(pc[E] > 0)              # conditioning breaks
    parts = [np.arange(n) < 2, np.arange(n) >= 2]; q = R.dirichlet(np.ones(2))
    pj = sum(q[i] * p * parts[i] / (p * parts[i]).sum() for i in range(2)); ok &= np.all(pj > 0)  # Jeffrey keeps support
    p0 = p.copy(); p0[0] = 0; p0 /= p0.sum(); a = R.uniform(0.1, 0.95)
    pd = a * p0 + (1 - a) / n; ok &= pd.min() >= (1 - a) / n - 1e-15                             # discounting heals
    K = R.dirichlet(np.ones(n), size=n); ok &= np.all(K.T @ p0 > 0)                              # no zero column: full support
    K2 = K.copy(); K2[:, 1] = 0; K2 /= K2.sum(1, keepdims=True); ok &= (K2.T @ p0)[1] == 0      # a zero column breaks
    pj0 = sum(qq * p * parts[i] / (p * parts[i]).sum() for i, qq in enumerate([1.0, 0.0])); ok &= np.all(pj0[parts[1]] == 0)  # zero target breaks
    Kpos = 0.1 * np.ones((n, n)) / n + 0.9 * np.eye(n); ok &= np.all(Kpos.T @ p0 > 0)       # positive kernel heals
    lo_band = a * p0; ok &= lo_band[0] == 0                                               # Shafer band keeps the zero in its lower envelope
check("D2 conditioning breaks; Jeffrey with positive targets keeps the support, with a zero target breaks; linear discounting "
      "heals to min >= (1-alpha)/n while the Shafer band keeps the zero lower bound; a positive kernel heals; a kernel breaks a "
      "full-support law only through a zero column", ok)

# ---------------------------------------------------------------- D3, D4, D6 three bits
cells = list(itertools.product([0, 1], repeat=3)); PAIRS = [(0, 1), (1, 2), (0, 2)]
def rows(S): return np.array([[1.0 if (c[S[0]], c[S[1]]) == v else 0.0 for c in cells] for v in itertools.product([0, 1], repeat=2)])
M = np.vstack([rows(S) for S in PAIRS])
def maxmin(b):
    """largest minimum cell over joints with margins b (None if incompatible)"""
    c = np.r_[np.zeros(8), -1.0]; A_eq = np.hstack([np.vstack([M, np.ones(8)]), np.zeros((13, 1))])
    A_ub = np.hstack([-np.eye(8), np.ones((8, 1))])
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(8), A_eq=A_eq, b_eq=np.r_[b, 1], bounds=[(0, None)] * 8 + [(None, None)], method="highs")
    return None if r.status != 0 else -r.fun
def tables(eps): return np.tile(np.array([eps / 2, (1 - eps) / 2, (1 - eps) / 2, eps / 2]), 3)
ok = True; trans = []
for _ in range(12):
    P0 = R.dirichlet(np.ones(8) * 3); b0 = M @ P0; b1 = tables(R.uniform(0, 0.25))     # compatible start, incompatible end
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if maxmin((1 - mid) * b0 + mid * b1) is None: hi = mid
        else: lo = mid
    v_at = maxmin((1 - lo) * b0 + lo * b1); v_before = maxmin((1 - lo + 0.05) * b0 + (lo - 0.05) * b1) if lo > 0.05 else None
    ok &= v_at is not None and abs(v_at) < 1e-6 and (v_before is None or v_before > 1e-4); trans.append(round(lo, 3))
check("D3 along paths from compatible to incompatible pairwise margins, the best minimum cell is 0 at the transition and "
      "positive before it", ok, f"transition points {trans[:6]}")
# D4 tolerance
ok = True; info = ""
Mpinv = np.linalg.pinv(M)
span = np.linalg.matrix_rank(M)
for _ in range(10):
    P0 = R.dirichlet(np.ones(8) * 3); delta = P0.min(); b0 = M @ P0
    kappa = np.abs(Mpinv).sum(axis=1).max()                      # ||M^+||_{inf -> inf}
    for _ in range(40):
        v = R.normal(size=8); v -= v.mean()                     # total mass unchanged: every table still sums to 1
        d = M @ v; d = d / np.abs(d).max() * 0.99 * delta / kappa       # a change of the margins within their span
        ok &= maxmin(b0 + d) is not None and maxmin(b0 + d) > 0
    info = f"delta {delta:.4f}, kappa {kappa:.3f}"
# tightness on the three-bit path: delta = (3 eps - 1)/4 and kappa = 3/2 give (3 eps - 1)/6, the sup-norm distance of the tables to eps = 1/3
tight = abs(kappa - 1.5) < 1e-9 and all(abs(((3 * e - 1) / 4) / kappa - (e - 1 / 3) / 2) < 1e-12 for e in (0.35, 0.4, 0.45))
check("D4 every change of the margins within their span of sup-norm below delta/kappa keeps them compatible, with a positive joint; "
      "kappa = 3/2 for pairwise tables of three bits, and the bound is attained on the three-bit path", ok and tight, info)
ok = all(abs(min((3 * e - 1) / 4, (1 - e) / 4) - maxmin(tables(e))) < 1e-9 for e in np.linspace(0.34, 0.99, 27)) and maxmin(tables(0.3)) is None \
    and abs(maxmin(tables(1 / 3))) < 1e-9 and abs(maxmin(tables(1.0))) < 1e-9
check("D6 three-bit shards: incompatible below 1/3, boundary at 1/3 and 1, positive between; the best minimum cell equals the "
      "filler's (3 eps - 1)/4 for eps <= 1/2 and vanishes at both ends", ok)

# ---------------------------------------------------------------- D5 the 2 x 2 path
g1 = 0.3
mid = lambda f1: (max(0.0, f1 + g1 - 1) + min(f1, g1)) / 2
prod = lambda f1: f1 * g1
e = 1e-6
slope = lambda fn, x, s: (fn(x + s * e) - fn(x)) / (s * e)
kinks = [round(slope(mid, w, 1) - slope(mid, w, -1), 6) for w in (0.3, 0.7)]
smooth_else = all(abs(slope(mid, x, 1) - slope(mid, x, -1)) < 1e-6 for x in (0.1, 0.5, 0.85))
prod_smooth = all(abs(slope(prod, x, 1) - slope(prod, x, -1)) < 1e-6 for x in (0.3, 0.7))
xs = np.linspace(0.3001, 0.6999, 40001); coincide = xs[np.abs(np.array([prod(x) - mid(x) for x in xs])) < 1e-5]
length = lambda f1: min(f1, g1) - max(0.0, f1 + g1 - 1)
check("D5 2x2 path: the Floer-nontrivial point kinks exactly at the walls 0.3 and 0.7, the product is smooth there, they meet in the "
      "middle chamber only at f1 = 0.5, and the moment segment shrinks to a point at the boundary",
      all(abs(k) > 0.4 for k in kinks) and smooth_else and prod_smooth and len(coincide) > 0 and np.allclose(coincide, 0.5, atol=1e-3)
      and length(1e-9) < 1e-8 and length(1 - 1e-9) < 1e-8, f"slope jumps {kinks}")

# ---------------------------------------------------------------- D7 continuity of gluing as a middle probability vanishes
ok = True
f = np.array([0.5, 0.5]); h_rows = R.dirichlet(np.ones(3), size=3)
for s in (1e-1, 1e-2, 1e-3, 1e-4):
    g = np.array([s, 0.4 - s / 2, 0.6 - s / 2])
    alpha = np.outer(f, g)                                       # any coupling of f and g; its column y=0 has mass s
    K1, K2 = h_rows.copy(), h_rows.copy(); K2[0] = R.dirichlet(np.ones(3))   # two different (arbitrary) rows for the vanishing state
    diff = np.abs(alpha @ K1 - alpha @ K2).sum()
    ok &= diff <= 2 * s + 1e-15
check("D7 gluing stays continuous as a middle probability vanishes: changing the undefined row K_y moves the result by at most 2 g_y", ok)

# ---------------------------------------------------------------- D8 breaking shrinks the gluing algebra
def algebra_dims(fv):
    n = len(fv); D = np.diag(fv); B = [np.outer(fv, fv).ravel()]
    for i, j in itertools.product(range(n - 1), repeat=2):
        xm = np.zeros((n, n)); xm[i, j] = 1; xm[i, n - 1] = -1; xm[n - 1, j] = -1; xm[n - 1, n - 1] = 1; B.append(xm.ravel())
    B = np.array(B).T; dim = B.shape[1]; Di = np.linalg.inv(D)
    mult = lambda a, b: a @ Di @ b
    Mc = np.vstack([np.column_stack([(mult(B[:, c].reshape(n, n), B[:, k].reshape(n, n)) - mult(B[:, k].reshape(n, n), B[:, c].reshape(n, n))).ravel()
                                     for c in range(dim)]) for k in range(dim)])
    sv = np.linalg.svd(Mc, compute_uv=False)
    return dim, dim - int((sv > 1e-10).sum())                     # absolute tolerance: Mc may vanish identically
d3, d2, d1 = algebra_dims(np.array([0.2, 0.3, 0.5])), algebra_dims(np.array([0.4, 0.6])), algebra_dims(np.array([1.0]))
check("D8 breaking one outcome: dimension 5 -> 2 (n = 3 -> 2) with centre 2 -> 2; dimension 2 -> 1 (n = 2 -> 1) with centre 2 -> 1",
      d3 == (5, 2) and d2 == (2, 2) and d1 == (1, 1), f"(dim, centre): n=3 {d3}, n=2 {d2}, n=1 {d1}")

# ---------------------------------------------------------------- D9 the tower under evidence
rho = np.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]); w0 = np.array([0.5, 0.5]); obs = [0, 1, 0, 1, 2, 0, 1, 0]
def run(rh):
    w = w0.copy(); traj = []
    for x in obs:
        w = w * rh[:, x]; w = w / w.sum()
        bar = w @ rh; J = w[:, None] * rh; I = sum(J[k, y] * math.log(J[k, y] / (w[k] * bar[y])) for k in range(2) for y in range(3) if J[k, y] > 0)
        traj.append((w.copy(), H(bar), sum(w[k] * H(rh[k]) for k in range(2)), I))
    return traj
plain = run(rho); a = 0.9; disc = run(a * rho + (1 - a) / 3)
ident = all(abs(t[1] - (t[2] + t[3])) < 1e-12 for t in plain + disc)
cromwell = plain[4][0][0] == 0 and all(t[0][0] == 0 for t in plain[4:]) and plain[4][3] == 0
prot = all(t[0][0] > 0 for t in disc)
rd = a * rho + (1 - a) / 3; ratio = (rd.max(axis=1)[:, None] / rd.min(axis=1)[None, :]).max()
ratio_ok = abs((a + (1 - a) / 3) / ((1 - a) / 3) - (1 + a * 3 / (1 - a))) < 1e-12
quoted9 = ([round(t[0][0], 3) for t in plain] == [0.714, 0.806, 0.912, 0.946, 0.0, 0.0, 0.0, 0.0]
           and [round(t[0][0], 3) for t in disc] == [0.694, 0.783, 0.891, 0.929, 0.473, 0.671, 0.764, 0.88]
           and [round(t[3], 3) for t in plain[:5]] == [0.214, 0.186, 0.12, 0.088, 0.0]
           and [round(t[3], 3) for t in disc] == [0.146, 0.125, 0.08, 0.057, 0.152, 0.15, 0.13, 0.085])
check("D9 Cromwell: observing an outcome impossible under model 1 removes it for ever and sets the epistemic term to 0; with "
      "reliability 0.9 every weight stays positive and per-observation likelihood ratios are at most n/(1-alpha) = 30; "
      "H = E H + I at every step; the trajectories quoted in the paper", ident and cromwell and prot and ratio <= 1 + a * 3 / (1 - a) + 1e-12 and ratio_ok and quoted9,
      f"weight of model 1 after the impossible outcome: plain {plain[4][0][0]:.3f}, discounted {disc[4][0][0]:.3f}; "
      f"epistemic term before/after: plain {plain[3][3]:.3f}/{plain[4][3]:.3f}, discounted {disc[3][3]:.3f}/{disc[4][3]:.3f}")
print(f"\n{sum(res)}/{len(res)} checks passed")
