"""check_decision_segment.py -- decisions on the hiding segment (lab paper, Theorem D). Two types, pair cover.

  DS1 expansion: E[u] along the segment equals the polynomial U(r) of pce.segment_decision (m = 3 and 4, random utilities); for a
      utility on three variables the slope is v_T times the top Walsh coefficient sum_x u(x) chi(x)
  DS2 existence by walls: the worst value over the segment is attained at a wall (three variables) or at a wall or the single interior
      critical point (four variables): dense-grid minimum = closed-form minimum
  DS3 degree: four-variable utilities give quadratics with leading coefficient v_T * sum_x u(x) chi(x); three-variable ones are affine
  DS4 info-gap horizon around a nominal rhat: for affine U it is margin / |beta|, or saturated (+inf) when the bad direction reaches a
      wall first; equals bisection
  DS5 ties and flat acts: two acts admissible on the whole segment both saturate the info-gap horizon (the nominal value breaks the
      tie); the info-gap act and the set-rule act differ on some decisions (info-gap prefers flatter acts)
  DS6 invariance: the set rule computed from the pair tables of five laws on one fibre gives one act; the info-gap rule with a nominal
      point chosen from the pairs is not invariant: midpoint in r and midpoint in t pick different acts on some decisions
  DS7 stability: along a path of truths the set rule changes only where an act's worst value crosses u0 or two worst values cross
  DS8 settling: the settling radius by bisection equals the closed form min_j g_j / gamma_j (affine utilities, interval inside the
      segment)
  DS9 tempo of rhat: the Monte Carlo sd of rhat at n = 20000 matches tau / sqrt(n) within 10%, and the plug-in tauhat is within 5% of
      tau on average (three random truths with tau/sqrt(n) <= 0.1, not those of Study 11b; the
      sign of r depends on which type is called type 1, so rhat is aligned with the truth's labelling)
Writes ../lab/decision_segment_results.json is not needed; prints PASS/FAIL.  Run: python check_decision_segment.py
"""
import itertools, os, sys, warnings
import numpy as np
from numpy.polynomial import polynomial as Pn
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce.lab import law, moments, fit_v, segment, explanation, r_of_t, t_of_r
from pce import segment_decision as SD

rng = np.random.default_rng(20261027); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

def truth(m):
    while True:
        w = rng.uniform(0.15, 0.85); a1 = rng.uniform(0.05, 0.95, m); a2 = rng.uniform(0.05, 0.95, m)
        if np.min(np.abs(a1 - a2)) > 0.15: return w, a1, a2

def fibre(w, a1, a2):
    P = law(w, a1, a2); mu, M, mS = moments(P); v = fit_v(mu, M); return P, mu, M, mS, v

def rand_u(k):
    tab = {x: rng.uniform(-1, 1) for x in itertools.product((0, 1), repeat=k)}; return lambda x: tab[tuple(x)]

def EU(P, u, T):
    m = P.ndim; return sum(P[x] * u(tuple(x[i] for i in T)) for x in itertools.product((0, 1), repeat=m))

# DS1, DS3
err1, err_slope, err3 = 0.0, 0.0, 0.0
for m in (3, 4):
    for _ in range(6):
        w, a1, a2 = truth(m); P, mu, M, mS, v = fibre(w, a1, a2); lo, hi = segment(mu, v); T = tuple(range(m)); u = rand_u(m)
        p = SD.U_poly(mu, v, u, T)
        for t in np.exp(np.linspace(np.log(lo), np.log(hi), 7)):
            err1 = max(err1, abs(Pn.polyval(r_of_t(t), p) - EU(law(*explanation(mu, v, t)), u, T)))
        if m == 3:
            err_slope = max(err_slope, abs((p[1] if len(p) > 1 else 0) - np.prod(v) * SD.walsh_top(u, 3))); err3 = max(err3, abs(p[2]) if len(p) > 2 else 0)
        else:
            err3 = max(err3, abs(p[2] - np.prod(v) * SD.walsh_top(u, 4)))
check("DS1 E[u] along the segment = U(r); three-variable slope = v_T x top Walsh coefficient", err1 < 1e-12 and err_slope < 1e-12, f"{err1:.1e}, {err_slope:.1e}")
check("DS3 degree: three variables affine, four variables quadratic with leading coefficient v_T x top Walsh", err3 < 1e-12, f"{err3:.1e}")

# DS2
err2 = 0.0; interior = 0
for m in (3, 4):
    for _ in range(20):
        w, a1, a2 = truth(m); P, mu, M, mS, v = fibre(w, a1, a2); R = SD.r_walls(mu, v); p = SD.U_poly(mu, v, rand_u(m), range(m))
        g = np.linspace(R[0], R[1], 20001); err2 = max(err2, abs(SD.worst(p, R) - Pn.polyval(g, p).min()))
        if m == 4 and np.argmin(Pn.polyval(g, p)) not in (0, len(g) - 1): interior += 1
check("DS2 worst value over the segment = walls (+ one interior critical point for four variables)", err2 < 1e-8, f"max err {err2:.1e}; {interior} interior minima seen")

# DS4
err4 = 0.0; sat = 0
for _ in range(200):
    R = tuple(sorted(rng.uniform(-3, 3, 2))); rhat = rng.uniform(*R); beta = rng.normal(); u0 = 0.0; U0 = abs(rng.normal())
    p = np.array([U0 - beta * rhat, beta]); h = SD.horizon(p, rhat, R, u0); dist = (rhat - R[0]) if beta > 0 else (R[1] - rhat)
    cf = np.inf if U0 / abs(beta) >= dist else U0 / abs(beta); sat += np.isinf(cf)
    err4 = max(err4, 0.0 if (np.isinf(h) and np.isinf(cf)) else abs(h - cf))
check("DS4 info-gap horizon = margin/|beta| or saturated", err4 < 1e-9, f"max err {err4:.1e}; {sat} saturated of 200")

# DS5
R = (-1.0, 1.0); A1 = np.array([0.5, 0.1]); A2 = np.array([0.4, 0.0])         # both admissible on R at u0 = 0.2
ht = [SD.horizon(p, 0.0, R, 0.2) for p in (A1, A2)]; nr = SD.nominal_rule([A1, A2], 0.0, R, 0.2)
differ = 0
for _ in range(400):
    ps = [np.array([rng.uniform(0, 1), rng.normal(0, 0.5)]) for _ in range(3)]; rhat = rng.uniform(-0.5, 0.5); u0 = 0.4
    a, b = SD.set_rule(ps, R, u0), SD.nominal_rule(ps, rhat, R, u0)
    if a is not None and b is not None and a != b: differ += 1
check("DS5 saturation tie broken by nominal value; info-gap and set rule can differ", np.all(np.isinf(ht)) and nr == 0 and differ > 0,
      f"horizons {ht}; picks act {nr}; {differ}/400 decisions differ")

# DS6
inv_ok = True; noninv = 0
for _ in range(10):
    w, a1, a2 = truth(3); P, mu, M, mS, v = fibre(w, a1, a2); lo, hi = segment(mu, v); acts = [rand_u(3) for _ in range(3)]
    picks = set()
    for t in np.exp(np.linspace(np.log(lo) + 0.05, np.log(hi) - 0.05, 5)):
        Pt = law(*explanation(mu, v, t)); mt, Mt, _ = moments(Pt); vt = fit_v(mt, Mt); Rt = SD.r_walls(mt, vt)
        ps = [SD.U_poly(mt, vt, u, range(3)) for u in acts]; u0 = float(np.median([SD.worst(p, Rt) for p in ps]))
        picks.add((SD.set_rule(ps, Rt, round(u0, 6)), round(u0, 6)))
    inv_ok &= len(picks) == 1
for _ in range(300):
    w, a1, a2 = truth(3); P, mu, M, mS, v = fibre(w, a1, a2); lo, hi = segment(mu, v); R = SD.r_walls(mu, v)
    ps = [SD.U_poly(mu, v, rand_u(3), range(3)) for _ in range(3)]; u0 = float(np.median([Pn.polyval(np.mean(R), p) for p in ps])) - 0.05
    if SD.nominal_rule(ps, 0.5 * (R[0] + R[1]), R, u0) != SD.nominal_rule(ps, r_of_t(0.5 * (lo + hi)), R, u0): noninv += 1
check("DS6 set rule invariant on the fibre; info-gap with a pair-chosen nominal is not (midpoint in r vs in t)", inv_ok and noninv > 0, f"{noninv}/300 decisions change with the parametrisation")

# DS7
changes, explained = 0, 0
for path in range(6):
  w0, a10, a20 = truth(3); w1, a11, a21 = truth(3); acts = [rand_u(3) for _ in range(3)]
  def at(s):
      w = (1 - s) * w0 + s * w1; a1 = (1 - s) * a10 + s * a11; a2 = (1 - s) * a20 + s * a21; P, mu, M, mS, v = fibre(w, a1, a2)
      R = SD.r_walls(mu, v); ps = [SD.U_poly(mu, v, u, range(3)) for u in acts]; return ps, R
  ps0, R0 = at(0.0); u0 = float(np.median([SD.worst(p, R0) for p in ps0]))
  ss = np.linspace(0, 1, 401); dec = [SD.set_rule(*at(s), u0) for s in ss]
  for k in range(len(ss) - 1):
    if dec[k] != dec[k + 1]:
        changes += 1; lo_, hi_ = ss[k], ss[k + 1]
        for _ in range(50):
            mid = 0.5 * (lo_ + hi_); ps, R = at(mid)
            if SD.set_rule(ps, R, u0) == dec[k]: lo_ = mid
            else: hi_ = mid
        ps, R = at(hi_); wv = [SD.worst(p, R) for p in ps]
        gaps = [abs(x - u0) for x in wv] + [abs(wv[i] - wv[j]) for i, j in itertools.combinations(range(3), 2)]
        explained += min(gaps) < 1e-6
check("DS7 the set rule changes only at a wall-value crossing of u0 or a crossing of worst values", changes > 0 and explained == changes, f"{explained}/{changes} change points explained")

# DS8
err8 = 0.0; used = 0
for _ in range(1500):
    w, a1, a2 = truth(3); P, mu, M, mS, v = fibre(w, a1, a2); R = SD.r_walls(mu, v); r0 = mS((0, 1, 2)) / np.prod(v)
    ps = [SD.U_poly(mu, v, rand_u(3), range(3)) for _ in range(3)]; u0 = float(np.median([Pn.polyval(r0, p) for p in ps])) - 0.01
    if SD.oracle(ps, r0, u0) is None: continue
    cf = SD.settle_radius_affine(ps, r0, u0)
    if not np.isfinite(cf) or cf >= min(r0 - R[0], R[1] - r0): continue
    used += 1; err8 = max(err8, abs(SD.settle_radius(ps, r0, R, u0) - cf))
check("DS8 settling radius: bisection = closed form min_j g_j / gamma_j", used >= 10 and err8 < 1e-8, f"{used} decisions, max err {err8:.1e}")

# DS9
rel_sd, rel_tau = [], []; n = 20000
while len(rel_sd) < 3:
    w, a1, a2 = truth(3); P = law(w, a1, a2); tau, r0, v, mu = SD.tau_r(P); rh, th = [], []
    if tau / np.sqrt(n) > 0.1: continue                    # asymptotic regime only
    for _ in range(2000):
        e = SD.estimate(P, n, rng)
        if e is not None: rh.append(e[2] * np.sign(np.dot(e[1], v))); th.append(e[3])   # the type labelling is a convention: align it
    rel_sd.append(np.std(rh) * np.sqrt(n) / tau - 1); rel_tau.append(np.mean(th) / tau - 1)
check("DS9 sd of rhat = tau/sqrt(n) (10%), plug-in tauhat unbiased (5%)", max(map(abs, rel_sd)) < 0.10 and max(map(abs, rel_tau)) < 0.05,
      f"sd ratio - 1: {np.round(rel_sd, 3)}; tauhat ratio - 1: {np.round(rel_tau, 3)}")

print(f"{sum(results)}/{len(results)} passed")
