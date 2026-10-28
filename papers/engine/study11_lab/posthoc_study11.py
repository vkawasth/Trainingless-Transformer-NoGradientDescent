"""Post-hoc analyses for Study 11 (not pre-registered). PH3 uses exact bisection for robustness (the module's 101-point grid starts at
1 - lam*, so its resolution changes with lam* and can create spurious decreases of one grid step). Writes posthoc_study11.json.

PH1 (Q2 failed): the degree of P(all four) along the segment, per condition. In A, v_3 = 0 kills the r^2 term (coefficient v_0123), so
    the four-variable event is affine. The degree bound of Theorem A is an upper bound that drops on the stratum.
PH2 (L7 failed narrowly): within-truth Spearman of the graded VOI against |v_S| (registered) and against the information ratio
    |v_S| / sigma_S, where sigma_S^2 = E[((x - mu)_S)^2] sets the noise of the triple moment.
PH4 (L6 held with g = 0): with acts treat (u = 1 - c[E]) and nothing (0), E[u_treat] is affine in r on the segment (Theorem A), so
    every act's worst case lies at the same wall: a saddle point, and Wald's gap g is 0 for every cost. Savage's minimax regret has no
    saddle: at cost 1.235 the best act changes along G's segment, and the regret VOI of a pinning probe is the pre-probe minimax regret.
PH3 (L8 failed): robustness is monotone in the cover except for a jump to vacuity. For the drug family the script checks
    h1 >= h0 or h1 = -inf at every pi. It then shows a strongly conflicting BC window (harms 0.95, 0.95, independent) whose addition
    makes every act vacuous: negative VOI, by vacuity only.
"""
import itertools, json, os, sys
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import run_study11 as R
from pce.lab import law, moments, fit_v, r_of_t
OUT = {}
# PH1
ph1 = {}
for c in ("G", "A", "D"):
    P, mu, M, mS, v, vt = R.exact(c); ts, p4 = R.seg_curve(mu, v, R.ALL4); rs = r_of_t(ts)
    res = lambda y, d: float(np.max(np.abs(np.polyval(np.polyfit(rs, y, d), rs) - y)))
    ph1[c] = {"v_0123": float(np.prod(v)), "linear_residual": res(p4, 1), "quadratic_residual": res(p4, 2)}
OUT["PH1"] = ph1
# PH2
rng = np.random.default_rng(13 + 1000); rho_v, rho_info = [], []
for _ in range(300):
    w, a1, a2 = R.random_truth(rng); P = law(w, a1, a2); mu, M, _ = moments(P); v = fit_v(mu, M)
    vt = np.sqrt(w * (1 - w)) * (a1 - a2); v = v if np.dot(v, vt) >= 0 else -v
    ts, ps = R.seg_curve(mu, v, R.E3); rs = r_of_t(ts); before = max(0.0, R.u_treat(ps.max())); rtrue = r_of_t(np.sqrt((1 - w) / w)); row = []
    for S in R.TRIPLES:
        vS = abs(np.prod(v[list(S)])); ex2 = sum(P[x] * np.prod([(x[i] - mu[i]) for i in S]) ** 2 for x in itertools.product((0, 1), repeat=4))
        se = np.sqrt(ex2 / 1000) / vS; ins = (rs >= rtrue - 2 * se) & (rs <= rtrue + 2 * se)
        after = max(0.0, R.u_treat(ps[ins].max())) if ins.any() else max(0.0, R.u_treat(np.interp(rtrue, rs, ps)))
        row.append((vS, vS / np.sqrt(ex2), after - before))
    a, b, c_ = zip(*row)
    if max(c_) > 1e-12 and len(set(np.round(c_, 12))) > 1:
        r1 = spearmanr(a, c_).correlation; r2 = spearmanr(b, c_).correlation
        if np.isfinite(r1): rho_v.append(r1)
        if np.isfinite(r2): rho_info.append(r2)
OUT["PH2"] = {"mean_spearman_vS": float(np.mean(rho_v)), "mean_spearman_information_ratio": float(np.mean(rho_info)), "truths": len(rho_v),
              "note": "different random truths from the registered L7 (fresh seed); compare the two columns, not with L7"}
# PH3
from timesheaf_example import POP1, A_TYPES, ACTS, U0
from pce.level2 import lcm_law
from pce import Window
from pce.timesheaf import System, lam, robust_value
import numpy as _np
def robustness(S, u, u0, cover=None):
    """bisection (exact to 1e-10): V is non-decreasing in alpha"""
    l = lam(S, cover) - 1e-9; v = robust_value(S, u, l, cover)
    if v is None or v < u0 - 1e-12: return -_np.inf
    if robust_value(S, u, 0.0, cover) >= u0 - 1e-12: return 1.0
    lo, hi = 0.0, l
    for _ in range(40):
        m = 0.5 * (lo + hi); vm = robust_value(S, u, m, cover)
        if vm is not None and vm >= u0 - 1e-12: hi = m
        else: lo = m
    return float(1 - hi)
D2 = {"A": 2, "B": 2, "C": 2}; AB1 = POP1.sum(axis=2); AC1 = POP1.sum(axis=1)
S0 = System(D2, [Window(("A", "B"), AB1), Window(("A", "C"), AC1)]); h0 = {a: robustness(S0, u, U0) for a, u in ACTS.items()}
mono = True; rows = []
for pi in np.linspace(0, 1, 21):
    P2 = lcm_law(np.array([pi, 1 - pi]), A_TYPES); S1 = System(D2, [Window(("A", "B"), AB1), Window(("A", "C"), AC1), Window(("B", "C"), P2.sum(axis=0))])
    h1 = {a: robustness(S1, u, U0) for a, u in ACTS.items()}
    for a in ACTS:
        ok = (h1[a] >= h0[a] - 1e-9) or (h1[a] == -np.inf); mono &= ok
    rows.append({"pi": float(pi), "lam": float(lam(S1)), "h1": {a: (None if not np.isfinite(h) else float(h)) for a, h in h1.items()}})
BCx = np.outer([0.05, 0.95], [0.05, 0.95]); S2 = System(D2, [Window(("A", "B"), AB1), Window(("A", "C"), AC1), Window(("B", "C"), BCx)])
h2 = {a: robustness(S2, u, U0) for a, u in ACTS.items()}
OUT["PH3"] = {"h0": {a: (None if not np.isfinite(h) else float(h)) for a, h in h0.items()}, "monotone_except_vacuity": bool(mono), "family": rows,
              "strong_conflict_lam": float(lam(S2)), "strong_conflict_h": {a: (None if not np.isfinite(h) else float(h)) for a, h in h2.items()},
              "negative_voi_by_vacuity": bool(max(h0.values()) > -np.inf and all(h == -np.inf for h in h2.values()))}
# PH4
ph4 = {}
for c in ("G", "A", "M"):
    P, mu, M, mS, v, vt = R.exact(c); row = {}
    if c != "M":
        ts, ps = R.seg_curve(mu, v, R.E3)
        for cost in (1.2, 1.235):
            f = 1 - cost * ps                                                     # E[u_treat]; u_nothing = 0
            wald = float(np.min(np.maximum(0, f)) - max(0.0, float(f.min())))
            regret = float(min(np.max(np.maximum(0, -f)), np.max(np.maximum(0, f))))
            row[str(cost)] = {"wald_g": wald, "regret_voi_by_triple": [0.0 if abs(np.prod(v[list(S)])) < 1e-12 else regret for S in R.TRIPLES]}
    ph4[c] = row
OUT["PH4"] = ph4
json.dump(OUT, open(os.path.join(HERE, "posthoc_study11.json"), "w"), indent=1)
print(json.dumps({k: (v if k != "PH3" else {kk: vv for kk, vv in v.items() if kk != "family"}) for k, v in OUT.items()}, indent=1))
