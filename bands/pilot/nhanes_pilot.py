"""NHANES pilot: heterogeneous subgroups as contexts, probability bands, information ladder, walls, revisions.

Data: NHANESraw (NHANES 2009-10 and 2011-12, as distributed in the CRAN package NHANES), adults 20+,
MEC exam weights (WTMEC2YR / 2 when cycles are pooled). Attributes: A1 = age 55+, A2 = diabetes,
A3 = overweight (BMI >= 25), A4 = male. Outcome Y = high systolic blood pressure (average >= 140 mmHg).
Sampling uncertainty: delete-one-PSU stratified jackknife (JKn) on the design variables SDMVSTRA, SDMVPSU.
"""
import itertools, json, sys
import numpy as np, pandas as pd, pyreadr
from scipy.optimize import linprog

df = list(pyreadr.read_r("data/nhanes_pkg/data/NHANESraw.rda").values())[0]
d = df[(df.Age >= 20)].dropna(subset=["BMI", "BPSysAve", "Diabetes", "WTMEC2YR"]).copy()
d["A1"] = (d.Age >= 55).astype(int); d["A2"] = (d.Diabetes == "Yes").astype(int)
d["A3"] = (d.BMI >= 25).astype(int); d["A4"] = (d.Gender == "male").astype(int)
d["Y"] = (d.BPSysAve >= 140).astype(int)
d["cyc"] = d.SurveyYr.astype(str)
d["psu"] = d.cyc + "_" + d.SDMVSTRA.astype(int).astype(str) + "_" + d.SDMVPSU.astype(int).astype(str)
d["str"] = d.cyc + "_" + d.SDMVSTRA.astype(int).astype(str)
VARS = ["Y", "A1", "A2", "A3", "A4"]; NAMES = {"A1": "55+", "A2": "diabetic", "A3": "overweight", "A4": "male"}
CELLS = list(itertools.product([0, 1], repeat=5))           # (Y, A1, A2, A3, A4)
out = {}

def joint(sub, w):
    """Weighted joint distribution over the 32 cells."""
    key = sub[VARS].values @ (2 ** np.arange(4, -1, -1))
    p = np.bincount(key, weights=w, minlength=32); return p / p.sum()

def jk_replicates(sub):
    """Delete-one-PSU jackknife replicate weights (JKn); yields (weights, factor)."""
    w = sub.WTMEC2YR.values.astype(float)
    for s, g in sub.groupby("str"):
        psus = g.psu.unique(); n = len(psus)
        for u in psus:
            wr = w.copy(); ins = (sub.str == s).values; drop = (sub.psu == u).values
            wr[drop] = 0; wr[ins & ~drop] *= n / (n - 1)
            yield wr, (n - 1) / n

def risk_by_pattern(p):
    p = p.reshape(2, 16); return p[1] / p.sum(0), p.sum(0)    # P(Y=1 | pattern), P(pattern)

def est_with_se(sub):
    p = joint(sub, sub.WTMEC2YR.values)
    r, m = risk_by_pattern(p)
    reps = [(risk_by_pattern(joint(sub, wr))[0], f) for wr, f in jk_replicates(sub)]
    var = sum(f * (rr - r) ** 2 for rr, f in reps)
    return p, r, m, np.sqrt(var)

# ---------------------------------------------------------------- 1. pooled estimates and sampling bands
p_all, r_all, m_all, se_all = est_with_se(d)
pat = list(itertools.product([0, 1], repeat=4))
def lab(a): return ", ".join(NAMES[f"A{i+1}"] if a[i] else "not " + NAMES[f"A{i+1}"] for i in range(4))
n_cell = d.groupby(["A1", "A2", "A3", "A4"]).size().reindex(pat, fill_value=0).values
out["n_adults"] = int(len(d)); out["prevalence_Y"] = float((d.Y * d.WTMEC2YR).sum() / d.WTMEC2YR.sum())

# ---------------------------------------------------------------- 2. information ladder: LP bands for intersectional risk
def margin_constraints(level):
    """Rows: indicators of every cell of every marginal table on `level` of the 5 variables."""
    rows = []
    for S in itertools.combinations(range(5), level):
        for vals in itertools.product([0, 1], repeat=level):
            rows.append([1.0 if all(c[i] == v for i, v in zip(S, vals)) else 0.0 for c in CELLS])
    return np.array(rows)

ATTR_TABLE = np.array([[1.0 if c[1:] == a else 0.0 for c in CELLS] for a in itertools.product([0, 1], repeat=4)])
def ladder_rows(level):
    """Subgroup sizes (the full attribute table) are always known; level k adds every table of Y with k attributes."""
    rows = [ATTR_TABLE]
    for S in itertools.combinations(range(1, 5), level):
        for vals in itertools.product([0, 1], repeat=level + 1):
            rows.append(np.array([[1.0 if (c[0] == vals[0] and all(c[i] == v for i, v in zip(S, vals[1:]))) else 0.0 for c in CELLS]]))
    return np.vstack(rows)

def risk_band(p, a, level):
    """Bounds on P(Y=1 | pattern a) over all joints sharing p's marginal tables at this level (Charnes-Cooper LP)."""
    M = ladder_rows(level); b = M @ p
    i1 = CELLS.index((1,) + a); i0 = CELLS.index((0,) + a)
    # variables y (32) and t; y = t x; constraints M y - t b = 0, sum y - t = 0, y_i0 + y_i1 = 1
    A_eq = np.vstack([np.hstack([M, -b[:, None]]), np.hstack([np.ones(32), [-1.0]]),
                      np.hstack([np.eye(32)[i0] + np.eye(32)[i1], [0.0]])])
    b_eq = np.concatenate([np.zeros(len(b) + 1), [1.0]])
    c = np.zeros(33); c[i1] = 1
    lo = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * 33, method="highs")
    hi = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * 33, method="highs")
    if lo.status != 0 or hi.status != 0: return (0.0, 1.0)   # pattern can have mass 0 under these margins
    return (lo.fun, -hi.fun)

ladder = {}
for a in pat:
    ladder[a] = {L: risk_band(p_all, a, L) for L in (0, 1, 2, 3)}
target = (1, 1, 1, 1)
out["target"] = {"cell": lab(target), "n": int(n_cell[pat.index(target)]), "risk": float(r_all[pat.index(target)]),
                 "se": float(se_all[pat.index(target)]),
                 "bands": {L: [round(x, 4) for x in ladder[target][L]] for L in (0, 1, 2, 3)}}
out["ladder_width_mean"] = {L: float(np.mean([ladder[a][L][1] - ladder[a][L][0] for a in pat])) for L in (0, 1, 2, 3)}
out["ladder_contains_truth"] = all(ladder[a][L][0] - 1e-9 <= r_all[i] <= ladder[a][L][1] + 1e-9 for i, a in enumerate(pat) for L in (0, 1, 2, 3))
# Frechet band for the intersection mass from one-way margins of the attributes only
pa = [float(p_all.reshape(2, 2, 2, 2, 2).sum(axis=tuple(j for j in range(5) if j != i))[1]) for i in range(1, 5)]
out["frechet_mass_target"] = [max(0.0, sum(pa) - 3), min(pa)]; out["true_mass_target"] = float(m_all[pat.index(target)])

# ---------------------------------------------------------------- 3. walls: distance from each cell's band to a risk threshold
THR = 0.30
def wilson(r, se, n_raw=None, z=1.96):
    """Wilson interval with the design-based effective sample size n_eff = r(1-r)/se^2. When the cell has no events
    (or all events) the jackknife SE is 0 and n_eff is undefined; fall back to the raw count (a boundary case)."""
    if se <= 0 or r in (0.0, 1.0):
        if not n_raw: return (r, r)
        n = float(n_raw)
    else:
        n = max(r * (1 - r) / se ** 2, 1.0)
    den = 1 + z * z / n; c = (r + z * z / (2 * n)) / den
    h = z * np.sqrt(r * (1 - r) / n + z * z / (4 * n * n)) / den; return (c - h, c + h)
rows = []
for i, a in enumerate(pat):
    lo, hi = wilson(r_all[i], se_all[i], n_cell[i])
    side = "above" if lo > THR else ("below" if hi < THR else "straddles")
    rows.append(dict(cell=lab(a), n=int(n_cell[i]), risk=r_all[i], lo=lo, hi=hi, side=side,
                     margin=(lo - THR) if side == "above" else ((THR - hi) if side == "below" else -min(THR - lo, hi - THR)),
                     L0=ladder[a][0], L1=ladder[a][1], L2=ladder[a][2], L3=ladder[a][3]))
cells = pd.DataFrame(rows).sort_values("risk", ascending=False)
cells.to_csv("out/nhanes_cells.csv", index=False)
out["threshold"] = THR; out["sides"] = cells.side.value_counts().to_dict()

# ---------------------------------------------------------------- 4. revision: 2009-10 band vs 2011-12 estimate
c1, c2 = d[d.cyc == "2009_10"], d[d.cyc == "2011_12"]
_, r1, _, s1 = est_with_se(c1); _, r2, _, s2 = est_with_se(c2)
n1 = c1.groupby(["A1", "A2", "A3", "A4"]).size().reindex(pat, fill_value=0).values
n2 = c2.groupby(["A1", "A2", "A3", "A4"]).size().reindex(pat, fill_value=0).values
rev = []
for i, a in enumerate(pat):
    w1, w2 = wilson(r1[i], s1[i], n1[i]), wilson(r2[i], s2[i], n2[i])
    side1 = "above" if w1[0] > THR else ("below" if w1[1] < THR else "straddles")
    side2 = "above" if w2[0] > THR else ("below" if w2[1] < THR else "straddles")
    rev.append(dict(cell=lab(a), n_0910=int(n1[i]), n_1112=int(n2[i]), r_0910=r1[i], se_0910=s1[i], r_1112=r2[i], se_1112=s2[i], side_0910=side1, side_1112=side2,
                    inside_old_band=w1[0] <= r2[i] <= w1[1], lo_0910=w1[0], hi_0910=w1[1], lo_1112=w2[0], hi_1112=w2[1], width_change=s2[i] / s1[i] if s1[i] > 0 else np.nan))
rev = pd.DataFrame(rev); rev.to_csv("out/nhanes_revision.csv", index=False)
out["revision"] = {"cells_new_estimate_inside_old_95band": int(rev.inside_old_band.sum()), "cells": len(rev),
                   "side_changes": rev[rev.side_0910 != rev.side_1112][["cell", "side_0910", "side_1112", "r_0910", "r_1112"]].round(3).to_dict("records"),
                   "bands_widened": int((rev.width_change > 1).sum())}

# ---------------------------------------------------------------- 5. inconsistent sources: 2-way tables from different cycles
pj1, pj2 = joint(c1, c1.WTMEC2YR.values), joint(c2, c2.WTMEC2YR.values)
pairs = list(itertools.combinations(range(5), 2))
def inconsistency(blocks):
    """Least L1 change of the stated tables that makes them margins of one joint (LP); blocks = [(rows, target)]."""
    M = np.vstack([B for B, _ in blocks]); b = np.concatenate([t for _, t in blocks]); k = len(b)
    # variables x (32), u (k), v (k): M x - u + v = b, sum x = 1; minimise sum(u+v)
    A_eq = np.vstack([np.hstack([M, -np.eye(k), np.eye(k)]), np.hstack([np.ones(32), np.zeros(2 * k)])])
    res = linprog(np.concatenate([np.zeros(32), np.ones(2 * k)]), A_eq=A_eq, b_eq=np.concatenate([b, [1.0]]),
                  bounds=[(0, None)] * (32 + 2 * k), method="highs")
    return res.fun
def pair_rows(S):
    return np.array([[1.0 if all(c[i] == v for i, v in zip(S, vals)) else 0.0 for c in CELLS] for vals in itertools.product([0, 1], repeat=2)])
mixed = [(pair_rows(S), pair_rows(S) @ (pj1 if k % 2 == 0 else pj2)) for k, S in enumerate(pairs)]
same1 = [(pair_rows(S), pair_rows(S) @ pj1) for S in pairs]
# sampling scale: same inconsistency computed between two halves of the PSUs within one cycle
rng = np.random.default_rng(0)
half_vals = []
for rep in range(20):
    psus = c1.psu.unique(); h = set(rng.choice(psus, len(psus) // 2, replace=False))
    m1 = c1.psu.isin(h).values; q1 = joint(c1[m1], c1.WTMEC2YR.values[m1]); q2 = joint(c1[~m1], c1.WTMEC2YR.values[~m1])
    half_vals.append(inconsistency([(pair_rows(S), pair_rows(S) @ (q1 if k % 2 == 0 else q2)) for k, S in enumerate(pairs)]))
out["inconsistency"] = {"same_cycle": inconsistency(same1), "mixed_cycles": inconsistency(mixed),
                        "split_half_within_2009_10_median": float(np.median(half_vals))}

# ---------------------------------------------------------------- 6. Simpson reversals: exposure X, stratifier Z
def wrisk(sub, cond):
    s = sub[cond]; return (s.Y * s.WTMEC2YR).sum() / s.WTMEC2YR.sum()
simp = []
for X in ["A1", "A2", "A3", "A4"]:
    for Z in ["A1", "A2", "A3", "A4"]:
        if X == Z: continue
        rd = wrisk(d, d[X] == 1) - wrisk(d, d[X] == 0)
        rdz = [wrisk(d, (d[X] == 1) & (d[Z] == z)) - wrisk(d, (d[X] == 0) & (d[Z] == z)) for z in (0, 1)]
        simp.append(dict(exposure=NAMES[X], stratifier=NAMES[Z], rd_marginal=rd, rd_Z0=rdz[0], rd_Z1=rdz[1],
                         reversal=bool(np.sign(rd) != np.sign(rdz[0]) and np.sign(rd) != np.sign(rdz[1])),
                         partial=bool((np.sign(rd) != np.sign(rdz[0])) != (np.sign(rd) != np.sign(rdz[1])))))
def rd_se(X, Z, z):
    sub = d if Z is None else d[d[Z] == z]; base = None; var = 0.0
    def rdw(w):
        y, x = sub.Y.values, sub[X].values
        return (w * y * x).sum() / (w * x).sum() - (w * y * (1 - x)).sum() / (w * (1 - x)).sum()
    base = rdw(sub.WTMEC2YR.values.astype(float))
    for wr, f in jk_replicates(sub): var += f * (rdw(wr) - base) ** 2
    return np.sqrt(var)
for row in simp:
    X = [k for k, v in NAMES.items() if v == row["exposure"]][0]; Z = [k for k, v in NAMES.items() if v == row["stratifier"]][0]
    row["se_marginal"] = rd_se(X, None, None); row["se_Z0"] = rd_se(X, Z, 0); row["se_Z1"] = rd_se(X, Z, 1)
simp = pd.DataFrame(simp); simp.to_csv("out/nhanes_simpson.csv", index=False)
out["simpson"] = {"full_reversals": simp[simp.reversal][["exposure", "stratifier", "rd_marginal", "rd_Z0", "rd_Z1"]].round(3).to_dict("records"),
                  "sign_change_in_one_stratum": simp[simp.partial][["exposure", "stratifier", "rd_marginal", "rd_Z0", "se_Z0", "rd_Z1", "se_Z1"]].round(3).to_dict("records")}
# ---------------------------------------------------------------- 7. out-of-sample ladder: bands from 2009-10 tables, risks from 2011-12
pj1_ = joint(c1, c1.WTMEC2YR.values)
oos = {}
for L in (0, 1, 2, 3):
    inside = [risk_band(pj1_, a, L)[0] - 1e-9 <= r2[i] <= risk_band(pj1_, a, L)[1] + 1e-9 for i, a in enumerate(pat)]
    oos[L] = int(sum(inside))
# the same with each band widened by the 2011-12 sampling interval of the risk (structural + sampling)
oos_wide = {}
for L in (0, 1, 2, 3):
    k = 0
    for i, a in enumerate(pat):
        lo_, hi_ = risk_band(pj1_, a, L); w2 = wilson(r2[i], s2[i], n2[i])
        k += (w2[1] >= lo_ - 1e-9) and (w2[0] <= hi_ + 1e-9)       # the new estimate's interval meets the old band
    oos_wide[L] = int(k)
out["out_of_sample_ladder"] = {"inside": oos, "interval_meets_band": oos_wide, "cells": 16}
# expected number of 2011-12 estimates inside the 2009-10 95% bands if nothing changed, using the actual SEs
from scipy.stats import norm
exp_inside = sum(2 * norm.cdf(1.96 * s1[i] / np.sqrt(s1[i] ** 2 + s2[i] ** 2)) - 1 for i in range(16) if s1[i] > 0 and s2[i] > 0)
out["revision"]["expected_inside_if_no_change_cells_with_se"] = float(exp_inside)
out["revision"]["cells_with_positive_se_both"] = int(sum(1 for i in range(16) if s1[i] > 0 and s2[i] > 0))
out["revision"]["crossed_wall"] = int(sum(1 for _, r in rev.iterrows() if {r.side_0910, r.side_1112} == {"above", "below"}))
out["target_sampling_band"] = [float(x) for x in wilson(r_all[pat.index(target)], se_all[pat.index(target)], n_cell[pat.index(target)])]
out["target_revision"] = [float(r1[pat.index(target)]), float(r2[pat.index(target)])]
json.dump(out, open("out/nhanes_summary.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4)))
print(cells.round(3).to_string(index=False))
