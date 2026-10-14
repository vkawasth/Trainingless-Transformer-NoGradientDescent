"""explore_followup.py -- EXPLORATORY follow-up analyses (not pre-registered), prompted by Study 1 H2 and the first Study 2 samples.
F1  Study 2 (partial, 200 items): AUROC of the pre-registered detector, and of answer length alone (style confound check).
F2  Study 2 samples: window defect c among samples 1-3 (correctness by the kit's judge), with and without a difficulty
    centre D = number of samples 4-5 judged correct; Beck-Chevalley with the centre is Y1 _||_ Y3 | (Y2, D).
F3  Study 1 drift data: the H2 windows (4M,4J),(4J,35J) with the fourth run 35M added to the centre: Y4M _||_ Y35J | (Y4J, Y35M)?
F4  P3 power: probability that Spearman rho = 0.7 arises with n = 5 under no association.
"""
import json, sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kit_halueval import match, auroc
from scipy.stats import spearmanr
rng = np.random.default_rng(20261013); B = 2000
items = [json.loads(l) for l in open(os.environ.get("HALUEVAL_QA", "qa_data.json"))]
S = {json.loads(l)["id"]: json.loads(l)["samples"] for l in open("study2/samples.jsonl")}

def phi(t):
    r0, r1, c0, c1 = t[0].sum(), t[1].sum(), t[:, 0].sum(), t[:, 1].sum()
    if min(r0, r1, c0, c1) == 0: return None
    return (t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]) / np.sqrt(r0 * r1 * c0 * c1)
def c_strat(Y, a, m, b):
    """max over strata of the centre columns m (list) of |phi(Y_a, Y_b | centre)|; strata with degenerate margins skipped"""
    keys = {tuple(r[m]) for r in Y}; vals = []; used = 0
    for k in keys:
        sub = Y[np.all(Y[:, m] == k, axis=1)]; t = np.zeros((2, 2))
        for r in sub: t[r[a], r[b]] += 1
        f = phi(t)
        if f is not None: vals.append(abs(f)); used += len(sub)
    return (max(vals) if vals else np.nan), used
def markov_null(Y, a, m, b, c_obs):
    keys, idx = np.unique(Y[:, m], axis=0, return_inverse=True); idx = idx.ravel(); n = len(Y); null = []
    pa = {k: Y[idx == k, a].mean() for k in range(len(keys))}; pb = {k: Y[idx == k, b].mean() for k in range(len(keys))}
    for _ in range(B):
        ks = idx[rng.integers(0, n, n)]
        Z = np.zeros_like(Y); Z[:, m] = keys[ks]
        Z[:, a] = rng.random(n) < np.array([pa[k] for k in ks]); Z[:, b] = rng.random(n) < np.array([pb[k] for k in ks])
        v = c_strat(Z, a, m, b)[0]
        if not np.isnan(v): null.append(v)
    null = np.array(null); return float(np.median(null)), float((1 + (null >= c_obs).sum()) / (len(null) + 1))

out = {}
# F1
from kit_halueval import score_items
r = score_items(items, S); L, Y = [], []
for i in S:
    for ans, lab in ((items[i]["right_answer"], 0), (items[i]["hallucinated_answer"], 1)): L.append(len(ans)); Y.append(lab)
out["F1"] = dict(detector=r, length_only_auroc=float(auroc(np.array(L), np.array(Y))))
# F2
Yc = np.array([[int(match(items[i]["right_answer"], s)) for s in S[i]] for i in sorted(S)])
Yw = 1 - Yc                                   # 1 = sample wrong
D = Yw[:, 3] + Yw[:, 4]
Y = np.column_stack([Yw[:, 0], Yw[:, 1], Yw[:, 2], D])
c0, u0 = c_strat(Y, 0, [1], 2); med0, p0 = markov_null(Y, 0, [1], 2, c0)
c1, u1 = c_strat(Y, 0, [1, 3], 2); med1, p1 = markov_null(Y, 0, [1, 3], 2, c1)
out["F2"] = dict(n=len(Y), wrong_rate=float(Yw.mean()), c_without_centre=c0, null_median=med0, p=p0, used=u0,
                 c_with_difficulty_centre=c1, null_median_centre=med1, p_centre=p1, used_centre=u1)
# F3
from study1_llmdrift import load, TASKS
F3 = {}
for task in TASKS:
    A, R = load(task); rows = []
    for it in A.index:
        v = [A.at[it, k] for k in ("4M", "4J", "35J", "35M")]
        if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in v): continue
        rows.append([int(x != R[it]) for x in v])
    Y = np.array(rows)
    c0, _ = c_strat(Y, 0, [1], 2); med0, p0 = markov_null(Y, 0, [1], 2, c0)
    c1, used = c_strat(Y, 0, [1, 3], 2); med1, p1 = markov_null(Y, 0, [1, 3], 2, c1)
    F3[task] = dict(n=len(Y), c=c0, null=med0, p=p0, c_centre=c1, null_centre=med1, p_centre=p1, used_centre=used)
out["F3"] = F3
# F4
perm = [spearmanr(np.arange(5), rng.permutation(5)).statistic for _ in range(20000)]
out["F4"] = dict(p_two_sided_rho_ge_0_7=float(np.mean(np.abs(perm) >= 0.7 - 1e-9)))
json.dump(out, open("explore_followup.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))

# F5: weighted statistic and label centre
def cbar(Y, a, m, b):
    """mass-weighted conditional |phi|: sum over strata with non-degenerate margins of (n_k/n_used)|phi_k|"""
    keys = {tuple(r[m]) for r in Y}; tot = 0; acc = 0.0
    for k in keys:
        sub = Y[np.all(Y[:, m] == k, axis=1)]; t = np.zeros((2, 2))
        for r in sub: t[r[a], r[b]] += 1
        f = phi(t)
        if f is not None: acc += len(sub) * abs(f); tot += len(sub)
    return acc / tot if tot else np.nan
def perm_null(Y, a, m, b, stat):
    """exact conditional test: permute Y_b within strata of the centre (preserves both conditional margins)"""
    obs = stat(Y, a, m, b); keys, idx = np.unique(Y[:, m], axis=0, return_inverse=True); idx = idx.ravel(); null = []
    for _ in range(B):
        Z = Y.copy()
        for k in range(len(keys)):
            w = np.where(idx == k)[0]; Z[w, b] = Z[rng.permutation(w), b]
        v = stat(Z, a, m, b)
        if not np.isnan(v): null.append(v)
    null = np.array(null); return float(obs), float(np.median(null)), float((1 + (null >= obs).sum()) / (len(null) + 1))
F5 = {}
Y2 = np.column_stack([Yw[:, 0], Yw[:, 1], Yw[:, 2], D])
F5["samples_no_centre"] = perm_null(Y2, 0, [1], 2, cbar)
F5["samples_difficulty_centre"] = perm_null(Y2, 0, [1, 3], 2, cbar)
for task in TASKS:
    A, R = load(task); rows = []
    for it in A.index:
        v = [A.at[it, k] for k in ("4M", "4J", "35J", "35M")]
        if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in v): continue
        rows.append([int(x != R[it]) for x in v] + [hash(str(R[it])) % 1000003])
    Y = np.array(rows)
    F5[task] = {"shared_run": perm_null(Y, 0, [1], 2, cbar), "plus_35M": perm_null(Y, 0, [1, 3], 2, cbar)}
    if task == "PRIME_FULL":
        F5[task]["plus_label"] = perm_null(Y, 0, [1, 4], 2, cbar)
        F5[task]["label_counts"] = {str(k): int(v) for k, v in zip(*np.unique([str(R[i]) for i in A.index], return_counts=True))}
out["F5_weighted_cbar_(obs,null_median,p)"] = F5
json.dump(out, open("explore_followup.json", "w"), indent=1, default=float)
print(json.dumps(F5, indent=1, default=float))
