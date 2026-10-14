"""study2_analysis.py -- Study 2 analyses exactly as specified in AMENDMENT1.md (frozen, SHA-256 in FROZEN_AMENDMENT1.txt).
Primary set: items 200-1999 (confirmatory). Secondary: all 2000. Output: study2/study2_amendment1_results.json
"""
import json, re, numpy as np
import statsmodels.api as sm
from scipy.stats import chi2
from kit_halueval import match as match_contain, auroc

SEED, B = 20261013, 2000
STOP = set("the a an of in on at and or to is was were are by for with from as".split())
items = [json.loads(l) for l in open("/home/claude/halueval_data/qa_data.json")]
S = {d["id"]: d["samples"] for d in map(json.loads, open("study2/samples_2000.jsonl")) if d}

def toks(s): return {w for w in re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split() if w not in STOP}
def match_relaxed(a, s):
    A, T = toks(a), toks(s)
    if not A or not T: return False
    return A <= T or T <= A or len(A & T) / len(A | T) >= 0.5

def build(ids, match):
    sc, ln, y, it = [], [], [], []
    for i in ids:
        for ans, lab in ((items[i]["right_answer"], 0), (items[i]["hallucinated_answer"], 1)):
            sc.append(sum(not match(ans, s) for s in S[i]) / len(S[i])); ln.append(len(ans)); y.append(lab); it.append(i)
    return np.array(sc), np.array(ln, float), np.array(y), np.array(it)

def banded_auroc(sc, ln, y, cuts):
    q = np.digitize(ln, cuts); num = den = 0.0
    for b in np.unique(q):
        p, n = sc[(q == b) & (y == 1)], sc[(q == b) & (y == 0)]
        if len(p) and len(n):
            num += (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan, den

def boot(fn, it, rng):
    u = np.unique(it); pos = {k: np.where(it == k)[0] for k in u}; out = []
    for _ in range(B):
        idx = np.concatenate([pos[k] for k in rng.choice(u, len(u))]); v = fn(idx)
        if not np.isnan(v): out.append(v)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def analyse(ids, match, rng):
    sc, ln, y, it = build(ids, match)
    cuts = np.quantile(ln, [0.2, 0.4, 0.6, 0.8])
    r = {"items": len(ids)}
    r["detector_auroc"] = float(auroc(sc, y)); r["detector_ci"] = boot(lambda ix: auroc(sc[ix], y[ix]), it, rng)
    r["length_auroc"] = float(auroc(ln, y))
    ba, den = banded_auroc(sc, ln, y, cuts)
    r["length_adjusted_auroc"] = float(ba); r["within_band_pairs"] = int(den)
    r["length_adjusted_ci"] = boot(lambda ix: banded_auroc(sc[ix], ln[ix], y[ix], cuts)[0], it, rng)
    X0 = sm.add_constant(np.log(ln)); X1 = np.column_stack([X0, sc])
    m0 = sm.Logit(y, X0).fit(disp=0, maxiter=200); m1 = sm.Logit(y, X1).fit(disp=0, maxiter=200)
    lr = 2 * (m1.llf - m0.llf)
    r["logit"] = dict(coef_loglen=float(m1.params[1]), coef_detector=float(m1.params[2]), LR=float(lr),
                      p=float(chi2.sf(lr, 1)), llf_len_only=float(m0.llf), llf_full=float(m1.llf))
    # cluster-robust check (two answers per item are not independent): Wald z with item clusters
    m1c = sm.Logit(y, X1).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": it})
    r["logit"]["cluster_wald_p"] = float(m1c.pvalues[2])
    r["band_counts"] = {f"Q{b+1}": [int(((np.digitize(ln, cuts) == b) & (y == 0)).sum()), int(((np.digitize(ln, cuts) == b) & (y == 1)).sum())]
                        for b in range(5)}
    return r

def phi(t):
    r0, r1, c0, c1 = t[0].sum(), t[1].sum(), t[:, 0].sum(), t[:, 1].sum()
    if min(r0, r1, c0, c1) == 0: return None
    return (t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]) / np.sqrt(r0 * r1 * c0 * c1)
def cbar(Y, a, m, b):
    tot, acc = 0, 0.0
    for k in {tuple(r[m]) for r in Y}:
        sub = Y[np.all(Y[:, m] == k, axis=1)]; t = np.zeros((2, 2))
        for r in sub: t[r[a], r[b]] += 1
        f = phi(t)
        if f is not None: acc += len(sub) * abs(f); tot += len(sub)
    return acc / tot if tot else np.nan
def perm_null(Y, a, m, b, rng):
    obs = cbar(Y, a, m, b); keys, idx = np.unique(Y[:, m], axis=0, return_inverse=True); idx = idx.ravel(); null = []
    for _ in range(B):
        Z = Y.copy()
        for k in range(len(keys)):
            w = np.where(idx == k)[0]; Z[w, b] = Z[rng.permutation(w), b]
        v = cbar(Z, a, m, b)
        if not np.isnan(v): null.append(v)
    null = np.array(null)
    return dict(cbar=float(obs), null_median=float(np.median(null)), p=float((1 + (null >= obs).sum()) / (len(null) + 1)))

def windows(ids, rng):
    Yw = np.array([[int(not match_relaxed(items[i]["right_answer"], s)) for s in S[i]] for i in ids])
    Y = np.column_stack([Yw[:, 0], Yw[:, 1], Yw[:, 2], Yw[:, 3] + Yw[:, 4]])
    return dict(wrong_rate=float(Yw.mean()), per_sample_wrong=[float(x) for x in Yw.mean(0)],
                centre_Y2=perm_null(Y, 0, [1], 2, rng), centre_Y2_D=perm_null(Y, 0, [1, 3], 2, rng))

if __name__ == "__main__":
    out = {}
    for name, ids in (("confirmatory_200_1999", list(range(200, 2000))), ("all_2000", list(range(2000)))):
        rng = np.random.default_rng(SEED)
        out[name] = {"containment": analyse(ids, match_contain, rng), "relaxed": analyse(ids, match_relaxed, rng),
                     "windows_relaxed": windows(ids, rng)}
        c = out[name]["containment"]["logit"]["p"]
        out[name]["H1pp_detector_beyond_length"] = bool(c < 0.05)
        out[name]["H2p_window_BC_fails"] = bool(out[name]["windows_relaxed"]["centre_Y2"]["p"] < 0.05)
    json.dump(out, open("study2/study2_amendment1_results.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
