"""study1_llmdrift.py -- Study 1 of PREREG_hallucination.md (frozen; SHA-256 in FROZEN_PREREG.txt).
P1: AUROC of the cross-run disagreement score for target runs 4J (primary) and 35J (secondary).
P2: window Beck-Chevalley defect c for windows (4M,4J) and (4J,35J), with a parametric Markov-null bootstrap and Holm adjustment.
P3: exploratory Spearman correlation between c and the 4J AUROC across tasks.
Extraction rules are in llmdrift_extract.py (unchanged from the validation paper). Data folder: env LLMDRIFT_DIR."""
import os, sys, json, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llmdrift_extract import letter, yesno, boxed, norm_txt, D, MODELS

TASKS = ["USMLE", "PRIME_FULL", "COUNTAHAPPYNUMBER", "HOTPOTQA", "ARC"]
SEED, B = 20261013, 2000

def extract(task, s):
    if task == "USMLE": return letter(s)
    if task == "PRIME_FULL": return yesno(s)
    if task == "COUNTAHAPPYNUMBER": return boxed(s)
    if task == "HOTPOTQA":
        a = str(s)
        if a.startswith("Could not parse") or a.lower().startswith("agent stopped"): return None
        return norm_txt(a)
    if task == "ARC":
        a = str(s).strip(); return a.replace(" ", "") if a.startswith("[[") else None

def ref(task, s):
    if task == "USMLE": return letter(s)
    if task == "PRIME_FULL": return str(s).strip().lower()
    if task == "COUNTAHAPPYNUMBER": return boxed(s)
    if task == "HOTPOTQA": return norm_txt(s)
    if task == "ARC": return str(s).replace(" ", "")

def load(task):
    d = pd.read_csv(os.path.join(D, f"{task}_EVAL.csv"), low_memory=False)
    d["mshort"] = d.model.str.split("/").str[-1]
    if task == "USMLE":   # intact prompt = the longer one (llm_analysis.outcomes, variant='intact')
        d["qlen"] = d["query"].str.len(); d = d.sort_values(["mshort", "id", "qlen", "Unnamed: 0"])
        d = d.groupby(["mshort", "id"], as_index=False).nth(-1)
    d["item"] = d["id"]; assert not d.duplicated(["mshort", "item"]).any()
    d["ans"] = [extract(task, a) for a in d.answer]; d["ref"] = [ref(task, r) for r in d.ref_answer]
    inv = {v: k for k, v in MODELS.items()}
    A = d.pivot(index="item", columns="mshort", values="ans").rename(columns=inv)
    R = d.groupby("item")["ref"].first()
    return A, R

def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int); pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))

def p1(A, R, target):
    others = [r for r in ["35M", "35J", "4M", "4J"] if r != target]; S, Y = [], []
    for it in A.index:
        t = A.at[it, target]
        if t is None or (isinstance(t, float) and np.isnan(t)): continue
        vo = [A.at[it, o] for o in others if A.at[it, o] is not None and not (isinstance(A.at[it, o], float) and np.isnan(A.at[it, o]))]
        if not vo: continue
        S.append(sum(v != t for v in vo) / len(vo)); Y.append(int(t != R[it]))
    S, Y = np.array(S), np.array(Y); a = auroc(S, Y)
    rng = np.random.default_rng(SEED); bs = []
    for _ in range(B):
        ix = rng.integers(0, len(S), len(S)); bs.append(auroc(S[ix], Y[ix]))
    bs = np.array([x for x in bs if not np.isnan(x)])
    return dict(n=int(len(S)), wrong=int(Y.sum()), auroc=float(a), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))

def phi(t):  # t: 2x2 counts
    r0, r1 = t[0].sum(), t[1].sum(); c0, c1 = t[:, 0].sum(), t[:, 1].sum()
    if min(r0, r1, c0, c1) == 0: return None
    return (t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]) / np.sqrt(r0 * r1 * c0 * c1)

def c_stat(Y):  # Y: n x 3 array of (Y4M, Y4J, Y35J)
    vals = []
    for y in (0, 1):
        sub = Y[Y[:, 1] == y]
        t = np.zeros((2, 2))
        for a, b in zip(sub[:, 0], sub[:, 2]): t[a, b] += 1
        f = phi(t)
        if f is not None: vals.append(abs(f))
    return max(vals) if vals else np.nan

def p2(A, R):
    rows = []
    for it in A.index:
        v = [A.at[it, r] for r in ("4M", "4J", "35J")]
        if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in v): continue
        rows.append([int(x != R[it]) for x in v])
    Y = np.array(rows, int); n = len(Y); c = c_stat(Y)
    rng = np.random.default_rng(SEED)
    # Markov null fit
    pm = Y[:, 1].mean(); cond = {y: (Y[Y[:, 1] == y, 0].mean() if (Y[:, 1] == y).any() else 0.0,
                                  Y[Y[:, 1] == y, 2].mean() if (Y[:, 1] == y).any() else 0.0) for y in (0, 1)}
    null = []
    for _ in range(B):
        m = (rng.random(n) < pm).astype(int)
        a = np.array([rng.random() < cond[y][0] for y in m], int); b = np.array([rng.random() < cond[y][1] for y in m], int)
        null.append(c_stat(np.column_stack([a, m, b])))
    null = np.array([x for x in null if not np.isnan(x)])
    p = (1 + (null >= c).sum()) / (len(null) + 1) if not np.isnan(c) else np.nan
    bs = []
    for _ in range(B):
        ix = rng.integers(0, n, n); bs.append(c_stat(Y[ix]))
    bs = np.array([x for x in bs if not np.isnan(x)])
    counts = {f"{a}{m}{b}": int(((Y[:, 0] == a) & (Y[:, 1] == m) & (Y[:, 2] == b)).sum()) for a in (0, 1) for m in (0, 1) for b in (0, 1)}
    return dict(n=n, c=float(c), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p=float(p),
                null_median=float(np.median(null)), cells=counts)

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj

if __name__ == "__main__":
    out = {"prereg_sha256": open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "FROZEN_PREREG.txt")).read().split()[0]}
    for task in TASKS:
        A, R = load(task)
        out[task] = {"P1_4J": p1(A, R, "4J"), "P1_35J": p1(A, R, "35J"), "P2": p2(A, R)}
        print(task, json.dumps(out[task]))
    ps = np.array([out[t]["P2"]["p"] for t in TASKS]); adj = holm(ps)
    for t, a in zip(TASKS, adj): out[t]["P2"]["p_holm"] = float(a)
    h1 = sum(out[t]["P1_4J"]["lo"] > 0.5 for t in TASKS); h2 = sum(out[t]["P2"]["p_holm"] < 0.05 for t in TASKS)
    from scipy.stats import spearmanr
    rho = spearmanr([out[t]["P2"]["c"] for t in TASKS], [out[t]["P1_4J"]["auroc"] for t in TASKS]).statistic
    out["H1"] = {"tasks_with_CI_above_0.5": int(h1), "supported": bool(h1 >= 3)}
    out["H2"] = {"tasks_with_holm_p_below_0.05": int(h2), "supported": bool(h2 >= 1)}
    out["P3_spearman_c_vs_auroc"] = float(rho)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "study1_results.json"), "w"), indent=1)
    print("H1:", out["H1"], "H2:", out["H2"], "P3 rho:", round(rho, 3))
    for t in TASKS: print(f"  {t}: holm p = {out[t]['P2']['p_holm']:.4f}")
