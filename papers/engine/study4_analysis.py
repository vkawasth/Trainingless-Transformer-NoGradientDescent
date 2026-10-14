"""study4_analysis.py -- Study 4 exactly as frozen in PREREG_engine.md (SHA-256 in FROZEN_PREREG_engine.txt).
Data: LLM-drift <TASK>_EVAL.csv files (env LLMDRIFT_DIR); extraction rules from halluc_study/llmdrift_extract.py via study1_llmdrift.load.
Output: study4_results.json, study4_output.txt (stdout).
"""
import io, os, sys, json, hashlib, contextlib
import numpy as np
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "halluc_study"))
from pce.votes import features
from pce.selective import auroc, aurc, cv_logistic, logloss
with contextlib.redirect_stdout(io.StringIO()):
    import study1_llmdrift as s1

SEED, B = 20261014, 2000
FAM = {"GPT-4": ["4M", "4J"], "GPT-3.5": ["35M", "35J"]}
prereg = open(os.path.join(HERE, "PREREG_engine.md"), "rb").read()
frozen = open(os.path.join(HERE, "FROZEN_PREREG_engine.txt")).read().split()[0]
assert hashlib.sha256(prereg).hexdigest() == frozen, "pre-registration changed after freezing"

rng = np.random.default_rng(SEED)
per_task, all_d, all_task = {}, [], []
rows_all = []
for task in s1.TASKS:
    A, R = s1.load(task)
    keep = A[["4M", "4J", "35M", "35J"]].notna().all(axis=1) & R.notna()
    F, Y, patt = [], [], []
    for it in A.index[keep]:
        ans = {k: str(A.at[it, k]) for k in ["4M", "4J", "35M", "35J"]}
        f = features(ans, FAM, tiebreak="4J")
        F.append([f["v"], f["total"], f["between"]]); Y.append(int(f["answer"] == str(R[it])))
        vals = list(ans.values())
        two_two = len(set(vals)) == 2 and sorted([vals.count(x) for x in set(vals)]) == [2, 2]
        along = two_two and ans["4M"] == ans["4J"] and ans["35M"] == ans["35J"]
        patt.append("2-2 along families" if along else ("2-2 across families" if two_two else "other"))
    F, Y = np.array(F), np.array(Y)
    pb = cv_logistic(F[:, :2], Y, folds=5, seed=SEED); pe = cv_logistic(F, Y, folds=5, seed=SEED)
    lb, le = logloss(pb, Y), logloss(pe, Y); d = lb - le
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    coef = LogisticRegression(C=1.0, max_iter=1000).fit(F, Y).coef_[0].tolist() if len(set(Y)) > 1 else [float("nan")] * 3
    pat = {k: {"n": int(sum(1 for x in patt if x == k)),
               "accuracy": float(np.mean([y for y, x in zip(Y, patt) if x == k])) if any(x == k for x in patt) else None}
           for k in ("2-2 along families", "2-2 across families", "other")}
    per_task[task] = dict(n=int(len(Y)), accuracy=float(Y.mean()), delta=float(d.mean()),
                          ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                          p_one_sided=float((1 + (bs <= 0).sum()) / (B + 1)),
                          auroc_baseline=auroc(pb, Y), auroc_engine=auroc(pe, Y), aurc_baseline=aurc(pb, Y), aurc_engine=aurc(pe, Y),
                          exploratory_coef_v_H_B=coef, exploratory_patterns=pat)
    all_d.append(d); all_task.append(np.full(len(d), task)); rows_all.append((pb, pe, Y))

# primary: pooled Delta with bootstrap stratified by task
D = np.concatenate(all_d); tasks = np.concatenate(all_task)
idx = {t: np.where(tasks == t)[0] for t in s1.TASKS}
bs = np.array([np.concatenate([D[rng.choice(ix, len(ix))] for ix in idx.values()]).mean() for _ in range(B)])
lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
verdict = "supported" if lo > 0 else ("worse" if hi < 0 else "no improvement")
# Holm across tasks (one-sided Delta > 0)
order = sorted(s1.TASKS, key=lambda t: per_task[t]["p_one_sided"]); m = len(order); running = 0.0
for k, t in enumerate(order):
    running = max(running, min(1.0, (m - k) * per_task[t]["p_one_sided"])); per_task[t]["p_holm"] = running
PB = np.concatenate([r[0] for r in rows_all]); PE = np.concatenate([r[1] for r in rows_all]); YY = np.concatenate([r[2] for r in rows_all])
out = dict(prereg_sha256=frozen, n_items=int(len(D)), primary=dict(delta_bits=float(D.mean()), ci=[lo, hi], H4=verdict),
           pooled_auroc=dict(baseline=auroc(PB, YY), engine=auroc(PE, YY)), pooled_aurc=dict(baseline=aurc(PB, YY), engine=aurc(PE, YY)),
           per_task=per_task)
json.dump(out, open(os.path.join(HERE, "study4_results.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
