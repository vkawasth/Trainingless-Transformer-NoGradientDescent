"""EXPLORATORY (after the run, not pre-registered; does not change any verdict). Question: is the R0 advantage on E2 (H1 failed) information
that the trace lacks, or a limit of the frozen R1 feature map? Same histories (seed 27002, same generator), same learner and CV.
R1+: the frozen R1 features plus counts of covering pairs (e, f) of the dependence order; these are
functions of the trace (invariant under swaps of independent events)."""
import os, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_study27b.py")).read()
cut = src.index("models = {}")
os.environ["SMOKE27B"] = "0"; __file__ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_study27b.py"); exec(src[:cut])
def f_R1p(w):
    """counts of the covering pairs (e, f) of the dependence order (Hasse diagram of the trace): trace invariants"""
    n = len(w); below = [set() for _ in range(n)]
    for k in range(n):
        for j in range(k):
            if O.dependent(w[j], w[k]): below[k] |= below[j] | {j}
    x = np.zeros(196)
    for k in range(n):
        for j in below[k]:
            if not any(j in below[i] for i in below[k]): x[tok(*w[j]) * 14 + tok(*w[k])] += 1
    return np.r_[f_R1(w), x]
# invariance check of the new features under independent swaps
rng2 = np.random.default_rng(1); inv = all(np.array_equal(f_R1p(w), f_R1p(t_comm(w, rng2))) for w in Wte[:300])
Xtr = np.array([f_R1p(w) for w in Wtr]); Xte = np.array([f_R1p(w) for w in Wte])
gs = GridSearchCV(LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000), {"C": [0.01, 0.1, 1, 10]},
                  cv=StratifiedKFold(5, shuffle=True, random_state=27002), scoring="neg_log_loss").fit(Xtr, y2tr)
e = (gs.best_estimator_.predict(Xte) != y2te).mean()
print(f"trace-invariant extra features: {inv}; E2 error R1+ = {e:.4f} (frozen: R0 0.1400, R1 0.1943)")
