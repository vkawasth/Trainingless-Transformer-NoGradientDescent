"""synthetic.py -- demonstrations of the Posterior Coherence Engine on worlds with known truth (debugging and illustration only;
no parameter here was tuned on, or used to choose, the pre-registered real-data analysis).

  D1  averaging hides conflicts: queries whose three pairwise contexts come either from one global law (coherent) or from a
      hallucination-type source (pairwise agreement below 1/3). The averaged answer looks equally confident in both; the engine's
      consistency radius separates them. Reported as AUROC against sample size per context.
  D2  non-unique reconstruction: only the chain (X0,X1), (X1,X2) is observed. The engine reports the interval of P(X0 = X2) over all
      global laws; a single reconstruction (the Markov glue) gives a point. Reported: how often each contains the truth, and the error
      of the point.
  D3  where disagreement sits: items answered by four runs in two families. World A has family-specific blind spots; world B is a
      null with exchangeable runs. Selective prediction of the pooled answer's correctness with (v, H) versus (v, H, between).
Output: demos/synthetic_results.json and figs/fig_engine_demo.pdf
"""
import json, os, sys, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from pce import Window, coherence, markov_glue
from pce.votes import features
from pce.selective import auroc, cv_logistic, logloss

rng = np.random.default_rng(7)
out = {}
DOM = {"X0": 2, "X1": 2, "X2": 2}; PAIRS = [("X0", "X1"), ("X1", "X2"), ("X0", "X2")]

def sample_pair_table(P2, n):
    counts = rng.multinomial(n, P2.ravel()).reshape(2, 2); return counts / n

# D1
def compatible_family():
    """random compatible pair tables on a triangle of binary variables (Frechet bounds); about a quarter admit no global law."""
    m = rng.uniform(0.1, 0.9, 3); tabs = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        lo, hi = max(0, m[i] + m[j] - 1), min(m[i], m[j]); p11 = rng.uniform(lo, hi)
        tabs.append(np.array([[1 - m[i] - m[j] + p11, m[j] - p11], [m[i] - p11, p11]]))
    return tabs
d1 = {}
for n in (25, 100, 400, 1600):
    rad, conf, lab = [], [], []
    for _ in range(200):
        tabs = compatible_family()
        truth_incoh = coherence(DOM, [Window(p, t) for p, t in zip(PAIRS, tabs)]).status == "incoherent"
        emp = [sample_pair_table(t, n) for t in tabs]
        r = coherence(DOM, [Window(p, e) for p, e in zip(PAIRS, emp)])
        rad.append(r.radius); lab.append(int(truth_incoh))
        conf.append(np.mean([abs(e.sum(1)[1] - 0.5) for e in emp]))      # confidence of the averaged per-claim answers
    d1[n] = {"incoherent_share": float(np.mean(lab)), "auroc_radius": auroc(rad, lab), "auroc_averaged_confidence": auroc(conf, lab)}
out["D1_incoherence_detection"] = d1

# D2
cover, point_err, contains_pt = 0, [], 0
for _ in range(200):
    P = rng.dirichlet(np.ones(8)).reshape(2, 2, 2)
    truth = P[0, :, 0].sum() + P[1, :, 1].sum()
    r = coherence(DOM, [Window(("X0", "X1"), P.sum(2)), Window(("X1", "X2"), P.sum(0))])
    # interval of P(X0 = X2): solve directly by LP over the fibre
    from scipy.optimize import linprog
    atoms = list(itertools.product((0, 1), repeat=3))
    A = [[1.0 if (a[0], a[1]) == c else 0.0 for a in atoms] for c in itertools.product((0, 1), repeat=2)] + \
        [[1.0 if (a[1], a[2]) == c else 0.0 for a in atoms] for c in itertools.product((0, 1), repeat=2)]
    b = list(P.sum(2).ravel()) + list(P.sum(0).ravel())
    f = np.array([1.0 if a[0] == a[2] else 0.0 for a in atoms])
    lo = linprog(f, A_eq=A, b_eq=b, bounds=(0, None), method="highs").fun
    hi = -linprog(-f, A_eq=A, b_eq=b, bounds=(0, None), method="highs").fun
    Q = markov_glue(P.transpose(0, 1, 2)); pt = Q[0, :, 0].sum() + Q[1, :, 1].sum()
    cover += lo - 1e-9 <= truth <= hi + 1e-9; contains_pt += abs(pt - truth) < 1e-9; point_err.append(abs(pt - truth))
out["D2_nonunique"] = {"interval_contains_truth": cover / 200, "point_equals_truth": contains_pt / 200,
                       "point_abs_error_median": float(np.median(point_err)), "point_abs_error_max": float(np.max(point_err))}

# D3
FAM = {"F": ["F1", "F2"], "G": ["G1", "G2"]}
def world(blind_spots, n_items=1500, K=4):
    rows = []
    for _ in range(n_items):
        truth = 0
        if blind_spots:                                   # each family is either reliable or shares one wrong answer on this item
            ans = {}
            for fam, runs in FAM.items():
                blind = rng.random() < 0.3; wrong = rng.integers(1, K)
                for r in runs:
                    ans[r] = (wrong if rng.random() < 0.9 else truth) if blind else (truth if rng.random() < 0.85 else rng.integers(1, K))
        else:                                             # exchangeable runs with item difficulty, family labels meaningless
            d = rng.uniform(0.3, 0.95); wrong = rng.integers(1, K)
            ans = {r: (truth if rng.random() < d else (wrong if rng.random() < 0.6 else rng.integers(1, K))) for r in ["F1", "F2", "G1", "G2"]}
        f = features({k: int(v) for k, v in ans.items()}, FAM, tiebreak="F1")
        rows.append((f["v"], f["total"], f["between"], int(f["answer"] == truth)))
    R = np.array(rows); y = R[:, 3].astype(int)
    pb = cv_logistic(R[:, :2], y); pe = cv_logistic(R[:, :3], y)
    lb, le = logloss(pb, y), logloss(pe, y); d = lb - le
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    return {"accuracy": float(y.mean()), "auroc_baseline": auroc(pb, y), "auroc_engine": auroc(pe, y),
            "logloss_gain_bits": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
out["D3_world_A_blind_spots"] = world(True)
out["D3_world_B_null"] = world(False)

json.dump(out, open(os.path.join(HERE, "synthetic_results.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
