"""Open problem 6 of the posterior-tower paper: do obstruction levels / certificates predict errors of claims?
Part A (real, level one): CausalIDView. Admissible set = image of the monotone-IV response-type model. A backbone's claim
  at a query unit is its estimated cells; d1 = minimal L1 distance to the admissible set (our LP slack; Thm 10.8 form).
  Known error = endpoint error of that unit's IV bounds against the oracle. Metric: AUC of d1 for the top error quartile.
Part B (synthetic, level two): finite hypothesis class H (6 laws on 4 outcomes), truth h* in H, n1 = 20 observations.
  Claims are second-order states x2 in D(D(B)) made by:
    bayes      the posterior over H                                   (admissible)
    collapsed  delta at the pooled barycenter                          (fails level 2, passes level 1)
    partial    alpha delta_barycenter + (1 - alpha) bayes, alpha ~ U(0,1)
    hallucin.  bayes with each atom perturbed by noise of strength s ~ U(0, 0.3)
    empirical  delta at the empirical frequencies
  Scores: d1 of the barycenter (L1 distance to conv H, LP); exact d2 = 2 x mass off H (Prop. 10.9); graded d2W =
  sum_c w_c min_h ||rho_c - rho_h||_1 (a transport version, open problem 5).
  Errors: E0 = TV(rho*, barycenter) now; E1 = TV(rho*, barycenter after the claim updates on 50 future observations, each
  atom reweighted by its likelihood). Question: which score predicts E1 (top quartile), and does d2 add to d1?"""
import os, sys, json, numpy as np
from scipy.optimize import linprog
h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(h, "..", "causalidview")); sys.path.insert(0, os.path.join(h, "../.."))

def auc(score, y):
    s, y = np.asarray(score, float), np.asarray(y, bool); pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))

def part_a(n_worlds=40):
    import run as R
    out = {}
    fits = {"logistic": lambda W: R.estimate(W, R.fit_lr), "XGBoost": lambda W: R.estimate(W, R.fit_xgb), "structured": R.estimate_ours}
    for name, f in fits.items():
        d1, err = [], []
        for seed in range(n_worlds):
            W = R.world(seed); E = f(W); ivo = np.array([R.iv_bounds(W["q"][i]) for i in range(len(W["qry"]))])
            d1 += list(E["slack"]); err += list(np.maximum(np.abs(E["Li"] - ivo[:, 0]), np.abs(E["Ui"] - ivo[:, 1])))
        d1, err = np.array(d1), np.array(err); top = err >= np.quantile(err, 0.75); flag = d1 > 1e-7
        out[name] = dict(units=len(err), flagged=float(flag.mean()), auc_top_quartile=auc(d1, top),
                         err_flagged=float(err[flag].mean()) if flag.any() else None, err_unflagged=float(err[~flag].mean()),
                         precision_flag_in_top_quartile=float(top[flag].mean()) if flag.any() else None)
        print("A", name, json.dumps(out[name]), flush=True)
    return out

def d1_conv(x, S):
    """L1 distance from x to conv(S) (S rows): min ||x - S^T w||_1, w in simplex"""
    m, k = S.shape; c = np.r_[np.zeros(m), np.ones(k)]
    A = np.block([[S.T, -np.eye(k)], [-S.T, -np.eye(k)]]); b = np.r_[x, -x]
    r = linprog(c, A_ub=A, b_ub=b, A_eq=np.r_[np.ones(m), np.zeros(k)][None], b_eq=[1], bounds=(0, None), method="highs"); return r.fun

def part_b(trials=1500, K=4, m=6, n1=20, n2=50, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for t in range(trials):
        H = rng.dirichlet(np.ones(K), size=m); hs = rng.integers(m); rho = H[hs]
        x = rng.multinomial(n1, rho); fut = rng.multinomial(n2, rho)
        ll = x @ np.log(H.T); w = np.exp(ll - ll.max()); w /= w.sum(); bary = w @ H
        claims = {"bayes": (w, H), "collapsed": (np.ones(1), bary[None]),
                  "empirical": (np.ones(1), ((x + 0.5) / (n1 + 0.5 * K))[None])}
        a = rng.uniform(); claims["partial"] = (np.r_[a, (1 - a) * w], np.vstack([bary, H]))
        s = rng.uniform(0, 0.3); P = np.abs(H + s * rng.normal(size=H.shape)); P /= P.sum(1, keepdims=True); claims["hallucinated"] = (w, P)
        for name, (wc, A) in claims.items():
            b0 = wc @ A; on = np.array([np.abs(A[i] - H).sum(1).min() < 1e-12 for i in range(len(A))])
            lu = fut @ np.log(A.T); wu = wc * np.exp(lu - lu.max()); wu /= wu.sum(); b1 = wu @ A
            rows.append(dict(claim=name, d1=d1_conv(b0, H), d2=float(2 * wc[~on].sum()), d2W=float(wc @ np.array([np.abs(A[i] - H).sum(1).min() for i in range(len(A))])),
                             E0=0.5 * float(np.abs(b0 - rho).sum()), E1=0.5 * float(np.abs(b1 - rho).sum())))
    import collections
    E1 = np.array([r["E1"] for r in rows]); E0 = np.array([r["E0"] for r in rows]); top1 = E1 >= np.quantile(E1, 0.75); top0 = E0 >= np.quantile(E0, 0.75)
    res = {"auc_E1_top_quartile": {k: auc([r[k] for r in rows], top1) for k in ("d1", "d2", "d2W")},
           "auc_E0_top_quartile": {k: auc([r[k] for r in rows], top0) for k in ("d1", "d2", "d2W")}}
    inside = np.array([r["d1"] < 1e-9 for r in rows])               # level-one admissible claims only
    res["among_level1_admissible"] = dict(n=int(inside.sum()), auc_E1_d2=auc(np.array([r["d2"] for r in rows])[inside], top1[inside]),
                                          auc_E1_d2W=auc(np.array([r["d2W"] for r in rows])[inside], top1[inside]))
    by = collections.defaultdict(list)
    for r in rows: by[r["claim"]].append(r)
    res["by_claim"] = {k: {q: float(np.mean([r[q] for r in v])) for q in ("d1", "d2", "d2W", "E0", "E1")} for k, v in by.items()}
    for k, v in res.items(): print("B", k, json.dumps(v, indent=None if k != "by_claim" else 1), flush=True)
    return res

if __name__ == "__main__":
    out = dict(B=part_b())
    if os.environ.get("CAUSALIDVIEW_DIR"): out["A"] = part_a(int(os.environ.get("NW", 40)))
    json.dump(out, open(os.path.join(h, "problem6.json"), "w"), indent=1, default=float)
