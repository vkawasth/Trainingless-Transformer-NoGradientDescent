"""CausalIDView partial-identification benchmark (Jung et al., arXiv 2609.36881), Manski and monotone-IV views.

The released CausalIDView code covers only the point-estimation figure; the partial-identification companion SCM is
described in the paper (Algorithm 2, Appendix C.5) but its code and calibrated amplitudes are not released. We rebuild
it from Algorithm 2 with their released effect families, nuisance functions and seeds (CAUSALIDVIEW_DIR, the cloned
repository), and the stated constraints (A_b + A_tau / 2 <= 0.45; loadings 0.55 / 0.75 / 1.5 from their Table 4):
    e_j(x, u) = expit(b_T + a_X x.v_T + a_U u + a_I j)            b_T calibrated to mean propensity 1/2
    b(x, u)   = 1/2 + A_b tanh(f_b(x) + c_U u),  tau0(x) = A_tau tanh(q_f(x)),  p_t = b + (t - 1/2) tau0
    A_b = 0.2, A_tau = 0.4, c_U = 0.8; X ~ N(0, I_50), U = +-1, I ~ Bernoulli(1/2), T = T(I), Y = Y(T)
40 worlds (seeds 0..39), 1024 context and 100 query units. Population oracle cells at the query covariates are exact
(mixing over U), and the oracle bounds are the Manski formula and the Balke-Pearl monotone-IV linear programme.
Estimators, as in their Appendix E.2.4: a backbone estimates P(T, Y | X) (Manski) and P(T, Y | X, I) (IV); the bound
functional is fixed. Backbones: multinomial logistic regression and XGBoost (theirs), and our gradient-free multinomial
logistic regression by Newton/IRLS steps. TabPFN-v3.5 and the causal foundation models need checkpoints we cannot
download here; their published values are quoted for reference.
Metrics, as theirs: endpoint RMSE per world (mean +- sd over worlds) and true-CATE containment. Added by the engine:
  * the IV feasibility diagnostic: the minimal L1 slack that makes estimated IV cells consistent with the monotone-IV
    model (0 = feasible; > 0 = the instrumental inequality fails -- the analogue of a positive contextual fraction).
    Bounds for infeasible cells are taken on the optimal face (the projection), and the slack is REPORTED, not hidden;
  * coverage: a context bootstrap (refit the backbone) gives 90% intervals for each endpoint; we report how often they
    contain the oracle endpoints (their containment metric is about tau0, not about the bounds);
  * compute: seconds per world for the backbone and for the bound stage.
"""
import os, sys, json, time
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
CIV = os.environ.get("CAUSALIDVIEW_DIR", "/tmp/ds/CausalIDView")
sys.path.insert(0, CIV)
import numpy as np
from scipy.special import expit
from scipy.optimize import linprog
from data import scm as S                       # their released SCM helpers (families, seeds, unit vectors)

A_B, A_TAU, C_U, A_X, A_U, A_I = 0.2, 0.4, 0.8, 0.55, 0.75, 1.5
D, NC, NQ, NCAL = 50, 1024, 100, 50000
TYPES = [(d0, d1, y0, y1) for (d0, d1) in ((0, 0), (0, 1), (1, 1)) for y0 in (0, 1) for y1 in (0, 1)]
AIV = np.array([[1.0 if (r[j] == t and r[2 + t] == y) else 0.0 for r in TYPES] for j in (0, 1) for t in (0, 1) for y in (0, 1)])
EFF = np.array([r[3] - r[2] for r in TYPES], float)


def world(seed):
    fam = S.CATE_FAMILIES[seed % len(S.CATE_FAMILIES)]
    x = S._rng(seed, "X").normal(size=(NC + NQ, D)); xc = S._rng(seed, "X_calibration").normal(size=(NCAL, D))
    cp = S._cate_parameters(seed, D, fam); rc = S._raw_cate(xc, fam, cp)
    qf = lambda X: (S._raw_cate(X, fam, cp) - rc.mean()) / rc.std()
    rng = S._rng(seed, "nuisance_parameters"); mu_w = S._unit_vector(rng, D, 14); vT = S._unit_vector(rng, D, 14)
    fb = lambda X: 0.7 * (X @ mu_w) + 0.25 * np.sin(X @ np.roll(mu_w, 1))
    cal = np.concatenate([A_X * (xc @ vT) + A_U * u + A_I * j for u in (-1, 1) for j in (0, 1)])
    bT = S._calibrate_intercept(cal)
    e = lambda X, u, j: expit(bT + A_X * (X @ vT) + A_U * u + A_I * j)
    b = lambda X, u: 0.5 + A_B * np.tanh(fb(X) + C_U * u)
    tau = lambda X: A_TAU * np.tanh(qf(X))
    u = np.where(S._rng(seed, "U").random(NC + NQ) < 0.5, -1, 1); I = (S._rng(seed, "I").random(NC + NQ) < 0.5).astype(int)
    vt = S._rng(seed, "V_T").random(NC + NQ); vy = S._rng(seed, "V_Y").random(NC + NQ)
    T = (vt <= e(x, u, I)).astype(int); pT = b(x, u) + (T - 0.5) * tau(x); Y = (vy <= pT).astype(int)
    perm = S._rng(seed, "split_permutation").permutation(NC + NQ); ctx, qry = perm[:NC], perm[NC:]
    # oracle cells at query covariates: q[j, t, y] = mean_u P(T=t | x,u,j) P(Y(t)=y | x,u)
    Xq = x[qry]; q = np.zeros((len(qry), 2, 2, 2))
    for uu in (-1, 1):
        for j in (0, 1):
            ej = e(Xq, uu, j)
            for t in (0, 1):
                pt = b(Xq, uu) + (t - 0.5) * tau(Xq); pt_t = ej if t else 1 - ej
                for y in (0, 1):
                    q[:, j, t, y] += 0.5 * pt_t * (pt if y else 1 - pt)
    return dict(X=x, T=T, Y=Y, I=I, ctx=ctx, qry=qry, q=q, tau=tau(Xq), family=fam)


def manski(qm):                       # qm[:, t, y] = P(T=t, Y=y | X)
    return -qm[:, 1, 0] - qm[:, 0, 1], qm[:, 1, 1] + qm[:, 0, 0]


def iv_bounds(qj):
    """qj[j, t, y] = P(T=t, Y=y | I=j, X=x). Exact LP if feasible; otherwise minimal L1 slack, then the bounds over the
    optimal face (slack fixed at its minimum). Returns lo, hi, slack."""
    qj = np.asarray(qj, float); qj = qj / qj.reshape(2, -1).sum(1)[:, None, None]      # float32 backbones: renormalise
    b = qj.reshape(-1); n = AIV.shape[1]; m = AIV.shape[0]
    Aeq = np.hstack([AIV, np.eye(m), -np.eye(m)]); c = np.concatenate([np.zeros(n), np.ones(2 * m)])
    Aeq1 = np.vstack([Aeq, np.concatenate([np.ones(n), np.zeros(2 * m)])]); beq1 = np.concatenate([b, [1.0]])
    r = linprog(c, A_eq=Aeq1, b_eq=beq1, bounds=(0, None), method="highs"); slack = float(r.fun)
    Aub = c.reshape(1, -1); bub = [slack * (1 + 1e-6) + 1e-7]
    lo = linprog(np.concatenate([EFF, np.zeros(2 * m)]), A_ub=Aub, b_ub=bub, A_eq=Aeq1, b_eq=beq1, bounds=(0, None), method="highs").fun
    hi = -linprog(-np.concatenate([EFF, np.zeros(2 * m)]), A_ub=Aub, b_ub=bub, A_eq=Aeq1, b_eq=beq1, bounds=(0, None), method="highs").fun
    return float(lo), float(hi), slack


# ------------------------------------------------------------------------------------------- backbones
def fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=2000); m.fit(X, y); return lambda Z: _full(m.predict_proba(Z), m.classes_)


def fit_xgb(X, y):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.9, objective="multi:softprob", verbosity=0)
    cls = np.unique(y); mp = {c: i for i, c in enumerate(cls)}; m.fit(X, np.array([mp[v] for v in y]))
    return lambda Z: _full(m.predict_proba(Z), cls)


def fit_irls(X, y, ridge=1.0, iters=50):
    """multinomial logistic regression by Newton/IRLS steps (closed-form linear solves; no gradient descent)"""
    Z = np.hstack([np.ones((len(X), 1)), X]); K = 4; p = Z.shape[1]; W = np.zeros((p, K - 1)); Yo = np.eye(K)[y][:, 1:]
    for _ in range(iters):
        eta = np.hstack([np.zeros((len(Z), 1)), Z @ W]); P = np.exp(eta - eta.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); Pk = P[:, 1:]
        g = (Z.T @ (Yo - Pk)).T.ravel() - ridge * np.concatenate([np.r_[0, W[1:, k]] for k in range(K - 1)])
        H = np.zeros(((K - 1) * p, (K - 1) * p))
        for a in range(K - 1):
            for c in range(K - 1):
                w = Pk[:, a] * ((a == c) - Pk[:, c]); H[a * p:(a + 1) * p, c * p:(c + 1) * p] = (Z * w[:, None]).T @ Z
        H += ridge * np.eye(H.shape[0])
        step = np.linalg.solve(H, g); W += step.reshape(K - 1, p).T
        if np.abs(step).max() < 1e-8:
            break
    def pred(Q):
        Zq = np.hstack([np.ones((len(Q), 1)), Q]); eta = np.hstack([np.zeros((len(Zq), 1)), Zq @ W]); P = np.exp(eta - eta.max(1, keepdims=True))
        return P / P.sum(1, keepdims=True)
    return pred


def _full(P, cls):
    out = np.zeros((P.shape[0], 4)); out[:, cls] = np.asarray(P, float); return out / out.sum(1, keepdims=True)


BACKBONES = {"logistic regression": fit_lr, "XGBoost": fit_xgb, "IRLS logistic (ours, gradient-free)": fit_irls,
             "structured (ours, gradient-free)": None}


def estimate(Wd, fit):
    c, qy = Wd["ctx"], Wd["qry"]; X = Wd["X"]; cls = 2 * Wd["T"][c] + Wd["Y"][c]
    t0 = time.time()
    pm = fit(X[c], cls)(X[qy]).reshape(-1, 2, 2)
    fiv = fit(np.hstack([X[c], Wd["I"][c][:, None]]), cls)
    pj = np.stack([fiv(np.hstack([X[qy], np.full((len(qy), 1), j)])) for j in (0, 1)], 1).reshape(-1, 2, 2, 2)
    t_back = time.time() - t0; t0 = time.time()
    Lm, Um = manski(pm); ivr = np.array([iv_bounds(pj[i]) for i in range(len(qy))])
    t_bound = time.time() - t0
    return dict(Lm=Lm, Um=Um, Li=ivr[:, 0], Ui=ivr[:, 1], slack=ivr[:, 2], t_back=t_back, t_bound=t_bound)


# ------------------------------------------------------------------------------------------- our structured backbone
def _sig(z):
    return 1 / (1 + np.exp(-z))


def _irls_bin(Z, y, lam, iters=60, free=(0,)):
    """ridge logistic regression by Newton steps; coordinates in `free` (intercept, instrument) are not penalised"""
    p = Z.shape[1]; b = np.zeros(p); P = np.eye(p) * lam
    for i in free:
        P[i, i] = 0
    for _ in range(iters):
        mu = _sig(Z @ b); W = mu * (1 - mu) + 1e-9
        st = np.linalg.solve((Z * W[:, None]).T @ Z + P + 1e-8 * np.eye(p), Z.T @ (y - mu) - P @ b); b += st
        if np.abs(st).max() < 1e-8:
            break
    return b


def _cv_logistic(Z, y, lams=(1, 3, 10, 30, 100, 300, 1000, 3000), k=5, seed=0, free=(0,)):
    """ridge logistic regression, ridge chosen by k-fold cross-validated log-loss (closed-form Newton fits only)"""
    f = np.random.default_rng(seed).integers(0, k, len(y)); best = None
    for lam in lams:
        ll = 0.0
        for i in range(k):
            b = _irls_bin(Z[f != i], y[f != i], lam, free=free); p = np.clip(_sig(Z[f == i] @ b), 1e-6, 1 - 1e-6)
            ll -= float((y[f == i] * np.log(p) + (1 - y[f == i]) * np.log(1 - p)).sum())
        if best is None or ll < best[0]:
            best = (ll, lam)
    return _irls_bin(Z, y, best[1], free=free)


def _krr_linear_loo(Xtr, y, Xte, lams=(1, 3, 10, 30, 100, 300, 1000, 3000)):
    """kernel ridge regression (linear kernel) with the closed-form leave-one-out choice of the ridge"""
    m = y.mean(); yc = y - m; K = Xtr @ Xtr.T; ev, V = np.linalg.eigh(K); yt = V.T @ yc; best = None
    for lam in lams:
        d = ev / (ev + lam); loo = (yc - V @ (d * yt)) / (1 - (V ** 2) @ d)
        if best is None or np.mean(loo ** 2) < best[0]:
            best = (np.mean(loo ** 2), lam)
    alpha = V @ (yt / (ev + best[1]))
    return np.clip(m + (Xte @ Xtr.T) @ alpha, 0, 1)


def estimate_ours(Wd):
    """Manski: the bounds depend only on s(x) = P(Y = T | X) (L = s - 1, U = s), so estimate that one binary quantity
    directly (linear kernel ridge, closed-form LOO). IV: factorise the cells as P(T | X, I) P(Y | T, X, I), each a
    cross-validated ridge logistic regression (outcome fitted per treatment arm) with the instrument coefficient unpenalised.
    No gradient steps."""
    c, qy = Wd["ctx"], Wd["qry"]; X = Wd["X"]; T = Wd["T"][c].astype(float); Y = Wd["Y"][c].astype(float); I = Wd["I"][c].astype(float)
    t0 = time.time(); nq = len(qy)
    s = _krr_linear_loo(X[c], (Y == T).astype(float), X[qy])
    one = lambda n: np.ones((n, 1)); Zc = np.hstack([one(len(c)), X[c], I[:, None]]); Zq = [np.hstack([one(nq), X[qy], np.full((nq, 1), j)]) for j in (0, 1)]
    fr = (0, Zc.shape[1] - 1)     # intercept and the single instrument coefficient are left unpenalised: shrinking the
                                  # instrument's effect makes it look weaker and widens the IV bounds (measured bias -0.036/+0.024)
    bT = _cv_logistic(Zc, T, free=fr); bY = {t: _cv_logistic(Zc[T == t], Y[T == t], free=fr) for t in (0, 1)}
    pj = np.zeros((nq, 2, 2, 2))
    for j in (0, 1):
        e = _sig(Zq[j] @ bT)
        for t in (0, 1):
            pt = e if t else 1 - e; my = _sig(Zq[j] @ bY[t]); pj[:, j, t, 1] = pt * my; pj[:, j, t, 0] = pt * (1 - my)
    t_back = time.time() - t0; t0 = time.time()
    ivr = np.array([iv_bounds(pj[i]) for i in range(nq)])
    return dict(Lm=s - 1, Um=s, Li=ivr[:, 0], Ui=ivr[:, 1], slack=ivr[:, 2], t_back=t_back, t_bound=time.time() - t0)


def ep_rmse(L, U, Lo, Uo):
    return float(np.sqrt(0.5 * np.mean((L - Lo) ** 2 + (U - Uo) ** 2)))


def main(n_worlds=40, boot=30):
    rows = {k: [] for k in BACKBONES}; cov = []; t_oracle = []
    for seed in range(n_worlds):
        Wd = world(seed); t0 = time.time()
        Lmo, Umo = manski(Wd["q"].mean(1)); ivo = np.array([iv_bounds(Wd["q"][i]) for i in range(len(Wd["qry"]))]); t_oracle.append(time.time() - t0)
        Lio, Uio, slo = ivo[:, 0], ivo[:, 1], ivo[:, 2]; tau = Wd["tau"]
        for name, fit in BACKBONES.items():
            E = estimate(Wd, fit) if fit is not None else estimate_ours(Wd)
            big = np.abs(np.c_[E["Li"] - Lio, E["Ui"] - Uio]).max(1)
            rows[name].append(dict(manski=ep_rmse(E["Lm"], E["Um"], Lmo, Umo), iv=ep_rmse(E["Li"], E["Ui"], Lio, Uio),
                                   cont_manski=float(np.mean((E["Lm"] <= tau) & (tau <= E["Um"]))), cont_iv=float(np.mean((E["Li"] <= tau) & (tau <= E["Ui"]))),
                                   infeasible=float(np.mean(E["slack"] > 1e-7)), slack=float(E["slack"].mean()),
                                   err_infeasible=float(big[E["slack"] > 1e-7].mean()) if (E["slack"] > 1e-7).any() else np.nan,
                                   err_feasible=float(big[E["slack"] <= 1e-7].mean()) if (E["slack"] <= 1e-7).any() else np.nan,
                                   t_back=E["t_back"], t_bound=E["t_bound"]))
        if boot:
            rngb = np.random.default_rng(seed); B = []
            for _ in range(boot):
                Wb = dict(Wd); Wb["ctx"] = rngb.choice(Wd["ctx"], len(Wd["ctx"]), replace=True); B.append(estimate_ours(Wb))
            def cover(key, oracle):
                arr = np.array([b[key] for b in B]); lo, hi = np.quantile(arr, 0.05, 0), np.quantile(arr, 0.95, 0)
                return float(np.mean((lo <= oracle) & (oracle <= hi)))
            cov.append(dict(Lm=cover("Lm", Lmo), Um=cover("Um", Umo), Li=cover("Li", Lio), Ui=cover("Ui", Uio)))
        r = {k: v[-1] for k, v in rows.items()}
        print(f"world {seed:2d} ({Wd['family']:22s}) oracle IV slack {slo.max():.1e} | " + " | ".join(f"{k.split(' (')[0][:14]} M {v['manski']:.3f} IV {v['iv']:.3f} infeas {v['infeasible']:.2f}" for k, v in r.items())
              + (f" | boot cover IV {np.mean([cov[-1]['Li'], cov[-1]['Ui']]):.2f}" if boot else ""), flush=True)
    summ = {}
    for name, R in rows.items():
        summ[name] = {k: (float(np.nanmean([r[k] for r in R])), float(np.nanstd([r[k] for r in R]))) for k in R[0]}
    if cov:
        summ["bootstrap coverage (structured, 90%)"] = {k: float(np.mean([c[k] for c in cov])) for k in cov[0]}
    summ["oracle_bound_seconds_per_world"] = float(np.mean(t_oracle))
    summ["published (Jung et al., Fig. 6c, read off the bars)"] = {"logistic regression": dict(manski=0.135, iv=0.175), "XGBoost": dict(manski=0.185, iv=0.23), "TabPFN-v3.5": dict(manski=0.09, iv=0.12)}
    json.dump(dict(rows=rows, coverage=cov, summary=summ), open(os.path.join(here, "results.json"), "w"), indent=1, default=float)
    print("\nSUMMARY (mean +- sd over worlds)")
    for name in BACKBONES:
        s = summ[name]
        print(f"  {name:38s} endpoint RMSE Manski {s['manski'][0]:.3f}±{s['manski'][1]:.3f}  IV {s['iv'][0]:.3f}±{s['iv'][1]:.3f} | containment Manski {s['cont_manski'][0]:.2f} IV {s['cont_iv'][0]:.2f}"
              f" | IV cells infeasible {s['infeasible'][0]:.2f} (mean slack {s['slack'][0]:.3f}); endpoint error infeasible {s['err_infeasible'][0]:.3f} vs feasible {s['err_feasible'][0]:.3f}"
              f" | seconds backbone {s['t_back'][0]:.2f}, bounds {s['t_bound'][0]:.2f}")
    if cov:
        print("  bootstrap 90% interval coverage of the ORACLE endpoints (structured backbone):", summ["bootstrap coverage (structured, 90%)"])
    print("  published:", summ["published (Jung et al., Fig. 6c, read off the bars)"])
    plot(summ)


def plot(summ):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, green, grey, navy = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#b8b6ae", "#2c3e7a"
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7)
    P = summ["published (Jung et al., Fig. 6c, read off the bars)"]
    labels = ["XGBoost\n(published)", "logistic\n(published)", "TabPFN-v3.5\n(published)", "XGBoost\n(rebuilt)", "logistic\n(rebuilt)", "structured\n(ours)"]
    for j, view in enumerate(("manski", "iv")):
        vals = [P["XGBoost"][view], P["logistic regression"][view], P["TabPFN-v3.5"][view],
                summ["XGBoost"][view][0], summ["logistic regression"][view][0], summ["structured (ours, gradient-free)"][view][0]]
        err = [0, 0, 0, summ["XGBoost"][view][1], summ["logistic regression"][view][1], summ["structured (ours, gradient-free)"][view][1]]
        ax[0].bar(np.arange(6) + j * 7, vals, yerr=err, color=[grey, grey, navy, orange, green, blue], capsize=2)
    ax[0].set_xticks(list(range(6)) + [7 + i for i in range(6)]); ax[0].set_xticklabels(labels * 2, fontsize=5.2, rotation=90)
    ax[0].text(2.5, ax[0].get_ylim()[1] * 0.95, "Manski", ha="center", fontsize=8); ax[0].text(9.5, ax[0].get_ylim()[1] * 0.95, "monotone IV", ha="center", fontsize=8)
    ax[0].set_ylabel("endpoint RMSE (lower is better)", fontsize=7.5, color=muted); ax[0].set_title("(a) bound accuracy: published vs rebuilt worlds", fontsize=8.5, loc="left")
    names = ["logistic regression", "XGBoost", "structured (ours, gradient-free)"]; x = np.arange(3)
    ax[1].bar(x - 0.2, [summ[n]["infeasible"][0] for n in names], 0.4, color=orange, label="share of query units with infeasible IV cells")
    ax[1].bar(x + 0.2, [summ[n]["err_infeasible"][0] / max(summ[n]["err_feasible"][0], 1e-9) for n in names], 0.4, color=blue, label="endpoint error, infeasible / feasible")
    for i, n in enumerate(names):
        sh = summ[n]["infeasible"][0]; ra = summ[n]["err_infeasible"][0] / max(summ[n]["err_feasible"][0], 1e-9)
        ax[1].text(i - 0.2, sh + 0.04, f"{100 * sh:.1f}%", ha="center", fontsize=6.5, color=ink)
        ax[1].text(i + 0.2, min(ra, 3.4) + 0.04, f"{ra:.1f}x", ha="center", fontsize=6.5, color=ink)
    ax[1].axhline(1, color=ink, lw=0.6); ax[1].set_xticks(x); ax[1].set_xticklabels(["logistic", "XGBoost", "structured (ours)"], fontsize=7)
    ax[1].set_ylim(0, 3.8); ax[1].legend(fontsize=6.3, frameon=False, loc="upper center"); ax[1].set_title("(b) the feasibility flag their projection hides", fontsize=8.5, loc="left")
    C = summ.get("bootstrap coverage (structured, 90%)")
    if C:
        ax[2].bar(range(4), [C["Lm"], C["Um"], C["Li"], C["Ui"]], color=[green, green, blue, blue]); ax[2].axhline(0.9, color=ink, lw=0.8, ls="--")
        ax[2].set_xticks(range(4)); ax[2].set_xticklabels(["Manski L", "Manski U", "IV L", "IV U"], fontsize=7); ax[2].set_ylim(0, 1)
        ax[2].set_title("(c) do 90% bootstrap intervals cover\nthe ORACLE bounds? (structured backbone)", fontsize=8.5, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(here, "causalidview_chart.pdf")); fig.savefig(os.path.join(here, "causalidview_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json")))["summary"])
elif __name__ == "__main__":
    main(n_worlds=int(os.environ.get("WORLDS", "40")), boot=int(os.environ.get("BOOT", "30")))
