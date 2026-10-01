"""Paths over the gluing lattice that raise (or lower) the probability of an outcome.

Setting: cells (s, t) (source x topic) with successes k and trials n for a binary label. A REGION is a set of topics;
on a region the glued (loop-free) model is additive on the log-odds scale, logit p(s,t) = alpha_s + beta_t, fitted by
weighted least squares on empirical logits (weights n p (1 - p); closed form, no iterations). The region GLUES if the
weighted residual Q (the Hodge loop part of the certificate) passes chi^2(df) at level 0.05.
Outcome: a TARGET cell (s*, t*) is held out; the glued model on a region containing t* predicts
p_hat = expit(alpha_{s*} + beta_{t*}); the binary outcome is [p_hat > 1/2].
Over the lattice of glueable regions containing t*:
  envelope(target)   min and max of p_hat over all glueable regions up to a size (exhaustive), with the arg-regions;
  greedy_path(...)   add one topic at a time, staying glueable, choosing the topic that raises (or lowers) p_hat most.
The envelope width is the sensitivity of the conclusion to the choice of regions; the paths say which regions push it.
Selecting a region to reach a desired value is cherry-picking unless the whole envelope is reported.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy import stats


def _logit(k, n):
    p = (k + 0.5) / (n + 1.0)
    return np.log(p / (1 - p)), n * p * (1 - p)


def fit_region(cells, sources, topics):
    """cells: dict (s, t) -> (k, n). Returns alpha, beta, Q, df over the cells whose topic is in `topics`."""
    rows = [(s, t) for (s, t) in cells if t in topics and s in sources]
    S = sorted({s for s, _ in rows}); T = sorted({t for _, t in rows})
    if len(S) < 2 or len(T) < 1:
        return None
    X = np.zeros((len(rows), len(S) + len(T))); y = np.zeros(len(rows)); w = np.zeros(len(rows))
    for i, (s, t) in enumerate(rows):
        X[i, S.index(s)] = 1; X[i, len(S) + T.index(t)] = 1
        y[i], w[i] = _logit(*cells[(s, t)])
    WX = X * w[:, None]
    G = X.T @ WX; coef = np.linalg.pinv(G) @ (WX.T @ y)
    res = y - X @ coef; Q = float((w * res ** 2).sum())
    rank = np.linalg.matrix_rank(G); df = len(rows) - rank
    return dict(S=S, T=T, coef=coef, cov=np.linalg.pinv(G), Q=Q, df=int(df),
                p=float(stats.chi2.sf(Q, df)) if df > 0 else 1.0)


def predict(fit, s, t):
    if fit is None or s not in fit["S"] or t not in fit["T"]:
        return None
    i, j = fit["S"].index(s), len(fit["S"]) + fit["T"].index(t)
    v = np.zeros(len(fit["coef"])); v[i] = 1; v[j] = 1
    eta = float(v @ fit["coef"]); se = float(np.sqrt(max(v @ fit["cov"] @ v, 0)))
    return dict(p=float(1 / (1 + np.exp(-eta))), eta=eta, se=se)


def _without(cells, target):
    return {k: v for k, v in cells.items() if k != target}


def evaluate(cells, target, region, alpha=0.05):
    s, t = target
    fit = fit_region(_without(cells, target), sorted({a for a, _ in cells}), set(region))
    pr = predict(fit, s, t)
    if pr is None:
        return None
    return dict(region=tuple(sorted(region)), p=pr["p"], se=pr["se"], glue_p=fit["p"], glues=fit["p"] > alpha, df=fit["df"])


def envelope(cells, target, topics, max_size=6, alpha=0.05, min_size=1):
    """sizes count the target topic: regions of min_size..max_size topics that contain t*."""
    s, t = target
    others = [x for x in topics if x != t]
    best_hi = best_lo = None; n_glue = 0; all_ = []
    for k in range(max(1, min_size - 1), max_size):
        for comb in itertools.combinations(others, k):
            e = evaluate(cells, target, (t,) + comb, alpha)
            if e is None or not e["glues"]:
                continue
            n_glue += 1; all_.append((e["p"], e["glue_p"], len(e["region"])))
            if best_hi is None or e["p"] > best_hi["p"]:
                best_hi = e
            if best_lo is None or e["p"] < best_lo["p"]:
                best_lo = e
    return dict(n_glueable=n_glue, lo=best_lo, hi=best_hi, spread=(best_hi["p"] - best_lo["p"]) if best_hi else None,
                p_values=[a[0] for a in all_], regions=all_)


def greedy_path(cells, target, topics, direction=+1, alpha=0.05):
    s, t = target
    region = [t]; path = []; pool = [x for x in topics if x != t]
    while pool:
        cands = [(x, evaluate(cells, target, region + [x], alpha)) for x in pool]
        cands = [(x, e) for x, e in cands if e is not None and e["glues"]]
        if not cands:
            break
        x, e = max(cands, key=lambda c: direction * c[1]["p"])
        if path and direction * (e["p"] - path[-1]["p"]) <= 0:
            break
        region.append(x); pool.remove(x); path.append(dict(e, added=x))
    return path


def prune_path(cells, target, topics, direction=+1, alpha=0.05, min_size=5):
    """start from the full cover (or the largest glueable region found by dropping the worst-fitting topics) and drop
    one topic at a time, staying glueable and >= min_size, choosing the drop that raises (direction=+1) or lowers p."""
    s, t = target
    region = list(topics)
    e = evaluate(cells, target, region, alpha)
    while e is not None and not e["glues"] and len(region) > min_size:          # descend to a glueable region first
        cands = [(x, evaluate(cells, target, [y for y in region if y != x], alpha)) for x in region if x != t]
        x, e = max([c for c in cands if c[1] is not None], key=lambda c: c[1]["glue_p"])
        region.remove(x)
    if e is None or not e["glues"]:
        return []
    path = [dict(e, dropped=None)]
    while len(region) > min_size:
        cands = [(x, evaluate(cells, target, [y for y in region if y != x], alpha)) for x in region if x != t]
        cands = [(x, c) for x, c in cands if c is not None and c["glues"]]
        if not cands:
            break
        x, c = max(cands, key=lambda z: direction * z[1]["p"])
        if direction * (c["p"] - path[-1]["p"]) <= 0:
            break
        region.remove(x); path.append(dict(c, dropped=x))
    return path
