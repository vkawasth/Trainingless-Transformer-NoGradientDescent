import numpy as np
from amb_vigneaux.lattice_path import fit_region, predict, evaluate, envelope, prune_path


def _cells(eta, n=10 ** 6):
    P = 1 / (1 + np.exp(-eta))
    return {(s, t): (int(round(P[s, t] * n)), n) for s in range(eta.shape[0]) for t in range(eta.shape[1])}, P


def test_additive_world_prediction_is_region_invariant():
    rng = np.random.default_rng(0); a, b = rng.normal(size=5), rng.normal(size=6)
    cells, P = _cells(a[:, None] + b[None, :])
    env = envelope(cells, (2, 3), list(range(6)), max_size=6, min_size=3)
    assert env["spread"] < 1e-3 and abs(env["hi"]["p"] - P[2, 3]) < 1e-3       # no loops: every glued region agrees


def test_loops_make_the_prediction_depend_on_the_region_and_pruning_stays_glueable():
    rng = np.random.default_rng(1); a, b = rng.normal(size=6), rng.normal(size=8)
    eta = a[:, None] + b[None, :]; eta[:, 5] += rng.normal(0, 2, 6); eta[:, 6] += rng.normal(0, 2, 6)
    rng2 = np.random.default_rng(2)
    P = 1 / (1 + np.exp(-eta)); cells = {(s, t): (int(rng2.binomial(40, P[s, t])), 40) for s in range(6) for t in range(8)}
    path = prune_path(cells, (0, 0), list(range(8)), +1, min_size=4)
    assert path and all(e["glues"] for e in path)
    assert all(path[i + 1]["p"] > path[i]["p"] for i in range(len(path) - 1))   # the raise path is monotone
    full = evaluate(cells, (0, 0), list(range(8)))
    assert full is not None and not full["glues"]                             # contaminated topics break the full cover


def test_first_order_landscape_influence_is_the_loop_residual():
    import itertools
    from amb_vigneaux.lattice_path import _logit
    rng = np.random.default_rng(5); S, T = 6, 7
    a, b = rng.normal(size=S), rng.normal(size=T); eta = a[:, None] + b[None, :] + rng.normal(0, 0.4, (S, T))
    P = 1 / (1 + np.exp(-eta)); cells = {(s, t): (int(rng.binomial(200, P[s, t])), 200) for s in range(S) for t in range(T)}
    tg = (0, 0); others = list(range(1, T)); L = {}
    for k in range(2, len(others) + 1):
        for comb in itertools.combinations(others, k):
            e = evaluate(cells, tg, (0,) + comb, alpha=0.0)                  # alpha=0: every region counts
            L[frozenset(comb)] = np.log(e["p"] / (1 - e["p"]))
    keys = list(L); y = np.array([L[k] for k in keys])
    X = np.array([[1.0] + [1.0 if x in k else 0.0 for x in others] for k in keys])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    full = fit_region({k: v for k, v in cells.items() if k != tg}, list(range(S)), set(range(T)))
    res = []
    for x in others:
        yx, wx = _logit(*cells[(0, x)]); res.append(wx * (yx - full["coef"][0] - full["coef"][S + x]))
    assert r2 > 0.8 and np.corrcoef(c[1:], res)[0, 1] > 0.8
