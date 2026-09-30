import numpy as np
from amb_vigneaux.toggle import toggle_radius, split_functional
from amb_vigneaux.certificate import _design


def _X():
    edges = [(s, t) for s in "abc" for t in "xyz"]
    return edges, _design(edges, list("abc"), list("xyz"))


def test_loop_reading_outcome_cannot_be_flipped_naturally():
    edges, X = _X(); w = np.ones(len(edges))
    a = np.zeros(9); a[0], a[1], a[3], a[4] = 1, -1, -1, 1          # a 4-cycle: reads only the loop part
    b = np.random.default_rng(0).normal(size=9)
    r = toggle_radius(a, b, X, w)
    assert np.isinf(r["r_natural"]) and np.isfinite(r["r_loop"]) and r["share_exact"] < 1e-12


def test_potential_reading_outcome_cannot_be_flipped_by_loops():
    edges, X = _X(); w = np.ones(9)
    a = np.array([2 if s == "b" else -1 for s, t in edges], float) / 3   # centre vs the others, pooled over topics
    r = toggle_radius(a, np.random.default_rng(1).normal(size=9), X, w)
    assert np.isinf(r["r_loop"]) and abs(r["share_exact"] - 1) < 1e-12


def test_radius_is_exact_distance():
    edges, X = _X(); w = np.ones(9); a = np.random.default_rng(2).normal(size=9); b = np.random.default_rng(3).normal(size=9)
    r = toggle_radius(a, b, X, w)
    b2 = b - r["f"] * a / (a @ a)            # move along a by exactly r_all
    assert abs(np.linalg.norm(b2 - b) - r["r_all"]) < 1e-12 and abs(a @ b2) < 1e-12
