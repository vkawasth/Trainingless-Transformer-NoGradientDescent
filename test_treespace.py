import numpy as np
from amb_vigneaux.treespace import spider_point, dist, frechet_mean, energy_test


def test_spider_point_and_metric():
    assert spider_point([0.0, 0.1, 1.0]) == (0, 0.8)
    assert spider_point([0.0, 1.0, 2.0])[0] == -1                      # equal gaps: star tree
    assert dist((0, 0.5), (0, 0.2)) == 0.3 and dist((0, 0.5), (1, 0.2)) == 0.7


def test_frechet_mean_sticks_and_unsticks():
    pts = [(0, 1.0)] * 3 + [(1, 1.0)] * 3 + [(2, 1.0)] * 3
    assert frechet_mean(pts)[0] == (-1, 0.0)                           # balanced: sticky at the origin
    pts = [(0, 1.0)] * 7 + [(1, 1.0)] * 1 + [(2, 1.0)] * 1
    mu, _ = frechet_mean(pts)
    assert mu[0] == 0 and abs(mu[1] - (7 - 2) / 9) < 1e-12
    # brute-force check of the minimiser
    grid = [(a, t) for a in range(3) for t in np.linspace(0, 2, 2001)]
    f = lambda q: sum(dist(q, p) ** 2 for p in pts)
    best = min(grid, key=f)
    assert best[0] == 0 and abs(best[1] - mu[1]) < 1e-3


def test_energy_test_detects_leg_shift():
    rng = np.random.default_rng(0)
    X = [(int(rng.integers(3)), float(rng.exponential())) for _ in range(150)]
    Y = [(0 if rng.random() < 0.7 else int(rng.integers(3)), float(rng.exponential())) for _ in range(150)]
    assert energy_test(X, Y, B=199)["p"] < 0.01
    Z = [(int(rng.integers(3)), float(rng.exponential())) for _ in range(150)]
    assert energy_test(X, Z, B=199)["p"] > 0.01
