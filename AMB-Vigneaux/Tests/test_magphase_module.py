import itertools
import numpy as np
from amb_vigneaux.holonomy import Graph
from amb_vigneaux.magphase import *

G = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])


def test_sum_sq_unbiased_exact_expectation():
    # exact expectation over all multinomial outcomes for a small case
    from math import factorial
    p, N = np.array([0.5, 0.3, 0.2]), 5
    E = 0.0
    for c in itertools.product(range(N + 1), repeat=3):
        if sum(c) != N: continue
        w = factorial(N) / np.prod([factorial(x) for x in c]) * np.prod(p ** np.array(c))
        E += w * sum_sq_unbiased(c)
    assert abs(E - (p ** 2).sum()) < 1e-12


def test_i2_formula_and_modulus():
    for s in (0.0, 0.4, 0.9):
        T = np.array([[1 + s, 1 - s], [1 - s, 1 + s]]) * 1e6
        assert abs(modulus_from_i2(i2_unbiased(T)) - s) < 1e-3
        assert abs(modulus_from_mi(mi_plugin(T)) - s) < 1e-3


def test_dimension_deficit_planted():
    for p in (3, 5):
        assert dimension_deficit(G, [(1, 1), (1, 2), (1, 0), (1, p - 3)], p) == (0, p)
        assert dimension_deficit(G, [(1, 1), (1, 2), (1, 0), (1, p - 2)], p) == (1, 0)
        assert dimension_deficit(G, [(2, 0), (1, 0), (1, 0), (1, 0)], p) == (1, 1)


def test_dimension_is_cocycle_on_affine_supports_only():
    p, n, rng = 3, 3, np.random.default_rng(0)
    worst_aff, worst_gen = 0.0, 0.0
    for _ in range(100):
        k = rng.integers(1, n + 1)
        P = random_law_with_affine_support(rng.integers(0, p, (k, n)), rng.integers(0, p, n), p, rng)
        X, Y = rng.integers(0, p, (1, n)), rng.integers(0, p, (2, n))
        worst_aff = max(worst_aff, abs(cocycle_defect(lambda Q: dim_functional(Q, p), P, X, Y, p, 1.0)))
        assert cocycle_defect(s0_functional, P, X, Y, p, 0.0) == 0
        pts = [tuple(v) for v in rng.integers(0, p, (5, n))]
        Q = {}
        for v, w in zip(pts, rng.dirichlet(np.ones(5))): Q[v] = Q.get(v, 0) + w
        worst_gen = max(worst_gen, abs(cocycle_defect(lambda R: dim_functional(R, p), Q, X, Y, p, 1.0)))
        assert cocycle_defect(s0_functional, Q, X, Y, p, 0.0) == 0
    assert worst_aff < 1e-12 and worst_gen > 0.1
