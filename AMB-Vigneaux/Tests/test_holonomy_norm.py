import numpy as np

from amb_vigneaux import holonomy_norm as H

rng = np.random.default_rng(1)


def test_class_function():
    for _ in range(50):
        p = rng.dirichlet(np.ones(9)); g = rng.integers(0, 9, 4)
        g2 = (g + H.coboundary(rng.integers(0, 9, 4))) % 9
        assert abs(H.kl_shift(p, g) - H.kl_shift(p, g2)) < 1e-14
        assert abs(H.kl_shift(p, g) - H.kl_shift(p, np.roll(g, 1))) < 1e-14


def test_zero_iff_free_action():
    p = rng.dirichlet(np.ones(8))
    assert all(H.kl(p, H.shift(p, h)) > 0 for h in range(1, 8))
    u = np.full(8, 1 / 8)
    assert all(H.kl(u, H.shift(u, h)) < 1e-15 for h in range(8))          # counterexample to 'iff' without freeness


def test_not_a_norm_but_locally_fisher():
    n = 64; x = np.arange(n); p = np.exp(np.sin(2 * np.pi * x / n) + 0.5 * np.sin(4 * np.pi * x / n + 1)); p /= p.sum()
    assert abs(H.kl(p, H.shift(p, 5)) - H.kl(p, H.shift(p, -5))) > 1e-3   # asymmetric
    dp = (np.roll(p, -1) - np.roll(p, 1)) / 2; F = np.sum(dp ** 2 / p)
    assert abs(H.kl(p, H.shift(p, 1)) / (F / 2) - 1) < 0.05


def test_tilt_is_bregman_and_quadratic():
    p = rng.dirichlet(np.ones(7)); T = rng.normal(size=(7, 2)); h = np.array([0.003, -0.002])
    assert abs(H.kl_tilt(p, h, T) - H.bregman_psi(p, h, T)) < 1e-15
    assert abs(H.kl_tilt(p, h, T) / H.quad(p, h, T) - 1) < 0.02
