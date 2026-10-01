import numpy as np
from amb_vigneaux.curvature import efron_gamma2, pull_jets, shift_jets, mixture_lr

logit = lambda p: np.log(p / (1 - p))


def test_pull_jets_match_finite_differences_and_shift_is_flat():
    rng = np.random.default_rng(0); p = rng.uniform(0.2, 0.8, 30); q = 0.9; h = 1e-4
    e = lambda w: logit((1 - w) * p + w * q)
    d1, d2 = pull_jets(p, q)
    assert np.allclose(d1, (e(h) - e(-h)) / (2 * h), atol=1e-6)
    assert np.allclose(d2, (e(h) - 2 * e(0) + e(-h)) / h ** 2, atol=1e-3)
    I = rng.uniform(50, 200, 30)
    s1, s2 = shift_jets(p, 0.1)
    assert efron_gamma2(s1, s2, I) == 0.0 and efron_gamma2(d1, d2, I) > 0


def test_curvature_shrinks_with_information():
    rng = np.random.default_rng(1); p = rng.uniform(0.2, 0.8, 40); d1, d2 = pull_jets(p, 0.9)
    g_small = efron_gamma2(d1, d2, np.full(40, 10.0)); g_big = efron_gamma2(d1, d2, np.full(40, 1000.0))
    assert abs(g_small / g_big - 100) < 1e-6                      # gamma^2 scales as 1 / information


def test_hidden_mixture_em_recovers_planted_weight():
    rng = np.random.default_rng(2); n = 4000; s2 = np.full(n, 0.01)
    z = rng.random(n) < 0.3; r = rng.normal(0, 0.1, n) + 0.3 * z
    pi, lr = mixture_lr(r, s2, [0.3, -0.3])
    assert abs(pi[0] - 0.3) < 0.04 and pi[1] < 0.03 and lr > 50
    pi0, lr0 = mixture_lr(rng.normal(0, 0.1, n), s2, [0.3, -0.3])
    assert lr0 < 10
