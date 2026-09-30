import numpy as np
from amb_vigneaux.spectral import slepian_step_test, multitaper, spectral_change, harmonic_ftest


def test_step_vs_trend():
    rng = np.random.default_rng(1); n = 146
    y = rng.normal(size=n) * 0.3 + 1.0 * (np.arange(n) >= 73)
    assert abs(slepian_step_test(y, 73, NW=1)["z"]) > 3
    y2 = rng.normal(size=n) * 0.3 + np.linspace(-1, 1, n)
    assert abs(slepian_step_test(y2, 73, NW=1)["z"]) < 3


def test_multitaper_white_flat_and_line():
    rng = np.random.default_rng(2); n = 512
    f, S, se, K = multitaper(rng.normal(size=n), NW=3)
    assert K == 5 and 0.6 < np.median(S) < 1.6
    y = rng.normal(size=n) + 2 * np.sin(2 * np.pi * 0.2 * np.arange(n))
    h = harmonic_ftest(y, NW=3)
    assert abs(h["f_best"] - 0.2) < 0.01 and h["p_adj"] < 0.01


def test_spectral_change_persistence():
    rng = np.random.default_rng(3); n = 200
    a = rng.normal(size=n); e = rng.normal(size=n) * np.sqrt(1 - 0.81); b = np.zeros(n)
    for i in range(1, n): b[i] = 0.9 * b[i - 1] + e[i]
    hi = spectral_change(a, b, NW=3)[-1]
    assert hi["ratio"] < 0.5 and hi["p"] < 0.01
