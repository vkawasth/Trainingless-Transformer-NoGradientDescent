"""Magnitude (Layer 2) and phase (holonomy): logical Bell bound, CF from edge moduli + one bit,
Tsallis-2 edge information, and S_2 as normalised flag dimension."""
import numpy as np
from amb_vigneaux import models, analyse_outcomes
from amb_vigneaux.scenario import EmpiricalModel


def _cycle(s):
    return EmpiricalModel.from_function(models.chsh_scenario(),
                                        lambda C, o: (1 + s[C] * (1 if o[0] == o[1] else -1)) / 4)


def test_cf_equals_moduli_plus_holonomy_bit():
    rng = np.random.default_rng(3)
    sc = models.chsh_scenario()
    for _ in range(60):
        s = {C: rng.uniform(-1, 1) * (rng.random() < .9) for C in sc.contexts}
        cf = analyse_outcomes(_cycle(s)).contextual_fraction
        a = np.array([abs(v) for v in s.values()]); odd = np.prod(list(s.values())) < 0
        pred = max(0, (a.sum() - 2) / 2) if odd else max(0, (a.sum() - 2 * a.min() - 2) / 2)
        assert abs(cf - pred) < 1e-9


def test_logical_bell_lower_bound():
    rng = np.random.default_rng(5)
    for _ in range(30):
        v = rng.uniform(0, 1)
        m = models.noisy(models.pr_box(), v)
        p = (1 + v) / 2                                   # each PR parity formula holds w.p. p
        assert analyse_outcomes(m).contextual_fraction >= 4 * p - 3 - 1e-9


def test_tsallis2_edge_information():
    S2 = lambda p: 1 - np.sum(np.asarray(p) ** 2)
    for s in np.linspace(-1, 1, 9):
        J = np.array([[1 + s, 1 - s], [1 - s, 1 + s]]) / 4
        px = J.sum(1)
        I2 = S2(J.sum(0)) - sum(px[x] ** 2 * S2(J[x] / px[x]) for x in range(2))   # Vigneaux alpha-action
        assert abs(I2 - (1 + s * s) / 4) < 1e-12


def test_s2_is_normalised_flag_dimension():
    rng = np.random.default_rng(0)
    for _ in range(20):
        k = rng.integers(1, 6, size=rng.integers(2, 5)); n = k.sum()
        dim = sum(k[i] * k[j] for i in range(len(k)) for j in range(i + 1, len(k)))
        assert abs(2 * dim / n ** 2 - (1 - np.sum((k / n) ** 2))) < 1e-12
