import numpy as np
from amb_vigneaux.dynamics import natural_split


def test_natural_change_has_no_loop_change():
    rng = np.random.default_rng(0)
    eta_s, eta_t = rng.normal(size=4), rng.normal(size=3)
    delta = eta_s[:, None] + eta_t[None, :]
    r = natural_split(delta)
    assert np.abs(r["non_natural"]).max() < 1e-10
    assert max(abs(v) for v in r["delta_phi"].values()) < 1e-10


def test_selective_change_is_non_natural():
    delta = np.zeros((2, 2)); delta[1, 0] = 1.0            # source 1 shifts on topic 0 only
    r = natural_split(delta)
    assert abs(r["delta_phi"][(0, 1, 0, 1)] + 1.0) < 1e-12
    assert np.abs(r["non_natural"]).max() > 0.2
