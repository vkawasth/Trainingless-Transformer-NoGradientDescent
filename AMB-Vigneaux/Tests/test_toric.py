import io, contextlib
import numpy as np
from amb_vigneaux import toric as T
from amb_vigneaux.strata import design


def test_birch_point_is_ipf_and_theta_is_monotone():
    rng = np.random.default_rng(0)
    for n in (3, 4):
        assert T.binomial(n)["degree"] == 2 ** (n - 1)
        p = rng.dirichlet(np.ones(2 ** n)); Pi = T.point_from_margins(p, n); Pb, _ = T.birch(Pi)
        assert np.abs(Pb - Pi).max() < 1e-10 and abs(T.theta(Pi)) < 1e-10
        F = T.fibre(Pi); ts = np.linspace(F["t_lo"], F["t_hi"], 30)[1:-1]
        assert np.all(np.diff([T.theta(Pi + t * T.chi(n)) for t in ts]) > 0)
        # every point of the segment has the same (n-1)-margins
        mg = T.margins_of(Pi, n); mg2 = T.margins_of(Pi + ts[3] * T.chi(n), n)
        assert all(np.allclose(mg[m], mg2[m]) for m in mg)


def test_theta_hat_recovers_a_planted_interaction():
    rng = np.random.default_rng(1); n = 3; th0 = 0.4
    logp = rng.normal(size=8) * 0.3; logp += (th0 - T.chi(n) @ logp / 8) * T.chi(n)
    p = np.exp(logp); p /= p.sum(); assert abs(T.theta(p) - th0) < 1e-12
    est = [T.theta_hat(rng.multinomial(20000, p))["theta"] for _ in range(200)]
    se = T.theta_hat(rng.multinomial(20000, p))["se"]
    assert abs(np.mean(est) - th0) < 3 * se / np.sqrt(200) + 0.01 and 0.7 < np.std(est) / se < 1.3


def test_single_cells_are_never_cofacial_and_higher_pr_fibre_is_empty():
    rng = np.random.default_rng(2); p = rng.dirichlet(np.ones(8) * 3)
    with contextlib.redirect_stdout(io.StringIO()):
        S = T.strata_report(p, cells_idx=[0, 5])
    assert all(len(s["forced"]) >= 2 and s["radius"] >= s["saturated"] for s in S)
    A, _ = design((2,) * 4, T.margins(4))
    tri = np.array([1.0 if sum(x) % 2 == 0 else 0.0 for x in T.cells(3)]) / 4
    b = np.concatenate([tri] * 4); P0, *_ = np.linalg.lstsq(A, b, rcond=None)
    assert np.abs(A @ P0 - b).max() < 1e-12 and not T.fibre(P0)["nonempty"]
