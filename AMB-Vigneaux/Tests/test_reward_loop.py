import numpy as np

from amb_vigneaux import reward_loop as RL

GRID = np.linspace(0, 1, 41)


def test_value_and_policy_iteration_agree():
    K = RL.belief_kernel(GRID, 49, 25.0, 0.02, 0.10, nq=200)
    assert np.allclose(K.sum(1), 1)
    for mu in (0.0, 5.0):
        R = RL.rewards(GRID, 1.0, 5.0, 0.5, mu)
        pv, Vv, _ = RL.value_iteration(K, R, 0.95)
        pp, Vp, _ = RL.policy_iteration(K, R, 0.95)
        assert np.array_equal(pv, pp) and np.allclose(Vv, Vp, atol=1e-6)


def test_policy_has_hysteresis_with_switching_cost():
    K = RL.belief_kernel(GRID, 49, 25.0, 0.02, 0.10, nq=200)
    pol, _, _ = RL.value_iteration(K, RL.rewards(GRID, 1.0, 5.0, 2.0, 0.0), 0.95)
    # a kept topic is kept up to a higher belief than a dropped topic is re-admitted
    keep_cut = GRID[np.argmin(pol[:, 1])]; readmit_cut = GRID[np.argmin(pol[:, 0])]
    assert keep_cut >= readmit_cut


def test_filter_is_bayes_rule():
    rng = np.random.default_rng(0); b = 0.3; Q = 70.0
    f0, f1 = RL.lik(Q, 49, 25.0); bp = RL.predict(b, 0.02, 0.1)
    assert abs(RL.update(bp, Q, 49, 25.0) - bp * f1 / (bp * f1 + (1 - bp) * f0)) < 1e-12
