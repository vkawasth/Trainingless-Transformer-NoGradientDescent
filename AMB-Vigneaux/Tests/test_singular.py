import numpy as np
from amb_vigneaux import singular as S


def test_flop_posterior_certainty():
    ev, od = [0, 3, 5, 6], [1, 2, 4, 7]
    clear = np.array([5, 400, 300, 300, 300, 300, 300, 300])           # one cheap even, one cheap odd (index 1? no: 0 even, 2 odd)
    clear[2] = 4
    tie = np.array([5, 300, 6, 300, 6, 300, 300, 300])                  # two odd cells nearly equal
    assert max(S.flop_posterior(clear, ev, od)["pairs"].values()) > 0.95
    assert max(S.flop_posterior(tie, ev, od)["pairs"].values()) < 0.8


def test_learning_coefficient_separates_singular_from_regular():
    rng = np.random.default_rng(3); n = 2000
    lam = {}
    for name, a, b in (("singular", 0.0, 0.0), ("regular", 0.3, 1.0)):
        z = rng.random(n) < a; r = rng.normal(0, 1, n) + b * z
        lam[name] = S.wbic_lambda(S.mix_loglik(r, np.ones(n)), S.mix_logprior(2.0), [0.1, 0.5], n, steps=8000, burn=2000, rng=rng)["lam"]
    assert lam["singular"] < 0.75 < lam["regular"]


def test_thermodynamic_integration_matches_conjugate_evidence():
    rng = np.random.default_rng(4); y = rng.normal(0.5, 1, 50); n = len(y)
    ll = lambda th: float(-0.5 * ((y - th[0]) ** 2).sum() - 0.5 * n * np.log(2 * np.pi))
    lp = lambda th: float(-0.5 * th[0] ** 2 - 0.5 * np.log(2 * np.pi))
    exact = -0.5 * n * np.log(2 * np.pi) - 0.5 * np.log(n + 1) - 0.5 * ((y ** 2).sum() - y.sum() ** 2 / (n + 1))
    est, _, _ = S.log_evidence_ti(ll, lp, [0.0], steps=6000, burn=1500, scale=0.3, rng=rng)
    assert abs(est - exact) < 0.3


def test_lca_em_likelihood_increases_with_classes():
    rng = np.random.default_rng(5); c = rng.multinomial(1000, rng.dirichlet(np.ones(16)))
    ll = [S.lca_em(c, K, 4, starts=5, rng=rng)["loglik"] for K in (1, 2, 3)]
    assert ll[0] <= ll[1] + 1e-6 <= ll[2] + 2e-6


def test_lone_node_extra_symbols_are_free_window_is_not():
    rng = np.random.default_rng(6)
    c1 = rng.multinomial(2000, rng.dirichlet(np.ones(16)))
    l1 = [S.cat_lca_em(c1, K, 1, 16, starts=3, rng=rng)["loglik"] for K in (1, 2)]
    assert abs(l1[0] - l1[1]) < 1e-6                         # a mixture of single categorical draws is one categorical
    w = np.array([.5, .5]); Q = rng.dirichlet(np.ones(4), size=(2, 3)); X = S.cat_cells(3, 4)
    lik = np.ones((2, len(X)))
    for j in range(3):
        lik *= Q[:, j, X[:, j]]
    p = w @ lik; assert abs(p.sum() - 1) < 1e-12
    c3 = rng.multinomial(5000, p)
    l3 = [S.cat_lca_em(c3, K, 3, 4, starts=5, rng=rng)["loglik"] for K in (1, 2)]
    assert l3[1] - l3[0] > 50                                 # the window identifies the two parent symbols
