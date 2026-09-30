import numpy as np
from amb_vigneaux.giry import kleisli, bayes_inverse, pairwise_mi, coarse_grain, edges_at, hsic


def test_kleisli_is_associative_and_stochastic():
    rng = np.random.default_rng(0)
    A, B, C = (rng.dirichlet(np.ones(m), size=n) for n, m in ((3, 4), (4, 5), (5, 2)))
    assert np.allclose(kleisli(kleisli(A, B), C), kleisli(A, kleisli(B, C)))
    assert np.allclose(kleisli(A, B).sum(1), 1)


def test_block_posterior_is_bayesian_inversion_of_the_tree():
    # parent w -> (z_L, z_R) -> leaf pairs; posterior of z_L given ALL leaves = inside_L * outside_L (normalised)
    rng = np.random.default_rng(1); W, K, c = 2, 3, 4
    pw = rng.dirichlet(np.ones(W)); Th = rng.dirichlet(np.ones(K * K), size=W).reshape(W, K, K)
    th = rng.dirichlet(np.ones(c * c), size=K).reshape(K, c, c)
    x = (1, 2, 0, 3)
    joint = np.einsum("w,wab,a,b->wab", pw, Th, th[:, x[0], x[1]], th[:, x[2], x[3]])
    brute = joint.sum((0, 2)); brute /= brute.sum()
    inside_L, inside_R = th[:, x[0], x[1]], th[:, x[2], x[3]]
    outside_L = np.einsum("w,wab,b->a", pw, Th, inside_R)
    post = inside_L * outside_L; post /= post.sum()
    assert np.allclose(post, brute, atol=1e-15)
    # the same thing as a Bayesian inverse of the kernel z_L -> x_left with the prior 'outside'
    prior = outside_L / outside_L.sum()
    Kz = th.reshape(K, c * c)
    assert np.allclose(bayes_inverse(prior, Kz)[x[0] * c + x[1]], brute, atol=1e-15)
    # the inside vector does not depend on the other block's leaves (locality): only the posterior does
    inside_L2 = th[:, x[0], x[1]]
    assert np.array_equal(inside_L, inside_L2)


def test_mi_filtration_is_functorial_under_coarse_graining():
    rng = np.random.default_rng(2)
    for _ in range(20):
        J = rng.dirichlet(np.full(81, 0.3)).reshape(3, 3, 3, 3)
        M = pairwise_mi(J)
        Jc = coarse_grain(J, 1, [0, 0, 1])
        Mc = pairwise_mi(Jc)
        assert (Mc <= M + 1e-12).all()                                   # data processing
        for t in np.linspace(0, M.max(), 25):
            assert edges_at(Mc, t) <= edges_at(M, t)                     # inclusion of filtered complexes


def test_hsic_violates_data_processing():
    rng = np.random.default_rng(0); x = rng.normal(size=400); y = x + 0.5 * rng.normal(size=400)
    assert hsic(2 * np.sign(x), y) > hsic(x, y)                          # a coarse-graining INCREASES HSIC
    assert abs(hsic(10 * x, y) - hsic(x, y)) > 0.01                      # not even invariant under isomorphisms
