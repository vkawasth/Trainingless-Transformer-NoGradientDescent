import itertools
import numpy as np

from amb_vigneaux import blackwell as B


def _joint(seed=0, n=4):
    p = np.random.default_rng(seed).dirichlet(np.ones(2 ** (n + 1)))
    return p, n


def test_supersets_dominate_and_value_is_monotone():
    p, n = _joint()
    R = np.array([[1.0, -0.5], [0.2, 1.0], [0.6, 0.6]])
    for U in itertools.combinations(range(n), 2):
        for extra in range(n):
            if extra in U:
                continue
            V = tuple(sorted(U + (extra,)))
            mu, nu = B.standard_measure(p, U, n), B.standard_measure(p, V, n)
            assert B.blackwell_leq(mu, nu)
            assert B.value(mu, R) <= B.value(nu, R) + 1e-12


def test_frontier_contains_every_optimum():
    p, n = _joint(1, 5); cost = np.random.default_rng(2).uniform(0.5, 2, n)
    regs = [U for r in range(n + 1) for U in itertools.combinations(range(n), r)]
    M = [B.standard_measure(p, U, n) for U in regs]; c = np.array([cost[list(U)].sum() for U in regs])
    F, _ = B.frontier(M, c, [B.mutual_information(m, M[0][0][0]) for m in M])
    R = np.array([[1.0, 0.0], [0.0, 1.0]]); vals = np.array([B.value(m, R) for m in M])
    for price in np.linspace(0, 0.3, 13):
        best = (vals - price * c).max()
        assert max((vals - price * c)[F]) >= best - 1e-12


def test_parity_is_invisible_to_subsets():
    # Theta = X1 xor X2 xor X3: every proper subset is uninformative
    X = B.cells(4); p = np.array([1.0 if x[0] == (x[1] ^ x[2] ^ x[3]) else 0.0 for x in X]); p /= p.sum()
    acc = np.eye(2)
    for U in [(0,), (0, 1), (1, 2)]:
        assert abs(B.value(B.standard_measure(p, U, 3), acc) - 0.5) < 1e-12
    assert abs(B.value(B.standard_measure(p, (0, 1, 2), 3), acc) - 1.0) < 1e-12


def test_contextual_region_has_no_value():
    ctx = [(0, 1), (1, 2), (2, 0)]; eq = np.array([.5, 0, 0, .5]); ne = np.array([0, .5, .5, 0])
    assert B.robust_value(3, ctx, [eq, eq, ne], 0, (1, 2), np.eye(2)) is None
    u = np.full(4, .25); lo, hi = B.robust_value(3, ctx, [u, u, u], 0, (1, 2), np.eye(2))
    assert lo <= 0.5 + 1e-9 <= hi + 1e-9


def test_strassen_lp_matches_garbling_for_three_classes():
    rng = np.random.default_rng(5); prior = np.array([0.5, 0.3, 0.2])
    L2 = rng.dirichlet(np.ones(6), size=3)                        # 3 x 6 likelihood matrix
    M = rng.dirichlet(np.ones(3), size=6)                         # garbling 6 -> 3
    L1 = L2 @ M
    assert B.leq_k(L1, L2, prior)                                 # a garbling is less informative
    assert not B.leq_k(L2, L1, prior) or np.allclose(B.value_k(L1, prior, np.eye(3)), B.value_k(L2, prior, np.eye(3)))


def test_product_preserves_order_and_growth_is_exact():
    rng = np.random.default_rng(6); prior = np.array([0.5, 0.5])
    L2 = rng.dirichlet(np.ones(4), size=2); L1 = L2 @ rng.dirichlet(np.ones(2), size=4); W = rng.dirichlet(np.ones(3), size=2)
    assert B.leq_k(B.product(L1, W), B.product(L2, W), prior)
    blocks = [[(0.0, np.ones((2, 1)), ()), (1.0, L1, ("a",)), (1.0, L2, ("b",))], [(0.0, np.ones((2, 1)), ()), (2.0, W, ("w",))]]
    F, _, _ = B.grow_frontier(blocks, prior)
    labels = {f[2] for f in F}
    assert ("a",) not in labels and ("a", "w") not in labels           # the garbled option never survives


def test_dependent_blocks_break_growth():
    # N noise, C = theta xor N: {N} <= {B} but {N, C} reveals theta
    X = B.cells(4); p = np.array([0.25 * (0.6 if bw == t else 0.4) * (1.0 if c == (t ^ nn) else 0.0) for t, nn, bw, c in X]); p /= p.sum()
    prior = np.array([0.5, 0.5]); LN, _ = B.experiment(p, (0,), 3); LB, _ = B.experiment(p, (1,), 3)
    LNC, _ = B.experiment(p, (0, 2), 3); LBC, _ = B.experiment(p, (1, 2), 3)
    assert B.leq_k(LN, LB, prior) and not B.leq_k(LNC, LBC, prior)
