import numpy as np
import pytest
from amb_vigneaux.twisted_info import cocycle_space_dim, reduced_dim, band_twisting_is_trivial

TWISTS = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 1, 0, 0], [1, 1, 1, 0]]


@pytest.mark.parametrize("alpha", [0.5, 2.0, 3.0])
def test_tsallis_is_the_only_cocycle_and_twist_invariant(alpha):
    for g in TWISTS:
        assert reduced_dim(4, 3, g, alpha) == (1, 0)


def test_cocycle_is_tsallis_with_one_global_constant():
    d, B, idx = cocycle_space_dim(4, 3, [1, 0, 0, 0], 2.0)
    v = B[0]; c0 = v[idx[("c", 0)][0]]
    for o, (a, b) in idx.items():
        assert np.isclose(v[a], c0)                       # same K on every observable
        assert np.allclose(v[a + 1:b], -c0)               # phi = K (1 - sum Q^alpha)


def test_alpha_one_linear_terms_die_shannon_survives():
    for g in TWISTS:
        assert reduced_dim(4, 3, g, 1.0) == (0, 0)             # expectations are not cocycles (XX = X)
        assert reduced_dim(4, 3, g, 1.0, logs=True) == (1, 0)  # only Shannon, label-blind


def test_z2_odd_cycle():
    assert reduced_dim(3, 2, [1, 0, 0], 1.0, logs=True) == (1, 0)
    assert reduced_dim(3, 2, [1, 0, 0], 2.0) == (1, 0)


def test_band():
    assert band_twisting_is_trivial()


from amb_vigneaux.twisted_info import pointwise_cocycles, orbit_count


def _exact(sv):
    return int((sv < 1e-7).sum())


@pytest.mark.parametrize("n,g", [(2, [0, 0, 0, 0]), (2, [1, 0, 0, 0]), (3, [0, 0, 0, 0]), (3, [1, 0, 0, 0])])
def test_pointwise_full_support_only_surprisal(n, g):
    assert _exact(pointwise_cocycles(4, n, g, "full", samples=10, spectrum=True)) == 1


def test_pointwise_graph_stratum_reads_holonomy_orbits():
    k = {}
    for g in ([0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]):
        d = _exact(pointwise_cocycles(4, 3, g, "graph", samples=10, spectrum=True))
        k[tuple(g)] = d / orbit_count(3, sum(g))
    assert len(set(k.values())) == 1 and list(k.values())[0] == 7      # dim Z^1 = 7 x #orbits


def test_pointwise_soft_signal_near_stratum():
    for g, orbits in (([0, 0, 0, 0], 3), ([1, 0, 0, 0], 1)):
        sv = pointwise_cocycles(4, 3, g, "graph+eps", eps=3e-4, samples=12, spectrum=True)
        assert _exact(sv) == 1
        assert int(((sv >= 1e-7) & (sv < 20 * 3e-4)).sum()) == orbits


def test_tsallis_derivative_at_one_is_second_moment_of_surprisal():
    rng = np.random.default_rng(1)
    for _ in range(5):
        p = rng.dirichlet(np.ones(5)); h = 1e-3
        S = lambda a: (1 - (p ** a).sum()) / (a - 1)
        d = (S(1 + h) - S(1 - h)) / (2 * h)
        assert abs(d - (-0.5 * (p * np.log(p) ** 2).sum())) < 1e-5


def test_escort_monoid():
    p = np.random.default_rng(2).dirichlet(np.ones(6))
    E = lambda a, q: q ** a / (q ** a).sum()
    assert np.allclose(E(1.7, E(0.6, p)), E(1.7 * 0.6, p))


def test_loop_complex_and_comparison_map():
    """V0 = free R[Z/n]-module of rank r; h = shift.  ker(h-1) = coker(h-1) (loop complex, Euler char 0);
    the non-admissible part coker(V0^h -> V0) = im(h-1) has dim dim V0 (1 - 1/ord h)."""
    from math import gcd
    n, r = 6, 3
    for hh in range(n):
        S = np.roll(np.eye(n), hh, axis=1)
        H = np.kron(np.eye(r), S)
        rk = np.linalg.matrix_rank(H - np.eye(n * r))
        ker, coker, im = n * r - rk, n * r - rk, rk
        order = n // gcd(n, hh) if hh else 1
        assert ker == coker == (n * r) // order
        assert im * order == n * r * (order - 1)
