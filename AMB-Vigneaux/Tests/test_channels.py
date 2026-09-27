import itertools
import numpy as np
import pytest

from amb_vigneaux.channels import (birkhoff_noise, cf_noisy_cycle, cf_of, channel_information,
                                   character_eigenvalue, cycle, dual_certificate, edge_family,
                                   holonomy_from_spectrum, loop_operator, loop_spectrum, loop_walk,
                                   noisy_shift, shift)
from amb_vigneaux.holonomy import Graph, reduce
from amb_vigneaux.outcome import cohomological_obstruction

ROOM = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])


def _bent(L, h):
    g = np.zeros(L, int); g[0] = h; return g


def test_loop_walk_reproduces_group_holonomy():
    rng = np.random.default_rng(0)
    n = 7
    for _ in range(20):
        g = rng.integers(0, n, ROOM.E)
        hol = ROOM.holonomy(g, n)
        for i in range(ROOM.beta1):
            M = loop_operator(ROOM, [shift(n, x) for x in g], loop_walk(ROOM, i))
            assert np.allclose(M, shift(n, hol[i]))


@pytest.mark.parametrize("n,L", [(2, 3), (2, 6), (3, 4), (5, 3), (5, 5)])
def test_cf_closed_form(n, L):
    G = cycle(L)
    for h in {1, n - 1}:
        for eps in (0.05, 0.2, 0.35, 0.6):
            K = [noisy_shift(n, x, eps) for x in _bent(L, h)]
            assert cf_of(G, K) == pytest.approx(cf_noisy_cycle(L, eps), abs=1e-7)
        assert cf_of(G, [noisy_shift(n, 0, 0.2) for _ in range(L)]) < 1e-9   # flat control


@pytest.mark.parametrize("n,L,h", [(2, 3, 1), (3, 4, 2), (5, 3, 2), (5, 4, 1), (4, 3, 2)])
def test_dual_certificate_is_feasible_and_tight(n, L, h):
    """Every colouring's discrepancies sum to −h; the certificate covers each."""
    w = dual_certificate(n, h)
    for d in itertools.product(range(n), repeat=L):
        if sum(d) % n == (-h) % n:
            assert sum(w[x] for x in d) >= 1 - 1e-12
    eps = 0.1
    pd = np.full(n, eps / n); pd[0] = 1 - eps + eps / n
    assert L * (pd @ w) == pytest.approx(L * eps / 2)


def test_per_edge_information_is_blind_to_the_permutation():
    rng = np.random.default_rng(1)
    for _ in range(10):
        N = birkhoff_noise(6, 3, rng)
        P = np.eye(6)[rng.permutation(6)]
        assert channel_information(N @ P) == pytest.approx(channel_information(N))
        assert channel_information(P @ N) == pytest.approx(channel_information(N))


def test_covariant_noise_spectrum_carries_holonomy_exactly():
    n, L = 5, 4
    G = cycle(L)
    for h in range(n):
        for eps in (0.0, 0.3, 0.9):
            K = [noisy_shift(n, x, eps) for x in _bent(L, h)]
            M = loop_operator(G, K, loop_walk(G, 0))
            lam = character_eigenvalue(M, 1)
            assert abs(lam) == pytest.approx((1 - eps) ** L)
            r = holonomy_from_spectrum(M)
            assert min(abs(r - h), n - abs(r - h)) < 1e-9
            # loop-composite information does not depend on h
            K0 = [noisy_shift(n, 0, eps) for _ in range(L)]
            assert channel_information(M) == pytest.approx(
                channel_information(loop_operator(G, K0, loop_walk(G, 0))))


def test_gamma_dies_under_any_noise():
    n, L = 5, 3
    G = cycle(L)
    for eps, expect in ((0.0, 5), (1e-6, 0)):
        e = edge_family(G, [noisy_shift(n, x, eps) for x in _bent(L, 2)])
        sc, S = e.scenario, e.support(1e-12)
        got = sum(not cohomological_obstruction(sc, S, sc.contexts[0], s).obstruction_vanishes
                  for s in S[sc.contexts[0]])
        assert got == expect


def test_spectrum_is_base_point_invariant_for_lossy_channels():
    rng = np.random.default_rng(2)
    n, L = 5, 4
    G = cycle(L)
    for _ in range(10):
        K = [birkhoff_noise(n, 3, rng) for _ in range(L)]
        walk = loop_walk(G, 0)
        ref = np.sort_complex(np.round(np.linalg.eigvals(loop_operator(G, K, walk)), 8))
        for r in range(1, L):
            rot = walk[r:-1] + walk[:r + 1]
            got = np.sort_complex(np.round(np.linalg.eigvals(loop_operator(G, K, rot)), 8))
            assert np.allclose(ref, got, atol=1e-6)


def test_noncovariant_noise_makes_flat_systems_contextual():
    rng = np.random.default_rng(1)
    n, L, eps = 5, 3, 0.5
    G = cycle(L)
    flat_cf = []
    for _ in range(4):
        Ns = [birkhoff_noise(n, 4, rng) for _ in range(L)]
        flat_cf.append(cf_of(G, [((1 - eps) * np.eye(n) + eps * N) for N in Ns]))
    assert max(flat_cf) > 0.05


# ---------------------------------------------- several loops; general abelian groups
from amb_vigneaux.channels import (cf_noisy_graph, character, character_value, convolution,
                                   group_elements, holonomy_girth, translation)


def _theta(lengths):
    edges, nv = [], 2
    for Lp in lengths:
        prev = 0
        for _ in range(Lp - 1):
            edges.append((prev, nv)); prev = nv; nv += 1
        edges.append((prev, 1))
    return Graph(nv, edges)


GRAPHS = [ROOM, _theta((2, 3, 4)), _theta((2, 2, 5)),
          Graph(7, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 6), (6, 4)]),
          Graph(6, [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 5), (5, 2)])]


@pytest.mark.parametrize("gi", range(len(GRAPHS)))
def test_cf_is_governed_by_holonomy_girth(gi):
    G, n = GRAPHS[gi], 3
    for hol in itertools.product(range(n), repeat=G.beta1):
        if not any(hol):
            continue
        g = np.zeros(G.E, int)
        for i, k in enumerate(G.cotree):
            g[k] = hol[i] * G.C[i, k]
        for eps in (0.1, 0.4, 0.7):
            assert cf_of(G, [noisy_shift(n, x, eps) for x in g]) == pytest.approx(
                cf_noisy_graph(G, g, n, eps), abs=1e-7)


def test_girth_not_basis_decides():
    """θ(2,2,5): a flat 4-cycle and non-flat 7-cycles give ℓ_min = 7, not 4."""
    G = _theta((2, 2, 5))
    for hol in ((0, 1), (1, 0)):
        g = np.zeros(G.E, int)
        for i, k in enumerate(G.cotree):
            g[k] = hol[i] * G.C[i, k]
        lm = holonomy_girth(G, g, 3)
        assert lm in (4, 7)
        assert cf_of(G, [noisy_shift(3, x, 0.1) for x in g]) == pytest.approx(1 - lm * 0.05)


@pytest.mark.parametrize("orders", [(2, 2), (2, 4), (3, 3)])
def test_general_abelian_group(orders):
    rng = np.random.default_rng(3)
    els = group_elements(orders)
    N = len(els)
    G = cycle(3)
    h = els[1 + rng.integers(N - 1)]
    gs = [h, tuple(0 for _ in orders), tuple(0 for _ in orders)]
    # symmetric noise: phase of the character value is exactly χ(h)
    mu = rng.random(N); idx = {e: i for i, e in enumerate(els)}
    mu = mu + np.array([mu[idx[tuple((-a) % m for a, m in zip(e, orders))]] for e in els]); mu /= mu.sum()
    K = [convolution(orders, mu) @ translation(orders, x) for x in gs]
    M = loop_operator(G, K, loop_walk(G, 0))
    for k in els:
        val = character_value(orders, k, M)
        chi_h = character(orders, k)[idx[h]]
        mu_hat = character(orders, k).conj() @ mu      # real for symmetric μ
        assert val == pytest.approx(mu_hat ** 3 * chi_h, abs=1e-9)
    # the CF closed form holds verbatim for A = ⊕ Z/m_j
    for eps in (0.1, 0.3):
        Kn = [(1 - eps) * translation(orders, x) + eps / N for x in gs]
        assert cf_of(G, Kn) == pytest.approx(cf_noisy_cycle(3, eps), abs=1e-7)


def test_unlabelled_spectrum_only_sees_the_order_of_h():
    """On Z/5 every h ≠ 0 gives the same eigenvalue multiset; only the
    character-labelled eigenvalue reads h."""
    n, G = 5, cycle(3)
    specs, labelled = [], []
    for h in range(1, n):
        M = loop_operator(G, [noisy_shift(n, x, 0.2) for x in _bent(3, h)], loop_walk(G, 0))
        specs.append(np.sort_complex(np.round(np.linalg.eigvals(M), 9)))
        labelled.append(round(holonomy_from_spectrum(M)) % n)
    assert all(np.allclose(specs[0], s_) for s_ in specs)
    assert labelled == [1, 2, 3, 4]


def test_asymmetric_noise_contaminates_the_phase():
    n, G = 5, cycle(3)
    mu = np.array([0.7, 0.2, 0.0, 0.0, 0.1])
    C = np.array([np.roll(mu, x) for x in range(n)])
    M = loop_operator(G, [C @ shift(n, x) for x in (2, 0, 0)], loop_walk(G, 0))
    M0 = loop_operator(G, [C, C, C], loop_walk(G, 0))
    assert abs(holonomy_from_spectrum(M) - 2) > 0.1
    assert holonomy_from_spectrum(M) - holonomy_from_spectrum(M0) == pytest.approx(2)


def test_information_closed_forms():
    from amb_vigneaux.channels import loop_information, noisy_shift_information
    for n in (2, 3, 5, 8):
        for eps in (0.0, 0.1, 0.2, 0.7, 1.0):
            for g in (0, 1):
                assert channel_information(noisy_shift(n, g, eps)) == pytest.approx(noisy_shift_information(n, eps))
            for L in (3, 4):
                G = cycle(L)
                M = loop_operator(G, [noisy_shift(n, x, eps) for x in _bent(L, 1)], loop_walk(G, 0))
                assert channel_information(M) == pytest.approx(loop_information(n, eps, L))
                assert np.allclose(M, (1 - eps) ** L * shift(n, 1) + (1 - (1 - eps) ** L) / n)
    assert noisy_shift_information(5, 0.2) == pytest.approx(1.3676, abs=1e-4)


def test_entropy_is_not_a_function_of_fourier_moduli():
    """Homometric pair: same |FFT|, different entropy."""
    rng = np.random.default_rng(0)
    a, b = rng.dirichlet(np.ones(7)), rng.dirichlet(np.ones(7))
    conv = lambda u, v: np.real(np.fft.ifft(np.fft.fft(u) * np.fft.fft(v)))
    p, q = conv(a, b), conv(a, np.roll(b[::-1], 1))
    H = lambda x: float(-(x[x > 0] * np.log2(x[x > 0])).sum())
    assert np.allclose(np.abs(np.fft.fft(p)), np.abs(np.fft.fft(q)))
    assert abs(H(p) - H(q)) > 1e-3


def test_cf_not_determined_by_edge_information():
    """Same ε (hence identical per-edge I) but different ℓ_min ⇒ different CF."""
    n, eps = 3, 0.2
    cfs = []
    for L in (3, 5):
        G = cycle(L)
        cfs.append(cf_of(G, [noisy_shift(n, x, eps) for x in _bent(L, 1)]))
    assert cfs == pytest.approx([0.7, 0.5])
