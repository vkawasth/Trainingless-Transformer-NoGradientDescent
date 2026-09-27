"""Covariant perturbation split (corrected): loop eigenvalue factorisation, Layer-2 dependence on
magnitudes only, class dependence on phases only, and the true status of the CF coupling."""
import numpy as np
from amb_vigneaux.channels import noisy_shift, cf_of, cycle, channel_information
from amb_vigneaux.monodromy import loop_matrices, in_character_basis, operation_system, certificate_phase
from amb_vigneaux.holonomy import Graph


def test_loop_eigenvalue_factorises():
    rng = np.random.default_rng(0)
    G, p = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]), 5
    for _ in range(10):
        g = rng.integers(0, p, G.E); eta = rng.uniform(0, 0.9, G.E)
        Ms = loop_matrices(G, [noisy_shift(p, x, e) for x, e in zip(g, eta)])
        Mclean = loop_matrices(G, [noisy_shift(p, x, 0.0) for x in g])
        for M, Mc in zip(Ms, Mclean):
            D, Dc = np.diag(in_character_basis(M)), np.diag(in_character_basis(Mc))
            ratio = D[1:] / Dc[1:]                                  # the magnitude factor, same for all k != 0
            assert np.allclose(ratio, ratio[0]) and np.isclose(D[0], 1)


def test_layer2_ignores_phase_and_class_ignores_magnitude():
    p, G = 3, cycle(4)
    for g in ([1, 0, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0]):
        I = [channel_information(noisy_shift(p, x, 0.2)) for x in g]
        assert np.allclose(I, I[0])
    _, phi, _ = certificate_phase(*operation_system(G, [(1, 1), (1, 0), (1, 0), (1, 0)], p), p)
    assert phi[0] != 0                                               # class from the phases alone


def test_cf_coupling_status():
    # equal noise, non-flat: the proved cycle theorem
    for p in (3, 5):
        for L in (3, 4, 5):
            for e in (0.05, 0.2, 0.4):
                assert abs(cf_of(cycle(L), [noisy_shift(p, x, e) for x in [1] + [0] * (L - 1)]) - max(0, 1 - L * e / 2)) < 1e-8
    # Z/2, per-edge: exact, and a FLAT cycle can be contextual
    rng = np.random.default_rng(1)
    for L in (3, 4, 5):
        for _ in range(10):
            eta = rng.uniform(0, 1, L)
            flat = cf_of(cycle(L), [noisy_shift(2, 0, e) for e in eta])
            odd = cf_of(cycle(L), [noisy_shift(2, x, e) for x, e in zip([1] + [0] * (L - 1), eta)])
            assert abs(flat - max(0, (eta.max() - (eta.sum() - eta.max())) / 2)) < 1e-8
            assert abs(odd - max(0, 1 - eta.sum() / 2)) < 1e-8
    # Z/3, per-edge: the linear formula fails; one uniform edge gives 1 - 1/p, flat or not
    for g in ([0, 0, 0], [1, 0, 0]):
        assert abs(cf_of(cycle(3), [noisy_shift(3, x, e) for x, e in zip(g, [1, 0, 0])]) - 2 / 3) < 1e-6
