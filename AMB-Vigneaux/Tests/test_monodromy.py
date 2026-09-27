import numpy as np
from amb_vigneaux.monodromy import *
from amb_vigneaux.holonomy import Graph
from amb_vigneaux.channels import shift
from amb_vigneaux import models, analyse_outcomes


def random_graph(rng, V=6, extra=4):
    edges = [(i, i + 1) for i in range(V - 1)]
    while len(edges) < V - 1 + extra:
        a, b = sorted(rng.choice(V, 2, replace=False))
        if (a, b) not in edges and (b, a) not in edges:
            edges.append((int(a), int(b)))
    return Graph(V, edges)


def test_abelian_rep_is_the_class_and_diagonal():
    rng = np.random.default_rng(0)
    for _ in range(10):
        G, p = random_graph(rng), 5
        g = rng.integers(0, p, G.E)
        Ms = loop_matrices(G, [shift(p, x) for x in g])
        h = G.holonomy(g, p)
        for M, hi in zip(Ms, h):
            D = in_character_basis(M)
            assert np.allclose(D, np.diag(np.diag(D)))
            ph = np.round(np.angle(np.diag(D)) * p / (2 * np.pi)) % p
            assert all(ph[k] in ((k * hi) % p, (-k * hi) % p) for k in range(p))
        assert all(np.allclose(commutator(A, B), np.eye(p)) for A in Ms for B in Ms)


def test_twisted_betti_formula():
    rng = np.random.default_rng(1)
    for _ in range(20):
        G, p = random_graph(rng), 5
        g = rng.integers(0, p, G.E)
        h = G.holonomy(g, p)
        for k in range(p):
            h0, h1 = twisted_betti(G, g, p, k)
            flat = all((k * x) % p == 0 for x in h)
            assert h0 == int(flat) and h1 == G.E - G.n_vertices + h0


def test_nonabelian_rep_sees_more_than_the_abelian_class():
    p, G = 5, Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    A = [(1, 0), (1, 1), (1, 0), (2, 0), (1, 0)]
    B = [(1, 0), (1, 0), (1, 0), (2, 0), (1, 0)]
    MA = loop_matrices(G, [affine_perm_matrix(a, b, p) for a, b in A])
    MB = loop_matrices(G, [affine_perm_matrix(a, b, p) for a, b in B])
    mult = lambda M: int((M.argmax(1)[1] - M.argmax(1)[0]) % p)
    assert [mult(M) for M in MA] == [mult(M) for M in MB]            # same abelianised class
    assert [round(np.trace(M)) for M in MA] != [round(np.trace(M)) for M in MB]
    assert not np.allclose(commutator(*MA), np.eye(p)) and np.allclose(commutator(*MB), np.eye(p))


def test_reflection_on_characters():
    p = 5
    sigma, _ = monomial_part(in_character_basis(affine_perm_matrix(p - 1, 0, p)))
    assert sigma == [(-k) % p for k in range(p)]
    sigma, _ = monomial_part(in_character_basis(affine_perm_matrix(1, 2, p)))
    assert sigma == list(range(p))


def test_grid_obstruction_is_invisible_to_loop_monodromy():
    p = 3
    specs = []
    for rr, cc in (((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 1))):
        G, K = grid_transports(p, rr, cc)
        specs.append([np.sort_complex(np.round(np.linalg.eigvals(M), 8)) for M in grid_loop_matrices(G, K)])
    assert all(np.allclose(a, b) for a, b in zip(*specs))
    assert analyse_outcomes(models.parity_grid((0, 0, 0), (0, 0, 0), p), with_cf=False).n_global_sections > 0
    assert analyse_outcomes(models.parity_grid((0, 0, 0), (0, 0, 1), p), with_cf=False).n_global_sections == 0


def test_certificate_phase_grid_and_locality():
    from amb_vigneaux.monodromy import grid_system, certificate_phase, continuous_phase, _rref_left_null
    p = 3
    for rr, cc in (((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 1)), ((1, 2, 0), (2, 1, 0)), ((1, 1, 1), (0, 0, 0))):
        A, b = grid_system(rr, cc)
        Y, phi, loc = certificate_phase(A, b, p)
        assert Y.shape[0] == 1 and (Y != 0).all()                   # one certificate, all six contexts
        assert (phi[0] != 0) == ((sum(rr) - sum(cc)) % p != 0)
        assert phi[0] == loc.sum() % p                               # sum of local contributions
    # every proper sub-family is consistent for EVERY local data: no smaller certificate exists
    A, _ = grid_system((0, 0, 0), (0, 0, 0))
    for C in range(6):
        assert _rref_left_null(np.delete(A, C, 0), p).shape[0] == 0
    # a local perturbation of one context moves the shadow continuously and alone
    A, b = grid_system((0, 0, 0), (0, 0, 1)); Y, _, _ = certificate_phase(A, b, p)
    bb = b.astype(float); bb[5] = 0.0
    assert abs(continuous_phase(Y[0], bb, p) - 1) < 1e-12


def test_certificate_phase_on_operation_systems():
    from amb_vigneaux.monodromy import operation_system, certificate_phase
    rng = np.random.default_rng(2)
    for _ in range(20):
        G, p = random_graph(rng), 5
        g = rng.integers(0, p, G.E)
        A, b = operation_system(G, [(1, int(x)) for x in g], p)
        Y, phi, _ = certificate_phase(A, b, p)
        assert Y.shape[0] == G.beta1                                  # certificates = cycle space
        assert (phi != 0).any() == (G.holonomy(g, p) != 0).any()
    G = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    _, phiB, _ = certificate_phase(*operation_system(G, [(1, 0), (1, 0), (1, 0), (2, 0), (1, 0)], 5), 5)
    _, phiA, _ = certificate_phase(*operation_system(G, [(1, 0), (1, 1), (1, 0), (2, 0), (1, 0)], 5), 5)
    assert (phiB == 0).all() and (phiA != 0).any()
