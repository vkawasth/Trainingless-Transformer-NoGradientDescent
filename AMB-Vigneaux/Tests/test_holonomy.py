import numpy as np
import pytest

from amb_vigneaux.holonomy import (Graph, chernoff_class_error, edge_design, estimate_edges,
                                   field_fit_estimate, identifiable_loops, n_required,
                                   observe_edges, power_theory, reduce,
                                   tree_section_design)
from amb_vigneaux.holonomy import holonomy_test as wald_test

ROOM = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
LOOPS = Graph(8, [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)])


def _flat(G, rng, group):
    c = rng.random(G.n_vertices) if group in ("R", "R/Z") else rng.integers(0, group, G.n_vertices)
    return reduce(G.coboundary(c), group)


def test_room_matches_user_holonomy_script():
    """holonomy.py: consistent / one loop broken / both broken, Z/4096."""
    for ops, has_field in (([100, 250, -90, -260, 350], True), ([100, 250, -90, -260, 351], False),
                           ([100, 250, -88, -260, 351], False)):
        assert (ROOM.field_by_propagation(ops, 4096) is not None) == has_field
        assert bool(np.all(ROOM.holonomy(ops, 4096) == 0)) == has_field


@pytest.mark.parametrize("group", ["R/Z", 7, 4096])
def test_holonomy_vs_bruteforce(group):
    rng = np.random.default_rng(0)
    for G in (ROOM, LOOPS):
        for _ in range(50):
            g = rng.random(G.E) if group == "R/Z" else rng.integers(0, group, G.E)
            if rng.random() < 0.5:
                g = _flat(G, rng, group)
            assert (G.field_by_propagation(g, group) is not None) == bool(np.all(
                np.abs(G.holonomy(g, group)) < 1e-9))


def test_identifiability_table():
    table = {grp: (identifiable_loops(ROOM, edge_design(ROOM), grp),
                   identifiable_loops(ROOM, tree_section_design(ROOM), grp),
                   identifiable_loops(ROOM, 2 * ROOM.C, grp)) for grp in ("R", "R/Z", 3, 4)}
    assert table["R"] == ([True] * 2, [False] * 2, [True] * 2)
    assert table["R/Z"] == ([True] * 2, [False] * 2, [False] * 2)
    assert table[3] == ([True] * 2, [False] * 2, [True] * 2)
    assert table[4] == ([True] * 2, [False] * 2, [False] * 2)


def test_non_identifiability_witnesses():
    """Necessity, by explicit pairs of worlds with identical observations."""
    for i, k in enumerate(ROOM.cotree):
        # tree-section design never reads a cotree edge: shift it by t
        gp = np.zeros(ROOM.E); gp[k] = 0.3
        assert np.allclose(reduce(tree_section_design(ROOM) @ gp, "R/Z"), 0)
        assert abs(reduce(ROOM.C[i] @ gp, "R/Z")) > 0.1
        # loop observed twice over R/Z: shift its cotree edge by 1/2
        gp = np.zeros(ROOM.E); gp[k] = 0.5
        assert np.allclose(reduce(2 * ROOM.C @ gp, "R/Z"), 0)
        assert abs(reduce(ROOM.C[i] @ gp, "R/Z")) == pytest.approx(0.5)
        # same over Z/4 with shift 2
        gp = np.zeros(ROOM.E, int); gp[k] = 2
        assert np.all(reduce(2 * ROOM.C @ gp, 4) == 0) and reduce(ROOM.C[i] @ gp, 4) == 2


def test_field_fit_estimator_is_blind():
    rng = np.random.default_rng(1)
    g = _flat(LOOPS, rng, "R/Z").astype(float)
    g[LOOPS.cotree[0]] += 0.2 * LOOPS.C[0, LOOPS.cotree[0]]
    est = estimate_edges(observe_edges(g, 100, "R/Z", 0.05, rng), "R/Z")
    assert abs(LOOPS.holonomy(est.g, "R/Z")[0] - 0.2) < 0.05
    assert np.allclose(LOOPS.holonomy(field_fit_estimate(LOOPS, est, "R/Z"), "R/Z"), 0)


@pytest.mark.parametrize("sigma,n", [(0.05, 50), (0.3, 30)])
def test_chi2_calibrated_including_wrapping(sigma, n):
    rng = np.random.default_rng(2)
    rej = [wald_test(LOOPS, observe_edges(_flat(LOOPS, rng, "R/Z"), n, "R/Z", sigma, rng),
                         "R/Z").p_chi2 < 0.05 for _ in range(400)]
    assert 0.02 < np.mean(rej) < 0.09


def test_naive_se_is_liberal_under_wrapping():
    rng = np.random.default_rng(3)
    rej = [wald_test(LOOPS, observe_edges(_flat(LOOPS, rng, "R/Z"), 20, "R/Z", 0.3, rng),
                         "R/Z", se_kind="small").p_chi2 < 0.05 for _ in range(200)]
    assert np.mean(rej) > 0.3


def test_power_and_n_required():
    rng = np.random.default_rng(4)
    h = np.array([0.05, 0.0])
    n = n_required(LOOPS, h, 0.1)
    n6 = n_required(LOOPS, np.array([0.0, 0.05]), 0.1)
    assert n6 == pytest.approx(2 * n, rel=0.02)          # cost ∝ loop length
    acc = []
    for _ in range(300):
        g = _flat(LOOPS, rng, "R/Z").astype(float)
        g[LOOPS.cotree[0]] += 0.05 * LOOPS.C[0, LOOPS.cotree[0]]
        acc.append(wald_test(LOOPS, observe_edges(g, n, "R/Z", 0.1, rng), "R/Z").p_chi2 < 0.05)
    assert 0.72 < np.mean(acc) < 0.88


def test_chernoff_bound_holds():
    rng = np.random.default_rng(5)
    for m in (40, 80):
        err = np.mean([np.any(ROOM.holonomy(estimate_edges(observe_edges(
            _flat(ROOM, rng, 5), m, 5, 0.7, rng), 5).g, 5) != 0) for _ in range(1500)])
        assert err <= chernoff_class_error(ROOM, 5, 0.7, m)
