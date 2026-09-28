import itertools
import numpy as np
from amb_vigneaux.channels import cf_of, cycle, Graph, edge_family
from amb_vigneaux.outcome import analyse_outcomes


def K(e):
    p = (1 + e) / 2
    return np.array([[p, 1 - p], [1 - p, p]])


def test_one_weak_edge_cf_is_half_deficit():
    for L in (3, 4, 5):
        for e in (1, 0.5, 0, -0.5, -1):
            assert abs(cf_of(cycle(L), [K(1)] * (L - 1) + [K(e)]) - (1 - e) / 2) < 1e-9


def test_formula_random_cycles():
    rng = np.random.default_rng(0)
    for L in (3, 4, 5, 6):
        for _ in range(40):
            E = rng.uniform(-0.2, 1, L)
            best = max(sum((-1 if i in S else 1) * E[i] for i in range(L))
                       for k in range(1, L + 1, 2) for S in itertools.combinations(range(L), k))
            assert abs(cf_of(cycle(L), [K(e) for e in E]) - max(0, (best - (L - 2)) / 2)) < 1e-9


def test_uniform_weakening_and_tree_are_noncontextual():
    assert cf_of(cycle(4), [K(0.5)] * 4) < 1e-9
    path = Graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    assert cf_of(path, [K(1)] * 3 + [K(-1)]) < 1e-9


def test_amb_localises_weak_edge():
    r = analyse_outcomes(edge_family(cycle(4), [K(1)] * 3 + [K(0)]))
    assert not r.strongly_contextual and r.n_global_sections == 2
    assert {w[0] for w in r.cohomological_witnesses} == {("c3", "c0")}
