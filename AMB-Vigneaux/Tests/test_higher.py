import itertools, json
import numpy as np
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.outcome import contextual_fraction
from amb_vigneaux.bounds import loop_space, functional
from amb_vigneaux.higher import betti, path_2groupoid, complex_instrument


def test_boundary_of_simplex_covers_are_spheres_with_top_parity_free():
    for n in (3, 4, 5):
        M = [f"x{i}" for i in range(n)]; ctx = list(itertools.combinations(M, n - 1))
        b = betti(ctx); assert b[0] == 1 and b[-1] == 1 and all(v == 0 for v in b[1:-1]) and len(b) == n - 1
        sc = Scenario({m: (0, 1) for m in M}, tuple(ctx)); K = loop_space(sc)
        chi = functional(sc, lambda g: (-1.0) ** sum(g.values()))
        assert K.shape[1] == 1 and abs(abs(np.corrcoef(K[:, 0], chi)[0, 1]) - 1) < 1e-9


def test_tetrahedral_loops_are_all_filled_triangle_loop_is_not():
    g = path_2groupoid(list(itertools.combinations("abcd", 3)))
    assert g["graph_beta1"] == 3 and g["filled"] == 3 and len(g["two_cells"]) == 4
    g = path_2groupoid([("L", "C"), ("C", "R"), ("L", "R")])
    assert g["graph_beta1"] == 1 and g["filled"] == 0


def test_higher_pr_box_is_invisible_to_pairs():
    M = "abcd"; sc = Scenario({m: (0, 1) for m in M}, tuple(itertools.combinations(M, 3)))
    pr = {C: np.array([1.0 if sum(s) % 2 == 0 else 0.0 for s in sc.sections(C)]) / 4 for C in sc.contexts}
    for lam in (0.2, 0.5, 1.0):
        e = EmpiricalModel(sc, {C: lam * pr[C] + (1 - lam) / 8 for C in sc.contexts})
        assert abs(contextual_fraction(e).value - max(0.0, 9 / 8 * (lam - 1 / 3))) < 1e-7
        for C in sc.contexts:                                   # every pair marginal is uniform: first order sees nothing
            for P in itertools.combinations(C, 2):
                assert np.allclose(sc.restriction_matrix(C, P) @ e.tables[C], 0.25)
    inst = complex_instrument(EmpiricalModel(sc, pr))
    json.dumps(inst)
    assert inst["betti"] == [1, 0, 1] and len(inst["faces"]) == 4 and inst["outer_face_filled"]
