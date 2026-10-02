import itertools
import numpy as np

from amb_vigneaux.scenario import Scenario
from amb_vigneaux import bounds as Bd


def _rank(n, ctx, seed=0):
    sc = Scenario({m: (0, 1) for m in "ABCD"[:n]}, ctx); R = [sc.restriction_matrix(sc.measurements, C) for C in sc.contexts]
    p = np.random.default_rng(seed).dirichlet(np.ones(2 ** n) * 3); J = (np.diag(p) - np.outer(p, p))[:, 1:]
    F = sum((Rc @ J).T @ np.diag(1 / (Rc @ p)) @ (Rc @ J) for Rc in R); ev = np.linalg.eigvalsh(F)
    return int((ev > 1e-8 * ev.max()).sum()), Bd.loop_space(sc).shape[1]


def test_ridge_dimension_is_fisher_rank_drop():
    for n, ctx in ((3, [("A", "B"), ("B", "C"), ("C", "A")]), (4, [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]),
                   (4, list(itertools.combinations("ABCD", 3)))):
        rank, dimker = _rank(n, ctx)
        assert rank == 2 ** n - 1 - dimker
