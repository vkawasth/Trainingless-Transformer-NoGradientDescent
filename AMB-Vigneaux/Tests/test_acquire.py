import numpy as np

from amb_vigneaux import acquire as A

X3 = A.cells(3)
UNAN = (X3.min(1) == X3.max(1)).astype(float)
ALL1 = X3.prod(1).astype(float)
PAIRS = [(0, 1), (1, 2), (0, 2)]


def test_identification_rank_test():
    assert A.identified(3, PAIRS, UNAN)
    assert not A.identified(3, PAIRS, ALL1)
    assert A.identified(3, PAIRS + [(0, 1, 2)], ALL1)
    best = A.cheapest_identifying(3, PAIRS + [(0, 1, 2)], [1, 1, 1, 4], ALL1, have=(0, 1, 2))
    assert best[1] == (3,)


def test_robust_bounds_contain_truth_and_shrink():
    rng = np.random.default_rng(0); p = rng.dirichlet(np.ones(8)); Rs = [A.restriction(3, c) for c in PAIRS]
    lo0, hi0 = A.robust_bounds(3, Rs, [R @ p for R in Rs], [0.0] * 3, ALL1)
    assert lo0 - 1e-9 <= ALL1 @ p <= hi0 + 1e-9
    lo1, hi1 = A.robust_bounds(3, Rs, [R @ p for R in Rs], [0.05] * 3, ALL1)
    assert lo1 <= lo0 + 1e-9 and hi1 >= hi0 - 1e-9


def test_copt_shares_and_unidentified():
    p = np.full(8, 1 / 8); Rs = [A.restriction(3, c) for c in PAIRS]
    sh, _ = A.c_optimal(3, Rs, [1, 1, 1], UNAN, p)
    assert sh is not None and abs(sh.sum() - 1) < 1e-9
    sh2, v = A.c_optimal(3, Rs, [1, 1, 1], ALL1, p)
    assert sh2 is None and v == np.inf


def test_sequential_decision_is_correct_when_far():
    rng = np.random.default_rng(3); p = rng.dirichlet(np.ones(8) * 2); truth = UNAN @ p
    prob = A.Problem(3, PAIRS + [(0, 1, 2)], [1, 1, 1, 4], [0, 1, 2], UNAN, truth - 0.2)
    r = A.run(prob, p, "bounds-first", rng=rng)
    assert r["stopped"] and r["decision"] is True
