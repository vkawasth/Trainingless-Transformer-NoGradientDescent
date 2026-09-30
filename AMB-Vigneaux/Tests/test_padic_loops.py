import numpy as np
from amb_vigneaux.padic_loops import sum_claim_depth, vp, hierarchy_violations, random_tree, lca_depth


def test_even_cycle_depth_is_valuation_of_alternating_sum():
    for alt in (1, 3, 9, 27, 54):
        r12, r23, r41 = 5, 7, 2; r34 = alt - r12 + r23 + r41
        cl = [(0, 1, r12), (1, 2, r23), (2, 3, r34), (3, 0, r41)]
        assert sum_claim_depth(4, cl, 3, 6) == min(vp(alt, 3), 6)


def test_sign_twist_odd_cycles():
    for r in ((1, 2, 4), (3, 5, 1), (2, 4, 6)):
        cl = [(0, 1, r[0]), (1, 2, r[1]), (2, 0, r[2])]
        assert sum_claim_depth(3, cl, 3, 6) == 6                      # p odd: odd cycles always solvable
        assert (sum_claim_depth(3, cl, 2, 6) == 0) == (sum(r) % 2 == 1)  # p = 2: parity obstruction


def test_planted_digit_recovered():
    rng = np.random.default_rng(0); p, K = 5, 6
    x = rng.integers(0, p ** K, 8)
    cl = [(i, j, int(x[i] + x[j])) for i in range(8) for j in range(i + 1, 8)]
    assert sum_claim_depth(8, cl, p, K) == K
    for m in range(K):
        c2 = list(cl); i, j, r = c2[3]; c2[3] = (i, j, r + 2 * p ** m)
        assert sum_claim_depth(8, c2, p, K) == m


def test_hierarchy_consistent_and_nodewise_distortion_invisible():
    rng = np.random.default_rng(1)
    codes, arity = random_tree(4, rng)
    idx = rng.choice(len(codes), 40, replace=False)
    claims, shifted = [], []
    for _ in range(300):
        i, j = rng.choice(40, 2, replace=False)
        a, b = codes[idx[i]], codes[idx[j]]; d = lca_depth(a, b)
        claims.append((int(i), int(j), d, None))
        k = arity.get(tuple(a[:d]), 0)
        shifted.append((int(i), int(j), d + (k % 2), None))       # node-wise distortion
    assert hierarchy_violations(40, claims) == []
    assert hierarchy_violations(40, shifted) == []


def test_hierarchy_detects_unique_minimum_on_cycle():
    # triangle with depths 3, 3, 1: the shallow claim is the unique minimum -> violated
    bad = hierarchy_violations(3, [(0, 1, 3, None), (1, 2, 3, None), (0, 2, 1, None)])
    assert len(bad) == 1 and bad[0][2] == 1
