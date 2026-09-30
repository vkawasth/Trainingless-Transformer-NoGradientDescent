import itertools
import numpy as np
from amb_vigneaux.hull import hull_tree, coalition, disc_profile, obstruction_profile, padic_dist, coalition_test
from amb_vigneaux.padic_loops import sum_claim_depth


def test_filled_cell_linear_invariant_zero_but_tree_informative():
    x = [0.0, 0.1, 2.0]
    assert abs((x[0] - x[1]) + (x[1] - x[2]) + (x[2] - x[0])) < 1e-15      # holonomy inside the cell
    assert coalition(x) == (0, 1)
    assert coalition([v + 7.3 for v in x]) == (0, 1)                        # translation (event effect) invariant


def test_padic_hull_is_disc_tree():
    x = [0, 4, 2, 1]                                                     # 2-adically: {0,4} closest, then 2, then 1
    m = hull_tree(x, lambda a, b: padic_dist(a, b, 2))
    assert m[0][1] | m[0][2] == frozenset({0, 1}) and m[-1][0] == 1.0
    prof = disc_profile(x, 2, 3)
    assert [n for _, n, _ in prof] == [1, 2, 3, 4]                       # integer jumps; composition splits 4 = 3+1 = 2+1+1
    assert prof[1][2] == (3, 1) and prof[2][2] == (2, 1, 1)


def test_obstruction_profile_matches_depth():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = 5; p = 3; K = 6
        edges = [e for e in itertools.combinations(range(n), 2) if rng.random() < 0.6]
        if not edges:
            continue
        x = rng.integers(0, 3 ** K, size=n)
        claims = [(i, j, int(x[i] + x[j] + (3 ** int(rng.integers(0, K)) if rng.random() < 0.3 else 0))) for i, j in edges]
        prof = obstruction_profile(n, claims, p, K)
        ms = sum_claim_depth(n, claims, p, K)
        assert all(o == 0 for m, o in prof if m <= ms)
        assert ms == K or prof[ms + 1][1] > 0
        assert all(prof[i + 1][1] - prof[i][1] in (0, 1) for i in range(K))   # piecewise linear, integer slopes 0/1


def test_coalition_test_detects_planted_alliance():
    rng = np.random.default_rng(1); cells = []
    for _ in range(300):
        v = rng.normal()
        a = rng.normal(0, 1)
        # B follows A tightly on half the units (an alliance not explained by lean)
        b = a + rng.normal(0, 0.05) if rng.random() < 0.5 else rng.normal(0, 1)
        cells.append({"A": v + a, "B": v + b + 0.3, "C": v + rng.normal(0, 1) - 0.3})
    r = coalition_test(cells, ["A", "B", "C"], B=300)
    assert r["coalitions"]["('A', 'B')"]["obs"] > r["coalitions"]["('A', 'B')"]["null"] and r["coalitions"]["('A', 'B')"]["p"] < 0.01


def test_variogram_invariant_under_natural_change():
    from amb_vigneaux.hull import variogram
    rng = np.random.default_rng(3)
    cells = [{"A": rng.normal(), "B": rng.normal(), "C": rng.normal()} for _ in range(200)]
    u = {"A": 0.7, "B": -1.1, "C": 0.2}
    moved = [{s: c[s] + u[s] + (i * 0.01) for s in c} for i, c in enumerate(cells)]     # u_s + v_e
    g1 = np.array(variogram(cells, ["A", "B", "C"], B=5)["gamma"]); g2 = np.array(variogram(moved, ["A", "B", "C"], B=5)["gamma"])
    assert np.allclose(g1, g2, atol=1e-4)


def test_majority_reads_the_variogram():
    import itertools
    def stats_(P):
        p = [sum(pr for x, pr in P.items() if x[i]) for i in range(3)]
        g = [0.5 * sum(pr * (x[i] - x[j]) ** 2 for x, pr in P.items()) for i, j in ((0, 1), (0, 2), (1, 2))]
        t = sum(pr for x, pr in P.items() if all(x)); m = sum(pr for x, pr in P.items() if sum(x) >= 2)
        return p, g, t, m
    for P in ({x: np.prod([.45 if v else .55 for v in x]) for x in itertools.product([0, 1], repeat=3)},
              {(1, 1, 0): .225, (1, 0, 1): .225, (0, 1, 1): .225, (0, 0, 0): .325}, {(1, 1, 1): .45, (0, 0, 0): .55}):
        p, g, t, m = stats_(P)
        assert abs(m - (sum(p) - sum(g) - 2 * t)) < 1e-12 and np.allclose(p, 0.45)
    # same marginals (same b, same lean, same loops), majority on opposite sides of 1/2
    assert stats_({(1, 1, 0): .225, (1, 0, 1): .225, (0, 1, 1): .225, (0, 0, 0): .325})[3] > 0.5 > stats_({(1, 1, 1): .45, (0, 0, 0): .55})[3]
