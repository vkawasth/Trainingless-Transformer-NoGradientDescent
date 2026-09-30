import numpy as np
from amb_vigneaux.strata import design, ipf, cofacial_set, is_facial, toggle_radius_amb


def test_independence_model_forces_row_or_column():
    A, cells = design((3, 4), [(0,), (1,)])
    rng = np.random.default_rng(0)
    p = np.outer(rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(4))).ravel()
    for c in range(12):
        i, j = cells[c]
        row = [k for k, x in enumerate(cells) if x[0] == i]; col = [k for k, x in enumerate(cells) if x[1] == j]
        F, _ = cofacial_set(A, p, c)
        assert sorted(F) in (sorted(row), sorted(col))
        r = toggle_radius_amb(A, p, c)
        assert abs(r["mass_forced"] - min(p[row].sum(), p[col].sum())) < 1e-9     # exact for this model
        assert not is_facial(A, [k for k in range(12) if k != c])               # a single zero is not a stratum
        assert is_facial(A, [k for k in range(12) if k not in F])


def test_saturated_model_forces_only_the_cell():
    A, cells = design((2, 3), [(0, 1)])
    p = np.full(6, 1 / 6)
    r = toggle_radius_amb(A, p, 2)
    assert r["forced"] == [2] and abs(r["upper"] - r["lower"]) < 1e-12


def test_no3way_complement_is_facial_and_ipf_matches_margins():
    rng = np.random.default_rng(1)
    N = rng.integers(1, 20, size=(3, 3, 2)).astype(float)
    M = ipf(N, [(0, 1), (0, 2), (1, 2)])
    for m in [(0, 1), (0, 2), (1, 2)]:
        o = tuple(a for a in range(3) if a not in m)
        assert np.allclose(M.sum(o), N.sum(o), atol=1e-6)
    A, cells = design(N.shape, [(0, 1), (0, 2), (1, 2)]); p = (M / M.sum()).ravel()
    for c in (0, 7, 17):
        F, _ = cofacial_set(A, p, c)
        assert c in F and is_facial(A, [k for k in range(len(cells)) if k not in F]) and len(F) > 1


def test_exact_matches_bruteforce_small():
    import itertools
    from amb_vigneaux.strata import exact_toggle
    A, cells = design((2, 3, 2), [(0, 1), (0, 2), (1, 2)]); n = len(cells)
    rng = np.random.default_rng(4); p = rng.dirichlet(np.ones(n))
    cof = [F for F in (frozenset(k for k in range(n) if m >> k & 1) for m in range(1, 2 ** n))
           if is_facial(A, [k for k in range(n) if k not in F])]
    for c in (0, 5, 9):
        best = min(p[list(F)].sum() for F in cof if c in F)
        e = exact_toggle(A, p, c)
        assert abs(e["mass"] - best) < 1e-8 and is_facial(A, [k for k in range(n) if k not in e["forced"]])
