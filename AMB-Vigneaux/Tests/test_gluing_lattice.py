from amb_vigneaux.gluing_lattice import lattice, greedy_maximal


def test_exact_lattice_downward_closed_and_extremes():
    # contexts a..d; exactly the sets containing both 'a' and 'd' fail to glue
    glues = lambda S: 0.0 if {"a", "d"} <= set(S) else 1.0
    L = lattice("abcd", glues)
    assert L["monotonicity_violations"] == 0
    assert L["minimal_obstructed"] == [("a", "d")]
    assert sorted(L["maximal_glueable"]) == [("a", "b", "c"), ("b", "c", "d")]


def test_greedy_finds_maximal():
    glues = lambda S: 0.0 if {"a", "d"} <= set(S) else 1.0
    G = greedy_maximal("abcd", glues)
    assert set(G) == {("a", "b", "c"), ("b", "c", "d")}
