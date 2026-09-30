import numpy as np
from amb_vigneaux.arity_holonomy import holonomy, thin, grade_units


def test_full_grade_closes_exactly_and_lower_grade_carries_plant():
    rng = np.random.default_rng(0)
    V = {u: {s: float(rng.normal()) for s in "ABC"} for u in range(400)}
    h3 = holonomy(V, ("A", "B", "C"), 3)
    assert abs(h3["phi"]) < 1e-12 and h3["max_abs_unit_loop"] < 1e-12
    Vt = thin(V, 0.7, rng)
    for u, vs in Vt.items():                       # plant: C shifted by +1 only where B is missing
        if "C" in vs and "A" in vs and "B" not in vs:
            vs["C"] += 1.0
    assert abs(holonomy(Vt, ("A", "B", "C"), 3)["phi"]) < 1e-12
    h2 = holonomy(Vt, ("A", "B", "C"), 2)
    assert 0.5 < h2["phi"] < 1.5 and h2["z"] > 2


def test_grades():
    V = {1: {"A": 0, "B": 1}, 2: {"A": 0, "B": 1, "C": 2}, 3: {"C": 1}}
    assert grade_units(V, ("A", "B", "C")) == {1: 2, 2: 3, 3: 1}
