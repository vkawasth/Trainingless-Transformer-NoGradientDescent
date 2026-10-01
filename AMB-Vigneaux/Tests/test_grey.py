import numpy as np
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.bounds import grey_bounds, event, hypothesis, functional


def test_conditional_bound_is_frechet_without_covariates():
    sc = Scenario({"X": ("x",), "A": (0, 1), "Y": (0, 1)}, (("X", "Y"), ("X", "A")))
    for pY, pA in ((0.6, 0.1), (0.3, 0.7), (0.9, 0.5)):
        e = EmpiricalModel(sc, {("X", "Y"): np.array([1 - pY, pY]), ("X", "A"): np.array([1 - pA, pA])})
        b = grey_bounds(e, event(sc, lambda g: g["Y"] == 1 and g["A"] == 1), given=event(sc, lambda g: g["A"] == 1))
        assert abs(b["lo"] - max(0, (pY + pA - 1) / pA)) < 1e-6 and abs(b["hi"] - min(1, pY / pA)) < 1e-6


def test_hidden_truth_inside_and_hypotheses_outside_are_ruled_out():
    rng = np.random.default_rng(2)
    sc = Scenario({"X": (0, 1, 2), "A": (0, 1), "Y": (0, 1)}, (("X", "Y"), ("X", "A")))
    for _ in range(10):
        P = rng.dirichlet(np.ones(12)); e = EmpiricalModel.from_global(sc, P)
        YA, A = event(sc, lambda g: g["Y"] == 1 and g["A"] == 1), event(sc, lambda g: g["A"] == 1)
        b = grey_bounds(e, YA, given=A); t = (YA @ P) / (A @ P)
        assert b["lo"] - 1e-7 <= t <= b["hi"] + 1e-7
        if b["hi"] < 0.98:
            h = hypothesis(sc, lambda g: g["Y"] == 1, lambda g: g["A"] == 1, b["hi"] + 0.02, b["hi"] + 0.02)
            r = grey_bounds(e, event(sc, lambda g: g["Y"] == 1), rows=h)
            assert not r["feasible"] and r["slack"] > 0
        h = hypothesis(sc, lambda g: g["Y"] == 1, lambda g: g["A"] == 1, t, t)
        assert grey_bounds(e, event(sc, lambda g: g["Y"] == 1), rows=h)["feasible"]


def test_one_hypothesis_closes_the_triangle_loop():
    rng = np.random.default_rng(3)
    sc = Scenario({"L": (0, 1), "C": (0, 1), "R": (0, 1)}, (("L", "C"), ("C", "R"), ("L", "R")))
    P = rng.dirichlet(np.ones(8)) * 0.6 + 0.05; P /= P.sum(); e = EmpiricalModel.from_global(sc, P)
    maj = functional(sc, lambda g: float(g["L"] + g["C"] + g["R"] >= 2))
    free = grey_bounds(e, maj); assert free["hi"] - free["lo"] > 1e-3
    s = event(sc, lambda g: g["L"] == 1 and g["C"] == 1 and g["R"] == 1) @ P / (event(sc, lambda g: g["L"] == 1 and g["C"] == 1) @ P)
    b = grey_bounds(e, maj, rows=hypothesis(sc, lambda g: g["R"] == 1, lambda g: g["L"] == 1 and g["C"] == 1, s, s))
    assert b["hi"] - b["lo"] < 1e-5 and abs(b["lo"] - maj @ P) < 1e-5
