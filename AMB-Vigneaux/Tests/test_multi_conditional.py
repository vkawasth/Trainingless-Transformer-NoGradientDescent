import numpy as np
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.bounds import grey_bounds, event, hypothesis, condition_rank, loop_space, nerve_betti

TRI = Scenario({"L": (0, 1), "C": (0, 1), "R": (0, 1)}, (("L", "C"), ("C", "R"), ("L", "R")))


def test_nerves_have_no_h2():
    assert nerve_betti([("L", "C"), ("C", "R"), ("L", "R")]) == [1, 1]
    assert nerve_betti([("a0", "b0"), ("a0", "b1"), ("a1", "b0"), ("a1", "b1")]) == [1, 1]
    assert nerve_betti([("a", "b", "c"), ("a", "b"), ("b", "c")]) == [1, 0, 0]   # filled triangle: contractible


def test_two_triple_conditions_share_one_free_direction():
    rng = np.random.default_rng(4)
    P = rng.dirichlet(np.ones(8)) * 0.6 + 0.05; P /= P.sum(); e = EmpiricalModel.from_global(TRI, P)
    c1 = (lambda g: g["R"] == 1, lambda g: g["L"] == 1 and g["C"] == 1)
    c2 = (lambda g: g["L"] == 1, lambda g: g["C"] == 1 and g["R"] == 1)
    assert loop_space(TRI).shape[1] == 1
    s1 = event(TRI, lambda g: g["L"] == 1 and g["C"] == 1 and g["R"] == 1) @ P / (event(TRI, c1[1]) @ P)
    s2 = event(TRI, lambda g: g["L"] == 1 and g["C"] == 1 and g["R"] == 1) @ P / (event(TRI, c2[1]) @ P)
    assert condition_rank(TRI, [c1, c2], [s1, s2]) == 1
    b = grey_bounds(e, event(TRI, lambda g: c2[0](g) and c2[1](g)), given=event(TRI, c2[1]), rows=hypothesis(TRI, *c1, s1, s1))
    assert abs(b["lo"] - s2) < 1e-5 and abs(b["hi"] - s2) < 1e-5           # s1 determines s2
    r = grey_bounds(e, event(TRI, c1[1]), rows=hypothesis(TRI, *c1, s1, s1) + hypothesis(TRI, *c2, s2 + 0.05, s2 + 0.05))
    assert not r["feasible"] and r["slack"] > 0                              # each alone fine, jointly impossible


def test_two_filters_are_independent_conditions():
    rng = np.random.default_rng(5)
    sc = Scenario({"X": (0, 1, 2), "A": (0, 1), "F": (0, 1), "Y": (0, 1)}, (("X", "Y"), ("X", "A", "F")))
    P = rng.dirichlet(np.ones(24)); e = EmpiricalModel.from_global(sc, P)
    cA = (lambda g: g["Y"] == 1, lambda g: g["A"] == 1); cF = (lambda g: g["Y"] == 1, lambda g: g["F"] == 1)
    sA = event(sc, lambda g: g["Y"] == 1 and g["A"] == 1) @ P / (event(sc, cA[1]) @ P)
    sF = event(sc, lambda g: g["Y"] == 1 and g["F"] == 1) @ P / (event(sc, cF[1]) @ P)
    assert condition_rank(sc, [cA, cF], [sA, sF]) == 2
    tgt, giv = event(sc, lambda g: g["Y"] == 1 and g["A"] == 1 and g["F"] == 1), event(sc, lambda g: g["A"] == 1 and g["F"] == 1)
    free = grey_bounds(e, tgt, given=giv); cond = grey_bounds(e, tgt, given=giv, rows=hypothesis(sc, *cA, sA, sA) + hypothesis(sc, *cF, sF, sF))
    t = tgt @ P / (giv @ P)
    assert cond["lo"] - 1e-7 <= t <= cond["hi"] + 1e-7 and cond["hi"] - cond["lo"] <= free["hi"] - free["lo"] + 1e-9
