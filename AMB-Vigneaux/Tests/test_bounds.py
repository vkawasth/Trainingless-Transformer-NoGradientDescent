import numpy as np
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.models import pr_box, white_noise
from amb_vigneaux.bounds import functional, outcome_bounds, nc_part_bounds

TRI = Scenario({"L": (0, 1), "C": (0, 1), "R": (0, 1)}, (("L", "C"), ("C", "R"), ("L", "R")))


def test_chsh_dropped_context_bound_is_the_chsh_inequality():
    PR = pr_box(); sc = PR.scenario; WN = white_noise(sc); sub = Scenario(dict(sc.outcomes), sc.contexts[:3]); a, b = sc.contexts[3]
    f = functional(sub, lambda g: 1.0 if g[a] == g[b] else -1.0)
    for lam in np.linspace(0, 1, 11):
        bd = outcome_bounds(EmpiricalModel(sub, {C: lam * PR.tables[C] + (1 - lam) * WN.tables[C] for C in sub.contexts}), f)
        assert abs(bd["lo"] - max(-1, 3 * lam - 2)) < 1e-7 and abs(bd["hi"] - 1) < 1e-7
        assert (bd["lo"] - 1e-9 <= -lam) == (lam <= 0.5 + 1e-12)          # truth inside iff CF = 0
        assert bd["lo"] - 1e-9 <= bd["point"] <= bd["hi"] + 1e-9


def test_identified_iff_orthogonal_to_the_loop_direction():
    rng = np.random.default_rng(0)
    chi = functional(TRI, lambda g: (-1.0) ** (g["L"] + g["C"] + g["R"]))
    fs = {"all": functional(TRI, lambda g: g["L"] * g["C"] * g["R"]),
          "unanimous": functional(TRI, lambda g: float(g["L"] == g["C"] == g["R"]))}
    assert fs["unanimous"] @ chi == 0 and fs["all"] @ chi != 0
    for _ in range(20):
        e = EmpiricalModel.from_global(TRI, rng.dirichlet(np.ones(8)))
        assert abs(outcome_bounds(e, fs["unanimous"])["hi"] - outcome_bounds(e, fs["unanimous"])["lo"]) < 1e-7
        assert outcome_bounds(e, fs["all"])["hi"] - outcome_bounds(e, fs["all"])["lo"] > 1e-4


def test_bounds_are_directional_under_relabelling():
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(8)); e = EmpiricalModel.from_global(TRI, P)
    G = TRI.global_sections(); flip = [G.index(tuple(1 - x for x in g)) for g in G]
    ef = EmpiricalModel.from_global(TRI, P[np.argsort(flip)] if False else P[flip])
    f = functional(TRI, lambda g: float(g["L"] + g["C"] + g["R"] >= 2))
    b, bf = outcome_bounds(e, f), outcome_bounds(ef, f)
    assert abs(bf["lo"] - (1 - b["hi"])) < 1e-7 and abs(bf["hi"] - (1 - b["lo"])) < 1e-7


def test_contextual_family_falls_back():
    PR = pr_box(); f = functional(PR.scenario, lambda g: float(g["a0"] == g["b1"]))
    assert outcome_bounds(PR, f)["mode"] == "projected"
    e = EmpiricalModel(PR.scenario, {C: 0.7 * PR.tables[C] + 0.3 * white_noise(PR.scenario).tables[C] for C in PR.scenario.contexts})
    nb = nc_part_bounds(e, f); assert abs(nb["cf"] - 0.4) < 1e-6 and nb["lo"] <= nb["hi"]
