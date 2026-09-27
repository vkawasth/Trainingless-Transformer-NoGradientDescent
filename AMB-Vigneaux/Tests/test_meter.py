import numpy as np
from amb_vigneaux.meter import run_meter
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.outcome import contextual_fraction


def test_liar_cycle_within_unit_and_across_units():
    liar = [("u", 0, 1, 0), ("u", 1, 2, 0), ("u", 2, 0, 1)]
    r = run_meter(liar, deterministic=True)
    assert r.self_contradictions["u"] and not r.cross_unit_cycles
    split = [("a", 0, 1, 0), ("b", 1, 2, 0), ("c", 2, 0, 1)]
    r = run_meter(split, trusted=["a", "b"], deterministic=True)
    assert not any(r.self_contradictions.values()) and len(r.cross_unit_cycles) == 1
    cy = r.cross_unit_cycles[0]
    assert cy.blamed == (0, 2) and abs(cy.cf_lower - 1) < 1e-12


def test_overlap_disagreement_is_recorded_not_glued():
    r = run_meter([("a", 0, 1, 0), ("b", 0, 1, 1), ("a", 1, 2, 0)], deterministic=True)
    assert len(r.overlap_disagreements) == 1 and not r.cross_unit_cycles


def test_coherent_fabrication_is_invisible():
    r = run_meter([("a", 0, 1, 0), ("a", 1, 2, 1), ("b", 3, 4, 1)], deterministic=True)
    assert not any(r.self_contradictions.values()) and not r.overlap_disagreements and not r.cross_unit_cycles


def _cycle_model(s):
    n = len(s)
    outs = {f"c{v}": (0, 1) for v in range(n)}
    ctx = tuple((f"c{v}", f"c{(v + 1) % n}") for v in range(n))
    sc = Scenario(outs, ctx)
    sk = dict(zip(ctx, s))
    return EmpiricalModel.from_function(sc, lambda C, o: (1 + sk[C] * (1 if o[0] == o[1] else -1)) / 4)


def test_cf_lower_bound_holds_against_lp():
    rng = np.random.default_rng(2)
    for n in (3, 5):
        for _ in range(10):
            s = rng.uniform(0.3, 1, n); s[0] *= -1                 # odd holonomy
            claims = []
            for v in range(n):
                k = int(round(200 * (1 + s[v]) / 2))
                claims += [(f"u{v}", v, (v + 1) % n, 0)] * k + [(f"u{v}", v, (v + 1) % n, 1)] * (200 - k)
            r = run_meter(claims)
            assert len(r.cross_unit_cycles) == 1
            s_emp = [(2 * int(round(200 * (1 + x) / 2)) - 200) / 200 for x in s]
            cf = contextual_fraction(_cycle_model(s_emp)).value
            assert r.cross_unit_cycles[0].cf_lower <= cf + 1e-9
