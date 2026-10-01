import numpy as np
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.models import pr_box, white_noise
from amb_vigneaux.functors import extension_exists
from amb_vigneaux.kernel_presheaf import rip_order, sweep_glue, gluing_defects, signed_extension, descent_defect


def test_rip_order_detects_cycles():
    assert rip_order([("a", "b"), ("b", "c"), ("c", "d")]) is not None
    assert rip_order([("a", "b", "c"), ("b", "d"), ("c", "e")]) is not None
    assert rip_order([("a", "b"), ("b", "c"), ("c", "a")]) is None
    assert rip_order(pr_box().scenario.contexts) is None


def test_acyclic_sweep_glues_arbitrary_compatible_family():
    rng = np.random.default_rng(1)
    sc = Scenario({v: (0, 1, 2) for v in "abcd"}, (("c", "d"), ("a", "b"), ("b", "c")))
    ab = rng.dirichlet(np.ones(9)); b = ab.reshape(3, 3).sum(0)
    bc = (b[:, None] * rng.dirichlet(np.ones(3), size=3)).ravel(); c = bc.reshape(3, 3).sum(0)
    cd = (c[:, None] * rng.dirichlet(np.ones(3), size=3)).ravel()
    e = EmpiricalModel(sc, {("a", "b"): ab, ("b", "c"): bc, ("c", "d"): cd})
    assert descent_defect(e) < 1e-12
    assert max(gluing_defects(e, sweep_glue(e, rip_order(sc.contexts))).values()) < 1e-12


def test_conditional_product_exists_but_cycle_does_not_close():
    PR = pr_box(); sc = PR.scenario; WN = white_noise(sc); lam = 0.3
    e = EmpiricalModel(sc, {C: lam * PR.tables[C] + (1 - lam) * WN.tables[C] for C in sc.contexts})
    assert extension_exists(e)                                  # a global law exists (CF = 0)
    d = gluing_defects(e, sweep_glue(e, sc.contexts[:3]))
    assert max(d[C] for C in sc.contexts[:3]) < 1e-12
    assert abs(d[sc.contexts[3]] - (lam + lam ** 3) / 2) < 1e-12   # correlations compose along the chain


def test_linear_descent_always_glues():
    PR = pr_box()
    assert descent_defect(PR) < 1e-12 and signed_extension(PR)[1] < 1e-12
