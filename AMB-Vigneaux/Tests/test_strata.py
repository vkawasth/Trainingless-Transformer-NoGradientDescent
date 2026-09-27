"""Support stratification (Section: support strata): gamma and logical contextuality are
functions of the face; interior and vertices are both trivial for them; CF is not a face function."""
import itertools
import numpy as np
from amb_vigneaux import models, analyse_outcomes


def _trivial_possibilistic(r):
    return r.n_global_sections > 0 and not r.logical_witnesses and not r.cohomological_witnesses


def test_interior_has_no_possibilistic_contextuality_but_cf_can_be_positive():
    for m in (models.tsirelson_box(), models.noisy(models.pr_box(), 0.99), models.noisy(models.ghz_model(), 0.9)):
        r = analyse_outcomes(m)
        assert all(len(r.support[C]) == len(list(m.scenario.sections(C))) for C in m.scenario.contexts)
        assert _trivial_possibilistic(r) and r.contextual_fraction > 0.1


def test_vertices_are_noncontextual_and_carry_no_information():
    for base in (models.pr_box(), models.ghz_model(), models.parity_grid()):
        sc = base.scenario
        X = sc.measurements
        rng = np.random.default_rng(0)
        for _ in range(3):
            g = [sc.outcomes[x][rng.integers(len(sc.outcomes[x]))] for x in X]
            m = models.deterministic(sc, dict(zip(X, g)))
            r = analyse_outcomes(m)
            assert _trivial_possibilistic(r) and abs(r.contextual_fraction) < 1e-9


def test_cf_is_not_a_face_function():
    # same (interior) face, different CF
    a = analyse_outcomes(models.tsirelson_box()).contextual_fraction
    b = analyse_outcomes(models.white_noise(models.chsh_scenario())).contextual_fraction
    assert a > 0.4 and abs(b) < 1e-9


def test_intermediate_faces_carry_the_obstruction():
    pr, hardy = analyse_outcomes(models.pr_box()), analyse_outcomes(models.hardy_model())
    assert pr.n_global_sections == 0 and len(pr.cohomological_witnesses) > 0
    assert hardy.logical_witnesses and hardy.false_negatives        # logical, missed by gamma over Z


def _parity_chsh(q, odd):
    from amb_vigneaux.scenario import EmpiricalModel
    def f(C, s):
        a, b = s; x, y = int(C[0][1]), int(C[1][1])
        if (a ^ b) != ((x * y) if odd else 0):
            return 0.0
        return q if a == 0 else 1 - q
    return EmpiricalModel.from_function(models.chsh_scenario(), f)


def test_two_outcome_supports_all_or_nothing():
    from amb_vigneaux.information import information_profile
    # flat: a one-parameter family, non-contextual, information varies with q
    for q in (0.5, 0.7, 0.9):
        r = analyse_outcomes(_parity_chsh(q, False))
        assert r.n_global_sections == 2 and not r.logical_witnesses and abs(r.contextual_fraction) < 1e-9
    # odd holonomy: no-signalling forces q = 1/2 (the PR box); other q signal
    assert analyse_outcomes(_parity_chsh(0.9, True)).signalling_defect > 0.5
    pr = analyse_outcomes(_parity_chsh(0.5, True))
    assert pr.n_global_sections == 0 and pr.cohomological_witnesses and abs(pr.contextual_fraction - 1) < 1e-9
    # identical information profiles, CF 1 versus 0
    ip, iq = information_profile(_parity_chsh(0.5, True)), information_profile(_parity_chsh(0.5, False))
    for C in ip:
        assert abs(ip[C]["H_joint"] - iq[C]["H_joint"]) < 1e-12
        assert abs(ip[C]["total_correlation"] - iq[C]["total_correlation"]) < 1e-12
