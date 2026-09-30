import numpy as np
from amb_vigneaux.models import pr_box, white_noise
from amb_vigneaux.scenario import EmpiricalModel
from amb_vigneaux.outcome import contextual_fraction, analyse_outcomes
from amb_vigneaux.functors import max_entropy_extension


def _mix(lam):
    PR = pr_box(); WN = white_noise(PR.scenario)
    return EmpiricalModel(PR.scenario, {C: lam * PR.tables[C] + (1 - lam) * WN.tables[C] for C in PR.scenario.contexts})


def test_cf_continuous_gamma_jumps_ipf_iff_cf_zero():
    for lam in (0.3, 0.6, 0.8, 0.95):
        e = _mix(lam); cf = contextual_fraction(e).value
        assert abs(cf - max(0.0, 2 * lam - 1)) < 1e-7                          # CF = (2 lambda - 1)_+
        rep = analyse_outcomes(e, eps=0.02, with_cf=False)
        assert rep.gamma_h1_nonzero == (lam > 0.92)                            # gamma only after the support collapse
        assert max_entropy_extension(e, iters=3000, tol=1e-8).converged == (cf < 1e-9)
    # the region the two-box picture misses: CF > 0 while gamma = 0
    e = _mix(0.7)
    assert contextual_fraction(e).value > 0.3 and not analyse_outcomes(e, eps=0.02, with_cf=False).gamma_h1_nonzero
