"""Study 12 design (deterministic; no sampling). Imported by run_study12.py."""
import itertools, os, sys
import numpy as np
from numpy.polynomial import polynomial as Pn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "study11b_settle"))
from pce.lab import law, moments, fit_v
from pce import segment_decision as SD
from pce import injection as IJ
import design11b as D11

Z = 1.96; N = 100000
EPS_FLIP = (0.0, 0.02, 0.04, 0.05, 0.055, 0.06, 0.065, 0.07, 0.08, 0.10)
EPS_DET = (0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20)
N_TET = (2000, 10000, 50000); TILT = 0.3
G4 = (0.40, (0.75, 0.25, 0.70, 0.60), (0.15, 0.70, 0.20, 0.20))      # registered H4 truth
G4_SMOKE = (0.30, (0.70, 0.30, 0.80, 0.55), (0.20, 0.65, 0.10, 0.25))   # simulated in the first smoke run; demoted to smoke use

def hidden():
    """design D2 of Study 11b (truth G; treat A vs flat C); the hidden-channel injection toward the A/C switch"""
    tn, spec, u0, a0, d = D11.build("D2"); P = D11.truth(tn)[0]; acts = D11._acts(spec)
    base = IJ.pipeline(P, acts, u0); r0, R, vT = base["rhat"], base["R"], float(np.prod(base["v"]))
    ps = base["polys"]; rc = (Pn.polyval(0, ps[1]) - ps[0][0]) / ps[0][1]          # where U_A = U_C (C is flat)
    lo, hi = IJ.stealth_range(P, (0, 1, 2)); s = hi if np.sign(hi / vT) == np.sign(rc - r0) else lo
    Q = IJ.as_law(IJ.as_vec(P) + s * IJ.chi(3, (0, 1, 2)), 3)
    wall = R[0] if rc < r0 else R[1]
    return dict(P=P, Q=Q, acts=acts, u0=u0, a0=a0, s=s, r0=r0, R=R, vT=vT, rc=rc,
                eps_star=abs(rc - r0) * abs(vT) / abs(s), eps_det=abs(wall - r0) * abs(vT) / abs(s))

def tetrad_design(truth=None):
    w, a1, a2 = truth or G4; P = law(w, np.array(a1), np.array(a2)); Nj = IJ.tetrad_jacobian(P)
    tiltF = lambda X: (lambda mu, M, mS: mS((0, 1, 2)) / np.prod(fit_v(mu, M)[:3]))(*moments(X))
    tilt = IJ.influence(tiltF, P).ravel(); val, q = IJ.stealth_tilt_lp(P, tilt, Nj, 0.0); Qs = IJ.as_law(q, 4)
    k = int(np.argmax(tilt)); Qv = np.zeros_like(P); Qv[IJ.outcomes(4)[k]] = 1.0
    return dict(P=P, Qs=Qs, Qv=Qv, eps_s=TILT / val, eps_v=TILT / tilt[k], tilt_s=val, tilt_v=tilt[k])

if __name__ == "__main__":
    h = hidden(); print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in h.items() if k not in ("P", "Q", "acts")})
    t = tetrad_design(); print({k: round(v, 5) for k, v in t.items() if k not in ("P", "Qs", "Qv")})
