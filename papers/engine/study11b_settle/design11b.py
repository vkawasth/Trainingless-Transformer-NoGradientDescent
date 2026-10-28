"""Study 11b design (deterministic; no sampling). Imported by run_study11b.py."""
import os, sys
import numpy as np
from numpy.polynomial import polynomial as Pn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from pce.lab import law, moments
from pce import segment_decision as SD

Z = 1.96
TRUTHS = {"G": (0.30, (0.70, 0.30, 0.80), (0.20, 0.65, 0.10)), "H": (0.40, (0.80, 0.25, 0.70), (0.15, 0.70, 0.20))}
any_h = lambda x: float(sum(x) > 0); two_h = lambda x: float(sum(x) >= 2); x0_h = lambda x: float(x[0])

def truth(name):
    w, a1, a2 = TRUTHS[name]; P = law(w, np.array(a1), np.array(a2)); mu, M, mS = moments(P); v = SD.v_closed(M)
    tau, r0, _, _ = SD.tau_r(P); return P, mu, v, tau, r0, SD.r_walls(mu, v)

def _acts(spec):
    """spec: list of (kind, b) with kind in A (u = b - [any harm]), B (u = b - [at least two harmed]), C (u = b - [variable 0 harmed])"""
    f = {"A": any_h, "B": two_h, "C": x0_h}; return [(lambda k, b: (lambda x: b - f[k](x)))(k, b) for k, b in spec]

def build(name):
    """u0 sits three radii below the competitor's value, so that one constraint binds (the median formula needs one binding constraint).
    returns truth name, act specs, u0, the index the oracle chooses, and the radius d targeted by the design"""
    if name == "D1":                                   # one act; admissibility binds; pre-probe: abstain
        P, mu, v, tau, r0, R = truth("G"); pA = SD.U_poly(mu, v, _acts([("A", 1.0)])[0], range(3)); u0 = Pn.polyval(r0, pA) - abs(pA[1]) * 0.35
        return "G", [("A", 1.0)], float(u0), 0, 0.35
    if name in ("D2", "D4"):                           # A against a flat conservative act C; D4 puts the truth on the crossing (d = 0)
        P, mu, v, tau, r0, R = truth("G"); pA = SD.U_poly(mu, v, _acts([("A", 1.0)])[0], range(3)); d = 0.35 if name == "D2" else 0.0
        UC = Pn.polyval(r0, pA) - abs(pA[1]) * d; bC = UC + mu[0]; u0 = UC - abs(pA[1]) * (3 * d if name == "D2" else 0.85)
        return "G", [("A", 1.0), ("C", float(bC))], float(u0), 0, d
    if name == "D3":                                   # a decreasing act B against C: the boundary is on the other side of r0
        P, mu, v, tau, r0, R = truth("G"); pB = SD.U_poly(mu, v, _acts([("B", 0.5)])[0], range(3)); UC = Pn.polyval(r0, pB) - abs(pB[1]) * 0.25
        return "G", [("B", 0.5), ("C", float(UC + mu[0]))], float(UC - abs(pB[1]) * 3 * 0.25), 0, 0.25
    if name == "D5":                                   # a second truth
        P, mu, v, tau, r0, R = truth("H"); pA = SD.U_poly(mu, v, _acts([("A", 1.0)])[0], range(3)); UC = Pn.polyval(r0, pA) - abs(pA[1]) * 0.30
        return "H", [("A", 1.0), ("C", float(UC + mu[0]))], float(UC - abs(pA[1]) * 3 * 0.30), 0, 0.30
    raise KeyError(name)

DESIGNS = ("D1", "D2", "D3", "D4", "D5")

def summary(name):
    tn, spec, u0, a0, d = build(name); P, mu, v, tau, r0, R = truth(tn); polys = [SD.U_poly(mu, v, u, range(3)) for u in _acts(spec)]
    orc = SD.oracle(polys, r0, u0); pre = SD.set_rule(polys, R, u0); rad = SD.settle_radius_affine(polys, r0, u0)
    nstar = (Z * tau / rad) ** 2 if rad > 0 else np.inf
    return dict(truth=tn, acts=spec, u0=u0, r0=r0, R=R, tau=tau, oracle=orc, pre_probe=pre, radius=rad, nstar=nstar)

if __name__ == "__main__":
    for d in DESIGNS: print(d, {k: (np.round(v, 4) if isinstance(v, (float, tuple)) else v) for k, v in summary(d).items()})
