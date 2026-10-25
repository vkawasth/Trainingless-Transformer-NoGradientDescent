"""The origin fingerprint of a body of evidence: which model classes can explain it, and how much each saves.

Each model class is a functor from evidence to [0, 1] (the largest fraction of every table it can account for); the classes are nested,
so the fractions are ordered, and each is monotone under local post-processing (data processing). The fingerprint lists them, and the
origin is the smallest class that explains everything, with the direct-influence and contextuality diagnostics that decide the lower rungs.

  fingerprint(domains, windows, bell=None, joint3=None)
      windows: list of pce.Window. Returns a dict with
        signalling      {variable: TV between the marginals of its copies} (direct influence; Contextuality-by-Default)
        direct          sum of signalling
        linear_gap      0 iff a signed global law exists (no rung at all otherwise)
        cbd_c           Contextuality-by-Default degree c (> 0: contextual beyond direct influence); needs each variable in <= 2 windows
        lam_classical   lam* (classical noncontextual fraction)
        lam_quantum     dimension-free quantum fraction (NPA level 1+AB), if bell = (mA, mB) and windows are (A_x, B_y) pairs
        level2          the two-actor rung of a full three-variable binary table, if joint3 is given ("R>=0", "R", "C", ...)
        origin          a label: classical / quantum, not classical / beyond quantum / direct influence only /
                        contextual beyond direct influence / contextual (quantum status not computed)
        with direct influence (pairwise binary windows, linear gap > 0), also
        lam_classical_cbd, lam_quantum_cbd   fractions of the consistified system (pce.cbdfrac); parties as in lam_quantum_cbd
                        (default: bipartite when bell is given, else minimal commutation)
        shares          1 - lam* split into direct influence, quantum-explainable contextuality, beyond quantum
        origin_cbd      direct influence only / quantum-type contextuality with direct influence / beyond quantum, allowing
                        direct influence
"""
from __future__ import annotations
import numpy as np
from .integrate import gluable_fraction
from .ladder import level1, level2_m3
from .cbd import signalling, cbd


def fingerprint(domains, windows, bell=None, joint3=None, tol=1e-6, parties="auto", qtol=1e-4):
    out = {}
    try:
        s, _ = signalling(windows); out["signalling"] = {str(k): float(v) for k, v in s.items()}; out["direct"] = float(sum(s.values()))
        out["cbd_c"] = float(cbd(windows)["c"])
    except AssertionError:
        out["signalling"] = None; out["direct"] = None; out["cbd_c"] = None
    l1 = level1(domains, windows); out["linear_gap"] = l1["linear_gap"]; out["lam_classical"] = float(gluable_fraction(domains, windows).lam)
    out["lam_quantum"] = None
    if bell is not None and out["linear_gap"] < tol:
        from .npa import npa_fraction
        mA, mB = bell; P = np.zeros((mA, mB, 2, 2))
        for w in windows:
            x = int(w.vars[0][1:]); y = int(w.vars[1][1:]); P[x, y] = np.asarray(w.table, float)
        out["lam_quantum"] = float(npa_fraction(P, "1+AB"))
    if joint3 is not None:
        r = level2_m3(joint3); out["level2"] = r["rung"]; out["level2_negativity"] = r["negativity2"]; out["level2_imag"] = r["imag"]
    if out["linear_gap"] > tol and out["cbd_c"] is not None and all(len(w.vars) == 2 and np.asarray(w.table).shape == (2, 2) for w in windows):
        from .cbdfrac import lam_classical_cbd, lam_quantum_cbd, bipartite
        pt = (bipartite if bell is not None else None) if parties == "auto" else parties
        lc = lam_classical_cbd(windows); lq = lam_quantum_cbd(windows, parties=pt)
        out["lam_classical_cbd"] = lc; out["lam_quantum_cbd"] = lq
        out["shares"] = {"direct_influence": lc - out["lam_classical"], "quantum_contextual": lq - lc, "beyond_quantum": 1 - lq}
        out["origin_cbd"] = ("direct influence only" if lc > 1 - tol else
                             "quantum-type contextuality with direct influence" if lq > 1 - qtol else "beyond quantum, allowing direct influence")
    if out["linear_gap"] > tol:
        if out["cbd_c"] is not None and out["cbd_c"] > tol: out["origin"] = "contextual beyond direct influence"
        else: out["origin"] = "direct influence only"
    elif out["lam_classical"] > 1 - tol: out["origin"] = "classical"
    elif out["lam_quantum"] is not None: out["origin"] = "quantum, not classical" if out["lam_quantum"] > 1 - tol else "beyond quantum"
    else: out["origin"] = "contextual (quantum status not computed)"
    return out
