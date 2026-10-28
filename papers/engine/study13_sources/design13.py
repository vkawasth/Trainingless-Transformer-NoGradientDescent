"""Study 13 design (deterministic; no sampling). Imported by run_study13.py.

Three sources (G1, G2, C), K_C = 8 batches each. A batch: the triple window (the probe) with n records and three pair windows with
rho n records each (pair tables come from large registries). Source C is trusted and genuine in most of its batches; in some of them
a share eps_b of the records comes from the hidden channel Q = P + s chi (pairs unchanged, m_T moved by eps_b s). The adversary knows
P, n, K and the regulator's thresholds, not the sampling noise."""
import os, sys
import numpy as np
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from pce.lab import law
from pce import injection as IJ
from pce import sources as SR

Z = 1.96; K_C = 8; K = 3 * K_C; RHO = 25
C_B = float(norm.ppf(1 - 0.05 / (2 * K)))            # batch level: Bonferroni over the 24 batches, family-wise 0.05
C_S = float(norm.ppf(1 - 0.05 / 6))                  # source level: Bonferroni over the 3 sources, family-wise 0.05
NB = (500, 1000, 2000, 4000, 8000)                   # triple records per batch; N = 24 n triple records in all
A_RISK20 = C_S - float(norm.ppf(0.8))                # expected source statistic for exposure risk 0.2
SRC = np.array(["G1"] * K_C + ["G2"] * K_C + ["C"] * K_C)
TRUTHS = {"W": (0.6, (0.5, 0.1, 0.4), (0.05, 0.5, 0.1)),          # registered: the worked example of Section 17
          "I": (0.45, (0.75, 0.25, 0.70), (0.25, 0.70, 0.20))}     # invented, smoke only

def acts_for(name, P):
    if name == "W": return [lambda x: 1 - float(sum(x) > 0), lambda x: 0.872 - float(x[0] or x[1])], 0.30
    gm0 = SR.components(SR.SD.exact_tables(P)); mu, v, r0 = gm0[0], gm0[1], gm0[2]
    pA = SR.SD.U_poly(mu, v, lambda x: 1 - float(sum(x) > 0), range(3)); c = float(pA[0] + pA[1] * (r0 - 0.08 * np.sign(pA[1])))
    return [lambda x: 1 - float(sum(x) > 0), lambda x: c], -1.0

def setup(name="W"):
    w, a1, a2 = TRUTHS[name]; P = law(w, np.array(a1), np.array(a2)); acts, u0 = acts_for(name, P); gm = SR.gauss_model(P, acts, u0, 0)
    lo, hi = IJ.stealth_range(P, (0, 1, 2)); s = hi if np.sign(hi / gm["vT"]) == np.sign(gm["rc"] - gm["r0"]) else lo
    es = gm["d"] * abs(gm["vT"]) / abs(s); ed = (abs(gm["R"][0 if gm["rc"] < gm["r0"] else 1] - gm["r0"])) * abs(gm["vT"]) / abs(s)
    a2 = 0.18 if name == "W" else 1.5 * (es + ed)              # A2: C's share puts the pooled share near the middle of the stealth window
    return dict(P=P, Q=SR.hidden_Q(P, s), s=float(s), acts=acts, u0=u0, a0=0, gm=gm, eps_star=es, eps_det=ed, a2_share=a2)

ADVERSARIES = ("A0", "A1", "A1b", "A3", "A2")

def shifts(adv, n, sm, s, a2_share=0.18):
    """per-batch shares eps_b for C's 8 batches, and the implied m_T shifts (eps_b s)"""
    if adv == "A0": e = [0.0] * K_C
    elif adv in ("A1", "A1b"):                       # sometimes: 4 of 8 batches, just enough for exposure risk 0.2 (A1) or 0.5 (A1b)
        a = A_RISK20 if adv == "A1" else C_S; D = SR.tuned_shift(sm, n, K_C, K, a); e = [2 * D / abs(s)] * 4 + [0.0] * 4
    elif adv == "A3":                                # concentrated: the same source shift as A1 in a single batch
        D = SR.tuned_shift(sm, n, K_C, K, A_RISK20); e = [K_C * D / abs(s)] + [0.0] * 7
    elif adv == "A2":                                # tuned to naive pooling: share 0.18 in every batch (pooled 0.06, in (eps*, eps_det))
        e = [a2_share] * K_C
    e = [min(x, 1.0) for x in e]; return e, [x * s for x in e]

if __name__ == "__main__":
    d = setup("W"); print({k: (round(v, 5) if isinstance(v, float) else None) for k, v in d.items() if isinstance(v, float)})
    print("c_b", C_B, "c_s", C_S, "a(risk .2)", A_RISK20, "B", SR.band(C_S, 1 / 3), SR.band(A_RISK20, 1 / 3))
    for n in NB: print(n, {a: np.round(shifts(a, n, d["gm"]["sm"], d["s"])[0][:2], 4).tolist() for a in ADVERSARIES})
