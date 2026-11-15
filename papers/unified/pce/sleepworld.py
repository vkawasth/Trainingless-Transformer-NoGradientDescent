"""The sleep-onset test world (Study 27i). A synthetic, declared law whose variables and structure follow the three-layer sleep-onset
cascade (neurochemical/circadian gate, autonomic/thermal coupling, cortical commitment). Probabilities are set here, not estimated from
any dataset; the families of sources mirror the variable sets of three public datasets (MMASH, DREAMT, Sleep-EDF) and of the sleep-lab
anchors one would wish for. New module; it modifies nothing.
Variables: Z population (0 young healthy, 1 clinical; never reported), K caffeine, L evening light, P sleep pressure (high = 1),
C circadian night window (open = 1), A anxiety / sympathetic tone, T distal vasodilation, E EEG (0 alpha, 1 theta, 2 spindles),
O onset outcome (0 wake, 1 N1, 2 N2 within the window).
Decision quantity g: the warm-bath effect in the circadian-ready, low-pressure state (State C of the cascade):
  g = P(O = 2 | T = 1, P = 0, C = 1) - P(O = 2 | T = 0, P = 0, C = 1).
  law(shift)                  the joint law; shift: dict of scenario modifications (dst, popmix, boundary)
  FAMILIES                    contexts per family with replicates and sample sizes
  g(p), grad_g(p), g_parts    the decision quantity and its two ratios (numerator, denominator vectors)
"""
from __future__ import annotations
import itertools, math
import numpy as np
from . import snapengine as S
LEVELS = dict(Z=2, K=2, L=2, P=2, C=2, A=2, T=2, E=3, O=3)
W = S.World(LEVELS); V = W.vars; ST = np.array(W.states)
def col(v): return ST[:, V.index(v)]
def _softmax(a): a = np.asarray(a, float); e = np.exp(a - a.max(axis=0)); return e / e.sum(axis=0)
def law(popmix=0.5, cday=0.0, boundary=0.0):
    """popmix: P(Z = 1); cday: drop of P(C = 1) (a daylight-saving shift: circadian window opens less often at the usual bedtime);
    boundary: in [0, 1), the probability of high pressure after caffeine is multiplied by (1 - boundary) (the cell K = 1, P = 1 empties)."""
    Z, K, L, P_, C, A, T, E, O = (col(v) for v in V)
    pz = np.where(Z == 1, popmix, 1 - popmix)
    pk1 = np.where(Z == 1, 0.55, 0.45); pk = np.where(K == 1, pk1, 1 - pk1)
    pl1 = np.where(Z == 1, 0.45, 0.55); pl = np.where(L == 1, pl1, 1 - pl1)
    pp1 = np.clip(0.62 - 0.30 * K + 0.10 * Z, 0.01, 0.99) * np.where(K == 1, 1 - boundary, 1.0); pp = np.where(P_ == 1, pp1, 1 - pp1)
    pc1 = np.clip(0.30 + 0.35 * L - cday, 0.01, 0.99); pc = np.where(C == 1, pc1, 1 - pc1)
    pa1 = 0.30 + 0.20 * Z + 0.15 * K; pa = np.where(A == 1, pa1, 1 - pa1)
    pt1 = np.clip(0.30 + 0.35 * C - 0.20 * A, 0.02, 0.98); pt = np.where(T == 1, pt1, 1 - pt1)
    drive = 0.9 * P_ + 0.8 * C + 0.4 * T - 0.6 * A
    pe = _softmax([-1.2 * drive, 0 * drive, 1.2 * drive - 0.6])[E, np.arange(len(E))]
    s2 = 1.6 * (E == 2) + 0.5 * (E == 1) + 0.9 * T * (1 - P_) * C + 0.15 * T - 0.6 * A
    s1 = 0.9 * (E == 1) + 0.3 * (E == 2); s0 = 1.1 * (E == 0) + 0.5 * A
    po = _softmax([s0, s1, s2])[O, np.arange(len(O))]
    p = pz * pk * pl * pp * pc * pa * pt * pe * po
    # the joint must be a law: marginalising each factor over its own variable gives one (each factor is a conditional), so p sums to 1
    return p / p.sum()
_num1 = ((col("O") == 2) & (col("T") == 1) & (col("P") == 0) & (col("C") == 1)).astype(float); _den1 = ((col("T") == 1) & (col("P") == 0) & (col("C") == 1)).astype(float)
_num0 = ((col("O") == 2) & (col("T") == 0) & (col("P") == 0) & (col("C") == 1)).astype(float); _den0 = ((col("T") == 0) & (col("P") == 0) & (col("C") == 1)).astype(float)
G_PARTS = ((_num1, _den1), (_num0, _den0))
def g(p): return (_num1 @ p) / (_den1 @ p) - (_num0 @ p) / (_den0 @ p)
def grad_g(p):
    a1, b1, a0, b0 = _num1 @ p, _den1 @ p, _num0 @ p, _den0 @ p
    return _num1 / b1 - a1 * _den1 / b1 ** 2 - (_num0 / b0 - a0 * _den0 / b0 ** 2)
# families: (name, role, [(ctx, replicates, n)])
FAMILIES = [
    ("lifestyle", "source", [(("K", "A"), 2, 3000), (("A", "L"), 2, 3000), (("C", "L"), 2, 3000), (("C", "K"), 2, 3000)]),
    ("diary", "source", [(("C", "P"), 7, 1500), (("A", "O"), 7, 1500), (("K", "O"), 7, 1500), (("K", "P"), 7, 1500)]),
    ("wearable", "source", [(("O", "T"), 8, 2000), (("A", "T"), 8, 2000), (("E", "O", "T"), 8, 2000)]),
    ("psg", "source", [(("E", "O"), 8, 2500), (("E", "P"), 8, 2500), (("C", "E"), 8, 2500)]),
    ("vendor", "source", [(("C", "O", "P", "T"), 10, 4000)]),
    ("lab", "anchor", [(("C", "O", "P", "T"), 2, 1500)])]
def roster():
    """the list of (table id, family, role, ctx, n)."""
    out = []
    for fam, role, items in FAMILIES:
        for ctx, r, n in items:
            for k in range(r): out.append((f"{fam}:{'.'.join(ctx)}:{k}", fam, role, tuple(sorted(ctx, key=V.index)), n))
    return out
