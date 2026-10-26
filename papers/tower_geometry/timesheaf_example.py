"""The worked example of Section 17: three drugs, five times, covers chosen by robust satisficing.

Truth (Section 2, conflicting pair trials): two patient types, susceptibilities a_1 = (0.5, 0.1, 0.4), a_2 = (0.05, 0.5, 0.1) to the harms
H_A, H_B, H_C; population 1 has type weights (0.6, 0.4), population 2 has (0.25, 0.75).

  t0  AB and AC trials (population 1): a tree, it glues
  t1  + BC trial (population 2): a cycle; the trials conflict (lam* = 0.86)
  t2  awareness growth: the harm of C is split into mild and severe (H_C in {0, mild, severe}); the AC and BC tables are refined, and
      coarsening them gives back t1 exactly (no drift). Severe shares differ between populations (0.3 in population 1, 0.6 in 2)
  t3  drift: a second AB batch from population 2 is pooled with the first (weights 0.7, 0.3)
  t4  a biomarker D is measured with B in population 1 (a pendant window: new variable, attached at B)

Acts: give the triple A+B+C, give the pair A+B, give nothing. Utilities (benefit minus harm): triple 1 - 1.2 [any harm] - 0.5 [severe C harm],
pair 0.75 - 1.2 [harm from A or B], nothing 0. Requirement for "good enough": expected utility >= u0 = 0.05.
Decision plane: x = P(any harm of the triple), y = P(harm from A or B).
"""
import numpy as np
from pce import Window
from pce.level2 import lcm_law
from pce.timesheaf import System

A_TYPES = np.array([[0.5, 0.1, 0.4], [0.05, 0.5, 0.1]])
POP1 = lcm_law(np.array([0.6, 0.4]), A_TYPES); POP2 = lcm_law(np.array([0.25, 0.75]), A_TYPES)      # axes (H_A, H_B, H_C), binary
SEVERE = {1: 0.3, 2: 0.6}
U0 = 0.05; COST = 1.2; COST_SEVERE = 0.5


def refine_C(P, share):
    """split H_C = 1 into mild (1) and severe (2) with the given severe share"""
    R = np.zeros(P.shape[:-1] + (3,)); R[..., 0] = P[..., 0]; R[..., 1] = P[..., 1] * (1 - share); R[..., 2] = P[..., 1] * share; return R


def steps():
    D2 = {"A": 2, "B": 2, "C": 2}; D3 = {"A": 2, "B": 2, "C": 3}; D4 = {"A": 2, "B": 2, "C": 3, "D": 2}
    AB1 = POP1.sum(axis=2); AC1 = POP1.sum(axis=1); BC2 = POP2.sum(axis=0)
    AC1r = refine_C(POP1, SEVERE[1]).sum(axis=1); BC2r = refine_C(POP2, SEVERE[2]).sum(axis=0)
    AB3 = 0.7 * AB1 + 0.3 * POP2.sum(axis=2)
    pB = AB3.sum(axis=0); kD = np.array([[0.8, 0.2], [0.3, 0.7]])                       # P(D | H_B)
    BD = pB[:, None] * kD
    S = [System(D2, [Window(("A", "B"), AB1), Window(("A", "C"), AC1)], ["AB", "AC"]),
         System(D2, [Window(("A", "B"), AB1), Window(("A", "C"), AC1), Window(("B", "C"), BC2)], ["AB", "AC", "BC"]),
         System(D3, [Window(("A", "B"), AB1), Window(("A", "C"), AC1r), Window(("B", "C"), BC2r)], ["AB", "AC", "BC"]),
         System(D3, [Window(("A", "B"), AB3), Window(("A", "C"), AC1r), Window(("B", "C"), BC2r)], ["AB", "AC", "BC"]),
         System(D4, [Window(("A", "B"), AB3), Window(("A", "C"), AC1r), Window(("B", "C"), BC2r), Window(("B", "D"), BD)], ["AB", "AC", "BC", "BD"])]
    labels = ["t0: AB, AC", "t1: + BC (other population)", "t2: C harm refined", "t3: AB drifts", "t4: + biomarker BD"]
    return S, labels


REFINE_C = {"C": {0: 0, 1: 1, 2: 1}}
any_harm = lambda x: float(x["A"] or x["B"] or x["C"] > 0)
pair_harm = lambda x: float(x["A"] or x["B"])
severe = lambda x: float(x["C"] == 2)
ACTS = {"triple A+B+C": lambda x: 1.0 - COST * any_harm(x) - COST_SEVERE * severe(x),
        "pair A+B": lambda x: 0.75 - COST * pair_harm(x),
        "nothing": lambda x: 0.0}
