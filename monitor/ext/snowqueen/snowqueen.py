"""Positive control: the Snow Queen experiment (Cervantes and Dzhafarov 2018, arXiv:1711.00418, Tables 2 and 3), a cyclic system of
rank 4 with direct influences, reported contextual by Contextuality-by-Default. We recompute it with the programme's CbD code
(bridges/cbd/cbd.py: closed-form criterion and linear program) from the published proportions and sample sizes.
Contents: q0 = Gerda/Troll, q1 = Beautiful/Unattractive, q2 = Snow Queen/Old Finn Woman, q3 = Kind/Evil (first option = +1).
Contexts in cyclic order: c0 = (q0,q1) = paper's context 1; c1 = (q1,q2) = context 3 transposed; c2 = (q2,q3) = context 4;
c3 = (q3,q0) = context 2 transposed."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'open_clinical', 'lib')); from cbd import criterion, lp_noncontextual, cntx
# published proportions (rows: character, columns: characteristic), paper contexts 1-4
T2 = {1: [[0.887, 0.000], [0.000, 0.113]], 2: [[0.841, 0.000], [0.000, 0.159]], 3: [[0.837, 0.000], [0.000, 0.163]], 4: [[0.000, 0.626], [0.374, 0.000]]}
N2 = {1: 425, 2: 429, 3: 410, 4: 388}                        # correct responses
T3 = {1: [[0.843, 0.020], [0.029, 0.107]], 2: [[0.797, 0.035], [0.018, 0.150]], 3: [[0.769, 0.011], [0.070, 0.150]], 4: [[0.135, 0.536], [0.320, 0.009]]}
N3 = {1: 447, 2: 453, 3: 446, 4: 453}                        # all responses
def cyc(T): T = {k: np.array(v, float) / np.sum(v) for k, v in T.items()}; return [T[1], T[3].T, T[4], T[2].T]
def lhs(tabs): c, s, D = criterion(tabs); return s - 2 - D
def analyse(T, N, rng, B=100000):
    tabs = cyc(T); ns = [N[1], N[3], N[4], N[2]]; c, s, D = criterion(tabs); L = s - 2 - D
    boots = np.array([lhs([rng.multinomial(n, t.ravel()).reshape(2, 2) / n for t, n in zip(tabs, ns)]) for _ in range(B)])
    return dict(s_odd=round(s, 4), Delta=round(D, 4), LHS=round(L, 4), CNTX=round(cntx(tabs), 4), contextual_criterion=bool(c), contextual_LP=not lp_noncontextual(tabs),
                bootstrap_CI_99_99=[round(float(x), 3) for x in np.percentile(boots, [0.005, 99.995])], share_LHS_le_0=float(np.mean(boots <= 0)), B=B)
if __name__ == '__main__':
    rng = np.random.default_rng(20261010)
    res = dict(source='Cervantes & Dzhafarov (2018), arXiv:1711.00418, Tables 2-3', paper_reported=dict(correct_LHS=0.452, correct_CI=[0.226, 0.668], all_LHS=0.279, all_CI=[0.008, 0.506]),
               correct_responses=analyse(T2, N2, rng), all_responses=analyse(T3, N3, rng))
    json.dump(res, open('results_snowqueen.json', 'w'), indent=1); print(json.dumps(res, indent=1))
