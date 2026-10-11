"""Planted-contextuality check of the pipeline used in analyses D (canonical CbD over all dichotomizations, cbd_split, with the
boundary-null p-value). The Snow Queen control (snowqueen.py) exercises the rank-n cyclic criterion and LP; this exercises the pipeline
that produced the clinical and panel non-detections.

Two contexts each measure the 3x3 contents (D, A). Context X has the joint P (a depression-anxiety table with the strength of
association seen in the panels). Context Y has the joint Q = shift(P) + eps * M, where shift() moves the margins (direct influence, a
naive contextual fraction near the panels' 0.15) and M preserves the margins and weakens the coupling. eps = 0 is compatible (no
contextuality): its rejection rate is the size. Larger eps plants a known population deficit. Sample sizes are those of the panels.
Usage: python3 cbd_split_power.py  -> results_cbd_split_power.json"""
import os, sys, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, '..', 'open_clinical', 'lib'))
from mimic_monitor import cbd_split, boundary_null_p

P = np.array([[0.40, 0.07, 0.01], [0.08, 0.14, 0.05], [0.01, 0.06, 0.18]]); P /= P.sum()
def shifted(P, a=0.18):                                        # same copula-like pattern, more mass in higher bands (direct influence)
    r = np.array([1 - a, 1.0, 1 + 2 * a]); Q = P * r[:, None] * r[None, :]; return Q / Q.sum()
M = np.array([[-1, 1, 0], [1, -1, 0], [0, 0, 0]], float) + np.array([[0, 0, 0], [0, -1, 1], [0, 1, -1]], float)   # margin preserving
def plant(eps, shift=True):
    Q = shifted(P) if shift else P.copy(); Q = Q + eps * M; assert Q.min() > 0; return Q / Q.sum()

def run(eps, shift, nX, nY, reps, B, rng):
    Q = plant(eps, shift); pop = cbd_split(P, Q)['deficit']; naive = 0.5 * np.abs(P - Q).sum(); rej = 0; ps = []
    for _ in range(reps):
        TX = rng.multinomial(nX, P.ravel()).reshape(3, 3) + 1e-12; TY = rng.multinomial(nY, Q.ravel()).reshape(3, 3) + 1e-12
        TX, TY = TX / TX.sum(), TY / TY.sum(); d = cbd_split(TX, TY)['deficit']
        p = boundary_null_p(TX, TY, nX, nY, d, rng, B) if d > 0 else 1.0; ps.append(p); rej += p < 0.05
    return dict(eps=eps, margins_shifted=shift, nX=nX, nY=nY, population_deficit=round(float(pop), 4), naive_cf=round(float(naive), 3),
                reps=reps, B=B, rejections=int(rej), rate=round(rej / reps, 3))

if __name__ == '__main__':
    rng = np.random.default_rng(20261010); out = []
    for nX, nY in ((500, 500), (1200, 400)):
        for shift in (False, True):
            for eps in (0.0, 0.0025, 0.005, 0.01, 0.02):
                r = run(eps, shift, nX, nY, reps=40, B=100, rng=rng); out.append(r); print(r, flush=True)
    json.dump(out, open(os.path.join(H, 'results_cbd_split_power.json'), 'w'), indent=1)
