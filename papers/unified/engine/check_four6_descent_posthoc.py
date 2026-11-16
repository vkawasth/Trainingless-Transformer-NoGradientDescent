"""Study 35, second addendum (post hoc, written after Study 35 ran; labelled so): leximin descent on the 2000-table discovery population.
  LZ9 RECORD (post hoc) per table: defect |merge(fine leximin) - coarse leximin|; within-group margin imbalance (largest relative spread of
              f or g inside a merged group); level purity of each block (all its fine cells fixed at one leximin level); Spearman
              correlation of defect with imbalance; descent rate among level-pure and level-mixed tables
Merges as in Study 35: rows {0,1},{2,3}; columns {0,1},{2,3},{4,5} (all blocks of size 4, so plain and block-average coarse leximin agree).
"""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "check_four6.py")).read(); exec(src.split("# LZ1 ----")[0])
RG = [[0, 1], [2, 3]]; CG = [[0, 1], [2, 3], [4, 5]]; cc, Ac, Nc, Dc = shape(2, 3)
defect = []; imb = []; pure = []
for t, (f, g) in enumerate(pop):
    S = facets_float(f, g, cells, A); lv, x, _, _ = leximin_float(f, g, cells, A, S); lvl = {}
    for k, (h, Sj) in enumerate(lv):
        for cl in Sj: lvl[cl] = k
    fc = [f[0] + f[1], f[2] + f[3]]; gc = [g[0] + g[1], g[2] + g[3], g[4] + g[5]]; Sc = facets_float(fc, gc, cc, Ac); _, xc, _, _ = leximin_float(fc, gc, cc, Ac, Sc)
    X = x.reshape(4, 6); M = np.array([[X[np.ix_(r_, c_)].sum() for c_ in CG] for r_ in RG]); defect.append(float(np.abs(M - xc.reshape(2, 3)).max()))
    F = np.array([float(v) for v in f]); G = np.array([float(v) for v in g])
    imb.append(max([np.ptp(F[r_]) / F[r_].mean() for r_ in RG] + [np.ptp(G[c_]) / G[c_].mean() for c_ in CG]))
    pure.append(all(len({lvl.get((i, j), -1) for i in r_ for j in c_}) == 1 for r_ in RG for c_ in CG))
defect = np.array(defect); imb = np.array(imb); pure = np.array(pure); z = defect <= 1e-9
rho = spearmanr(imb, defect)[0]
print(f"PASS LZ9 RECORD (post hoc) leximin descent on the discovery population   [descent {z.mean():.3f}; Spearman(defect, within-group imbalance) {rho:.2f}; "
      f"level-pure tables {pure.sum()} (descent among them {z[pure].mean() if pure.any() else float('nan'):.3f}); level-mixed {(~pure).sum()} (descent {z[~pure].mean():.3f}); "
      f"median imbalance among descending {np.median(imb[z]):.3f} vs failing {np.median(imb[~z]):.3f}]")
json.dump(dict(defect=defect.tolist(), imbalance=imb.tolist(), pure=pure.tolist(), rho=rho), open(os.path.join(HERE, "study35_four6", "descent_posthoc.json"), "w"))
print("1/1 passed")
