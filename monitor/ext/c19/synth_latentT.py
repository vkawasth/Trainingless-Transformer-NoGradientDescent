"""Synthetic check of latentT: planted latent first-order Markov (should fit) and planted second-order latent memory (should be
rejected), four waves, n = 800, answers observed with error."""
import numpy as np, time, sys
from latentT import test
def sim(n, second, rng):
    R = np.array([[0.8, 0.15, 0.05], [0.15, 0.7, 0.15], [0.05, 0.15, 0.8]])
    T1 = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
    Z = np.zeros((n, 4), int); Z[:, 0] = rng.choice(3, n, p=[0.4, 0.35, 0.25]); Z[:, 1] = [rng.choice(3, p=T1[z]) for z in Z[:, 0]]
    for t in (2, 3):
        for i in range(n):
            if second and Z[i, t - 2] != Z[i, t - 1]: Z[i, t] = Z[i, t - 2] if rng.random() < 0.7 else Z[i, t - 1]   # return to the earlier state
            else: Z[i, t] = rng.choice(3, p=T1[Z[i, t - 1]])
    X = np.array([[rng.choice(3, p=R[z]) for z in row] for row in Z]); N = np.zeros((3,) * 4); np.add.at(N, tuple(X.T), 1); return N
if __name__ == '__main__':
  rng = np.random.default_rng(3); ok = True
  for second in (False, True):
      t0 = time.time(); r = test(sim(800, second, rng), 4, shared=True, B=60, seed=1, starts=8)
      print('planted', 'second-order memory' if second else 'first-order Markov', r, f'{time.time()-t0:.0f}s'); ok &= (r['p_boot'] < 0.05) == second
  print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
