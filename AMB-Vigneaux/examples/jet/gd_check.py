"""Counterexample check, not part of the engine: plain gradient ascent on the no-three-way log-linear model of the
JetClass-II (pT x |eta| x cell) grid reaches the same flat-transport fit, hence the same holonomy, as IPF.
Gradient descent is not unable to see the geometry; the engine avoids it for exactness, certificates and singular models."""
import os, sys, importlib.util, numpy as np
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, "."); from amb_vigneaux import infogeo2 as G
spec = importlib.util.spec_from_file_location("J", "examples/jets/run.py"); J = importlib.util.module_from_spec(spec); spec.loader.exec_module(J)
df = J.load(); C = np.zeros((8, 4, 16)); np.add.at(C, (df.ptb.values, df.etab.values, df.cell.values), 1); N = C.sum()
M_ipf = G.flat_fit(C)
a = np.zeros((8, 4)); b = np.zeros((8, 16)); c = np.zeros((4, 16)); lr = 0.5
for it in range(20000):                                   # Poisson log-likelihood, plain full-batch gradient ascent
    mu = np.exp(a[:, :, None] + b[:, None, :] + c[None, :, :]) * N / 1e3
    R = (C - mu) / N
    a += lr * R.sum(2) * 1e1; b += lr * R.sum(1) * 1e1; c += lr * R.sum(0) * 1e1
def G2(M):
    m = C > 0; return 2 * float((C[m] * np.log(C[m] / M[m])).sum())
print(f"G (holonomy statistic): IPF {G2(M_ipf):.2f}   gradient descent {G2(mu):.2f}   max |mu_gd - mu_ipf| / mu_ipf {np.max(np.abs(mu - M_ipf) / M_ipf):.2e}")
print(f"holonomy information: IPF {G2(M_ipf) / (2 * N) * 1e3:.3f}e-3   GD {G2(mu) / (2 * N) * 1e3:.3f}e-3 nats/jet")
