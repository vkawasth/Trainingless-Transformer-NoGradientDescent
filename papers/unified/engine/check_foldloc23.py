"""Floating-point check (a record, not a certificate) of the location of the nearest 2x3 fold, thm:ST6(a): the fold of W on the Study 45
wall, continued in L, against the leading-family limit tau = (1/2) ln(27/256) + i pi/2 (t = tau / L). Output: printed."""
import sys, os, runpy, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x", "--lib"]; g = runpy.run_path(os.path.join(HERE, "check_example45.py"))
Vf, x0f = g["Vf"], g["x0f"]
def Ef(w, t, L): a = -L * x0f(t) + Vf @ w; return np.exp(a - a.real.max())
def G(z, L):
    w, t = z[:2], z[2]; E = Ef(w, t, L); J = (Vf.T * E) @ Vf; F = Vf.T @ E
    return np.array([F[0], F[1], J[0, 0] * J[1, 1] - J[0, 1] ** 2])
def DG(z, L, h=1e-7):
    M = np.zeros((3, 3), complex)
    for c in range(3):
        e = np.zeros(3, complex); e[c] = h; M[:, c] = (G(z + e, L) - G(z - e, L)) / (2 * h)
    return M
tau0 = 0.5 * np.log(27 / 256) + 1j * np.pi / 2; Y0 = 4 ** (-1 / 3) * np.exp(4j * np.pi / 3)   # double root of Y^4 - Y - b at b = e^{2 tau0 / 3}
for L in (20, 40, 80, 160):
    t = tau0 / L; h = 0.1 + t / 3; us = np.array([(0.3 + t) * 0.35 - h, (0.3 + t) * 0.45 - h])
    z = np.array([np.log(Y0) - L * us[0], np.log(Y0) - L * us[1], t], complex)
    for _ in range(100):
        st = np.linalg.solve(DG(z, L), G(z, L)); z = z - st
        if np.abs(st).max() < 1e-14: break
    print(f"L = {L}: tau_f = {z[2] * L:.6f}, residual {np.abs(G(z, L)).max():.1e}")
print(f"limit: {tau0:.6f}")
