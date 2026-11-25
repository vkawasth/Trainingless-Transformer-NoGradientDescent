"""CALIBRATION (Study 92 Part E1): bifurcation points of the 2x3 mirror on a generic complex line through the Study 83 base point.
la(s) = la0 + s d (log coefficients), d random (calibration seed 9211). M(s) = prod_{a<b} (v_a - v_b)^2 is single-valued and holomorphic
in s; its zeros in the disc are the Maxwell points (order 2) and the A2 caustics (order 3). Located by the argument principle on a polar
grid; exploratory, not a check."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np, mirror83 as M
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 9211; R = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
rng = np.random.default_rng(SEED); a0 = M.coeffs(M.BASE_F, M.BASE_G, 20.0).astype(complex); d = a0 * (rng.normal(size=6) + 1j * rng.normal(size=6)) / np.sqrt(2)
la0 = np.log(M.coeffs(M.BASE_F, M.BASE_G, 20.0)).astype(complex); u0 = M.newton(M.u_base().astype(complex), la0)
def cont(u, s0, s1, n=8):
    for q in range(1, n + 1):
        s = s0 + (s1 - s0) * q / n; u = M.newton(u, np.log(a0 + s * d), it=8)
    return u
def Mfun(u, s):
    v = M.values(u, np.log(a0 + s * d)); return np.prod([(v[a] - v[b]) ** 2 for a in range(6) for b in range(a + 1, 6)])
NT, NR = 180, 120; th = 2 * np.pi * np.arange(NT) / NT; rr = R * np.arange(1, NR + 1) / NR; t0 = time.time()
G = np.zeros((NR + 1, NT), complex); U = {}
for k, a in enumerate(th):
    u = u0.copy(); s_prev = 0.0; G[0, k] = Mfun(u, 0.0)
    for q, r in enumerate(rr):
        s = r * np.exp(1j * a); u = cont(u, s_prev, s); s_prev = s; G[q + 1, k] = Mfun(u, s)
print(f"grid done {time.time() - t0:.0f}s", flush=True)
# argument principle on each cell (polar grid): winding of M around the cell boundary
zeros = []
for q in range(NR):
    for k in range(NT):
        k1 = (k + 1) % NT; ring = [G[q, k], G[q + 1, k], G[q + 1, k1], G[q, k1], G[q, k]]
        w = sum(np.angle(ring[i + 1] / ring[i]) for i in range(4)) / (2 * np.pi)
        if abs(w) > 0.5: zeros.append((q, k, int(round(w)), (rr[q] if q < NR else R) * np.exp(1j * (th[k] + np.pi / NT))))
print("cells with zeros:", len(zeros), "orders:", sorted({z[2] for z in zeros}), "total order", sum(z[2] for z in zeros))
for z in zeros: print(f"  s ~ {z[3]:.3f}  order {z[2]}")
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"e1aff_cal_{SEED}.npy"), dict(d=d, R=R, zeros=zeros), allow_pickle=True)
