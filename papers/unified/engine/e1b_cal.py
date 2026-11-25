"""CALIBRATION (Study 92 Part E1, step 2): braids of the loops around the bifurcation points found by e1_cal.py. Clusters of grid cells
are merged; each loop = straight ray from s = 0 to the circle of radius rho around the cluster centre, the circle counter-clockwise,
and back. The braid word of the six critical values is read from crossings of their projections Re(e^{i theta} v) (Artin generators,
sign by the imaginary order). Exploratory."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np, mirror83 as M
HERE = os.path.dirname(os.path.abspath(__file__)); SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 9211
D = np.load(os.path.join(HERE, f"e1_cal_{SEED}.npy"), allow_pickle=True).item(); d = D["d"]
la0 = np.log(M.coeffs(M.BASE_F, M.BASE_G, 20.0)).astype(complex); u0 = M.newton(M.u_base().astype(complex), la0); THETA = 0.1
def clusters(zs, tol=0.25):
    out = []
    for z in zs:
        for c in out:
            if min(abs(z[3] - w) for w in c) < tol: c.append(z[3]); break
        else: out.append([z[3]])
    return [np.mean(c) for c in out]
def Mval(u, s):
    v = M.values(u, la0 + s * d); return np.prod([(v[a] - v[b]) ** 2 for a in range(6) for b in range(a + 1, 6)])
def walk(u, pts, sub=6):
    """continue along a polyline, returning the sequence of (s, u, values) at fine steps (adaptive by value motion)"""
    out = [(pts[0], u, M.values(u, la0 + pts[0] * d))]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(sub, int(abs(b - a) / 0.004))
        for q in range(1, n + 1):
            s = a + (b - a) * q / n; u = M.newton(u, la0 + s * d, it=10); out.append((s, u, M.values(u, la0 + s * d)))
    return out
def word_of(seq):
    """Artin word from projection crossings; strands labelled continuously (critical points continued)"""
    rot = np.exp(1j * THETA); w = []
    for (_, _, va), (_, _, vb) in zip(seq[:-1], seq[1:]):
        a, b = (rot * va).real, (rot * vb).real; oa = list(np.argsort(a))
        for i in range(6):
            for j in range(i + 1, 6):
                if np.sign(a[i] - a[j]) != np.sign(b[i] - b[j]):
                    pi, pj = oa.index(i), oa.index(j)
                    if abs(pi - pj) != 1: return None                                  # step too coarse: non-adjacent crossing
                    lo, hi = (i, j) if pi < pj else (j, i); w.append((min(pi, pj) + 1, 1 if (rot * vb[lo]).imag < (rot * vb[hi]).imag else -1))
    return w
def free_reduce(w):
    out = []
    for g in w:
        if out and out[-1][0] == g[0] and out[-1][1] == -g[1]: out.pop()
        else: out.append(g)
    return out
def perm_of(w):
    p = list(range(6))
    for k, e in w: p[k - 1], p[k] = p[k], p[k - 1]
    return p
cs = clusters(D["zeros"]); rho = 0.2
for c in cs:
    th = np.linspace(0, 2 * np.pi, 241); circ = c + rho * np.exp(1j * (np.angle(-c) + th))   # start on the side facing s = 0
    start = c + rho * np.exp(1j * np.angle(-c))
    seq = walk(u0, [0.0, start]); u1 = seq[-1][1]; seq2 = walk(u1, list(circ)); seq3 = walk(seq2[-1][1], [start, 0.0])
    full = seq + seq2[1:] + seq3[1:]; w = word_of(full)
    back = np.abs(np.sort_complex(full[-1][2]) - np.sort_complex(full[0][2])).max()
    Ms = [Mval(x[1], x[0]) for x in seq2]; wnum = sum(np.angle(Ms[i + 1] / Ms[i]) for i in range(len(Ms) - 1)) / (2 * np.pi)
    if w is None: print(f"cluster {c:.3f}: step too coarse"); continue
    wr = free_reduce(w); pm = perm_of(wr)
    print(f"cluster {c:.3f}: winding of M {wnum:+.2f}; word length {len(w)} -> {len(wr)}; exponent sum {sum(e for _, e in wr)}; perm {pm}; returns {back:.1e}")
    print("   ", wr)
