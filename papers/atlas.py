"""atlas.py -- the Posterior Tower Atlas as a script: the Hardy-Weinberg example in seven figures for the paper.

Writes PDF figures into ./figs:
  fig_algebra.pdf   two poolings on a 2x2 table: orders at point 1 and the alternating-projection rate (eps = 1/2)
  fig_tower.pdf     the tower as a martingale in the simplex, with the energies E||M_k||^2
  fig_secant.pdf    chords through rho (the 2-mixture fibre) and the fibre in (theta1, theta2) coordinates
  fig_tropical.pdf  log_t-scaled mixtures at t = 0.3, 1e-2, 1e-6 converging onto the tropical secant
  fig_walls.pdf     posterior maps at n = 3, 30, 300 with the zero-temperature walls
  fig_fibres.pdf    tropical fibres (generic and collision) with their assignment-cell covers and nerves
  fig_betti.pdf     b0 of the tropical fibre over the target chart (requires betti_scan.json from fibre_scan.py)
All numbers are computed here from the formulas in the paper; nothing is sampled from data.
"""
import json, os, sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fibre_topology import nerve, betti, rates, ALPHAS, SIGNS, cell_constraints
from scipy.optimize import linprog
from fractions import Fraction as F

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"); os.makedirs(OUT, exist_ok=True)
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"; ACC = "#3b4fb8"; INK = "#151a22"; INK3 = "#7b8597"; GRID = "#e3e7ee"; TROP = "#9aa3b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK3, "axes.labelcolor": INK, "xtick.color": INK3,
                     "ytick.color": INK3, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 200})
HW = lambda th: np.array([th * th, 2 * th * (1 - th), (1 - th) ** 2])
def T(p): return np.array([p[1] + p[2] / 2, np.sqrt(3) / 2 * p[2]])
def tri(ax):
    V = [T(e) for e in np.eye(3)]
    ax.plot(*np.array(V + [V[0]]).T, color=INK3, lw=0.8)
    for i, (dx, dy) in enumerate(((-0.07, -0.05), (0.02, -0.05), (-0.03, 0.02))): ax.text(V[i][0] + dx, V[i][1] + dy, f"x={i}", color=INK3, fontsize=7.5)
    ax.set_aspect("equal"); ax.axis("off")
def curve(ax, **kw):
    P = np.array([T(HW(t)) for t in np.linspace(0, 1, 200)]); ax.plot(*P.T, **kw)

# ------------------------------------------------------------------ algebra
def fig_algebra(eps=0.5):
    p = np.array([(1 + eps) / 4, (1 - eps) / 4, (1 - eps) / 4, (1 + eps) / 4]); A, B = [0, 0, 1, 1], [0, 1, 0, 1]
    def E(lab):
        M = np.zeros((4, 4))
        for i in range(4):
            cell = [j for j in range(4) if lab[j] == lab[i]]; M[i, cell] = p[cell] / p[cell].sum()
        return M
    EA, EB, EM = E(A), E(B), E([0] * 4); Z = np.arange(1, 5.)
    D = np.diag(np.sqrt(p)); Di = np.diag(1 / np.sqrt(p))
    norms = [np.linalg.norm(D @ (np.linalg.matrix_power(EA @ EB, n) - EM) @ Di, 2) for n in range(1, 9)]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw={"width_ratios": [1, 1.4]})
    ax[0].axis("off")
    ax[0].text(0, 0.95, f"weights p = ({', '.join(f'{x:.3f}' for x in p)})", fontsize=8, family="monospace", transform=ax[0].transAxes)
    ax[0].text(0, 0.75, f"(E_cols E_rows Z)(1) = {(EB @ EA @ Z)[0]:.4f}", fontsize=8, family="monospace", transform=ax[0].transAxes)
    ax[0].text(0, 0.60, f"(E_rows E_cols Z)(1) = {(EA @ EB @ Z)[0]:.4f}", fontsize=8, family="monospace", transform=ax[0].transAxes)
    sv = np.linalg.svd(2 * np.array([[p[0], p[1]], [p[2], p[3]]]), compute_uv=False)
    ax[0].text(0, 0.45, f"singular values of K: {sv[0]:.2f}, {sv[1]:.2f}", fontsize=8, family="monospace", transform=ax[0].transAxes)
    ax[0].text(0, 0.25, "meet trivial; not independent,\nso the orders differ", fontsize=8, transform=ax[0].transAxes, color=INK3)
    n = np.arange(1, 9)
    ax[1].semilogy(n, norms, "o-", color=S1, lw=1.6, ms=4, label="computed")
    ax[1].semilogy(n, eps ** (2 * n - 1), "--", color=INK3, lw=1, label=r"$c^{2n-1}$, $c=|\varepsilon|$")
    ax[1].set_xlabel("n"); ax[1].set_ylabel(r"$\|(E_AE_B)^n-E_{A\wedge B}\|$"); ax[1].grid(color=GRID, lw=0.6); ax[1].legend(frameon=False, fontsize=7.5)
    fig.savefig(os.path.join(OUT, "fig_algebra.pdf")); plt.close(fig)
    return (EB @ EA @ Z)[0], (EA @ EB @ Z)[0], norms

# ------------------------------------------------------------------ tower
TH = [0.2, 0.5, 0.8]; ATOMS = [HW(t) for t in TH]; PIBAR = np.array([0.35, 0.3, 0.35]); DIR = np.array([0.25, 0, -0.25])
def fig_tower(s=0.7):
    P1, P2 = PIBAR + 1.36 * s * DIR, PIBAR - 1.36 * s * DIR
    rho = PIBAR @ np.array(ATOMS); m1 = [P1 @ np.array(ATOMS), P2 @ np.array(ATOMS)]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
    a = ax[0]; tri(a); curve(a, color=GRID, lw=1.5)
    V = [T(e) for e in np.eye(3)]
    for at in ATOMS:
        for x in range(3): a.plot(*np.array([T(at), V[x]]).T, color=INK3, lw=0.6, alpha=0.15 + 0.6 * at[x])
    for j, m in enumerate(m1):
        for i, at in enumerate(ATOMS): a.plot(*np.array([T(m), T(at)]).T, color=ACC, lw=0.8, alpha=min(1, 0.15 + 1.2 * [P1, P2][j][i]))
        a.plot(*np.array([T(rho), T(m)]).T, color=ACC, lw=1.2)
    for x in range(3): a.scatter(*V[x], s=40 + 300 * rho[x], marker="s", color=INK3, alpha=0.6, zorder=3)
    for i, at in enumerate(ATOMS): a.scatter(*T(at), s=40 + 500 * PIBAR[i], color=[S1, S2, S3][i], edgecolor="white", lw=1.2, zorder=4)
    for m in m1: a.scatter(*T(m), s=40, marker="D", color=ACC, edgecolor="white", zorder=5)
    a.scatter(*T(rho), s=45, color=INK, edgecolor="white", zorder=6)
    E = [rho @ rho, 0.5 * (m1[0] @ m1[0] + m1[1] @ m1[1]), sum(PIBAR[i] * ATOMS[i] @ ATOMS[i] for i in range(3)), 1.0]
    b = ax[1]; names = [r"$M_0=\rho$", r"$M_1$", r"$M_2=\Pi$", r"$M_3=D(\eta)\rho$"]
    b.bar(range(4), E, color=[INK3, ACC, INK3, INK3], alpha=0.85, width=0.6)
    for k, e in enumerate(E): b.text(k, e + 0.015, f"{e:.3f}", ha="center", fontsize=7.5)
    b.set_xticks(range(4)); b.set_xticklabels(names, fontsize=7.5); b.set_ylim(0.3, 1.08); b.set_ylabel(r"$E\|M_k\|_2^2$"); b.grid(axis="y", color=GRID, lw=0.6)
    leg = [Line2D([], [], marker="o", ls="", color=INK, label=r"$M_0$"), Line2D([], [], marker="D", ls="", color=ACC, label=r"$M_1$"),
           Line2D([], [], marker="o", ls="", color=S2, label=r"$M_2$ atoms"), Line2D([], [], marker="s", ls="", color=INK3, label=r"$M_3$ vertices")]
    a.legend(handles=leg, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(-0.05, 1.02))
    fig.savefig(os.path.join(OUT, "fig_tower.pdf")); plt.close(fig)
    return rho, m1, E

# ------------------------------------------------------------------ secant
def secant_fibre(lam):
    rho = (1 - lam) * HW(0.5) + lam * np.array([0.5, 0, 0.5]); P = T(rho)
    cr = lambda u, v: u[0] * v[1] - u[1] * v[0]; fib = []
    grid = np.linspace(0, 1, 601)
    for t1 in grid[1:-1]:
        A = T(HW(t1)); u = A - P; vals = np.array([cr(u, T(HW(t2)) - P) for t2 in grid])
        for j in range(len(grid) - 1):
            if np.sign(vals[j]) != np.sign(vals[j + 1]) and vals[j] != 0:
                lo, hi = grid[j], grid[j + 1]
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if np.sign(cr(u, T(HW(mid)) - P)) == np.sign(cr(u, T(HW(lo)) - P)): lo = mid
                    else: hi = mid
                t2 = (lo + hi) / 2; vv = T(HW(t2)) - P
                if u @ vv < 0 and abs(t2 - t1) > 1e-3:
                    fib.append((t1, t2, np.linalg.norm(vv) / (np.linalg.norm(vv) + np.linalg.norm(u))))
    return rho, np.array(fib)
def fig_secant(lams=(0.45, 0.08)):
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1, 1, 1]})
    rho, fib = secant_fibre(lams[0])
    a = ax[0]; tri(a)
    P = np.array([T(HW(t)) for t in np.linspace(0, 1, 200)]); a.fill(*P.T, color=S1, alpha=0.1, lw=0); curve(a, color=S1, lw=1.4)
    for t1, t2, w in fib[:: max(1, len(fib) // 14)]: a.plot(*np.array([T(HW(t1)), T(HW(t2))]).T, color=S2, lw=0.7, alpha=0.8)
    a.scatter(*T(rho), color=INK, s=20, zorder=5); a.set_title(r"chords through $\rho$", fontsize=8)
    for k, lam in enumerate(lams):
        _, fb = secant_fibre(lam); b = ax[k + 1]
        b.plot([0, 1], [0, 1], "--", color=INK3, lw=0.8); b.scatter(fb[:, 0], fb[:, 1], s=1.2, color=S2)
        b.set_xlim(0, 1); b.set_ylim(0, 1); b.set_aspect("equal"); b.set_xlabel(r"$\theta_1$"); b.set_ylabel(r"$\theta_2$"); b.grid(color=GRID, lw=0.5)
        b.set_title(f"fibre, distance {lam}", fontsize=8)
    fig.savefig(os.path.join(OUT, "fig_secant.pdf")); plt.close(fig)
    return len(fib)

# ------------------------------------------------------------------ tropical
def fig_tropical():
    rng = np.random.default_rng(7); S = np.column_stack([rng.uniform(-3, 3, 900), rng.uniform(-3, 3, 900), rng.uniform(-4, 4, 900)])
    u = lambda d: np.array([0, d, 2 * d]) - min(0, 2 * d)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    tr = np.array([(lambda v: (v[1] - v[0], v[2] - v[0]))(np.minimum(u(d1), b + u(d2))) for d1, d2, b in S])
    ax.scatter(tr[:, 0], tr[:, 1], s=5, color=TROP, alpha=0.55, lw=0, label=r"tropical secant $\mathrm{Mix}_2^{\mathrm{T}}$")
    ax.plot([-3, 3.5], [-6, 7], color=INK, lw=1.6, label="tropical curve")
    maxdev = {}
    for t, col in ((0.3, "#b9c6ee"), (1e-2, "#6f86d6"), (1e-6, "#2f3f9c")):
        lg = np.log(t); sh = np.log(2) / lg
        ax.plot([-3 + sh, 3.5 + sh], [-6, 7], "--", color=col, lw=1)
        def r(d):
            z = t ** d; den = (1 + z) ** 2; return np.array([1 / den, 2 * z / den, z * z / den])
        pts = []
        for d1, d2, b in S[:260]:
            z = t ** b; p = r(d1) / (1 + z) + z / (1 + z) * r(d2); pts.append((np.log(p[1] / p[0]) / lg, np.log(p[2] / p[0]) / lg))
        pts = np.array(pts); ax.scatter(pts[:, 0], pts[:, 1], s=6, color=col, lw=0, label=f"t = {t:g}")
        maxdev[t] = float(np.max(np.abs(pts - tr[:260])))
    ax.set_xlim(-4, 4); ax.set_ylim(-6, 7); ax.grid(color=GRID, lw=0.5)
    ax.set_xlabel(r"$\log_t(p_1/p_0)$"); ax.set_ylabel(r"$\log_t(p_2/p_0)$"); ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(os.path.join(OUT, "fig_tropical.pdf")); plt.close(fig)
    return maxdev

# ------------------------------------------------------------------ walls
def fig_walls(ns=(3, 30, 300)):
    lr = np.log(np.array(ATOMS)); lp = np.log(PIBAR); cols = np.array([[0x2a, 0x78, 0xd6], [0xeb, 0x68, 0x34], [0x1b, 0xaf, 0x7a]]) / 255
    N = 260; fig, ax = plt.subplots(1, len(ns), figsize=(7.0, 2.4))
    for k, n in enumerate(ns):
        img = np.ones((N, N, 4)); img[..., 3] = 0; hard = -np.ones((N, N), int)
        for iy in range(N):
            for ix in range(N):
                q2 = iy / (N - 1) / (np.sqrt(3) / 2); q1 = ix / (N - 1) - q2 / 2; q0 = 1 - q1 - q2
                if q2 > 1 or min(q0, q1, q2) < 0: continue
                q = np.array([q0, q1, q2]); sc = lp + n * (lr @ q); e = np.exp(sc - sc.max()); e /= e.sum()
                img[iy, ix, :3] = e @ cols; img[iy, ix, 3] = 1; hard[iy, ix] = int(np.argmax(lr @ q))
        a = ax[k]; a.imshow(img, origin="lower", extent=(0, 1, 0, np.sqrt(3) / 2), interpolation="nearest")
        wall = np.zeros((N, N), bool); wall[:, 1:] |= (hard[:, 1:] != hard[:, :-1]) & (hard[:, 1:] >= 0) & (hard[:, :-1] >= 0)
        wall[1:, :] |= (hard[1:, :] != hard[:-1, :]) & (hard[1:, :] >= 0) & (hard[:-1, :] >= 0)
        yy, xx = np.nonzero(wall); a.scatter(xx / (N - 1), yy / (N - 1) * np.sqrt(3) / 2, s=0.15, color=INK)
        tri(a); a.set_title(f"n = {n}  (t = e$^{{-{n}}}$)", fontsize=8)
    fig.savefig(os.path.join(OUT, "fig_walls.pdf")); plt.close(fig)

# ------------------------------------------------------------------ fibres and nerves
def cell_segment(C):
    Aeq, beq, Aub, bub = C; pts = []
    for d in (np.array([0, 1, 0.3]), np.array([0, -1, -0.3]), np.array([1, 0.2, 0.1]), np.array([-1, -0.2, -0.1]), np.array([0.1, 0.3, 1]), np.array([-0.1, -0.3, -1])):
        res = linprog(d, A_ub=np.array(Aub, float), b_ub=np.array(bub, float), A_eq=np.array(Aeq, float), b_eq=np.array(beq, float),
                      bounds=[(-5, 5)] * 3, method="highs")
        if res.status == 0: pts.append(res.x)
    return np.array(pts)
def fig_fibres(cases):
    fig, ax = plt.subplots(2, len(cases), figsize=(7.0, 6.0), gridspec_kw={"hspace": 0.45})
    info = []
    for k, (name, v) in enumerate(cases):
        verts, simp, C = nerve(v, 2); b = betti(simp, 1); info.append((name, [str(x) for x in v], len(verts), len(simp[1]), len(simp[2]), b))
        a = ax[0, k]; palette = plt.cm.tab20(np.linspace(0, 1, max(1, len(verts))))
        for i, c in enumerate(verts):
            P = cell_segment(C[c])
            if len(P) == 0: continue
            hull = P[np.lexsort((P[:, 2], P[:, 1]))]
            a.plot(hull[:, 1], hull[:, 2], "-o", color=palette[i], lw=1.4, ms=2)
        a.set_xlim(-5.2, 5.2); a.set_ylim(-5.2, 5.2); a.set_aspect("equal"); a.grid(color=GRID, lw=0.5)
        a.set_xlabel(r"$d_1$"); a.set_ylabel(r"$d_2$"); a.set_title(f"{name}: $v$=({', '.join(str(x) for x in v)})", fontsize=8)
        g = ax[1, k]; ang = np.linspace(0, 2 * np.pi, len(verts), endpoint=False); pos = {c: (np.cos(t), np.sin(t)) for c, t in zip(verts, ang)}
        for s in simp[1]: g.plot(*np.array([pos[c] for c in s]).T, color=INK3, lw=0.5, alpha=0.9 if len(simp[1]) < 20 else 0.18)
        for i, c in enumerate(verts): g.scatter(*pos[c], color=palette[i], s=22, zorder=3)
        g.set_aspect("equal"); g.axis("off"); g.set_title(f"nerve: {len(verts)} cells, {len(simp[1])} edges\n{len(simp[2])} triangles;  $b_0={b[0]}$, $b_1={b[1]}$", fontsize=7.5)
    fig.savefig(os.path.join(OUT, "fig_fibres.pdf")); plt.close(fig)
    return info

def fig_betti(path):
    data = json.load(open(path)); fig, ax = plt.subplots(figsize=(6.0, 3.6))
    cmap = {0: "#ffffff00", 1: S3, 2: S1}
    for a_, c_, b, nc in data:
        a_, c_ = float(F(a_)), float(F(c_))
        if b[0] == 0: ax.scatter(a_, c_, s=18, facecolor="none", edgecolor=GRID, lw=0.6)
        else: ax.scatter(a_, c_, s=26, color=cmap.get(b[0], S2), marker="s" if b[1] == 0 else "*")
    ax.plot([-3, 3.5], [-6, 7], color=INK, lw=1, label=r"collision locus $\mathrm{Mix}_1^{\mathrm{T}}$")
    ax.scatter([], [], s=26, color=S1, marker="s", label=r"$b_0=2$, $b_1=0$ (two labelled branches)")
    ax.scatter([], [], s=26, color=S3, marker="s", label=r"$b_0=1$, $b_1=0$ (branches joined)")
    ax.scatter([], [], s=18, facecolor="none", edgecolor=INK3, label="empty fibre (outside the tropical secant)")
    ax.set_xlabel(r"$v_1-v_0$"); ax.set_ylabel(r"$v_2-v_0$"); ax.grid(color=GRID, lw=0.5); ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(os.path.join(OUT, "fig_betti.pdf")); plt.close(fig)
    from collections import Counter
    return Counter(tuple(b) for _, _, b, _ in data)

if __name__ == "__main__":
    print("algebra:", fig_algebra())
    rho, m1, E = fig_tower(); print("tower energies:", [round(e, 4) for e in E])
    print("secant fibre points:", fig_secant())
    print("tropical max deviation from tropical limit:", fig_tropical())
    fig_walls(); print("walls done")
    cases = [("generic", [F(1, 2), F(1), F(0)]), ("collision", [F(0), F(1), F(2)])]
    print("fibres:", fig_fibres(cases))
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "betti_scan.json")
    if os.path.exists(p): print("betti map:", fig_betti(p))
