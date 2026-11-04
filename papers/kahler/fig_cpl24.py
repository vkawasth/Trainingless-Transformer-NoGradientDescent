"""Figures for Part IV of the companion note: 2 x 4 couplings (module engine/pce/cpl24.py).
  fig_tf_polytopes.pdf  the three fibres A, B, C with the candidate sets of selection: sheet -> line -> point (B: sheet -> point)
  fig_tf_spectrum.pdf   the eigenvalue valuations of Q/W at the positive object against 1/L, converging to h_j - h_1 with multiplicity r_j
  fig_tf_objects.pdf    every critical point (object) with its fibre and leading holonomies, drawn as dials"""
import os, sys, itertools
import numpy as np
import mpmath as mp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import cpl24 as C
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
EX = {"A": C.EX_A, "B": C.EX_B, "C": C.EX_C}
TITLE = {"A": "A: $(\\mathbb{CP}^1)^3$, Fano, decoupled", "B": "B: $\\mathrm{Bl}_{pt}\\mathbb{CP}^3$, Fano, coupled", "C": "C: weak Fano, 7 facets, coupled"}


def candidate_vertices(f, g, j):
    """vertices of the candidate set after level j (j = 0: the fibre): fixed cells at their levels, free cells >= h_j"""
    P = C.polytope(f, g); V, lam, F = P["V"], P["lam"], P["facets"]; lx = C.leximin_exact(f, g); eq = {}; free = list(F)
    if j > 0:
        for i in range(j):
            for nm in lx["sets"][i]: k = C.NAMES.index(nm); eq[k] = lx["levels"][i]; free.remove(k)
    h = lx["levels"][j - 1] if j > 0 else Fr(0)
    rows = [(k, eq[k]) for k in eq] + [(k, h) for k in free]; pts = []
    for S in itertools.combinations(rows, 3):
        x, D = C._solve3([V[k] for k, _ in S], [lam[k] + t for k, t in S])
        if x is None: continue
        if all(C._dot(V[k], x) - lam[k] == t for k, t in eq.items()) and all(C._dot(V[k], x) - lam[k] >= h for k in free):
            if x not in pts: pts.append(x)
    return np.array([[float(c) for c in x] for x in pts])


# ---------------------------------------------------------------- polytopes
fig = plt.figure(figsize=(12.6, 5.6))
for n, (k, (f, g)) in enumerate(EX.items()):
    a = fig.add_subplot(1, 3, n + 1, projection="3d"); lx = C.leximin_exact(f, g)
    Pv = candidate_vertices(f, g, 0); hull = ConvexHull(Pv)
    faces = []
    for eqn in np.unique(np.round(hull.equations, 9), axis=0):
        on = Pv[np.abs(Pv @ eqn[:3] + eqn[3]) < 1e-9]
        if len(on) < 3: continue
        c0 = on.mean(0); nrm = eqn[:3]; e1 = on[0] - c0; e1 /= np.linalg.norm(e1); e2 = np.cross(nrm, e1)
        ang = np.arctan2((on - c0) @ e2, (on - c0) @ e1); faces.append(on[np.argsort(ang)])
    a.add_collection3d(Poly3DCollection(faces, facecolor="#dfe7f1", edgecolor=INK2, linewidths=0.6, alpha=0.18))
    cols = [ORANGE, ROSE, VIOLET]; labs = []
    for j in range(1, len(lx["levels"]) + 1):
        if sum(lx["drops"][:j]) == sum(lx["drops"][:j - 1]): continue
        Q = candidate_vertices(f, g, j); dim = 3 - sum(lx["drops"][:j]); col = cols[j - 1]
        lab = f"level {j}: $h_{j}={lx['levels'][j - 1]}$, " + {2: "a sheet", 1: "a line", 0: "a point"}[dim]
        if dim == 2:
            c0 = Q.mean(0); M = Q - c0; _, _, vt = np.linalg.svd(M); ang = np.arctan2(M @ vt[1], M @ vt[0]); Q = Q[np.argsort(ang)]
            a.add_collection3d(Poly3DCollection([Q], facecolor=col, edgecolor=col, alpha=0.45)); a.plot([], [], color=col, lw=6, alpha=0.5, label=lab)
        elif dim == 1:
            a.plot(Q[:, 0], Q[:, 1], Q[:, 2], color=col, lw=3.2, label=lab)
        else:
            a.scatter(*Q[0], color=col, s=55, depthshade=False, zorder=10, label=lab)
    u = [float(c) for c in lx["point"]]; a.scatter(*u, marker="*", color=INK, s=70, depthshade=False, zorder=11)
    a.set_xlabel("$a_{11}$", labelpad=-12); a.set_ylabel("$a_{12}$", labelpad=-12); a.set_zlabel("$a_{13}$", labelpad=-12); a.set_xticklabels([]); a.set_yticklabels([]); a.set_zticklabels([])
    a.view_init(elev=22, azim=-58); a.set_proj_type("ortho"); a.set_box_aspect(np.ptp(Pv, axis=0) + 1e-3, zoom=1.12)
    a.set_title(TITLE[k] + f"\n$f={tuple(str(x) for x in f)}$, $g={tuple(str(x) for x in g)}$".replace("'", ""), fontsize=8.6)
    a.legend(frameon=False, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.25, wspace=0.08)
fig.savefig(os.path.join(HERE, "fig_tf_polytopes.pdf")); fig.savefig(os.path.join(HERE, "fig_tf_polytopes.png"), dpi=150); plt.close(fig)

# ---------------------------------------------------------------- spectrum
Ls = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.9), sharey=False)
for a, (k, (f, g)) in zip(ax, EX.items()):
    lx = C.leximin_exact(f, g); data = []
    for L in Ls:
        y = C.positive_point(f, g, L); ev, W = C.clifford_spectrum(f, g, L, y); data.append(sorted(float(-mp.log(abs(e)) / L) for e in ev))
    data = np.array(data); x = 1 / np.array(Ls)
    pred = []; h1 = lx["levels"][0]
    for j, r in enumerate(lx["drops"]):
        if j > 0: pred += [lx["levels"][j] - h1] * r
    for p in sorted(set(pred)):
        m = pred.count(p); a.axhline(float(p), color=GREY, ls="--", lw=1)
        a.text(0.0205, float(p) - 0.0006, f"$h_j-h_1={p}$" + (f"  (twice: $r_j={m}$)" if m > 1 else ""), fontsize=7.6, va="top", ha="right", color=INK2)
    a.axhline(0, color=GREY, ls="--", lw=1)
    for i, col in enumerate([BLUE, ORANGE, ROSE]): a.plot(x, data[:, i], "o-", color=col, ms=4, lw=1.3)
    a.set_xlabel("$1/L$  ($T=e^{-L}$)"); a.set_ylabel("$-\\log|\\mu|/L$ for the eigenvalues $\\mu$ of $Q/W$"); a.set_xlim(-0.0008, 0.0215)
    a.set_title(TITLE[k] + "\nlevels " + ", ".join(str(h) for h in lx["levels"]) + f"; drops {tuple(lx['drops'])}", fontsize=8.8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_tf_spectrum.pdf")); fig.savefig(os.path.join(HERE, "fig_tf_spectrum.png"), dpi=150); plt.close(fig)

# ---------------------------------------------------------------- objects
def dial(a, x0, y0, ph, col):
    t = np.linspace(0, 2 * np.pi, 60); a.plot(x0 + 0.38 * np.cos(t), y0 + 0.38 * np.sin(t), color=GREY, lw=0.6)
    a.annotate("", (x0 + 0.34 * np.cos(ph), y0 + 0.34 * np.sin(ph)), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=7))
fig, ax = plt.subplots(1, 3, figsize=(12.0, 6.4), gridspec_kw={"width_ratios": [1, 1, 1]}); rows = []
for a, (k, (f, g)) in zip(ax, EX.items()):
    L = 400.0; pts, b = C.critical_points_exact(f, g, L); groups = {}
    for y in pts:
        u = tuple(Fr(round(float(-mp.log(abs(v)) / L) * 240), 240) for v in y); groups.setdefault(u, []).append([float(mp.arg(v)) for v in y])
    row = 0; a.set_xlim(-0.6, 4.2); a.axis("off")
    for u, objs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        a.text(-0.55, -row + 0.05, f"fibre $u=({', '.join(str(c) for c in u)})$: {len(objs)} object" + ("s" if len(objs) > 1 else ""), fontsize=8, color=INK, va="bottom")
        row += 0.75
        for ph in sorted(objs, key=lambda p: [round(q, 2) for q in p]):
            for i in range(3): dial(a, 1.0 * i + 0.4, -row, ph[i], [BLUE, ORANGE, ROSE][i])
            row += 1.0
        row += 0.2
    for i, nm in enumerate(["$y_1$", "$y_2$", "$y_3$"]): a.text(1.0 * i + 0.4, 0.75, nm, ha="center", fontsize=8.5, color=[BLUE, ORANGE, ROSE][i])
    rows.append(row); a.set_aspect("equal")
    a.set_title(TITLE[k] + f"\n{len(pts)} critical points = normalised volume {b} = $\\chi$", fontsize=8.8)
for a in ax: a.set_ylim(-max(rows) - 0.2, 1.3)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_tf_objects.pdf")); fig.savefig(os.path.join(HERE, "fig_tf_objects.png"), dpi=150); plt.close(fig); print("ok")
