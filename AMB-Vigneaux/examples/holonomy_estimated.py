"""Problem 26 — holonomy from ESTIMATED operations.  Writes examples/holonomy.png.

A  calibration of the Wald test under the flat null, vs wrapping noise σ:
   naive σ/√m standard error, delta-method circular standard error, bootstrap.
B  power vs planted holonomy h for a 3-edge and a 6-edge loop, empirical vs
   non-central χ² theory.
C  readings per edge needed for 80% power at h = 0.05: theory curve, with
   empirical checks; grows like |L|·exp(4π²σ²).
D  Z/5 symmetric channel: P(false non-zero class) vs readings per edge, and the
   Chernoff bound.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from amb_vigneaux.holonomy import (Graph, chernoff_class_error, chernoff_threshold, estimate_edges,
                                   holonomy_test, n_required, observe_edges, power_theory, reduce,
                                   identifiable_loops, edge_design, tree_section_design, route_design,
                                   field_fit_estimate)

BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
rng = np.random.default_rng(0)
ROOM = Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
G = Graph(8, [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)])
L = np.abs(G.C).sum(1)

def flat(Gr, group):
    c = rng.random(Gr.n_vertices) if group == "R/Z" else rng.integers(0, group, Gr.n_vertices)
    return reduce(Gr.coboundary(c), group)

def planted(li, h):
    g = flat(G, "R/Z").astype(float); k = G.cotree[li]; g[k] += h * G.C[li, k]; return g

# ------------------------------------------------ identifiability table
print("IDENTIFIABILITY of each basis loop (room graph)")
extra = ROOM.root_path(2) + ROOM.C[0]            # a second route to vertex 2, around loop 0
designs = {"edge-local": edge_design(ROOM), "tree section (T0)": tree_section_design(ROOM),
           "tree + 2nd route to v2": route_design(ROOM, [extra]), "each loop, twice": 2 * ROOM.C}
print(f"  {'design':<24}" + "".join(f"{str(g):>10}" for g in ("R", "R/Z", 3, 4)))
for name, D in designs.items():
    print(f"  {name:<24}" + "".join(f"{str(identifiable_loops(ROOM, D, g)):>10}"
                                    .replace("True", "Y").replace("False", "n").replace(", ", "")
                                    for g in ("R", "R/Z", 3, 4)))

# ------------------------------------------------ A calibration
sigmas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
cal = {"naive": [], "delta": [], "boot": []}
for s in sigmas:
    rn, rd, rb = [], [], []
    for r in range(500):
        obs = observe_edges(flat(G, "R/Z"), 30, "R/Z", s, rng)
        rn.append(holonomy_test(G, obs, "R/Z", se_kind="small").p_chi2 < 0.05)
        t = holonomy_test(G, obs, "R/Z", n_boot=(49 if r < 120 else 0), rng=rng)
        rd.append(t.p_chi2 < 0.05)
        if t.p_boot is not None: rb.append(t.p_boot < 0.05)
    cal["naive"].append(np.mean(rn)); cal["delta"].append(np.mean(rd)); cal["boot"].append(np.mean(rb))

# ------------------------------------------------ B power
hs = np.linspace(0, 0.08, 9); sigma, n = 0.05, 50
se = np.full(G.E, sigma / np.sqrt(n))
powB = {}
for li in range(2):
    emp = []
    for h in hs:
        emp.append(np.mean([holonomy_test(G, observe_edges(planted(li, h), n, "R/Z", sigma, rng), "R/Z").p_chi2 < 0.05
                            for _ in range(300)]))
    hv = np.linspace(0, 0.08, 81)
    th = [power_theory(G, np.eye(2)[li] * x, se) for x in hv]
    powB[li] = (emp, hv, th)

# ------------------------------------------------ C n required
sg = np.linspace(0.02, 0.36, 60)
nreq = {li: [n_required(G, np.eye(2)[li] * 0.05, s) for s in sg] for li in range(2)}
checks = []
for s in (0.1, 0.2, 0.3):
    for li in range(2):
        nn = n_required(G, np.eye(2)[li] * 0.05, s)
        pw = np.mean([holonomy_test(G, observe_edges(planted(li, 0.05), nn, "R/Z", s, rng), "R/Z").p_chi2 < 0.05
                      for _ in range(200)])
        checks.append((s, li, nn, pw))

# ------------------------------------------------ D discrete
ms = [5, 10, 20, 30, 40, 60, 80, 100]
errD, bnd = [], []
for m in ms:
    errD.append(np.mean([np.any(ROOM.holonomy(estimate_edges(observe_edges(flat(ROOM, 5), m, 5, 0.7, rng), 5).g, 5) != 0)
                         for _ in range(3000)]))
    bnd.append(chernoff_class_error(ROOM, 5, 0.7, m))

# ------------------------------------------------ figure
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axs = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
def grid(ax): ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

ax = axs[0, 0]; grid(ax)
ax.plot(sigmas, cal["naive"], "-o", color=ORANGE, lw=2, ms=5, label="Wald, naive SE σ/√m")
ax.plot(sigmas, cal["delta"], "-o", color=BLUE, lw=2, ms=5, label="Wald, circular delta-method SE")
ax.plot(sigmas, cal["boot"], "-o", color=AQUA, lw=2, ms=5, label="bootstrap from nearest flat system")
ax.axhline(0.05, color=INK2, ls="--", lw=1); ax.text(sigmas[0], 0.07, "nominal 5%", color=INK2, fontsize=8)
ax.set_xlabel("edge noise σ (turns, wrapped normal)"); ax.set_ylabel("false-positive rate, flat world")
ax.set_title("A · calibration under the flat null (m = 30 per edge)", loc="left"); ax.legend(frameon=False)

ax = axs[0, 1]; grid(ax)
for li, col in ((0, BLUE), (1, ORANGE)):
    emp, hv, th = powB[li]
    ax.plot(hv, th, color=col, lw=2, label=f"{L[li]}-edge loop, theory")
    ax.plot(hs, emp, "o", color=col, ms=6, mec="white", mew=1.5, label=f"{L[li]}-edge loop, measured")
ax.set_xlabel("planted holonomy h (turns)"); ax.set_ylabel("power at α = 0.05")
ax.set_title("B · power: non-central χ² vs measured (σ = 0.05, m = 50)", loc="left"); ax.legend(frameon=False, loc="lower right")

ax = axs[1, 0]; grid(ax)
for li, col in ((0, BLUE), (1, ORANGE)):
    ax.plot(sg, nreq[li], color=col, lw=2, label=f"{L[li]}-edge loop, theory")
    pts = [(s, nn) for s, l_, nn, pw in checks if l_ == li]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=col, ms=6, mec="white", mew=1.5)
for s, li, nn, pw in checks:
    ax.annotate(f"{pw:.2f}", (s, nn), textcoords="offset points", xytext=(6, -10 if li == 0 else 4), fontsize=7, color=INK2)
ax.set_yscale("log"); ax.set_xlabel("edge noise σ (turns)"); ax.set_ylabel("readings per edge for 80% power")
ax.set_title("C · sample cost to detect h = 0.05 (dots: measured power)", loc="left"); ax.legend(frameon=False, loc="upper left")

ax = axs[1, 1]; grid(ax)
ax.plot(ms, errD, "-o", color=BLUE, lw=2, ms=5, label="measured P(false non-zero class)")
ax.plot(ms, bnd, "-", color=ORANGE, lw=2, label="Chernoff bound")
thr = chernoff_threshold(ROOM, 5, 0.7, 0.01)
ax.axvline(thr, color=INK2, ls=":", lw=1); ax.text(thr, 0.3, f" m* = {thr} for δ = 0.01", color=INK2, fontsize=8)
ax.set_yscale("log"); ax.set_ylim(1e-4, 1.5); ax.set_xlabel("readings per edge m")
ax.set_ylabel("probability"); ax.legend(frameon=False, loc="lower left")
ax.set_title("D · Z/5, symmetric channel q = 0.7: exact class vs readings", loc="left")
out = os.path.join(os.path.dirname(__file__), "holonomy.png")
fig.savefig(out, dpi=160, facecolor="#fcfcfb")

print("\nA  false-positive rate at σ:", dict(zip(sigmas, [f"naive {a:.2f} / delta {b:.3f} / boot {c:.3f}"
                                                       for a, b, c in zip(cal["naive"], cal["delta"], cal["boot"])])))
print("C  checks (σ, loop, n_required, measured power):", [(s, int(L[l]), nn, round(pw, 3)) for s, l, nn, pw in checks])
print("D  measured:", dict(zip(ms, np.round(errD, 4))), "\n   bound:", dict(zip(ms, np.round(bnd, 4))), " m* =", thr)
print("wrote", out)
