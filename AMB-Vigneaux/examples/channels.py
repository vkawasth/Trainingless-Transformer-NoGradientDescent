"""Lossy operations on a cycle (L = 4, Z/5, holonomy h = 2 vs flat).  Writes examples/channels.png."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from amb_vigneaux.channels import (birkhoff_noise, cf_noisy_cycle, cf_of, channel_information,
                                   character_eigenvalue, cycle, holonomy_from_spectrum,
                                   loop_operator, loop_walk, noisy_shift)

BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
n, L, h = 5, 4, 2
G = cycle(L); walk = loop_walk(G, 0)
gb = np.zeros(L, int); gb[0] = h; gf = np.zeros(L, int)
eps = np.linspace(0, 0.95, 39)
rows = []
for e in eps:
    Kb = [noisy_shift(n, x, e) for x in gb]; Kf = [noisy_shift(n, x, e) for x in gf]
    Mb, Mf = loop_operator(G, Kb, walk), loop_operator(G, Kf, walk)
    rows.append(dict(Ie=channel_information(Kb[0]), Ief=channel_information(Kf[0]),
                     Il=channel_information(Mb), Ilf=channel_information(Mf),
                     mod=abs(character_eigenvalue(Mb, 1)), hb=holonomy_from_spectrum(Mb),
                     hf=holonomy_from_spectrum(Mf) if abs(character_eigenvalue(Mf,1))>0 else 0,
                     cfb=cf_of(G, Kb), cff=cf_of(G, Kf)))
R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
R["hf"] = np.where(R["hf"] > n / 2, R["hf"] - n, R["hf"])

rng = np.random.default_rng(1); nc = []
for t in range(40):
    Ns = [birkhoff_noise(n, 4, rng) for _ in range(L)]
    for e in (0.2, 0.5):
        Kf = [((1 - e) * np.eye(n) + e * N) for N in Ns]
        Kb = [K @ np.roll(np.eye(n), x, axis=1) for K, x in zip(Kf, gb)]
        Mb = loop_operator(G, Kb, walk)
        nc.append((e, cf_of(G, Kf), cf_of(G, Kb), holonomy_from_spectrum(Mb)))
nc = np.array(nc)

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
def grid(ax): ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

ax = axs[0, 0]; grid(ax)
ax.plot(eps, R["Ief"], color=BLUE, lw=4, alpha=0.35, label="per-edge I, flat")
ax.plot(eps, R["Ie"], color=BLUE, lw=1.5, ls="--", label="per-edge I, holonomy 2")
ax.plot(eps, R["Ilf"], color=ORANGE, lw=4, alpha=0.35, label="loop-composite I, flat")
ax.plot(eps, R["Il"], color=ORANGE, lw=1.5, ls="--", label="loop-composite I, holonomy 2")
ax.set_xlabel("noise ε"); ax.set_ylabel("bits"); ax.legend(frameon=False)
ax.set_title("A · Layer 2: flat and holonomy-2 curves coincide", loc="left")

ax = axs[0, 1]; grid(ax)
ax.plot(eps, R["hb"], color=BLUE, lw=2, label="phase readout, holonomy 2")
ax.plot(eps, R["hf"], color=ORANGE, lw=2, label="phase readout, flat")
ax.set_ylim(-0.5, 3); ax.set_xlabel("noise ε"); ax.set_ylabel("h read from arg λ₁ (units of Z/5)")
ax2 = ax.twinx(); ax2.plot(eps, R["mod"], color=INK2, lw=1, ls=":"); ax2.set_ylabel("|λ₁| = (1−ε)^L  (dotted)", color=INK2)
ax2.set_yscale("log"); ax2.spines["top"].set_visible(False)
ax.legend(frameon=False, loc="center left")
ax.set_title("B · character-labelled eigenvalue: phase = h, modulus = loss", loc="left")

ax = axs[1, 0]; grid(ax)
ax.plot(eps, [cf_noisy_cycle(L, e) for e in eps], color=INK2, lw=1, ls="--", label="max(0, 1 − Lε/2)")
ax.plot(eps, R["cfb"], "o", color=BLUE, ms=4, label="CF, holonomy 2 (LP)")
ax.plot(eps, R["cff"], "o", color=ORANGE, ms=4, label="CF, flat (LP)")
ax.axvline(2 / L, color=INK2, lw=0.8, ls=":"); ax.text(2 / L, 0.85, "  ε = 2/L", color=INK2, fontsize=8)
ax.annotate("γ ≠ 0 only at ε = 0", (0, 1), xytext=(0.12, 1.02), fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_xlabel("noise ε"); ax.set_ylabel("contextual fraction"); ax.legend(frameon=False, loc="upper right")
ax.set_title("C · Layer 3: CF closed form; γ dies at once", loc="left")

ax = axs[1, 1]; grid(ax)
for e, col in ((0.2, BLUE), (0.5, ORANGE)):
    m = nc[:, 0] == e
    ax.plot(nc[m, 1], nc[m, 2], "o", color=col, ms=5, mec="white", mew=1, label=f"ε = {e}")
ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
ax.set_xlim(-0.02, 0.6); ax.set_ylim(0, 1)
ax.set_xlabel("CF of the FLAT system (noise alone)"); ax.set_ylabel("CF with holonomy 2")
ax.legend(frameon=False, loc="lower right")
ax.set_title("D · non-covariant noise: flat systems are already contextual", loc="left")
out = os.path.join(os.path.dirname(__file__), "channels.png")
fig.savefig(out, dpi=160, facecolor="#fcfcfb")
print("non-covariant: flat CF range", nc[:,1].min().round(3), nc[:,1].max().round(3),
      "| phase readout range (true 2):", nc[:,3].min().round(2), nc[:,3].max().round(2))
print("wrote", out)
