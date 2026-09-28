"""Three closed-loop experiments for the AMB–Vigneaux engine.

A. Learning the PR box: the latent obstruction γ ∈ Ȟ¹ switches on as the
   learned law's small entries cross the support threshold.
B. Driven quantum world: visibility v(t) of the Tsirelson box ramps 0.5 → 1;
   the contextual learner's CF crosses zero at v = 1/√2, while the
   hidden-variable learner hits a KL wall there.
C. Support-threshold crossing: noisy PR box v(t) ramps 0.97 → 1; the realised
   obstruction appears once N·(1−v)/4 ≪ 1 and forbidden sections stop showing up.

Run:  python examples/demo.py   (writes examples/demo.png and prints a summary)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from amb_vigneaux import ClosedLoopEngine, ContextualFamily, HiddenVariableFamily
from amb_vigneaux.models import chsh_scenario, noisy, pr_box, tsirelson_box

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
sc = chsh_scenario()

# ---------------------------------------------------------------- A
famA = ContextualFamily(sc)
engA = ClosedLoopEngine(famA, famA.init_theta(), eta=0.5, n_samples=400, world=pr_box(), seed=1)
engA.run(40)

# ---------------------------------------------------------------- B
T_B = 120
vB = np.linspace(0.5, 1.0, T_B)
worldB = lambda t, P: noisy(tsirelson_box(), vB[min(t, T_B - 1)])
engB = {}
for name, Fam in [("contextual", ContextualFamily), ("hidden-variable", HiddenVariableFamily)]:
    f = Fam(sc)
    e = ClosedLoopEngine(f, f.init_theta(), eta=0.5, n_samples=5000, world=worldB, seed=2)
    e.run(T_B)
    engB[name] = e

# ---------------------------------------------------------------- C
T_C = 80
vC = np.linspace(0.97, 1.0, T_C)
worldC = lambda t, P: noisy(pr_box(), vC[min(t, T_C - 1)])
famC = ContextualFamily(sc)
engC = ClosedLoopEngine(famC, famC.init_theta(), eta=0.5, n_samples=300, world=worldC, seed=3)
engC.run(T_C)

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)

def grid(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

def flag_band(ax, t, flags, color, label, y0, y1):
    for i, f in zip(t, flags):
        if f:
            ax.axvspan(i - 0.5, i + 0.5, ymin=y0, ymax=y1, color=color, alpha=0.9, lw=0)
    ax.text(t[0], (y0 + y1) / 2, " " + label, transform=ax.get_xaxis_transform(),
            va="center", fontsize=8, color=INK2)

# A
ax = axs[0, 0]; grid(ax)
t = engA.series("t")
ax.plot(t, engA.series("cf_model"), color=BLUE, lw=2, label="CF(P_θ)")
ax.plot(t, engA.series("kl_world"), color=ORANGE, lw=2, label="KL(world ‖ P_θ)")
ax.set_ylim(-0.25, 1.05)
flag_band(ax, t, engA.series("gamma_latent"), AQUA, "γ≠0 on supp_ε(P_θ)", 0.0, 0.08)
ax.set_title("A · learning the PR box: latent obstruction switches on", loc="left")
ax.set_xlabel("loop iteration"); ax.legend(frameon=False, loc="center right")

# B
ax = axs[0, 1]; grid(ax)
for name, col in [("contextual", BLUE), ("hidden-variable", ORANGE)]:
    ax.plot(vB, engB[name].series("cf_model"), color=col, lw=2, label=f"CF, {name} learner")
cf_true = np.clip(np.sqrt(2) * vB - 1, 0, None)          # exact: CF = max(0, √2·v − 1)
ax.plot(vB, cf_true, color=INK2, lw=1, ls="--", label="CF of the world (exact)")
ax.axvline(1 / np.sqrt(2), color=INK2, lw=0.8, ls=":")
ax.text(1 / np.sqrt(2), 0.2, " v = 1/√2", color=INK2, fontsize=8)
ax.set_title("B · driven quantum world: contextuality threshold", loc="left")
ax.set_xlabel("world visibility v(t)"); ax.set_ylabel("contextual fraction")
ax.legend(frameon=False, loc="upper left")

# B'
ax = axs[1, 0]; grid(ax)
for name, col in [("contextual", BLUE), ("hidden-variable", ORANGE)]:
    ax.plot(vB, engB[name].series("kl_world"), color=col, lw=2, label=f"{name} learner")
ax.axvline(1 / np.sqrt(2), color=INK2, lw=0.8, ls=":")
ax.set_title("B · residual KL: the hidden-variable manifold cannot follow", loc="left")
ax.set_xlabel("world visibility v(t)"); ax.set_ylabel("KL(world ‖ P_θ)  [nats]")
ax.legend(frameon=False, loc="upper left")

# C
ax = axs[1, 1]; grid(ax)
t = engC.series("t")
ax.plot(t, engC.series("cf_empirical"), color=BLUE, lw=2, label="CF(P̂_N)")
ax.plot(t, engC.series("mean_mutual_info"), color=ORANGE, lw=2, label="mean I(a;b) of P_θ [bits]")
ax.set_ylim(-0.3, 1.05)
flag_band(ax, t, engC.series("gamma_realised"), AQUA, "γ≠0 on realised support", 0.0, 0.07)
flag_band(ax, t, engC.series("gamma_latent"), INK2, "γ≠0 on supp_ε(P_θ)", 0.09, 0.16)
ax2 = ax.secondary_xaxis("top", functions=(lambda x: np.interp(x, t, vC[:len(t)]),
                                             lambda v: np.interp(v, vC[:len(t)], t)))
ax2.set_xlabel("noise-free fraction v(t)", color=INK2)
ax.set_title("C · noisy PR box: support threshold crossing", loc="left", pad=28)
ax.set_xlabel("loop iteration"); ax.legend(frameon=False, loc="center left")

out = os.path.join(os.path.dirname(__file__), "demo.png")
fig.savefig(out, dpi=160, facecolor="#fcfcfb")

# ---------------------------------------------------------------- summary
def first(flags):
    idx = np.flatnonzero(flags)
    return int(idx[0]) if idx.size else None

print("A  PR box: γ latent first at t =", first(engA.series("gamma_latent")),
      "| final CF(P_θ) = %.3f, KL = %.4f" % (engA.history[-1].cf_model, engA.history[-1].kl_world),
      "| max cocycle defect %.1e" % engA.series("max_cocycle_defect").max())
for name, e in engB.items():
    cf = e.series("cf_model")
    on = first(cf > 0.03)   # above the finite-N signalling noise floor (~0.01)
    print(f"B  {name:16s}: CF>0.03 first at v = {vB[on]:.3f}" if on is not None else f"B  {name:16s}: CF stays 0",
          "| final KL = %.4f, final CF = %.3f" % (e.history[-1].kl_world, cf[-1]))
g = engC.series("gamma_realised")
print("C  noisy PR: realised γ≠0 in %d/%d steps; first at v = %s; latent γ first at t = %s"
      % (g.sum(), len(g), None if first(g) is None else f"{vC[first(g)]:.4f}", first(engC.series("gamma_latent"))))
print("wrote", out)
