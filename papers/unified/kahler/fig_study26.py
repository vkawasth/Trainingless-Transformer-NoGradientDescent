"""Study 26 figures: (1) the pipeline, theorem by theorem; (2) four results. Numbers read from engine/study26_e2e/results_study26.json."""
import json, os
from auxref import N   # theorem numbers from the .aux file
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "..", "engine", "study26_e2e", "results_study26.json")))
COL = ["#1f5f9e", "#c27c0e", "#8e3b8e", "#2e7d32"]; INK, MUTED, GRID = "#222222", "#666666", "#d9d9d9"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

# ---------------------------------------------------------------------------------------------------- figure 1: pipeline
fig, ax = plt.subplots(figsize=(10.5, 5.6)); ax.set_xlim(0, 10.5); ax.set_ylim(0, 5.6); ax.axis("off")
def box(x, y, w, h, text, fc, ec=MUTED, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=0.9))
box(0.05, 4.25, 1.9, 1.15, "300 development\ntables, 9 contexts\n(sources vary)", "#f3f3f3")
box(2.25, 4.25, 1.9, 1.15, "flag and pool\nleave-one-out test;\nelastic attribution", "#f3f3f3")
box(4.45, 4.25, 1.9, 1.15, f"fibre J\nThm {N("thm:projection")}: dim 9\nProp {N("prop:fisher")}: Fisher kernel", "#e6eef7")
box(6.65, 4.25, 1.75, 1.15, "hypothesis\n3 paths, union;\nprojection ladder", "#e6eef7")
box(8.65, 4.25, 1.8, 1.15, f"decision\nThm {N("thm:decision")}: open\nThm {N("thm:determination")}: VOI range", "#e6eef7")
for x in (1.95, 4.15, 6.35, 8.4): arrow(x, 4.82, x + 0.3, 4.82)
box(0.05, 2.45, 2.5, 1.35, f"selection (coupling)\nThms {N("thm:maxmincoupling")}, {N("thm:selection")}: leximin,\ntoric limit; walls", "#f1e6f1")
box(2.8, 2.45, 2.5, 1.35, f"composition\nThms {N("thm:k4")}, {N("thm:sections")}: maximum\nentropy commutes,\nleximin does not", "#f1e6f1")
box(5.55, 2.45, 2.35, 1.35, f"history\nThm {N("thm:history")}: trace in\nH_D + F_MRY;\nThm {N("thm:decision")}: reachable slice", "#f7ead6")
box(8.15, 2.45, 2.3, 1.35, f"protection\nThm {N("thm:protect")}: rho_q vs\n2 eta; coalition\nevades a small inspection", "#f7ead6")
box(0.05, 0.3, 3.3, 1.65, "100 holdout tables\n{M,C,R,Y}: dim 9 -> 6, VOI fixed\n{M,Y}: inspection\n{M,R,Y}, {M,H,Y}: calibration", "#eaf3ea")
box(3.6, 0.3, 3.4, 1.65, f"cycle of four instruments\nfibre: Thm {N("thm:ncycle")}, Prop {N("prop:twist")} (twist)\nProp {N("prop:imprecctx")}: imprecise, not contextual\nnoncontextual fraction", "#eaf3ea")
box(7.25, 0.3, 3.2, 1.65, f"three layers kept apart (Rem {N("rem:layers")})\nsheaf: supports, global sections\nfibre: global law or not\ncocycle: entropy, system independence", "#eaf3ea")
for x in (5.4, 7.5): arrow(x, 4.25, x - 0.0, 3.8)
arrow(3.35, 1.12, 3.6, 1.12); arrow(7.0, 1.12, 7.25, 1.12)
ax.text(0.05, 5.5, "development (300): what the data identify, what they leave open, and how decisions respond", fontsize=8.5, color=MUTED)
ax.text(0.05, 2.05, "holdout (100): value of information, calibration, contextuality", fontsize=8.5, color=MUTED)
fig.savefig(os.path.join(HERE, "fig_study26_pipeline.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_study26_pipeline.png"), dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------------------------------------------- figure 2: results
fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.0))
ax = axs[0, 0]; lad = R["ladder"]; names = ["none", "C = 0", "C = 0, R = 1", "C = 0, R = 1, H = 0"]; tau = 0.10
for i, r in enumerate(lad):
    y = len(lad) - 1 - i
    ax.plot(r["delta_robust"], [y - 0.13] * 2, color=COL[0], alpha=0.45, lw=5, solid_capstyle="butt")
    ax.plot(r["delta_oracle"], [y + 0.13] * 2, color=COL[0], lw=5, solid_capstyle="butt")
    if r["delta_oracle"][1] - r["delta_oracle"][0] < 1e-6: ax.plot(r["delta_oracle"][0], y + 0.13, "|", color=COL[0], ms=11, mew=2.5)
    ax.plot(r["delta_true"], y + 0.13, "o", color="white", mec=INK, ms=5, zorder=5)
ax.axvline(tau, color=COL[1], lw=1.2, ls="--"); ax.text(tau + 0.02, 3.55, "tau = 0.10", color=COL[1], fontsize=8)
ax.set_yticks(range(len(lad))); ax.set_yticklabels(names[::-1]); ax.set_xlabel("effect Delta(K) = P(Y=1 | M=1, K) - P(Y=1 | M=0, K)")
ax.set_title("projection ladder: exact data (dark), estimated (light), truth (circle)", fontsize=9, color=INK, loc="left")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_ylim(-0.6, 3.8)

ax = axs[0, 1]; H = R["history"]
ax.plot(H["fibre"], [2, 2], color=COL[0], lw=6, solid_capstyle="butt"); ax.text(H["fibre"][0], 2.3, "whole fibre (any edit)", fontsize=8, color=COL[0])
ax.plot(H["slice"], [1, 1], color=COL[2], lw=6, solid_capstyle="butt"); ax.text(H["slice"][0], 1.3, "edits on {M,R,Y} only (numerical)", fontsize=8, color=COL[2])
ax.plot(H["honest"], 0, "o", color=COL[3], ms=7); ax.text(H["honest"], -0.38, f"honest {H['honest']:+.3f}", ha="center", fontsize=8, color=COL[3])
ax.plot(H["histB"], 0, "s", color=COL[1], ms=7); ax.text(H["histB"], 0.25, f"history B {H['histB']:+.3f}", ha="center", fontsize=8, color=COL[1])
ax.axvline(tau, color=COL[1], lw=1.2, ls="--"); ax.set_yticks([]); ax.set_ylim(-0.8, 2.7); ax.set_xlabel("Delta in the target stratum (C = 0, R = 1, H = 0)")
ax.set_title("history: same final tables, different end laws", fontsize=9, color=INK, loc="left"); ax.grid(axis="x", color=GRID, lw=0.6)

ax = axs[1, 0]; cyc = R["cycle"]
vals = [cyc["consistent"]["best"][0], cyc["twisted"]["best"][0]]; ses = [cyc["consistent"]["best"][1], cyc["twisted"]["best"][1]]
ax.bar([0, 1], vals, yerr=[3 * s for s in ses], color=[COL[3], COL[2]], width=0.55, capsize=4, ecolor=INK)
ax.axhline(2, color=INK, lw=1); ax.text(1.32, 2.03, "noncontextual\nbound 2", fontsize=8, color=INK, va="bottom")
for i, c in enumerate(("consistent", "twisted")):
    ax.text(i, vals[i] + 3 * ses[i] + 0.06, f"global law: {'yes' if cyc[c]['feasible'] else 'no'}\nNCF {cyc[c]['ncf']:.3f}\nglobal sections {cyc[c]['global_sections']}",
            ha="center", fontsize=7.5, color=INK)
ax.set_xticks([0, 1]); ax.set_xticklabels(["consistent coding", "app coding changed\nin one pair of studies"]); ax.set_ylim(0, 3.2); ax.set_xlim(-0.6, 1.9)
ax.set_ylabel("largest twisted-pattern sum (3 SE bars)"); ax.set_title("cycle of four instruments: sheaf, fibre, inequality", fontsize=9, color=INK, loc="left")
ax.grid(axis="y", color=GRID, lw=0.6)

ax = axs[1, 1]; P_ = R["protection"]
labels = ["rho_q (margin)", "2 eta (twice the\ninspection error)", "Delta_q honest", "Delta_q coalition", "kappa (flag level)"]
vals = [P_["rho"], 2 * P_["eta"], P_["dq_honest"], P_["dq_coalition"], P_["eta"]]; cols = [COL[0], COL[1], COL[3], COL[2], MUTED]
ax.barh(range(5)[::-1], vals, color=cols, height=0.55)
for i, v in enumerate(vals): ax.text(v + 0.003, 4 - i, f"{v:.3f}", va="center", fontsize=8, color=INK)
ax.set_yticks(range(5)[::-1]); ax.set_yticklabels(labels); ax.set_xlim(0, max(vals) * 1.25); ax.set_xlabel("l1 distance on the {M,Y} table")
ax.set_title(f"protection: decision g >= {P_['tau2']:.3f}; pooled g = {P_['g_pooled']:.3f}, truth {P_['gstar']:.3f}", fontsize=9, color=INK, loc="left")
ax.grid(axis="x", color=GRID, lw=0.6)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_study26_results.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_study26_results.png"), dpi=150, bbox_inches="tight")
print("ok")
