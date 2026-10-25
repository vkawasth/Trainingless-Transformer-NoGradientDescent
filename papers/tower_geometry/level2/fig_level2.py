"""fig_level2.py -- figures for the note on partial gluing and level two (data from engine/check_level2.py -> level2_results.json).
(a) graded explanations for the drug example with conflicting trials: the prediction band for P(any harm | A+B+C) as the saved fraction
alpha grows, the most probable (max-entropy) explanation, and the two populations' truths; (b) level one against two latent types;
(c) Puiseux separation of the types across the independence locus; (d) the second-order obstruction around the tangent cone;
(e) the smallest singular value of dF along a role-exchanging half loop (3 vs 4 variables); (f) the 4-variable half loop: the types'
parameters trade places (Z/2 monodromy) while the observed law closes up."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "level2_results.json")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, GREY = "#2a78d6", "#eb6834", "#2e9e6b", "#c8423b", "#8a8a8a"
fig, ax = plt.subplots(2, 3, figsize=(7.6, 5.0)); (a, b, c), (d, e, f) = ax

G = R["graded"]; rows = np.array(G["rows"])
a.fill_between(rows[:, 0], rows[:, 1], rows[:, 2], color=BLUE, alpha=0.18, lw=0, label="prediction band over $E_\\alpha$")
a.plot(rows[:, 0], rows[:, 1], color=BLUE, lw=1); a.plot(rows[:, 0], rows[:, 2], color=BLUE, lw=1)
a.plot(rows[:, 0], rows[:, 3], color="black", lw=1.6, label="most probable explanation")
for t, lab, col in zip(G["truths"], ["population 1", "population 2"], [GREEN, ORANGE]): a.axhline(t, color=col, lw=1, ls="--", label=lab)
a.axvline(G["lam"], color=RED, lw=1, ls=":"); a.text(G["lam"] - 0.01, 0.05, "$\\lambda^*$", color=RED, ha="right")
a.set_xlim(0, 0.9); a.set_ylim(0, 1.02); a.set_xlabel("$\\alpha$  (structure saved)"); a.set_ylabel("P(any harm | A+B+C)")
a.legend(frameon=False, fontsize=6, loc="lower left"); a.set_title("(a) sitting at the gluing failure", loc="left", fontsize=8.5)

S = R["sharpen"]
for y, (lo, hi), lab, col in [(1, S["level1"], "pair tables only\n(level one)", BLUE), (0, S["level2"], "two latent types\n(level two)", ORANGE)]:
    b.plot([lo, hi], [y, y], color=col, lw=6, solid_capstyle="butt"); b.text(hi + 0.002, y, f"[{lo:.4f}, {hi:.4f}]", va="center", fontsize=6.5)
b.axvline(S["truth"], color="black", lw=1, ls="--"); b.text(S["truth"], 1.45, "truth", ha="center", fontsize=7)
b.set_yticks([0, 1]); b.set_yticklabels(["two latent types\n(level two)", "pair tables only\n(level one)"], fontsize=7)
b.set_ylim(-0.6, 1.7); b.set_xlim(0.65, 0.715); b.set_xlabel("P(any harm | A+B+C)")
b.set_title("(b) latent structure sharpens prediction", loc="left", fontsize=8.5)

P = R["puiseux"]; s = np.array(P["s"]); ss = np.linspace(0, 0.045, 200)
c.plot(ss, np.sqrt(ss), color=GREY, lw=1, label="$\\sqrt{s}$"); c.plot(s, P["gap"], "o", color=ORANGE, ms=4, label="gap between the types")
c.plot(P["neg_s"], P["neg_res"], "s", color=RED, ms=4, label="best two-type misfit")
c.axvline(0, color="black", lw=0.8); c.text(-0.044, 0.035, "no two-type\nexplanation", fontsize=6.5, color=RED)
c.text(0.017, 0.02, "two branches\n$c \\pm \\sqrt{s}\\, e$", fontsize=6.5, color=ORANGE)
c.set_xlabel("s  (path of observed laws)"); c.set_xlim(-0.046, 0.046); c.set_ylim(-0.01, 0.23)
c.legend(frameon=False, fontsize=6, loc="upper left"); c.set_title("(c) mass leaves through a singularity", loc="left", fontsize=8.5)

K = R["cone"]; phi = np.array(K["phi"])
d.plot(phi, K["kappa"], color=BLUE, lw=1.4)
for x, lab in [(0, "split $X_1$"), (np.pi / 2, "split $X_2$"), (np.pi, "split $X_1$")]: d.plot([x], [0], "o", color=GREEN, ms=5); d.text(x, max(K["kappa"]) * 0.08, lab, ha="center", fontsize=6.5, color=GREEN)
d.set_xticks([0, np.pi / 2, np.pi]); d.set_xticklabels(["0", "$\\pi/2$", "$\\pi$"])
d.set_xlabel("direction of the split in the $(X_1, X_2)$ plane"); d.set_ylabel("$|\\kappa(v)|$  (second-order obstruction)")
d.set_title("(d) tangent cone = union of branches", loc="left", fontsize=8.5)

M = R["monodromy"]; t3 = np.linspace(0, np.pi, len(M["m3"]["smin_path"])); t4 = np.linspace(0, np.pi, len(M["m4"]["smin_path"]))
e.semilogy(t3, M["m3"]["smin_path"], color=RED, lw=1.4, label="3 variables"); e.semilogy(t4, M["m4"]["smin_path"], color=BLUE, lw=1.4, label="4 variables")
e.set_xticks([0, np.pi / 2, np.pi]); e.set_xticklabels(["0", "$\\pi/2$", "$\\pi$"]); e.set_xlabel("$\\theta$ along the half loop")
e.set_ylabel("smallest singular value of dF"); e.legend(frameon=False, fontsize=6.5); e.set_title("(e) does the exchange pass a singularity?", loc="left", fontsize=8.5)

T = np.array(M["m4"]["trace"])                                            # steps x 2 types x 4 variables
f.plot(T[:, 0, 0], T[:, 0, 1], color=BLUE, lw=1.4, label="type 1 (tracked)"); f.plot(T[:, 1, 0], T[:, 1, 1], color=ORANGE, lw=1.4, label="type 2 (tracked)")
for h, col in [(0, BLUE), (1, ORANGE)]:
    f.plot(T[0, h, 0], T[0, h, 1], "o", color=col, ms=5); f.plot(T[-1, h, 0], T[-1, h, 1], "x", color=col, ms=6, mew=1.6)
f.set_xlabel("$a_{h1}$"); f.set_ylabel("$a_{h2}$"); f.legend(frameon=False, fontsize=6, loc="upper left")
f.set_title("(f) four variables: the roles trade places", loc="left", fontsize=8.5); f.set_aspect("equal", adjustable="datalim")
f.text(0.98, 0.03, "o start   x end of half loop", transform=f.transAxes, ha="right", fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_level2.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_level2.png"), dpi=130, bbox_inches="tight")
print("ok")
