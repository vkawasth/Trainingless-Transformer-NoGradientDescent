"""fig_ladder.py -- the ladder of base semirings (note Section 8; data from engine/check_ladder.py -> ladder_results.json).
(a) level one: negativity of the best signed global section against 1 - lam*, for no-signalling families in five scenarios; each scenario
lies on or between lines of slope nu (its own contextual part's negativity). (b) level two, along a path of laws: which rung the two-type
explanation lives on (complex pair, signed, positive weights off the box, inside the box), with the size of the failure on each rung:
imaginary part (C), negative weight (R), and the largest |d| (dotted), which diverges at the walls between rungs."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "ladder_results.json")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1, 1.45]})
COLS = {"3-cycle": "#2a78d6", "CHSH": "#2e9e6b", "5-cycle": "#eb6834", "4-cycle": "#9b59b6", "3-outcome": "#c8423b"}
S = R["level1"]["scatter"]; xs = np.linspace(0, 1, 50)
for slope in (0.25, 0.5, 2 / 3): a.plot(xs, slope * xs, color="#bbb", lw=0.8, ls="--"); a.text(1.0, slope * 1.0, f" {slope:.2g}" if slope != 2 / 3 else " 2/3", fontsize=6.5, va="center", color="#777")
for name, col in COLS.items():
    pts = np.array([[x, y] for n, x, y in S if n == name])
    if len(pts): a.plot(pts[:, 0], pts[:, 1], "o", ms=3, color=col, label=name, alpha=0.85)
a.set_xlabel("$1-\\lambda^*$  (structure that cannot be saved)"); a.set_ylabel("negativity of the best signed section")
a.set_xlim(0, 1.08); a.set_ylim(0, 0.7); a.legend(frameon=False, fontsize=6.5, loc="upper left")
a.set_title("(a) level one: $ℝ$ but not $ℝ≥0$", loc="left", fontsize=8.5)

P = R["path"]; s = np.array([p["s"] for p in P]) * 1e3; rung = [p["rung"] for p in P]
BAND = {"C": "#e8b4b0", "R": "#f3d9a4", "R>=0 off box": "#d6e6f3", "R>=0": "#9cc3e8"}
i0 = 0
for i in range(1, len(P) + 1):
    if i == len(P) or rung[i] != rung[i0]:
        b.axvspan(s[i0], s[i - 1], color=BAND[rung[i0]], lw=0); i0 = i
imag = np.array([p["imag"] if p["rung"] == "C" else np.nan for p in P]); neg = np.array([p["neg"] if p["rung"] == "R" else np.nan for p in P])
b.plot(s, imag, color="#c8423b", lw=1.5, label="imaginary part ($ℂ$ rung)"); b.plot(s, neg, color="#b07d10", lw=1.5, label="negative weight ($ℝ$ rung)")
b.set_ylim(0, 1.2); b.set_xlabel("$m_{12}=s$ ($\\times10^{-3}$)"); b.set_ylabel("size of the failure")
b2 = b.twinx(); dm = np.array([np.nan if p["dmax"] is None else p["dmax"] for p in P]); b2.semilogy(s, dm, color="black", lw=0.9, ls=":")
b2.set_ylabel("max $|d_i|$ (dotted)"); b2.spines["top"].set_visible(False)
for x, lab, ha in [(R["path_params"]["s_det"] * 1e3, "Det = 0 ", "right"), (0, " $m_{12}=0$", "left"), (R["path_params"]["s_box"] * 1e3, "box wall", "center")]:
    b.axvline(x, color="#555", lw=0.6); b.text(x, 1.22, lab, ha=ha, fontsize=6.3, clip_on=False)
for x, lab in [(-2.6, "$ℂ$"), (-0.55, "$ℝ$"), (3.8, "$ℝ≥0$\noff box"), (9.8, "$ℝ≥0$\nin box")]:
    b.text(x, 1.02, lab, ha="center", fontsize=7.5, va="top")
b.legend(frameon=False, fontsize=6.3, loc="center right", bbox_to_anchor=(1.0, 0.55))
b.set_title("(b) level two: the rungs along a path", loc="left", fontsize=8.5, pad=12)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ladder.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_ladder.png"), dpi=130, bbox_inches="tight")
print("ok")
