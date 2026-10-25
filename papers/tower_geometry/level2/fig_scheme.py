"""fig_scheme.py -- scheme structure (note Section 7; data from engine/check_scheme.py -> scheme_results.json).
(a) a slice of the law space (m13 = m23 = k, mean fixed) coloured by what the two-type fibre's real points are: genuine mixtures
(Delta > 0), signed mixtures (Det > 0 > Delta), or a complex-conjugate pair (Det < 0); (b) the simplicial complex whose Stanley-Reisner
scheme is the fibre over a product law (m = 3); (c) along the path m12 = s on the slice: the weights of the two explanations, and the
largest |d| (which blows up as the path crosses m12 = 0, a point of the closure that is not in the image)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon, Patch
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "scheme_results.json")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
COL = ["#e8b4b0", "#f3d9a4", "#b9d6ee"]                                     # complex pair, signed mixture, genuine mixture
fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw={"width_ratios": [1.1, 0.9, 1.2]})

x = np.array(R["x"]) * 1e3; y = np.array(R["y"]) * 1e3
a.imshow(np.array(R["region"]), origin="lower", extent=[x[0], x[-1], y[0], y[-1]], aspect="auto", cmap=ListedColormap(COL), vmin=0, vmax=2)
xx = np.linspace(x[0], 0, 200); k = R["k"] * 1e3
a.plot(xx, np.sqrt(-4 * xx * k * k / 1e3), color="#555", lw=0.8); a.plot(xx, -np.sqrt(-4 * xx * k * k / 1e3), color="#555", lw=0.8)
a.axvline(0, color="black", lw=1); a.axhline(R["m123"] * 1e3, color="black", lw=0.8, ls="--")
a.set_xlabel("$m_{12}$ ($\\times10^{-3}$)"); a.set_ylabel("$m_{123}$ ($\\times10^{-3}$)")
a.legend(handles=[Patch(color=COL[2], label="mixture ($\\Delta>0$)"), Patch(color=COL[1], label="signed ($\\mathrm{Det}>0>\\Delta$)"),
                  Patch(color=COL[0], label="complex pair ($\\mathrm{Det}<0$)")], fontsize=5.8, frameon=True, loc="lower right")
a.set_title("(a) real points of the fibre", loc="left", fontsize=8.5)

V = {"w1": (0.5, 1.0), "w2": (0.5, -0.05), "d1": (0.05, 0.42), "d2": (0.95, 0.42), "d3": (0.62, 0.58)}
for top in ("w1", "w2"):
    b.add_patch(Polygon([V[top], V["d1"], V["d2"]], closed=True, color="#2a78d6" if top == "w1" else "#eb6834", alpha=0.18, lw=0))
for u, v in [("d1", "d2"), ("d2", "d3"), ("d1", "d3")] + [(t, dd) for t in ("w1", "w2") for dd in ("d1", "d2", "d3")]:
    b.plot(*zip(V[u], V[v]), color="#444", lw=0.9, ls="--" if "d3" in (u, v) and u != "d1" and v != "d1" and u != "d2" and v != "d2" else "-")
b.plot(*zip(V["w1"], V["w2"]), color="#2e9e6b", lw=1.8)
for n, (px, py) in V.items():
    b.plot(px, py, "o", color="black", ms=4)
    b.text(px + (0.05 if n != "d1" else -0.05), py + (0.04 if n == "w1" else -0.07 if n == "w2" else 0.03), f"${n[0]}_{n[1]}$",
           ha="left" if n != "d1" else "right", fontsize=8)
b.text(0.5, -0.25, "facets: $\\{w_1,d_1,d_2,d_3\\}$, $\\{w_2,d_1,d_2,d_3\\}$,\n$\\{w_1,w_2,d_i\\}$ (green axis + one $d_i$)", ha="center", fontsize=6.3)
b.set_xlim(-0.15, 1.15); b.set_ylim(-0.38, 1.12); b.axis("off"); b.set_title("(b) the complex of the fibre", loc="left", fontsize=8.5)

P = np.array([[r[0], np.nan if r[1] is None else r[1], np.nan if r[2] is None else r[2], np.nan if r[3] is None else r[3]] for r in R["path"]])
s = P[:, 0] * 1e3
for lo, hi, col in [(s.min(), 0, None), (0, s.max(), COL[2])]:
    pass
cx = s[np.isnan(P[:, 1])]
c.axvspan(s.min(), cx.max() if cx.size else s.min(), color=COL[0], alpha=0.8, lw=0)
c.axvspan(cx.max() if cx.size else s.min(), 0, color=COL[1], alpha=0.8, lw=0); c.axvspan(0, s.max(), color=COL[2], alpha=0.8, lw=0)
c.plot(s, P[:, 1], color="#2a78d6", lw=1.4, label="$w_1$, explanation 1"); c.plot(s, P[:, 2], color="#eb6834", lw=1.4, label="$w_1$, explanation 2")
c.axhline(0, color="#555", lw=0.6); c.axhline(1, color="#555", lw=0.6)
c.set_ylim(-0.6, 1.6); c.set_xlabel("$m_{12}=s$ ($\\times10^{-3}$) along the dashed line in (a)"); c.set_ylabel("weight $w_1$")
c2 = c.twinx(); c2.semilogy(s, P[:, 3], color="black", lw=1, ls=":", label="max $|d_i|$"); c2.set_ylabel("max $|d_i|$ (dotted)")
c2.spines["top"].set_visible(False); c.legend(frameon=False, fontsize=6, loc="upper left")
c.set_title("(c) crossing the walls", loc="left", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_scheme.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_scheme.png"), dpi=130, bbox_inches="tight")
print("ok")
