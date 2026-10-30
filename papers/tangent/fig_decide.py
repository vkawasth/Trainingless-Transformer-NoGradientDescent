"""Figure for the decision-boundary sections of "A tangent structure for injected data". (a) the lab's post-probe decision boundary in
the plane through a boundary point spanned by the normal and the most curved tangent direction, with a mixture flip; (b) the decision
gap along the two flip chords against the injected share; (c) attack budgets: first order, second order (the 2-jet) and exact."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(HERE, "..", "engine"); sys.path.insert(0, ENG)
import warnings; warnings.filterwarnings("ignore")
from pce import tangent as TG, tangent_decide as TD, lifting as LF, injection as IJ
from pce.tangent import HD
D = json.load(open(os.path.join(ENG, "..", "lab", "tangent_decide.json")))
BLUE, ORANGE, VIOLET, AQUA, GREY, CRIT = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8c8c8c", "#d03b3b"; INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
W = LF.we(); c = list(W["ps"][0] - np.r_[W["UP"], 0.0]); F = lambda X: TG.poly_hd(TG.rhat_hd(X), c); B = np.linalg.svd(np.ones((1, 8)))[2][1:].T
P0 = np.array(D["boundary"]["P0"]); ev, V, g0, H0 = TD.second_form(F, P0, B); nvec = B @ (g0 / np.linalg.norm(g0)); k = int(np.argmax(ev)); xi = B @ V[:, k]
val = lambda p: F(HD.var(p, [])).c[0]
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0))
a = ax[0]; S = np.linspace(-0.035, 0.035, 141); E = np.linspace(-0.0012, 0.0012, 121); Z = np.array([[val(P0 + s * xi + e * nvec) for s in S] for e in E])
cs = a.contourf(S, E * 1e3, Z * 1e4, levels=np.linspace(-6, 6, 25), cmap="RdBu", extend="both"); a.contour(S, E * 1e3, Z, levels=[0], colors=INK, linewidths=1.6)
fl = D["flips"]["T,T -> P"]; Pp, Pm = np.array(fl["P_plus"]), np.array(fl["P_minus"])
co = lambda p: ((p - P0) @ xi / (xi @ xi), (p - P0) @ nvec / (nvec @ nvec) * 1e3)
for p, lab in ((Pp, "$P_+$"), (Pm, "$P_-$"), (0.5 * (Pp + Pm), "mixture")):
    x, y = co(p); a.scatter([x], [y], color=INK, s=18, zorder=5); a.text(x + 0.002, y + 0.05, lab, fontsize=8)
x1, y1 = co(Pp); x2, y2 = co(Pm); a.plot([x1, x2], [y1, y2], color=INK, lw=1, ls="--")
a.set_xlabel("tangent direction $\\xi$ (largest curvature)"); a.set_ylabel("normal direction ($\\times10^{-3}$)")
a.text(-0.033, 1.0, "blue: act T ($F>0$)", fontsize=7.5, color=BLUE); a.text(0.012, -1.1, "red: act P ($F<0$)", fontsize=7.5, color=INK)
a.set_title("(a) the post-probe boundary curves: two laws\nchoosing T whose mixture chooses P", fontsize=9)
a = ax[1]; es = np.linspace(0, 1, 401)
for name, col in (("T,T -> P", BLUE), ("P,P -> T", ORANGE)):
    f = D["flips"][name]; Pp, Pm = np.array(f["P_plus"]), np.array(f["P_minus"]); a.plot(es, [val((1 - e) * Pp + e * Pm) * 1e4 for e in es], color=col, lw=2, label=f"{name}: flips for share in {f['flip_shares']}")
a.axhline(0, color=INK2, lw=0.8); a.set_xlabel("injected share $\\varepsilon$ of $P_-$ into $P_+$"); a.set_ylabel("decision gap $F$ ($\\times10^{-4}$)")
a.legend(frameon=False, fontsize=7.2, loc="lower center"); a.set_title("(b) pooling paradox: impossible for affine $F$,\nforced by curvature of either sign", fontsize=9)
a = ax[2]; bud = D["budgets"]; keys = [k for k in ("111", "001", "100") if bud[k]["exact"]]
x = np.arange(len(keys)); a.bar(x - 0.25, [bud[k]["eps1"] for k in keys], width=0.25, color=GREY, label="first order $\\varepsilon_1$")
a.bar(x, [bud[k]["eps2"] for k in keys], width=0.25, color=VIOLET, label="second order (2-jet) $\\varepsilon_2$"); a.bar(x + 0.25, [bud[k]["exact"] for k in keys], width=0.25, color=INK, label="exact")
a.set_xticks(x); a.set_xticklabels([f"record {k}" for k in keys]); a.set_ylabel("smallest injected share that flips the act")
a.set_ylim(0, 0.021); a.text(-0.4, 0.0205, "010: $\\varepsilon_1=0.43$, but no flip (curvature $>0$)\n101: no first-order flip, exact $0.46$ (curvature $<0$)\n$\\chi$: $F''=0$, first order exact", fontsize=7.3, color=INK2, va="top")
a.legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(0.0, 0.8)); a.set_title("(c) vertex attacks: the 2-jet corrects the budget\nand can reverse the verdict", fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(HERE, "fig_decide.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_decide.png"), dpi=150, bbox_inches="tight"); print("ok")
