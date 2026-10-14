"""fig_places.py -- Figure for Section 17: the three-token context presheaf read at the places of Q, drawn on the tree M(Z).
Example eps = 2/5, s = 0 (all numbers recomputed here from the exact formulas; see check_places.py M1-M7).
Output: figs/fig_places.pdf (and .png)
"""
from fractions import Fraction as F
import itertools, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

eps = F(2, 5); a, b = (3 * eps - 1) / 4, (1 - eps) / 4
ATOMS = list(itertools.product((0, 1), repeat=3))
J = {x: {0: a, 3: a, 1: b, 2: b}[sum(x)] for x in ATOMS}
D0 = J[(0, 0, 0)] * J[(1, 0, 1)] - J[(0, 0, 1)] * J[(1, 0, 0)]
def vp(x, p):
    n, d, v = x.numerator, x.denominator, 0
    while n % p == 0: n //= p; v += 1
    while d % p == 0: d //= p; v -= 1
    return v
def mod_support(p):
    if any(vp(v, p) < 0 for v in J.values()): return None
    return ["".join(map(str, x)) for x, v in J.items() if v.numerator * pow(v.denominator, -1, p) % p]
c = abs(2 * eps - 1) / (2 * eps)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.72, 1.55); ax.set_aspect("equal"); ax.axis("off")
INK, ACC, BAD, MUT = "#18222d", "#2b5fa8", "#c2410c", "#5b6876"

branches = [("inf", 90), (2, 162), (3, 222), (5, 318), (7, 18)]
L = 1.0
for lab, ang in branches:
    t = math.radians(ang); x, y = L * math.cos(t), L * math.sin(t)
    col = BAD if lab == "inf" else ACC
    ax.plot([0, x], [0, y], color=col, lw=2.2, solid_capstyle="round", zorder=1)
    for k in (0.33, 0.66):                                    # alpha ticks along the branch
        ax.plot([k * x], [k * y], marker="o", ms=2.5, color=col, zorder=2)
    ax.plot([x], [y], marker="o", ms=6.5, color=col, mec="white", zorder=3)

ax.plot([0], [0], marker="o", ms=11, color=INK, mec="white", zorder=4)
ax.text(0.10, 0.06, "$a_0$", fontsize=10, color=INK, ha="left", va="bottom")
ax.annotate("", xy=(0, -0.06), xytext=(0, -1.12), arrowprops=dict(arrowstyle="-", color="#9aa6b2", lw=0.8, ls=":"))
ax.text(0, -1.14, "centre: trivial seminorm, germs $=\\mathbb{Q}$\n"
        "signed sections $J+s\\chi$ exist (here $a=1/20$)\n"
        f"CI determinant $\\Delta_0={D0.numerator}/{D0.denominator}\\neq0$: Beck–Chevalley fails",
        ha="center", va="top", fontsize=8, color=INK,
        bbox=dict(boxstyle="round,pad=0.35", fc="#eef2f6", ec="#c7d0da", lw=0.8))

def box(x, y, text, col, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=7.8, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=0.9))

box(0, 1.27, "archimedean branch $|\\cdot|_\\infty^{\\varepsilon}$: order lives here\n"
             f"positive fibre non-empty ($\\varepsilon\\geq1/3$), $\\Phi={float(3*(1-eps)):.1f}\\leq2$\n"
             f"defect $c={c.numerator}/{c.denominator}$, the norm $\\|E_AE_B-E_{{A\\wedge B}}\\|$", BAD)
for lab, ang in branches[1:]:
    t = math.radians(ang); x, y = 1.0 * math.cos(t), 1.0 * math.sin(t)
    sup = mod_support(lab)
    end = "end $\\mathbb{F}_%d$: J not %d-integral" % (lab, lab) if sup is None else \
          ("end $\\mathbb{F}_%d$: support {%s}" % (lab, ", ".join(sup)) if len(sup) < 8 else "end $\\mathbb{F}_%d$: all 8 atoms" % lab)
    txt = f"$p={lab}$:  $v_{lab}(\\Delta_0)={vp(D0, lab)}$\n{end}"
    ox, oy = {2: (-0.32, 0.18), 3: (-0.42, -0.24), 5: (0.42, -0.24), 7: (0.30, 0.20)}[lab]
    box(x + ox, y + oy, txt, ACC)
ax.text(-1.72, 1.52, "$\\varepsilon=2/5$, $s=0$", ha="left", va="top", fontsize=9, color=INK)

os.makedirs("figs", exist_ok=True)
fig.savefig("figs/fig_places.pdf", bbox_inches="tight"); fig.savefig("figs/fig_places.png", dpi=150, bbox_inches="tight")
print("wrote figs/fig_places.pdf; Delta0 =", D0, "c =", c, {p: vp(D0, p) for p in (2, 3, 5, 7)})
