"""Patch the author-supplied Gemini summary for sec:mcg: fix garbled formulas and redraw the fibre and thimble panels exactly."""
import io, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
SRC = "/mnt/user-data/attachments/6cdf1ee2-7dbd-5dd7-9fad-7a5f343db875/194b062b-image.jpg"
A = Image.open(SRC).convert("RGB"); d = ImageDraw.Draw(A)
NAVY, TEAL, VIOL, INK = "#23285e", "#2a8c8c", "#5b3fa0", "#1f2937"
def render(w, h, draw, scale=3):
    fig = plt.figure(figsize=(w * scale / 100, h * scale / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    draw(ax); buf = io.BytesIO(); fig.savefig(buf, dpi=100, facecolor="white"); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB").resize((w, h), Image.LANCZOS)
def text_patch(box, s, size, ha="left", va="center"):
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
    def dr(ax): ax.text(1 if ha == "left" else w / 2, h / 2, s, fontsize=size * 3 * 0.75, ha=ha, va=va, color=INK)
    A.paste(render(w, h, dr), (x0, y0))
# 1. rho_br line: remove the garbled second arrow
d.rectangle((158, 568, 300, 602), fill="white")
# 2. Picard-Lefschetz formula and intersection matrix
text_patch((598, 836, 745, 866), r"$T_V(x)=x\pm\langle x,V\rangle\,V$", 15)
text_patch((760, 826, 893, 892), r"$I_{ij}=\langle V_i,V_j\rangle$", 15)
# 3. comparison panel: B_6, H_1, 96x96
text_patch((702, 1020, 718, 1034), r"$B_6$", 10)
text_patch((606, 1067, 622, 1080), r"$H_1$", 10)
text_patch((768, 1130, 802, 1141), r"96$\times$96", 9)
# 4. fibre panel: exact schematic on a torus (square with opposite sides identified)
def fibre(ax):
    for k, (x0, title, cross) in enumerate(((20, "collision pair: disjoint,\ncommuting twists", False), (200, "caustic pair: meet once,\nbraid relation $T_aT_bT_a=T_bT_aT_b$", True))):
        s = 62; y0 = 6; ax.add_patch(plt.Rectangle((x0 + 55, y0), s, s, fill=False, ec=INK, lw=1.2))
        for (xa, ya, xb, yb) in ((x0 + 55, y0 + s, x0 + 55 + s, y0 + s), (x0 + 55, y0, x0 + 55 + s, y0)):
            ax.annotate("", (xa + s * 0.58, ya), (xa + s * 0.42, ya), arrowprops=dict(arrowstyle="->", color=INK, lw=1))
        for (xa, ya) in ((x0 + 55, y0), (x0 + 55 + s, y0)):
            ax.annotate("", (xa, ya + s * 0.42), (xa, ya + s * 0.58), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))
        if cross:
            ax.plot([x0 + 55, x0 + 55 + s], [y0 + s / 2, y0 + s / 2], color=NAVY, lw=2.2); ax.plot([x0 + 55 + s / 2] * 2, [y0, y0 + s], color=TEAL, lw=2.2)
            ax.plot(x0 + 55 + s / 2, y0 + s / 2, "o", color=VIOL, ms=5)
        else:
            ax.plot([x0 + 55, x0 + 55 + s], [y0 + s * 0.33] * 2, color=NAVY, lw=2.2); ax.plot([x0 + 55, x0 + 55 + s], [y0 + s * 0.67] * 2, color=TEAL, lw=2.2)
        ax.text(x0 + 55 + s / 2, y0 + s + 8, title, ha="center", va="top", fontsize=9.6 * 3 * 0.75, color=INK)
    ax.text(4, 40, "$F$", fontsize=13 * 3 * 0.75, color=INK, va="center")
A.paste(render(385, 108, fibre), (18, 818)); d.rectangle((403, 900, 407, 927), fill='white')
# 5. thimble sketch: straight distinguished paths fanning out from s0 to the critical values, thimbles above
def thimbles(ax):
    s0 = (18, 52); C = [(250, 12), (270, 38), (262, 66), (238, 88)]
    ax.plot(*s0, "o", color=INK, ms=4); ax.text(s0[0] - 2, s0[1] + 14, "$s_0$", fontsize=11 * 2.25, color=INK, ha="center")
    for k, (x, y) in enumerate(C):
        ax.plot([s0[0], x], [s0[1], y], color=[NAVY, TEAL, VIOL, NAVY][k], lw=1.8); ax.plot(x, y, "x", color="#C2185B", ms=7, mew=2)
        t = 0.55; mx, my = s0[0] + t * (x - s0[0]), s0[1] + t * (y - s0[1])
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse((x - 18, y - 2), 26, 12, fc="#c7d2fe", ec=NAVY, lw=0.8, alpha=0.9))
    ax.text(140, 10, r"$\gamma_1$", fontsize=10 * 2.25, color=NAVY); ax.text(130, 92, r"$\gamma_N$", fontsize=10 * 2.25, color=NAVY)
A.paste(render(318, 100, thimbles), (18, 656))
A.save("graphical_summary_mcg.png")
