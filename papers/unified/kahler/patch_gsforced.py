"""Patch the author-supplied Gemini summary for sec:mcg part 4 (forced structure) -> graphical_summary_forced.png.
Fixes: (1) table embedding: old 2x3 = rows (AgI, unseeded) x (below, normal, above); new = salt row + 'none' column (a face);
(2) Lefschetz sketch redrawn: singular fibre with two nodes p, q over a point c of the base; (3A) collision drawn exactly: crossing
strands (half-twist) and a loop that encircles the collision point; (3B) end labels 'state 1', 'state 2'; (3C) Stokes event redrawn:
value crosses a vanishing path, before / after related by a Hurwitz move."""
import io, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
SRC = "/mnt/user-data/attachments/6cdf1ee2-7dbd-5dd7-9fad-7a5f343db875/3e97c823-image.jpg"
A = Image.open(SRC).convert("RGB")
INK = "#1f2937"; BLUE = (142 / 255, 176 / 255, 214 / 255); GREEN = (168 / 255, 205 / 255, 172 / 255)
ABG = (214 / 255, 227 / 255, 236 / 255); BBG = (251 / 255, 233 / 255, 197 / 255); CBG = (243 / 255, 211 / 255, 208 / 255)
def render(w, h, draw, bg="white", scale=3):
    fig = plt.figure(figsize=(w * scale / 100, h * scale / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    draw(ax); buf = io.BytesIO(); fig.savefig(buf, dpi=100, facecolor=bg); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB").resize((w, h), Image.LANCZOS)
def patch(box, draw, bg="white"):
    x0, y0, x1, y1 = box; A.paste(render(x1 - x0, y1 - y0, draw, bg), (x0, y0))
def label(box, s, size, bg="white", ha="center"):
    def dr(ax):
        w, h = box[2] - box[0], box[3] - box[1]; ax.text(w / 2 if ha == "center" else (w - 2 if ha == "right" else 2), h / 2, s, fontsize=size * 3.3, ha=ha, va="center", color=INK)
    patch(box, dr, bg)
FS = 13
# (1) table labels: left 'seeded' -> 'AgI'; right rows: AgI (blue), unseeded (blue), salt (green, new)
label((18, 160, 126, 199), "AgI", FS, ha="right")
label((487, 203, 584, 227), "unseeded", FS, bg=BLUE, ha="right")
label((487, 231, 584, 254), "salt", FS, bg=GREEN, ha="right")
# (2) Lefschetz sketch: base plane with c, singular fibre above with two nodes p, q
def lef(ax):
    ax.add_patch(plt.Polygon([(40, 72), (250, 72), (290, 52), (80, 52)], closed=True, fill=False, ec=INK, lw=3))
    ax.plot([165], [62], "o", color=INK, ms=10); ax.text(172, 64, "$c$", fontsize=FS * 3.3, va="center", color=INK)
    x = np.linspace(40, 290, 400); y = 22 + 9 * np.sin((x - 40) / 250 * 4 * np.pi)
    ax.plot(x, y, color=INK, lw=3.5); ax.plot(x, 44 - y, color=INK, lw=3.5)
    for xn, nm in ((102.5, "$p$"), (227.5, "$q$")):
        ax.plot([xn], [22], "o", color=INK, ms=11); ax.text(xn + 6, 34, nm, fontsize=FS * 3.3, ha="left", va="top", color=INK)
    ax.annotate("", (165, 58), (165, 36), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5, ls="--"))
    ax.text(170, 44, "$W$", fontsize=FS * 3.0, va="center", color=INK)
    ax.text(292, 22, "fibre over $c$", fontsize=FS * 2.6, ha="right", va="center", color="#555555") if False else None
patch((50, 360, 360, 448), lef)
# (3A) value collision: crossing strands (half-twist) and a loop around the collision point
def coll(ax):
    ax.plot([20, 120], [12, 52], color=INK, lw=3.5); ax.plot([20, 120], [52, 12], color=INK, lw=3.5); ax.plot([70], [32], "o", color=INK, ms=10)
    ax.text(70, 58, "values cross", fontsize=FS * 2.6, ha="center", va="top", color=INK)
    ax.plot([215], [32], "o", color=INK, ms=10); th = np.linspace(0, 2 * np.pi, 200); ax.plot(215 + 22 * np.cos(th), 32 + 18 * np.sin(th), color=INK, lw=3.5)
    ax.annotate("", (215 + 22 * np.cos(0.3), 32 - 18 * np.sin(0.3)), (215 + 22 * np.cos(0.5), 32 - 18 * np.sin(0.5)), arrowprops=dict(arrowstyle="-|>", color=INK, lw=3))
    ax.text(215, 58, "small loop", fontsize=FS * 2.6, ha="center", va="top", color=INK)
patch((15, 598, 80, 640), lambda ax: None, bg=ABG)
patch((18, 622, 298, 681), coll, bg=ABG)
patch((165, 681, 298, 696), lambda ax: None, bg=ABG)
# (3B) end labels
label((530, 641, 590, 660), "state 1", 12, bg=BBG, ha="left")
label((527, 692, 590, 711), "state 2", 12, bg=BBG, ha="left")
# (3C) Stokes event: before / after
def stokes(ax):
    def panel(x0, after):
        s0 = (x0 + 12, 70); ends = [(x0 + 95, 20), (x0 + 95, 70), (x0 + 95, 120)]
        ax.plot(*s0, "o", color=INK, ms=11); ax.text(s0[0] - 2, s0[1] + 14, "$s_0$", fontsize=FS * 3.0, ha="center", color=INK)
        for (ex, ey) in ends: ax.plot([ex], [ey], "o", color=INK, ms=11)
        ax.plot([s0[0], ends[0][0]], [s0[1], ends[0][1]], color=INK, lw=3); ax.plot([s0[0], ends[2][0]], [s0[1], ends[2][1]], color=INK, lw=3)
        mv = (x0 + 58, 52) if not after else (x0 + 58, 82)                      # the moving critical value
        ax.plot(*mv, "o", color="#b02a2a", ms=13, zorder=5)
        if not after:
            ax.plot([s0[0], ends[1][0]], [s0[1], ends[1][1]], color=INK, lw=3)
            ax.annotate("", (x0 + 58, 90), (x0 + 58, 60), arrowprops=dict(arrowstyle="-|>", color="#b02a2a", lw=2.5))
        else:
            t = np.linspace(0, 1, 200); xs = s0[0] + t * (ends[1][0] - s0[0]); ys = 70 + 24 * np.exp(-((xs - mv[0]) / 12) ** 2)
            ax.plot(xs, ys, color=INK, lw=3)                                       # the crossed path now passes around the value
        ax.text(x0 + 55, 131, "after" if after else "before", fontsize=FS * 3.0, ha="center", va="center", color=INK)
    panel(5, False); panel(170, True)
    ax.annotate("", (168, 70), (118, 70), arrowprops=dict(arrowstyle="-|>", color=INK, lw=3.5))
    ax.text(143, 52, "Hurwitz\nmove", fontsize=FS * 2.7, ha="center", va="center", color=INK)
patch((600, 600, 888, 738), stokes, bg=CBG)
A.save("graphical_summary_forced.png")
# (3C footer) reconciled wording: a Stokes event is a letter of the path's braid word relative to s0, not a local event of the values
A_ = A
def foot(ax):
    ax.text(141, 11, "a letter relative to $s_0$, not a local event ·", fontsize=8.5 * 3.3, ha="center", va="center", color=INK)
    ax.text(141, 31, "intersection numbers must be computed", fontsize=8.5 * 3.3, ha="center", va="center", color=INK)
patch((600, 740, 882, 782), foot, bg=(215 / 255, 120 / 255, 116 / 255))
A.save("graphical_summary_forced.png")
