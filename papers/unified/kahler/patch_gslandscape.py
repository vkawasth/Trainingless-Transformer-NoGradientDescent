"""Patch the author-supplied Gemini 'geometric landscape' image -> graphical_summary_landscape.png (sec:overview).
Fixes: state labels I0, I1, I2, I3, It; collision box 'crossing sigma / loop sigma^2' (stray glyph removed); caustic box
'A2 fold, sigma^3, <V,V'> = +-1' with 'side: convention'; Stokes 'computed'; 'Escape loci (3x3)'; duplicate labels replaced
('Hypotheses J_k', ring 'one cut' label); 'Vanishing cycle'; upper duplicate boundary label removed; legend line rewritten
(one state seen in every layer)."""
import io, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
SRC = "/mnt/user-data/attachments/6cdf1ee2-7dbd-5dd7-9fad-7a5f343db875/c24f9a08-image.jpg"
A = Image.open(SRC).convert("RGB"); ARR = np.asarray(A).astype(float)
FG = "#e8edf3"
def bgof(box):
    x0, y0, x1, y1 = box; b = np.concatenate([ARR[y0, x0:x1], ARR[y1 - 1, x0:x1], ARR[y0:y1, x0], ARR[y0:y1, x1 - 1]]); return tuple(np.median(b, 0) / 255)
def render(w, h, draw, bg, scale=3):
    fig = plt.figure(figsize=(w * scale / 100, h * scale / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    draw(ax); buf = io.BytesIO(); fig.savefig(buf, dpi=100, facecolor=bg); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB").resize((w, h), Image.LANCZOS)
def patch(box, draw=lambda ax: None, bg=None):
    bg = bg or bgof(box); A.paste(render(box[2] - box[0], box[3] - box[1], draw, bg), (box[0], box[1]))
def text(box, s, size, ha="center", color=FG, bg=None, **kw):
    def dr(ax):
        w, h = box[2] - box[0], box[3] - box[1]; x = {"center": w / 2, "left": 2, "right": w - 2}[ha]
        ax.text(x, h / 2, s, fontsize=size * 2.9, ha=ha, va="center", color=color, **kw)
    patch(box, dr, bg)
# state labels
text((204, 497, 230, 526), "$I_0$", 15)
text((548, 418, 574, 447), "$I_2$", 15)
text((748, 370, 774, 397), "$I_3$", 15)
# stray glyph box removed; collision box content
patch((436, 346, 500, 412))
text((511, 348, 629, 389), "crossing: $\\sigma$\nloop: $\\sigma^2$", 10.5, bg=bgof((511, 348, 629, 389)))
# caustic box and its label
text((644, 300, 727, 357), "$A_2$ fold: $\\sigma^3$\n$\\langle V,V'\\rangle=\\pm1$", 9, bg=bgof((644, 300, 727, 357)))
text((644, 361, 738, 381), "side: convention", 8.5, ha="left")
# Stokes label
text((770, 356, 896, 376), "intersections: computed", 8, ha="left")
# escape loci
text((548, 155, 705, 186), "Escape loci (3$\\times$3)", 11, ha="left")
# upper duplicate boundary label removed
patch((128, 323, 302, 368))
# duplicate thimble label, vanishing cycle
text((842, 145, 962, 192), "Hypotheses $J_k$", 11)
text((1178, 228, 1340, 255), "Vanishing cycle", 11.5, ha="left")
# rings: left duplicate labels -> one cut
text((931, 368, 1030, 466), "one cut per\nindependent\ncycle (×)", 10.5)
# legend line
text((255, 723, 1122, 751), "$I_0,\\dots,I_t$: one information state at successive times, seen in every layer at once  ·  $\\sigma$: half-twist  ·  CF: contextual fraction", 10)
A.save("graphical_summary_landscape.png")
