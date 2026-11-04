"""Figures for section 12.8 of the companion note (Stage IV): the sheet of column margins and a path through the column-binding example.
  fig_s4_sheet.pdf    the (g2, g3) sheet at f = (2/5, 3/5): Kahler chambers (blow-up walls), leximin types (walls of level 1 and 2), the path and the example
  fig_s4_objects.pdf  the five objects over the pentagon at four points of the path (exact fibres and multiplicities)
  fig_s4_track.pdf    along the path: real critical points and the collision; the Clifford scale of the positive object; Stokes alignments"""
import os, sys
import numpy as np
import mpmath as mp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import pentpath as PP, couplingfibre as CF, datawalls as DW
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
f = (0.4, 0.6); WHICH = set(sys.argv[1:]) or {"sheet", "objects", "track"}

# ---------------------------------------------------------------- the sheet
def lextype(g2, g3):
    g = (1 - g2 - g3, g2, g3); Fc = CF.facet_cells(list(f), list(g)); V, L = CF.cell_functions(f[0], g[0], g[1])
    u, lev = DW.lex_maxmin([V[k] for k in Fc], [float(L[k]) for k in Fc])
    return len(Fc), tuple(tuple(sorted(Fc[i] for i in S)) for _, S in lev[:2])
n = 120; xs = np.linspace(0.004, 0.996, n); types = {}; Tf = np.full((n, n), np.nan); Tl = np.full((n, n), np.nan); keys = {}
for i, g2 in enumerate(xs if "sheet" in WHICH else []):
    for j, g3 in enumerate(xs):
        if g2 + g3 >= 0.996: continue
        nf, lt = lextype(g2, g3); Tf[j, i] = nf; Tl[j, i] = keys.setdefault(lt, len(keys))
if "sheet" in WHICH:
  fig, a = plt.subplots(figsize=(7.4, 6.6))
  a.add_patch(Polygon([(0, 0), (1, 0), (0, 1)], closed=True, fc="none", ec=INK2, lw=1))
  cm = plt.get_cmap("Pastel2"); im = np.ma.masked_invalid(Tl)
  a.pcolormesh(xs, xs, im % 8, cmap=cm, shading="nearest", alpha=0.55, rasterized=True)
  for c in (0.4, 0.6):
      a.plot([c, c], [0, 1 - c], color=INK, lw=1.2); a.plot([0, 1 - c], [c, c], color=INK, lw=1.2); a.plot([0, c], [c, 0], color=INK, lw=1.2)
  dx = xs[1] - xs[0]
  for i in range(n):
      for j in range(n):
          if np.isnan(Tl[j, i]): continue
          if i + 1 < n and not np.isnan(Tl[j, i + 1]) and Tl[j, i + 1] != Tl[j, i]: a.plot([xs[i] + dx / 2] * 2, [xs[j] - dx / 2, xs[j] + dx / 2], color=ROSE, lw=0.8)
          if j + 1 < n and not np.isnan(Tl[j + 1, i]) and Tl[j + 1, i] != Tl[j, i]: a.plot([xs[i] - dx / 2, xs[i] + dx / 2], [xs[j] + dx / 2] * 2, color=ROSE, lw=0.8)
  a.plot([0.4, 4 / 15], [0, 4 / 15], color=VIOLET, lw=1.6, ls="--", zorder=4)          # the exact level-2 wall g2 + g3/2 = 2/5 in the pentagon chamber
  a.plot([4 / 15, 0.4], [4 / 15, 4 / 15], color=ORANGE, lw=1.6, ls="--", zorder=4)        # the exact level-1 wall g3 = 4/15 in the pentagon chamber
  a.plot([0.3, 0.3], [0.15, 0.275], color=BLUE, lw=3, zorder=5); a.annotate("", (0.3, 0.282), (0.3, 0.26), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
  a.plot(0.3, 0.2, "o", color=INK, ms=7, zorder=6); a.plot(0.3, 4 / 15, "s", color=ORANGE, ms=6, zorder=6)
  a.annotate("the example $\\gamma=1/5$\n(on a level-2 wall)", (0.3, 0.2), (0.42, 0.12), fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
  a.annotate("level-1 wall $\\gamma=4/15$", (0.3, 4 / 15), (0.42, 0.3), fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
  for i in range(0, n, 9):
      for j in range(0, n, 9):
          if not np.isnan(Tf[j, i]) and xs[i] + xs[j] < 0.93: a.text(xs[i], xs[j], str(int(Tf[j, i])), fontsize=6, color=GREY, ha="center", va="center")
  a.plot([], [], color=INK, lw=1.2, label="blow-up (Kähler) walls $f_1=g(J)$"); a.plot([], [], color=ROSE, lw=0.9, ls="--", label="leximin walls (on a $120\\times120$ grid)")
  a.plot([], [], color=VIOLET, lw=1.6, ls="--", label="exact level-2 wall $g_2+g_3/2=2/5$")
  a.plot([], [], color=ORANGE, lw=1.6, ls="--", label="exact level-1 wall $g_3=4/15$")
  a.plot([], [], color=BLUE, lw=3, label="the path: $g_2=3/10$, $g_3=\\gamma\\in[3/20,11/40]$")
  a.set_xlabel("$g_2$"); a.set_ylabel("$g_3=\\gamma$  ($g_1=1-g_2-g_3$)"); a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal")
  a.legend(frameon=False, fontsize=8, loc="upper right"); a.set_title("The sheet of column margins at $f=(2/5,3/5)$. Grey numbers: facets of the fibre.\nColours: leximin types (the first two level sets)", fontsize=9.5)
  fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_s4_sheet.pdf")); fig.savefig(os.path.join(HERE, "fig_s4_sheet.png"), dpi=150); plt.close(fig)

# ---------------------------------------------------------------- objects over the pentagon
fig, ax = plt.subplots(1, 4, figsize=(12.6, 3.6))
for a, g in zip(ax, [Fr(17, 100), Fr(1, 5), Fr(47, 200), Fr(27, 100)]):
    fF, gg = PP.margins(g); Pp, _ = CF.polygon(fF, gg); PF = [(float(x), float(y)) for x, y in Pp]
    a.add_patch(Polygon(PF, closed=True, fc="#f5f5f5", ec=INK2, lw=1)); lx = PP.leximin(g); u = [float(x) for x in lx["point"]]
    if g == Fr(1, 5): fib = [((Fr(3, 20), Fr(3, 20)), 2), ((Fr(2, 15), Fr(1, 6)), 3)]
    else: fib = PP.newton(g)[2]
    for (x, y), m in fib:
        lex = abs(float(x) - u[0]) < 1e-9 and abs(float(y) - u[1]) < 1e-9
        a.plot(float(x), float(y), "o", ms=6 + 3 * m, color=BLUE if lex else ROSE, alpha=0.85, zorder=5)
        a.annotate(f"{m}", (float(x), float(y)), (float(x) + 0.014, float(y) + 0.012) if lex else (float(x) - 0.03, float(y) - 0.02), fontsize=9, color=INK, zorder=6)
    a.plot(*u, "*", color=INK, ms=9, zorder=7)
    lab = {Fr(17, 100): "column binding,\nlevel 2: column 2", Fr(1, 5): "the example:\na level-2 wall", Fr(47, 200): "column binding,\nlevel 2: row 1", Fr(27, 100): "row binding"}[g]
    a.set_title(f"$\\gamma={g}$: {lab}", fontsize=8.8); a.set_xlim(0, 0.42); a.set_ylim(0, 0.32); a.set_aspect("equal"); a.set_xlabel("$a_{11}$", fontsize=8)
    if a is ax[0]: a.set_ylabel("$a_{12}$", fontsize=8)
    a.tick_params(labelsize=7)
fig.suptitle("Where the five objects sit: blue = the leximin fibre (star), rose = other fibres; the number is how many objects", fontsize=9.5, y=1.0)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_s4_objects.pdf")); fig.savefig(os.path.join(HERE, "fig_s4_objects.png"), dpi=150); plt.close(fig)

# ---------------------------------------------------------------- along the path
if "track" in WHICH:
  fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.8))
  a = ax[0]; gr = np.linspace(0.13, 0.275, 140)
  for L, col in ((50.0, ORANGE), (100.0, BLUE)):
      nr = [sum(PP.kind(o) != "cx" for o in PP.roots(g, L)) for g in gr]; a.step(gr, nr, where="mid", color=col, lw=1.6, label=f"$L={int(L)}$")
      for z in PP.disc_zeros(L): a.axvline(float(z), color=col, ls=":", lw=1)
  for w in (0.2, 4 / 15): a.axvline(w, color=GREY, lw=0.8)
  a.set_ylim(2, 6); a.set_xlabel("$\\gamma$"); a.set_ylabel("real critical points"); a.legend(frameon=False, fontsize=8)
  a.set_title("(a) Five real points become three:\na collision (dotted) at the level-2 wall only", fontsize=9)
  b = ax[1]; gp = np.linspace(0.14, 0.275, 28); sc = []
  for g in gp:
      r4 = PP.roots(g, 400.0); r8 = PP.roots(g, 800.0); P4 = next(o for o in r4 if PP.kind(o) == "++"); P8 = next(o for o in r8 if PP.kind(o) == "++")
      e4 = sorted(mp.eig(P4["Qn"])[0], key=abs); e8 = sorted(mp.eig(P8["Qn"])[0], key=abs); sc.append(float(-(mp.log(abs(e8[0])) - mp.log(abs(e4[0]))) / 400))
  gg = np.linspace(0.14, 0.275, 400); pred = np.where(gg < 0.2, 0.15 - gg / 2, np.where(gg < 4 / 15, 0.2 - 0.75 * gg, 0))
  b.plot(gg, pred, color=GREY, lw=1.2, ls="--", label="$h_2-h_1$ (exact)"); b.plot(gp, sc, "o", color=BLUE, ms=4, label="small eigenvalue of $Q/W$, slope $L=400\\to800$")
  for w in (0.2, 4 / 15): b.axvline(w, color=GREY, lw=0.8)
  b.set_xlabel("$\\gamma$"); b.set_ylabel("valuation"); b.legend(frameon=False, fontsize=7.6)
  b.set_title("(b) The positive object's Clifford scale:\na kink at the level-2 wall, closing at the level-1 wall", fontsize=9)
  c = ax[2]; gams = np.linspace(0.12, 0.275, 1551)
  for L, mk in ((25.0, "^"), (50.0, "s"), (100.0, "o")):
      P_, W_ = PP.track(L, gams)
      for th, col in ((0.3, BLUE), (1.0, ORANGE), (np.pi / 2, ROSE)):
          for g, i, j, conj in PP.stokes(L, gams, th, P_, W_): c.plot(1 / L, g, mk, color=col, ms=5)
  x = np.linspace(0, 0.045, 10)
  for w in (0.2, 4 / 15):
      c.axhline(w, color=GREY, lw=0.8); c.fill_between(x, w - x, w + x, color=GREY, alpha=0.12)
  for th, col in ((0.3, BLUE), (1.0, ORANGE), (np.pi / 2, ROSE)): c.plot([], [], "o", color=col, label=f"phase {th:.2f}")
  for L, mk in ((25, "^"), (50, "s"), (100, "o")): c.plot([], [], mk, color=GREY, label=f"$L={L}$")
  c.set_xlabel("$1/L$"); c.set_ylabel("$\\gamma$ of a Stokes alignment"); c.legend(frameon=False, fontsize=7.4, loc="upper right"); c.set_xlim(0, 0.045)
  c.set_title("(c) Stokes alignments at three phases stay in\nthe band $|\\gamma-\\gamma_{wall}|\\leq 1/L$ (shaded)", fontsize=9)
  fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_s4_track.pdf")); fig.savefig(os.path.join(HERE, "fig_s4_track.png"), dpi=150); plt.close(fig); print("ok")
