"""Study 74 figures: (a) A-side vs real-cycle agreement along the three 3x3 paths (exploratory grid + held-out points from
engine/study74_strand/check_strand74_output.txt); (b) shell sums of the own and adjacent chamber expansions on path F (exploratory dump74b)
with the predicted rate L|x0.E_C|. Output: fig_strand74a.pdf/png, fig_strand74b.pdf/png"""
import os, pickle, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); DS = os.path.join(HERE, "..", "engine", "study74_strand")
COL = {"C": "#2a78d6", "F": "#eb6834", "X": "#1baf7a"}; INK = "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": "#0b0b0b", "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
d = pickle.load(open(os.path.join(DS, "dump74.pkl"), "rb"))
fig, ax = plt.subplots(figsize=(5.6, 3.2))
LAB = {"C": "C: margin-order tie (no wall)", "F": "F: flop wall", "X": "X: leximin + flop wall"}
for name in ("C", "F", "X"):
    ts = sorted(t for (n, t) in d if n == name); ys = []
    for t in ts:
        r = d[(name, t)]; b = r["B"][1]; ys.append(min(abs(r["A"][c][1] - b) / abs(b) for c in r["chambers"]))
    ax.plot(ts, np.log10(np.maximum(ys, 1e-16)), "-o", color=COL[name], lw=2, ms=4, label=LAB[name] + " (exploratory)")
held = os.path.join(DS, "check_strand74_output.txt")
if os.path.exists(held):
    txt = open(held).read()
    for m in re.finditer(r"VS1 (\w) t ([+-][\d.]+): .*?\|A16-B\|/\|B\| ([\d.e+-]+)", txt):
        ax.plot(float(m.group(2)), np.log10(max(float(m.group(3)), 1e-16)), "D", ms=8, mfc="white", mec=COL[m.group(1)], mew=2)
    for m in re.finditer(r"VS2 (\w) t ([+-][\d.]+): .*?\|A16-B\|/\|B\| ([\d.e+-]+)", txt):
        ax.plot(float(m.group(2)), np.log10(max(float(m.group(3)), 1e-16)), "s", ms=8, mfc="white", mec=COL[m.group(1)], mew=2)
ax.axvline(0, color=INK, lw=0.6, ls=":"); ax.set_xlabel("path parameter $t$ (wall of F and X at $t=0$)")
ax.set_ylabel(r"$\log_{10}|Z_A(\mathcal{O})-\Theta_R|/|\Theta_R|$"); ax.set_ylim(-15.5, -3.5)
ax.plot([], [], "D", mfc="white", mec=INK, label="held out, Option A (VS1)"); ax.plot([], [], "s", mfc="white", mec=INK, label="held out, Option C (VS2)")
ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.grid(axis="y", color="#e4e3df", lw=0.6)
fig.tight_layout(); [fig.savefig(os.path.join(HERE, "fig_strand74a." + e), dpi=200) for e in ("pdf", "png")]
b = pickle.load(open(os.path.join(DS, "dump74b.pkl"), "rb"))
fig, ax = plt.subplots(figsize=(5.6, 3.2)); n = np.arange(15)
for t, c in ((0.035, "#2a78d6"), (0.015, "#eb6834"), (0.005, "#1baf7a")):
    for lab, ls in (("own", "-"), ("other", "--")):
        sh = b[("F", t, lab)]; ok = sh > 0; ax.plot(n[ok], np.log10(sh[ok]), ls, color=c, lw=2, marker="o" if lab == "own" else "^", ms=3.5)
    rate = 50 * t; sh = b[("F", t, "own")]; k = 10
    for sgn, key in ((-1, "own"), (1, "other")):
        y0 = np.log(b[("F", t, key)][k]); nn = np.arange(7, 15)
        ax.plot(nn, (y0 + sgn * rate * (nn - k) - 2 * np.log(nn / k)) / np.log(10), color=INK, lw=0.8, ls=":")
    ax.text(14.2, np.log10(b[("F", t, "other")][14]), f"$t={t}$", color=c, fontsize=7, va="center")
ax.plot([], [], "-o", color=INK, ms=3.5, label="own chamber"); ax.plot([], [], "--^", color=INK, ms=3.5, label="adjacent chamber")
ax.plot([], [], ":", color=INK, label=r"predicted $e^{\mp L|x_0\cdot E_C|\,n}n^{-2}$")
ax.set_xlabel("shell $n=\\max_{ij}|E_{ij}|$"); ax.set_ylabel(r"$\log_{10}|S_n|$ (path F)"); ax.set_xlim(0, 15.5)
ax.legend(fontsize=7, frameon=False, loc="lower left"); ax.grid(axis="y", color="#e4e3df", lw=0.6)
fig.tight_layout(); [fig.savefig(os.path.join(HERE, "fig_strand74b." + e), dpi=200) for e in ("pdf", "png")]
print("ok")
