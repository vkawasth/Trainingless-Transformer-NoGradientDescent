"""Study 89 figures: the caustic (where two critical points merge) and the sigma^3 loop around it."""
import os, sys, copy
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(os.path.dirname(HERE), "engine"); sys.path.insert(0, ENG); os.chdir(ENG)
INK, MUTED, BLUE, ORANGE, GREEN, PURPLE, GRAY = "#1F2937", "#6B7280", "#1D4ED8", "#C2410C", "#047857", "#7C3AED", "#9CA3AF"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
res = {}
for L in ("20", "30"):
    src = open("ks89.py").read().replace('OUT = open(os.path.join(D, f"ks89_L{int(L)}_output.txt"), "a")', 'OUT = open(os.devnull, "w")').replace("pickle.dump(", "_unused = (")
    src = src.replace("cur = Tq.vals[k1] - Tq.vals[k2]; dargs.append(np.angle(cur / prev)); prev = cur", "cur = Tq.vals[k1] - Tq.vals[k2]; dargs.append(np.angle(cur / prev)); prev = cur; TRACE.append(cur)")
    src = src.replace("NL = 400; dargs = []", "NL = 400; dargs = []; TRACE = []")
    sys.argv = ["ks89.py", L]; g = {"__file__": os.path.join(ENG, "ks89.py"), "__name__": "x"}
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()): exec(src, g)
    res[L] = g
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8))
a = axs[0]
for L, col in (("20", BLUE), ("30", ORANGE)):
    g = res[L]; tr = np.array(g["TRACE"]); arg = np.unwrap(np.angle(tr)); al = np.linspace(0, 2 * np.pi, len(arg) + 1)[1:]
    a.plot(al, (arg - arg[0]) / np.pi, color=col, lw=1.5, label=f"$L={L}$")
a.plot([0, 2 * np.pi], [0, 3], color=INK, lw=0.8, ls=(0, (4, 4))); a.text(0.3, 2.4, "model $u^{3/2}$:\nthree half-turns", fontsize=7, color=INK)
a.set_xlabel("loop angle $\\alpha$ around the caustic"); a.set_ylabel("turn of $v_a-v_b$ ($\\times\\pi$)"); a.legend(fontsize=7, frameon=False); a.spines[["top", "right"]].set_visible(False)
a.set_title("critical values of the merging pair", fontsize=8)
b = axs[1]; g = res["20"]; tr = np.array(g["TRACE"]); b.plot(tr.real * 1e5, tr.imag * 1e5, color=BLUE, lw=1.2); b.plot(0, 0, "o", color=PURPLE, ms=5)
b.plot(tr[0].real * 1e5, tr[0].imag * 1e5, "o", color=INK, ms=3); b.set_aspect("equal"); b.set_xlabel("Re$(v_a-v_b)\\times10^5$"); b.set_ylabel("Im$(v_a-v_b)\\times10^5$")
b.set_title("$v_a-v_b$ winds 1.5 times\naround 0 ($L=20$)", fontsize=8); b.spines[["top", "right"]].set_visible(False)
c = axs[2]; xs = np.linspace(0, 1, 50)
for q in range(2): c.plot([0, 1], [q, q], color=GRAY, lw=0.6)
pos = [0.2, 0.5, 0.8]; y = [0, 1]; cols = [GREEN, ORANGE]; state = [0, 1]
segs = {0: [(0, 0)], 1: [(0, 1)]}
for p in pos:
    for s in (0, 1): segs[s].append((p - 0.08, state[s]))
    state = [state[1], state[0]]
    for s in (0, 1): segs[s].append((p + 0.08, state[s]))
for s in (0, 1): segs[s].append((1, state[s])); xx, yy = zip(*segs[s]); c.plot(xx, yy, color=cols[s], lw=2)
for p in pos: c.plot(p, 0.5, "o", mfc="white", mec=ORANGE, mew=1.5, ms=7)
c.set_xticks([]); c.set_yticks([0, 1]); c.set_yticklabels(["$V_3$", "$V_6$"], fontsize=8); c.set_ylim(-0.4, 1.4)
c.set_title("word $\\sigma^{\\pm3}$: pair exchanged,\none object returns shifted", fontsize=8)
for s_ in ("top", "right", "bottom"): c.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig89_K1.pdf")); fig.savefig(os.path.join(HERE, "fig89_K1.png"), dpi=150); plt.close(fig)
print("written", res["20"]["wind"], res["30"]["wind"])
