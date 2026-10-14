"""fig_agents.py -- figure for the companion note "Pooling Informed Agents".
(a) stop-loss transforms t -> E(q - t)+ of agents a, b, c (Example 3): c lies below a, b crosses both.
(b) expected log loss of the log-odds pool logit F = logit q_i + gamma * logit q_j against gamma, for the two couplings of Example 5;
    the shaded band is the set of gamma for which the pool is at least as good as agent i alone (the KL ball along this family).
Output: figs/fig_agents.pdf
"""
import math, itertools, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

laws = {"a (accuracy 0.8)": {0.2: 0.5, 0.8: 0.5},
        "b (reveals X w.p. 0.2)": {0.0: 0.1, 0.5: 0.8, 1.0: 0.1},
        "c (a garbled)": {0.35: 0.5, 0.65: 0.5}}
cols = {"a (accuracy 0.8)": "#2b5fa8", "b (reveals X w.p. 0.2)": "#c2410c", "c (a garbled)": "#0f7b6c"}
lg = lambda p: math.log(p / (1 - p)); sg = lambda z: 1 / (1 + math.exp(-z))

def loss_bits(gam, coupling):
    L = 0.0
    for x, a, b in itertools.product((0, 1), repeat=3):
        pa = 0.8 if a == x else 0.2
        pb = (0.75 if b == a else 0.25) if coupling == "garbling" else (0.65 if b == x else 0.35)
        p = 0.5 * pa * pb; qi = 0.8 if a else 0.2; qj = 0.65 if b else 0.35
        F = sg(lg(qi) + gam * lg(qj)); L += p * -math.log2(F if x else 1 - F)
    return L

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"wspace": 0.32})
t = np.linspace(0, 1, 401)
for k, law in laws.items():
    ax1.plot(t, [sum(m * max(q - s, 0) for q, m in law.items()) for s in t], color=cols[k], lw=1.8, label=k)
ax1.set_xlabel("$t$"); ax1.set_ylabel(r"stop-loss  $\mathbb{E}(q-t)_+$"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 0.52)
ax1.set_title("(a) Blackwell order by stop-loss", fontsize=9.5, loc="left")
ax1.legend(frameon=False, fontsize=8, loc="upper right")
for s in (0.5, 0.8): ax1.axvline(s, color="#9aa6b2", lw=0.7, ls=":")
ax1.annotate("b below a, c", xy=(0.5, 0.05), xytext=(0.12, 0.16), fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.6))
ax1.annotate("b above a, c", xy=(0.8, 0.02), xytext=(0.74, 0.17), fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.6))

g = np.linspace(0, 2.5, 251)
base = loss_bits(0, "independent")
for coupling, col in (("garbling", "#5b6876"), ("independent", "#2b5fa8")):
    ax2.plot(g, [loss_bits(x, coupling) for x in g], color=col, lw=1.8, label=f"{coupling} coupling")
ind = np.array([loss_bits(x, "independent") for x in g])
ax2.fill_between(g, 0.66, 0.90, where=ind <= base + 1e-12, color="#2b5fa8", alpha=0.10, lw=0)
ax2.axhline(base, color="#18222d", lw=0.8, ls="--"); ax2.text(2.48, base + 0.004, "agent $i$ alone", ha="right", va="bottom", fontsize=7.5)
ax2.plot([1], [loss_bits(1, "independent")], "o", color="#2b5fa8", ms=4)
ax2.annotate("Bayes pool ($\\gamma=1$)", xy=(1, loss_bits(1, "independent")), xytext=(1.25, 0.668), fontsize=7.5,
             arrowprops=dict(arrowstyle="->", lw=0.6))
ax2.set_xlabel(r"weight $\gamma$ on agent $j$ in $\mathrm{logit}\,F=\mathrm{logit}\,q_i+\gamma\,\mathrm{logit}\,q_j$")
ax2.set_ylabel("expected log loss (bits)"); ax2.set_xlim(0, 2.5); ax2.set_ylim(0.66, 0.90)
ax2.set_title("(b) the KL ball along the log-odds family", fontsize=9.5, loc="left")
ax2.legend(frameon=False, fontsize=8, loc="upper left")
os.makedirs("figs", exist_ok=True)
fig.savefig("figs/fig_agents.pdf", bbox_inches="tight"); fig.savefig("figs/fig_agents.png", dpi=150, bbox_inches="tight")
inside = g[ind <= base + 1e-12]
print(f"wrote figs/fig_agents.pdf; independent coupling: pool beats i for gamma in ({inside[inside > 0].min():.2f}, {inside.max():.2f})")
