"""Study 88 figures (held-out seed 8802). SYNTHETIC DATA ONLY; defensive analysis."""
import os, sys, pickle, itertools
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(os.path.dirname(HERE), "engine"); sys.path.insert(0, ENG); os.chdir(ENG)
import lib88 as A
INK, MUTED, BLUE, ORANGE, GREEN, PURPLE, GRAY = "#1F2937", "#6B7280", "#1D4ED8", "#C2410C", "#047857", "#7C3AED", "#9CA3AF"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
R = pickle.load(open(os.path.join(ENG, "study88_adversarial", "check88_8802.pkl"), "rb")); SC = {s["key"]: s for s in A.build(8802)}
def clean(ax): ax.spines[["top", "right"]].set_visible(False)
# ------------------------------------------------------------------ H1 margin-preserving attack --------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.4)); ys = np.arange(len(R))[::-1]
for y, r in zip(ys, R):
    sc = SC[r["key"]]; n = sc["C"].sum(0); q0 = A.conventional(n)[0]; lo, hi, _ = A.credal(n)
    ax.plot([lo, hi], [y, y], color=GREEN, lw=5, alpha=0.35, solid_capstyle="butt")
    ax.annotate("", xy=(q0 + r["AD1"]["dq"], y), xytext=(q0, y), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
    ax.plot(q0, y, "o", color=ORANGE, ms=3.5)
ax.axvline(A.THETA, color=INK, lw=0.9, ls="--"); ax.text(A.THETA, len(R) - 0.3, " threshold", fontsize=7, color=INK)
ax.set_yticks(ys); ax.set_yticklabels([r["key"] for r in R], fontsize=7); ax.set_xlabel("$q=P(\\mathrm{above}\\mid\\mathrm{seeded},\\ \\mathrm{wet})$"); ax.set_xlim(0, 1)
ax.plot([], [], color=GREEN, lw=5, alpha=0.35, label="credal band (all laws with the trusted margins): unchanged by the attack")
ax.plot([], [], color=ORANGE, marker="o", ms=3.5, label="conventional estimate, before $\\to$ after the margin-preserving attack")
ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12)); ax.set_title("Attack A: three colluding stations rewrite 30% of their reports without changing any trusted margin", fontsize=8.5); clean(ax)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig88_H1.pdf")); fig.savefig(os.path.join(HERE, "fig88_H1.png"), dpi=150); plt.close(fig)
# ------------------------------------------------------------------ H2 injection path --------------------------------------------------
r = max([r for r in R if "AD3" in r and r["AD3"]["ok"] and r["AD2"]["k_conv"]], key=lambda r: len(r["AD3"]["events"]), default=None) or [r for r in R if "AD3" in r and r["AD3"]["ok"]][0]
sc = SC[r["key"]]; n = sc["C"].sum(0); tgt = r["AD2"]["target"]; cm = 5 if tgt == "seed" else 0; unit = A.fabricated(1, 6, cm); kg = r["AD2"]["k_credal"]
ks = np.linspace(0, kg * 1.1, 120).round().astype(int); qs = [A.conventional(n + k * unit)[0] for k in ks]; bd = np.array([A.credal(n + k * unit)[:2] for k in ks])
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
a1.fill_between(ks, bd[:, 0], bd[:, 1], color=GREEN, alpha=0.2, lw=0, label="credal band"); a1.plot(ks, qs, color=ORANGE, lw=1.5, label="conventional estimate")
a1.axhline(A.THETA, color=INK, lw=0.9, ls="--"); a1.set_ylabel("$q$"); a1.legend(fontsize=7, frameon=False, loc="upper left")
if r["AD2"]["k_conv"]: a1.axvline(r["AD2"]["k_conv"], color=ORANGE, lw=0.8, ls=":")
a1.axvline(kg, color=GREEN, lw=0.8, ls=":"); a1.text(kg, 0.02, " credal decision flips", fontsize=7, color=GREEN, transform=a1.get_xaxis_transform())
a1.set_title(f"Attack B on held-out {r['key']}: fabricated reports (seeded, wet, {'above' if tgt == 'seed' else 'below'}) push the decision toward '{tgt}'", fontsize=8.5); clean(a1)
for e in r["AD3"]["events"]:
    a2.plot(e[0], 1 if e[1] == "event" else 0.5, "o" if e[1] == "event" else "D", color=ORANGE if e[1] == "event" else PURPLE, ms=5)
for w in r["AD3"]["walls"]: a2.plot(w, 0, "|", color=GRAY, ms=12, mew=2)
a2.set_yticks([0, 0.5, 1]); a2.set_yticklabels(["chamber wall", "collision", "Stokes event"], fontsize=7); a2.set_ylim(-0.4, 1.4); a2.set_xlabel("number of fabricated reports"); clean(a2)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig88_H2.pdf")); fig.savefig(os.path.join(HERE, "fig88_H2.png"), dpi=150); plt.close(fig)
# ------------------------------------------------------------------ H3 order -------------------------------------------------------------
sel = [r for r in R if r["AD4"]["decs"][0] != r["AD4"]["decs"][1]][:6]; col = {"s": GREEN, "d": ORANGE, "u": "#E5E7EB"}
fig, ax = plt.subplots(figsize=(7.2, 0.45 * 2 * len(sel) + 1.0))
for i, r in enumerate(sel):
    for j, (nm, dec) in enumerate(zip(("first weeks", "last weeks"), r["AD4"]["decs"])):
        y = -(2 * i + j)
        for w, c in enumerate(dec): ax.add_patch(plt.Rectangle((w, y), 0.92, 0.8, color=col[c]))
        ax.text(-0.3, y + 0.4, f"{r['key']}, {nm}", ha="right", va="center", fontsize=7)
    ax.text(13.2, -(2 * i) - 0.1, "same end band, same thimble state" if r["AD4"]["state_eq"] else "", fontsize=7, color=INK, va="center")
ax.set_xlim(-4.5, 19); ax.set_ylim(-2 * len(sel) + 0.6, 1.2); ax.set_xticks(np.arange(13) + 0.46); ax.set_xticklabels(range(1, 14), fontsize=7); ax.set_yticks([])
for s in ("left", "right", "top"): ax.spines[s].set_visible(False)
ax.set_xlabel("week (weekly credal decision: green seed, orange do not seed, grey undetermined)")
ax.set_title("Order attack: the same 90 fabricated reports injected early vs late in each month", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig88_H3.pdf")); fig.savefig(os.path.join(HERE, "fig88_H3.png"), dpi=150); plt.close(fig)
# ------------------------------------------------------------------ H4 Shapley and interaction -----------------------------------------
r = [r for r in R if "AD5" in r and r["AD5"]["recovered"] == sorted(r["colluders"])][0]; sc = SC[r["key"]]; C = sc["C"]; n = C.sum(0); colu = sc["colluders"]; outl = sc["outlier"]
tgt = r["AD2"]["target"]; cm = 5 if tgt == "seed" else 0; ke = r["AD5"]["k_each"]; per = {c: A.fabricated(ke, 6, cm) for c in colu}; full = n + sum(per.values())
without = lambda Sx: full - sum(C[s] + per.get(s, 0) for s in Sx); lo, hi, _ = A.credal(full); cache = {}
def Dv(Sx):
    Sx = frozenset(Sx)
    if Sx not in cache: l, h, _ = A.credal(without(Sx)); cache[Sx] = max(abs(l - lo), abs(h - hi))
    return cache[Sx]
players = list(range(10, 20)); phi, _ = A.shapley(lambda Sx: Dv(Sx) if Sx else 0.0, players)
Im = np.full((10, 10), np.nan)
for i, j in itertools.combinations(players, 2): Im[i - 10, j - 10] = Im[j - 10, i - 10] = Dv([i, j]) - Dv([i]) - Dv([j])
fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
cols = [ORANGE if s in colu else BLUE if s == outl else GRAY for s in players]
b1.bar(range(10), [phi[s] for s in players], color=cols); b1.set_xticks(range(10)); b1.set_xticklabels([f"{s - 9}" for s in players], fontsize=7)
b1.set_xlabel("target station"); b1.set_ylabel("Shapley value of band displacement"); clean(b1)
from matplotlib.patches import Patch
b1.legend(handles=[Patch(color=ORANGE, label="planted colluders"), Patch(color=BLUE, label="honest outlier (wetter site)"), Patch(color=GRAY, label="honest")], fontsize=7, frameon=False, loc="upper left")
b1.set_title("who moves the decision (influence)", fontsize=8.5)
im = b2.imshow(Im, cmap="RdBu_r", vmin=-0.1, vmax=0.1); b2.set_xticks(range(10)); b2.set_xticklabels(range(1, 11), fontsize=7); b2.set_yticks(range(10)); b2.set_yticklabels(range(1, 11), fontsize=7)
for i, j in itertools.combinations(range(10), 2):
    if Im[i, j] > 0.01: b2.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
b2.set_title("who acts together (pairwise interaction; boxed > 0.01)", fontsize=8.5); fig.colorbar(im, ax=b2, fraction=0.046)
fig.suptitle(f"held-out {r['key']}: influence is not coordination", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig88_H4.pdf")); fig.savefig(os.path.join(HERE, "fig88_H4.png"), dpi=150); plt.close(fig)
print("figures written", r["key"])
