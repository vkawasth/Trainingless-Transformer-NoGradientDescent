"""Figure for Study 14 (magnesium in acute MI). (a) the trials: odds ratios with 95% intervals by source; (b) the margin M at each
threshold h before and after ISIS-4, with the observed non-rejection band B; (c) leave-one-out within the small trials before ISIS-4."""
import json, math, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "engine", "study14_magnesium")
sys.path.insert(0, D); import run_study14 as R
res = json.load(open(os.path.join(D, "results_study14.json"))); rows, _ = R.load(os.path.join(D, "data_study14.csv"))
BLUE, ORANGE, VIOLET, GREY, INK2 = "#2a78d6", "#eb6834", "#6250d6", "#8c8c8c", "#52514e"
COL = {"S": BLUE, "L": ORANGE, "I": VIOLET}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.3), gridspec_kw=dict(width_ratios=[1.15, 1.2, 1]))
a = ax[0]; ys = list(range(len(rows) + 3, 0, -1)); k = 0; labels = []
for t in rows:
    th, v = R.log_or(t); se = math.sqrt(v); y = ys[k]; k += 1
    a.plot([math.exp(th - 1.96 * se), math.exp(th + 1.96 * se)], [y, y], color=COL[t["source"]], lw=1.6)
    a.plot(math.exp(th), y, "s", color=COL[t["source"]], ms=3 + 2.2 * math.log10(1 / v + 1))
    labels.append((y, t["trial"], False))
for lab, key, y in (("pooled 1991 (Peto)", ("T0", "peto"), ys[k]), ("pooled before ISIS-4", ("T1", "pool"), ys[k + 1]), ("pooled after ISIS-4", ("T2", "pool"), ys[k + 2])):
    p = res[key[0]][key[1]]; a.plot([math.exp(p["theta"] - 1.96 * p["se"]), math.exp(p["theta"] + 1.96 * p["se"])], [y, y], color="black", lw=1.6)
    a.plot(math.exp(p["theta"]), y, "D", color="black", ms=5); labels.append((y, lab, True))
a.axvline(1, color=GREY, lw=0.8, ls=":"); a.set_xscale("log"); a.set_xlim(0.02, 12); a.set_yticks([y for y, _, _ in labels]); a.set_yticklabels([n for _, n, _ in labels], fontsize=7.5)
for tl, (_, _, bold) in zip(a.get_yticklabels(), labels): tl.set_fontweight("bold" if bold else "normal")
a.tick_params(axis="y", length=0); a.spines["left"].set_visible(False)
a.set_xlabel("odds ratio of death, magnesium vs control"); a.set_title("(a) trials: small (blue), LIMIT-2 (orange), ISIS-4 (violet)", fontsize=9)

b = ax[1]; hs = [0, 0.5, 1, 2]
for t, col, off, lab in (("T1", ORANGE, -0.08, "before ISIS-4"), ("T2", VIOLET, 0.08, "after ISIS-4")):
    B = res[t]["B"]; Ms = [res[t]["verdicts"]["0.1"][str(h / 100)]["M"] for h in hs]
    for h, M in zip(hs, Ms):
        b.plot([h + off, h + off], [M - B, M + B], color=col, lw=5, alpha=0.35, solid_capstyle="butt")
    b.plot([h + off for h in hs], Ms, "o-", color=col, lw=1.4, ms=5, label=f"{lab} (bars: $\\pm B$, $B$={B:.2f})")
b.axhline(1.96, color=GREY, lw=0.9, ls="--"); b.axhline(-1.96, color=GREY, lw=0.9, ls="--"); b.axhline(0, color=GREY, lw=0.5)
b.text(2.25, 2.1, "for", fontsize=8, color=INK2); b.text(2.25, -2.6, "against", fontsize=8, color=INK2)
b.set_xticks(hs); b.set_xlabel("threshold $h$: required mortality reduction (percentage points)"); b.set_ylabel("margin $M$ (pooled s.e.)")
b.set_xlim(-0.3, 2.6); b.legend(frameon=False, fontsize=7.5, loc="lower left")
b.set_title("(b) settled for treatment needs $M-B>1.96$;\nagainst needs $M+B<-1.96$ ($p_0=0.10$)", fontsize=9)

c = ax[2]; S = [t for t in rows if t["source"] == "S"]; L = [t for t in rows if t["source"] == "L"]; vals = []
for i in range(len(S)):
    S2 = S[:i] + S[i + 1:]; p = R.iv(S2 + L); ct = R.contrast(S2, L); B = 1.96 * math.sqrt(ct["pi"] / (1 - ct["pi"]))
    vals.append((S[i]["trial"], R.verdicts(p, B, 0.10)["0.0"]["M"] - B))
full = res["T1"]["verdicts"]["0.1"]["0.0"]["M"] - res["T1"]["B"]
c.barh(range(len(vals)), [v for _, v in vals], color=[BLUE if v > 1.96 else GREY for _, v in vals], height=0.6)
c.axvline(1.96, color=INK2, lw=0.9, ls="--"); c.axvline(full, color=ORANGE, lw=1.2); c.text(full + 0.03, -0.55, f"all seven: {full:.2f}", color=ORANGE, fontsize=7.5)
c.set_yticks(range(len(vals))); c.set_yticklabels([f"without {n}" for n, _ in vals], fontsize=7.5); c.invert_yaxis()
c.set_xlabel("$M-B$ before ISIS-4, $h=0$"); c.set_title("(c) the $h=0$ verdict is fragile: 3 of 7 drops\nfall below 1.96 (grey)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_magnesium.pdf")); fig.savefig(os.path.join(HERE, "fig_magnesium.png"), dpi=150); print("ok")
