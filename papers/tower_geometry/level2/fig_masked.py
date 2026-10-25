"""fig_masked.py -- given measurements, partly masked (note Section 10; data from engine/check_masked.py -> masked_results.json).
(a) angular error of the recovered hidden directions against the masking level, split by whether each side keeps known directions
spanning R^3 (the linear fit can locate them) or not; (b) cost: least-squares fits evaluated (exhaustive lattice L^h for h <= 2, lattice
sweeps beyond) and the success rate of plain local search; (c) one hidden direction with spanning known directions: the linear fit's
residual over the lattice against the angle to the truth (one basin); (d) planar known directions: the linear fit is uninformative, the
cone is not: the CHSH maximum over the feasible (positive) explanations, against the truth."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "masked_results.json")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 4, figsize=(8.6, 2.8), gridspec_kw={"width_ratios": [1.15, 1.1, 1, 1]}); a, b, c, d = ax
levs = list(R["levels"]); xs = {lev: i for i, lev in enumerate(levs)}
for lev, recs in R["levels"].items():
    for k, r in enumerate(recs):
        if r["h"] == 0: continue
        x = xs[lev] + (k - len(recs) / 2) * 0.07
        a.semilogy(x, max(r["err"], 0.05), "o" if r["identifiable"] else "s", ms=4, color="#2a78d6" if r["identifiable"] else "#c8423b",
                   mfc="none" if not r["covered"] else None)
a.semilogy([], [], "o", color="#2a78d6", label="each side spans $\\mathbb{R}^3$"); a.semilogy([], [], "s", color="#c8423b", label="a side does not")
a.set_xticks(range(1, len(levs))); a.set_xticklabels(levs[1:]); a.set_xlim(0.5, len(levs) - 0.5); a.set_xlabel("measurements masked")
a.set_ylabel("error of hidden directions (deg)"); a.legend(frameon=False, fontsize=6, loc="lower right", bbox_to_anchor=(1.0, 0.0)); a.set_title("(a) recovery", loc="left", fontsize=8.5)
hs = []; ev = []; ls = []
for lev, recs in R["levels"].items():
    if recs[0]["h"] == 0: continue
    hs.append(recs[0]["h"]); ev.append(np.median([r["evals"] for r in recs])); ls.append(np.mean([r["local_success"] for r in recs]))
hh = np.arange(1, 6); b.semilogy(hh, 300.0 ** hh, color="#888", ls="--", lw=0.9, label="exhaustive $300^h$")
b.semilogy(hs, ev, "o-", color="black", ms=4, label="fits used"); b.set_xlabel("hidden directions $h$"); b.set_ylabel("least-squares fits")
b2 = b.twinx(); b2.plot(hs, ls, "s-", color="#eb6834", ms=4); b2.set_ylim(0, 1.05); b2.set_ylabel("local-search success", color="#eb6834")
b2.spines["top"].set_visible(False); b.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(0.0, 0.45)); b.set_xticks([1, 2, 3, 4, 5]); b.set_title("(b) cost", loc="left", fontsize=8.5)
G = np.array(R["cone"]["span_rss"][0]); rss = np.array(R["cone"]["span_rss"][1]); t = np.array(R["cone"]["true_span"])
ang = np.degrees(np.arccos(np.clip(G @ t, -1, 1))); c.semilogy(ang, rss, ".", ms=2, color="#2e9e6b")
c.set_xlabel("angle to the true direction (deg)"); c.set_ylabel("residual of the linear fit"); c.set_title("(c) spanning: one basin", loc="left", fontsize=8.5)
ch = np.array(R["cone"]["chsh_samples"]); d.hist(ch, bins=30, color="#9b59b6", alpha=0.8); d.axvline(R["cone"]["chsh_true"], color="black", lw=1.2)
d.text(R["cone"]["chsh_true"], d.get_ylim()[1] * 0.92, " truth", fontsize=6.5)
d.set_xlabel("CHSH maximum of positive explanations"); d.set_ylabel("samples")
d.set_title("(d) planar: only the cone", loc="left", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_masked.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_masked.png"), dpi=130, bbox_inches="tight")
print("ok")
