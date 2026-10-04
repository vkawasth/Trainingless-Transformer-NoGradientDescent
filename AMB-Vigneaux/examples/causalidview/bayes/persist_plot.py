"""Chart for persist.py: Manski / IV RMSE along the data stream (persistent chains vs reruns vs stale), and work per update."""
import os, glob, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
h = os.path.dirname(os.path.abspath(__file__)); R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(h, "persist", "w*.json")))]
n = np.array(R[0]["sizes"]); ink, muted, blue, orange, green, grey, red = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#b8b6ae", "#d6402a"
fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
for a in ax:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.tick_params(colors=muted, labelsize=7)
for j, view in enumerate(("Manski", "IV")):
    for k, col in ((100, orange), (300, blue)):
        P = np.array([r["persistent_%d" % k]["path"] for r in R])[:, :, j].mean(0); ax[j].plot(n, P, "-o", ms=3, color=col, label="persistent, %d it/update" % k)
    ax[j].plot([768, 1024], [np.mean([r["rerun_%d" % m]["score"][j] for r in R]) for m in (768, 1024)], "s", ms=7, color=green, label="rerun from scratch")
    ax[j].axhline(np.mean([r["stale_at_1024"][j] for r in R]), color=grey, ls="--", lw=1, label="stale (n = 512)")
    ax[j].set_xlabel("context units seen", fontsize=7.5, color=muted); ax[j].set_ylabel("%s endpoint RMSE" % view, fontsize=7.5, color=muted)
    ax[j].set_title("(%s) %s bounds along the stream" % ("ab"[j], view), fontsize=8.5, loc="left"); ax[j].legend(fontsize=6.3, frameon=False)
w = [np.mean([r["persistent_100"]["update_work"] / 8 for r in R]), np.mean([r["persistent_300"]["update_work"] / 8 for r in R]), np.mean([r["rerun_1024"]["work"] for r in R])]
ax[2].bar(range(3), w, color=[orange, blue, green]); ax[2].set_yscale("log"); ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["persistent\n100", "persistent\n300", "rerun"], fontsize=7)
for i, v in enumerate(w): ax[2].text(i, v * 1.15, "%.1e" % v, ha="center", fontsize=6.5)
ax[2].set_ylabel("likelihood work per update (unit-iterations)", fontsize=7.5, color=muted); ax[2].set_title("(c) cost of one update", fontsize=8.5, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(h, "persist_chart.pdf")); fig.savefig(os.path.join(h, "persist_chart.png"), dpi=150)
