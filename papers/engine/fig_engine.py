"""fig_engine.py -- figure for the engine report, from demos/synthetic_results.json and study4/study4_results.json.
(a) incoherence detection: AUROC of the consistency radius versus the confidence of averaged answers, against data per context (D1);
(b) Study 4: gain in log loss from keeping where disagreement sits, per task and pooled, with 95% intervals (pre-registered).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
syn = json.load(open(os.path.join(ROOT, "demos", "synthetic_results.json")))
s4 = json.load(open(os.path.join(ROOT, "study4", "study4_results.json")))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5), gridspec_kw={"wspace": 0.35})
d1 = syn["D1_incoherence_detection"]; ns = [int(n) for n in d1]
a1.plot(ns, [d1[str(n)]["auroc_radius"] for n in ns], "o-", color="#2b5fa8", label="consistency radius (engine)")
a1.plot(ns, [d1[str(n)]["auroc_averaged_confidence"] for n in ns], "s--", color="#c2410c", label="confidence of averaged answers")
a1.axhline(0.5, color="#9aa6b2", lw=0.8, ls=":"); a1.set_xscale("log"); a1.set_ylim(0.3, 1.0)
a1.set_xlabel("observations per pairwise context"); a1.set_ylabel("AUROC for detecting incoherence")
a1.set_title("(a) averaging cannot see incoherence", loc="left", fontsize=9.5); a1.legend(frameon=False, fontsize=8, loc="upper left")

names = {"USMLE": "USMLE", "PRIME_FULL": "primes", "COUNTAHAPPYNUMBER": "happy numbers", "HOTPOTQA": "HotpotQA", "ARC": "ARC"}
tasks = list(s4["per_task"]); y = list(range(len(tasks) + 1))[::-1]
vals = [s4["per_task"][t]["delta"] for t in tasks] + [s4["primary"]["delta_bits"]]
cis = [s4["per_task"][t]["ci"] for t in tasks] + [s4["primary"]["ci"]]
labels = [f"{names[t]} (n={s4['per_task'][t]['n']})" for t in tasks] + [f"pooled (n={s4['n_items']})"]
for yi, v, (lo, hi), lab in zip(y, vals, cis, labels):
    col = "#18222d" if lab.startswith("pooled") else "#2b5fa8"
    a2.plot([lo, hi], [yi, yi], color=col, lw=1.6); a2.plot([v], [yi], "o", color=col, ms=5 if lab.startswith("pooled") else 4)
a2.axvline(0, color="#9aa6b2", lw=0.8, ls=":"); a2.set_yticks(y); a2.set_yticklabels(labels, fontsize=8)
a2.set_xlabel("log-loss gain over averaging (bits per item)")
a2.set_title("(b) Study 4: where disagreement sits (pre-registered)", loc="left", fontsize=9.5)
fig.savefig(os.path.join(HERE, "fig_engine.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_engine.png"), dpi=150, bbox_inches="tight")
print("wrote report/fig_engine.pdf")
