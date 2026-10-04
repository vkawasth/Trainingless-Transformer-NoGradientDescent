"""Tables and a figure from docs_{ARCH}_n{N}.json: decision matrices, theta-band coverage, unknown bounds."""
import os, json, glob, numpy as np
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

h = os.path.dirname(os.path.abspath(__file__))
CASES = ["copy A", "copy B", "switching", "blend 0.2", "blend 0.5", "blend 0.8", "unknown", "half unknown"]
DEC = ["A", "B", "blend", "switching", "unknown"]
TRUE = {"copy A": "A", "copy B": "B", "switching": "switching", "blend 0.2": "blend", "blend 0.5": "blend",
        "blend 0.8": "blend", "unknown": "unknown", "half unknown": "unknown"}


def table(path):
    J = json.load(open(path)); out = {}
    for tag, R in J["results"].items():
        rows = R["rows"]; T = {}
        for c in CASES:
            sub = [r for r in rows if r["case"] == c]
            if not sub: continue
            d = Counter(r["decision"] for r in sub); n = len(sub)
            e = dict(n=n, frac={k: d.get(k, 0) / n for k in DEC}, correct=d.get(TRUE[c], 0) / n,
                     theta=float(np.nanmean([r["theta"] for r in sub])), mu=float(np.mean([r["mu"] for r in sub])),
                     mu_lower=float(np.mean([r["mu_lower"] for r in sub])))
            if c.startswith("blend"):
                lam = float(c.split()[1])
                e["cover"] = float(np.mean([r["theta_band"][0] <= lam <= r["theta_band"][1] for r in sub]))
                e["width"] = float(np.mean([r["theta_band"][1] - r["theta_band"][0] for r in sub]))
            T[c] = e
        known = [r for r in rows if TRUE[r["case"]] != "unknown"]
        T["_false_unknown"] = float(np.mean([r["decision"] == "unknown" for r in known]))
        T["_crit"], T["_budget"] = R["crit"], R["budget"]
        out[tag] = T
    return J["N"], out


def main():
    res = {}
    for p in sorted(glob.glob(os.path.join(h, "docs_*_n*.json"))):
        arch = os.path.basename(p).split("_")[1]
        if arch not in ("docs", "ctx"): continue
        N, T = table(p); res[(arch, N)] = T
    lines = []
    for (arch, N), T in sorted(res.items()):
        for tag, t in T.items():
            lines.append("\n### ARCH=%s, n=%d tokens, %s  (switch threshold %.2f, unknown budget %.2f, false-unknown on known text %.2f)\n"
                         % (arch, N, tag, t["_crit"], t["_budget"], t["_false_unknown"]))
            lines.append("| case | " + " | ".join(DEC) + " | correct | theta | mu | mu lower | band cover | band width |")
            lines.append("|" + "---|" * (len(DEC) + 7))
            for c in CASES:
                if c not in t: continue
                e = t[c]
                lines.append("| %s | " % c + " | ".join("%.2f" % e["frac"][k] for k in DEC) + " | **%.2f** | %.2f | %.2f | %.2f | %s | %s |" % (
                    e["correct"], e["theta"], e["mu"], e["mu_lower"], ("%.2f" % e["cover"]) if "cover" in e else "", ("%.2f" % e["width"]) if "width" in e else ""))
    open(os.path.join(h, "SUMMARY.md"), "w").write("# Context attribution: decision tables\n" + "\n".join(lines) + "\n")
    json.dump({"%s_n%d" % k: v for k, v in res.items()}, open(os.path.join(h, "summary.json"), "w"), indent=1)

    # figure: accuracy by case vs n (docs), plus the ctx failure on half unknown
    docsN = sorted(N for (a, N) in res if a == "docs")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, tag in enumerate(["different-event pair (easy)", "same-event pair (hard)"]):
        for c in CASES:
            ys = [res[("docs", N)][tag][c]["correct"] for N in docsN if tag in res[("docs", N)] and c in res[("docs", N)][tag]]
            ax[j].plot(docsN[:len(ys)], ys, "o-", label=c)
        ax[j].set_title("ARCH=docs, " + tag); ax[j].set_xlabel("generated tokens n"); ax[j].set_ylabel("fraction correct"); ax[j].set_ylim(0, 1.02); ax[j].grid(alpha=.3)
    ax[0].legend(fontsize=7, loc="lower right")
    # theta recovery with bands, n=100 docs, easy and hard
    if ("docs", 100) in res:
        J = json.load(open(os.path.join(h, "docs_docs_n100.json")))
        for k, (tag, col) in enumerate([("different-event pair (easy)", "C0"), ("same-event pair (hard)", "C3")]):
            rows = J["results"][tag]["rows"]
            for c in ["copy B", "blend 0.2", "blend 0.5", "blend 0.8", "copy A"]:
                lam = {"copy B": 0.0, "copy A": 1.0}[c] if c.startswith("copy") else float(c.split()[-1])
                sub = [r for r in rows if r["case"] == c][:15]
                for i, r in enumerate(sub):
                    x = lam + (k - 0.5) * 0.04 + (i - 7) * 0.002
                    ax[2].plot([x, x], r["theta_band"], color=col, alpha=.35, lw=1); ax[2].plot(x, r["theta"], ".", color=col, ms=3)
            ax[2].plot([], [], color=col, label=tag)
        ax[2].plot([0, 1], [0, 1], "k--", lw=.8); ax[2].set_xlabel("true share of A"); ax[2].set_ylabel("estimated theta with 95% band")
        ax[2].set_title("theta and its band, n=100"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(h, "attribution.png"), dpi=130)
    print(open(os.path.join(h, "SUMMARY.md")).read())


if __name__ == "__main__":
    main()
