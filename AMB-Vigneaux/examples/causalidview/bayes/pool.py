"""Pool every saved chain per world (genericcells_w{seed}_*.npy: posterior-mean cells per chain), recompute Manski and
IV endpoint RMSE from the pooled posterior-mean cells, compare with the structured backbone and the published TabPFN,
and draw the chart. Also reports the error as a function of the number of chains pooled (the compute lever)."""
import os, sys, glob, json, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R

def evaluate(cells, W, ivo):
    Lo, Uo = R.manski(W["q"].mean(1)); post = cells.mean(0); Lm, Um = R.manski(post.mean(1))
    iv = np.array([R.iv_bounds(post[i]) for i in range(len(post))])
    return R.ep_rmse(Lm, Um, Lo, Uo), R.ep_rmse(iv[:, 0], iv[:, 1], ivo[:, 0], ivo[:, 1]), float(np.mean(iv[:, 2] > 1e-7))

if __name__ == "__main__":
    S = json.load(open(os.path.join(_h, "..", "results.json")))["rows"]["structured (ours, gradient-free)"]; rows = []; curve = {}
    for seed in range(40):
        fs = sorted(glob.glob(os.path.join(_h, "chains", "genericcells_w%d_*.npy" % seed)))
        if not fs: continue
        cells = np.concatenate([np.load(f) for f in fs]); W = R.world(seed)
        ivo = np.array([R.iv_bounds(W["q"][i]) for i in range(len(W["qry"]))])
        m, iv, inf = evaluate(cells, W, ivo)
        for k in sorted({1, 2, 4, 8, len(cells)}):
            if k <= len(cells): curve.setdefault(k, []).append(evaluate(cells[:k], W, ivo)[:2])
        rows.append(dict(seed=seed, family=W["family"], chains=len(cells), manski=m, iv=iv, infeasible=inf, s_manski=S[seed]["manski"], s_iv=S[seed]["iv"]))
        print(seed, W["family"][:10], "chains %d | Manski %.3f (structured %.3f) | IV %.3f (structured %.3f)" % (len(cells), m, S[seed]["manski"], iv, S[seed]["iv"]), flush=True)
    a = lambda k: np.array([r[k] for r in rows])
    summ = dict(worlds=len(rows), chains_min=int(a("chains").min()), manski=(float(a("manski").mean()), float(a("manski").std())), iv=(float(a("iv").mean()), float(a("iv").std())),
                structured=(float(a("s_manski").mean()), float(a("s_iv").mean())), wins=(int((a("manski") < a("s_manski")).sum()), int((a("iv") < a("s_iv")).sum())),
                infeasible=float(a("infeasible").mean()), tabpfn=(0.09, 0.12), logistic=(0.132, 0.169),
                by_family={f: (float(np.mean([r["manski"] for r in rows if r["family"] == f])), float(np.mean([r["iv"] for r in rows if r["family"] == f])),
                               float(np.mean([r["s_manski"] for r in rows if r["family"] == f])), float(np.mean([r["s_iv"] for r in rows if r["family"] == f])))
                           for f in sorted({r["family"] for r in rows})},
                chain_curve={int(k): (float(np.mean([v[0] for v in vs])), float(np.mean([v[1] for v in vs])), len(vs)) for k, vs in curve.items()})
    json.dump(dict(rows=rows, summary=summ), open(os.path.join(_h, "pooled.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summ, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, green, grey, navy = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#b8b6ae", "#2c3e7a"
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for x in ax:
        for sp in ("top", "right"): x.spines[sp].set_visible(False)
        x.tick_params(colors=muted, labelsize=7)
    labs = ["logistic", "structured", "Bayes\n(ours)", "TabPFN"]
    for j, (view, name) in enumerate((("manski", "Manski"), ("iv", "monotone IV"))):
        v = [summ["logistic"][j], summ["structured"][j], summ[view][0], summ["tabpfn"][j]]
        ax[0].bar(np.arange(4) + 5 * j, v, color=[grey, green, blue, navy])
        for i, y in enumerate(v): ax[0].text(i + 5 * j, y + 0.002, "%.3f" % y, ha="center", fontsize=6.5)
        ax[0].text(1.5 + 5 * j, 0.185, name, ha="center", fontsize=8)
    ax[0].set_xticks(list(range(4)) + [5 + i for i in range(4)]); ax[0].set_xticklabels(labs * 2, fontsize=6.5, rotation=45, ha="right"); ax[0].set_ylim(0, 0.195)
    ax[0].set_ylabel("endpoint RMSE (40 worlds)", fontsize=7.5, color=muted); ax[0].set_title("(a) a posterior under a generic prior, no gradients", fontsize=8.5, loc="left")
    fams = list(summ["by_family"]); x = np.arange(len(fams))
    ax[1].bar(x - 0.2, [summ["by_family"][f][2] for f in fams], 0.4, color=green, label="structured backbone")
    ax[1].bar(x + 0.2, [summ["by_family"][f][0] for f in fams], 0.4, color=blue, label="generic-prior Bayes")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f.split("_")[0] for f in fams], fontsize=7); ax[1].set_ylabel("Manski RMSE", fontsize=7.5, color=muted)
    ax[1].legend(fontsize=6.5, frameon=False); ax[1].set_title("(b) by effect family", fontsize=8.5, loc="left")
    ks = sorted(summ["chain_curve"]); ax[2].plot(ks, [summ["chain_curve"][k][0] for k in ks], "-o", ms=4, color=blue, label="Manski")
    ax[2].plot(ks, [summ["chain_curve"][k][1] for k in ks], "-o", ms=4, color=orange, label="IV")
    ax[2].axhline(0.09, color=blue, ls="--", lw=0.8); ax[2].axhline(0.12, color=orange, ls="--", lw=0.8)
    ax[2].set_xscale("log", base=2); ax[2].set_xticks(ks); ax[2].set_xticklabels(ks)
    ax[2].set_xlabel("chains pooled (CPU only)", fontsize=7.5, color=muted); ax[2].legend(fontsize=6.5, frameon=False)
    ax[2].set_title("(c) more chains help, then plateau:\nthe rest is the prior (dashed: TabPFN)", fontsize=8.5, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(_h, "bayes_chart.pdf")); fig.savefig(os.path.join(_h, "bayes_chart.png"), dpi=150)
