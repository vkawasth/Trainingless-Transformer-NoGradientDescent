"""Chart for the support-guided-prior study. Numbers are the printed means of the scripts in this folder
(oracle_support / oracle_subspace / subspace_prior / crossfit_prior / eb_prior / law_prepass / law_oracle: first 10 worlds; eb_hyper: first 6)."""
import os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
R = {"constant (no covariates)": 0.131, "linear ridge, LOO (our backbone)": 0.106,
     "+ TRUE coordinate support": 0.1015, "+ TRUE subspace, linear": 0.0783, "+ TRUE subspace, RBF": 0.072,
     "plug-in learned subspace (pooled)": 0.1142, "plug-in, sequential chain": 0.1157,
     "half data, no prior": 0.1114, "cross-fitted prior (A->B, B->A)": 0.1093,
     "empirical Bayes, shared Sigma": 0.1259, "EB + inverse-Wishart, best nu": 0.1048,
     "pre-pass law (rank, sparsity) + cross-fit": 0.1082, "TRUE law (rank, sparsity) + cross-fit": 0.1096}
NU = {"nu": [1, 5, 20, 50, 200, 1000], "rank 1": [0.1274, 0.1239, 0.1183, 0.1119, 0.1054, 0.1048], "rank 5": [0.1239, 0.1203, 0.1156, 0.1109, 0.1052, 0.1048], "base": 0.1046}
json.dump(dict(manski_rmse=R, eb_hyper=NU, tabpfn_published=0.09), open(os.path.join(here, "results.json"), "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ink, muted, blue, orange, green, grey, navy, red = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#b8b6ae", "#2c3e7a", "#d6402a"
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
for a in ax:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.tick_params(colors=muted, labelsize=7)
names = list(R); vals = [R[k] for k in names]
col = [grey, blue, green, green, green, orange, orange, grey, orange, red, orange, orange, green]
y = np.arange(len(names))[::-1]; ax[0].barh(y, vals, color=col)
for yi, v in zip(y, vals): ax[0].text(v + 0.001, yi, f"{v:.3f}", va="center", fontsize=6.5, color=ink)
ax[0].axvline(0.09, color=navy, ls="--", lw=1); ax[0].text(0.0905, len(names) - 0.6, "TabPFN (published) 0.09", fontsize=6.5, color=navy)
ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=7); ax[0].set_xlim(0.06, 0.14)
ax[0].set_xlabel("Manski endpoint RMSE (lower is better)", fontsize=7.5, color=muted)
ax[0].set_title("(a) a KNOWN subspace beats the transformer; LEARNING it (orange/red)\ndoes not, and even the TRUE law of supports (last bar) adds nothing", fontsize=8, loc="left")
ax[1].semilogx(NU["nu"], NU["rank 1"], "-o", ms=4, color=orange, label="learned rank-1 subspace")
ax[1].semilogx(NU["nu"], NU["rank 5"], "-o", ms=4, color=red, label="learned rank-5 subspace")
ax[1].axhline(NU["base"], color=blue, lw=1, label="isotropic prior (our backbone)")
ax[1].set_xlabel("hyperprior strength ν (pseudo-tasks pulling Σ to isotropic)", fontsize=7.5, color=muted); ax[1].set_ylabel("Manski RMSE", fontsize=7.5, color=muted)
ax[1].legend(fontsize=6.5, frameon=False); ax[1].set_title("(b) more trust in the learned support is monotonely worse:\nthe optimum is no learned support", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(here, "support_chart.pdf")); fig.savefig(os.path.join(here, "support_chart.png"), dpi=150)
