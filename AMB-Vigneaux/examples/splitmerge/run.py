"""Split-merge and singular learning theory: learning coefficients of merge strata, and free energy as the merge criterion.

Model of one grammar node: a parent symbol z in {1..K} emits d children independently, each categorical on V symbols
(a latent class model; with d >= 3 views it is generically identifiable up to relabelling, the 'two-level window' of
local EM). A merge identifies two parent symbols; the merge stratum is where their child tables coincide.
(A) Learning coefficients by K, for
      lone node: d = 1 child drawn from V^2 = 16 cells (a mixture of single categorical draws: every K gives the same
                 family, so extra symbols are free -- the root / lone-node non-identifiability), and
      window:    d = 3 children on V = 4 (identifiable), true K0 = 2.
    lambda_hat (6 chains) against d/2.
(B) The merge decision for a spurious split (truth K0 = 2, candidate K = 3) and for a real but weak split (truth K0 = 3
    with two parent symbols close), across n: held-out log-likelihood gain of the split, BIC, WBIC and the log Bayes
    factor (thermodynamic integration). How often each criterion accepts the merge.
(C) The free-energy criterion: merge iff n * (likelihood cost of the merge) < (lambda_split - lambda_merged) log n,
    i.e. compare F_n = n S_n + lambda log n across strata -- not the RLCT alone (a merge always lowers lambda).
"""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
from amb_vigneaux import singular as S

rng = np.random.default_rng(0)
PRIOR = lambda th: float(-0.5 * (th / 3.0) @ (th / 3.0) - len(th) * 0.5 * np.log(2 * np.pi * 9.0))


def law(w, Q, d, V):
    X = S.cat_cells(d, V); lik = np.ones((len(w), len(X)))
    for j in range(d):
        lik *= Q[:, j, X[:, j]]
    return np.asarray(w) @ lik


def truth(K0, d, V, close=None):
    w = np.full(K0, 1.0 / K0); Q = rng.dirichlet(np.ones(V) * 0.7, size=(K0, d))
    if close is not None:                                      # make symbol K0-1 a small perturbation of symbol K0-2
        Q[K0 - 1] = (1 - close) * Q[K0 - 2] + close * rng.dirichlet(np.ones(V), size=d)
    return w, Q


def lam(counts, K, d, V, chains=4, steps=16000):
    em = S.cat_lca_em(counts, K, d, V, rng=rng)
    r = S.wbic_lambda_rep(S.cat_lca_loglik(counts, K, d, V), PRIOR, em["theta"], int(np.sum(counts)), chains=chains, steps=steps, burn=4000, scale=0.04, rng=rng)
    return dict(r, loglik=em["loglik"], dim=(K - 1) + K * d * (V - 1))


def part_a(n=3000):
    out = {}
    print("(A) learning coefficients by number of parent symbols K")
    for name, d, V, K0 in (("lone node (1 child on 16 cells)", 1, 16, 2), ("window (3 children on 4)", 3, 4, 2)):
        w, Q = truth(K0, d, V); c = rng.multinomial(n, law(w, Q, d, V)); rows = {}
        for K in (1, 2, 3, 4):
            r = lam(c, K, d, V); rows[K] = r
            print(f"    {name:32s} K={K}: d/2 {r['dim'] / 2:5.1f}  lambda_hat {r['lam']:5.2f} (MC se {r['lam_se']:.2f})  max loglik {r['loglik']:9.1f}")
        out[name] = {str(k): v for k, v in rows.items()}
    return out


def evidence(counts, K, d, V):
    em = S.cat_lca_em(counts, K, d, V, rng=rng)
    logZ, _, _ = S.log_evidence_ti(S.cat_lca_loglik(counts, K, d, V), PRIOR, em["theta"], betas=np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 16)]),
                                   steps=3000, burn=1000, scale=0.04, rng=rng)
    return em, logZ


def part_b(reps=4, d=3, V=4):
    out = {}
    print("\n(B) the merge decision: accept the merge (choose the smaller K)?")
    for case, K0, close in (("spurious split (truth K0=2, candidate K=3)", 2, None), ("weak real split (truth K0=3, two symbols close)", 3, 0.25)):
        out[case] = {}
        for n in (300, 1000, 3000, 10000):
            print(f'      n={n} ...', flush=True)
            acc = dict(heldout=0, bic=0, wbic=0, bayes=0); gains = []
            for _ in range(reps):
                w, Q = truth(K0, d, V, close); p = law(w, Q, d, V)
                tr = rng.multinomial(n, p); te = rng.multinomial(n, p)
                small, big = (2, 3)
                em_s, z_s = evidence(tr, small, d, V); em_b, z_b = evidence(tr, big, d, V)
                ho = float(te @ np.log(em_b["probs"]) - te @ np.log(em_s["probs"])); gains.append(ho / n)
                ds, db = (small - 1) + small * d * (V - 1), (big - 1) + big * d * (V - 1)
                bic_s = -em_s["loglik"] + ds / 2 * np.log(n); bic_b = -em_b["loglik"] + db / 2 * np.log(n)
                ws = S.wbic_lambda_rep(S.cat_lca_loglik(tr, small, d, V), PRIOR, em_s["theta"], n, chains=2, steps=10000, burn=2500, scale=0.04, rng=rng)["wbic"]
                wb = S.wbic_lambda_rep(S.cat_lca_loglik(tr, big, d, V), PRIOR, em_b["theta"], n, chains=2, steps=10000, burn=2500, scale=0.04, rng=rng)["wbic"]
                acc["heldout"] += ho <= 0; acc["bic"] += bic_s <= bic_b; acc["wbic"] += ws <= wb; acc["bayes"] += z_s >= z_b
            out[case][str(n)] = dict({k: v / reps for k, v in acc.items()}, heldout_gain_per_unit=float(np.mean(gains)))
            o = out[case][str(n)]
            print(f"    {case:48s} n={n:5d}: merge accepted by held-out LL {o['heldout']:.2f} | BIC {o['bic']:.2f} | WBIC {o['wbic']:.2f} | Bayes factor {o['bayes']:.2f}"
                  f"   (held-out gain of the split {o['heldout_gain_per_unit']:+.4f} nats/unit)")
    return out


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, grey, green = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#b8b6ae", "#1baf7a"
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7.5)
    A = OUT["A"]
    for (name, R), col in zip(A.items(), (orange, blue)):
        Ks = sorted(R, key=int)
        ax[0].plot([int(k) for k in Ks], [R[k]["dim"] / 2 for k in Ks], ls=":", color=col, lw=1.2)
        ax[0].errorbar([int(k) for k in Ks], [R[k]["lam"] for k in Ks], yerr=[1.96 * R[k]["lam_se"] for k in Ks], fmt="-o", color=col, capsize=3, lw=1.8, label=name)
    ax[0].set_xlabel("parent symbols K", fontsize=8, color=muted); ax[0].set_ylabel("λ̂ (solid) and d/2 (dotted)", fontsize=8, color=muted)
    ax[0].set_xticks([1, 2, 3, 4]); ax[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    ax[0].set_title("(a) λ̂ plateaus: extra symbols cost ≪ d/2", fontsize=8.5, loc="left", color=ink)
    B = OUT["B"]; crit = (("heldout", "held-out LL", grey), ("bic", "BIC", blue), ("wbic", "WBIC", green), ("bayes", "Bayes factor", orange))
    for a, (case, R) in zip(ax[1:], B.items()):
        ns = sorted(R, key=int)
        for k, lab, col in crit:
            a.plot([int(n) for n in ns], [R[n][k] for n in ns], "-o", color=col, lw=1.6, ms=4, label=lab)
        a.set_xscale("log"); a.set_ylim(-0.05, 1.05); a.set_xlabel("sample size n", fontsize=8, color=muted); a.set_ylabel("rate the merge is accepted", fontsize=8, color=muted)
        a.text(0.98, 0.04, "4 data sets per point", transform=a.transAxes, fontsize=6.5, color=muted, ha="right")
        a.set_title(("(b) " if "spurious" in case else "(c) ") + case.split(" (")[0] + "\n(" + case.split(" (")[1], fontsize=8, loc="left", color=ink)
    ax[1].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(here, "splitmerge_chart.pdf")); fig.savefig(os.path.join(here, "splitmerge_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    OUT = dict(A=part_a(), B=part_b())
    js = json.loads(json.dumps(OUT, default=float))
    json.dump(js, open(os.path.join(here, "results.json"), "w"), indent=1)
    plot(js)
