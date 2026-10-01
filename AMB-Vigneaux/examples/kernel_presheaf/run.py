"""Component [L]: singular statistics and optimal learning -- the net difference they make to outcomes.

Each part compares the regular / plug-in answer with the singular-aware / Bayes answer on the same data.
(A) Calibration of the learning-coefficient estimator on the shift mixture (1-a) N(0,1) + a N(b,1), n = 2000:
    singular truth (a b = 0; theory lambda = 1/2, multiplicity 2) vs regular truth (a = 0.3, b = 1; lambda = d/2 = 1).
(B) [C3] hidden sponsorship among nonpartisan 2016 polls, single bent component with free shift b: lambda_hat, WBIC
    of 'mixture' vs 'none', the regular BIC decision vs the singular decision, MLE vs posterior mean of a.
(C) The flop: plug-in forced pair (argmin, reported with certainty) vs the Dirichlet posterior probability of that pair,
    and the posterior interval of the toggle radius (MBIC 3, AllSides 3, MBIC 4).
(D) Latent class models K = 1..4 on the MBIC four-group table (secant varieties; singular): BIC vs WBIC choice of K,
    lambda_hat vs d/2, and held-out generalisation (20 random halves): plug-in ML vs Bayes predictive (posterior
    average), and the predicted P(all four biased) under each choice against the observed value.
Env: POLLS2016_CSV, MBIC_XLSX, ALLSIDES_DIR.
"""
import os, sys, json, io, contextlib
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../toric"), os.path.join(here, "../curvature")]
import numpy as np
from amb_vigneaux import singular as S, deform as D, toric as T
from amb_vigneaux.curvature import mixture_lr

rng = np.random.default_rng(0)


def _mod(name, sub):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, "..", sub, "run.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def part_a(reps=10):
    out = {}
    print("(A) calibration of lambda_hat (n = 2000, 10 data sets each)")
    for name, a, b, theory in (("singular", 0.0, 0.0, 0.5), ("regular", 0.3, 1.0, 1.0)):
        lams = []
        for _ in range(reps):
            n = 2000; z = rng.random(n) < a; r = rng.normal(0, 1, n) + b * z
            lams.append(S.wbic_lambda(S.mix_loglik(r, np.ones(n)), S.mix_logprior(2.0), [0.1, 0.5], n, steps=16000, burn=4000, rng=rng)["lam"])
        out[name] = dict(theory=theory, lam=lams, mean=float(np.mean(lams)), sd=float(np.std(lams)))
        print(f"    {name:9s}: lambda_hat {np.mean(lams):.2f} +/- {np.std(lams):.2f} (theory {theory}; regular d/2 = 1)")
    return out


def part_b():
    CV = _mod('cv_run', 'curvature')
    d = CV.load(); tau2 = CV.tau2_mom(d); b, X, w = CV.fit_shift(d, tau2)
    base = X.drop(columns=["D", "R"]).to_numpy() @ b.drop(["D", "R"]).to_numpy()
    m = (d.D == 0).to_numpy() & (d.R == 0).to_numpy(); r = (d["y"].to_numpy() - base)[m]; s2 = 1.0 / w[m]; n = int(m.sum())
    ll = S.mix_loglik(r, s2); ll0 = ll([0.0, 0.0])
    best = max(((mixture_lr(r, s2, [bb])[1], bb) for bb in np.linspace(-0.3, 0.3, 121) if abs(bb) > 1e-9), key=lambda t: t[0])
    lr, bhat = best; pi_hat = mixture_lr(r, s2, [bhat])[0][0]
    grid = [bb for bb in np.linspace(-0.3, 0.3, 61) if abs(bb) > 1e-9]; null = []
    for _ in range(200):                                  # max-over-b LR under a = 0: the Davies-type singular null
        rs = rng.normal(0, np.sqrt(s2)); null.append(max(mixture_lr(rs, s2, [bb], iters=100)[1] for bb in grid))
    null = np.array(null); p_boot = float((null >= lr).mean())
    wl = S.wbic_lambda_rep(ll, S.mix_logprior(0.2), [0.05, 0.05], n, chains=6, steps=16000, burn=4000, scale=0.02, rng=rng)
    post = []
    x = np.array([0.05, 0.05]); s = 0.02; cur = ll(x) + S.mix_logprior(0.2)(x)
    for t in range(16000):
        y = x + s * rng.normal(size=2); lp = S.mix_logprior(0.2)(y)
        if np.isfinite(lp):
            c = ll(y) + lp
            if np.log(rng.random()) < c - cur:
                x, cur = y, c
        if t >= 4000:
            post.append(x.copy())
    post = np.array(post)
    # exact(-ish) Bayes factor by thermodynamic integration (proper prior: a ~ U(0,1), b ~ N(0, 0.2^2))
    lp_norm = lambda th: S.mix_logprior(0.2)(th) - 0.5 * np.log(2 * np.pi * 0.04)
    logZ, _, _ = S.log_evidence_ti(ll, lp_norm, [0.01, -0.1], steps=6000, burn=1500, scale=0.02, rng=rng)
    log_bf = logZ - ll0
    bic_mix = -(ll0 + lr / 2) + 1.0 * np.log(n); bic0 = -ll0                      # d = 2 -> (d/2) log n
    sing_mix = -(ll0 + lr / 2) + 0.5 * np.log(n) - (2 - 1) * np.log(np.log(n))    # lambda = 1/2, m = 2
    out = dict(n=n, lr=float(lr), p_boot=p_boot, log_bf=float(log_bf), decision_bayes="mixture" if log_bf > 0 else "none", null_q95=float(np.quantile(null, .95)), chi2_2_q95=5.99, b_hat=float(bhat), a_hat=float(pi_hat), lambda_hat=wl["lam"], lambda_se=wl["lam_se"], wbic_mix=wl["wbic"], wbic_mix_se=wl["wbic_se"], wbic_null=float(-ll0),
               decision_bic="mixture" if bic_mix < bic0 else "none", decision_singular="mixture" if sing_mix < bic0 else "none",
               decision_wbic="mixture" if wl["wbic"] < -ll0 else "none",
               post_mean_a=float(post[:, 0].mean()), post_a_q=[float(np.quantile(post[:, 0], .025)), float(np.quantile(post[:, 0], .975))],
               post_mean_ab=float((post[:, 0] * post[:, 1]).mean()), mle_ab=float(pi_hat * bhat))
    print(f"\n(B) hidden sponsorship ({n} nonpartisan polls; single bent component, shift free)")
    print(f"    MLE a {pi_hat:.3f}, b {bhat:+.3f} (a b {pi_hat * bhat:+.4f}); LR {lr:.2f}; lambda_hat {wl['lam']:.2f} (regular d/2 = 1)")
    print(f"    bootstrap null of max-over-b LR: 95% point {out['null_q95']:.2f} (chi2_2: 5.99); p = {p_boot:.3f}")
    print(f"    decision: regular BIC -> {out['decision_bic']}; singular BIC (lambda 1/2, m 2) -> {out['decision_singular']}; WBIC -> {out['decision_wbic']}"
          f" (WBIC {wl['wbic']:.1f} +/- {wl['wbic_se']:.1f} MC vs none {-ll0:.1f})")
    print(f"    thermodynamic-integration log Bayes factor (mixture vs none) {log_bf:+.2f} -> {out['decision_bayes']}")
    print(f"    posterior mean a {out['post_mean_a']:.3f} (95% [{out['post_a_q'][0]:.3f}, {out['post_a_q'][1]:.3f}]); posterior mean a b {out['post_mean_ab']:+.4f} vs MLE {out['mle_ab']:+.4f}")
    return out


def part_c():
    TR = _mod('tr_run', 'toric')
    m3, m4 = TR.mbic_tables(); tables = {"MBIC 3 groups": m3, "AllSides 3 sides": TR.allsides_table(), "MBIC 4 groups": m4}
    out = {}
    print("\n(C) the flop: plug-in forced pair vs its posterior probability")
    for name, Tm in tables.items():
        c = TR.counts_from(Tm); n = int(np.log2(len(c))); ev, od = D.parity_classes(n); p = (c + .5) / (c + .5).sum()
        plug = (int(min(ev, key=lambda i: p[i])), int(min(od, key=lambda i: p[i])))
        fp = S.flop_posterior(c, ev, od, rng=rng); pr = fp["pairs"].get(plug, 0.0)
        rad = D.radii(p)["toggle_radius"]
        out[name] = dict(plugin_pair=[T.cells(n)[plug[0]], T.cells(n)[plug[1]]], plugin_certainty=1.0, posterior_prob=pr,
                         n_pairs_with_mass=int(sum(v > 0.05 for v in fp["pairs"].values())), radius_plugin=rad,
                         radius_post=[fp["radius_lo"], fp["radius_hi"]], radius_post_mean=fp["radius_mean"])
        print(f"    {name}: plug-in pair {out[name]['plugin_pair']} -> posterior probability {pr:.2f}; pairs with > 5% mass {out[name]['n_pairs_with_mass']};"
              f" radius plug-in {rad:.3f}, posterior mean {fp['radius_mean']:.3f} [{fp['radius_lo']:.3f}, {fp['radius_hi']:.3f}]")
    return out


def bayes_predictive(counts, K, d, steps=8000, burn=2000, thin=20):
    em = S.lca_em(counts, K, d, starts=8, rng=rng)
    ll = S.lca_loglik(counts, K, d); x = em["theta"].copy(); cur = ll(x) + S.lca_logprior(x); s = 0.05; acc = 0; P = []
    for t in range(steps):
        y = x + s * rng.normal(size=x.shape); c = ll(y) + S.lca_logprior(y)
        if np.log(rng.random()) < c - cur:
            x, cur = y, c; acc += 1
        if t < burn and t % 200 == 199:
            s *= 1.25 if acc / 200 > 0.3 else 0.8; acc = 0
        if t >= burn and t % thin == 0:
            P.append(S.lca_probs(x, K, d))
    return em["probs"], np.mean(P, 0)


def part_d(reps=40):
    TR = _mod('tr_run', 'toric')
    _, m4 = TR.mbic_tables(); c = TR.counts_from(m4); N = int(c.sum()); d = 4; obs_all = float(c[-1] / N)
    print(f"\n(D) latent class models on the MBIC four-group table ({N} sentences)")
    sel = {}
    for K in (1, 2, 3, 4):
        dim = (K - 1) + K * d; em = S.lca_em(c, K, d, rng=rng)
        bic = -em["loglik"] + dim / 2 * np.log(N)
        wl = S.wbic_lambda_rep(S.lca_loglik(c, K, d), S.lca_logprior, em["theta"], N, chains=6, steps=20000, burn=5000, scale=0.05, rng=rng)
        sel[K] = dict(dim=dim, loglik=em["loglik"], bic=float(bic), wbic=wl["wbic"], wbic_se=wl["wbic_se"], lam=wl["lam"], lam_se=wl["lam_se"], p_all=float(em["probs"][-1]))
        print(f"    K={K}: d {dim:2d}, d/2 {dim / 2:4.1f}, lambda_hat {wl['lam']:5.2f} (MC se {wl['lam_se']:.2f}) | BIC {bic:8.1f} | WBIC {wl['wbic']:8.1f} (MC se {wl['wbic_se']:.2f})"
              f" | ML P(all four) {em['probs'][-1]:.3f}")
    kb = min(sel, key=lambda k: sel[k]["bic"]); kw = min(sel, key=lambda k: sel[k]["wbic"])
    print(f"    BIC picks K={kb}, WBIC picks K={kw}; observed P(all four) {obs_all:.3f}")
    # held-out generalisation: ML plug-in vs Bayes predictive
    units = np.repeat(np.arange(16), c.astype(int)); gen = {K: dict(ml=[], bayes=[]) for K in (2, 3)}
    for _ in range(reps):
        idx = rng.permutation(len(units)); tr = np.bincount(units[idx[: len(units) // 2]], minlength=16); te = np.bincount(units[idx[len(units) // 2:]], minlength=16)
        for K in (2, 3):
            pml, pb = bayes_predictive(tr, K, d)
            gen[K]["ml"].append(float(-(te @ np.log(pml)) / te.sum())); gen[K]["bayes"].append(float(-(te @ np.log(pb)) / te.sum()))
    g = {K: dict(ml=float(np.mean(v["ml"])), bayes=float(np.mean(v["bayes"])), diff=float(np.mean(np.array(v["bayes"]) - np.array(v["ml"]))),
                 diff_se=float(np.std(np.array(v["bayes"]) - np.array(v["ml"])) / np.sqrt(reps)), bayes_better=float(np.mean(np.array(v["bayes"]) < np.array(v["ml"]))))
         for K, v in gen.items()}
    for K, v in g.items():
        print(f"    held-out log-loss per sentence, K={K}: ML plug-in {v['ml']:.4f}, Bayes predictive {v['bayes']:.4f};"
              f" Bayes - ML {v['diff']:+.5f} (se {v['diff_se']:.5f}); Bayes better in {v['bayes_better']:.0%} of splits")
    return dict(models={str(k): v for k, v in sel.items()}, k_bic=kb, k_wbic=kw, observed_all=obs_all, generalisation={str(k): v for k, v in g.items()})


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#b8b6ae"
    fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.4), gridspec_kw=dict(width_ratios=[1, 1.05, 1.1, 1.05]))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7.5)
    A = OUT["A"]
    for i, (k, col) in enumerate((("singular", orange), ("regular", blue))):
        ax[0].scatter(np.full(len(A[k]["lam"]), i) + rng.normal(0, 0.04, len(A[k]["lam"])), A[k]["lam"], color=col, s=14, alpha=0.7)
        ax[0].hlines(A[k]["theory"], i - 0.3, i + 0.3, color=col, lw=2)
    ax[0].scatter([2], [OUT["B"]["lambda_hat"]], color=ink, s=26, marker="D")
    ax[0].set_xticks([0, 1, 2]); ax[0].set_xticklabels(["synthetic\nsingular", "synthetic\nregular", "polls\n[C3]"], fontsize=7)
    ax[0].axhline(1.0, color=muted, lw=0.6, ls=":"); ax[0].set_ylabel("learning coefficient λ̂", fontsize=8, color=muted)
    ax[0].set_title("(a) λ: singular ½ vs regular d/2", fontsize=8.5, loc="left", color=ink)
    C = OUT["C"]; names = list(C); y = np.arange(len(names))[::-1]
    ax[1].barh(y + 0.18, [1.0] * len(names), height=0.34, color=grey, label="plug-in (argmin)")
    ax[1].barh(y - 0.18, [C[k]["posterior_prob"] for k in names], height=0.34, color=orange, label="posterior")
    ax[1].set_yticks(y); ax[1].set_yticklabels(names, fontsize=7); ax[1].set_xlim(0, 1.05); ax[1].set_xlabel("certainty of the forced pair", fontsize=8, color=muted)
    ax[1].legend(fontsize=6.5, frameon=False, loc="lower right"); ax[1].set_title("(b) the flop: argmin vs posterior", fontsize=8.5, loc="left", color=ink)
    M = OUT["D"]["models"]; Ks = sorted(M, key=int)
    ax[2].plot([int(k) for k in Ks], [M[k]["dim"] / 2 for k in Ks], "-o", color=grey, lw=1.5, label="regular d/2")
    ax[2].errorbar([int(k) for k in Ks], [M[k]["lam"] for k in Ks], yerr=[1.96 * M[k].get("lam_se", 0) for k in Ks], fmt="-o", color=orange, lw=1.8, capsize=3, label="λ̂ (6 chains, 95% MC)")
    ax[2].set_xlabel("latent classes K", fontsize=8, color=muted); ax[2].set_ylabel("penalty per log n", fontsize=8, color=muted)
    ax[2].set_title("(c) latent classes, MBIC 4: λ̂ plateaus", fontsize=8.5, loc="left", color=ink)
    ax[2].legend(fontsize=6.5, frameon=False, loc="upper left"); ax[2].set_xticks([int(k) for k in Ks])
    G = OUT["D"]["generalisation"]; ks = sorted(G, key=int)
    ax[3].bar([int(k) for k in ks], [1e3 * G[k]["diff"] for k in ks], yerr=[1.96e3 * G[k]["diff_se"] for k in ks], color=[orange if G[k]["diff"] < 0 else blue for k in ks], width=0.5, capsize=3)
    ax[3].axhline(0, color=muted, lw=0.8); ax[3].set_xticks([int(k) for k in ks]); ax[3].set_xlabel("latent classes K", fontsize=8, color=muted)
    ax[3].set_ylabel("held-out log-loss, Bayes − ML (×10⁻³)", fontsize=8, color=muted)
    ax[3].set_title("(d) Bayes vs ML, held out", fontsize=8.5, loc="left", color=ink)
    fig.tight_layout(); fig.savefig(os.path.join(here, "singular_chart.pdf")); fig.savefig(os.path.join(here, "singular_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    OUT = dict(A=part_a(), B=part_b(), C=part_c(), D=part_d())
    js = json.loads(json.dumps(OUT, default=lambda o: o.tolist() if hasattr(o, "tolist") else (float(o) if isinstance(o, (np.floating,)) else str(o))))
    json.dump(js, open(os.path.join(here, "results.json"), "w"), indent=1)
    plot(js)
