"""Reward loop: policy / value iteration on geometric beliefs under scheme rules (amb_vigneaux/reward_loop.py).

S = 50 sources, X = 60 topics, T = 100 periods; contamination chain q01 = 0.02, q10 = 0.10 (about 17% of topics looped
at any time); loop amplitude 0.6 on 20 sources, target-source lean shift +1.0 per looped topic.
Reward: +1 per clean topic kept, -5 per looped topic kept, -0.5 per switch, -20 per violated rule (glue; lean right).
Policies (each tuned on 5 training streams, evaluated on 10 test streams; identical data across policies):
  oracle         keeps exactly the clean topics (knows z) -- a reference, not achievable
  all-in         keeps every topic
  myopic-test    keeps a topic iff its loop statistic passes a chi2 test this period (level tuned)
  myopic-Bayes   keeps iff the filtered belief is below a threshold (tuned): probability, no lookahead
  value-iter     the belief-MDP policy from value iteration, price mu per looped topic kept (tuned); policy iteration
                 is run as a check (same policy)
The loop noncentrality lam is estimated by EM from a burn-in stream (no access to z).
"""
import os, sys, json, time
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
from scipy import stats
from amb_vigneaux import reward_loop as RL

V, C, KAPPA, PEN, GAMMA = 1.0, 5.0, 0.5, 20.0, 0.95
AMP = 0.6
GRID = np.linspace(0, 1, 101)


def cache(seed, T=100):
    st = RL.Stream(T=T, seed=seed, amp=AMP, amp_star=AMP * 5 / 3); data = []
    for t in range(st.T):
        y, w = st.period(t); data.append((y, w, RL.loop_stats(y, w)))
    return st, data


def simulate(st, data, decide, df, lam):
    X = st.X; b = np.full(X, st.q01 / (st.q01 + st.q10)); kept = np.ones(X, int); tot = dict(reward=0.0, glue_fail=0, lean_fail=0,
                                                                                               switches=0, looped_kept=0, clean_kept=0, clean=0, looped=0)
    for t, (y, w, Q) in enumerate(data):
        b = RL.update(RL.predict(b, st.q01, st.q10), Q, df, lam)
        new = decide(t, Q, b, kept).astype(int)
        if new.sum() < 2:
            new[np.argsort(b)[:2]] = 1
        U = [int(x) for x in np.where(new == 1)[0]]
        z = st.z[t]; sw = int(np.abs(new - kept).sum())
        g, l = RL.rules_ok(y, w, U, st.true_lean)
        r = V * ((new == 1) & (z == 0)).sum() - C * ((new == 1) & (z == 1)).sum() - KAPPA * sw - PEN * (not g) - PEN * (not l)
        tot["reward"] += r; tot["glue_fail"] += (not g); tot["lean_fail"] += (not l); tot["switches"] += sw
        tot["looped_kept"] += int(((new == 1) & (z == 1)).sum()); tot["clean_kept"] += int(((new == 1) & (z == 0)).sum())
        tot["clean"] += int((z == 0).sum()); tot["looped"] += int((z == 1).sum()); kept = new
    T = len(data)
    return dict(reward_per_period=tot["reward"] / T, glue_ok=1 - tot["glue_fail"] / T, lean_ok=1 - tot["lean_fail"] / T,
                switches_per_period=tot["switches"] / T, looped_kept=tot["looped_kept"] / max(tot["looped"], 1), clean_kept=tot["clean_kept"] / tot["clean"])


def make_policies(df, lam, q01, q10):
    K = RL.belief_kernel(GRID, df, lam, q01, q10)
    pols = {"oracle": [None], "all-in": [None], "myopic-test": [0.9, 0.95, 0.99, 0.999], "myopic-Bayes": [0.1, 0.2, 0.3, 0.5, 0.7],
            "value-iter": [0.0, 2.0, 5.0, 10.0, 20.0]}
    tables = {}; checks = {}
    for mu in pols["value-iter"]:
        R = RL.rewards(GRID, V, C, KAPPA, mu)
        t0 = time.time(); pv, _, itv = RL.value_iteration(K, R, GAMMA); tv = time.time() - t0
        t0 = time.time(); pp, _, itp = RL.policy_iteration(K, R, GAMMA); tp = time.time() - t0
        tables[mu] = pv; checks[mu] = dict(vi_iters=itv, pi_iters=itp, same=bool(np.array_equal(pv, pp)), vi_s=tv, pi_s=tp,
                                           keep_threshold_if_kept=float(GRID[np.argmin(pv[:, 1])] if (pv[:, 1] == 0).any() else 1.0),
                                           keep_threshold_if_dropped=float(GRID[np.argmin(pv[:, 0])] if (pv[:, 0] == 0).any() else 1.0))
    def decide(name, par, st):
        if name == "oracle":
            return lambda t, Q, b, kept: (st.z[t] == 0)
        if name == "all-in":
            return lambda t, Q, b, kept: np.ones_like(kept)
        if name == "myopic-test":
            cut = stats.chi2.ppf(par, df); return lambda t, Q, b, kept: Q < cut
        if name == "myopic-Bayes":
            return lambda t, Q, b, kept: b < par
        tab = tables[par]
        return lambda t, Q, b, kept: tab[np.clip(np.round(b * (len(GRID) - 1)).astype(int), 0, len(GRID) - 1), kept]
    return pols, decide, checks


def main(tag=""):
    S = 50; df = S - 1
    burn_st, burn = cache(999, T=30)
    lam, pi1 = RL.em_lambda(np.concatenate([q for _, _, q in burn]), df)
    lam_true = float(np.mean([np.sum(burn[0][1][:, x] * (burn_st.g[x] - (burn[0][1][:, x] * burn_st.g[x]).sum() / burn[0][1][:, x].sum()) ** 2) for x in range(burn_st.X)]))
    print(f"EM from burn-in: lam_hat {lam:.1f} (Fisher norm^2 of the planted loops ~ {lam_true:.1f}), share looped {pi1:.2f}")
    pols, decide, checks = make_policies(df, lam, burn_st.q01, burn_st.q10)
    for mu, c in checks.items():
        print(f"   mu {mu:5.1f}: value iteration {c['vi_iters']} sweeps ({c['vi_s'] * 1e3:.0f} ms), policy iteration {c['pi_iters']} steps ({c['pi_s'] * 1e3:.0f} ms),"
              f" same policy {c['same']}; keep while belief < {c['keep_threshold_if_kept']:.2f} if kept, re-admit below {c['keep_threshold_if_dropped']:.2f}")
    train = [cache(100 + i) for i in range(5)]; test = [cache(200 + i) for i in range(10)]
    OUT = dict(lam_hat=lam, lam_planted=lam_true, checks={str(k): v for k, v in checks.items()}, tuning={}, test={})
    # the same belief policies with the planted noncentrality (a 'geometry oracle'): isolates the evidence estimate
    pols_o, decide_o, _ = make_policies(df, lam_true, burn_st.q01, burn_st.q10)
    runs = [(n, p, decide, lam) for n, p in pols.items()] + [(n + "*", pols_o[n], decide_o, lam_true) for n in ("myopic-Bayes", "value-iter")]
    for name, pars, dec, lm in runs:
        base = name.rstrip("*")
        best = None
        for par in pars:
            m = np.mean([simulate(st, d, dec(base, par, st), df, lm)["reward_per_period"] for st, d in train])
            OUT["tuning"].setdefault(name, {})[str(par)] = float(m)
            if best is None or m > best[0]:
                best = (m, par)
        res = [simulate(st, d, dec(base, best[1], st), df, lm) for st, d in test]
        agg = {k: float(np.mean([r[k] for r in res])) for k in res[0]}; agg["reward_sd"] = float(np.std([r["reward_per_period"] for r in res]) / np.sqrt(len(res)))
        agg["param"] = best[1]; OUT["test"][name] = agg
        print(f"{name:13s} (param {best[1]}): reward/period {agg['reward_per_period']:7.2f} ± {agg['reward_sd']:.2f} | glue ok {agg['glue_ok']:.2f}"
              f" lean ok {agg['lean_ok']:.2f} | looped kept {agg['looped_kept']:.2f} clean kept {agg['clean_kept']:.2f} | switches/period {agg['switches_per_period']:.1f}", flush=True)
    return OUT


def plot(ALL):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, green, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#b8b6ae"
    col = {"oracle": ink, "all-in": grey, "myopic-test": orange, "myopic-Bayes": green, "value-iter": blue}
    keys = list(ALL); fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7)
    x = np.arange(len(keys)); w = 0.2
    w = 0.16
    col.update({"myopic-Bayes*": "#8fd9b6", "value-iter*": "#9cc3ee"})
    for j, n in enumerate(("myopic-test", "myopic-Bayes", "value-iter", "value-iter*", "oracle")):
        ax[0].bar(x + (j - 2) * w, [ALL[k]["test"][n]["reward_per_period"] for k in keys], w, yerr=[1.96 * ALL[k]["test"][n]["reward_sd"] for k in keys],
                  color=col[n], label=n, capsize=2)
    ax[0].set_xticks(x); ax[0].set_xticklabels([k.replace(", ", "\n") for k in keys], fontsize=6.5); ax[0].legend(fontsize=6.3, frameon=False, ncol=2)
    ax[0].set_ylabel("reward per period (test streams)", fontsize=7.5, color=muted); ax[0].set_title("(a) reward by regime (* = planted noncentrality)", fontsize=8.5, loc="left")
    gap = lambda k, a, b: ALL[k]["test"][a]["reward_per_period"] - ALL[k]["test"][b]["reward_per_period"]
    ax[1].bar(x - 0.27, [gap(k, "myopic-Bayes", "myopic-test") for k in keys], 0.27, color=green, label="probability: Bayes filter − per-period test")
    ax[1].bar(x, [gap(k, "value-iter", "myopic-Bayes") for k in keys], 0.27, color=blue, label="lookahead: value iteration − Bayes threshold")
    ax[1].bar(x + 0.27, [gap(k, "value-iter*", "value-iter") for k in keys], 0.27, color="#9cc3ee", label="geometry: planted − estimated noncentrality")
    ax[1].axhline(0, color=ink, lw=0.6); ax[1].set_xticks(x); ax[1].set_xticklabels([k.replace(", ", "\n") for k in keys], fontsize=6.5)
    ax[1].legend(fontsize=6.3, frameon=False); ax[1].set_ylabel("gain in reward per period", fontsize=7.5, color=muted)
    ax[1].set_title("(b) what each layer adds", fontsize=8.5, loc="left")
    for j, n in enumerate(("myopic-test", "myopic-Bayes", "value-iter", "value-iter*", "oracle")):
        ax[2].bar(x + (j - 2) * w, [ALL[k]["test"][n]["switches_per_period"] for k in keys], w, color=col[n], label=n)
    ax[2].set_xticks(x); ax[2].set_xticklabels([k.replace(", ", "\n") for k in keys], fontsize=6.5); ax[2].set_ylabel("topics switched per period", fontsize=7.5, color=muted)
    ax[2].set_title("(c) churn", fontsize=8.5, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(here, "reward_loop_chart.pdf")); fig.savefig(os.path.join(here, "reward_loop_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    ALL = {}
    for amp, kappa in ((0.6, 0.5), (0.6, 2.0), (0.35, 0.5), (0.35, 2.0), (0.25, 0.5), (0.25, 2.0)):
        globals()["AMP"], globals()["KAPPA"] = amp, kappa
        print(f"\n=== loop amplitude {amp}, switching cost {kappa}")
        ALL[f"amp {amp}, kappa {kappa}"] = main()
    json.dump(json.loads(json.dumps(ALL, default=float)), open(os.path.join(here, "results.json"), "w"), indent=1)
    plot(json.load(open(os.path.join(here, "results.json"))))
