"""Planted benchmark for dynamic programming on the gluing lattice (amb_vigneaux/lattice_dp.py).

Instance: S = 50 sources x X topics (X = 50 .. 400), binomial counts (about 35 per cell), additive true logits
alpha_s + beta_x. 15% of topics are contaminated in CLUSTERS of 2-4 topics:
each cluster shares a source-interaction vector (a planted loop) on 20 sources, amplitude 0.8, and shifts the target
source by +2.0, so the full cover biases the target outcome. Clean truth = the uncontaminated topics.
Rules (what a solution must satisfy; nothing is optimised):
  local   size >= 60% of X; at least 5 topics of a 10-topic quota group; the target topic is in; 8 random exclusive
          pairs (not both in)
  global  the region glues (additive-model deviance p >= 0.05); the target outcome 'lean of source s* >= eta0'
          (alpha_s* - mean alpha) is stable (|lean_hat - eta0| / se >= 2), eta0 halfway between the truth and the
          full-cover estimate (either side counts: stability, not a desired value).
Methods (each proposes regions; a proposal counts if it passes every rule, checked exactly):
  DP-2nd   exact sampling from pi(U) ~ exp(-[D~ - 2 df]/2) with first + second-order jets, bags from the D'' graph (cap 8)
  DP-1st   the same with first-order jets only (independent topics)
  DP-2nd+rule  DP-2nd with the target-stability rule carried in the state through the outcome's own jets
  greedy   consistency pruning with randomised restarts: repeatedly drop one of the 3 topics with the largest loop
           residual (exact refit each step) until every rule holds (checked exactly each step), respecting the local rules
  random   uniform among regions meeting the local counting rules (the DP with zero score)
Also: accuracy of the jets (exact D(U) vs first / second-order prediction on random regions) and wall-clock scaling.
"""
import os, sys, json, time
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
from amb_vigneaux import lattice_dp as L

rng = np.random.default_rng(23)


def instance(X, S=50, frac=0.15, amp=0.8, amp_star=2.0, seed=0):
    r = np.random.default_rng(seed)
    a = r.normal(0, 0.5, S); b = r.normal(-0.5, 0.6, X); eta = a[:, None] + b[None, :]
    topics = r.permutation(X); n_cont = int(round(frac * X)); cont = []; clusters = []; i = 0
    s_star = 0
    while len(cont) < n_cont:
        size = int(r.integers(2, 5)); cl = list(topics[i:i + size]); i += size
        cl = cl[:n_cont - len(cont)]; cont += cl; clusters.append([int(c) for c in cl])
        srcs = r.choice(np.arange(1, S), 20, replace=False); g = np.zeros(S); g[srcs] = r.choice([-1, 1], 20) * amp; g[s_star] = amp_star
        for x in cl:
            eta[:, x] += g
    clean = [x for x in range(X) if x not in cont]
    n = r.poisson(30, (S, X)) + 5
    t_star = int(clean[0])
    p = 1 / (1 + np.exp(-eta)); k = r.binomial(n, p)
    y = np.log((k + 0.5) / (n - k + 0.5)); w = 1 / (1 / (k + 0.5) + 1 / (n - k + 0.5))
    true_eta = a[s_star] - a.mean()
    full_eta = L.target_stability(y, w, list(range(X)), s_star, t_star, 0.0)[1]
    eta0 = (true_eta + full_eta) / 2
    group = [int(x) for x in r.choice(clean[1:], 7, replace=False)] + [int(x) for x in r.choice(cont, 3, replace=False)]
    excl = []
    while len(excl) < 8:
        i, j = map(int, r.choice(X, 2, replace=False))
        if t_star not in (i, j) and i in clean and j in clean:
            excl.append((i, j))
    rules = L.Rules(k_min=int(0.6 * X), group=group, m_min=5, must=[t_star], exclusive=excl)
    return dict(y=y, w=w, cont=set(int(c) for c in cont), clusters=clusters, rules=rules, target=(s_star, t_star, eta0),
                true_eta=true_eta, full_eta=full_eta)


def jet_accuracy(I, J, reps=150):
    X = I["y"].shape[1]; ex, p1, p2 = [], [], []
    for _ in range(reps):
        m = int(rng.integers(1, int(0.3 * X))); rem = list(rng.choice(X, m, replace=False))
        U = [x for x in range(X) if x not in rem]
        ex.append(L.additive_fit(I["y"], I["w"], U)[2]); p1.append(L.jet_predict(J, rem, 1)); p2.append(L.jet_predict(J, rem, 2))
    ex, p1, p2 = map(np.array, (ex, p1, p2))
    r2 = lambda p: float(1 - np.sum((ex - p) ** 2) / np.sum((ex - ex.mean()) ** 2))
    return dict(r2_first=r2(p1), r2_second=r2(p2), mae_first=float(np.mean(np.abs(ex - p1))), mae_second=float(np.mean(np.abs(ex - p2))),
                exact=ex.tolist(), first=p1.tolist(), second=p2.tolist())


def greedy(I, budget_s, max_props=200):
    y, w, R = I["y"], I["w"], I["rules"]; X = y.shape[1]; G = set(R.group); out = []; t0 = time.time()
    while time.time() - t0 < budget_s and len(out) < max_props:
        U = list(range(X))
        for i, j in R.exclusive:                                       # resolve exclusions at random
            if i in U and j in U:
                U.remove(i if rng.random() < 0.5 else j)
        while True:
            if L.check(y, w, U, R, I["target"])["feasible"] or len(U) <= R.k_min:
                break
            alpha, beta, _, _ = L.additive_fit(y, w, U)
            res = (w[:, U] * (y[:, U] - alpha[:, None] - beta[None, :]) ** 2).sum(0)
            order = [U[i] for i in np.argsort(-res)]
            cand = [x for x in order if x not in R.must and not (x in G and sum(u in G for u in U) <= R.m_min)][:3]
            if not cand:
                break
            U.remove(cand[int(rng.integers(len(cand)))])
        out.append(sorted(U))
    return out, time.time() - t0


def evaluate(I, props, secs):
    res = [L.check(I["y"], I["w"], U, I["rules"], I["target"]) for U in props]
    feas = [U for U, r in zip(props, res) if r["feasible"]]
    uniq = {tuple(U) for U in feas}
    jac = []
    F = list(uniq)
    for i in range(min(len(F), 40)):
        for j in range(i + 1, min(len(F), 40)):
            a, b = set(F[i]), set(F[j]); jac.append(1 - len(a & b) / len(a | b))
    excl_cont = [len(I["cont"] - set(U)) / len(I["cont"]) for U in feas]
    kept_clean = [len(set(U) - I["cont"]) / (I["y"].shape[1] - len(I["cont"])) for U in feas]
    s_, t_, e0 = I["target"]; truth_side = I["true_eta"] >= e0
    correct = [(L.target_stability(I["y"], I["w"], U, s_, t_, e0)[1] >= e0) == truth_side for U in feas]
    return dict(proposals=len(props), feasible=len(feas), rate=len(feas) / max(len(props), 1), distinct=len(uniq),
                diversity=float(np.mean(jac)) if jac else 0.0, cont_excluded=float(np.mean(excl_cont)) if feas else np.nan,
                clean_kept=float(np.mean(kept_clean)) if feas else np.nan, correct_side=float(np.mean(correct)) if feas else np.nan, seconds=secs, per_feasible_s=secs / max(len(feas), 1),
                glues=float(np.mean([r["glues"] for r in res])), stable=float(np.mean([r["stable"] for r in res])))


def run_size(X, n_props=200, seed=0):
    I = instance(X, seed=seed); out = dict(X=X)
    t = time.time(); J = L.jets(I["y"], I["w"]); out["jets_s"] = time.time() - t
    out["jet_accuracy"] = jet_accuracy(I, J, reps=60 if X > 200 else 150)
    bags, tau = L.bags_from_graph(J["d2"], cap=8, forced_pairs=I["rules"].exclusive)
    cont_bagged = sum(1 for cl in I["clusters"] if any(set(cl) <= set(b) for b in bags))
    out["bags"] = dict(n=len(bags), max=max(len(b) for b in bags), clusters_in_one_bag=cont_bagged, clusters=len(I["clusters"]))
    meth = {}
    singles, _ = L.bags_from_graph(np.zeros((X, X)), cap=X, forced_pairs=I["rules"].exclusive, quantile=1.0)
    t = time.time(); LJ = L.lean_jets(I["y"], I["w"], bags, I["target"][0]); out["lean_jets_s"] = time.time() - t
    S = I["y"].shape[0]
    for name, kw in (("DP-2nd+rule", dict(bags=bags, order=2, lean=LJ, target=(I["target"][2], 2.0))),
                     ("DP-2nd", dict(bags=bags, order=2)), ("DP-1st", dict(bags=singles, order=1)),
                     ("random", dict(bags=singles, order=1, beta=0.0, c_pen=0.0))):
        t = time.time(); Smp = L.Sampler(X, kw.pop("bags"), J["d1"], J["d2"], I["rules"], S=S, **kw); build = time.time() - t
        props = [Smp.sample(rng) for _ in range(n_props)]; secs = time.time() - t
        meth[name] = dict(evaluate(I, props, secs), build_s=build)
    props, secs = greedy(I, budget_s=max(meth["DP-2nd+rule"]["seconds"], 5.0), max_props=n_props)
    meth["greedy"] = evaluate(I, props, secs)
    out["methods"] = meth
    if X == 200:                                                       # the temperature dial: feasibility vs diversity
        out["beta_sweep"] = {}
        for beta in (1.0, 0.3, 0.1, 0.03):
            t = time.time(); Smp = L.Sampler(X, bags, J["d1"], J["d2"], I["rules"], S=S, order=2, beta=beta)
            props = [Smp.sample(rng) for _ in range(n_props)]; e = evaluate(I, props, time.time() - t)
            out["beta_sweep"][str(beta)] = e
            print(f"   beta {beta:<5}: feasible {e['rate']:.2f}, distinct {e['distinct']}, diversity {e['diversity']:.3f}, contaminated excluded {e['cont_excluded']:.2f}", flush=True)
    out["full_cover"] = L.check(I["y"], I["w"], list(range(X)), I["rules"], I["target"])
    print(f"X={X}: jets {out['jets_s']:.1f}s; R2 of D(U): first {out['jet_accuracy']['r2_first']:.3f} second {out['jet_accuracy']['r2_second']:.3f};"
          f" bags {out['bags']['n']} (max {out['bags']['max']}; {cont_bagged}/{len(I['clusters'])} planted clusters inside one bag);"
          f" full cover glue p {out['full_cover']['glue_p']:.1e}, target r {out['full_cover']['r']:.2f}")
    for k, v in meth.items():
        print(f"   {k:7s} feasible {v['feasible']:3d}/{v['proposals']:3d} ({v['rate']:.2f}), distinct {v['distinct']:3d}, diversity {v['diversity']:.3f},"
              f" contaminated excluded {v['cont_excluded']:.2f}, clean kept {v['clean_kept']:.2f}, outcome correct {v['correct_side']:.2f}, glues {v['glues']:.2f}, stable {v['stable']:.2f},"
              f" {v['seconds']:.1f}s ({v['per_feasible_s']:.2f}s per feasible)", flush=True)
    return out


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, green, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#b8b6ae"
    col = {"DP-2nd+rule": "#7a3fc2", "DP-2nd": blue, "DP-1st": green, "greedy": orange, "random": grey}
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.3))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7)
    R = OUT["sizes"]; Xs = [r["X"] for r in R]
    A = [r for r in R if r["X"] == 200][0]["jet_accuracy"] if any(r["X"] == 200 for r in R) else R[-1]["jet_accuracy"]
    ax[0].scatter(A["exact"], A["first"], s=8, color=green, label=f"first order (R² {A['r2_first']:.2f})")
    ax[0].scatter(A["exact"], A["second"], s=8, color=blue, label=f"second order (R² {A['r2_second']:.2f})")
    mn, mx = min(A["exact"]), max(A["exact"]); ax[0].plot([mn, mx], [mn, mx], ":", color=ink)
    ax[0].set_xlabel("exact deviance D(U)", fontsize=7.5, color=muted); ax[0].set_ylabel("jet prediction", fontsize=7.5, color=muted)
    ax[0].legend(fontsize=6.3, frameon=False); ax[0].set_title("(a) jets predict region consistency (X = 200)", fontsize=8, loc="left")
    for m in ("DP-2nd+rule", "DP-2nd", "DP-1st", "greedy", "random"):
        ax[1].plot(Xs, [r["methods"][m]["rate"] for r in R], "-o", ms=4, color=col[m], label=m)
        ax[2].plot(Xs, [r["methods"][m]["distinct"] for r in R], "-o", ms=4, color=col[m], label=m)
        ax[3].plot(Xs, [r["methods"][m]["per_feasible_s"] if r["methods"][m]["feasible"] else np.nan for r in R], "-o", ms=4, color=col[m], label=m)
    ax[1].set_xlabel("topics X (S = 50 sources)", fontsize=7.5, color=muted); ax[1].set_ylabel("share of proposals passing every rule", fontsize=7.5, color=muted)
    ax[1].set_title("(b) feasibility", fontsize=8, loc="left"); ax[1].legend(fontsize=6.3, frameon=False); ax[1].set_ylim(-0.03, 1.03)
    ax[2].set_xlabel("topics X", fontsize=7.5, color=muted); ax[2].set_ylabel("distinct feasible regions (of 200 proposals)", fontsize=7.5, color=muted)
    ax[2].set_title("(c) many solutions, not one", fontsize=8, loc="left")
    ax[3].set_yscale("log"); ax[3].set_xlabel("topics X", fontsize=7.5, color=muted); ax[3].set_ylabel("seconds per feasible region", fontsize=7.5, color=muted)
    ax[3].set_title("(d) cost", fontsize=8, loc="left")
    B = [r for r in R if "beta_sweep" in r]
    if B:
        bs = B[0]["beta_sweep"]; xs = [float(k) for k in bs]
        ax[4].plot(xs, [bs[k]["rate"] for k in bs], "-o", ms=4, color=blue, label="share feasible")
        ax[4].plot(xs, [bs[k]["diversity"] for k in bs], "-s", ms=4, color=orange, label="diversity (Jaccard)")
        ax[4].plot(xs, [bs[k]["distinct"] / 200 for k in bs], "-^", ms=4, color=green, label="distinct / 200")
        ax[4].set_xscale("log"); ax[4].set_xlabel("β (inverse temperature), X = 200", fontsize=7.5, color=muted)
        ax[4].legend(fontsize=6.3, frameon=False); ax[4].set_title("(e) one dial: safe vs diverse", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(here, "lattice_dp_chart.pdf")); fig.savefig(os.path.join(here, "lattice_dp_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    sizes = [int(s) for s in os.environ.get("SIZES", "50,100,200,400").split(",")]
    OUT = dict(sizes=[run_size(X) for X in sizes])
    OUT = json.loads(json.dumps(OUT, default=lambda o: o.tolist() if hasattr(o, "tolist") else (list(o) if isinstance(o, set) else float(o))))
    json.dump(OUT, open(os.path.join(here, "results.json"), "w")); plot(OUT)
