"""The flat ridge as a diagnostic: dim ker R predicts the Fisher rank drop and the learning coefficient.

When only context margins are observed, the likelihood of a global law depends on it only through R p, so it is
constant along the fibre (P + ker R) inside the simplex: a flat ridge of dimension dim ker R. For a regular
parametrisation of the joint (K - 1 logits) this is a smooth redundancy, so
    Fisher rank = (K - 1) - dim ker R,     learning coefficient lambda = ((K - 1) - dim ker R) / 2   (Watanabe),
i.e. the RLCT deficit from d/2 is dim ker R / 2. Checked on five covers with an interior truth, margin samples of
3000 per context, lambda estimated by WBIC (Watanabe's two-temperature estimator, 4 chains).
"""
import os, sys, json, itertools
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
from amb_vigneaux.scenario import Scenario
from amb_vigneaux import bounds as Bd, singular as S

rng = np.random.default_rng(2)
COVERS = {
    "full joint (3 binary)": (3, [("A", "B", "C")]),
    "tree: AB, BC": (3, [("A", "B"), ("B", "C")]),
    "triangle: AB, BC, CA": (3, [("A", "B"), ("B", "C"), ("C", "A")]),
    "4-cycle: AB, BC, CD, DA": (4, [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]),
    "sphere: all triples of 4": (4, [c for c in itertools.combinations("ABCD", 3)]),
}


def setup(n, ctx):
    names = "ABCD"[:n]; sc = Scenario({m: (0, 1) for m in names}, ctx)
    R = [sc.restriction_matrix(sc.measurements, C) for C in sc.contexts]
    return sc, R


def run(n_per=3000, chains=4):
    out = {}
    for name, (n, ctx) in COVERS.items():
        sc, R = setup(n, ctx); K = 2 ** n; d = K - 1
        dimker = Bd.loop_space(sc).shape[1]
        p = rng.dirichlet(np.ones(K) * 3)
        counts = [rng.multinomial(n_per, Rc @ p) for Rc in R]
        def probs(th):
            e = np.concatenate([[0.0], th]); q = np.exp(e - e.max()); return q / q.sum()
        def ll(th):
            q = probs(th); return float(sum(c @ np.log(np.maximum(Rc @ q, 1e-300)) for c, Rc in zip(counts, R)))
        lp = lambda th: float(-0.5 * (th / 3) @ (th / 3))
        # Fisher information of the margin likelihood at the truth (in logit coordinates)
        q = p; J = np.diag(q) - np.outer(q, q); J = J[:, 1:]                        # dp/dtheta
        F = sum(n_per * (Rc @ J).T @ np.diag(1 / (Rc @ q)) @ (Rc @ J) for Rc in R)
        ev = np.linalg.eigvalsh(F); rank = int((ev > 1e-8 * ev.max()).sum())
        th0 = np.log(p[1:] / p[0])
        r = S.wbic_lambda_rep(ll, lp, th0, n_per * len(R), chains=chains, steps=20000, burn=5000, scale=0.02, rng=rng)
        out[name] = dict(d=d, dim_ker_R=dimker, fisher_rank=rank, lam_theory=(d - dimker) / 2, lam_hat=r["lam"], lam_se=r["lam_se"], d_half=d / 2,
                         contexts=len(R))
        o = out[name]
        print(f"  {name:26s} d {d:2d}  dim ker R {dimker}  Fisher rank {rank:2d}  lambda: theory {o['lam_theory']:4.1f}  estimated {o['lam_hat']:5.2f} ± {o['lam_se']:.2f}  (d/2 = {d / 2})")
    return out


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#b8b6ae"
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    names = list(OUT); x = np.arange(len(names))
    ax.bar(x - 0.2, [OUT[n]["d_half"] for n in names], 0.4, color=grey, label="d/2 (no ridge)")
    ax.bar(x + 0.2, [OUT[n]["lam_theory"] for n in names], 0.4, color=blue, label="(d − dim ker R)/2")
    ax.errorbar(x + 0.2, [OUT[n]["lam_hat"] for n in names], yerr=[1.96 * OUT[n]["lam_se"] for n in names], fmt="o", color=orange, ms=4, capsize=3, label="λ̂ (WBIC)")
    ax.set_xticks(x); ax.set_xticklabels([n.split(":")[0].split(" (")[0] for n in names], fontsize=7)
    ax.set_ylabel("learning coefficient", fontsize=8, color=muted); ax.tick_params(colors=muted, labelsize=7)
    ax.legend(fontsize=6.5, frameon=False); ax.set_title("The ridge lowers λ by dim ker R / 2", fontsize=8.5, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(here, "ridge_chart.pdf")); fig.savefig(os.path.join(here, "ridge_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    OUT = run(); json.dump(OUT, open(os.path.join(here, "results.json"), "w"), indent=1); plot(OUT)
