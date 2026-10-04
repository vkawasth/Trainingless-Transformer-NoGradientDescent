"""Planted checks for second-order information geometry [G2]: every quantity is run where the answer is known.

(a) Paths on 16 cells (9 bins on s in [-3, 3]). shift: eta(s) = eta0 + s c (e-geodesic); pull: p(s) = (1 - w) q0 + w q1, w = sigmoid(s)
    (m-geodesic); great circle on the Fisher sphere (FR geodesic); bent: eta0 + s c + s^2 d. Exact curvatures from
    finite-difference jets, and the same curvatures ESTIMATED from multinomial counts (n = 2e4, 2e5 per bin), bootstrap bias-corrected by local_jets.
(b) Holonomy on a 6 x 6 grid: flat transport eta_uv = beta + a_u + b_v vs a planted twist + lam u v h. Debiased
    information lost (nats per unit) and its z-score; n per grid cell 10000 (as in the jet grids).
(c) Efron-Hinkley: n Var(J / nI) at the MLE vs gamma2_e, for the e-flat family (d = 0) and bent families.
(d) Nodes: a table at a node with the two odd partners equal (branch posterior ~ 1/2) and one far from it (~ 1).
(e) Top-order information: debiased D_top under no 4-way interaction (~ 0) and with a planted theta.
"""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
from amb_vigneaux import infogeo2 as G, toric as T

rng = np.random.default_rng(7)
K = 16


def exact_jets(fn, s, h=1e-3):
    """natural jets of a path given as s -> p(s), by central differences on log p"""
    L = lambda t: np.log(fn(t))
    return fn(s), (L(s + h) - L(s - h)) / (2 * h), (L(s + h) - 2 * L(s) + L(s - h)) / h ** 2


def paths():
    eta0 = rng.normal(0, 0.8, K); c = rng.normal(0, 0.6, K); d = rng.normal(0, 0.4, K)
    q0, q1 = G.softmax(eta0), G.softmax(rng.normal(0, 0.8, K))
    P = {"shift (e-geodesic)": lambda s: G.softmax(eta0 + c * s),
         "pull (m-geodesic)": lambda s: (1 - 1 / (1 + np.exp(-s))) * q0 + 1 / (1 + np.exp(-s)) * q1,
         "great circle (FR geodesic)": lambda s: G.geodesic(q0, q1, "fr", 1 / (1 + np.exp(-s))),
         "bent (eta quadratic)": lambda s: G.softmax(eta0 + c * s + d * s * s)}
    out = {}
    s_grid = np.linspace(-3.0, 3.0, 9)
    for name, fn in P.items():
        ex = G.curvatures(*exact_jets(fn, 0.0))
        est = {}
        for n in (20000, 200000):
            tab = np.array([rng.multinomial(n, fn(s)) for s in s_grid])
            r = G.local_curvatures(tab, s_grid, 4, half=2, B=60, rng=rng)
            est[n] = {k: r[k] for k in r if k.startswith(("gamma2", "kappa2"))}
        out[name] = dict(exact=ex, est=est)
        print(f"  {name:28s} exact: e {ex['gamma2_e']:.3f}  m {ex['gamma2_m']:.3f}  FR {ex['kappa2_fr']:.3f}  | n=200000: " +
              "  ".join(f"{lab} {est[200000][k]:.3f}±{est[200000][k+'_sd']:.3f} (raw {est[200000][k+'_raw']:.3f})" for lab, k in (("e", "gamma2_e"), ("m", "gamma2_m"), ("FR", "kappa2_fr"))))
    return out, (eta0, c, d)


def holonomy():
    U = V = 6; beta = rng.normal(0, 0.7, K); a = rng.normal(0, 0.3, (U, K)); b = rng.normal(0, 0.3, (V, K)); h = rng.normal(0, 1, K)
    out = {}
    for lam in (0.0, 0.02, 0.04, 0.06, 0.1):
        tots, zs = [], []
        for rep in range(20):
            C = np.array([[rng.multinomial(10000, G.softmax(beta + a[u] + b[v] + lam * (u - 2.5) * (v - 2.5) * h)) for v in range(V)] for u in range(U)])
            r = G.plaquettes(C); tots.append(r["total_nats_e"]); zs.append(r["z_e"])
        P_true = np.array([[G.softmax(beta + a[u] + b[v] + lam * (u - 2.5) * (v - 2.5) * h) for v in range(V)] for u in range(U)])
        truth = G.plaquettes(P_true * 1e12)["total_nats_e"]
        out[str(lam)] = dict(truth=truth, mean=float(np.mean(tots)), sd=float(np.std(tots)), z_mean=float(np.mean(zs)),
                             detect=float(np.mean(np.array(zs) > 1.645)))
        print(f"  twist lam={lam:<5}: true holonomy {truth:.4f} nats  estimated {np.mean(tots):+.4f} ± {np.std(tots):.4f}  z {np.mean(zs):+5.2f}  detected {out[str(lam)]['detect']:.2f}")
    return out


def info_loss(jets):
    eta0, c, d = jets; out = {}
    for name, dd in (("e-flat (d = 0)", 0 * d), ("bent (d)", d), ("bent (2d)", 2 * d)):
        g2 = G.curvatures(G.softmax(eta0), c, 2 * dd)["gamma2_e"]; rows = {}
        for n in (25, 100, 400, 1600):
            rows[n] = G.efron_hinkley(eta0, c, dd, n, reps=1500, rng=rng)
        out[name] = dict(gamma2_e=g2, rows={str(k): v for k, v in rows.items()})
        print(f"  {name:16s} gamma2_e {g2:.3f} | n Var(J/nI): " + "  ".join(f"n={k}: {v['nvar']:.3f}" for k, v in rows.items()))
    return out


def nodes():
    ev, od = [i for i in range(16) if T.chi(4)[i] > 0], [i for i in range(16) if T.chi(4)[i] < 0]
    base = np.full(16, 2000.0)
    near = base.copy(); near[ev[0]] = 20; near[ev[1]] = 60; near[od[0]] = near[od[1]] = 25       # node, equal odd partners
    far = base.copy(); far[ev[0]] = 20; far[ev[1]] = 60; far[od[0]] = 25; far[od[1]] = 200        # partners far apart
    out = {}
    for name, c in (("node, equal partners", near), ("partners separated", far)):
        r = G.node_posterior(c, rng=rng); out[name] = r
        print(f"  {name:22s}: top branch {r['branch_top']:.2f}  branches {r['n_branches']}  margin {r['margin_obs']:.2f} [{r['margin_lo']:.2f}, {r['margin_hi']:.2f}]")
    return out


def topinfo():
    out = {}
    for th in (0.0, 0.05, 0.1):
        eta = rng.normal(0, 0.5, 16) * 0
        X = np.array(T.cells(4)); lin = X @ rng.normal(0, 0.3, 4)
        pair = sum(rng.normal(0, 0.2) * X[:, i] * X[:, j] for i in range(4) for j in range(i + 1, 4))
        p = G.softmax(lin + pair + th * T.chi(4) * 16 / 16)
        Ds = [G.top_info(rng.multinomial(20000, p))["D_top_debiased"] for _ in range(30)]
        out[str(th)] = dict(truth=G.top_info(p * 1e12)["D_top"], mean=float(np.mean(Ds)), sd=float(np.std(Ds)))
        print(f"  theta-scale {th}: true D_top {out[str(th)]['truth']:.5f}  estimated {np.mean(Ds):+.5f} ± {np.std(Ds):.5f} (n=20000)")
    return out


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, green, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#b8b6ae"
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7)
    names = list(OUT["paths"]); x = np.arange(len(names)); w = 0.26
    for j, (k, lab, col) in enumerate((("gamma2_e", "e", blue), ("gamma2_m", "m", orange), ("kappa2_fr", "Fisher-Rao", green))):
        ax[0].bar(x + (j - 1) * w, [OUT["paths"][n]["exact"][k] for n in names], w, color=col, label=lab)
        ax[0].errorbar(x + (j - 1) * w, [OUT["paths"][n]["est"]["200000"][k] for n in names], yerr=[OUT["paths"][n]["est"]["200000"][k + "_sd"] for n in names], fmt="o", color=ink, ms=2.5, lw=0.8, zorder=3)
    ax[0].set_xticks(x); ax[0].set_xticklabels([n.split(" (")[0] for n in names], fontsize=7); ax[0].set_yscale("symlog", linthresh=1e-2)
    ax[0].legend(fontsize=6.5, frameon=False); ax[0].set_title("(a) each geodesic has zero curvature\nin its own connection (dots: estimated)", fontsize=8, loc="left")
    H = OUT["holonomy"]; lams = [float(k) for k in H]
    ax[1].plot(lams, [H[k]["truth"] for k in H], ":", color=muted, label="true")
    ax[1].errorbar(lams, [H[k]["mean"] for k in H], yerr=[H[k]["sd"] for k in H], fmt="-o", color=blue, capsize=3, ms=4, label="debiased estimate")
    ax[1].set_xlabel("planted twist λ", fontsize=7.5, color=muted); ax[1].set_ylabel("holonomy (nats per unit)", fontsize=7.5, color=muted)
    ax[1].legend(fontsize=6.5, frameon=False); ax[1].set_title("(b) holonomy 2-form: flat → 0", fontsize=8, loc="left")
    E = OUT["info_loss"]
    for (name, R), col in zip(E.items(), (grey, blue, orange)):
        ns = [int(k) for k in R["rows"]]
        ax[2].plot(ns, [R["rows"][str(k)]["nvar"] for k in ns], "-o", color=col, ms=4, label=f"{name}")
        ax[2].axhline(R["gamma2_e"], color=col, ls=":", lw=1)
    ax[2].set_xscale("log"); ax[2].set_xlabel("n", fontsize=7.5, color=muted); ax[2].set_ylabel("n·Var(J/nI)  (dotted: γ²ₑ)", fontsize=7.5, color=muted)
    ax[2].legend(fontsize=6.5, frameon=False); ax[2].set_title("(c) curvature = information wobble", fontsize=8, loc="left")
    Tn = OUT["topinfo"]; th = [float(k) for k in Tn]
    ax[3].plot(th, [Tn[k]["truth"] for k in Tn], ":", color=muted, label="true")
    ax[3].errorbar(th, [Tn[k]["mean"] for k in Tn], yerr=[Tn[k]["sd"] for k in Tn], fmt="-o", color=green, capsize=3, ms=4, label="debiased estimate")
    ax[3].set_xlabel("planted top interaction", fontsize=7.5, color=muted); ax[3].set_ylabel("D_top (nats per unit)", fontsize=7.5, color=muted)
    ax[3].legend(fontsize=6.5, frameon=False); ax[3].set_title("(d) order-4 information", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(here, "planted_chart.pdf")); fig.savefig(os.path.join(here, "planted_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "planted.json"))))
elif __name__ == "__main__":
    print("(a) path curvatures (exact | estimated from counts)"); Pa, jets = paths()
    print("(b) holonomy 2-form"); Hb = holonomy()
    print("(c) Efron-Hinkley information wobble"); Ec = info_loss(jets)
    print("(d) nodes"); Nd = nodes()
    print("(e) top-order information"); Te = topinfo()
    OUT = json.loads(json.dumps(dict(paths=Pa, holonomy=Hb, info_loss=Ec, nodes=Nd, topinfo=Te), default=float))
    json.dump(OUT, open(os.path.join(here, "planted.json"), "w"), indent=1)
    plot(OUT)
