"""Second-order information geometry on JetClass-II [G2]: probability on a curved space, made visible.

Data and binary observables as in census.py (M mass>50, P2 tau21<0.5, P3 tau32<0.65, N constituents>40; 16 cells).
A second table pairs detector-level with generator-level truth: (M_reco, M_gen, N_reco, N_gen>64) -- near-diagonal,
hence near a node. Populations: all, QCD (labels 161-187), resonance (0-160). Planted checks: planted.py.

[1] The pT path on the Fisher sphere. 12 pT quantile bins; each bin's law is a point x = 2 sqrt(p) on the radius-2
    sphere; its Dirichlet posterior is a cloud there. Fisher-Rao length vs chord. Curvatures along the path in the
    three connections (e: Efron's gamma^2, m, Fisher-Rao), bootstrap bias-corrected, with sd. Global LR tests of the
    path shape. Is the pooled path a mixture PULL (m-flat) of fixed class laws? TV of pooled vs fixed-class mixture.
[2] Curvature as information wobble (Efron-Hinkley): from the fitted local family at the centre of each path,
    n Var(J / nI) at the MLE across n; the predicted plateau is gamma2_e. Relative error-bar wobble sqrt(gamma2/n).
[3] Holonomy 2-form on covariate grids (pT x |eta|, pT x phi, |eta| x phi): debiased information lost by flat
    transport (nats per jet) per plaquette and in total, calibrated against resampled flat grids. phi grids are the
    symmetry null.
[4] Nodes on the reco-vs-gen table, per pT bin: posterior over the toggle branch (which perfect matching), partner
    margin, node radius; the shape table as the far-from-node control.
[5] Order-4 information per pT bin: debiased D_top (nats per jet) with posterior band, co-information.
PLOT_ONLY=1 replots from results.json.
"""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
import pandas as pd
from amb_vigneaux import infogeo2 as G, deform as D

PATH = os.environ.get("JETCLASS2_PARQUET", "/tmp/ds/jetclass2/jetclass2_300k.parquet")
rng = np.random.default_rng(11)


def load():
    df = pd.read_parquet(PATH)
    t21 = df.jet_tau2 / df.jet_tau1.replace(0, np.nan); t32 = df.jet_tau3 / df.jet_tau2.replace(0, np.nan)
    df = df[t21.notna() & t32.notna() & (df.genjet_pt > 0)].copy()
    t21 = df.jet_tau2 / df.jet_tau1; t32 = df.jet_tau3 / df.jet_tau2
    df["cell"] = ((df.jet_sdmass > 50) * 8 + (t21 < 0.5) * 4 + (t32 < 0.65) * 2 + (df.jet_nparticles > 40)).astype(int)
    df["rg"] = ((df.jet_sdmass > 50) * 8 + (df.genjet_sdmass > 50) * 4 + (df.jet_nparticles > 40) * 2 + (df.genjet_nparticles > 64)).astype(int)
    df["qcd"] = (df.jet_label >= 161).astype(int)
    df["lpt"] = np.log(df.jet_pt)
    df["pt12"] = pd.qcut(df.jet_pt, 12, labels=False); df["ptb"] = pd.qcut(df.jet_pt, 8, labels=False)
    df["etab"] = pd.qcut(df.jet_eta.abs(), 4, labels=False)
    df["phib"] = np.floor((df.jet_phi + np.pi) / (2 * np.pi) * 8).clip(0, 7).astype(int)
    return df


POPS = {"all": lambda d: d, "QCD": lambda d: d[d.qcd == 1], "resonance": lambda d: d[d.qcd == 0]}


def table(d, by, col="cell", nb=None):
    nb = nb or int(d[by].max()) + 1
    return np.array([np.bincount(d[d[by] == t][col], minlength=16) for t in range(nb)]).astype(float)


def bin_pos(df, by, nb):
    s = np.array([df[df[by] == t].lpt.mean() for t in range(nb)])
    return (s - s.mean()) / s.std()


# ------------------------------------------------------------------------------------------- [1] + [2]
def part_path(df):
    s = bin_pos(df, "pt12", 12); out = {}
    print("[1] the pT path on the Fisher sphere (12 bins; curvatures bias-corrected, ± bootstrap sd)")
    for name, f in POPS.items():
        d = f(df); tab = table(d, "pt12", nb=12)
        prof = G.path_profile(tab, s, half=2, B=60, rng=rng); tests = G.path_tests(tab, s)
        out[name] = dict(profile=prof, tests=tests, tab=tab.tolist(), s=s.tolist())
        print(f"  {name:10s} FR length {prof['fr_length']:.3f} vs chord {prof['fr_chord']:.3f} (ratio {prof['fr_length'] / prof['fr_chord']:.2f}); "
              f"e-flat path rejected: LR(bend) {tests['lr_bend']:.0f}/{tests['df_bend']}, quadratic still off {tests['dev_quad']:.0f}/{tests['df_quad']}")
        for r in prof["rows"]:
            print(f"     pT bin {r['i']:2d}: gamma2_e {r['gamma2_e']:7.2f}±{r['gamma2_e_sd']:.2f}  gamma2_m {r['gamma2_m']:7.2f}±{r['gamma2_m_sd']:.2f}"
                  f"  kappa2_FR {r['kappa2_fr']:7.2f}±{r['kappa2_fr_sd']:.2f}  speed {r['speed']:.3f}")
    # mixture pull? pooled vs fixed class laws weighted by the per-bin resonance fraction
    tq, tr = table(df[df.qcd == 1], "pt12", nb=12), table(df[df.qcd == 0], "pt12", nb=12)
    w = np.array([[df[(df.pt12 == t)].qcd.mean(), 1 - df[(df.pt12 == t)].qcd.mean()] for t in range(12)])
    tv_fixed = G.mixture_path_check([tq, tr], w)
    tv_moving = [float(0.5 * np.abs(G.smooth(tq[t] + tr[t]) - (w[t, 0] * G.smooth(tq[t]) + w[t, 1] * G.smooth(tr[t]))).sum()) for t in range(12)]
    out["mixture"] = dict(qcd_frac=w[:, 0].tolist(), tv_fixed_class_laws=tv_fixed, tv_moving_class_laws=tv_moving)
    print(f"  pooled = pull between FIXED class laws (an m-flat mixture path)? TV per bin {np.round(tv_fixed, 3).tolist()} (mean {np.mean(tv_fixed):.3f})")
    # sphere picture: PCA of all bin points (3 populations) + posterior clouds + geodesics between the end bins (all)
    pts = {n: G.sphere(G.smooth(np.array(out[n]["tab"]))) for n in POPS}
    X = np.vstack(list(pts.values())); mu = X.mean(0); U = np.linalg.svd(X - mu, full_matrices=False)[2][:3]
    proj = lambda Y: ((Y - mu) @ U.T).tolist()
    clouds = {n: [proj(G.sphere(rng.dirichlet(np.array(out[n]["tab"][t]) + 0.5, size=60))) for t in range(12)] for n in POPS}
    P_all = G.smooth(np.array(out["all"]["tab"]))
    geo = {k: proj(np.array([G.sphere(G.geodesic(P_all[0], P_all[-1], k, t)) for t in np.linspace(0, 1, 40)])) for k in ("e", "m", "fr")}
    out["sphere"] = dict(points={n: proj(v) for n, v in pts.items()}, clouds=clouds, geodesics=geo,
                         explained=(np.linalg.svd(X - mu, compute_uv=False)[:3] ** 2 / (np.linalg.svd(X - mu, compute_uv=False) ** 2).sum()).tolist())
    print("[2] curvature as information wobble (Efron-Hinkley) at the centre of each path")
    out["wobble"] = {}
    for name, f in POPS.items():
        tab = np.array(out[name]["tab"]); p0, d1, d2 = G.local_jets(tab, s, 6, half=2)
        g2 = G.curvatures(p0, d1, d2)["gamma2_e"]; rows = {}
        for n in (10, 25, 100, 400, 1600):
            rows[str(n)] = G.efron_hinkley(np.log(p0), d1, d2 / 2, n, reps=1500, rng=rng)
        out["wobble"][name] = dict(gamma2_e=g2, rows=rows)
        print(f"  {name:10s} gamma2_e(centre) {g2:6.2f} | n Var(J/nI): " + "  ".join(f"n={k}: {v['nvar']:.2f}" for k, v in rows.items())
              + f" | error-bar wobble (sd of observed/expected information) at n=25: ±{100 * np.sqrt(rows['25']['nvar'] / 25):.0f}%, n=100: ±{100 * np.sqrt(rows['100']['nvar'] / 100):.0f}%")
    return out


# ------------------------------------------------------------------------------------------- [3]
def part_holonomy(df):
    out = {}
    print("[3] holonomy 2-form: information lost by flat transport (debiased nats per jet; null = resampled flat grid)")
    for name, f in POPS.items():
        d = f(df); out[name] = {}
        for u, v in (("ptb", "etab"), ("ptb", "phib"), ("etab", "phib")):
            U, V = int(df[u].max()) + 1, int(df[v].max()) + 1; C = np.zeros((U, V, 16))
            np.add.at(C, (d[u].values, d[v].values, d.cell.values), 1)
            h = G.holonomy_info(C, B=100, rng=rng); r = G.plaquettes(C)
            out[name][f"{u}x{v}"] = dict(h, plaq_e=r["nats_e"].tolist(), plaq_total_e=r["total_nats_e"], plaq_total_m=r["total_nats_m"], n=int(C.sum()))
            print(f"  {name:10s} {u} x {v}: holonomy information {1e3 * h['excess_nats']:+6.2f} ± {1e3 * h['sd_nats']:.2f} (1e-3 nats/jet; raw {1e3 * h['nats']:.2f}, "
                  f"flat null {1e3 * h['null_nats']:.2f}) z {h['z']:+5.1f} p {h['p']:.2f} | G/df {h['G'] / h['df']:.2f}")
    return out


# ------------------------------------------------------------------------------------------- [4] + [5]
def part_nodes(df):
    out = {}
    print("[4] nodes: toggle-branch posterior per pT bin (reco-vs-gen table near a node; shape table = control)")
    for name in ("all", "QCD"):
        d = POPS[name](df); out[name] = {}
        for col in ("rg", "cell"):
            rows = []
            for t in range(8):
                c = np.bincount(d[d.ptb == t][col], minlength=16).astype(float); r = G.node_posterior(c, B=2000, rng=rng)
                r["flop_rate"] = D.flop_switch_rate(G.smooth(c), 0.1, reps=400, rng=rng); r["n"] = int(c.sum()); r.pop("pairs"); rows.append(r)
            out[name][col] = rows
            print(f"  {name:4s} {'reco-vs-gen' if col == 'rg' else 'shape      '}: top branch " + " ".join(f"{r['branch_top']:.2f}" for r in rows)
                  + " | margin " + " ".join(f"{r['margin_obs']:.2f}" for r in rows) + " | node radius " + " ".join(f"{r['node_radius']:.3f}" for r in rows))
    print("[5] order-4 information per pT bin: debiased D_top (1e-3 nats per jet) [90% band]")
    out5 = {}
    for name, f in POPS.items():
        d = f(df); out5[name] = {}
        for col in ("cell", "rg"):
            rows = []
            for t in range(8):
                c = np.bincount(d[d.ptb == t][col], minlength=16).astype(float)
                r = G.top_info(c); r.update(G.top_info_posterior(c, B=100, rng=rng)); r["n"] = int(c.sum()); rows.append(r)
            out5[name][col] = rows
            print(f"  {name:10s} {'shape' if col == 'cell' else 'reco-gen'}: " + "  ".join(f"{1e3 * r['D_top_debiased']:.2f}[{1e3 * r['D_lo']:.2f},{1e3 * r['D_hi']:.2f}]" for r in rows))
    return out, out5


# ------------------------------------------------------------------------------------------- plotting
def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, green, grey, red = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#b8b6ae", "#d6402a"
    pc = {"all": ink, "QCD": blue, "resonance": orange}
    fig, ax = plt.subplots(3, 3, figsize=(13, 11))
    for a in ax.ravel():
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7)
    P = OUT["path"]; S = P["sphere"]
    a = ax[0, 0]
    for n, pts in S["points"].items():
        pts = np.array(pts)
        for t, cl in enumerate(S["clouds"][n]):
            cl = np.array(cl); a.scatter(cl[:, 0], cl[:, 1], s=1, color=pc[n], alpha=0.15, lw=0)
        a.plot(pts[:, 0], pts[:, 1], "-o", color=pc[n], ms=3, lw=1.4, label=n)
        a.annotate("low pT", pts[0, :2], fontsize=6, color=pc[n]); a.annotate("high pT", pts[-1, :2], fontsize=6, color=pc[n])
    for k, ls in (("e", "--"), ("m", ":"), ("fr", "-.")):
        g = np.array(S["geodesics"][k]); a.plot(g[:, 0], g[:, 1], ls, color=grey, lw=1, label=f"{k}-geodesic (all)")
    a.set_title(f"(a) pT paths on the Fisher sphere x = 2√p\n(PCA, {100 * sum(S['explained'][:2]):.0f}% of spread; clouds = posteriors)", fontsize=8, loc="left")
    a.legend(fontsize=5.5, frameon=False, loc="best")
    a = ax[0, 1]
    for n in ("QCD", "resonance", "all"):
        R = P[n]["profile"]["rows"]; sx = [r["s"] for r in R]
        a.errorbar(sx, [r["gamma2_e"] for r in R], yerr=[r["gamma2_e_sd"] for r in R], fmt="-o", color=pc[n], ms=3, capsize=2, lw=1.4, label=f"{n}: e (Efron γ²)")
        a.plot(sx, [r["gamma2_m"] for r in R], ":", color=pc[n], lw=1.2, label=f"{n}: m")
    a.axhline(1 / 8, color=grey, lw=0.8); a.text(sx[0], 1 / 8 * 1.15, "1/8", fontsize=6, color=muted)
    a.set_yscale("symlog", linthresh=1); a.set_ylim(-3, 300); a.set_xlabel("standardised log pT", fontsize=7.5, color=muted); a.set_ylabel("curvature per jet", fontsize=7.5, color=muted)
    a.set_title("(b) the path bends in every connection", fontsize=8, loc="left"); a.legend(fontsize=5.5, frameon=False, ncol=2)
    a = ax[0, 2]
    for n, R in P["wobble"].items():
        ns = [int(k) for k in R["rows"]]
        a.plot(ns, [R["rows"][str(k)]["nvar"] for k in ns], "-o", color=pc[n], ms=3, lw=1.4, label=f"{n} (γ²ₑ = {R['gamma2_e']:.1f})")
        a.axhline(R["gamma2_e"], color=pc[n], ls=":", lw=1)
    a.set_xscale("log"); a.set_xlabel("jets n", fontsize=7.5, color=muted); a.set_ylabel("n·Var(observed / expected information)", fontsize=7.5, color=muted)
    a.set_title("(c) curvature = how much the error bar wobbles\n(dotted: Efron–Hinkley prediction γ²ₑ)", fontsize=8, loc="left"); a.legend(fontsize=6, frameon=False)
    H = OUT["holonomy"]
    for a, key, lab in ((ax[1, 0], "ptbxetab", ("pT bin", "|η| bin")), (ax[1, 1], "ptbxphib", ("pT bin", "φ bin"))):
        M = np.array(H["all"][key]["map_excess"]) * 1e3; vm = max(abs(M).max(), 1e-9)
        im = a.imshow(M.T, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
        a.set_xlabel(lab[0], fontsize=7.5, color=muted); a.set_ylabel(lab[1], fontsize=7.5, color=muted)
        plt.colorbar(im, ax=a, fraction=0.046).ax.tick_params(labelsize=6)
        o = H["all"][key]
        a.set_title(f"({'d' if key == 'ptbxetab' else 'e'}) where flat transport fails, all jets (1e-3 nats/jet)\nexcess {1e3 * o['excess_nats']:.2f} ± {1e3 * o['sd_nats']:.2f}, z {o['z']:.1f}", fontsize=8, loc="left")
    a = ax[1, 2]; keys = ("ptbxetab", "ptbxphib", "etabxphib"); x = np.arange(3); w = 0.26
    for j, n in enumerate(("all", "QCD", "resonance")):
        a.bar(x + (j - 1) * w, [H[n][k]["z"] for k in keys], w, color=pc[n], label=n)
    a.axhline(2, color=red, lw=0.8, ls="--"); a.axhline(-2, color=red, lw=0.8, ls="--")
    a.set_xticks(x); a.set_xticklabels(["pT × |η|", "pT × φ (null)", "|η| × φ (null)"], fontsize=7); a.set_ylabel("z vs resampled flat grids", fontsize=7.5, color=muted)
    a.set_yscale("symlog", linthresh=2); a.legend(fontsize=6, frameon=False); a.set_title("(f) holonomy only where symmetry allows it", fontsize=8, loc="left")
    N = OUT["nodes"]; a = ax[2, 0]
    for n, col, ls, lab in (("all", "rg", "-", "all, reco-vs-gen"), ("QCD", "rg", "-", "QCD, reco-vs-gen"), ("all", "cell", ":", "all, shape (control)"), ("QCD", "cell", ":", "QCD, shape (control)")):
        R = N[n][col]; a.plot(range(8), [r["branch_top"] for r in R], ls, marker="o", ms=3, color=pc[n], lw=1.4, label=lab)
    a.axhline(0.5, color=grey, lw=0.8); a.set_ylim(0.3, 1.05); a.set_xlabel("pT bin", fontsize=7.5, color=muted); a.set_ylabel("posterior of the likeliest toggle branch", fontsize=7.5, color=muted)
    a.set_title("(g) near a node the branch is uncertain (flop)", fontsize=8, loc="left"); a.legend(fontsize=6, frameon=False)
    a = ax[2, 1]
    for n, col, mk, lab in (("all", "rg", "o", "all, reco-vs-gen"), ("QCD", "rg", "o", "QCD, reco-vs-gen"), ("all", "cell", "^", "all, shape"), ("QCD", "cell", "^", "QCD, shape")):
        R = N[n][col]
        a.errorbar([r["margin_obs"] for r in R], [r["branch_top"] for r in R],
                   xerr=[[max(r["margin_obs"] - r["margin_lo"], 0) for r in R], [max(r["margin_hi"] - r["margin_obs"], 0) for r in R]],
                   fmt=mk, color=pc[n], ms=4, lw=0.6, alpha=0.8, mfc=pc[n] if col == "rg" else "white", label=lab)
    a.axhline(0.5, color=grey, lw=0.8); a.set_xlabel("partner margin (0 = on the flop wall)", fontsize=7.5, color=muted)
    a.set_ylabel("posterior of the likeliest branch", fontsize=7.5, color=muted); a.legend(fontsize=6, frameon=False)
    a.set_title("(h) branch certainty is set by distance to the wall\n(one point per pT bin; bars 90% posterior)", fontsize=8, loc="left")
    T5 = OUT["topinfo"]; a = ax[2, 2]
    for n in ("QCD", "resonance", "all"):
        R = T5[n]["cell"]; a.plot(range(8), [1e3 * r["D_top_debiased"] for r in R], "-o", ms=3, color=pc[n], lw=1.4, label=f"{n}, shape")
        if n != "QCD":
            a.fill_between(range(8), [1e3 * r["D_lo"] for r in R], [1e3 * r["D_hi"] for r in R], color=pc[n], alpha=0.12, lw=0)
    R = T5["all"]["rg"]; a.plot(range(8), [1e3 * r["D_top_debiased"] for r in R], "--s", ms=3, color=green, lw=1.2, label="all, reco-vs-gen")
    a.axhline(0, color=grey, lw=0.8); a.set_xlabel("pT bin", fontsize=7.5, color=muted); a.set_ylabel("order-4 information (1e-3 nats/jet)", fontsize=7.5, color=muted)
    a.set_title("(i) information only the 4-way joint carries\n(bands 90% posterior; QCD band omitted: ±0.5)", fontsize=8, loc="left"); a.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(here, "jets_chart.pdf")); fig.savefig(os.path.join(here, "jets_chart.png"), dpi=140)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    df = load(); print(f"{len(df)} jets ({df.qcd.mean():.3f} QCD)")
    OUT = dict(path=part_path(df), holonomy=part_holonomy(df))
    OUT["nodes"], OUT["topinfo"] = part_nodes(df)
    js = json.loads(json.dumps(OUT, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    json.dump(js, open(os.path.join(here, "results.json"), "w"))
    plot(js)
