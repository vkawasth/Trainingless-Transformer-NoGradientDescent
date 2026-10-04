"""JetClass-II census: is there curved structure, holonomy and populated nodes before we build on it?

Data: a 300k-jet subsample of JetClass-II (jet-level columns only; JETCLASS2_PARQUET, default
/tmp/ds/jetclass2/jetclass2_300k.parquet). Labels 0..160 = resonance decays (X -> 2, 3, 4 prongs), 161..187 = QCD.
Four binary jet observables (arity 4, 16 cells):
    M  soft-drop mass > 50 GeV        P2 tau2/tau1 < 0.5  (two-prong-like)
    P3 tau3/tau2 < 0.65 (three-prong-like)   N  constituents > 40
Covariates: pT (8 quantile bins), |eta| (4 bins), phi (8 bins on the circle).
Populations: all jets, QCD only, resonance only (simulation truth = the planted answer).

[1] Curvature along pT. The path t -> p_t (16 cells). e-flat path: log mu_{x,t} = a_t + b_x + c_x s_t (a straight line
    in natural coordinates). Curved: + d_x s_t^2. Saturated: free per bin. LR tests by IRLS on the Poisson form.
    Efron's statistical curvature gamma^2 at the middle of the path from the quadratic fit (per observation;
    the n-sample curvature is gamma^2 / n).
[2] Holonomy on covariate grids. Flat transport = no (cell x cov1 x cov2) interaction: log mu = a_{uv} + b_{xu} + c_{xv}
    (IPF). Its LR statistic vs saturated is the sum of squared plaquette holonomies. Grids (pT, |eta|), (pT, phi),
    (|eta|, phi). Null by symmetry: anything with phi.
[3] Node census per pT bin: top interaction theta_hat, near-empty cells, toggle and node radii, flop margin.
"""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
import pandas as pd
from scipy import stats
from amb_vigneaux import toric as T, deform as D

PATH = os.environ.get("JETCLASS2_PARQUET", "/tmp/ds/jetclass2/jetclass2_300k.parquet")
NAMES = ("M", "P2", "P3", "N")


def load():
    df = pd.read_parquet(PATH)
    t21 = df.jet_tau2 / df.jet_tau1.replace(0, np.nan); t32 = df.jet_tau3 / df.jet_tau2.replace(0, np.nan)
    df = df.assign(M=(df.jet_sdmass > 50).astype(int), P2=(t21 < 0.5).astype(int), P3=(t32 < 0.65).astype(int),
                   N=(df.jet_nparticles > 40).astype(int))
    df = df[t21.notna() & t32.notna()].copy()
    df["cell"] = df.M * 8 + df.P2 * 4 + df.P3 * 2 + df.N
    df["qcd"] = (df.jet_label >= 161).astype(int)
    df["ptb"] = pd.qcut(df.jet_pt, 8, labels=False)
    df["etab"] = pd.qcut(df.jet_eta.abs(), 4, labels=False)
    df["phib"] = np.floor((df.jet_phi + np.pi) / (2 * np.pi) * 8).clip(0, 7).astype(int)
    df["lpt"] = np.log(df.jet_pt)
    return df


POPS = {"all": lambda d: d, "QCD": lambda d: d[d.qcd == 1], "resonance": lambda d: d[d.qcd == 0]}


# ------------------------------------------------------------------------------------- [1] curvature along pT
def poisson_irls(X, y, iters=100):
    mu = y + 0.5; eta = np.log(mu)
    for _ in range(iters):
        z = eta + (y - mu) / mu; w = mu
        beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], z * np.sqrt(w), rcond=None)[0]
        new = X @ beta
        if np.max(np.abs(new - eta)) < 1e-10:
            eta = new; break
        eta = new; mu = np.exp(eta)
    mu = np.exp(eta)
    return mu, beta


def dev(y, mu):
    m = y > 0
    return float(2 * (np.sum(y[m] * np.log(y[m] / mu[m])) - np.sum(y - mu)))


def curvature_path(d):
    tab = np.zeros((8, 16)); s = np.zeros(8)
    for t in range(8):
        g = d[d.ptb == t]; tab[t] = np.bincount(g.cell, minlength=16); s[t] = g.lpt.mean()
    s = (s - s.mean()) / s.std()
    y = tab.ravel(); tt = np.repeat(np.arange(8), 16); xx = np.tile(np.arange(16), 8)
    At = np.eye(8)[tt]; Bx = np.eye(16)[xx][:, 1:]
    X1 = np.hstack([At, Bx, Bx * s[tt][:, None]]); X2 = np.hstack([X1, Bx * (s[tt] ** 2)[:, None]])
    mu1, _ = poisson_irls(X1, y); mu2, b2 = poisson_irls(X2, y)
    D1, D2 = dev(y, mu1), dev(y, mu2); df_sat1 = 8 * 15 - 30; df_sat2 = df_sat1 - 15
    # jets at s = 0 (natural coordinates of the 16-cell law, cell 0 as reference)
    c = np.concatenate([[0], b2[8 + 15: 8 + 30]]); dq = np.concatenate([[0], b2[8 + 30:]])
    d1, d2 = c, 2 * dq
    p0 = mu2.reshape(8, 16)[3:5].sum(0); p0 = p0 / p0.sum()
    I = np.diag(p0) - np.outer(p0, p0)
    a = d1 @ I @ d1; b = d2 @ I @ d2; cc = d1 @ I @ d2
    g2 = float((a * b - cc ** 2) / a ** 3)
    return dict(n=int(tab.sum()), dev_eflat=D1, df_eflat=df_sat1, p_eflat=float(stats.chi2.sf(D1, df_sat1)),
                dev_quad=D2, df_quad=df_sat2, p_quad=float(stats.chi2.sf(D2, df_sat2)),
                lr_curved=D1 - D2, p_curved=float(stats.chi2.sf(D1 - D2, 15)), gamma2=g2,
                n_where_gamma2n_is_1_8=float(8 * g2), speed=float(np.sqrt(a)))


# ------------------------------------------------------------------------------------- [2] holonomy on grids
def no3way(Y, iters=2000, tol=1e-9):
    M = np.ones_like(Y, float)
    for _ in range(iters):
        old = M.copy()
        for ax in (0, 1, 2):
            m_obs = Y.sum(ax, keepdims=True); m_fit = M.sum(ax, keepdims=True)
            M = M * np.where(m_fit > 0, m_obs / np.maximum(m_fit, 1e-300), 0)
        if np.max(np.abs(M - old)) < tol:
            break
    return M


def holonomy(d, u, v):
    U, V = d[u].max() + 1, d[v].max() + 1
    Y = np.zeros((16, U, V))
    np.add.at(Y, (d.cell.values, d[u].values, d[v].values), 1)
    M = no3way(Y); G = dev(Y.ravel(), M.ravel()); df = 15 * (U - 1) * (V - 1)
    # plaquette holonomies on the log scale for the commonest cell contrast (cell vs cell-sum reference), for plotting
    L = np.log(Y + 0.5); L = L - L.mean(0, keepdims=True)
    H = L[:, 1:, 1:] - L[:, 1:, :-1] - L[:, :-1, 1:] + L[:, :-1, :-1]
    return dict(G=G, df=df, ratio=G / df, p=float(stats.chi2.sf(G, df)), max_abs_plaquette=float(np.abs(H).max()),
                rms_plaquette=float(np.sqrt((H ** 2).mean())))


# ------------------------------------------------------------------------------------- [3] node census
def nodes(d):
    out = []
    for t in range(8):
        cnt = np.bincount(d[d.ptb == t].cell, minlength=16).astype(float); p = (cnt + 0.5) / (cnt + 0.5).sum()
        th = T.theta_hat(cnt); r = D.radii(p)
        out.append(dict(ptb=t, n=int(cnt.sum()), theta=th["theta"], theta_se=th["se"], empty_lt5=int((cnt < 5).sum()),
                        min_cell=float(p.min()), toggle_radius=r["toggle_radius"], node_radius=r["node_radius"],
                        partner_margin=r["partner_margin"], flop_rate=D.flop_switch_rate(p, 0.1, reps=400)))
    return out


def main():
    df = load(); OUT = {}
    print(f"{len(df)} jets; QCD {df.qcd.mean():.3f}; cell occupancy (all): {np.bincount(df.cell, minlength=16).tolist()}")
    for name, f in POPS.items():
        d = f(df); R = {}
        R["curvature"] = c = curvature_path(d)
        print(f"\n== {name} (n={len(d)})")
        print(f"[1] pT path: e-flat dev {c['dev_eflat']:.0f}/{c['df_eflat']} (p={c['p_eflat']:.1e}); curved term LR {c['lr_curved']:.0f}/15 (p={c['p_curved']:.1e});"
              f" quad dev {c['dev_quad']:.0f}/{c['df_quad']}; gamma^2 = {c['gamma2']:.3g} per obs (n-sample curvature >= 1/8 only for n < {c['n_where_gamma2n_is_1_8']:.1f})")
        R["holonomy"] = {}
        for u, v in (("ptb", "etab"), ("ptb", "phib"), ("etab", "phib")):
            h = holonomy(d, u, v); R["holonomy"][f"{u}x{v}"] = h
            print(f"[2] holonomy {u} x {v}: G {h['G']:.0f}/{h['df']} = {h['ratio']:.2f} (p={h['p']:.2g}); rms plaquette {h['rms_plaquette']:.3f}")
        R["nodes"] = nd = nodes(d)
        for r in nd:
            print(f"[3] pT bin {r['ptb']}: n {r['n']:6d} theta {r['theta']:+.3f}±{r['theta_se']:.3f} empty(<5) {r['empty_lt5']:2d} min p {r['min_cell']:.1e}"
                  f" toggle r {r['toggle_radius']:.4f} node r {r['node_radius']:.4f} partner margin {r['partner_margin']:.2f} flop {r['flop_rate']:.2f}")
        OUT[name] = R
    json.dump(json.loads(json.dumps(OUT, default=float)), open(os.path.join(here, "census.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
