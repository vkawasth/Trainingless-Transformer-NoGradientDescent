"""Is statistical holonomy a norm on the cohomology class? Checks of amb_vigneaux/holonomy_norm.py.

(a) Class function: KL is unchanged by coboundaries (vertex gauge) and by moving the base vertex (200 random cases).
(b) Zero set: generic law -> KL > 0 for every non-zero class; uniform law -> KL = 0 for all classes; a law with a
    period-n/2 symmetry -> KL = 0 at the non-zero class h = n/2. So 'KL = 0 iff [g] = 0' needs a free action.
(c) Shifts on Z/32 of a smooth skewed law: KL(h) against 1/2 h^2 F_loc (F_loc = sum (p')^2 / p): quadratic near 0,
    asymmetric (KL(h) != KL(-h)), periodic, not homogeneous -- not a norm, but locally the Fisher norm squared / 2.
(d) Tilts (A = R^k): KL equals the Bregman divergence of psi exactly; the quadratic form's relative error is O(|h|);
    an unfaithful statistic (a constant component) gives a degenerate form.
(e) Real data: the plaquette holonomies H of JetClass-II (pT x |eta|, all jets) are tilts with T = cell indicators;
    exact KL(p || tilt_H p) against 1/2 H^T I H on every plaquette (needs JETCLASS2_PARQUET; skipped if absent).
"""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
from amb_vigneaux import holonomy_norm as H

rng = np.random.default_rng(5)


def part_a(n=12, m=5, cases=200):
    worst = 0.0
    for _ in range(cases):
        p = rng.dirichlet(np.ones(n)); g = rng.integers(0, n, m)
        base = H.kl_shift(p, g)
        g2 = (g + H.coboundary(rng.integers(0, n, m))) % n          # vertex gauge
        g3 = np.roll(g, rng.integers(m))                               # move the base vertex
        worst = max(worst, abs(H.kl_shift(p, g2) - base), abs(H.kl_shift(p, g3) - base))
    print(f"(a) class function: max |change of KL| under gauge / base change over {cases} cases = {worst:.1e}")
    return dict(max_change=worst)


def part_b(n=12):
    p = rng.dirichlet(np.ones(n)); u = np.full(n, 1 / n)
    half = rng.dirichlet(np.ones(n // 2)); sym = np.concatenate([half, half]) / 2
    rows = {name: [H.kl(q, H.shift(q, h)) for h in range(n)] for name, q in (("generic", p), ("uniform", u), ("period n/2", sym))}
    for name, r in rows.items():
        zeros = [h for h in range(1, n) if r[h] < 1e-12]
        print(f"(b) {name:10s}: non-zero classes with KL = 0: {zeros or 'none'}; stabiliser {H.stabiliser({'generic': p, 'uniform': u, 'period n/2': sym}[name])}")
    return rows


def part_c(n=32):
    x = np.arange(n); p = np.exp(1.2 * np.sin(2 * np.pi * x / n) + 0.6 * np.sin(4 * np.pi * x / n + 1)); p /= p.sum()
    dp = (np.roll(p, -1) - np.roll(p, 1)) / 2; F = float(np.sum(dp ** 2 / p))
    hs = np.arange(-n // 2, n // 2 + 1); kls = [H.kl(p, H.shift(p, h)) for h in hs]
    asym = max(abs(H.kl(p, H.shift(p, h)) - H.kl(p, H.shift(p, -h))) for h in range(1, n // 2))
    print(f"(c) Z/{n}: F_loc {F:.4f}; KL(1) {kls[n // 2 + 1]:.4f} vs F/2 {F / 2:.4f}; KL(2)/KL(1) {kls[n // 2 + 2] / kls[n // 2 + 1]:.2f} (quadratic: 4);"
          f" KL(8)/KL(4) {kls[n // 2 + 8] / kls[n // 2 + 4]:.2f}; max |KL(h) - KL(-h)| {asym:.3f}")
    return dict(h=hs.tolist(), kl=kls, F=F, p=p.tolist(), asym=asym)


def part_d(K=10, k=3):
    p = rng.dirichlet(np.ones(K)); T = rng.normal(size=(K, k)); u = rng.normal(size=k); u /= np.linalg.norm(u)
    scales = np.geomspace(1e-3, 2, 20); exact, brg, qd = [], [], []
    for s in scales:
        h = s * u; exact.append(H.kl_tilt(p, h, T)); brg.append(H.bregman_psi(p, h, T)); qd.append(H.quad(p, h, T))
    exact, brg, qd = map(np.array, (exact, brg, qd))
    Tu = np.column_stack([T[:, :2], np.ones(K)])                     # unfaithful: third statistic constant
    ev = np.linalg.eigvalsh(H.fisher_tilt(p, Tu))
    print(f"(d) tilts: max |KL - Bregman| {np.max(np.abs(exact - brg)):.1e}; relative error of the quadratic form at |h| = 0.01 / 0.1 / 1:"
          f" {abs(exact[np.argmin(abs(scales - .01))] / qd[np.argmin(abs(scales - .01))] - 1):.3f} / {abs(exact[np.argmin(abs(scales - .1))] / qd[np.argmin(abs(scales - .1))] - 1):.3f}"
          f" / {abs(exact[np.argmin(abs(scales - 1))] / qd[np.argmin(abs(scales - 1))] - 1):.3f}; unfaithful T: smallest Fisher eigenvalue {ev.min():.1e}")
    return dict(scales=scales.tolist(), exact=exact.tolist(), quad=qd.tolist(), unfaithful_min_eig=float(ev.min()))


def part_e():
    path = os.environ.get("JETCLASS2_PARQUET", "/tmp/ds/jetclass2/jetclass2_300k.parquet")
    if not os.path.exists(path):
        print("(e) skipped (no JetClass-II file)"); return None
    sys.path.insert(0, os.path.join(here, "../jets"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("jets_run", os.path.join(here, "../jets/run.py")); J = importlib.util.module_from_spec(spec); spec.loader.exec_module(J)
    df = J.load(); C = np.zeros((8, 4, 16)); np.add.at(C, (df.ptb.values, df.etab.values, df.cell.values), 1)
    P = (C + 0.5) / (C + 0.5).sum(-1, keepdims=True); L = np.log(P)
    ex, qd = [], []
    for u in range(7):
        for v in range(3):
            Hh = L[u + 1, v + 1] - L[u + 1, v] - L[u, v + 1] + L[u, v]; p = P[u, v]
            T = np.eye(16)
            ex.append(H.kl_tilt(p, Hh, T)); qd.append(H.quad(p, Hh, T))
    ex, qd = np.array(ex), np.array(qd)
    print(f"(e) JetClass-II plaquettes (pT x |eta|): exact tilt KL vs 1/2 H^T I H: median ratio {np.median(ex / qd):.3f},"
          f" range {np.min(ex / qd):.3f}-{np.max(ex / qd):.3f}; |H| up to {np.max(np.sqrt(2 * qd)):.2f} in Fisher units")
    return dict(exact=ex.tolist(), quad=qd.tolist())


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, green, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#b8b6ae"
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7)
    B = OUT["b"]
    for (name, r), col in zip(B.items(), (blue, grey, orange)):
        ax[0].plot(range(len(r)), r, "-o", ms=3, color=col, label=name)
    ax[0].set_xlabel("class h ∈ Z/12", fontsize=7.5, color=muted); ax[0].set_ylabel("KL(p ‖ τ_h p)", fontsize=7.5, color=muted)
    ax[0].legend(fontsize=6.5, frameon=False); ax[0].set_title("(a) KL = 0 off the zero class\nwhen the law has symmetry", fontsize=8, loc="left")
    C = OUT["c"]; h = np.array(C["h"])
    ax[1].plot(h, C["kl"], "-o", ms=3, color=blue, label="KL(p ‖ τ_h p)")
    ax[1].plot(h, 0.5 * C["F"] * h ** 2, ":", color=ink, label="½ F h²")
    ax[1].set_ylim(0, max(C["kl"]) * 1.3); ax[1].set_xlabel("shift h on Z/32", fontsize=7.5, color=muted)
    ax[1].legend(fontsize=6.5, frameon=False); ax[1].set_title("(b) locally the Fisher form, globally\nasymmetric and periodic: not a norm", fontsize=8, loc="left")
    D = OUT["d"]; s = np.array(D["scales"])
    ax[2].loglog(s, D["exact"], "-o", ms=3, color=blue, label="KL (= Bregman of ψ)"); ax[2].loglog(s, D["quad"], ":", color=ink, label="½ hᵀFh")
    ax[2].set_xlabel("|h| (tilt)", fontsize=7.5, color=muted); ax[2].legend(fontsize=6.5, frameon=False)
    ax[2].set_title("(c) tilts: exact Bregman, quadratic to O(|h|³)", fontsize=8, loc="left")
    E = OUT.get("e")
    if E:
        ax[3].scatter(E["quad"], E["exact"], s=12, color=green)
        mx = max(max(E["quad"]), max(E["exact"])); ax[3].plot([0, mx], [0, mx], ":", color=ink)
        ax[3].set_xlabel("½ HᵀIH (plaquette)", fontsize=7.5, color=muted); ax[3].set_ylabel("exact KL", fontsize=7.5, color=muted)
        ax[3].set_title("(d) JetClass-II plaquettes: the\nquadratic norm is the holonomy", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(here, "holonomy_norm_chart.pdf")); fig.savefig(os.path.join(here, "holonomy_norm_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    OUT = dict(a=part_a(), b=part_b(), c=part_c(), d=part_d(), e=part_e())
    OUT = json.loads(json.dumps(OUT, default=float)); json.dump(OUT, open(os.path.join(here, "results.json"), "w"), indent=1); plot(OUT)
