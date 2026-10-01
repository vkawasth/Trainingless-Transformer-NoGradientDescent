"""Toric geometry of the sphere covers, component by component (amb_vigneaux/toric.py).

Components, each computed on its own:
  [F] fibre: what the (n-1)-way data alone allow (a segment; mixture coordinate m = <chi, P>)
  [T] toric variety: theta = 0 (one binomial of degree 2^{n-1}); the Birch point = IPF = max-ent gluing
  [I] interaction: theta_hat +/- CI from units where all n are observed; where the observed law sits on the segment
  [S] strata: exact AMB toggle radius of every cell in the no-top-interaction model vs the saturated radius
(0) synthetic checks (n = 3, 4, 5): Birch = IPF, binomial residual, monotonicity; the higher PR box (empty [F]).
(1) real tables where the full joint is observed: MBIC three ideology groups (1,600 sentences), AllSides three sides
    (736 triple-covered stories), MBIC four ideology groups (1,200 sentences).
(2) one outcome through the components ('all n biased/adversarial'): [F] gives bounds, [F]+[T] the max-ent point,
    [I] the observed value; the gap between the last two is theta's effect on the outcome.
Env: MBIC_XLSX, ALLSIDES_DIR.
"""
import os, sys, json, collections, io, contextlib
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../bias3"), os.path.join(here, "../allsides")]
import numpy as np, pandas as pd
from amb_vigneaux import toric as T

rng = np.random.default_rng(0)


def quiet(f, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return f(*a, **k)


def part0():
    out = {}
    print("(0) synthetic")
    for n in (3, 4, 5):
        d_birch, resid, mono = [], [], True
        for _ in range(50):
            p = rng.dirichlet(np.ones(2 ** n)); Pi = T.point_from_margins(p, n); Pb, _ = T.birch(Pi)
            d_birch.append(np.abs(Pb - Pi).max()); resid.append(abs(np.log(np.prod(Pi[T.chi(n) > 0]) / np.prod(Pi[T.chi(n) < 0]))))
            F = T.fibre(Pi); ts = np.linspace(F["t_lo"], F["t_hi"], 50)[1:-1]
            th = [T.theta(Pi + t * T.chi(n)) for t in ts]; mono &= bool(np.all(np.diff(th) > 0))
        out[n] = dict(max_birch_minus_ipf=float(max(d_birch)), max_log_binomial_residual=float(max(resid)), monotone=mono, degree=2 ** (n - 1))
        print(f"    n={n}: binomial degree {2 ** (n - 1)}; max |Birch - IPF| {max(d_birch):.1e}; max |log binomial ratio| at IPF {max(resid):.1e}; theta increasing along the fibre: {mono}")
    # higher PR box: triple tables uniform on even parity. Signed laws with these 3-margins form the line P0 + t chi;
    # [F] = that line inside the simplex, which is empty
    from amb_vigneaux.strata import design
    A, _ = design((2,) * 4, T.margins(4))
    tri = np.array([1.0 if sum(x) % 2 == 0 else 0.0 for x in T.cells(3)]) / 4
    b = np.concatenate([tri for _ in T.margins(4)])
    P0, *_ = np.linalg.lstsq(A, b, rcond=None)
    F = T.fibre(P0)
    out["higher_pr"] = dict(t_lo=F["t_lo"], t_hi=F["t_hi"], nonempty=F["nonempty"], margin_residual=float(np.abs(A @ P0 - b).max()))
    print(f"    higher PR box: signed law with its 3-margins found (residual {out['higher_pr']['margin_residual']:.1e});"
          f" positivity needs t in [{F['t_lo']:.4f}, {F['t_hi']:.4f}] -> [F] {'non-empty' if F['nonempty'] else 'empty'}")
    return out


def counts_from(Tm):
    n = Tm.shape[1]; c = np.zeros(2 ** n)
    for x in Tm:
        c[int("".join(map(str, x)), 2)] += 1
    return c


def mbic_tables():
    import mbic
    d = mbic.load()
    g3 = pd.cut(d["political_ideology"], [-11, -4, 3, 11], labels=list("LCR"))
    S3 = d.assign(g=g3).groupby(["sentence_id", "g"], observed=True)["b"].mean().unstack().dropna()
    g4 = pd.cut(d["political_ideology"], [-11, -5, 0, 5, 11], labels=list("abcd"))
    S4 = d.assign(g=g4).groupby(["sentence_id", "g"], observed=True)["b"].mean().unstack().dropna()
    return (S3[list("LCR")].to_numpy() >= .5).astype(int), (S4[list("abcd")].to_numpy() >= .5).astype(int)


def allsides_table():
    import load
    st = collections.defaultdict(dict)
    for r in load.load():
        k = (r["date"], r["topic"]); s = r["side"][0].upper(); st[k][s] = max(st[k].get(s, 0), r["outcome"])
    return np.array([[v["L"], v["C"], v["R"]] for v in st.values() if len(v) == 3])


def part1(tables):
    out = {}
    print("\n(1) real tables (full joint observed)")
    for name, Tm in tables.items():
        c = counts_from(Tm); r = quiet(T.report, counts=c)
        S = r["S"]; ratio = np.array([s["radius"] / s["saturated"] for s in S])
        r["S_summary"] = dict(median_ratio=float(np.median(ratio)), min_ratio=float(ratio.min()), max_ratio=float(ratio.max()),
                              forced_sizes=sorted(collections.Counter(len(s["forced"]) for s in S).items()))
        out[name] = dict(r, n_units=int(len(Tm)))
        I = r["I"]; F = r["F"]
        print(f"  {name} ({len(Tm)} units, n={r['n']}):")
        print(f"    [F] segment m in [{F['m_lo']:.3f}, {F['m_hi']:.3f}]; observed m {F['m_obs']:.3f}")
        print(f"    [T] binomial degree {r['binomial_degree']}; theta at IPF {r['T']['theta_at_ipf']:.1e}; |Birch - IPF| {r['T']['birch_minus_ipf']:.1e}")
        print(f"    [I] theta_hat {I['theta']:+.3f} (95% CI [{I['lo']:+.3f}, {I['hi']:+.3f}], z {I['z']:+.2f}); observed law at t = {I['t_obs']:+.4f}"
              f" on the segment; TV(observed, Birch) {I['tv_obs_to_birch']:.3f}")
        print(f"    [S] toggle radius / saturated radius: median {r['S_summary']['median_ratio']:.2f} (range {r['S_summary']['min_ratio']:.2f}-{r['S_summary']['max_ratio']:.2f});"
              f" forced-set sizes {r['S_summary']['forced_sizes']}")
    return out


def part2(tables, rep):
    print("\n(2) one outcome through the components: P(all n positive)")
    out = {}
    for name, Tm in tables.items():
        n = Tm.shape[1]; c = counts_from(Tm); p = (c + .5) / (c + .5).sum()
        f = np.array([float(all(x)) for x in T.cells(n)])
        Pi = T.point_from_margins(p, n); F = T.fibre(Pi); x = T.chi(n)
        lo, hi = sorted([f @ (Pi + F["t_lo"] * x), f @ (Pi + F["t_hi"] * x)])
        obs = float(f @ p); me = float(f @ Pi)
        se = np.sqrt(obs * (1 - obs) / len(Tm))
        out[name] = dict(F_bounds=[float(lo), float(hi)], FT_maxent=me, I_observed=obs, observed_se=float(se), theta_effect=obs - me)
        print(f"  {name}: [F] bounds [{lo:.3f}, {hi:.3f}] | [F]+[T] max-ent {me:.3f} | [I] observed {obs:.3f} (se {se:.3f}) | theta's effect {obs - me:+.3f}")
    return out


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#b8b6ae"
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[1.15, 1, 1, 1.1]))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7.5)
    R = OUT["real"]; names = list(R); key = names[-1]; r = R[key]
    # (a) [F]+[T]+[I] along the segment of the MBIC 4-group table: theta(t)
    F = r["F"]; ts = np.linspace(F["t_lo"], F["t_hi"], 400)[1:-1]
    c = np.array(r["counts"]); p = (c + .5) / (c + .5).sum(); n = r["n"]; Pi = T.point_from_margins(p, n); x = T.chi(n)
    th = [T.theta(Pi + t * x) for t in ts]; m = [x @ (Pi + t * x) for t in ts]
    ax[0].plot(m, th, color=blue, lw=2, label="θ along the fibre [F]")
    ax[0].axhline(0, color=muted, lw=0.8, ls=":"); ax[0].scatter([x @ Pi], [0], color=ink, s=26, zorder=4, label="Birch point = IPF [T]")
    I = r["I"]; ax[0].errorbar([F["m_obs"]], [I["theta"]], yerr=[[I["theta"] - I["lo"]], [I["hi"] - I["theta"]]], fmt="o", color=orange, ms=5, capsize=3, label="observed θ̂ ± 95% [I]")
    ax[0].set_ylim(-1.2, 1.2); ax[0].set_xlabel("mixture coordinate m = ⟨χ, P⟩ (the fibre)", fontsize=8, color=muted); ax[0].set_ylabel("θ (top interaction)", fontsize=8, color=muted)
    ax[0].set_title(f"(a) {key}: fibre, variety, data", fontsize=8.5, loc="left", color=ink); ax[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    # (b) theta_hat with CI per table
    y = np.arange(len(names))[::-1]
    for yi, nm in zip(y, names):
        I = R[nm]["I"]; ax[1].plot([I["lo"], I["hi"]], [yi, yi], color=orange, lw=3, alpha=0.6); ax[1].scatter([I["theta"]], [yi], color=ink, s=16, zorder=3)
    ax[1].axvline(0, color=muted, lw=0.8, ls=":"); ax[1].set_yticks(y); ax[1].set_yticklabels(names, fontsize=7)
    ax[1].set_xlabel("θ̂ (top interaction), 95% CI", fontsize=8, color=muted); ax[1].set_title("(b) [I] is the top interaction there?", fontsize=8.5, loc="left", color=ink)
    # (c) one outcome through the components
    O = OUT["outcome"]
    for yi, nm in zip(y, names):
        o = O[nm]; ax[2].plot(o["F_bounds"], [yi, yi], color=blue, lw=4, alpha=0.35)
        ax[2].scatter([o["FT_maxent"]], [yi], color=blue, marker="|", s=120, zorder=3)
        ax[2].errorbar([o["I_observed"]], [yi], xerr=[[1.96 * o["observed_se"]], [1.96 * o["observed_se"]]], fmt="o", color=orange, ms=4, capsize=2)
    ax[2].set_yticks(y); ax[2].set_yticklabels(names, fontsize=7)
    ax[2].set_xlabel("P(all positive)", fontsize=8, color=muted); ax[2].set_title("(c) outcome: [F] band, [T] tick, [I] dot", fontsize=8.5, loc="left", color=ink)
    # (d) strata: toggle radius vs saturated, MBIC 4 groups
    S = r["S"]; sat = [s["saturated"] for s in S]; rad = [s["radius"] for s in S]
    ax[3].scatter(sat, rad, color=blue, s=18); mx = max(rad) * 1.05; ax[3].plot([0, mx], [0, mx], color=muted, lw=0.8, ls=":")
    ax[3].set_xlabel("saturated radius −log(1 − p_c)", fontsize=8, color=muted); ax[3].set_ylabel("toggle radius, no top interaction", fontsize=8, color=muted)
    ax[3].set_title("(d) [S] toggle radius (MBIC 4)", fontsize=8.5, loc="left", color=ink)
    fig.tight_layout(); fig.savefig(os.path.join(here, "toric_chart.pdf")); fig.savefig(os.path.join(here, "toric_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    m3, m4 = mbic_tables()
    tables = {"MBIC 3 groups": m3, "AllSides 3 sides": allsides_table(), "MBIC 4 groups": m4}
    OUT = dict(synthetic=part0(), real=part1(tables), outcome=None)
    for nm, Tm in tables.items():
        OUT["real"][nm]["counts"] = counts_from(Tm).tolist()
    OUT["outcome"] = part2(tables, OUT["real"])
    json.dump(OUT, open(os.path.join(here, "results.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    plot(OUT)
