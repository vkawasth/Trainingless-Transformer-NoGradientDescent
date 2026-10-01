"""Second-order holonomy on a cover whose nerve is a sphere (amb_vigneaux/higher.py).

(0) Covers: triangle {LC, CR, LR}, CHSH 4-cycle, tetrahedral {abc, abd, acd, bcd}: Betti numbers of the nerve, loops of
    the context graph and how many are filled by 2-cells (path 2-groupoid), dim ker R and what spans it.
(1) Higher PR box on the tetrahedral cover (each triple uniform on even parity), mixed with noise at weight lambda:
    first-order view (the 6 pair contexts: what any pairwise / loop-based method sees) vs second-order view (the
    4 triple contexts): CF, gamma, and the bound on the 4-way parity.
(2) MBIC with four ideology groups A..D (political_ideology bins [-10,-5], [-4,0], [1,5], [6,10]; group outcome =
    group majority 'biased'); 1,200 sentences rated by all four. 100 random 5-way splits: each triple context from
    its own fifth, the 4-way truth from the last fifth. Bounds (plug-in, and bootstrap-widened, + truth noise) on
    all four biased, at least three of four, unanimity of four, and even parity; identification <f, chi_4>;
    CF and gamma of the plug-in tables; the pair-only view. Instrumentation of the nerve for one split.
(3) Figure: the triangle complex, the tetrahedral complex (MBIC instrumentation), CF vs lambda in both views, MBIC bounds.
Env: MBIC_XLSX.
"""
import os, sys, json, itertools
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../bias3")]
import numpy as np, pandas as pd
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.outcome import contextual_fraction, analyse_outcomes
from amb_vigneaux.bounds import functional, outcome_bounds, loop_space
from amb_vigneaux.higher import path_2groupoid, complex_instrument, plot_complex

rng = np.random.default_rng(0)
M4 = ("a", "b", "c", "d")
TET = Scenario({m: (0, 1) for m in M4}, tuple(itertools.combinations(M4, 3)))
PAIRS = Scenario({m: (0, 1) for m in M4}, tuple(itertools.combinations(M4, 2)))
CHI4 = functional(TET, lambda g: (-1.0) ** sum(g.values()))
F = {"all four": functional(TET, lambda g: float(sum(g.values()) == 4)),
     "at least three": functional(TET, lambda g: float(sum(g.values()) >= 3)),
     "unanimous": functional(TET, lambda g: float(sum(g.values()) in (0, 4))),
     "even parity": functional(TET, lambda g: float(sum(g.values()) % 2 == 0))}
TRUTH = {"all four": lambda T: (T.sum(1) == 4).mean(), "at least three": lambda T: (T.sum(1) >= 3).mean(),
         "unanimous": lambda T: np.isin(T.sum(1), (0, 4)).mean(), "even parity": lambda T: (T.sum(1) % 2 == 0).mean()}


def part0():
    covers = {"triangle {LC,CR,LR}": [("L", "C"), ("C", "R"), ("L", "R")],
              "CHSH 4-cycle": [("a0", "b0"), ("a0", "b1"), ("a1", "b0"), ("a1", "b1")],
              "tetrahedral {abc,abd,acd,bcd}": list(itertools.combinations(M4, 3))}
    out = {}
    print("(0) covers")
    for name, c in covers.items():
        g = path_2groupoid(c); meas = sorted({m for x in c for m in x})
        sc = Scenario({m: (0, 1) for m in meas}, tuple(c)); K = loop_space(sc)
        par = functional(sc, lambda gg: (-1.0) ** sum(gg.values())); spanned = K.shape[1] == 1 and abs(abs(np.corrcoef(K[:, 0], par)[0, 1]) - 1) < 1e-9
        out[name] = dict(betti=g["betti"], graph_loops=g["graph_beta1"], filled=g["filled"], two_cells=len(g["two_cells"]), dim_kerR=int(K.shape[1]), top_parity=bool(spanned))
        print(f"    {name:32s} nerve Betti {g['betti']}; context-graph loops {g['graph_beta1']}, filled by 2-cells {g['filled']};"
              f" dim ker R {K.shape[1]}" + (" (the top parity)" if spanned else ""))
    return out


def part1():
    pr = {C: np.array([1.0 if sum(s) % 2 == 0 else 0.0 for s in TET.sections(C)]) / 4 for C in TET.contexts}
    rows = []
    print("\n(1) higher PR box (each triple uniform on even parity) + noise")
    for lam in np.round(np.linspace(0, 1, 21), 3):
        tri = EmpiricalModel(TET, {C: lam * pr[C] + (1 - lam) / 8 for C in TET.contexts})
        pairs = EmpiricalModel(PAIRS, {P: TET.restriction_matrix(next(C for C in TET.contexts if set(P) <= set(C)), P) @ tri.tables[next(C for C in TET.contexts if set(P) <= set(C))] for P in PAIRS.contexts})
        cf2 = contextual_fraction(tri).value; cf1 = contextual_fraction(pairs).value
        rep = analyse_outcomes(tri, eps=1e-9, with_cf=False)
        b = outcome_bounds(tri, F["even parity"])
        rows.append(dict(lam=float(lam), CF_pairs=cf1, CF_triples=cf2, gamma=bool(rep.gamma_h1_nonzero), parity_lo=b["lo"], parity_hi=b["hi"], mode=b["mode"]))
    for r in rows[::4] + [rows[-1]]:
        print(f"    lambda {r['lam']:.2f}: first-order (pairs) CF {r['CF_pairs']:.3f} | second-order (triples) CF {r['CF_triples']:.3f}, gamma {r['gamma']},"
              f" P(even parity) in [{r['parity_lo']:.3f}, {r['parity_hi']:.3f}] ({r['mode']})")
    first = min(r["lam"] for r in rows if r["CF_triples"] > 1e-9); g0 = min(r["lam"] for r in rows if r["gamma"])
    print(f"    second-order CF > 0 from lambda = {first:.2f}; gamma from {g0:.2f}; first-order CF = 0 at every lambda: {all(r['CF_pairs'] < 1e-9 for r in rows)}")
    return rows


def mbic_units():
    import mbic
    d = mbic.load(); g = pd.cut(d["political_ideology"], [-11, -5, 0, 5, 11], labels=list("abcd"))
    S = d.assign(g=g).groupby(["sentence_id", "g"], observed=True)["b"].mean().unstack().dropna()
    return (S[list("abcd")].to_numpy() >= 0.5).astype(int)


def tet_model(parts, smooth=0.5):
    tabs = {}
    for C, T in parts.items():
        idx = [M4.index(m) for m in C]; t = np.zeros(8)
        for x in T:
            t[4 * x[idx[0]] + 2 * x[idx[1]] + x[idx[2]]] += 1
        tabs[C] = (t + smooth) / (t + smooth).sum()
    return EmpiricalModel(TET, tabs)


def part2(T, n_rep=100, B=30):
    print(f"\n(2) MBIC four ideology groups: {len(T)} sentences rated by all four; {n_rep} random 5-way splits")
    print("    identification <f, chi_4>: " + ", ".join(f"{k} {F[k] @ CHI4:+.0f}" for k in F))
    res = {k: dict(cov=0, cov_boot=0, cov_pred=0, width=[], width_boot=[], err=[]) for k in F}; cfs, gams = [], []
    inst = None
    for rep in range(n_rep):
        idx = rng.permutation(len(T)); P = np.array_split(idx, 5)
        parts = {C: T[P[i]] for i, C in enumerate(TET.contexts)}; truth = T[P[4]]
        e = tet_model(parts); cfs.append(contextual_fraction(e).value)
        gams.append(analyse_outcomes(tet_model(parts, smooth=0.0), eps=1e-9, with_cf=False).gamma_h1_nonzero)
        if rep == 0:
            inst = complex_instrument(e, labels=["abc", "abd", "acd", "bcd"])
        boots = [tet_model({C: v[rng.integers(len(v), size=len(v))] for C, v in parts.items()}) for _ in range(B)]
        for k, f in F.items():
            b = outcome_bounds(e, f); tr = TRUTH[k](truth); r = res[k]
            r["cov"] += b["lo"] - 1e-9 <= tr <= b["hi"] + 1e-9; r["width"].append(b["hi"] - b["lo"]); r["err"].append(abs(b["point"] - tr))
            bb = [outcome_bounds(m, f) for m in boots]
            lo, hi = np.quantile([x["lo"] for x in bb], 0.025), np.quantile([x["hi"] for x in bb], 0.975)
            z = 1.96 * np.sqrt(max(tr * (1 - tr), 1e-12) / len(truth))
            r["cov_boot"] += lo - 1e-9 <= tr <= hi + 1e-9; r["cov_pred"] += lo - z - 1e-9 <= tr <= hi + z + 1e-9; r["width_boot"].append(hi - lo)
    out = {}
    for k, r in res.items():
        out[k] = dict(identified=bool(abs(F[k] @ CHI4) < 1e-12), coverage=r["cov"] / n_rep, coverage_boot=r["cov_boot"] / n_rep, coverage_pred=r["cov_pred"] / n_rep,
                      width=float(np.median(r["width"])), width_boot=float(np.median(r["width_boot"])), maxent_err=float(np.mean(r["err"])))
        o = out[k]
        print(f"    {k:15s} {'identified' if o['identified'] else 'free      '}  width {o['width']:.3f} (boot {o['width_boot']:.3f})"
              f"  coverage plug-in {o['coverage']:.2f}, boot {o['coverage_boot']:.2f}, + truth noise {o['coverage_pred']:.2f}  max-ent error {o['maxent_err']:.3f}")
    full = tet_model({C: T for C in TET.contexts})
    pairs_full = EmpiricalModel(PAIRS, {P: TET.restriction_matrix(next(C for C in TET.contexts if set(P) <= set(C)), P) @ full.tables[next(C for C in TET.contexts if set(P) <= set(C))] for P in PAIRS.contexts})
    print(f"    CF of the split tables: median {np.median(cfs):.3f} (max {np.max(cfs):.3f}); gamma != 0 in {np.mean(gams):.2f} of splits;"
          f"  same sentences for every context: CF {contextual_fraction(full).value:.3f}, pairs-only CF {contextual_fraction(pairs_full).value:.3f}")
    return dict(results=out, cf_median=float(np.median(cfs)), cf_max=float(np.max(cfs)), gamma_frac=float(np.mean(gams)), instrument=inst, n=int(len(T)))


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
    fig = plt.figure(figsize=(11, 6.6)); gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    a0, a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]); b0 = fig.add_subplot(gs[1, :2]); b1 = fig.add_subplot(gs[1, 2])
    tri = EmpiricalModel(Scenario({"L": (0, 1), "C": (0, 1), "R": (0, 1)}, (("L", "C"), ("C", "R"), ("L", "R"))),
                         {C: np.array([0.5, 0.0, 0.0, 0.5]) if C != ("L", "R") else np.array([0.0, 0.5, 0.5, 0.0]) for C in (("L", "C"), ("C", "R"), ("L", "R"))})
    plot_complex(complex_instrument(tri, labels=["LC", "CR", "LR"]), a0, "(a) triangle cover, contradictory pairs: hollow")
    pr = {C: np.array([1.0 if sum(s) % 2 == 0 else 0.0 for s in TET.sections(C)]) / 4 for C in TET.contexts}
    plot_complex(complex_instrument(EmpiricalModel(TET, pr), labels=["abc", "abd", "acd", "bcd"]), a1, "(b) tetrahedral cover, higher PR box")
    plot_complex(OUT["mbic"]["instrument"], a2, "(c) tetrahedral cover, MBIC (one split)")
    for a in (b0, b1):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7.5)
    R = OUT["higher_pr"]; L = [r["lam"] for r in R]
    b0.plot(L, [r["CF_pairs"] for r in R], color=muted, lw=1.8, label="first order: pair contexts (any loop / pairwise method)")
    b0.plot(L, [r["CF_triples"] for r in R], color=blue, lw=2, label="second order: triple contexts (nerve = sphere)")
    g = [r["lam"] for r in R if r["gamma"]]
    if g:
        b0.scatter(g, [r["CF_triples"] for r in R if r["gamma"]], color=orange, s=30, zorder=4, label="γ ≠ 0 (only at λ = 1)")
    b0.set_xlabel("λ (weight of the higher PR box)", fontsize=8, color=muted); b0.set_ylabel("contextual fraction", fontsize=8, color=muted)
    b0.set_title("(d) the obstruction is invisible to first-order (pairwise) views", fontsize=8.5, loc="left", color=ink); b0.legend(fontsize=7, frameon=False, loc="upper left")
    res = OUT["mbic"]["results"]; ks = list(res); y = np.arange(len(ks))[::-1]
    for yi, k in zip(y, ks):
        r = res[k]; b1.barh(yi, r["width_boot"], left=0, color=blue if not r["identified"] else orange, alpha=0.6, height=0.5)
        b1.text(r["width_boot"] + 0.01, yi, f"cov {r['coverage_pred']:.2f}", fontsize=7, va="center", color=ink)
    b1.set_yticks(y); b1.set_yticklabels([k + (" (identified)" if res[k]["identified"] else "") for k in ks], fontsize=7)
    b1.set_xlabel("bootstrap width of the bounds", fontsize=8, color=muted); b1.set_xlim(0, max(r["width_boot"] for r in res.values()) * 1.5)
    b1.set_title("(e) MBIC 4-group outcomes: width, coverage", fontsize=8.5, loc="left", color=ink)
    fig.text(0.01, 0.005, "Nerve drawings are representational: vertices = contexts (size = share of the Bell certificate), edges = overlaps (width = signalling),\n"
             "shaded triangles = 2-cells (triple overlaps, labelled by the shared measurement; the outer face of (b, c) is the fourth 2-cell).", fontsize=6.5, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(here, "higher_chart.pdf")); fig.savefig(os.path.join(here, "higher_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    OUT = dict(covers=part0(), higher_pr=part1(), mbic=part2(mbic_units()))
    json.dump(OUT, open(os.path.join(here, "results.json"), "w"), indent=1, default=float)
    plot(OUT)
