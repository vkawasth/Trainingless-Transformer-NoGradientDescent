"""Component [D]: deformations of the toric model at its boundary (amb_vigneaux/deform.py).

(1) Boundary supports: which zero sets occur in the closure of the no-top-interaction model (n = 3 all 256; n = 4 up to
    four zeros), by (#even, #odd) zeros.
(2) Local structure by (a, b) = (#even, #odd) zeros: smooth or singular, Tjurina number of x1..xa = y1..yb (Groebner
    basis), dimension of the singular locus, transversal type at a generic singular point; T^2 = 0 (hypersurface);
    equisingularity of the theta-level sets (Tjurina number of x1 x2 = e^c y1 y2 for several c).
(3) The flop: near a node (two even and two odd cells small) the cheapest odd partner of an even cell flips under tiny
    perturbations; near a smooth boundary point (one even, one odd small) it does not. Exact MILP toggles confirm
    that the forced set is {cheapest even, cheapest odd}.
(4) Real tables (MBIC 3 groups, AllSides 3 sides, MBIC 4 groups): toggle radius vs node radius, partner margin, and
    how often the toggle's forced pair changes under multinomial resampling of the table (the flop under sampling).
Env: MBIC_XLSX, ALLSIDES_DIR.
"""
import os, sys, json, io, contextlib
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../toric")]
import numpy as np
from amb_vigneaux import deform as D, toric as T
from amb_vigneaux.strata import design, exact_toggle

rng = np.random.default_rng(0)


def part1():
    print("(1) boundary supports (zero sets in the closure of the model), by (#even, #odd) zeros")
    out = {}
    for n, mz in ((3, None), (4, 4)):
        r = D.boundary_supports(n, mz)
        ok = all((v["facial"] > 0) == (a >= 1 and b >= 1) and (v["nonfacial"] == 0) == (a >= 1 and b >= 1) for (a, b), v in r.items())
        out[n] = dict(table={f"{a},{b}": v for (a, b), v in r.items()}, rule_holds=bool(ok))
        print(f"    n={n}" + (" (up to 4 zeros)" if mz else " (all subsets)") + f": occurs on the boundary iff it meets both parity classes: {ok}")
    return out


def part2():
    print("\n(2) local structure at a boundary point with a even and b odd zeros (f ~ x1..xa - y1..yb)")
    rows = []
    for a, b in ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (2, 4)):
        tj = D.tjurina(a, b) if a + b <= 6 else None
        trans = D.tjurina(2, 2) if D.singular(a, b) else 0
        rows.append(dict(a=a, b=b, singular=D.singular(a, b), tjurina=tj, sing_dim=D.singular_locus_dim(a, b), transversal_tjurina=trans))
        print(f"    (a,b)=({a},{b}): {'singular' if D.singular(a, b) else 'smooth  '}  Tjurina {tj if tj is not None else 'inf (non-isolated)'}"
              f"  singular-locus dim {D.singular_locus_dim(a, b)}  transversal type {'node (Tjurina 1)' if D.singular(a, b) else 'smooth'}")
    eq = {c: D.tjurina(2, 2, c=c) for c in (0, 1, -2)}
    print(f"    T^2 = 0 at every point (hypersurface). theta-level sets x1 x2 = e^c y1 y2: Tjurina {eq} -> equisingular")
    return dict(rows=rows, equisingular=eq)


def near(n, kind, delta=0.004):
    ev, od = D.parity_classes(n); p = rng.dirichlet(np.ones(2 ** n) * 5)
    small = [ev[0], od[0]] + ([ev[1], od[1]] if kind == "node" else [])
    for i in small:
        p[i] = delta * (1 + 0.02 * rng.normal())
    return p / p.sum()


def part3():
    print("\n(3) the flop: does the toggle's odd partner flip under perturbation?")
    out = {}
    A, _ = design((2,) * 3, T.margins(3))
    for kind in ("smooth", "node"):
        p = near(3, kind)
        with contextlib.redirect_stdout(io.StringIO()):
            ev, od = D.parity_classes(3); e = min(ev, key=lambda i: p[i]); r = exact_toggle(A, p, e)
        partner = sorted(set(r["forced"]) - {e})
        rates = {eps: D.flop_switch_rate(p, eps, rng=rng) for eps in (0.01, 0.05, 0.2)}
        out[kind] = dict(milp_forced=[T.cells(3)[i] for i in r["forced"]], cheapest_odd=T.cells(3)[min(od, key=lambda i: p[i])],
                         partner_margin=D.radii(p)["partner_margin"], switch=rates)
        print(f"    near a {kind} boundary point: MILP forced set {out[kind]['milp_forced']} (cheapest odd {out[kind]['cheapest_odd']});"
              f" partner margin {out[kind]['partner_margin']:.3f}; switch rate at eps 0.01/0.05/0.2: " + " / ".join(f"{v:.2f}" for v in rates.values()))
    return out


def part4():
    import run as TR                              # examples/toric/run.py helpers
    m3, m4 = TR.mbic_tables(); tables = {"MBIC 3 groups": m3, "AllSides 3 sides": TR.allsides_table(), "MBIC 4 groups": m4}
    print("\n(4) real tables")
    out = {}
    for name, Tm in tables.items():
        c = TR.counts_from(Tm); N = int(c.sum()); p = (c + .5) / (c + .5).sum(); n = int(np.log2(len(p))); ev, od = D.parity_classes(n)
        r = D.radii(p); e0 = min(ev, key=lambda i: p[i]); o0 = min(od, key=lambda i: p[i])
        sw = 0; B = 500
        for _ in range(B):
            cb = rng.multinomial(N, c / N); pb = (cb + .5) / (cb + .5).sum()
            sw += (min(ev, key=lambda i: pb[i]), min(od, key=lambda i: pb[i])) != (e0, o0)
        out[name] = dict(r, resample_switch=sw / B, forced_pair=[T.cells(n)[e0], T.cells(n)[o0]])
        print(f"    {name}: toggle pair {out[name]['forced_pair']} mass {r['toggle_mass']:.3f} (radius {r['toggle_radius']:.3f}); node mass {r['node_mass']:.3f}"
              f" (radius {r['node_radius']:.3f}); partner margin {r['partner_margin']:.2f}; forced pair changes in {sw / B:.2f} of resamples")
    return out


def plot(OUT):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, blue, orange, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#b8b6ae"
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=[1, 1.1, 1.1]))
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=muted, labelsize=7.5)
    # (a) the (a,b) grid: boundary / smooth / node / deeper
    for a_ in range(0, 5):
        for b_ in range(0, 5):
            if a_ == 0 and b_ == 0:
                continue
            if a_ == 0 or b_ == 0:
                col, lab = "white", "×"
            elif min(a_, b_) == 1:
                col, lab = "#cfe0f5", "s"
            elif a_ == 2 and b_ == 2:
                col, lab = orange, "node"
            else:
                col, lab = "#f6c9b4", f"{D.singular_locus_dim(a_, b_)}"
            ax[0].add_patch(plt.Rectangle((a_ - .5, b_ - .5), 1, 1, facecolor=col, edgecolor="#d8d6cf"))
            ax[0].text(a_, b_, lab, ha="center", va="center", fontsize=7, color=ink)
    ax[0].set_xlim(-.5, 4.5); ax[0].set_ylim(-.5, 4.5); ax[0].set_xlabel("# even-parity zeros", fontsize=8, color=muted); ax[0].set_ylabel("# odd-parity zeros", fontsize=8, color=muted)
    ax[0].set_title("(a) boundary of [T] (n=3): × not a support,\ns smooth, node = conifold, k = singular-locus dim", fontsize=8, loc="left", color=ink)
    ax[0].set_aspect("equal")
    # (b) flop switch rates
    F = OUT["flop"]; eps = [0.01, 0.05, 0.2]
    for kind, col in (("smooth", blue), ("node", orange)):
        ax[1].plot(eps, [F[kind]["switch"][str(e)] if str(e) in F[kind]["switch"] else F[kind]["switch"][e] for e in eps], "-o", color=col, lw=1.8, label=f"near a {kind} point")
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 1); ax[1].set_xlabel("perturbation size ε (log-scale noise)", fontsize=8, color=muted)
    ax[1].set_ylabel("rate the forced partner flips", fontsize=8, color=muted); ax[1].legend(fontsize=7, frameon=False, loc="upper left")
    ax[1].set_title("(b) the flop: support toggle near a node is unstable", fontsize=8.5, loc="left", color=ink)
    # (c) real tables: toggle vs node radius, resample switch
    R = OUT["real"]; names = list(R); y = np.arange(len(names))[::-1]
    for yi, nm in zip(y, names):
        r = R[nm]; ax[2].plot([r["toggle_radius"], r["node_radius"]], [yi, yi], color=grey, lw=1)
        ax[2].scatter([r["toggle_radius"]], [yi], color=blue, s=22, zorder=3); ax[2].scatter([r["node_radius"]], [yi], color=orange, s=22, zorder=3)
        ax[2].text((r["toggle_radius"] + r["node_radius"]) / 2, yi + 0.15, f"pair flips in {r['resample_switch']:.0%} of resamples", fontsize=6.5, color=ink, ha="center")
    ax[2].scatter([], [], color=blue, s=22, label="toggle radius (smooth boundary)"); ax[2].scatter([], [], color=orange, s=22, label="node radius")
    ax[2].set_yticks(y); ax[2].set_yticklabels(names, fontsize=7.5); ax[2].set_xlabel("−log(1 − mass) to reach the stratum", fontsize=8, color=muted)
    ax[2].legend(fontsize=6.5, frameon=False, loc="lower right"); ax[2].set_xlim(0, 0.36); ax[2].set_ylim(-0.5, len(names) - 0.3); ax[2].set_title("(c) real tables: boundary and node", fontsize=8.5, loc="left", color=ink)
    fig.tight_layout(); fig.savefig(os.path.join(here, "deform_chart.pdf")); fig.savefig(os.path.join(here, "deform_chart.png"), dpi=150)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    plot(json.load(open(os.path.join(here, "results.json"))))
elif __name__ == "__main__":
    OUT = dict(boundary=part1(), local=part2(), flop=part3(), real=part4())
    json.dump(OUT, open(os.path.join(here, "results.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    plot(json.loads(json.dumps(OUT, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))))
