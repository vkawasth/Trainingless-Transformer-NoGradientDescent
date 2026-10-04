"""AMB toggle radii on MBIC and AllSides: how much evidence must vanish to make a local outcome impossible?

Tables: source x topic x label.  MBIC: outlet x topic x (biased / not) by sentence majority.
AllSides: side x topic group x (adversarial / not), whole period and 26 weeks before / 19 after COVID.
Models: SATURATED (every cell free) vs GLUED = no-three-way log-linear model (source-topic, source-label, topic-label
margins; the model in which all loops vanish). For each cell c the AMB outcome "label y is possible in context (s,t)"
toggles when c vanishes; the glued model forces the co-facial set F(c) to vanish with it.
Reported in SENTENCES / ARTICLES: N * p(F) (upper bound, achievable) vs N * p_c (saturated = lower bound).
Data via MBIC_XLSX and ALLSIDES_DIR.
"""
import os, sys, json, datetime
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../bias3"), os.path.join(here, "../allsides")]
import numpy as np
from amb_vigneaux.strata import design, ipf, toggle_radius_amb, is_facial, exact_toggle

MARG = [(0, 1), (0, 2), (1, 2)]


def analyse(N, names, label, top=6):
    M = ipf(N, MARG); p = (M / M.sum()).ravel(); n = N.sum()
    A, cells = design(N.shape, MARG)
    ctx = N.sum(2).ravel()                                     # empty source x topic contexts are structural zeros
    live = [c for c in range(len(cells)) if N.sum(2)[cells[c][0], cells[c][1]] > 0]
    mle_interior = bool(min(M.ravel()[c] for c in live) > 1e-6)
    rows = []
    def kind(F, c):
        s0, t0, y0 = cells[c]; cs = [cells[j] for j in F]
        if all(x[:2] == (s0, t0) for x in cs): return "context"
        if all(x[1] == t0 and x[2] == y0 for x in cs): return "topic slice"
        if all(x[0] == s0 and x[2] == y0 for x in cs): return "source slice"
        return "mixed (loop-shaped)"
    for c in live:
        e = exact_toggle(A, p, c)
        F = [j for j in e["forced"] if j in set(live)] if e else toggle_radius_amb(A, p, c)["forced"]
        mass = float(p[F].sum())
        rows.append(dict(cell=[names[a][cells[c][a]] for a in range(3)], obs=int(N.ravel()[c]), fitted=float(M.ravel()[c]),
                         sat=float(n * p[c]), glued=float(n * mass), n_forced=len(F), optimal=bool(e and e["optimal"]),
                         kind=kind(F, c), forced=[[names[a][cells[j][a]] for a in range(3)] for j in F],
                         forced_contexts=sorted({f"{names[0][cells[j][0]]}|{names[1][cells[j][1]]}" for j in F})))
    ratio = np.array([r["glued"] / r["sat"] for r in rows])
    rows.sort(key=lambda r: r["glued"])
    print(f"\n== {label}: {int(n)} units, table {N.shape}, {len(live)} live cells "
          f"({len(cells) - len(live)} in empty contexts), glued-model MLE interior on live cells: {mle_interior}")
    print(f"   evidence that must vanish to toggle one cell: saturated (the cell alone) vs glued (its co-facial set)")
    print(f"   median glued/saturated = {np.median(ratio):.1f}x   (range {ratio.min():.1f}-{ratio.max():.1f}x);"
          f"  exact (MILP optimal) for {sum(r['optimal'] for r in rows)}/{len(rows)} cells")
    import collections
    kinds = collections.Counter(r["kind"] for r in rows)
    print("   cheapest toggle is: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    for r in rows[:top]:
        print(f"   {'/'.join(map(str, r['cell'])):48s} obs {r['obs']:4d}  saturated {r['sat']:6.1f}  glued {r['glued']:7.1f}"
              f"  [{r['kind']}: {'; '.join('/'.join(map(str, x)) for x in r['forced'][:4])}{' ...' if len(r['forced']) > 4 else ''}]")
    return dict(n=float(n), mle_interior=mle_interior, median_ratio=float(np.median(ratio)), min_ratio=float(ratio.min()),
                max_ratio=float(ratio.max()), kinds=dict(kinds), nearest=rows[:top])


if __name__ == "__main__":
    R = {}
    import mbic
    d = mbic.load()
    s = d.groupby("sentence_id").agg(y=("b", "mean"), o=("outlet", "first"), t=("topic", "first"))
    s["lab"] = np.where(s.y >= 0.5, "biased", "not")
    O, T, Y = sorted(s.o.unique()), sorted(s.t.unique()), ["biased", "not"]
    N = np.zeros((len(O), len(T), 2))
    for o, t, l in s[["o", "t", "lab"]].itertuples(index=False):
        N[O.index(o), T.index(t), Y.index(l)] += 1
    R["mbic"] = analyse(N, [O, T, Y], "MBIC outlet x topic x majority label")

    import load
    rows = [r for r in load.load() if r["group"] != "other"]
    G = sorted({r["group"] for r in rows if r["group"] != "coronavirus"}); S = ["left", "center", "right"]; Y2 = ["adversarial", "not"]
    def table(rs, groups):
        N = np.zeros((3, len(groups), 2))
        for r in rs:
            if r["group"] in groups:
                N[S.index(r["side"]), groups.index(r["group"]), 0 if r["outcome"] else 1] += 1
        return N
    R["allsides_all"] = analyse(table(rows, G), [S, G, Y2], "AllSides side x topic group x adversarial, 2015-2020")
    tau = datetime.date(2020, 3, 13); nb = datetime.timedelta(weeks=26)
    pre = [r for r in rows if tau - nb <= r["date"] < tau]; post = [r for r in rows if r["date"] >= tau]
    Gc = [g for g in G if all(sum(1 for r in rr if r["group"] == g and r["side"] == sd) >= 10 for rr in (pre, post) for sd in S)]
    for lab, rr in (("before COVID (26 wk)", pre), ("after COVID (19 wk)", post)):
        Nt = table(rr, Gc); key = f"allsides_{lab.split()[0]}"
        R[key] = analyse(Nt, [S, Gc, Y2], f"AllSides {lab}, groups {Gc}", top=4)
        # the adversarial cells of the centre: how close to 'the centre is never adversarial on topic t'?
        M = ipf(Nt, MARG); p = (M / M.sum()).ravel(); A, cells = design(Nt.shape, MARG)
        cen = {}
        for j, g in enumerate(Gc):
            c = cells.index((1, j, 0)); r = exact_toggle(A, p, c); cen[g] = float(Nt.sum() * r["mass"])
        R[key]["centre_adversarial_toggle"] = cen
        print("   centre 'adversarial possible' toggle (articles, glued; % of window): " +
              "  ".join(f"{g} {v:.0f} ({100 * v / Nt.sum():.1f}%)" for g, v in cen.items()))
    json.dump(R, open(os.path.join(here, "amb_toggle_results.json"), "w"), indent=1, default=str)
