"""Tree-space tests for coalition changes (BHV spider T_3; amb_vigneaux/treespace.py).

MBIC: each sentence with left / centre / right annotators -> a 3-leaf tree of the group bias rates.
AllSides: each story covered by left / center / right -> a tree of the tone of Trump sentences.
Reported: the sticky Frechet mean (does a population coalition exist?), its label-permutation null (MBIC),
and energy-distance permutation tests between outlet types, article years and before/after events.
Data via MBIC_XLSX and ALLSIDES_DIR.
"""
import os, sys, json, datetime, collections
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../bias3"), os.path.join(here, "../allsides")]
import numpy as np
from amb_vigneaux.treespace import spider_point, frechet_mean, energy_test

rng = np.random.default_rng(0)
LEGS = {0: "L-C", 1: "L-R", 2: "C-R", -1: "star"}


def fm_str(pts):
    (a, v), m = frechet_mean(pts)
    return f"Frechet mean on {LEGS[a]} at {v:.4f} (leg moments L-C {m[0]:.4f}, L-R {m[1]:.4f}, C-R {m[2]:.4f})", (a, v, m.tolist())


if __name__ == "__main__":
    R = {}
    import mbic
    d = mbic.load(); pi = d.political_ideology
    d["grp"] = np.where(pi <= -4, "left", np.where(pi >= 3, "right", "centre"))
    d["yr"] = d.news_link.str.extract(r"/(20\d\d)/")[0]
    units = []
    for _, g in d.groupby("sentence_id"):
        if all((g.grp == x).any() for x in ("left", "centre", "right")):
            units.append((g.b.to_numpy(), g.grp.to_numpy(), g.type.iloc[0], g.yr.iloc[0]))
    pt = lambda b, gr: spider_point([b[gr == x].mean() for x in ("left", "centre", "right")])
    P = [pt(b, gr) for b, gr, _, _ in units]
    s, info = fm_str(P); print(f"== MBIC ({len(P)} sentences): {s}")
    null = []
    for _ in range(500):
        Q = [pt(b, rng.permutation(gr)) for b, gr, _, _ in units]
        (a, v), m = frechet_mean(Q); null.append((a, v, m))
    nm = np.array([x[2] for x in null])
    p_leg = [(nm[:, k] >= info[2][k]).mean() for k in range(3)]
    print(f"   label-permutation null: leg moments mean L-C {nm[:, 0].mean():.4f}, L-R {nm[:, 1].mean():.4f}, C-R {nm[:, 2].mean():.4f};"
          f"  p(moment >= observed): L-C {p_leg[0]:.3f}, L-R {p_leg[1]:.3f}, C-R {p_leg[2]:.3f};"
          f"  null means off the origin in {np.mean([x[0] >= 0 for x in null]):.2f} of permutations")
    R["mbic"] = dict(n=len(P), mean=info, null_moment_mean=nm.mean(0).tolist(), p_moment=p_leg)
    for lab, fa, fb in (("left vs right outlets", lambda u: u[2] == "left", lambda u: u[2] == "right"),
                        ("articles <= 2019 vs 2020", lambda u: isinstance(u[3], str) and u[3] <= "2019", lambda u: u[3] == "2020")):
        X = [pt(u[0], u[1]) for u in units if fa(u)]; Y = [pt(u[0], u[1]) for u in units if fb(u)]
        e = energy_test(X, Y, B=999); R[f"mbic_{lab}"] = dict(e, mean_x=fm_str(X)[1], mean_y=fm_str(Y)[1])
        print(f"   {lab}: energy {e['energy']:.4f}, p {e['p']:.3f} (n {e['n_x']}/{e['n_y']});  {fm_str(X)[0]}  |  {fm_str(Y)[0]}")

    import load
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load.load():
        by[(r["date"], r["topic"])][r["side"]].append(r["tone"])
    stories = [(k[0], spider_point([np.mean(g[s]) for s in ("left", "center", "right")]))
               for k, g in by.items() if all(s in g for s in ("left", "center", "right"))]
    Pall = [p for _, p in stories]
    s, info = fm_str(Pall); print(f"\n== AllSides ({len(Pall)} stories, tone): {s}"); R["allsides"] = dict(n=len(Pall), mean=info)
    for name, cut in (("election 2016", datetime.date(2016, 11, 8)), ("COVID emergency", datetime.date(2020, 3, 13))):
        X = [p for dt, p in stories if dt < cut]; Y = [p for dt, p in stories if dt >= cut]
        e = energy_test(X, Y, B=999); R[f"allsides_{name}"] = dict(e, mean_x=fm_str(X)[1], mean_y=fm_str(Y)[1])
        print(f"   before/after {name}: energy {e['energy']:.4f}, p {e['p']:.3f} (n {e['n_x']}/{e['n_y']});  {fm_str(X)[0]}  |  {fm_str(Y)[0]}")
    json.dump(R, open(os.path.join(here, "treespace_results.json"), "w"), indent=1, default=str)
