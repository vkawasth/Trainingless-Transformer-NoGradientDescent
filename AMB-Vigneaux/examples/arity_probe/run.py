"""Probing inside arity: coalitions of filled cells (hull trees) on real data, and the p-adic radius profile.

Inside a filled cell (one unit, k sources) all loop sums vanish, but the configuration of the k values does not.
Its first merge -- the COALITION, the two sources that agree first -- is tested against an additive null
(unit effect + source lean + residuals permuted within source), so lean alone cannot produce it.
  AllSides  story groups covered by left, center and right (tone of Trump sentences, VADER); by period.
  MBIC      sentences labelled by left, centre and right annotators (bias rate per group; cut <= -4 / >= 3).
  NewsWCL50 event x target cells covered by all five outlets (favourability log-odds, >= 3 codes each).
  p-adic    order profile e(m) of the obstruction class of sum claims: piecewise linear, integer slopes, breakpoint m*.
Data via ALLSIDES_DIR, MBIC_XLSX, NEWSWCL50_CSV (not redistributed).
"""
import os, sys, json, collections, datetime
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../bias3"), os.path.join(here, "../allsides")]
import numpy as np
from amb_vigneaux.hull import coalition_test, coalition_test_binomial, coalition_test_labelperm, obstruction_profile, coalition
from amb_vigneaux.padic_loops import sum_claim_depth

rng = np.random.default_rng(0)


def allsides_cells():
    import load
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load.load():
        by[(r["date"], r["topic"])][r["side"]].append(r["tone"])
    cells = []
    for (d, t), g in by.items():
        if all(s in g for s in ("left", "center", "right")):
            c = {s: float(np.mean(g[s])) for s in ("left", "center", "right")}; c["_date"] = d; cells.append(c)
    return cells


def period_shift(cells, cut, sources, B=2000):
    """share of each coalition before vs after `cut`; permutation of period labels."""
    lab = np.array([c["_date"] >= cut for c in cells])
    co = []
    for c in cells:
        cc = coalition([c[s] for s in sources]); co.append(f"{sources[cc[0]]}-{sources[cc[1]]}" if cc else "tie")
    co = np.array(co); keys = sorted(set(co)); out = {}
    for k in keys:
        x = (co == k).astype(float); d = x[lab].mean() - x[~lab].mean()
        null = np.array([(lambda l: x[l].mean() - x[~l].mean())(rng.permutation(lab)) for _ in range(B)])
        out[k] = dict(before=float(x[~lab].mean()), after=float(x[lab].mean()), p=float((np.abs(null) >= abs(d)).mean()))
    return dict(n_before=int((~lab).sum()), n_after=int(lab.sum()), shares=out)


def mbic_cells():
    import mbic
    d = mbic.load()
    d["grp"] = np.where(d.political_ideology <= -4, "left", np.where(d.political_ideology >= 3, "right", "centre"))
    k = d.groupby(["sentence_id", "grp"])["b"].sum().unstack(); n = d.groupby(["sentence_id", "grp"])["b"].count().unstack()
    ok = n[["left", "centre", "right"]].notna().all(axis=1)
    return [{g: (int(k.loc[i, g]), int(n.loc[i, g])) for g in ("left", "centre", "right")} for i in n.index[ok]]


def mbic_rater_cells():
    import mbic
    d = mbic.load()
    d["grp"] = np.where(d.political_ideology <= -4, "left", np.where(d.political_ideology >= 3, "right", "centre"))
    out = []
    for _, g in d.groupby("sentence_id"):
        if all((g.grp == x).any() for x in ("left", "centre", "right")):
            out.append((g.b.to_numpy(), g.grp.to_numpy(), g.type.iloc[0]))
    return out


def newswcl_cells():
    import newswcl50 as N
    p = N.load(); cells = []
    for (e, t), g in p.groupby(["event_id", "tgt"]):
        c = {}
        for o in N.OUT:
            s = g[g.publisher_id == o]
            if len(s) >= 3:
                c[o] = float(np.log((s.pos.sum() + 0.5) / ((1 - s.pos).sum() + 0.5)))
        if len(c) == 5:
            cells.append(c)
    return cells


if __name__ == "__main__":
    R = {}
    print("== AllSides: story groups covered by L, C, R; tone of Trump sentences")
    A = allsides_cells(); S3 = ["left", "center", "right"]
    r = coalition_test([{s: c[s] for s in S3} for c in A], S3); R["allsides"] = r
    print(f"   n = {r['n']} cells; lean {r['lean']}")
    for k, v in r["coalitions"].items():
        print(f"   {k:28s} obs {v['obs']:4d}  additive null {v['null']:7.1f}  p {v['p']:.3f}")
    for name, cut in (("election 2016", datetime.date(2016, 11, 8)), ("COVID emergency", datetime.date(2020, 3, 13))):
        sh = period_shift(A, cut, S3); R[f"allsides_shift_{name}"] = sh
        print(f"   coalition shares before/after {name} (n {sh['n_before']}/{sh['n_after']}): " +
              "  ".join(f"{k} {v['before']:.2f}->{v['after']:.2f} (p {v['p']:.2f})" for k, v in sh["shares"].items()))

    print("\n== MBIC: sentences labelled by left / centre / right annotators (bias rate per annotator group)")
    M = mbic_cells(); S3m = ["left", "centre", "right"]
    MC = mbic_rater_cells()
    r = coalition_test_labelperm([(v, g) for v, g, _ in MC], S3m); R["mbic_labelperm"] = r
    print(f"   n = {r['n']} sentences; null 1: permute annotator groups within each sentence (keeps ties, sizes, rater spread)")
    for k, v in r["coalitions"].items():
        print(f"   {k:28s} obs {v['obs']:4d}  null {v['null']:7.1f}  p {v['p']:.3f}")
    for ty in ("left", "center", "right"):
        rr = coalition_test_labelperm([(v, g) for v, g, t in MC if t == ty], S3m, seed=1); R[f"mbic_labelperm_{ty}_outlets"] = rr
        print(f"   sentences from {ty} outlets (n {rr['n']}): " + "  ".join(f"{k} {v['obs']} vs {v['null']:.0f} (p {v['p']:.3f})"
              for k, v in rr["coalitions"].items() if k != "tie"))
    r = coalition_test_binomial(M, S3m); R["mbic_binomial"] = r
    print(f"   null 2: binomial with lean {r['lean']} (no rater heterogeneity, so it makes too many ties; shown for comparison)")
    for k, v in r["coalitions"].items():
        print(f"   {k:28s} obs {v['obs']:4d}  null {v['null']:7.1f}  p {v['p']:.3f}")

    print("\n== NewsWCL50: event x target cells covered by all five outlets (favourability log-odds)")
    W = newswcl_cells(); S5 = ["LL", "L", "M", "R", "RR"]
    r = coalition_test(W, S5, B=2000); R["newswcl50"] = r
    print(f"   n = {r['n']} cells; lean {r['lean']}")
    for k, v in r["coalitions"].items():
        if v["obs"] or v["null"] > 1:
            print(f"   {k:28s} obs {v['obs']:4d}  additive null {v['null']:7.1f}  p {v['p']:.3f}")

    print("\n== p-adic: order profile e(m) of the obstruction class of sum claims (p = 3, precision 3^6)")
    x = [5, 11, 40, 7, 23]; E = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]          # K_{2,3}: two even loops
    for err in ({}, {(1, 4): 81}, {(0, 2): 9, (1, 4): 81}, {(0, 2): 9 * 2}):
        claims = [(i, j, x[i] + x[j] + err.get((i, j), 0)) for i, j in E]
        prof = obstruction_profile(5, claims, 3, 6); ms = sum_claim_depth(5, claims, 3, 6)
        print(f"   errors {err or 'none'}: e(m) = {[o for _, o in prof]}  (m* = {ms}; the local hulls glue at radius 3^-{ms})")
    R["padic_profile"] = dict(profile=prof, m_star=ms)
    json.dump(R, open(os.path.join(here, "arity_probe_results.json"), "w"), indent=1, default=str)
