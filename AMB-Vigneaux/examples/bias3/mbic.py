"""MBIC (Spinde et al. 2021): 1,700 sentences, 8 outlets, 14 topics, ~10 crowd labels per sentence,
each label carrying the annotator's self-reported political ideology (-10 .. +10).

Theorem (coboundary). If every comparison is a difference of ONE global assignment on the SAME units,
d = delta b, then every loop sum is (delta delta b) = 0 identically. We check this: annotator groups
compared on the same sentences close every loop exactly. A global failure needs different units per region.

Test A  outlet x topic (different sentences per outlet)
        y_s = fraction of annotators calling sentence s biased.  Additive model  y = u(outlet) + v(topic) + e.
        Loop residuals on the outlet-topic bipartite graph = interaction. Global test: interaction F in the
        unbalanced two-way model, Freedman-Lane permutation of residuals under the additive model.
        Local regions: every pair of topics (4-cycles over all outlets); the whole cover.
Test B  annotator ideology x outlet type (hostile-media loop)
        per sentence, d_s = b(left annotators, s) - b(right annotators, s)   (paired: same sentence)
        Phi = mean d over right-leaning outlets - mean d over left-leaning outlets
        = loop  L-annotators - right-outlets - R-annotators - left-outlets. Exact-conditional permutation of
        outlet-type labels across sentences (d fixed).
"""
import itertools, json
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
RAW = "/tmp/claude-0/BABE/data/raw_labels_MBIC.xlsx"


def load():
    d = pd.read_excel(RAW)
    d["b"] = (d["label_bias"] == "Biased").astype(float)
    return d


# ---------------------------------------------------------------- theorem check
def theorem_check(d):
    g = pd.cut(d["political_ideology"], [-11, -4, 3, 11], labels=["L", "C", "R"])
    S = d.assign(g=g).groupby(["sentence_id", "g"], observed=True)["b"].mean().unstack()
    same = S.dropna()                                   # sentences rated by all three groups
    loop_same = (same["L"] - same["C"]) + (same["C"] - same["R"]) + (same["R"] - same["L"])
    # the same loop computed on DISJOINT sentence sets for its three edges
    idx = rng.permutation(same.index); a, b, c = np.array_split(idx, 3)
    loop_disjoint = (same.loc[a, "L"] - same.loc[a, "C"]).mean() + (same.loc[b, "C"] - same.loc[b, "R"]).mean() \
        + (same.loc[c, "R"] - same.loc[c, "L"]).mean()
    return dict(n_sentences=int(len(same)), max_abs_loop_same=float(np.abs(loop_same).max()),
                loop_disjoint=float(loop_disjoint))


# ---------------------------------------------------------------- test A
def two_way(y, o, t):
    """least-squares additive fit y ~ outlet + topic (dummy coding); returns fitted, rss"""
    O = pd.get_dummies(o, drop_first=True).to_numpy(float); T = pd.get_dummies(t, drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(y)), O, T])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    f = X @ beta
    return f, float(((y - f) ** 2).sum())


def interaction_F(y, o, t):
    f, rss_add = two_way(y, o, t)
    cell = pd.Series(y).groupby([o.values, t.values]).transform("mean").to_numpy()
    rss_full = float(((y - cell) ** 2).sum())
    return (rss_add - rss_full) / max(rss_full, 1e-12), f


def freedman_lane(y, o, t, n=2000):
    F, f = interaction_F(y, o, t)
    r = y - f; cnt = 0
    for _ in range(n):
        Fp, _ = interaction_F(f + rng.permutation(r), o, t)
        cnt += Fp >= F - 1e-12
    return F, (cnt + 1) / (n + 1)


def test_A(d):
    s = d.groupby("sentence_id").agg(y=("b", "mean"), outlet=("outlet", "first"), topic=("topic", "first"))
    y, o, t = s["y"].to_numpy(), s["outlet"], s["topic"]
    F, p = freedman_lane(y, o, t)
    out = dict(n=len(s), global_F=F, global_p=p)
    # level 2: outlet propensity and topic effect (additive fit, reported as cell-free marginal means)
    out["outlet_mean"] = s.groupby("outlet")["y"].mean().round(3).to_dict()
    # local regions: every pair of topics, all outlets (the 4-cycles through those two topics)
    loc = []
    for t1, t2 in itertools.combinations(sorted(t.unique()), 2):
        m = t.isin([t1, t2]).to_numpy()
        Fl, pl = freedman_lane(y[m], o[m], t[m], n=300)
        loc.append((t1, t2, Fl, pl))
    ps = np.array([x[3] for x in loc])
    out["pair_regions"] = dict(n=len(loc), frac_p_below_05=float((ps < 0.05).mean()),
                               min_p=float(ps.min()), worst=sorted(loc, key=lambda x: x[3])[:5])
    # strongest single loops (outlet pair x topic pair), difference in differences of cell means
    cm = s.groupby(["outlet", "topic"])["y"].agg(["mean", "var", "count"])
    loops = []
    for (o1, o2) in itertools.combinations(sorted(o.unique()), 2):
        for (t1, t2) in itertools.combinations(sorted(t.unique()), 2):
            try:
                c = [cm.loc[(o1, t1)], cm.loc[(o1, t2)], cm.loc[(o2, t1)], cm.loc[(o2, t2)]]
            except KeyError:
                continue
            if min(x["count"] for x in c) < 5:
                continue
            phi = c[0]["mean"] - c[1]["mean"] - c[2]["mean"] + c[3]["mean"]
            se = np.sqrt(sum(x["var"] / x["count"] for x in c))
            loops.append((o1, o2, t1, t2, float(phi), float(phi / se)))
    loops.sort(key=lambda x: -abs(x[5]))
    out["n_loops"] = len(loops); out["top_loops"] = loops[:8]
    return out


# ---------------------------------------------------------------- test B
def test_B(d, cut=(-4, 3)):
    g = pd.cut(d["political_ideology"], [-11, cut[0], cut[1], 11], labels=["L", "C", "R"])
    S = d.assign(g=g).groupby(["sentence_id", "g"], observed=True)["b"].mean().unstack()
    typ = d.groupby("sentence_id")["type"].first()
    both = S[["L", "R"]].dropna().join(typ)
    dd = (both["L"] - both["R"]).to_numpy(); ty = both["type"].to_numpy()
    lev = {k: float(dd[ty == k].mean()) for k in ("left", "center", "right")}
    phi = lev["right"] - lev["left"]
    m = np.isin(ty, ["left", "right"]); dsub, tsub = dd[m], ty[m]
    cnt, n = 0, 20000
    for _ in range(n):
        tp = rng.permutation(tsub)
        cnt += abs(dsub[tp == "right"].mean() - dsub[tp == "left"].mean()) >= abs(phi) - 1e-12
    # level 2 per group per outlet type
    grid = {gname: {k: float(both.loc[both["type"] == k, gname].mean()) for k in ("left", "center", "right")}
            for gname in ("L", "R")}
    return dict(cut=cut, n_sentences=int(len(both)), d_by_type=lev, phi=phi, p=(cnt + 1) / (n + 1), grid=grid)


if __name__ == "__main__":
    d = load()
    R = {}
    R["theorem"] = th = theorem_check(d)
    print(f"THEOREM CHECK: {th['n_sentences']} sentences rated by L, C and R annotators.")
    print(f"   loop L-C-R-L on the same sentences: max |sum| = {th['max_abs_loop_same']:.2e}  (identically 0)")
    print(f"   the same loop on disjoint sentence thirds: {th['loop_disjoint']:+.3f}  (free to be non-zero)")
    R["A"] = A = test_A(d)
    print(f"\nTEST A outlet x topic ({A['n']} sentences): outlet bias propensity {A['outlet_mean']}")
    print(f"   GLOBAL interaction (loops over the whole outlet-topic graph): F = {A['global_F']:.4f}, "
          f"Freedman-Lane p = {A['global_p']:.4f}")
    pr = A["pair_regions"]
    print(f"   LOCAL regions (each pair of topics, {pr['n']} regions): fraction with p < 0.05 = "
          f"{pr['frac_p_below_05']:.2f}, min p = {pr['min_p']:.4f}")
    print("   worst local regions:", [(a, b, round(p, 3)) for a, b, _, p in pr["worst"]])
    print(f"   strongest single loops of {A['n_loops']} (outlet pair x topic pair: Phi, z):")
    for x in A["top_loops"]:
        print(f"      {x[0]:>10s}/{x[1]:<10s} x {x[2]:>22s}/{x[3]:<22s} Phi = {x[4]:+.3f}  z = {x[5]:+.2f}")
    R["B"] = {}
    for cut in ((-4, 3), (-1, 0), (-6, 5)):
        B = test_B(d, cut); R["B"][str(cut)] = B
        print(f"\nTEST B annotator ideology x outlet type (L <= {cut[0]}, R > {cut[1]}; {B['n_sentences']} sentences):")
        print(f"   bias rate  L-annotators: {({k: round(v, 3) for k, v in B['grid']['L'].items()})}")
        print(f"              R-annotators: {({k: round(v, 3) for k, v in B['grid']['R'].items()})}")
        print(f"   d = L - R by outlet type: {({k: round(v, 3) for k, v in B['d_by_type'].items()})}")
        print(f"   hostile-media loop Phi = d(right outlets) - d(left outlets) = {B['phi']:+.3f}   exact permutation p = {B['p']:.4g}")
    json.dump(R, open("mbic_results.json", "w"), indent=1, default=str)
