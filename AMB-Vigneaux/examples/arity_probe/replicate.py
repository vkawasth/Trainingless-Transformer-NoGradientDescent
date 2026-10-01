"""Replication of the coalition probe: finer ideology groups, BABE expert annotators, article year; lean-invariant
variogram gamma*_ij = 1/2 Var(x_i - x_j) - 1/2 (sampling var_i + var_j) (noise-corrected for groups of few raters).
Data: MBIC_XLSX (raw_labels_MBIC.xlsx) and BABE_DIR (the BABE repo: data/raw_labels_SG1.csv, data/raw_labels_SG2.csv,
annotator_demographics.csv). Not redistributed.
"""
import os, sys, json, itertools
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../bias3")]
import numpy as np, pandas as pd
from amb_vigneaux.hull import variogram, coalition_test_labelperm

BABE = os.environ.get("BABE_DIR", os.path.join(here, "../bias3/data/BABE"))
rng = np.random.default_rng(0); R = {}


def group_variogram(d, grp, S, value="b", unit="sentence_id", B=500, min_n=2):
    g = d.assign(g=grp).groupby([unit, "g"])[value]
    m = g.mean().unstack(); n = g.count().unstack(); v = (m * (1 - m) / (n - 1)).where(n > 1)
    cells, noise = [], []
    for u in m.index:
        ok = [s for s in S if s in m.columns and not np.isnan(m.loc[u, s]) and n.loc[u, s] >= min_n]   # >= 2 raters: unbiased noise
        cells.append({s: m.loc[u, s] for s in ok}); noise.append({s: v.loc[u, s] for s in ok})
    out = variogram(cells, S, B=B, noise=noise)
    G, se = np.array(out["gamma"]), np.array(out["se"])
    out["pairs"] = {f"{S[i]}-{S[j]}": (round(float(G[i, j]), 4), round(float(se[i, j]), 4)) for i, j in itertools.combinations(range(len(S)), 2)}
    return out


def show(label, out):
    print(f"   {label}: first merge {out['first_merge']} (bootstrap support {out['support']:.2f});  gamma* " +
          "  ".join(f"{k} {a:+.4f}±{b:.4f}" for k, (a, b) in out["pairs"].items()))


if __name__ == "__main__":
    import mbic
    d = mbic.load(); pi = d.political_ideology
    g3 = np.where(pi <= -4, "L", np.where(pi >= 3, "R", "C"))
    print("== MBIC crowd, noise-corrected variogram")
    R["mbic_3"] = o = group_variogram(d, g3, ["L", "C", "R"]); show("3 groups", o)
    g5 = pd.cut(pi, [-11, -7, -2, 1, 6, 10], labels=["LL", "L", "C", "R", "RR"]).astype(str)
    R["mbic_5"] = o = group_variogram(d, g5, ["LL", "L", "C", "R", "RR"]); show("5 groups (<=-7, -6..-2, -1..1, 2..6, >=7)", o)
    for ty in ("left", "center", "right"):
        dd = d[d.type == ty]; p2 = dd.political_ideology
        R[f"mbic_3_{ty}"] = o = group_variogram(dd, np.where(p2 <= -4, "L", np.where(p2 >= 3, "R", "C")), ["L", "C", "R"])
        show(f"3 groups, {ty} outlets", o)
    print("== MBIC coalition by article year (label-permutation null)")
    d["yr"] = d.news_link.str.extract(r"/(20\d\d)/")[0]; d["grp"] = np.where(pi <= -4, "left", np.where(pi >= 3, "right", "centre"))
    for lab, f in (("articles <= 2019", lambda y: isinstance(y, str) and y <= "2019"), ("articles 2020", lambda y: y == "2020")):
        cells = [(g.b.to_numpy(), g.grp.to_numpy()) for _, g in d.groupby("sentence_id")
                 if f(g.yr.iloc[0]) and all((g.grp == x).any() for x in ("left", "centre", "right"))]
        r = coalition_test_labelperm(cells, ["left", "centre", "right"], B=300); R[f"mbic_year_{lab}"] = r
        print(f"   {lab} (n {r['n']}): " + "  ".join(f"{k} {v['obs']} vs {v['null']:.0f} (p {v['p']:.3f})" for k, v in r["coalitions"].items()))

    print("== BABE expert annotators (ideology from annotator_demographics.csv)")
    dem = pd.read_csv(os.path.join(BABE, "annotator_demographics.csv"), sep=";")
    ide = dict(zip(dem.annotator_id, dem["political orientation"]))
    for f in ("raw_labels_SG1.csv", "raw_labels_SG2.csv"):
        b = pd.read_csv(os.path.join(BABE, "data", f), sep=";")
        t = b.pivot_table(index="text", columns="annotator_id", values="Label_bias_0-1", aggfunc="mean")
        A = sorted(t.columns); cells = [{a: r[a] for a in A if not np.isnan(r[a])} for _, r in t.iterrows()]
        v = variogram(cells, A, B=100); G, N = np.array(v["gamma"]), np.array(v["n"])
        pairs = [(i, j) for i, j in itertools.combinations(range(len(A)), 2) if N[i, j] > 50]
        g = np.array([G[i, j] for i, j in pairs]); lab = np.array([ide[a] for a in A])
        dist = lambda L: np.array([abs(L[i] - L[j]) for i, j in pairs])
        r = float(np.corrcoef(g, dist(lab))[0, 1]); null = np.array([np.corrcoef(g, dist(rng.permutation(lab)))[0, 1] for _ in range(5000)])
        grp = {a: ("L" if ide[a] <= -5 else ("C" if ide[a] == 0 else "R")) for a in A}
        members = {x: [int(a) for a in A if grp[a] == x] for x in "LCR"}
        if min(len(v) for v in members.values()) < 2:
            o = dict(first_merge=None, support=None, pairs={}, note="a group has a single annotator: noise cannot be corrected")
        else:
            o = group_variogram(b, b.annotator_id.map(grp), ["L", "C", "R"], value="Label_bias_0-1", unit="text", B=300)
        R[f"babe_{f}"] = dict(annotators=A, ideology=lab.tolist(), mantel_r=r, mantel_p=float((null >= r).mean()), groups=o,
                              members=members)
        print(f"   {f}: annotators {A}, ideology {lab.tolist()}")
        print(f"      Mantel r(gamma_ij, |ideology_i - ideology_j|) = {r:+.3f}, permutation p = {(null >= r).mean():.3f} ({len(pairs)} pairs)")
        if o.get("note"):
            print(f"      groups {members}: {o['note']}")
        else:
            show(f"groups {members}", o)
    json.dump(R, open(os.path.join(here, "replicate_results.json"), "w"), indent=1, default=str)
