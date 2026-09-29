"""Three-level bias analysis on BASIL (two sources describing the same events).

Level 1  same event, two descriptions (BASIL triplets: fox / nyt / hpo).
Level 2  bias b(s, e): the annotated article-level `relative_stance`
         (left -2, liberal -1, center 0, conservative +1, right +2), aggregated to b(s, topic, window).
Level 3  relations between biases. Paired decomposition, with each event as its own control:
             d(e) = b(A, e) - b(B, e)                   (removes every event effect)
             house lean      u  = mean_e d(e)
             selective bias  Phi(t1, t2) = mean_{t1} d - mean_{t2} d
         Phi is the loop sum on the source-topic bipartite graph (A-t1-B-t2-A) after event effects are
         removed. Phi = 0 on every loop  <=>  bias = house lean + topic effect (a coboundary).
Tests (exact / conditional):
  * continuous: permutation of topic labels across events (conditions on the multiset of d)
  * toric: binary y = [stance right of centre]; log-linear model with (source, event), (event, y) and
    (source, y) margins; sufficient statistics fixed => the fiber is the set of direction assignments to
    discordant events with a fixed count. Exact p by hypergeometric enumeration; checked against MCMC
    with the degree-4 Markov-basis moves (swap the discordance directions of two events).
Gluing map: regions = theme x window cells; a union of cells "glues" if one house lean fits it
(homogeneity of d across its cells not rejected).
"""
import json, glob, collections, itertools, sys
import numpy as np
from scipy.stats import hypergeom
from event_labels import L, triplet_labels
TL = triplet_labels()

BASIL = "/tmp/claude-0/BASIL"
STANCE = {"left": -2, "liberal": -1, "center": 0, "conservative": 1, "right": 2}
WIN = lambda y: "2010-13" if y <= 2013 else ("2014-16" if y <= 2016 else "2017-19")
rng = np.random.default_rng(0)


def load():
    # join annotation <-> article by FILENAME (four HPO annotation files carry a duplicated uuid)
    ev = collections.defaultdict(dict)
    for a in glob.glob(f"{BASIL}/annotations/*/*_ann.json"):
        A = json.load(open(a))
        d = json.load(open(a.replace("/annotations/", "/articles/").replace("_ann.json", ".json")))
        e = d["triplet-uuid"]
        ev[e]["event"] = d["main-event"]; ev[e]["year"] = int(d["date"][:4])
        ev[e][d["source"].lower()] = STANCE[A["article-level-annotations"]["relative_stance"]]
    out = []
    for e, r in ev.items():
        th, camp, _ = TL[e]
        out.append(dict(id=e, event=r["event"], year=r["year"], win=WIN(r["year"]), theme=th, camp=camp,
                        fox=r["fox"], nyt=r["nyt"], hpo=r["hpo"]))
    return out


def perm_p(d, labels, groups, stat, n=20000):
    obs = stat(d, labels, groups)
    lab = np.array(labels); cnt = 0
    for _ in range(n):
        cnt += stat(d, rng.permutation(lab), groups) >= obs - 1e-12
    return obs, (cnt + 1) / (n + 1)


def between_var(d, labels, groups):
    """omnibus statistic: weighted between-group variance of d (one-way ANOVA numerator)"""
    lab = np.asarray(labels); m = d.mean()
    return sum((lab == g).sum() * (d[lab == g].mean() - m) ** 2 for g in groups if (lab == g).any())


def absdiff(g1, g2):
    def f(d, labels, groups):
        lab = np.asarray(labels)
        a, b = d[lab == g1], d[lab == g2]
        return abs(a.mean() - b.mean()) if len(a) and len(b) else 0.0
    return f


def toric_exact(yA, yB, labels, g1, g2):
    """Exact conditional test of source x topic x y interaction between topics g1, g2 (binary y).
    Fiber: discordant events keep their count of 'A right'; directions are exchangeable.
    Statistic: number of 'A right' discordances in g1. Two-sided p by hypergeometric enumeration."""
    lab = np.asarray(labels)
    disc = (yA != yB) & ((lab == g1) | (lab == g2))
    n = int(disc.sum()); k = int((yA[disc] > yB[disc]).sum()); n1 = int((disc & (lab == g1)).sum())
    x = int(((yA > yB) & disc & (lab == g1)).sum())
    if n == 0 or n1 in (0, n):
        return dict(n_disc=n, p=1.0, x=x, n1=n1, k=k)
    rv = hypergeom(n, k, n1)
    px = rv.pmf(x); p = float(sum(rv.pmf(j) for j in range(max(0, k + n1 - n), min(k, n1) + 1) if rv.pmf(j) <= px + 1e-12))
    return dict(n_disc=n, k=k, n1=n1, x=x, p=min(1.0, p))


def toric_mcmc(yA, yB, labels, g1, g2, steps=200000):
    """Same test by Markov-basis MCMC: degree-4 moves swap the discordance directions of two events
    (preserving every sufficient statistic); estimate P(|x - E x| >= |x_obs - E x|)."""
    lab = np.asarray(labels)
    idx = np.flatnonzero((yA != yB) & ((lab == g1) | (lab == g2)))
    if len(idx) < 2:
        return 1.0
    dirs = (yA[idx] > yB[idx]).astype(int); ing1 = (lab[idx] == g1)
    x_obs = int(dirs[ing1].sum()); Ex = dirs.mean() * ing1.sum()
    cur = dirs.copy(); hits = 0
    for _ in range(steps):
        i, j = rng.integers(len(idx), size=2)
        if cur[i] != cur[j]:
            cur[i], cur[j] = cur[j], cur[i]                  # the Markov-basis move
        hits += abs(cur[ing1].sum() - Ex) >= abs(x_obs - Ex) - 1e-12
    return hits / steps


def analyse(E, A, B, key):
    d = np.array([e[A] - e[B] for e in E], float)
    lab = [e[key] for e in E]
    groups = sorted(set(lab))
    yA = np.array([e[A] > 0 for e in E], int); yB = np.array([e[B] > 0 for e in E], int)
    res = dict(pair=f"{A}-{B}", key=key, house_lean=float(d.mean()), n=len(d),
               by_group={g: dict(n=int((np.array(lab) == g).sum()), mean_d=float(d[np.array(lab) == g].mean()))
                         for g in groups})
    obs, p = perm_p(d, lab, groups, between_var)
    res["omnibus"] = dict(stat=float(obs), p_perm=p)
    loops = []
    for g1, g2 in itertools.combinations(groups, 2):
        o, pp = perm_p(d, lab, groups, absdiff(g1, g2), n=10000)
        te = toric_exact(yA, yB, lab, g1, g2)
        loops.append(dict(t1=g1, t2=g2, Phi=float(d[np.array(lab) == g1].mean() - d[np.array(lab) == g2].mean()),
                          p_perm=pp, toric=te))
    res["loops"] = loops
    return res


def gluing_map(E, A, B, alpha=0.05):
    """regions = theme x window cells; test whether one house lean fits: rows, columns, whole."""
    cells = sorted({(e["theme"], e["win"]) for e in E})
    d = np.array([e[A] - e[B] for e in E], float)
    cell_of = [(e["theme"], e["win"]) for e in E]
    out = {}
    def test(region):
        m = np.array([c in region for c in cell_of])
        if len({c for c, k in zip(cell_of, m) if k}) < 2:
            return None
        labs = [str(c) for c, k in zip(cell_of, m) if k]
        o, p = perm_p(d[m], labs, sorted(set(labs)), between_var, n=5000)
        return dict(n=int(m.sum()), lean=float(d[m].mean()), p=p, glues=p > alpha)
    for th in sorted({c[0] for c in cells}):
        out[f"theme {th} across windows"] = test({c for c in cells if c[0] == th})
    for w in sorted({c[1] for c in cells}):
        out[f"window {w} across themes"] = test({c for c in cells if c[1] == w})
    out["whole cover"] = test(set(cells))
    out["cells"] = {f"{c[0]} {c[1]}": dict(n=int(sum(1 for x in cell_of if x == c)),
                                           lean=float(d[[x == c for x in cell_of]].mean())) for c in cells}
    return out


if __name__ == "__main__":
    E = load()
    print(f"{len(E)} events; mean stance fox {np.mean([e['fox'] for e in E]):+.2f}, nyt {np.mean([e['nyt'] for e in E]):+.2f}, "
          f"hpo {np.mean([e['hpo'] for e in E]):+.2f}")
    R = {}
    for A, B in (("fox", "nyt"), ("fox", "hpo"), ("hpo", "nyt")):
        for key in ("camp", "theme", "win"):
            r = analyse(E, A, B, key); R[f"{A}-{B}:{key}"] = r
            print(f"\n== {A} vs {B}, topic = {key}:  house lean {r['house_lean']:+.2f}  omnibus p = {r['omnibus']['p_perm']:.4f}")
            for g, v in r["by_group"].items():
                print(f"     {g:8s} n={v['n']:3d}  mean d = {v['mean_d']:+.2f}")
            for lp in r["loops"]:
                t = lp["toric"]
                print(f"     loop {lp['t1']}-{lp['t2']}: Phi = {lp['Phi']:+.2f}  p_perm = {lp['p_perm']:.4f}   "
                      f"toric exact p = {t['p']:.4f} (discordant {t['n_disc']})")
        R[f"{A}-{B}:gluing"] = g = gluing_map(E, A, B)
        print(f"   gluing map ({A} vs {B}):")
        for k, v in g.items():
            if k != "cells" and v:
                print(f"     {k:32s} n={v['n']:3d} lean {v['lean']:+.2f}  p={v['p']:.3f}  {'glues' if v['glues'] else 'DOES NOT GLUE'}")
    # Markov-basis MCMC cross-check of the exact toric test
    chk = []
    for A, B, key in (("fox", "nyt", "camp"), ("fox", "nyt", "theme"), ("fox", "hpo", "camp")):
        lab = [e[key] for e in E]
        yA = np.array([e[A] > 0 for e in E], int); yB = np.array([e[B] > 0 for e in E], int)
        for g1, g2 in itertools.combinations(sorted(set(lab)), 2):
            ex = toric_exact(yA, yB, lab, g1, g2)["p"]; mc = toric_mcmc(yA, yB, lab, g1, g2, steps=60000)
            chk.append((A, B, g1, g2, ex, mc))
    print("\n  toric exact vs Markov-basis MCMC:", max(abs(a - b) for *_, a, b in chk), "max |difference|")
    R["mcmc_check"] = chk
    json.dump(R, open("bias3_results.json", "w"), indent=1, default=str)
