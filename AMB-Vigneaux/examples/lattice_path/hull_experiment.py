"""Berkovich / dendrogram hull covers as a way to choose regions on the gluing lattice, across corpora; attempts to
'stitch' good regions by a loop metric; and the two models that do use the loop (rank-1 interaction; tropical rank 2).

Cells (source, topic) -> (k, n) with a binary label (MBIC: majority-biased sentence; NewsWCL50: positive framing;
AllSides: adversarial / directed framing; BASIL: positive span). Each cell with n >= 10 is held out in turn.

(1) Hull cover. Single-linkage dendrogram (Euclidean; the real hull; its merge heights are an ultrametric, so discs of
    one radius partition the topics) on one feature per topic, computed without the target:
      value    pooled bias rate of the topic (1-D);
      profile  column of empirical logits over sources (missing -> the source's mean over topics);
      loop     column of weighted loop residuals sqrt(w)(y - a_s - b_t) of the full-cover additive fit.
    Chain of discs containing the target topic (root -> leaf); canonical region = largest disc with >= MIN topics that
    glues (chi^2 loop test); else the smallest disc with >= MIN topics.
    Reported: MAE vs held-out truth, how often a proper sub-region is chosen, spread of p_hat along the chain, spread
    over ALL glueable regions containing the target topic (exhaustive), oracle best glueable region (uses the truth).
(2) Stitching by a loop metric: region = target topic + topics whose loop-residual column correlates positively with
    the target topic's over the OTHER sources (needs >= 3 other sources).
(3) Models of the loop: rank-1 interaction logit = a_s + b_t + u_s v_t, and tropical rank 2 (columns on a tropical
    line, i.e. a tree): logit = min / max of two additive regimes. Both fitted by alternating closed-form weighted
    least squares (no gradient steps).
(4) Direction: the hull, its radii and its chains are invariant under relabelling (k -> n - k); the signed loop
    influence flips sign. Checked on every corpus.
Env: MBIC_XLSX, NEWSWCL50_CSV, ALLSIDES_DIR, BASIL_DIR.
"""
import os, sys, json, itertools, collections
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../certificate"), os.path.join(here, "../bias3"),
                os.path.join(here, "../allsides")]
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from amb_vigneaux.lattice_path import evaluate, fit_region, _logit, _without

rng = np.random.default_rng(0)


def cells_from(units):
    c = collections.defaultdict(lambda: [0, 0])
    for s, t, v, _ in units:
        c[(s, t)][0] += int(v >= 0.5); c[(s, t)][1] += 1
    return {k: tuple(v) for k, v in c.items()}


def mbic_cells():
    import mbic
    d = mbic.load(); s = d.groupby("sentence_id").agg(y=("b", "mean"), o=("outlet", "first"), t=("topic", "first"))
    s["lab"] = (s.y >= 0.5).astype(int)
    return {(o, t): (int(g.lab.sum()), int(len(g))) for (o, t), g in s.groupby(["o", "t"])}


def residuals(cw, S_all, T_all):
    fit = fit_region(cw, S_all, set(T_all)); S, T, c = fit["S"], fit["T"], fit["coef"]
    H = np.full((len(S_all), len(T_all)), np.nan)
    for (o, t), kn in cw.items():
        if o in S and t in T:
            y, w = _logit(*kn); H[S_all.index(o), T_all.index(t)] = np.sqrt(w) * (y - c[S.index(o)] - c[len(S) + T.index(t)])
    return H


def features(cw, kind, S_all, T_all):
    if kind == "value":
        return np.array([[sum(cw[(o, t)][0] for o in S_all if (o, t) in cw) / max(1, sum(cw[(o, t)][1] for o in S_all if (o, t) in cw))] for t in T_all])
    F = (np.array([[_logit(*cw[(o, t)])[0] if (o, t) in cw else np.nan for o in S_all] for t in T_all]) if kind == "profile"
         else residuals(cw, S_all, T_all).T)
    return np.nan_to_num(np.where(np.isnan(F), np.nanmean(F, 0, keepdims=True), F))


def chain(F, T_all, ti):
    Z = linkage(F, "single"); discs = []
    for k in range(1, len(T_all) + 1):
        lab = fcluster(Z, k, "maxclust"); D = frozenset(T_all[i] for i in np.where(lab == lab[ti])[0])
        if not discs or D != discs[-1]:
            discs.append(D)
    return discs, Z


def table(cw, S_all, T_all):
    Y = np.full((len(S_all), len(T_all)), np.nan); W = np.zeros_like(Y)
    for (o, t), kn in cw.items():
        Y[S_all.index(o), T_all.index(t)], W[S_all.index(o), T_all.index(t)] = _logit(*kn)
    return Y, W


def rank1(cw, S_all, T_all, iters=200):
    Y, W = table(cw, S_all, T_all); Y0 = np.nan_to_num(Y)
    a = np.zeros(len(S_all)); b = np.zeros(len(T_all)); u = np.full(len(S_all), .1); v = np.full(len(T_all), .1)
    for _ in range(iters):
        a = (W * (Y0 - b[None] - np.outer(u, v))).sum(1) / np.maximum(W.sum(1), 1e-12)
        b = (W * (Y0 - a[:, None] - np.outer(u, v))).sum(0) / np.maximum(W.sum(0), 1e-12)
        R = Y0 - a[:, None] - b[None]
        u = (W * R * v[None]).sum(1) / np.maximum((W * v[None] ** 2).sum(1), 1e-12)
        v = (W * R * u[:, None]).sum(0) / np.maximum((W * u[:, None] ** 2).sum(0), 1e-12)
    return a[:, None] + b[None] + np.outer(u, v)


def _wls_add(Y, W, M):
    S, T = Y.shape; X, y, w = [], [], []
    for i, j in zip(*np.where(M)):
        r = np.zeros(S + T); r[i] = 1; r[S + j] = 1; X.append(r); y.append(Y[i, j]); w.append(W[i, j] + 1e-9)
    X, y, w = np.array(X), np.array(y), np.array(w)
    c = np.linalg.pinv(X.T @ (X * w[:, None]) + 1e-6 * np.eye(S + T)) @ (X.T @ (w * y))
    return c[:S, None] + c[None, S:]


def trop2(cw, S_all, T_all, sense, iters=50, starts=10):
    Y, W = table(cw, S_all, T_all); obs = ~np.isnan(Y); best = None; g = np.random.default_rng(0)
    for _ in range(starts):
        lab = g.integers(2, size=Y.shape)
        for _ in range(iters):
            P = np.stack([_wls_add(Y, W, obs & (lab == r)) if (obs & (lab == r)).any() else np.zeros(Y.shape) for r in (0, 1)])
            new = P.argmax(0) if sense == "max" else P.argmin(0)
            if (new[obs] == lab[obs]).all():
                break
            lab = new
        pred = P.max(0) if sense == "max" else P.min(0); loss = np.nansum(W * (Y - pred) ** 2)
        if best is None or loss < best[0]:
            best = (loss, pred)
    return best[1]


expit = lambda e: 1 / (1 + np.exp(-e))


def run(name, cells, MIN):
    S_all = sorted({s for s, _ in cells}); T_all = sorted({t for _, t in cells})
    E = collections.defaultdict(list); sub = collections.Counter(); inv = 0; flip = 0; n_dir = 0
    for tg in [c for c, (k, n) in cells.items() if n >= 10]:
        k, n = cells[tg]; truth = k / n; cw = _without(cells, tg); fe = evaluate(cells, tg, T_all)
        if fe is None:
            continue
        si, ti = S_all.index(tg[0]), T_all.index(tg[1])
        E["full"].append(abs(fe["p"] - truth)); E["full_glues"].append(fe["glues"])
        others = [t for t in T_all if t != tg[1]]; ps = []
        for r in range(MIN - 1, len(others) + 1):
            for comb in itertools.combinations(others, r):
                e = evaluate(cells, tg, (tg[1],) + comb)
                if e is not None and e["glues"]:
                    ps.append(e["p"])
        if ps:
            E["envelope"].append(max(ps) - min(ps)); E["oracle"].append(min(abs(p - truth) for p in ps))
        flipped = {c: (m - kk, m) for c, (kk, m) in cw.items()}
        for kind in ("value", "profile", "loop"):
            discs, Z = chain(features(cw, kind, S_all, T_all), T_all, ti)
            discs_f, Zf = chain(features(flipped, kind, S_all, T_all), T_all, ti)
            inv += discs == discs_f and np.allclose(Z[:, 2], Zf[:, 2])
            ev = [(D, evaluate(cells, tg, D)) for D in discs if len(D) >= MIN]; ev = [(D, e) for D, e in ev if e is not None]
            pick = next((e for _, e in ev if e["glues"]), ev[-1][1]) if ev else fe
            sub[kind] += len(pick["region"]) < len(T_all); E[kind].append(abs(pick["p"] - truth))
            E[kind + "_chain"].append(max(e["p"] for _, e in ev) - min(e["p"] for _, e in ev) if ev else 0.0)
        H = residuals(cw, S_all, T_all); Hf = residuals(flipped, S_all, T_all); m = ~np.isnan(H[si])
        if m.any():
            n_dir += 1; flip += np.allclose(Hf[si, m], -H[si, m], atol=1e-8)
        sim = {}
        for j, x in enumerate(T_all):
            mm = ~np.isnan(H[:, ti]) & ~np.isnan(H[:, j]); mm[si] = False
            if j != ti and mm.sum() >= 3 and H[mm, ti].std() > 0 and H[mm, j].std() > 0:
                sim[x] = np.corrcoef(H[mm, ti], H[mm, j])[0, 1]
        pos = [x for x, r in sim.items() if r > 0 and (tg[0], x) in cw]
        e = evaluate(cells, tg, [tg[1]] + pos) if pos else None
        E["stitch"].append(abs((e or fe)["p"] - truth)); E["stitch_applicable"].append(bool(sim))
        E["rank1"].append(abs(expit(rank1(cw, S_all, T_all)[si, ti]) - truth))
        for sense in ("min", "max"):
            E["trop2_" + sense].append(abs(expit(trop2(cw, S_all, T_all, sense)[si, ti]) - truth))
    N = len(E["full"]); full = np.array(E["full"])
    wins = {m: int((np.array(E[m]) < full).sum()) for m in ("rank1", "trop2_min", "trop2_max")}
    out = dict(corpus=name, sources=len(S_all), topics=len(T_all), targets=N, min_topics=MIN, full_glues=int(sum(E["full_glues"])),
               mae={m: float(np.mean(E[m])) for m in ("full", "value", "profile", "loop", "stitch", "rank1", "trop2_min", "trop2_max")},
               sub_region={k: int(v) for k, v in sub.items()},
               chain_spread={k: float(np.median(E[k + "_chain"])) for k in ("value", "profile", "loop")},
               envelope_spread=float(np.median(E["envelope"])) if E["envelope"] else 0.0,
               oracle=float(np.mean(E["oracle"])) if E["oracle"] else None,
               stitch_applicable=int(sum(E["stitch_applicable"])),
               beats_full=wins, sign_p={m: float(stats.binomtest(w, N).pvalue) for m, w in wins.items()},
               hull_invariant_under_flip=f"{inv}/{3 * N}", loop_sign_flips=f"{flip}/{n_dir}")
    mae = out["mae"]
    print(f"\n== {name}: {len(S_all)} x {len(T_all)}, {N} targets, MIN {MIN}; full cover glues {out['full_glues']}/{N}")
    print(f"   MAE full {mae['full']:.3f} | hull value {mae['value']:.3f} profile {mae['profile']:.3f} loop {mae['loop']:.3f}"
          f" (sub-region {dict(sub)}) | chain spread {out['chain_spread']} | lattice envelope {out['envelope_spread']:.3f} | oracle {out['oracle']}")
    print(f"   stitch {mae['stitch']:.3f} (applicable {out['stitch_applicable']}/{N}) | rank-1 {mae['rank1']:.3f} | trop2 min {mae['trop2_min']:.3f}"
          f" max {mae['trop2_max']:.3f} | beats full {wins} | sign p {out['sign_p']}")
    print(f"   hull (chains + merge heights) invariant under k -> n-k: {out['hull_invariant_under_flip']}; signed loop residuals flip: {out['loop_sign_flips']}")
    return out


if __name__ == "__main__":
    import run_all as RA
    res = [run("MBIC (outlet x topic)", mbic_cells(), 5),
           run("NewsWCL50 (outlet x target)", cells_from(RA.newswcl50()), 2),
           run("AllSides adversarial (side x topic)", cells_from(RA.allsides("outcome")), 2),
           run("AllSides directed (side x topic)", cells_from(RA.allsides("outcome_dir")), 2),
           run("BASIL (source x target party)", cells_from(RA.basil(False)), 2)]
    json.dump(res, open(os.path.join(here, "hull_experiment_results.json"), "w"), indent=1)
