#!/usr/bin/env python3
"""LOCAL EM: learn the hierarchy one level at a time, bottom up, from random starts. Gradient-free.

Why. Global EM must align every level at once; its bad basins come from that coupling. Given the INSIDE messages of
level l-1 (soft evidence for each child node), level l is a one-level latent-class model:
    p(evidence of node n) = sum_z pi_z sum_{x,y} theta_z[x, y] beta_left[n, x] beta_right[n, y],
whose only symmetry is relabelling of z (identifiable below the root: Allman-Matias-Rhodes; merge_strata.py).
So each level is fitted LOCALLY -- only its own nodes and their children, never the whole tree -- and we go deeper.

A lone child pair is ONE categorical draw from its symbol, and a mixture of single categorical draws is not
identifiable (only the pooled mixture is). What identifies z is how its pair co-occurs with its SIBLING's pair under
the same parent. So the local neighbourhood is a TWO-LEVEL WINDOW: parent -> (left, right) -> child pairs; level l is
kept, the parent table is refitted in the next window (l+1, l+2).
Per window:
  * R restarts from random Dirichlet rule tables (symmetry breaking at the start, the EM analogue of random init);
  * optional tempered E-steps (T0 > 1). Measured: annealing from T0 = 2 drives every symbol to the SAME table
    (the symmetric fixed point, exactly), after which EM cannot separate them -- so the default is T0 = 1;
  * best restart by local log-likelihood; its inside messages (soft, per-node rescaled) are passed up.
Root. The root level is not identifiable (merge_strata.py): the likelihood sees it only through the mixture of
its rules, so it is fitted as ONE symbol whose rule table is that mixture.
Depth check. For every level we report the smallest held-out cost of merging two fitted symbols against its se;
a level whose symbols cannot be told apart at this N is flagged (go no deeper than the data resolve).
Optional global polish: a few sweeps of full inside-outside EM from the local solution.

    python3 local_em.py --data runs/V16L4 --n-train 20000 --json results/V16L4_local.json
"""
import json, argparse, time, itertools
import numpy as np
import o6_score as S

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--n-train", type=int, default=20000)
ap.add_argument("--n-val", type=int, default=2000)
ap.add_argument("--restarts", type=int, default=6)
ap.add_argument("--t0", type=float, default=1.0, help="annealing start; >1 collapses all symbols to one (see doc)")
ap.add_argument("--anneal", type=int, default=25)
ap.add_argument("--iters", type=int, default=40)
ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration of random starts")
ap.add_argument("--sm-rounds", type=int, default=0, help="targeted local split-merge rounds per window")
ap.add_argument("--sm-iters", type=int, default=20)
ap.add_argument("--delta", type=float, default=0.25)
ap.add_argument("--polish", type=int, default=0, help="global inside-outside sweeps after the local fit")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=65536)
ap.add_argument("--json", default="")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
def load(f, n):
    x = np.array(json.load(open(f"{a.data}/{f}")), dtype=np.int64)
    return x[:(len(x) // SEQ) * SEQ].reshape(-1, SEQ)[:n]
TR, VA = load("train_ids.json", a.n_train), load("val_ids.json", a.n_val)


def onehot(X):
    o = np.zeros(X.shape + (NLEAF,)); np.put_along_axis(o, X[..., None], 1.0, -1); return o


def evid(lo, hi, th):
    """L[n, z] = sum_xy th[z,x,y] lo[n,x] hi[n,y], chunked"""
    out = np.empty((lo.shape[0], th.shape[0]))
    for s in range(0, lo.shape[0], a.chunk):
        t = np.einsum("nx,zxy->nzy", lo[s:s + a.chunk], th, optimize=True)
        out[s:s + a.chunk] = np.einsum("nzy,ny->nz", t, hi[s:s + a.chunk], optimize=True)
    return out


def em_window(loL, hiL, loR, hiR, th, Th, pw, iters, T0=1.0, anneal=1):
    n, c = loL.shape; K = th.shape[0]; Kp = Th.shape[0]; ll = -np.inf
    for T in list(np.geomspace(T0, 1.0, anneal)) + [1.0] * iters:
        eL = np.maximum(evid(loL, hiL, th), 1e-300); eR = np.maximum(evid(loR, hiR, th), 1e-300)
        sL = eL.max(1, keepdims=True); sR = eR.max(1, keepdims=True); eL /= sL; eR /= sR
        Cth = np.zeros_like(th); CTh = np.zeros_like(Th); cw = np.zeros(Kp); ll = 0.0
        for s in range(0, n, a.chunk):
            j = pw[None, :, None, None] * Th[None] * eL[s:s + a.chunk, None, :, None] * eR[s:s + a.chunk, None, None, :]
            tot = j.sum((1, 2, 3)); ll += float((np.log(np.maximum(tot, 1e-300)) + np.log(sL[s:s + a.chunk, 0]) + np.log(sR[s:s + a.chunk, 0])).sum())
            if T != 1.0:
                j = j ** (1.0 / T); tot = j.sum((1, 2, 3))
            j /= np.maximum(tot, 1e-300)[:, None, None, None]
            CTh += j.sum(0); cw += j.sum((0, 2, 3))
            qL = j.sum((1, 3)) / eL[s:s + a.chunk]; qR = j.sum((1, 2)) / eR[s:s + a.chunk]
            lL, hL, lR, hR = loL[s:s + a.chunk] / sL[s:s + a.chunk], hiL[s:s + a.chunk], loR[s:s + a.chunk] / sR[s:s + a.chunk], hiR[s:s + a.chunk]
            for z in range(K):
                Cth[z] += (lL * qL[:, z:z + 1]).T @ hL + (lR * qR[:, z:z + 1]).T @ hR
        Cth *= th
        th = (Cth + 1e-12) / (Cth + 1e-12).sum((1, 2), keepdims=True)
        Th = (CTh + 1e-12) / (CTh + 1e-12).sum((1, 2), keepdims=True); pw = (cw + 1e-12) / (cw + 1e-12).sum()
    return th, Th, pw, ll


def fit_window(loL, hiL, loR, hiR, K, Kp, T0):
    """two-level window: parent w (Kp symbols) -> (z_left, z_right) (K symbols each) -> child pairs.
    The sibling is the second view that identifies z (a lone child pair is a single categorical draw)."""
    n, c = loL.shape
    best = None
    for r in range(a.restarts):
        th = rng.dirichlet(np.full(c * c, a.alpha), size=K).reshape(K, c, c)
        Th = rng.dirichlet(np.full(K * K, a.alpha), size=Kp).reshape(Kp, K, K); pw = np.full(Kp, 1.0 / Kp)
        th, Th, pw, ll = em_window(loL, hiL, loR, hiR, th, Th, pw, a.iters, T0, a.anneal)
        if best is None or ll > best[0]:
            best = (ll, th, Th, pw, r)
    return best


def refine(tr, va, th, Th, pw, nps_parent):
    """targeted local split-merge: while the merge diagnostic finds a redundant pair, merge it, split the most
    loaded symbol (usage x entropy of its rule row) with a +-delta perturbation, re-run window EM; keep only if the
    held-out window likelihood improves."""
    log = []
    cur = float(window_ll(*va, th, Th, pw).sum())
    for rd in range(a.sm_rounds):
        base = window_ll(*va, th, Th, pw)
        use = (Th * pw[:, None, None]).sum((0, 2)) + (Th * pw[:, None, None]).sum((0, 1))
        best = None
        for s, t in itertools.combinations(range(th.shape[0]), 2):
            w = use[[s, t]] / max(use[[s, t]].sum(), 1e-300); row = w[0] * th[s] + w[1] * th[t]
            th2 = np.delete(th, t, 0); th2[s if s < t else s - 1] = row
            T2 = Th.copy(); T2[:, s, :] += T2[:, t, :]; T2 = np.delete(T2, t, 1); T2[:, :, s] += T2[:, :, t]; T2 = np.delete(T2, t, 2)
            d = (base - window_ll(*va, th2, T2, pw)).reshape(-1, nps_parent).sum(1)
            cost, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
            if best is None or cost < best[0]:
                best = (cost, se, s, t, th2, T2)
        cost, se, s, t, th2, T2 = best
        if cost > 2 * se:
            log.append(dict(round=rd, stop="no redundant pair", min_merge=float(cost))); break
        use2 = (T2 * pw[:, None, None]).sum((0, 2)) + (T2 * pw[:, None, None]).sum((0, 1))
        ent = -(th2 * np.log(np.maximum(th2, 1e-300))).sum((1, 2))
        k = int(np.argmax(use2 * ent))
        pert = 1.0 + a.delta * rng.uniform(-1, 1, size=(2,) + th2.shape[1:])
        rows = th2[k][None] * pert; rows /= rows.sum((1, 2), keepdims=True)
        th3 = np.concatenate([th2, rows[1:2]], 0); th3[k] = rows[0]
        T3 = np.concatenate([T2, T2[:, k:k + 1, :] / 2], 1); T3[:, k, :] /= 2
        T3 = np.concatenate([T3, T3[:, :, k:k + 1] / 2], 2); T3[:, :, k] /= 2
        T3 /= T3.sum((1, 2), keepdims=True)
        th3, T3, pw3, _ = em_window(*tr, th3, T3, pw.copy(), a.sm_iters)
        new = float(window_ll(*va, th3, T3, pw3).sum())
        ok = new > cur
        log.append(dict(round=rd, merged=(int(s), int(t)), merge_cost=float(cost), split=k, val_gain=new - cur, accepted=ok))
        if ok:
            th, Th, pw, cur = th3, T3, pw3, new
        else:
            break
    return th, Th, pw, log


def window_ll(loL, hiL, loR, hiR, th, Th, pw):
    """per-node log-likelihood of the two-level window"""
    eL = np.maximum(evid(loL, hiL, th), 1e-300); eR = np.maximum(evid(loR, hiR, th), 1e-300)
    out = np.empty(len(eL))
    for s in range(0, len(eL), a.chunk):
        j = np.einsum("w,wab,na,nb->n", pw, Th, eL[s:s + a.chunk], eR[s:s + a.chunk], optimize=True)
        out[s:s + a.chunk] = np.log(np.maximum(j, 1e-300))
    return out


def merge_check(win, th, Th, pw, nodes_per_seq):
    """held-out window cost (nats/seq) of merging each pair of fitted level-l symbols (rows averaged by usage,
    parent axes collapsed). A lone-node mixture cannot be used: merging is exactly free there (single draws)."""
    base = window_ll(*win, th, Th, pw)
    use = (Th * pw[:, None, None]).sum((0, 2)) + (Th * pw[:, None, None]).sum((0, 1))
    res = []
    for s, t in itertools.combinations(range(th.shape[0]), 2):
        w = use[[s, t]] / max(use[[s, t]].sum(), 1e-300); row = w[0] * th[s] + w[1] * th[t]
        th2 = np.delete(th, t, 0); th2[s if s < t else s - 1] = row
        T2 = Th.copy(); T2[:, s, :] += T2[:, t, :]; T2 = np.delete(T2, t, 1); T2[:, :, s] += T2[:, :, t]; T2 = np.delete(T2, t, 2)
        d = (base - window_ll(*win, th2, T2, pw)).reshape(-1, nodes_per_seq).sum(1)
        res.append((float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))))
    i = int(np.argmin([r[0] for r in res]))
    return dict(min_merge=res[i][0], se=res[i][1], resolvable=bool(res[i][0] > 2 * res[i][1]),
                median_merge=float(np.median([r[0] for r in res])))


def up_msgs(lo, hi, th, shape):
    b = np.maximum(evid(lo, hi, th), 1e-300); b /= b.sum(1, keepdims=True); return b.reshape(shape)


def polish(P, X, sweeps):
    for _ in range(sweeps):
        tabs, _ = S.inside(X, P, L, NLEAF); outs = S.outside(tabs, P, L); C = {}
        for l in range(1, L + 1):
            lo, hi = tabs[l - 1][:, 0::2], tabs[l - 1][:, 1::2]
            raw = S.up(P[l], lo, hi); po = outs[l] / np.maximum((outs[l] * raw).sum(-1, keepdims=True), 1e-300)
            B, n, p = po.shape; lo2, hi2, po2 = lo.reshape(B * n, -1), hi.reshape(B * n, -1), po.reshape(B * n, p)
            C[l] = np.stack([(lo2 * po2[:, z:z + 1]).T @ hi2 for z in range(p)]) * P[l]
        P = {l: (C[l] + 1e-12) / (C[l] + 1e-12).sum((1, 2), keepdims=True) for l in C}
    return P


if __name__ == "__main__":
    t0 = time.time(); P = {}; info = {}
    btr, bva = onehot(TR), onehot(VA)
    for l in range(1, L):
        # level-l nodes as (left, right) siblings under level-(l+1) parents
        c = btr.shape[2]
        def quad(b):
            q = b.reshape(b.shape[0], -1, 4, c)
            return (q[:, :, 0].reshape(-1, c), q[:, :, 1].reshape(-1, c), q[:, :, 2].reshape(-1, c), q[:, :, 3].reshape(-1, c))
        loL, hiL, loR, hiR = quad(btr)
        Kp = V if l + 1 < L else 1                               # root: one symbol (the identifiable mixture)
        tl = time.time()
        ll, th, Th, pw, r = fit_window(loL, hiL, loR, hiR, V, Kp, a.t0)
        smlog = []
        if a.sm_rounds:
            th, Th, pw, smlog = refine((loL, hiL, loR, hiR), quad(bva), th, Th, pw, SEQ >> (l + 1))
        P[l] = th
        if l + 1 == L:
            P[L] = Th
        lo, hi = btr[:, 0::2].reshape(-1, c), btr[:, 1::2].reshape(-1, c)
        lov, hiv = bva[:, 0::2].reshape(-1, c), bva[:, 1::2].reshape(-1, c)
        nps = SEQ >> l
        chk = merge_check(quad(bva), th, Th, pw, SEQ >> (l + 1))
        info[l] = dict(K=V, window_ll_per_seq=ll / TR.shape[0], best_restart=r, sec=time.time() - tl, split_merge=smlog, **chk)
        print(f"  level {l} (window {l}-{l + 1}{', root = 1 symbol' if Kp == 1 else ''}): window ll/seq {ll / TR.shape[0]:.4f}"
              f"  (restart {r})  {time.time() - tl:.1f}s  min merge cost {chk['min_merge']:.4f} +- {chk['se']:.4f}"
              f" ({'resolvable' if chk['resolvable'] else 'NOT resolvable'})"
              + (f"  split-merge: {sum(1 for x in smlog if x.get('accepted'))} accepted / {len(smlog)} rounds" if a.sm_rounds else ""), flush=True)
        btr = up_msgs(lo, hi, th, (TR.shape[0], nps, V)); bva = up_msgs(lov, hiv, th, (VA.shape[0], nps, V))
    PT = S.truth_tables(a.data); st = S.score(a.data, PT)
    out = dict(L=L, V=V, n_train=int(TR.shape[0]), levels_fit=info, sec_local=time.time() - t0)
    def report(tag, Pm):
        sm = S.score(a.data, Pm); gap = st["ll_per_seq"] - sm["ll_per_seq"]
        rec = {l: (sm["levels"][l]["nmi"], st["levels"][l]["nmi"]) for l in sm["levels"]}
        print(f"  {tag}: test gap {gap:.4f} nats/seq;  NMI model/ceiling by level: " +
              "  ".join(f"L{l} {m:.3f}/{c:.3f}{'' if int(l) < L else ' (root, not identifiable)'}" for l, (m, c) in rec.items()))
        return dict(gap=gap, levels=sm["levels"], ceiling=st["levels"])
    out["local"] = report("local EM", P)
    if a.polish:
        tp = time.time(); Pp = polish(P, TR, a.polish); out["polished"] = report(f"local + {a.polish} global sweeps", Pp)
        out["sec_polish"] = time.time() - tp
        np.savez(a.json.replace(".json", "_polished.npz") if a.json else "/tmp/local_polished.npz", **{f"P{l}": Pp[l] for l in Pp})
    print(f"  total {time.time() - t0:.0f}s")
    if a.json:
        np.savez(a.json.replace(".json", ".npz"), **{f"P{l}": P[l] for l in P})
        json.dump(out, open(a.json, "w"), indent=1, default=float)
