#!/usr/bin/env python3
"""How far is the true grammar from the singular strata where two symbols coincide?  (gradient-free, exact inside)

The parameter space of a hierarchical grammar modulo relabelling of symbols is singular along the strata where two
symbols at one level coincide (the model then has V-1 symbols there). For every level l and every pair (s, s') of
level-l symbols we form the MERGED grammar: rows s, s' of P[l] averaged (weighted by their usage), the child axes of
level l+1 collapsed (s and s' become one child), the root prior summed at l = L. Then
    Delta_l(s, s') = mean over held-out sequences of  [log p_true(x) - log p_merged(x)]      (nats / sequence)
is the likelihood distance from the truth to that stratum, with a paired standard error over sequences.
A pair is RESOLVABLE at held-out size n if Delta > 2 se (and > 1e-9).
Proposition (the root level is not identifiable). log p(x) depends on the root level only through the mixture
sum_s pi_s P_L[s]: merging any two root symbols with usage weights leaves the likelihood EXACTLY unchanged
(Delta = 0 for every root pair). Root-symbol structure lies on a positive-dimensional fibre of constant likelihood
of the moduli of grammars; no learner can recover it from data, and recovery of root symbols against the truth is
not a well-posed target. If Delta at the top levels is below the likelihood gaps
that EM + split-merge leaves (BASELINE.md), likelihood cannot tell the true top-level structure from a merged one:
this is the singular-strata explanation of 'likelihood does not certify structure', measured, not conjectured.

    python3 merge_strata.py --data runs/V8L6 --n-val 4000 --json results/merge_V8L6.json
"""
import json, argparse, itertools
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--n-val", type=int, default=4000)
ap.add_argument("--max-pairs", type=int, default=28)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--json", default="")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
va = np.array(json.load(open(f"{a.data}/val_ids.json")), dtype=np.int64)
X = va[:(len(va) // SEQ) * SEQ].reshape(-1, SEQ)[:a.n_val]


def card0(l): return NLEAF if l == 1 else V


PT = {}
for l in range(1, L + 1):
    t = np.zeros((V, card0(l), card0(l)))
    for s, r in M["rules_all"][str(l)].items():
        for (x, y) in r:
            t[int(s), x, y] += 1.0 / len(r)
    PT[l] = t
PI = np.full(V, 1.0 / V)                                       # root prior


def per_seq_loglik(X, P, pi):
    soft = np.zeros(X.shape + (NLEAF,)); np.put_along_axis(soft, X[..., None], 1.0, -1)
    logs = np.zeros(X.shape[0])
    for l in range(1, L + 1):
        lo, hi = soft[:, 0::2], soft[:, 1::2]
        t = np.einsum("bnx,sxy,bny->bns", lo, P[l], hi, optimize=True)
        s = np.maximum(t.sum(-1, keepdims=True), 1e-300)
        logs += np.log(s[..., 0]).sum(-1); soft = t / s
    return logs + np.log(np.maximum((soft[:, 0, :] * pi[None, :]).sum(-1), 1e-300))


def usage(P, pi):
    """expected number of times each symbol is used per sequence at each level (top-down)"""
    u = {L: pi.copy()}
    for l in range(L, 1, -1):
        T = P[l] * u[l][:, None, None]
        u[l - 1] = T.sum((0, 2)) + T.sum((0, 1))
    return u


def merged(P, pi, l, s, s2, u):
    Q = {k: v.copy() for k, v in P.items()}; keep = [i for i in range(P[l].shape[0]) if i != s2]
    w = u[l][[s, s2]]; w = w / max(w.sum(), 1e-300)
    row = w[0] * P[l][s] + w[1] * P[l][s2]
    Q[l] = P[l][keep]; Q[l][keep.index(s)] = row
    if l < L:
        T = P[l + 1].copy()
        T[:, s, :] += T[:, s2, :]; T = np.delete(T, s2, axis=1)
        T[:, :, s] += T[:, :, s2]; T = np.delete(T, s2, axis=2)
        Q[l + 1] = T
        pim = pi
    else:
        pim = pi.copy(); pim[s] += pim[s2]; pim = np.delete(pim, s2)
    return Q, pim


if __name__ == "__main__":
    base = per_seq_loglik(X, PT, PI); u = usage(PT, PI)
    out = dict(L=L, V=V, nleaf=NLEAF, n_val=int(X.shape[0]), ll_true=float(base.mean()), levels={})
    print(f"L={L} V={V} leaves={NLEAF}  held-out n={X.shape[0]}  true log-lik {base.mean():.4f} nats/seq")
    print(f"{'level':>5s} | {'median Delta':>12s} | {'min Delta':>10s} | {'median se':>9s} | resolvable pairs (Delta > 2 se)")
    for l in range(1, L + 1):
        pairs = list(itertools.combinations(range(V), 2))
        if len(pairs) > a.max_pairs:
            pairs = [pairs[i] for i in rng.choice(len(pairs), a.max_pairs, replace=False)]
        D, S, used = [], [], []
        for s, s2 in pairs:
            used.append(u[l][s] > 1e-12 and u[l][s2] > 1e-12)
            Q, pim = merged(PT, PI, l, s, s2, u)
            diff = base - per_seq_loglik(X, Q, pim)
            D.append(diff.mean()); S.append(diff.std(ddof=1) / np.sqrt(len(diff)))
        D, S, used = np.array(D), np.array(S), np.array(used); res = float((D > np.maximum(2 * S, 1e-9)).mean())
        res_used = float((D[used] > np.maximum(2 * S[used], 1e-9)).mean()) if used.any() else 0.0
        unused = [i for i in range(V) if u[l][i] < 1e-12]
        out["levels"][l] = dict(median=float(np.median(D)), min=float(D.min()), max=float(D.max()),
                                median_used=float(np.median(D[used])) if used.any() else 0.0,
                                min_used=float(D[used].min()) if used.any() else 0.0, frac_resolvable_used=res_used,
                                median_se=float(np.median(S)), frac_resolvable=res, n_pairs=len(pairs),
                                exactly_zero=int((np.abs(D) < 1e-9).sum()), unused_symbols=unused)
        print(f"{l:5d} | {np.median(D):12.4f} | {D.min():10.4f} | {np.median(S):9.4f} | {res:.2f} of {len(pairs)}"
              f"   (exactly 0: {int((np.abs(D) < 1e-9).sum())}; unused symbols {unused}; used pairs: median"
              f" {np.median(D[used]) if used.any() else 0:.4f}, min {D[used].min() if used.any() else 0:.4f}, resolvable {res_used:.2f})")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
