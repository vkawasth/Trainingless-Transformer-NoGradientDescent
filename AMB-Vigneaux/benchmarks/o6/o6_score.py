#!/usr/bin/env python3
"""O6 scorer: held-out likelihood and per-level structure recovery for a submitted grammar.

    python o6_score.py --data DIR --model submission.npz     (keys P1..PL, P_l[parent, left, right])
    python o6_score.py --data DIR --truth                     (scores the generating grammar)

Metrics on the TEST split (never used for fitting or model selection):
  ll_per_seq      mean log p(sequence) under the submitted grammar (root uniform over its alphabet)
  gap             ll_true - ll_model   (nats/sequence; 0 = as good as the generating grammar)
  level l:
    nmi           normalised mutual information between the model's MAP symbol at each level-l node
                  and the true latent symbol (label permutations do not matter)
    acc_matched   accuracy after the best one-to-one matching of model symbols to true symbols
The submission may use any alphabet size per level; hierarchy codes must be LEARNED from train_ids only.
"""
import argparse, json
import numpy as np
from scipy.optimize import linear_sum_assignment


def up(P, lo, hi):
    t = np.einsum("bnx,pxy->bnpy", lo, P)
    return np.einsum("bnpy,bny->bnp", t, hi)


def inside(X, P, L, nleaf):
    oh = np.zeros(X.shape + (nleaf,)); np.put_along_axis(oh, X[..., None], 1.0, -1)
    tabs, logs = [oh], np.zeros(X.shape[0])
    for l in range(1, L + 1):
        t = up(P[l], tabs[-1][:, 0::2], tabs[-1][:, 1::2])
        s = t.sum(-1, keepdims=True).clip(1e-300)          # per-node rescaling, tracked in logs
        logs += np.log(s[..., 0]).sum(-1)
        tabs.append(t / s)
    return tabs, logs


def outside(tabs, P, L):
    B = tabs[0].shape[0]; vr = P[L].shape[0]
    outs = [None] * (L + 1); outs[L] = np.full((B, 1, vr), 1.0 / vr)
    for l in range(L, 0, -1):
        lo, hi = tabs[l - 1][:, 0::2], tabs[l - 1][:, 1::2]
        t = np.einsum("bnp,pxy->bnxy", outs[l], P[l])
        lo_o = np.einsum("bnxy,bny->bnx", t, hi); hi_o = np.einsum("bnxy,bnx->bny", t, lo)
        ch = np.zeros((B, lo.shape[1] * 2, lo.shape[2])); ch[:, 0::2] = lo_o; ch[:, 1::2] = hi_o
        ch /= ch.sum(-1, keepdims=True).clip(1e-300)
        outs[l - 1] = ch
    return outs


def nmi(a, b):
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    C = np.zeros((len(ua), len(ub))); np.add.at(C, (ia, ib), 1); C /= C.sum()
    pa, pb = C.sum(1), C.sum(0)
    nz = C > 0
    I = (C[nz] * np.log(C[nz] / np.outer(pa, pb)[nz])).sum()
    H = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    d = np.sqrt(H(pa) * H(pb))
    return float(I / d) if d > 0 else 1.0, C


def score(data, P, batch=256):
    M = json.load(open(f"{data}/rhm_meta.json"))
    L, SEQ, NLEAF = M["depth"], M["seq_len"], M["nleaf"]
    X = np.array(json.load(open(f"{data}/test_ids.json")), dtype=np.int64).reshape(-1, SEQ)
    LAT = json.load(open(f"{data}/test_latents.json"))
    ll, maps = [], {l: [] for l in range(1, L + 1)}
    for s in range(0, len(X), batch):
        tabs, logs = inside(X[s:s + batch], P, L, NLEAF)
        vr = P[L].shape[0]
        ll.append(logs + np.log(tabs[L][:, 0, :].sum(-1).clip(1e-300) / vr))
        outs = outside(tabs, P, L)
        for l in range(1, L + 1):
            maps[l].append((tabs[l] * outs[l]).argmax(-1))
    out = dict(ll_per_seq=float(np.concatenate(ll).mean()), levels={})
    for l in range(1, L + 1):
        pred = np.concatenate(maps[l]).ravel()
        true = np.array([lat[l - 1] for lat in LAT]).ravel()
        v, C = nmi(pred, true)
        r, c = linear_sum_assignment(-C)
        out["levels"][l] = dict(nmi=round(v, 4), acc_matched=round(float(C[r, c].sum()), 4),
                                identifiable=(l < L))   # the root level is not identifiable (see merge_strata.py)
    return out


def truth_tables(data):
    M = json.load(open(f"{data}/rhm_meta.json"))
    L, V, NLEAF = M["depth"], M["nsym"], M["nleaf"]
    P = {}
    for l in range(1, L + 1):
        c = NLEAF if l == 1 else V
        t = np.zeros((V, c, c))
        for s_, r in M["rules_all"][str(l)].items():
            for (x, y) in r:
                t[int(s_), x, y] += 1.0 / len(r)
        P[l] = t
    return P


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--truth", action="store_true")
    a = ap.parse_args()
    PT = truth_tables(a.data)
    st = score(a.data, PT)
    if a.truth:
        print(json.dumps(st, indent=1))
    else:
        Z = np.load(a.model); L = len(PT)
        sm = score(a.data, {l: Z[f"P{l}"] for l in range(1, L + 1)})
        sm["ll_true"] = st["ll_per_seq"]; sm["gap"] = st["ll_per_seq"] - sm["ll_per_seq"]
        sm["levels_truth"] = st["levels"]
        print(json.dumps(sm, indent=1))
