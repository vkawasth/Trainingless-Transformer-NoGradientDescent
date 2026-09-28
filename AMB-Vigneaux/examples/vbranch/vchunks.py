"""Bottom-up chunk induction for Corpus V, then initialisation of the positional learner.

Level by level (tokens -> level-1 symbols -> level-2 symbols -> root):
  1. candidate chunks = substrings of length 1..KMAX+1 seen >= MINC times
  2. EM for a unigram chunk model over segmentations (forward-backward), then Viterbi
  3. cluster chunks into V symbols by their neighbouring-chunk contexts (spectral + k-means)
  4. relabel every sentence as its sequence of symbol ids; recurse
The Viterbi structure gives counts (b, k, position, child) for the positional model.
"""
from __future__ import annotations

import collections, math
import numpy as np

KMAX = 5


def candidates(seqs, maxlen=KMAX + 1, minc=10):
    c = collections.Counter()
    for s in seqs:
        n = len(s)
        for i in range(n):
            for w in range(1, min(maxlen, n - i) + 1):
                c[tuple(s[i:i + w])] += 1
    return {k: v for k, v in c.items() if v >= minc}


def segment_em(seqs, lex, iters=15, maxlen=KMAX + 1, prune=1e-5):
    """Unigram chunk model; returns probabilities and Viterbi segmentations."""
    p = {k: float(v) for k, v in lex.items()}
    z = sum(p.values()); p = {k: v / z for k, v in p.items()}
    for _ in range(iters):
        cnt = collections.defaultdict(float)
        for s in seqs:
            n = len(s)
            a = np.zeros(n + 1); a[0] = 1.0
            for j in range(1, n + 1):
                for w in range(1, min(maxlen, j) + 1):
                    q = p.get(tuple(s[j - w:j]))
                    if q:
                        a[j] += a[j - w] * q
                sc = a[j]
                if sc > 0:                          # rescale to avoid underflow
                    pass
            b = np.zeros(n + 1); b[n] = 1.0
            for i in range(n - 1, -1, -1):
                for w in range(1, min(maxlen, n - i) + 1):
                    q = p.get(tuple(s[i:i + w]))
                    if q:
                        b[i] += q * b[i + w]
            Z = a[n]
            if Z <= 0:
                continue
            for i in range(n):
                for w in range(1, min(maxlen, n - i) + 1):
                    ch = tuple(s[i:i + w]); q = p.get(ch)
                    if q:
                        cnt[ch] += a[i] * q * b[i + w] / Z
        tot = sum(cnt.values())
        p = {k: v / tot for k, v in cnt.items() if v / tot > prune}
    segs = []
    for s in seqs:
        n = len(s)
        best = np.full(n + 1, -np.inf); best[0] = 0.0; back = [0] * (n + 1)
        for j in range(1, n + 1):
            for w in range(1, min(maxlen, j) + 1):
                q = p.get(tuple(s[j - w:j]))
                if q and best[j - w] + math.log(q) > best[j]:
                    best[j] = best[j - w] + math.log(q); back[j] = w
        out, j = [], n
        while j > 0:
            w = back[j]
            if w == 0:                               # unsegmentable: fall back to single items
                out.append(tuple(s[j - 1:j])); j -= 1
            else:
                out.append(tuple(s[j - w:j])); j -= w
        segs.append(out[::-1])
    return p, segs


def prune_concatenations(p, rounds=3, seqs=None):
    """Drop chunk types that are the concatenation of two other kept types (single items are
    always kept); re-run EM after each round.  Counters the unigram model's bias toward
    under-segmentation."""
    segs = None
    for _ in range(rounds):
        keep = set(p)
        drop = set()
        for c in p:
            if len(c) < 2:
                continue
            if any(c[:i] in keep and c[i:] in keep for i in range(1, len(c))):
                drop.add(c)
        if not drop:
            break
        lex = {c: 1.0 for c in keep - drop}
        for sq in seqs:                             # singles always available
            for a in sq:
                lex.setdefault((a,), 1.0)
        p, segs = segment_em(seqs, lex)
    if segs is None:
        _, segs = segment_em(seqs, p, iters=0)
    return p, segs


def cluster_chunks(segs, V, rng, dim=None):
    """Spectral clustering of chunk types by left/right neighbouring chunk identity."""
    types = sorted({c for sg in segs for c in sg}, key=lambda c: (len(c), c))
    idx = {c: i for i, c in enumerate(types)}
    T = len(types)
    F = np.zeros((T, 2 * T + 2))
    for sg in segs:
        for t, c in enumerate(sg):
            i = idx[c]
            F[i, idx[sg[t - 1]] if t > 0 else 2 * T] += 1
            F[i, T + idx[sg[t + 1]] if t + 1 < len(sg) else 2 * T + 1] += 1
    freq = F.sum(1)
    X = F / np.maximum(freq[:, None], 1)
    X = np.sqrt(X)                                   # Hellinger
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    d = dim or min(V * 2, len(S))
    Y = U[:, :d] * S[:d]
    # weighted k-means++ (weights = chunk frequency)
    w = freq / freq.sum()
    centers = [Y[rng.choice(T, p=w)]]
    for _ in range(1, V):
        dd = np.min([((Y - c) ** 2).sum(1) for c in centers], 0) * w
        centers.append(Y[rng.choice(T, p=dd / dd.sum())] if dd.sum() > 0 else Y[rng.integers(T)])
    C = np.array(centers)
    for _ in range(100):
        lab = np.argmin(((Y[:, None] - C[None]) ** 2).sum(-1), 1)
        newC = np.array([np.average(Y[lab == k], 0, weights=freq[lab == k]) if (lab == k).any() else C[k]
                         for k in range(V)])
        if np.allclose(newC, C):
            break
        C = newC
    return {c: int(lab[idx[c]]) for c in types}


def induce(train, V=8, L=3, seed=0, minc=10, verbose=True):
    rng = np.random.default_rng(seed)
    seqs = [list(s) for s in train]
    levels = []
    for l in range(1, L + 1):
        if l < L:
            lex = candidates(seqs, minc=minc)
            p, segs = segment_em(seqs, lex)
            p, segs = prune_concatenations(p, seqs=seqs)
            lab = cluster_chunks(segs, V, rng)
        else:                                        # the root: each whole sequence is one production
            segs = [[tuple(s)] for s in seqs]
            lab = {c: 0 for sg in segs for c in sg}
            if len({len(c) for sg in segs for c in sg if len(c) > KMAX + 1}):
                pass
        levels.append((segs, lab))
        if verbose:
            types = {c for sg in segs for c in sg}
            print(f"  level {l}: {len(types)} chunk types, mean chunk length "
                  f"{np.mean([len(c) for sg in segs for c in sg]):.2f}")
        seqs = [[lab[c] for c in sg] for sg in segs]
    return levels


def init_from_induction(levels, V, A, L, smooth=0.1, rng=None):
    """Positional parameters from the induced (Viterbi) structure.
    The root level is fitted with V symbols by assigning each sentence-production to a
    random root symbol (the root carries no information about the data; EM refines it)."""
    rng = rng or np.random.default_rng(0)
    P = {}
    for l in range(1, L + 1):
        segs, lab = levels[l - 1]
        Cn = A if l == 1 else V
        K = np.full((V, KMAX + 1), smooth); H = np.full((V, KMAX + 1, Cn), smooth)
        R = np.full((V, KMAX + 1, KMAX, Cn), smooth)
        for sg in segs:
            for ch in sg:
                if len(ch) > KMAX + 1:
                    continue
                b = lab[ch] if l < L else int(rng.integers(V))
                k = len(ch) - 1
                K[b, k] += 1; H[b, k, ch[0]] += 1
                for t, c in enumerate(ch[1:]):
                    R[b, k, t, c] += 1
        P[f"K{l}"] = K / K.sum(-1, keepdims=True)
        P[f"H{l}"] = H / H.sum(-1, keepdims=True)
        P[f"R{l}"] = R / R.sum(-1, keepdims=True)
    return P


def score_level1(levels, grammar):
    """Chunk inventory vs the true level-1 productions, and cluster purity."""
    true = {tuple(c): b for b, c, _ in grammar.rules[1]}
    segs, lab = levels[0]
    found = {c for sg in segs for c in sg}
    rec = np.mean([c in found for c in true])
    freq = collections.Counter(c for sg in segs for c in sg)
    prec = sum(v for c, v in freq.items() if c in true) / sum(freq.values())
    # purity of clusters over true level-1 productions
    by = collections.defaultdict(collections.Counter)
    for c, b in true.items():
        if c in lab:
            by[lab[c]][b] += freq[c]
    pur = sum(max(v.values()) for v in by.values()) / max(sum(sum(v.values()) for v in by.values()), 1)
    return dict(recall_true_productions=float(rec), token_precision=float(prec), cluster_purity=float(pur))
