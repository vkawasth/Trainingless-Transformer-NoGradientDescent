"""Split-merge over symbol assignments of the induced productions (Corpus V).

The chunk inventory fixes the productions; what remains is which symbol each production
belongs to.  Moves on one level:
  SPLIT  a symbol into two, over every bipartition of its productions
  MERGE  two symbols
Each candidate grammar is rebuilt from the (deterministic) training parses — root collapsed to
one symbol — and scored by EXACT held-out log-likelihood; a move is accepted only if it improves
held-out likelihood (the acceptance rule O2 argues for).  Splits and merges alternate until neither helps.
"""
import json, sys, time, itertools, collections, numpy as np
from vtree import VGrammar, chart_io
from vchunks import induce

NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
D = json.load(open("cV/corpus.json")); g_true = VGrammar.from_json(D["grammar"])
tr, va = D["train"][:NTR], D["val"][:500]
t0 = time.time()
lv = induce(tr, verbose=False)
L, A = 3, 12
# structure: level-l chunk types; assignment[l][chunk] = symbol id
assign = {l: dict(lv[l - 1][1]) for l in (1, 2)}
segs = {l: lv[l - 1][0] for l in (1, 2)}      # segs[l][sentence] = list of chunks (over level-(l-1) labels as induced)
# the level-2 chunks are tuples of level-1 cluster ids as induced; keep them as keys


def build(assign):
    """Grammar from assignments; level-2 children use level-1 symbols, root productions are the
    sequences of level-2 symbols of each training sentence (one root symbol)."""
    V = max(max(a.values()) for a in assign.values()) + 1
    rules = {}
    for l in (1, 2):
        cnt = collections.Counter()
        for sg in segs[l]:
            for ch in sg:
                cnt[(assign[l][ch], ch)] += 1
        tot = collections.Counter()
        for (b, _), v in cnt.items():
            tot[b] += v
        rules[l] = [(b, ch, v / tot[b]) for (b, ch), v in sorted(cnt.items())]
    # level-2 productions are stated in induced level-1 ids, which ARE the level-1 symbols here
    root = collections.Counter(tuple(assign[2][ch] for ch in sg) for sg in segs[2])
    n = sum(root.values())
    rules[3] = [(0, ch, v / n) for ch, v in sorted(root.items())]
    prior = np.zeros(V); prior[0] = 1.0
    return VGrammar(L, V, A, rules, prior)


def heldout(g):
    ll = [chart_io(g, s)[0] for s in va]
    return float(np.mean(ll)), int(np.sum(np.isinf(ll)))


def split_candidates(assign, l):
    syms = sorted(set(assign[l].values()))
    new = max(max(a.values()) for a in assign.values()) + 1
    for b in syms:
        prods = [c for c, s in assign[l].items() if s == b]
        if len(prods) < 2:
            continue
        for r in range(1, len(prods) // 2 + 1):
            for grp in itertools.combinations(prods, r):
                if r * 2 == len(prods) and prods[0] not in grp:
                    continue
                a2 = {k: dict(v) for k, v in assign.items()}
                for c in grp:
                    a2[l][c] = new
                yield ("split", l, b, grp), a2


def merge_candidates(assign, l):
    syms = sorted(set(assign[l].values()))
    for a, b in itertools.combinations(syms, 2):
        a2 = {k: dict(v) for k, v in assign.items()}
        for c, s in a2[l].items():
            if s == b:
                a2[l][c] = a
        yield ("merge", l, a, b), a2


def relabel(assign):
    """Compact symbol ids per level (keeps level-2 chunk keys valid: they are level-1 ids)."""
    return assign


log = []
g = build(assign); best, unp = heldout(g)
print(f"start: held-out {best:.3f} (unparsable {unp}); truth {heldout(g_true)[0]:.3f}  [{time.time()-t0:.0f}s]")
log.append(("start", best))
improved = True
while improved:
    improved = False
    for kind in ("split", "merge"):
        for l in (2, 1):
            gen = split_candidates(assign, l) if kind == "split" else merge_candidates(assign, l)
            scored = []
            for move, a2 in gen:
                v, u = heldout(build(a2))
                if u == 0:
                    scored.append((v, move, a2))
            if not scored:
                continue
            v, move, a2 = max(scored, key=lambda z: z[0])
            if v > best + 1e-9:
                best, assign, improved = v, a2, True
                print(f"  accept {move[0]} at level {l}: held-out {best:.3f}  ({len(scored)} candidates) "
                      f"[{time.time()-t0:.0f}s]", flush=True)
                log.append((str(move), best))
g = build(assign)
nroot = len(g.rules[3])
print(f"final held-out {best:.3f}; root productions {nroot}; symbols per level "
      f"{ {l: len(set(assign[l].values())) for l in (1, 2)} }")
# purity of the final level-2 assignment against the truth
t1 = {tuple(c): b for b, c, _ in g_true.rules[1]}
m1 = {assign[1][c]: t1[c] for c in assign[1] if c in t1}
t2 = {tuple(c): b for b, c, _ in g_true.rules[2]}
by = collections.defaultdict(collections.Counter)
for c, s in assign[2].items():
    tc = tuple(m1.get(x, -1) for x in c)
    if tc in t2:
        by[s][t2[tc]] += 1
print("level-2 symbols -> true symbols:", {k: dict(v) for k, v in sorted(by.items())})
json.dump(dict(log=log, final=best, truth=heldout(g_true)[0], root_productions=nroot),
          open(f"cV/splitmerge_N{NTR}.json", "w"), indent=1)
