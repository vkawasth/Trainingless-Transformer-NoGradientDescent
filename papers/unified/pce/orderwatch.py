"""Order watch (section sec:orderwatch): signed counts of ordered chains in the dependence order of an event history, a Milnor-type
coefficient for 'rare events in a certain order', computed online. New module; it modifies nothing.
Events are ((i, j), sign, round) on the revision graph; dependence = sharing a source (outerrelay.dependent).
  poset(word)                         the dependence partial order (transitive closure) of a word
  chain_count(word, path, k=0)        signed count of embeddings of the chain 'path' (a list of edges from the initiator to the acting
                                      member, in order): observed links in path order; adjacent links ordered in the dependence order;
                                      links separated by up to k hidden links ordered by round; sign = product of the links' signs
  prefix_len(word, path)              the longest initial segment of the path embedded with positive sign (early-warning progress)
  score(word, paths, k)               max over paths of chain_count
  strict_gap=True                     across a gap, require a strictly later round (Study 27h used the default, which breaks ties within a
                                      round by position: that is not a trace invariant, finding ZH2; the strict form is, ZH13)
  signed=False                        counts only embeddings whose links all carry sign +1 (the positive count, nu): it resists a decoy
                                      reversal but is not invariant under padding with cancelling pairs (sec:orderwatch)
"""
from __future__ import annotations
import itertools
from . import outerrelay as O
def poset(word):
    n = len(word); below = [set() for _ in range(n)]
    for b in range(n):
        for a in range(b):
            if O.dependent(word[a][:2], word[b][:2]): below[b] |= below[a] | {a}
    return below
def _matches(ev, edge): return tuple(sorted(ev[0])) == tuple(sorted(edge))
def chain_count(word, path, k=0, signed=True, strict_gap=False):
    """word: list of (edge, sign, round); path: list of edges. Counts signed embeddings using at least len(path) - k links."""
    below = poset([(ev[0], ev[1]) for ev in word]); L = len(path); total = 0
    occ = [[i for i, ev in enumerate(word) if _matches(ev, edge)] for edge in path]
    for keep in itertools.combinations(range(L), L - k):
        lists = [occ[j] for j in keep]
        if any(not l for l in lists): continue
        def rec(pos, prev_idx, prev_link, sign):
            if pos == len(keep): return sign
            s = 0; j = keep[pos]
            for i in lists[pos]:
                if not signed and word[i][1] < 0: continue
                if prev_idx is not None:
                    if j == prev_link + 1:
                        if prev_idx not in below[i]: continue
                    else:
                        if strict_gap:
                            if not word[i][2] > word[prev_idx][2]: continue
                        elif not (word[i][2] > word[prev_idx][2] or (word[i][2] == word[prev_idx][2] and i > prev_idx)): continue
                s += rec(pos + 1, i, j, sign * word[i][1])
            return s
        total += rec(0, None, None, 1)
        if k == 0: break
    return total
def prefix_len(word, path):
    best = 0
    for l in range(1, len(path) + 1):
        if chain_count(word, path[:l], 0) > 0: best = l
        else: break
    return best
def score(word, paths, k=0, signed=True): return max(chain_count(word, p, k, signed) for p in paths)
