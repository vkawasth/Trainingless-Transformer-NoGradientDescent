"""The dual (Birman-Ko-Lee / Bessis) left normal form in B_n, computed with permutations (for Study 27j). New module; it modifies
nothing. Validated against the brute-force search of pce/hurwitz.py (check DG1-DG3).
Atoms are the band generators a_ij (i > j), whose permutation is the transposition (i j); the Garside element delta = a_(n,n-1) ... a_21
has permutation the n-cycle c. A simple element is determined by its permutation x with l_T(x) + l_T(x^-1 c) = l_T(c) = n - 1 (Bessis:
the interval [1, c] in the absolute order), l_T(x) = n - (number of cycles of x).
  left_normal_form(word, n)  word: list of ((i, j), e) atoms a_ij^e (i > j, e = +-1). Returns (inf, [simple permutations]), with
                             sup = inf + len(factors) and canonical length = len(factors)
  from_artin(word)           an Artin word sigma_k^e as atoms: sigma_k = a_(k+1, k)
"""
from __future__ import annotations
def _mul(p, q):
    """(p q)(x) = p(q(x)); permutations as tuples on 0..n-1."""
    return tuple(p[q[i]] for i in range(len(p)))
def _inv(p):
    r = [0] * len(p)
    for i, x in enumerate(p): r[x] = i
    return tuple(r)
def _cycles(p):
    seen = [False] * len(p); c = 0
    for i in range(len(p)):
        if not seen[i]:
            c += 1; j = i
            while not seen[j]: seen[j] = True; j = p[j]
    return c
def lT(p): return len(p) - _cycles(p)
def _tr(i, j, n):
    t = list(range(n)); t[i - 1], t[j - 1] = t[j - 1], t[i - 1]; return tuple(t)
def coxeter(n):
    """the permutation of delta = a_(n,n-1) ... a_21, composed left to right as braids act (product of transpositions)."""
    p = tuple(range(n))
    for k in range(n, 1, -1): p = _mul(p, _tr(k, k - 1, n))
    return p
def is_simple(x, c): return lT(x) + lT(_mul(_inv(x), c)) == lT(c)
def tau(x, c):
    """the conjugation that moves delta^-1 leftwards past a simple: y delta^-1 = delta^-1 (delta y delta^-1); on permutations c x c^-1.
    (The opposite convention, c^-1 x c, disagreed with the brute-force search in 53 of 150 words in B_4; check DG1.)"""
    return _mul(_mul(c, x), _inv(c))
def _left_weight(s, t, c, n, atoms):
    """make the pair of simples (s, t) left-weighted: move atoms a <= t (left divisors of t) into s while s a stays simple."""
    changed = True
    while changed:
        changed = False
        for a in atoms:
            at = _mul(a, t)                                   # a^-1 t (a is an involution as a permutation)
            if lT(at) == lT(t) - 1 and is_simple(_mul(s, a), c) and lT(_mul(s, a)) == lT(s) + 1:
                s, t = _mul(s, a), at; changed = True; break
    return s, t
def left_normal_form(word, n):
    c = coxeter(n); e = tuple(range(n)); atoms = [_tr(i, j, n) for i in range(2, n + 1) for j in range(1, i)]
    # x = delta^-r P: write each a^-1 as delta^-1 (delta a^-1) and move the delta^-1 to the front with tau
    r = sum(1 for _, s in word if s < 0); factors = []                  # P as a product of simples, built left to right
    pending = []
    k = 0
    for (i, j), s in word:
        a = _tr(i, j, n)
        if s > 0: simple = a
        else: simple = _mul(c, a); k += 1                             # delta a^-1, permutation c a
        # this simple sits after k delta^-1's that were moved to the front past the earlier letters: conjugate by delta^(r-k)? We move
        # each delta^-1 leftwards past everything before it, which conjugates those earlier factors; equivalently, conjugate this
        # factor by tau^(number of delta^-1 still to come) at the end. Record (simple, k) and fix up after the loop.
        pending.append((simple, k))
    for simple, kk in pending:
        x = simple
        for _ in range(r - kk): x = tau(x, c)
        factors.append(x)
    # left-greedy normal form of the positive element (product of simples), adding one simple at a time
    nf = []
    for x in factors:
        nf.append(x)
        for i in range(len(nf) - 1, 0, -1):
            s, t = _left_weight(nf[i - 1], nf[i], c, n, atoms)
            if (s, t) == (nf[i - 1], nf[i]): break
            nf[i - 1], nf[i] = s, t
        nf = [y for y in nf if y != e] if nf and nf[-1] == e else nf
    nf = [y for y in nf if y != e]
    inf = -r
    while nf and nf[0] == c: nf.pop(0); inf += 1
    return inf, nf
def from_artin(word): return [((k + 1, k), e) for k, e in word]
