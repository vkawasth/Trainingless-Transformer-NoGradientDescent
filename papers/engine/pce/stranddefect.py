"""The strand-addition defect (Fadell-Neuwirth). New module; it modifies nothing.

Conventions as in braidknot (words of +-i, strands 1..n by position at the start, Artin's faithful right action). Strands are labelled by
their starting position; all braids here are pure, so every strand also ends where it began.

  pure_gen(i, j)        Artin's pure generator A_ij (i < j): (sigma_{j-1} ... sigma_{i+1}) sigma_i^2 (sigma_{i+1}^{-1} ... sigma_{j-1}^{-1})
  forget(word, n, k)    delete strand k (1-based label) from a braid on n strands; the result is a word on n - 1 strands
  section_far(word, n)  s_far: add a straight strand at position n + 1 that never crosses anything (the newcomer far away)
  section_back(word, n, k)  s_k: add a straight strand at position k + 1 that every other strand passes over (k = 0..n); s_n = s_far
  section_echo(word, n, i)  e_i: add a strand running parallel to strand i, just to its right (blackboard framing): the newcomer echoes i
  defect(bp, b, n, sec) Delta_s(bp, b) = bp s(b)^{-1}, defined only when forget(bp) = b (raises otherwise)
  loop_word(f, n)       for f in the kernel of forget_{n+1} (newcomer at position n + 1): the newcomer's loop as a reduced word in
                        Artin's combing generators A_{i,n+1} of the free kernel, read from Artin's action (f(x_{n+1}) = l x_{n+1} l^{-1};
                        delete the x_{n+1} letters, reverse because the action is on the right, change basis x -> z). Checked in
                        check_stranddefect: A_{i,n+1} -> letter i, a homomorphism, and from_loop_word(loop_word(f)) = f (the combing)
  magnus_coords(w, n)   degree-1 and degree-2 Magnus coefficients of a free-group word (x_j -> 1 + X_j)
  linking(word, n)      linking numbers of pure-braid strands, half the signed crossings between them
"""
from __future__ import annotations
from . import braidknot as BK

inv = BK._inv; red = BK._red


def pure_gen(i, j):
    pre = list(range(j - 1, i, -1))
    return pre + [i, i] + [-x for x in reversed(pre)]


def forget(word, n, k):
    order = list(range(1, n + 1)); out = []
    for g in word:
        i = abs(g); a, b = order[i - 1], order[i]
        if k not in (a, b):
            pk = order.index(k) + 1; out.append((i - 1 if pk < i else i) * (1 if g > 0 else -1))
        order[i - 1], order[i] = b, a
    return out


def section_far(word, n): return list(word)


def section_back(word, n, k):
    """the newcomer sits at position k+1 and every other strand passes over it; strands left of it keep their indices"""
    out = []
    for g in word:
        i, e = abs(g), (1 if g > 0 else -1)
        if i < k: out.append(g)
        elif i > k: out.append((i + 1) * e)
        else: out += [k, (k + 1) * e, -k]
    return out


def section_echo(word, n, i):
    order = list(range(1, n + 1)); out = []
    for g in word:
        m, e = abs(g), (1 if g > 0 else -1); a, b = order[m - 1], order[m]; P = order.index(i) + 1
        if i == a: out += [(m + 1) * e, m * e]              # i (with its echo) moves right past b
        elif i == b: out += [(m - 1 + 1) * e, (m + 1) * e]  # b moves right past i, then past the echo
        else: out.append((m if m < P else m + 1) * e)
        order[m - 1], order[m] = b, a
    return out


def section(word, n, sec):
    kind = sec[0]
    if kind == "far": return section_far(word, n)
    if kind == "back": return section_back(word, n, sec[1])
    if kind == "echo": return section_echo(word, n, sec[1])
    raise ValueError(sec)


def newcomer_label(n, sec):
    """the starting position of the added strand in s(b)"""
    if sec[0] == "far": return n + 1
    if sec[0] == "back": return sec[1] + 1
    return sec[1] + 1                                            # echo of i sits just right of i at the start


def defect(bp, b, n, sec=("far",)):
    if not BK.same_braid(forget(bp, n + 1, newcomer_label(n, sec)), b, n): raise ValueError("compatibility fails: forget(bp) != b")
    return list(bp) + inv(section(b, n, sec))


def _x_in_z(n):
    """x_j written in the basis z_i = (x_n ... x_{i+1}) x_i (x_n ... x_{i+1})^{-1}: x_n = z_n, x_i = c_i^{-1} z_i c_i, c_i = x_n ... x_{i+1}"""
    X = {}
    for i in range(n, 0, -1):
        c = []
        for j in range(n, i, -1): c += X[j]
        X[i] = red(inv(c) + [i] + c)
    return X


def loop_word(f, n):
    """the defect as a reduced word in Artin's combing generators A_{1,n+1}, ..., A_{n,n+1} (letter +-i stands for A_{i,n+1}^{+-1})"""
    img = BK.artin(f, n + 1)[n + 1]; m = (len(img) - 1) // 2
    if img[m] != n + 1 or red(img[:m] + [n + 1] + inv(img[:m])) != img: raise ValueError("newcomer strand is not pure")
    lx = red([x for x in reversed(img[:m]) if abs(x) != n + 1])          # Artin's action is a right action: reverse the letters
    X = _x_in_z(n); out = []
    for g in lx: out += X[g] if g > 0 else inv(X[-g])
    return red(out)


def from_loop_word(w, n):
    out = []
    for g in w: out += pure_gen(abs(g), n + 1) if g > 0 else inv(pure_gen(-g, n + 1))
    return out


def magnus_coords(w, n):
    M = BK.magnus(w, 2)
    d1 = {j: round(M.get((j,), 0.0)) for j in range(1, n + 1)}
    d2 = {(i, j): round(M.get((i, j), 0.0)) for i in range(1, n + 1) for j in range(1, n + 1) if i != j}
    return d1, d2


def linking(word, n):
    order = list(range(1, n + 1)); lk = {}
    for g in word:
        i = abs(g); a, b = order[i - 1], order[i]; key = (min(a, b), max(a, b))
        lk[key] = lk.get(key, 0) + (1 if g > 0 else -1); order[i - 1], order[i] = b, a
    return {k: v // 2 for k, v in lk.items() if v}
