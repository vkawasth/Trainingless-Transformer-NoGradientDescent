"""Belief presheaves over admitted sources, tamper budgets, conflicts and explanations (sec. 'Beliefs under contamination' of the
companion note). New module; it modifies nothing.
Setting: n binary variables (+-1), the 2^n atoms of a joint law p, and observation cells. A cell is (name, context, table): context a
tuple of variable indices, table a dict from assignments of the context to probabilities. A cell constrains p by M_c p = b_c.
  marg(n, ctx)                      the marginal map of a context (rows: context assignments in lexicographic order)
  system(n, cells)                  stacked (M, b) with a normalisation row
  feasible(n, cells, tol)           is there a law with every cell's table within L1 tolerance tol?  (tol = 0: exact)
  residual(n, cells)                the smallest total L1 discrepancy of any law from the cells (0 iff coherent)
  interval(n, cells, a)             [min, max] of a.p over the fibre (None if empty)
  forced_residual(n, honest, coalition)   min over laws matching the honest cells exactly of the L1 distance of the coalition's reports
                                    from the law's tables: the incoherence a report forces
  farkas(n, cells)                  for an exactly infeasible system, a sparse certificate y (A^T y >= 0, b.y = -1) and the cells it touches
  conflicts(n, cells, tol)          minimal conflicts by enumeration
  explanations(n, cells, k, tol)    every removable set of size <= k (not only the minimal ones)
  robust_interval(n, cells, a, k)   the union over removable sets H, |H| <= k, of the interval of a over the fibre of the rest
  exposure(cells)                   overlap exposure (shares a variable with another cell) and cycle exposure (lies on a cycle of the
                                    cell-intersection graph)
  artin_trivial(word, m)            is a braid word on m strands trivial? (Artin's faithful action on the free group)
  bubble_word(before, after, sign)  the braid word of a change of order (adjacent transpositions)
  trace_normal_form(word, indep)    Cartier-Foata normal form of a word of event labels under an independence relation
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog


def atoms(n): return list(itertools.product((1, -1), repeat=n))


def marg(n, ctx):
    A = atoms(n); keys = list(itertools.product((1, -1), repeat=len(ctx)))
    M = np.zeros((len(keys), len(A)))
    for j, x in enumerate(A): M[keys.index(tuple(x[i] for i in ctx)), j] = 1.0
    return M, keys


def system(n, cells):
    Ms, bs = [np.ones((1, 2 ** n))], [np.array([1.0])]
    for name, ctx, table in cells:
        M, keys = marg(n, ctx); Ms.append(M); bs.append(np.array([table[k] for k in keys]))
    return np.vstack(Ms), np.concatenate(bs)


def _l1_lp(n, cells, extra_eq=None):
    """min sum |M p - b| over cells, p a law (and optional extra exact equalities)"""
    N = 2 ** n; M, b = system(n, cells); M, b = M[1:], b[1:]; r = len(b)
    c = np.concatenate([np.zeros(N), np.ones(2 * r)])
    Aeq = [np.concatenate([np.ones(N), np.zeros(2 * r)])]; beq = [1.0]
    for i in range(r):
        row = np.concatenate([M[i], np.zeros(2 * r)]); row[N + i] = -1; row[N + r + i] = 1; Aeq.append(row); beq.append(b[i])
    if extra_eq is not None:
        for row, v in zip(*extra_eq): Aeq.append(np.concatenate([row, np.zeros(2 * r)])); beq.append(v)
    res = linprog(c, A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=[(0, None)] * (N + 2 * r), method="highs")
    return res


def residual(n, cells):
    if not cells: return 0.0
    res = _l1_lp(n, cells); return float(res.fun) if res.status == 0 else np.inf


def feasible(n, cells, tol=0.0):
    return residual(n, cells) <= tol + 1e-9


def interval(n, cells, a, tol=0.0):
    N = 2 ** n; a = np.asarray(a, float)
    if tol == 0.0:
        M, b = system(n, cells); out = []
        for s in (1, -1):
            res = linprog(s * a, A_eq=M, b_eq=b, bounds=[(0, None)] * N, method="highs")
            if res.status != 0: return None
            out.append(s * res.fun)
        return (out[0], out[1])
    M, b = system(n, cells); M, b = M[1:], b[1:]; r = len(b); out = []          # tolerance: total L1 discrepancy <= tol
    for s in (1, -1):
        c = np.concatenate([s * a, np.zeros(2 * r)])
        Aeq = [np.concatenate([np.ones(N), np.zeros(2 * r)])]; beq = [1.0]
        for i in range(r):
            row = np.concatenate([M[i], np.zeros(2 * r)]); row[N + i] = -1; row[N + r + i] = 1; Aeq.append(row); beq.append(b[i])
        Aub = [np.concatenate([np.zeros(N), np.ones(2 * r)])]; bub = [tol]
        res = linprog(c, A_ub=np.array(Aub), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq, bounds=[(0, None)] * (N + 2 * r), method="highs")
        if res.status != 0: return None
        out.append(s * res.fun)
    return (out[0], out[1])


def forced_residual(n, honest, coalition):
    Mh, bh = system(n, honest); N = 2 ** n
    res = _l1_lp(n, coalition, extra_eq=(Mh, bh))
    return float(res.fun) if res.status == 0 else np.inf


def farkas(n, cells):
    """sparse y with A^T y >= 0 and b.y = -1 (exists iff A p = b, p >= 0 is infeasible); minimise ||y||_1"""
    A, b = system(n, cells); m = len(b)
    c = np.ones(2 * m)                                                  # y = y+ - y-
    Aub = np.hstack([-A.T, A.T]); bub = np.zeros(A.shape[1])
    Aeq = np.concatenate([b, -b])[None, :]; beq = [-1.0]
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (2 * m), method="highs")
    if res.status != 0: return None, []
    y = res.x[:m] - res.x[m:]; touched = []; pos = 1
    for name, ctx, table in cells:
        k = 2 ** len(ctx)
        if np.any(np.abs(y[pos:pos + k]) > 1e-9): touched.append(name)
        pos += k
    return y, touched


def conflicts(n, cells, tol=0.0):
    names = [c[0] for c in cells]; out = []
    for r in range(1, len(cells) + 1):
        for S in itertools.combinations(range(len(cells)), r):
            if any(set(C) <= set(S) for C in out): continue
            if not feasible(n, [cells[i] for i in S], tol): out.append(S)
    return [tuple(names[i] for i in C) for C in out]


def explanations(n, cells, k, tol=0.0):
    names = [c[0] for c in cells]; out = []
    for r in range(0, k + 1):
        for H in itertools.combinations(range(len(cells)), r):
            rest = [c for i, c in enumerate(cells) if i not in H]
            if feasible(n, rest, tol): out.append(tuple(names[i] for i in H))
    return out


def robust_interval(n, cells, a, k, tol=0.0):
    """list of (explanation, interval) and the union's hull; the union itself may have gaps"""
    parts = []
    for H in explanations(n, cells, k, tol):
        rest = [c for c in cells if c[0] not in H]; I = interval(n, rest, a, tol)
        if I is not None: parts.append((H, I))
    if not parts: return parts, None
    return parts, (min(I[0] for _, I in parts), max(I[1] for _, I in parts))


def gyo_core(cells):
    """GYO reduction of the context hypergraph: repeatedly delete a variable that lies in only one remaining context, and a context
    contained in another remaining context (ties broken by order). The family is acyclic (has a join tree) iff nothing remains; the
    cells that remain form the cyclic core."""
    edges = {c[0]: set(c[1]) for c in cells}; changed = True
    while changed and edges:
        changed = False
        for u in list(edges):
            for v in list(edges[u]):
                if sum(v in edges[w] for w in edges) == 1: edges[u].discard(v); changed = True
        for u in list(edges):
            if u in edges and (not edges[u] or any(w != u and edges[u] <= edges[w] for w in edges)): del edges[u]; changed = True
    return set(edges)


def exposure(cells):
    """overlap exposure: the cell shares a variable with another cell (a neighbour can contradict it); cycle exposure: the cell is in the
    GYO core (a closed loop of contexts can reveal a contradiction that no single overlap shows)"""
    names = [c[0] for c in cells]; ctx = {c[0]: set(c[1]) for c in cells}
    overlap = {u: any(ctx[u] & ctx[v] for v in names if v != u) for u in names}
    core = gyo_core(cells)
    return overlap, {u: u in core for u in names}


def _free_reduce(w):
    out = []
    for x in w:
        if out and out[-1] == -x: out.pop()
        else: out.append(x)
    return out


def artin_trivial(word, m):
    """Artin action: sigma_i sends x_i -> x_i x_{i+1} x_i^-1, x_{i+1} -> x_i, fixes the rest (letters are +-(1..m)). Faithful, so the braid
    is trivial iff every generator is fixed."""
    def inv(w): return [-x for x in reversed(w)]
    img = {j: [j] for j in range(1, m + 1)}
    for g in word:                                                      # apply in order: the image of x_j under the composite
        i = abs(g); new = dict(img)
        if g > 0:
            new[i] = _free_reduce(img[i] + img[i + 1] + inv(img[i])); new[i + 1] = img[i]
        else:
            new[i] = img[i + 1]; new[i + 1] = _free_reduce(inv(img[i + 1]) + img[i] + img[i + 1])
        img = new
    return all(img[j] == [j] for j in img)


def bubble_word(before, after, sign_of):
    """before, after: lists of strand labels ordered bottom to top. Bubble-sort 'before' into 'after'; each adjacent swap of positions
    (k, k+1) gives the letter +-(k+1), sign from sign_of(lower_label, upper_label) (the declared over/under rule)."""
    cur = list(before); word = []
    target = {lab: i for i, lab in enumerate(after)}
    changed = True
    while changed:
        changed = False
        for k in range(len(cur) - 1):
            if target[cur[k]] > target[cur[k + 1]]:
                word.append((k + 1) * sign_of(cur[k], cur[k + 1])); cur[k], cur[k + 1] = cur[k + 1], cur[k]; changed = True
    return word


def trace_normal_form(word, indep):
    """Cartier-Foata normal form: a list of layers (sorted tuples). Letters a, b commute iff (a, b) or (b, a) is in indep; a letter never
    commutes with itself. Each layer collects the occurrences that commute with everything before them."""
    def comm(a, b): return a != b and ((a, b) in indep or (b, a) in indep)
    word = list(word); layers = []
    while word:
        take = [i for i, x in enumerate(word) if all(comm(word[j], x) for j in range(i))]
        layers.append(tuple(sorted(word[i] for i in take))); word = [x for i, x in enumerate(word) if i not in take]
    return layers
