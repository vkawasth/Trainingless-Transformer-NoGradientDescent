"""Step 2 of "Probability, Context and Differentiation: The Layers": the cone tangent structure T of the tangent companion, carried
through the Step 1 operations (provenance comonad, distributive law, marginalisation, conditioning, pooling). New module.

Laws on X are vectors of length n (cells 0..n-1); a law on L x X is an array of shape (|L|, n). The feasible cone of a law p is
C_p = {u : sum u = 0, u_x >= 0 where p_x = 0}; points of T^k are Weil-algebra arrays (pce.tangent.HD).

  feasible(p, u)                     u in C_p
  rand_feasible(p, rng)              a random integer direction in C_p
  lam_vec(l, p, nL)                  the distributive law (l, p) -> delta_l (x) p as an (|L|, n) array (works on HD coefficients too)
  forget(q)                          D eps: forgetting provenance, the column sums of q
  split_face(q, l)                   for q = delta_l (x) p and u in C_q: the part along the face (row l) and the transverse part
  marg_matrix(shape, keep)           the 0/1 matrix of marginalisation onto the coordinates `keep` of a law on a product of finite sets
  marg_lift(p, v, M)                 a direction u in C_p with M u = v (v in C_{Mp}): u(x) = v(x_S) p(x | x_S), uniform where p(x_S) = 0
  cond_vec(p, A)                     C_A p (A a 0/1 mask)
  dcond(p, A, u)                     the derivative of C_A at p along u: (N_A u - u(A) C_A p) / p(A)
  cond_lift(p, A, v)                 a direction u in C_p with DC_A u = v: u = p(A) v (zero off A)
  pool_hd(w, ps)                     sum_i w_i p_i on HD arguments (the monad multiplication restricted to n sources)
  so_ok_blocks(f, x, blocks)         the second-order corner criterion of the tangent companion on a product of simplices
  direct_T2_blocks(f, x, blocks, ..) the direct test: random points of T^2 of the product, mapped by T^2 f, tested by the lexicographic lemma
"""
from __future__ import annotations
import numpy as np
from . import tangent as TG
from .tangent import HD


def feasible(p, u, tol=1e-12):
    p = np.ravel(p); u = np.ravel(u)
    return abs(u.sum()) < 1e-9 and all(u[i] >= -tol for i in range(len(p)) if p[i] <= tol)


def rand_feasible(p, rng, lo=-3, hi=4):
    p = np.ravel(np.asarray(p, float)); n = len(p)
    for _ in range(1000):
        u = rng.integers(lo, hi, size=n).astype(float)
        u[p <= 0] = np.abs(u[p <= 0])
        s = u.sum(); sup = np.flatnonzero(p > 0)
        u[sup[0]] -= s
        if feasible(p, u): return u
    raise RuntimeError("no feasible direction found")


def lam_vec(l, p, nL):
    p = np.asarray(p, float); q = np.zeros((nL,) + p.shape); q[l] = p; return q


def forget(q): return np.asarray(q, float).sum(axis=0)


def split_face(q, l, u):
    """row l and the rest; the rest is >= 0 for u in C_q (q vanishes off row l) and is the mass injected under other labels"""
    u = np.asarray(u, float); t = np.zeros_like(u); t[l] = u[l]; return t, u - t


def marg_matrix(shape, keep):
    import itertools
    cells = list(itertools.product(*[range(s) for s in shape])); outs = list(itertools.product(*[range(shape[i]) for i in keep]))
    idx = {o: j for j, o in enumerate(outs)}; M = np.zeros((len(outs), len(cells)))
    for c, x in enumerate(cells): M[idx[tuple(x[i] for i in keep)], c] = 1.0
    return M


def marg_lift(p, v, M):
    p = np.asarray(p, float); v = np.asarray(v, float); u = np.zeros(M.shape[1]); m = M @ p
    for j in range(M.shape[0]):
        fib = np.flatnonzero(M[j]);
        if m[j] > 0: u[fib] = v[j] * p[fib] / m[j]
        else: u[fib] = v[j] / len(fib)
    return u


def cond_vec(p, A):
    p = np.asarray(p, float); A = np.asarray(A, float); a = (A * p).sum(); return A * p / a


def dcond(p, A, u):
    p = np.asarray(p, float); A = np.asarray(A, float); u = np.asarray(u, float); a = (A * p).sum()
    return (A * u - (A * u).sum() * cond_vec(p, A)) / a


def cond_lift(p, A, v):
    p = np.asarray(p, float); A = np.asarray(A, float); return (A * p).sum() * np.asarray(v, float) * A


def pool_hd(w, ps):
    out = w[0] * ps[0]
    for i in range(1, len(ps)): out = out + w[i] * ps[i]
    return out


# ------------------------------------------------------------------------------------------------ products of simplices
def _blocks_cone(x, blocks, tol=1e-12):
    """lineality basis and extreme rays of the feasible cone of x in the product of simplices (blocks = list of index arrays)"""
    n = len(x); I = np.eye(n); L = []; R = []
    for b in blocks:
        sup = [i for i in b if x[i] > tol]; Z = [i for i in b if x[i] <= tol]; s0 = sup[0]
        L += [I[s] - I[s0] for s in sup[1:]]; R += [I[z] - I[s0] for z in Z]
    return L, R


def _jets(fy, x):
    n = len(x); I = np.eye(n); g = np.zeros(n); H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            Y = fy(HD.var(x, [I[i], I[j]])); H[i, j] = H[j, i] = Y.c[3]; g[i] = Y.c[1]
    return g, H


def so_ok_blocks(f, x, blocks, tol=1e-9):
    """f maps an HD point (vector over the concatenated coordinates) to a list of HD output cells. For every output cell y with
    f_y(x) = 0: first order (grad >= 0 on the rays, = 0 on the lineality), and the second-order criterion: the Hessian vanishes on
    L x K_y and is >= 0 on R_K x R_K, K_y = C_x cap ker grad"""
    x = np.asarray(x, float); vals = [float(y.c[0]) if isinstance(y, HD) else float(y) for y in f(HD.var(x, []))]
    L, R = _blocks_cone(x, blocks); ok = True; info = []
    for y, v in enumerate(vals):
        if v > tol: continue
        g, H = _jets(lambda X, y=y: f(X)[y], x)
        first = all(abs(g @ l) < tol for l in L) and all(g @ r > -tol for r in R)
        RK = [r for r in R if abs(g @ r) < tol]; gens = L + RK
        lin = max([abs(l @ H @ k) for l in L for k in gens] + [0.0]); ray = min([r @ H @ s for r in RK for s in RK] + [0.0])
        good = first and lin < tol and ray > -tol; ok &= good; info.append((y, first, lin, ray))
    return ok, info


def in_T_blocks(c, blocks, k):
    return all(TG.in_T(c[:, b], k) for b in blocks)


def rand_T2_blocks(x, blocks, rng, tries=2000):
    """a random point of T^2 of the product of simplices over x: c1, c2 feasible; c3 zero-sum per block, made non-negative on the
    zero cells of x half of the time (otherwise left free, so that the lexicographic lemma does the filtering)"""
    x = np.asarray(x, float); n = len(x)
    for _ in range(tries):
        c = np.zeros((4, n)); c[0] = x
        for b in blocks:
            xb = x[b]; c[1, b] = rand_feasible(xb, rng, -2, 3); c[2, b] = rand_feasible(xb, rng, -2, 3)
            z = rng.integers(-2, 3, size=len(b)).astype(float)
            if rng.random() < 0.5: z[xb <= 0] = np.abs(z[xb <= 0])
            z[np.flatnonzero(xb > 0)[0]] -= z.sum(); c[3, b] = z
        if in_T_blocks(c, blocks, 2): return c
    raise RuntimeError("no T^2 point found")


def direct_T2_blocks(f, x, blocks, rng, samples=300):
    tested = bad = 0
    for _ in range(samples):
        c = rand_T2_blocks(x, blocks, rng); ys = f(HD(c, 2)); img = np.stack([y.c for y in ys], axis=1); tested += 1
        bad += not TG.in_T(img, 2)
    return bad == 0, tested, bad


# ------------------------------------------------------------------------------------------------ T^k of the half-line
def weil_mul(a, b, k):
    """product in W_k of two coefficient vectors of length 2^k"""
    out = np.zeros(2 ** k)
    for A in range(2 ** k):
        if a[A] == 0: continue
        for B in range(2 ** k):
            if A & B == 0: out[A | B] += a[A] * b[B]
    return out


def tensor_decompose(c, k):
    """write c in T^k H as a non-negative combination of elementary tensors u_1(e_1)...u_k(e_k), u_i = x_i + y_i e_i in T H. Returns a
    list of (coefficient > 0, [(x_1, y_1), ..., (x_k, y_k)]). Recursion on the outer generator: c = alpha + beta e_k =
    alpha (1 - e_k / s) + (beta + alpha / s) e_k, with s > 0 chosen so that alpha + s beta lies in T^{k-1} H"""
    c = np.asarray(c, float)
    if k == 0: return [(float(c[0]), [])] if c[0] > 0 else []
    h = 2 ** (k - 1); al, be = c[:h], c[h:]
    if not np.any(al): return [(w, f + [(0.0, 1.0)]) for w, f in tensor_decompose(be, k - 1)]
    s = 1.0
    while not TG.in_TH(al + s * be, k - 1): s /= 2
    return [(w, f + [(1.0, -1.0 / s)]) for w, f in tensor_decompose(al, k - 1)] + \
           [(w, f + [(0.0, 1.0)]) for w, f in tensor_decompose(be + al / s, k - 1)]


def tensor_expand(terms, k):
    out = np.zeros(2 ** k)
    for w, fs in terms:
        t = np.zeros(2 ** k); t[0] = w
        for i, (x, y) in enumerate(fs):
            u = np.zeros(2 ** k); u[0] = x; u[1 << i] = y; t = weil_mul(t, u, k)
        out += t
    return out
