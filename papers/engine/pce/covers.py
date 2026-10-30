"""Two latent types on arbitrary covers (companion paper "Two latent types on arbitrary covers"). New module; nothing frozen.

Model: binary variables 0..m-1, two types with weight w = 1/(1 + t^2), success probabilities a1 = mu + v t, a2 = mu - v / t
(the lab's parametrisation; label swap is (v, t) -> (-v, 1/t)). Pair covariances are c_ij = v_i v_j.

Phase I (the pair layer on a graph G = (V, E) of observed pairs, singles observed for every vertex):
  components(m, edges)             connected components (isolated vertices included)
  bipartition(comp, edges)         {vertex: +1 / -1} for a bipartite component, None otherwise
  incidence(m, edges)              the unsigned incidence matrix B (rows edges, columns vertices)
  b_count(m, edges)                number of bipartite components (isolated vertices count)
  factorise(m, edges, c)           the real factorisations v of c on G: zero strata, the nonzero graph G*, per component a base
                                   solution, its type (rigid / scale family) and the existence conditions (sign class, modulus class)
Phase II (the hiding set):
  sheet_constraints(mu, F, sigma)  the hiding set of one sign sheet in log coordinates (y = log t, x_k = log lambda_k): rows
                                   A [y, x] <= b, two per vertex with v_i != 0, every row of slope +-1 in (y, x_k)
  sheet_dimension(A, b)            dimension of the polytope (Chebyshev radius and affine hull)
  explain(mu, F, sigma, y, x, free) a two-type law (w, a1, a2) from a point of a sheet; free values for vertices with v free
  law_m(w, a1, a2)                 the 2^m-cell law
  marg(P, S)                       a marginal table
Phase III (windows):
  s_poly(k, r)                     the lab's s_k(r): s_2 = 1, s_3 = r, s_{k+1} = r s_k + s_{k-1}
  window_jacobian(F, sigma, y, x, windows)   Jacobian of the window moments m_T = v_T s_|T|(r) in (y, x_k), |T| >= 3
"""
from __future__ import annotations
import itertools
from collections import deque
import numpy as np
from scipy.optimize import linprog


# ------------------------------------------------------------------------------------------------ graphs
def components(m, edges):
    adj = {i: set() for i in range(m)}
    for i, j in edges: adj[i].add(j); adj[j].add(i)
    seen, out = set(), []
    for s in range(m):
        if s in seen: continue
        comp, q = [], deque([s]); seen.add(s)
        while q:
            u = q.popleft(); comp.append(u)
            for w in adj[u]:
                if w not in seen: seen.add(w); q.append(w)
        out.append(sorted(comp))
    return out


def bipartition(comp, edges):
    cs = set(comp); adj = {i: [] for i in comp}
    for i, j in edges:
        if i in cs and j in cs: adj[i].append(j); adj[j].append(i)
    side = {comp[0]: 1}; q = deque([comp[0]])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if w not in side: side[w] = -side[u]; q.append(w)
            elif side[w] == side[u]: return None
    return side


def incidence(m, edges):
    B = np.zeros((len(edges), m))
    for k, (i, j) in enumerate(edges): B[k, i] = B[k, j] = 1.0
    return B


def b_count(m, edges):
    return sum(1 for comp in components(m, edges) if bipartition(comp, edges) is not None)


def cycle_rank(m, edges): return len(edges) - m + len(components(m, edges))


# ------------------------------------------------------------------------------------------------ Phase I: factorisation
def factorise(m, edges, c, tol=1e-12):
    """c: dict edge -> covariance. Returns a dict:
      ok            a real factorisation exists
      reason        why not (zero edge inside N, sign class, modulus class)
      N, zero, free vertices with v != 0 forced, v = 0 forced, v unconstrained by nonzero edges
      comps         list of dicts (vertices, edges, bip side or None, base v0 with the right signs up to a component sign)
      sign_ok, mod_res   per component: the Z/2 sign cocycle is a coboundary; the residual of log|c| outside im B
    """
    E = [tuple(e) for e in edges]; nz = [e for e in E if abs(c[e]) > tol]; zr = [e for e in E if abs(c[e]) <= tol]
    N = sorted({i for e in nz for i in e}); Ns = set(N)
    out = dict(ok=True, reason="", N=N, nz=nz, zr=zr, comps=[])
    bad = [e for e in zr if e[0] in Ns and e[1] in Ns]
    if bad: out.update(ok=False, reason=f"zero moment on {bad[0]} between two vertices with nonzero moments"); return out
    zero = sorted({i for e in zr for i in e if (e[0] in Ns) != (e[1] in Ns) and i not in Ns})
    out["zero"] = zero; out["free"] = [i for i in range(m) if i not in Ns and i not in zero]
    for comp in components(m, nz):
        if len(comp) == 1 and comp[0] not in Ns: continue
        ce = [e for e in nz if e[0] in comp]; side = bipartition(comp, ce); idx = {v: k for k, v in enumerate(comp)}
        # signs: BFS sigma_i sigma_j = sgn c_ij
        sg = {comp[0]: 1.0}; q = deque([comp[0]]); adj = {i: [] for i in comp}; sign_ok = True
        for (i, j) in ce: adj[i].append(j); adj[j].append(i)
        while q:
            u = q.popleft()
            for w_ in adj[u]:
                e = (u, w_) if (u, w_) in c else (w_, u)
                s = np.sign(c[e]) * sg[u]
                if w_ not in sg: sg[w_] = s; q.append(w_)
                elif sg[w_] != s: sign_ok = False
        Bc = np.zeros((len(ce), len(comp))); y = np.zeros(len(ce))
        for k, (i, j) in enumerate(ce): Bc[k, idx[i]] = Bc[k, idx[j]] = 1.0; y[k] = np.log(abs(c[(i, j)]))
        xls = np.linalg.lstsq(Bc, y, rcond=None)[0]; res = float(np.abs(Bc @ xls - y).max()) if len(ce) else 0.0
        v0 = {i: sg[i] * float(np.exp(xls[idx[i]])) for i in comp}
        out["comps"].append(dict(vertices=comp, edges=ce, side=side, v0=v0, sign_ok=sign_ok, mod_res=res,
                                 rankB=int(np.linalg.matrix_rank(Bc)) if len(ce) else 0))
        if not sign_ok: out.update(ok=False, reason=f"sign class non-trivial on component {comp}")
        elif res > 1e-9: out.update(ok=False, reason=f"modulus class non-zero on component {comp} (residual {res:.2e})")
    out["n_scale"] = sum(1 for C in out["comps"] if C["side"] is not None)
    return out


def family_v(F, sigma, x):
    """the factorisation with component signs sigma (list, one per component) and log-scales x (one per bipartite component)"""
    v = {}; k = 0
    for C, s in zip(F["comps"], sigma):
        lam = 0.0
        if C["side"] is not None: lam = x[k]; k += 1
        for i in C["vertices"]:
            eps = C["side"][i] if C["side"] is not None else 0
            v[i] = s * C["v0"][i] * np.exp(eps * lam)
    return v


# ------------------------------------------------------------------------------------------------ Phase II: the hiding set
def sheet_constraints(mu, F, sigma):
    """rows of A [y, x_1..x_b] <= b for the sheet sigma. For vertex i with v_i = s a e^{z}, z = eps_i x_k:
       s > 0:  z + y <= log((1 - mu_i)/a),   z - y <= log(mu_i / a)
       s < 0:  z + y <= log(mu_i / a),       z - y <= log((1 - mu_i)/a)"""
    nb = F["n_scale"]; rows, rhs, who = [], [], []; k = 0
    for C, s in zip(F["comps"], sigma):
        kk = None
        if C["side"] is not None: kk = k; k += 1
        for i in C["vertices"]:
            a = abs(C["v0"][i]); sgn = np.sign(s * C["v0"][i]); eps = C["side"][i] if kk is not None else 0
            z = np.zeros(1 + nb)
            if kk is not None: z[1 + kk] = eps
            r1 = z.copy(); r1[0] += 1.0; r2 = z.copy(); r2[0] -= 1.0
            up, lo = np.log((1 - mu[i]) / a), np.log(mu[i] / a)
            rows += [r1, r2]; rhs += ([up, lo] if sgn > 0 else [lo, up]); who += [(i, "+"), (i, "-")]
    return np.array(rows), np.array(rhs), who


def chebyshev(A, b):
    n = A.shape[1]; nr = np.linalg.norm(A, axis=1)
    res = linprog(np.r_[np.zeros(n), -1.0], A_ub=np.c_[A, nr], b_ub=b, bounds=[(None, None)] * n + [(0, 10)], method="highs")
    if res.status != 0: return None, -1.0
    return res.x[:n], float(res.x[n])


def sheet_dimension(A, b, tol=1e-9):
    """dimension of {A z <= b}: -1 if empty; else the dimension of its affine hull (implicit equalities found by LP)"""
    n = A.shape[1]; res = linprog(np.zeros(n), A_ub=A, b_ub=b, bounds=[(None, None)] * n, method="highs")
    if res.status != 0: return -1
    eq = []
    for r in range(A.shape[0]):
        lp = linprog(A[r], A_ub=A, b_ub=b, bounds=[(None, None)] * n, method="highs")
        if lp.status == 0 and A[r] @ lp.x > b[r] - tol: eq.append(A[r])
    return n - (int(np.linalg.matrix_rank(np.array(eq), tol=1e-9)) if eq else 0)


def J_free(mu_i, t):
    """the interval of v_i for a vertex not constrained by any nonzero pair moment"""
    return max(-mu_i / t, -(1 - mu_i) * t), min((1 - mu_i) / t, mu_i * t)


def explain(mu, F, sigma, y, x, free_vals=None, m=None):
    t = float(np.exp(y)); v = family_v(F, sigma, x); m = m or len(mu); vv = np.zeros(m)
    for i, val in v.items(): vv[i] = val
    for i in F.get("free", []):
        lo, hi = J_free(mu[i], t); vv[i] = (free_vals or {}).get(i, 0.5 * (lo + hi))
    w = 1 / (1 + t * t); return w, np.asarray(mu) + vv * t, np.asarray(mu) - vv / t, vv


def law_m(w, a1, a2):
    m = len(a1); P = np.zeros((2,) * m)
    for xx in itertools.product((0, 1), repeat=m):
        x_ = np.array(xx); P[xx] = w * np.prod(np.where(x_ == 1, a1, 1 - a1)) + (1 - w) * np.prod(np.where(x_ == 1, a2, 1 - a2))
    return P


def marg(P, S):
    return P.sum(axis=tuple(i for i in range(P.ndim) if i not in S))


def truth_data(w, a1, a2, edges):
    a1 = np.asarray(a1, float); a2 = np.asarray(a2, float); mu = w * a1 + (1 - w) * a2; v = np.sqrt(w * (1 - w)) * (a1 - a2)
    return mu, {tuple(e): float(v[e[0]] * v[e[1]]) for e in edges}, v


# ------------------------------------------------------------------------------------------------ Phase III: windows
def s_poly(k, r):
    s = [None, None, 1.0, r]
    for j in range(3, k): s.append(r * s[j] + s[j - 1])
    return s[k]


def window_moments(F, sigma, y, x, windows):
    t = np.exp(y); r = t - 1 / t; v = family_v(F, sigma, x)
    return np.array([np.prod([v[i] for i in T]) * s_poly(len(T), r) for T in windows])


def window_jacobian(F, sigma, y, x, windows, h=1e-6):
    z0 = np.r_[y, x]; f = lambda z: window_moments(F, sigma, z[0], z[1:], windows)
    return np.array([(f(z0 + h * e) - f(z0 - h * e)) / (2 * h) for e in np.eye(len(z0))]).T
