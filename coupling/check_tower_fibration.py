"""check_tower_fibration.py -- a worked example of the base-relative tower as a fibration over Kl(D):
transport along Markov kernels B = {0,1} -> B' = {u,v,w} (and on to B'' = {P,Q}).

Admissibility data on B (not saturated): S1 = {d0, d1, h} with h = (1/2,1/2);
S2 = {1/2 d_{d0} + 1/2 d_{d1}, d_h}  (a fair coin is admissible only as certainty in h, or as a mixture of the
two decisive posteriors). Then E2 = conv S2, E1 = {h}, L1 = whole simplex, L2 = D(S1).

  W1 a claim x1 = (7/10, 3/10) lies in L1 but not in E1: it passes the local test and has no coherent lift;
     certificate f = (1, -1)
  W2 transport along a generic kernel k: the failure persists; a certificate found on B' pulls back to B
     (exact identity <k |> f', rho> = <f', k# rho>)
  W3 transport along a constant kernel: every obstruction disappears (failure levels can only rise)
  W4 secondary obstruction o3 of Pi = d_{d0}: 1 on B, 1 after the generic kernel, 0 after the constant one
  W5 coherent contexts stay coherent; A_{h.k} = A_h A_k on a context (exact)
  W6 grades transport: level-1 TV and level-2, level-3 Kantorovich distances between A_f(c) and A_g(c) are
     bounded by d(f, g) (LP)
  W7 conditioning on an outcome observed after the channel = conditioning on the pulled-back predicate (exact)
"""
import itertools
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))

# distributions as dicts {outcome: prob}; second-order states as dicts {first-order (frozen as tuple): prob}
def D(d): return tuple(sorted((k, v) for k, v in d.items() if v != 0))
def dd(t): return dict(t)
def kext(k, rho):
    out = {}
    for b, p in dd(rho).items():
        for bp, q in k[b].items(): out[bp] = out.get(bp, 0) + p * q
    return D(out)
def push(fn, P):
    out = {}
    for x, p in dd(P).items():
        y = fn(x); out[y] = out.get(y, 0) + p
    return D(out)
def mu(P):
    out = {}
    for rho, p in dd(P).items():
        for b, q in dd(rho).items(): out[b] = out.get(b, 0) + p * q
    return D(out)
def compose(k, h): return {b: dd(kext(h, D(k[b]))) for b in k}
def tv(p, q):
    a, b = dd(p), dd(q); keys = set(a) | set(b)
    return sum(abs(a.get(x, 0) - b.get(x, 0)) for x in keys) / 2
def inner(f, rho): return sum(f[b] * p for b, p in dd(rho).items())

d0, d1 = D({0: F(1)}), D({1: F(1)}); h = D({0: F(1, 2), 1: F(1, 2)})
S1 = [d0, d1, h]
S2 = [D({d0: F(1, 2), d1: F(1, 2)}), D({h: F(1)})]

# ------------------------------------------------------------------ W1
x1 = D({0: F(7, 10), 1: F(3, 10)})
E1 = [mu(P) for P in S2]
fB = {0: F(1), 1: F(-1)}
check("W1 E1 = {h}; x1 = (7/10, 3/10) is in L1 but not E1; f = (1,-1) separates it",
      set(E1) == {h} and inner(fB, x1) > max(inner(fB, e) for e in E1), f"<f,x1> = {inner(fB, x1)} > <f,h> = 0")

# ------------------------------------------------------------------ kernels
kgen = {0: {"u": F(1, 2), "v": F(1, 4), "w": F(1, 4)}, 1: {"u": F(1, 5), "v": F(1, 5), "w": F(3, 5)}}
kconst = {0: {"u": F(1, 3), "v": F(1, 3), "w": F(1, 3)}, 1: {"u": F(1, 3), "v": F(1, 3), "w": F(1, 3)}}
hP = {"u": {"P": F(1)}, "v": {"P": F(1, 2), "Q": F(1, 2)}, "w": {"Q": F(1)}}

# ------------------------------------------------------------------ W2 generic kernel
x1p = kext(kgen, x1); E1p = [kext(kgen, e) for e in E1]
diff = {b: dd(x1p).get(b, 0) - dd(E1p[0]).get(b, 0) for b in "uvw"}
fp = diff                                          # separating functional on B'
pull = {b: sum(kgen[b][bp] * fp[bp] for bp in "uvw") for b in (0, 1)}
ok_id = all(inner(pull, r) == inner(fp, kext(kgen, r)) for r in S1 + [x1])
margin = inner(fp, x1p) - inner(fp, E1p[0])
check("W2 generic kernel: the failure persists; the certificate f' on B' pulls back to k |> f' on B",
      margin > 0 and ok_id and inner(pull, x1) > inner(pull, h),
      f"k#x1 = {[str(dd(x1p)[b]) for b in 'uvw']}, E1' = {[str(dd(E1p[0])[b]) for b in 'uvw']}, pulled-back f = ({pull[0]}, {pull[1]})")

# ------------------------------------------------------------------ W3 constant kernel
x1c = kext(kconst, x1); E1c = {kext(kconst, e) for e in E1}
check("W3 constant kernel: k#x1 equals the transported effective point, so the obstruction disappears",
      E1c == {x1c})

# ------------------------------------------------------------------ W4 secondary obstruction o3
def o3(Pi, S1_, S2_):
    idx = {r: i for i, r in enumerate(S1_)}; n, m = len(S1_), len(S2_)
    v = np.zeros(n)
    for r, w in dd(Pi).items(): v[idx[r]] += float(w)
    M = np.zeros((n, m))
    for j, P in enumerate(S2_):
        for r, w in dd(P).items(): M[idx[r], j] += float(w)
    c = np.r_[np.zeros(m), np.ones(n)]
    A = np.r_[np.c_[-M, -np.eye(n)], np.c_[M, -np.eye(n)]]
    return linprog(c, A_ub=A, b_ub=np.r_[-v, v], A_eq=[np.r_[np.ones(m), np.zeros(n)]], b_eq=[1],
                   bounds=[(0, None)] * (m + n)).fun
Pi = D({d0: F(1)})
vals = []
for k in (None, kgen, kconst):
    if k is None: vals.append(o3(Pi, S1, S2)); continue
    e = lambda r: kext(k, r)
    S1k = list(dict.fromkeys(e(r) for r in S1)); S2k = [push(e, P) for P in S2]
    vals.append(o3(push(e, Pi), S1k, S2k))
check("W4 o3(d_{d0}) is 1 on B, 1 after the generic kernel (injective on S1), 0 after the constant kernel",
      abs(vals[0] - 1) < 1e-9 and abs(vals[1] - 1) < 1e-9 and abs(vals[2]) < 1e-9, str([round(v, 6) for v in vals]))

# ------------------------------------------------------------------ W5 coherence and functoriality
THETA = ("t1", "t2")
Sigma = D({("t1", d0): F(1, 4), ("t2", d1): F(1, 4), ("t1", h): F(1, 2)})
PiS = push(lambda hy: hy[1], Sigma); Om = D({S2[0]: F(1, 2), D({h: F(1)}): F(1, 2)})
# make the context coherent: Pi = D(q) Sigma, and Omega with mu(Omega) = Pi
Om = D({D({d0: F(1, 2), d1: F(1, 2)}): F(1, 2), D({h: F(1)}): F(1, 2)})
ctx = (mu(PiS), PiS, Om, Sigma)
def A(k, c):
    e = lambda r: kext(k, r)
    return (e(c[0]), push(e, c[1]), push(lambda P: push(e, P), c[2]), push(lambda hy: (hy[0], e(hy[1])), c[3]))
def coherent(c):
    return mu(c[1]) == c[0] and mu(c[2]) == c[1] and push(lambda hy: hy[1], c[3]) == c[1]
check("W5 the context is coherent, A_k keeps it coherent, and A_{h.k} = A_h A_k",
      coherent(ctx) and coherent(A(kgen, ctx)) and A(compose(kgen, hP), ctx) == A(hP, A(kgen, ctx)))

# ------------------------------------------------------------------ W6 grades transport
g2 = {0: {"u": F(1, 3), "v": F(1, 3), "w": F(1, 3)}, 1: {"u": F(1, 10), "v": F(3, 10), "w": F(3, 5)}}
dfg = max(tv(D(kgen[b]), D(g2[b])) for b in (0, 1))
def kant(P, Q, metric):
    xs, ys = [x for x, _ in P], [y for y, _ in Q]
    C = np.array([[float(metric(a, b)) for b in ys] for a in xs]); n, m = len(xs), len(ys)
    A_eq, b_eq = [], []
    for i, (_, p) in enumerate(P):
        r = np.zeros(n * m); r[i * m:(i + 1) * m] = 1; A_eq.append(r); b_eq.append(float(p))
    for j, (_, q) in enumerate(Q):
        r = np.zeros(n * m); r[j::m] = 1; A_eq.append(r); b_eq.append(float(q))
    return linprog(C.ravel(), A_eq=np.array(A_eq), b_eq=b_eq, bounds=[(0, None)] * (n * m)).fun
cf, cg = A(kgen, ctx), A(g2, ctx)
lv1 = float(tv(cf[0], cg[0])); lv2 = kant(cf[1], cg[1], tv)
lv3 = kant(cf[2], cg[2], lambda P, Q: kant(P, Q, tv))
check("W6 grade transport: level-1 TV, level-2 and level-3 Kantorovich distances are <= d(f,g)",
      lv1 <= float(dfg) + 1e-12 and lv2 <= float(dfg) + 1e-9 and lv3 <= float(dfg) + 1e-9,
      f"d(f,g) = {dfg} = {float(dfg):.4f}; levels: {lv1:.4f}, {lv2:.4f}, {lv3:.4f}")

# ------------------------------------------------------------------ W7 conditioning through the channel
def cond(S, L):
    w = {hy: L(hy) * p for hy, p in dd(S).items()}; z = sum(w.values())
    return D({hy: x / z for hy, x in w.items() if x})
ep = "w"
after = cond(push(lambda hy: (hy[0], kext(kgen, hy[1])), Sigma), lambda hy: dd(hy[1]).get(ep, 0))
pulled = lambda hy: sum(p * kgen[b][ep] for b, p in dd(hy[1]).items())
before = push(lambda hy: (hy[0], kext(kgen, hy[1])), cond(Sigma, pulled))
weights = {str(hy[1]): str(p) for hy, p in dd(cond(Sigma, pulled)).items()}
check("W7 observing w on B' after k = conditioning on k |> 1_w on B, then transporting", after == before,
      f"posterior weights on hypotheses: {list(dict(cond(Sigma, pulled)).values())}")
print(f"\n{sum(res)}/{len(res)} checks passed")
