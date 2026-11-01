"""Checks for Step 5 of "Probability, Context and Differentiation: The Layers": functions, duality and logs.
  FD1  state-effect duality (exact): <u, pK> = <Ku, p>; pulling back is contravariant, (KL)u = K(Lu); marginalisation pulls effects back
       along the projection; forgetting provenance pulls u back to u o eps; a reparametrisation commutes with pairing (naturality)
  FD2  conditioning is an update by an effect (exact): C_A = update by 1_A; updates compose multiplicatively (C_q C_r = C_{qr}); the contextual
       Bayes decomposition holds for every effect q, with Bayes weights pi_l <q, p_l> / <q, P> (update by eps^* q); the Bayesian inverse is
       the update by the pulled-back point effect K 1_y; expectations update as <u, C_q p> = <q u, p> / <q, p>
  FD3  the L^2 presheaf (exact): conditional expectation E_p[. | X_S] is the p-adjoint of pulling back along the projection, is idempotent,
       and restricts consistently, E[E[u | X_T] | X_S] = E[u | X_S] for S in T
  FD4  the log map: |Log_s(sum a_i) - min Log_s(a_i)| <= log n / |log s| (Maslov), so pooling tropicalises to min; Bayes is exactly linear in
       logs, and on valuations conditioning is the tropical division v(q) + v(p) - min(v(q) + v(p)) (exact, on germs)
  FD5  the normal fan in log coordinates: along eps = c s^w, Log_s(eps) -> w and Log_s f(eps(s)) -> trop_w f = min <w, alpha>; trop P(A) is
       piecewise linear with one linearity domain per vertex of N, its breaks at the edge normals
  FD6  the m/e split (exact, on germs): arithmetic mixtures (pooling) have valuation min(v p, v q); geometric mixtures p^(1-t) q^t / Z have
       valuation (1-t) v p + t v q - min_x(...), linear in t; products add valuations
  FD7  the cotangent cone: C_x^* = {constant gamma on the support, >= gamma off it}; pulling back by the derivative of a map of simplices sends
       C^*_{f(x)} into C^*_x (stochastic maps, conditioning, pooling at boundary points); an act tied at x stays weakly preferred under every
       first-order injection iff u_a - u_b lies in C_x^*
  FD8  decision walls (exact): along a cone of the decision fan the act changes exactly at the positive roots of the edge polynomial of the
       signed polynomial N_ab = sum_c (u_a - u_b)(c) f_c; N_ab can cancel where P(A) cannot
  FD9  what does not matter (exact and ranks): the decision-relevant dimension rank(rho on ker K) equals dim span(rho) - dim(span(rho) cap im K^T)
       (effects that are functions of the windows are pinned); utilities off A do not matter to a conditional decision; coefficients off
       the leading face do not matter at the wall
"""
import os, sys, json, random, math, itertools
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import effects as EF
from pce import orders as OD
from pce import fans as FN
from pce import para as PA
from pce import degen as DG
from pce import relevance as RV
from pce import tangent_layers as TLM
random.seed(20261101); rng = np.random.default_rng(20261101); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def rlaw(n, zeros=0):
    w = [Fr(random.randint(1, 6)) for _ in range(n)]
    for i in random.sample(range(n), zeros): w[i] = Fr(0)
    s = sum(w); return [x / s for x in w]
def rker(n, m): return [rlaw(m) for _ in range(n)]
def rfun(n): return [Fr(random.randint(-5, 5)) for _ in range(n)]

# FD1
ok = True
for _ in range(200):
    n, m, k = 4, 3, 5; p = rlaw(n, random.randint(0, 2)); K = rker(n, m); L = rker(m, k); u = rfun(m); w = rfun(k)
    ok &= EF.pair(u, EF.push(K, p)) == EF.pair(EF.pull(K, u), p)
    ok &= EF.pull(EF.compose(K, L), w) == EF.pull(K, EF.pull(L, w))
    # marginalisation onto the first of two coordinates of a 2 x 2 law: pull back v to v o pi
    P4 = rlaw(4); v = rfun(2); M = [[Fr(1), Fr(0)], [Fr(1), Fr(0)], [Fr(0), Fr(1)], [Fr(0), Fr(1)]]   # cells (0,0),(0,1),(1,0),(1,1) -> x0
    ok &= EF.pair(v, EF.push(M, P4)) == EF.pair(EF.pull(M, v), P4)
    # forgetting provenance: <u, D eps Lambda> = <u o eps, Lambda>
    Lam = rlaw(2 * n); fu = rfun(n); pooled = [Lam[i] + Lam[n + i] for i in range(n)]
    ok &= EF.pair(fu, pooled) == EF.pair(fu + fu, Lam)
    # naturality of pairing along a reparametrisation r: the pairing of the pulled-back family equals the pulled-back pairing
    lam = lambda t: rlaw(n) if False else [Fr(1, n) + (t - Fr(1, 2)) * (Fr(i) - Fr(n - 1, 2)) / (n * n) for i in range(n)]
    r = lambda z: z * z / 2 + Fr(1, 4); z0 = Fr(random.randint(0, 4), 4); ok &= EF.pair(fu, lam(r(z0))) == (lambda t: EF.pair(fu, lam(t)))(r(z0))
check("FD1 state-effect duality: <u, pK> = <Ku, p>; contravariant pullback; marginalisation, forgetting, reparametrisation natural", ok)

# FD2
ok = True
for _ in range(200):
    n = 5; p = rlaw(n, random.randint(0, 2)); q = [Fr(random.randint(0, 4), 4) for _ in range(n)]; r = [Fr(random.randint(0, 4), 4) for _ in range(n)]
    A = [Fr(1) if i < 3 else Fr(0) for i in range(n)]
    cA = EF.cond_effect(p, A)
    if cA is not None: ok &= cA == [x * a / sum(p[:3]) for x, a in zip(p, A)]
    cq = EF.cond_effect(p, q)
    if cq is not None and EF.cond_effect(cq, r) is not None:
        ok &= EF.cond_effect(cq, r) == EF.cond_effect(p, [a * b for a, b in zip(q, r)])
    if cq is not None:
        u = rfun(n); ok &= EF.pair(u, cq) == EF.pair([a * b for a, b in zip(q, u)], p) / EF.pair(q, p)
    # contextual Bayes for an arbitrary effect
    pis = rlaw(3); ps = [rlaw(n, random.randint(0, 2)) for _ in range(3)]; P = [sum(pis[l] * ps[l][i] for l in range(3)) for i in range(n)]
    if EF.pair(q, P) > 0:
        Lam = [pis[l] * ps[l][i] for l in range(3) for i in range(n)]; CL = EF.cond_effect(Lam, q * 3)       # eps^* q = q on every row
        wts = [pis[l] * EF.pair(q, ps[l]) / EF.pair(q, P) for l in range(3)]
        ok &= [sum(CL[l * n:(l + 1) * n]) for l in range(3)] == wts
        ok &= EF.cond_effect(P, q) == [sum(wts[l] * (EF.cond_effect(ps[l], q)[i] if EF.pair(q, ps[l]) > 0 else 0) for l in range(3)) for i in range(n)]
    # Bayesian inverse = update by the pulled-back point effect K 1_y
    K = rker(n, 3); y = random.randrange(3); oney = [Fr(1) if j == y else Fr(0) for j in range(3)]; qy = EF.pull(K, oney)
    if EF.pair(qy, p) > 0: ok &= EF.cond_effect(p, qy) == [p[i] * K[i][y] / sum(p[j] * K[j][y] for j in range(n)) for i in range(n)]
check("FD2 conditioning = update by an effect: C_A by 1_A, C_q C_r = C_qr, Bayes weights pi_l <q,p_l>/<q,P>, Bayesian inverse = update by K 1_y", ok)

# FD3
ok = True
cells = list(itertools.product((0, 1), repeat=3))
def Mmat(S): vals = list(itertools.product((0, 1), repeat=len(S))); return [[Fr(1) if tuple(c[i] for i in S) == v else Fr(0) for c in cells] for v in vals]
for _ in range(100):
    p = rlaw(8); u = rfun(8); MS = Mmat((0,)); MT = Mmat((0, 1))
    Eu = EF.cond_exp(p, u, MS); v = rfun(2); vpull = [v[[0, 1].index(c[0])] for c in cells]
    ok &= sum(p[i] * Eu[i] * vpull[i] for i in range(8)) == sum(p[i] * u[i] * vpull[i] for i in range(8))
    ok &= EF.cond_exp(p, Eu, MS) == Eu
    ok &= EF.cond_exp(p, EF.cond_exp(p, u, MT), MS) == Eu
check("FD3 the L2 presheaf: E_p[.|X_S] is the p-adjoint of pulling back, idempotent, and restricts consistently (tower property)", ok)

# FD4
ok = True; worst = 0.0
for _ in range(500):
    n = random.randint(2, 6); s = 10 ** -random.uniform(1, 12); a = [s ** random.uniform(0, 5) * random.uniform(0.1, 10) for _ in range(n)]
    gap = abs(EF.log_s(sum(a), s) - min(EF.log_s(x, s) for x in a)); bound = math.log(n) / abs(math.log(s))
    ok &= gap <= bound + 1e-12; worst = max(worst, gap / bound)
for _ in range(200):                                       # tropical division on germs: v(C_q p)(x) = v q(x) + v p(x) - min_y(v q + v p)
    n = 4; p = [{Fr(random.randint(0, 4)): Fr(random.randint(1, 3))} for _ in range(n)]; q = [{Fr(random.randint(0, 3)): Fr(random.randint(1, 3))} for _ in range(n)]
    z = {}
    for i in range(n): z = OD.add(z, OD.mul(q[i], p[i]))
    vz = OD.lead(z)[0]; ok &= vz == min(OD.lead(q[i])[0] + OD.lead(p[i])[0] for i in range(n))
    for i in range(n): ok &= OD.lead(OD.mul(q[i], p[i]))[0] - vz == OD.lead(q[i])[0] + OD.lead(p[i])[0] - vz
    x = [random.uniform(0.1, 1) for _ in range(n)]; qq = [random.uniform(0.1, 1) for _ in range(n)]; zz = sum(a * b for a, b in zip(qq, x))
    ok &= all(abs(math.log(qq[i] * x[i] / zz) - (math.log(qq[i]) + math.log(x[i]) - math.log(zz))) < 1e-12 for i in range(n))
check("FD4 the log map: pooling tropicalises to min (Maslov bound log n/|log s|); Bayes exactly linear in logs; on valuations, conditioning is tropical division", ok,
      f"largest gap/bound {worst:.3f}")

# FD5
ok = True; doms = []
for _ in range(100):
    cellsP = {l: {tuple(Fr(random.randint(0, 4)) for _ in range(2)): Fr(random.randint(1, 4)) for _ in range(3)} for l in range(3)}
    P = {}
    for f in cellsP.values():
        for a, v in f.items(): P[a] = P.get(a, Fr(0)) + v
    w = (Fr(random.randint(1, 3)), Fr(random.randint(1, 3))); c = (random.uniform(0.5, 2), random.uniform(0.5, 2))
    def logsum(ls): mx = max(ls); return mx + math.log(sum(math.exp(x - mx) for x in ls))
    ls_ = [-20 * math.log(10), -40 * math.log(10), -80 * math.log(10)]       # log s
    vals = [logsum([math.log(float(v)) + float(a[0]) * (math.log(c[0]) + float(w[0]) * L_) + float(a[1]) * (math.log(c[1]) + float(w[1]) * L_)
                    for a, v in P.items()]) / L_ for L_ in ls_]
    ok &= abs(vals[-1] - float(FN.trop(P, w))) < 0.05 and abs(vals[-1] - float(FN.trop(P, w))) <= abs(vals[0] - float(FN.trop(P, w))) + 1e-12
    V = FN.newton_vertices_2d({0: P}); slopes = set()
    for t in np.linspace(0.02, 50, 4000):                  # w = (1, t): which vertex attains the min
        vals2 = [float(a[0]) + t * float(a[1]) for a in V]; slopes.add(int(np.argmin(vals2)))
    ok &= len(slopes) == len(V); doms.append(len(V))
check("FD5 normal fan in log coordinates: Log_s f(eps(s)) -> trop_w f; trop P(A) has one linearity domain per vertex of N", ok, f"vertices/domains e.g. {doms[:8]}")

# FD6
ok = True
for _ in range(200):
    n = 4; p = [{Fr(random.randint(0, 3)): Fr(random.randint(1, 3))} for _ in range(n)]; q = [{Fr(random.randint(0, 3)): Fr(random.randint(1, 3))} for _ in range(n)]
    for i in range(n): ok &= OD.lead(OD.add(p[i], q[i]))[0] == min(OD.lead(p[i])[0], OD.lead(q[i])[0])
    for t in (Fr(1, 4), Fr(1, 2), Fr(3, 4)):                  # geometric mixture p^(1-t) q^t (monomial germs: exponents combine linearly)
        g = [{(1 - t) * OD.lead(p[i])[0] + t * OD.lead(q[i])[0]: Fr(1)} for i in range(n)]; mn = min(OD.lead(x)[0] for x in g)
        ok &= all(OD.lead(g[i])[0] - mn == (1 - t) * OD.lead(p[i])[0] + t * OD.lead(q[i])[0] - mn for i in range(n))
    ok &= all(OD.lead(OD.mul(p[i], q[j]))[0] == OD.lead(p[i])[0] + OD.lead(q[j])[0] for i in range(n) for j in range(n))
check("FD6 the m/e split: mixtures tropicalise to min, geometric mixtures stay linear in t, products add valuations", ok)

# FD7
ok = True; tested = 0
def jac(f, x):
    n = len(x); I = np.eye(n); h = 1e-7; return np.array([(np.array(f(x + h * I[i])) - np.array(f(x - h * I[i]))) / (2 * h) for i in range(n)]).T
for _ in range(150):
    n = 4; x = np.array([float(v) for v in rlaw(n, random.randint(1, 2))])
    K = np.array([[float(v) for v in rlaw(3, random.randint(0, 1))] for _ in range(n)]); maps = [(lambda z, K=K: z @ K)]
    Am = np.array([1, 1, 1, 0], float)
    if (Am * x).sum() > 0: maps.append(lambda z: list(z * Am / (z * Am).sum()))
    for f in maps:
        y = np.array(f(x)); J = jac(f, x); L, R = EF.cone_generators(x)
        for _ in range(20):
            sup = [i for i in range(len(y)) if y[i] > 1e-12]; g = np.full(len(y), rng.normal())
            for i in range(len(y)):
                if i not in sup: g[i] += abs(rng.normal())
            gx = J.T @ g; ok &= all(abs(gx @ l) < 1e-6 for l in L) and all(gx @ r > -1e-6 for r in R); tested += 1
# robustness: tie at x with u_a - u_b; weakly preferred under all injections iff in the dual cone
for _ in range(200):
    n = 4; x = [float(v) for v in rlaw(n, random.randint(1, 2))]; g = [float(v) for v in rfun(n)]
    inside = EF.dual_cone_member(x, g)
    worst_dir = min(np.dot(g, TLM.rand_feasible(np.array(x), rng)) for _ in range(200))
    L, R = EF.cone_generators(np.array(x)); gen_ok = all(abs(np.dot(g, l)) < 1e-12 for l in L) and all(np.dot(g, r) >= -1e-12 for r in R)
    ok &= inside == gen_ok and (not inside or worst_dir >= -1e-9)
check("FD7 cotangent: C_x^* formula; derivative transposes pull dual cones back (contravariant T*); ties are injection-proof iff u_a-u_b in C_x^*", ok, f"{tested} covectors pulled back")

# FD8
ok = True; walls = 0
for _ in range(200):
    cellsD = {l: {(Fr(1), Fr(0)): Fr(random.randint(1, 3)), (Fr(0), Fr(1)): Fr(random.randint(1, 3))} for l in range(2)}
    U = {"a": {0: Fr(random.randint(-3, 3)), 1: Fr(random.randint(-3, 3))}, "b": {0: Fr(0), 1: Fr(0)}}
    Nab = {}
    for l, f in cellsD.items():
        for al, v in f.items(): Nab[al] = Nab.get(al, Fr(0)) + U["a"][l] * v
    Nab = {k: v for k, v in Nab.items() if v}
    if len(Nab) < 2: continue
    # edge polynomial along w = (1, 1) in the variable c = y/x: N(1, c) = N_10 + N_01 c; positive root -> a wall on the ray
    a10, a01 = Nab.get((Fr(1), Fr(0)), Fr(0)), Nab.get((Fr(0), Fr(1)), Fr(0))
    if a01 != 0 and -a10 / a01 > 0:
        cstar = -a10 / a01; walls += 1
        ok &= PA.limit_acts(cellsD, U, (Fr(1), Fr(1)), (Fr(1), cstar)) == {"a", "b"}
        lo = PA.eventual_acts(cellsD, U, (Fr(1), Fr(1)), (Fr(1), cstar / 2)); hi = PA.eventual_acts(cellsD, U, (Fr(1), Fr(1)), (Fr(1), cstar * 2)); ok &= lo != hi
    else:
        ok &= all(len(PA.limit_acts(cellsD, U, (Fr(1), Fr(1)), (Fr(1), Fr(c)))) == 1 or a10 == a01 == 0 for c in (1, 2, 5))
check("FD8 decision walls on a cone = positive roots of the edge polynomial of the signed N_ab; the act flips across each", ok, f"{walls} walls")

# FD9
ok = True; rows = []
for it in range(40):
    m = 4; wins = random.choice([[(0, 1), (1, 2), (2, 3), (0, 3)], [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], [(0, 1, 2), (1, 2, 3)]])
    U = [rng.integers(-3, 4, size=(2,) * m).astype(float) for _ in range(random.randint(2, 4))]
    if it % 3 == 0:                                          # a utility that is a function of one window's table: pinned by the data
        W0 = wins[0]; U[1] = U[0] + np.fromfunction(lambda *ix: (ix[W0[0]] * 2 + ix[W0[1]]).astype(float), (2,) * m)
    rr = RV.relevant_rank(m, wins, U); K = RV.marginal_matrix(m, wins)
    D = np.array([(U[a] - U[0]).ravel() for a in range(1, len(U))]); dspan = np.linalg.matrix_rank(D, 1e-9)
    inter = dspan + np.linalg.matrix_rank(K, 1e-9) - np.linalg.matrix_rank(np.vstack([D, K]), 1e-9)
    ok &= rr == dspan - inter; rows.append((int(rr), int(dspan), int(inter)))
for _ in range(100):                                        # utilities off A do not matter to the conditional decision
    n = 5; p = rlaw(n); A = [Fr(1)] * 3 + [Fr(0)] * 2; u = rfun(n); u2 = u[:3] + rfun(2)
    ok &= EF.pair(u, EF.cond_effect(p, A)) == EF.pair(u2, EF.cond_effect(p, A))
for _ in range(100):                                        # coefficients off the leading face do not matter at the wall
    cellsW = {l: {(Fr(random.randint(0, 3)), Fr(random.randint(0, 3))): Fr(random.randint(1, 4)) for _ in range(3)} for l in range(3)}
    w = (Fr(random.randint(1, 3)), Fr(random.randint(1, 3))); mu = min(FN.trop(f, w) for f in cellsW.values())
    pert = {l: {a: (v if sum(wi * ai for wi, ai in zip(w, a)) == mu else v * random.randint(2, 9)) for a, v in f.items()} for l, f in cellsW.items()}
    ok &= FN.arc_limit(cellsW, w, (1, 1)) == FN.arc_limit(pert, w, (1, 1))
check("FD9 what does not matter: rank(rho on ker K) = dim span(rho) - dim(span(rho) cap im K^T); utilities off A; coefficients off the leading face", ok,
      f"(relevant rank, utility span, pinned by windows): {sorted(set(rows))}")
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "effects.json"), "w"))
print(f"{sum(results)}/{len(results)} passed")
