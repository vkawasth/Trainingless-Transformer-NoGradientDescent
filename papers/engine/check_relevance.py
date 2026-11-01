"""Checks for "what does not matter" (lab Section "What does not matter: the decision-relevant quotient").
  RL1  gauge: an act whose utility depends on observables has the same expected utility under every realisation of a law (label
       permutations); an act indexed by a latent type does not
  RL2  the decision-relevant dimension: at m = 4 the pair cover's hidden kernel has dimension 5, but random two- and three-act problems
       see rank 1 and 2 of it, and acts on pair events see rank 0 (settled by the pairs alone)
  RL3  settling: an act is optimal on the whole fibre iff every utility-difference range (two linear programmes each) lies on one side
       of 0; checked against 200 random fibre points per problem
  RL4  redundant windows: dropping a window can leave every utility-difference range unchanged; a greedy pass gives a decision-sufficient
       subcover with the same ranges
  RL5  decision value is a Vigneaux coboundary: V[X] = R - X.R satisfies V[XY] = V[X] + X.V[Y] on random laws and on two-type laws, and
       the two-type family is stable under conditioning on observed variables (an adapted, curved family)
  RL6  irrelevance: V[X](P) = 0 iff one act is optimal under every conditional law P | X = x
  RL7  rectangular sets of laws: the ex-ante maxmin value of observing X equals the worst-case value of prior-by-prior updating (300 problems)
  RL8  the lab's hiding segment is not rectangular: updating loses value in a positive fraction of problems, and can make observing X
       worth less than not observing it
  RL9  wall-crossing for the value of information: V[X](P) = min_a <l_a, P> - sum_x min_a <l_a 1_{X=x}, P> exactly; along the hiding
       segment V is piecewise linear, its kinks lie on decision walls of P or of a conditional, and each gradient jump equals the change
       of the active loss vector
"""
import itertools, json, os, sys
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import relevance as RV, three as TH
rng = np.random.default_rng(20261031); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def rand_law(m): return rng.dirichlet(np.ones(2 ** m)).reshape((2,) * m)

# RL1
ok = True; moved = 0.0
for _ in range(30):
    w = rng.dirichlet([2] * 3); a = rng.uniform(0.05, 0.95, (3, 4)); P = TH.law(w, a); u = rng.normal(size=(2,) * 4)
    for p in itertools.permutations(range(3)):
        P2 = TH.law(w[list(p)], a[list(p)]); ok &= abs((P * u).sum() - (P2 * u).sum()) < 1e-14
        latent = lambda ww, aa: float(ww[0] * TH.law(np.array([1.0]), aa[:1]).ravel() @ u.ravel())     # utility of "act on type 1"
        moved = max(moved, abs(latent(w, a) - latent(w[list(p)], a[list(p)])))
check("RL1 gauge: observable acts are invariant under relabelling; an act indexed by a latent type is not", ok and moved > 1e-3, f"latent-indexed change up to {moved:.3f}")

# RL2
m = 4; pairs = [list(e) for e in itertools.combinations(range(4), 2)]; kd = RV.kernel(m, pairs).shape[1]
r2 = {RV.relevant_rank(m, pairs, [rng.normal(size=(2,) * m) for _ in range(2)]) for _ in range(20)}
r3 = {RV.relevant_rank(m, pairs, [rng.normal(size=(2,) * m) for _ in range(3)]) for _ in range(20)}
r0 = set()
for _ in range(20):
    i, j = sorted(rng.choice(4, 2, replace=False)); U = [np.zeros((2,) * m), np.zeros((2,) * m)]
    idx = [slice(None)] * 4; idx[i] = 1; idx[j] = 1; U[1][tuple(idx)] = rng.normal(); U[1] += rng.normal(); r0.add(RV.relevant_rank(m, pairs, U))
OUT["RL2"] = dict(kernel=kd, two=sorted(r2), three=sorted(r3), pair_event=sorted(r0))
check("RL2 decision-relevant dimension: hidden kernel 5 at m = 4; two acts see 1, three see 2, pair-event acts see 0", kd == 5 and r2 == {1} and r3 == {2} and r0 == {0},
      str(OUT["RL2"]))

# RL3
K = RV.marginal_matrix(m, pairs); ok3 = True; nset = 0
for _ in range(15):
    P = rand_law(m); U = [rng.normal(size=(2,) * m) * 0.2 for _ in range(3)]; a = RV.settled(P, pairs, U); nset += a is not None
    b = K @ P.ravel(); opt = set()
    for _ in range(200):
        r = linprog(rng.normal(size=16), A_eq=K, b_eq=b, bounds=[(0, None)] * 16, method="highs"); Q = r.x
        opt.add(int(np.argmax([Q @ Ua.ravel() for Ua in U])))
    ok3 &= (a is not None) == (len(opt) == 1) and (a is None or opt == {a})
check("RL3 settled iff every utility-difference range lies on one side of 0 (vs 200 fibre points per problem)", ok3, f"{nset} of 15 settled")

# RL4
wins = pairs + [[0, 1, 2], [1, 2, 3]]; P = rand_law(m)
U = [np.zeros((2,) * m), np.zeros((2,) * m)]; U[1][1, 1, 1, :] = 1.0; U[1] -= 0.08
base = RV.rho_range(P, wins, U[1] - U[0]); cur = list(wins); changed = True
while changed:
    changed = False
    for W in list(cur):
        rest = [V for V in cur if V != W]
        if rest and np.allclose(RV.rho_range(P, rest, U[1] - U[0]), base, atol=1e-10): cur = rest; changed = True; break
OUT["RL4"] = dict(kept=cur, dropped=[W for W in wins if W not in cur])
check("RL4 a decision-sufficient subcover: windows dropped greedily without changing the utility-difference range", len(cur) < len(wins) and np.allclose(RV.rho_range(P, cur, U[1] - U[0]), base),
      f"kept {cur} of {len(wins)} windows")

# RL5
L = [rng.normal(size=(2,) * 4) for _ in range(3)]; err = 0.0
for _ in range(100):
    P = rand_law(4) if _ % 2 else TH.law(rng.dirichlet([2, 2]), rng.uniform(0.05, 0.95, (2, 4)))
    for X, Y in (([0], [1]), ([1], [0]), ([2], [0, 3])):
        lhs = RV.value(P, X + Y, L); rhs = RV.value(P, X, L) + RV.act_on(lambda Q: RV.value(Q, Y, L), P, X); err = max(err, abs(lhs - rhs))
adapted = 0.0
for _ in range(20):
    w = rng.dirichlet([2, 2]); a = rng.uniform(0.05, 0.95, (2, 4)); P = TH.law(w, a)
    for x in (0, 1):
        Q, p = RV.condition(P, [0], (x,)); wx = w * np.where(x, a[:, 0], 1 - a[:, 0]); wx /= wx.sum()
        a2 = a.copy(); a2[:, 0] = x; adapted = max(adapted, np.abs(Q - TH.law(wx, a2)).max())
check("RL5 decision value V = R - X.R is a Vigneaux 1-cocycle (chain rule), also on the curved two-type family, which is stable under conditioning",
      err < 1e-12 and adapted < 1e-12, f"chain-rule error {err:.1e}, conditioning error {adapted:.1e}")

# RL6
ok6 = True; zeros = 0
for _ in range(300):
    P = rand_law(3); L3 = [rng.normal(size=(2, 2, 2)) for _ in range(3)]
    if _ % 3 == 0:                                                       # make one act optimal everywhere given X0
        L3[0] = np.minimum.reduce(L3) - 0.1
    v = RV.value(P, [0], L3)
    conds = [RV.condition(P, [0], (x,))[0] for x in (0, 1)]
    common = any(all(float((Q * L3[a]).sum()) <= min(float((Q * l).sum()) for l in L3) + 1e-12 for Q in conds) for a in range(3))
    ok6 &= (abs(v) < 1e-12) == common; zeros += abs(v) < 1e-12
check("RL6 V[X](P) = 0 iff one act is optimal under every conditional", ok6, f"{zeros} irrelevant cases of 300")

# RL7, RL8
chi = np.array([[[(2 * a - 1) * (2 * b - 1) * (2 * c - 1) for c in (0, 1)] for b in (0, 1)] for a in (0, 1)], float)
def rect_set():
    px = [rng.dirichlet([2, 2]) for _ in range(2)]; q = {x: [rng.dirichlet(np.ones(4)).reshape(2, 2) for _ in range(2)] for x in (0, 1)}; out = []
    for p in px:
        for q0 in q[0]:
            for q1 in q[1]:
                P = np.zeros((2, 2, 2)); P[0] = p[0] * q0; P[1] = p[1] * q1; out.append(P)
    return out
def segment():
    while True:
        P = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); smin = -P[chi > 0].min(); smax = P[chi < 0].min()
        if smax - smin > 0.05: return [P + smin * chi, P + smax * chi]
def noinfo(Qs, U):
    M = np.array([[float((Q * Ua).sum()) for Ua in U] for Q in Qs]); k = len(U)
    r = linprog(np.r_[np.zeros(k), -1], A_ub=np.hstack([-M, np.ones((len(Qs), 1))]), b_ub=np.zeros(len(Qs)), A_eq=[list(np.ones(k)) + [0]], b_eq=[1],
                bounds=[(0, None)] * k + [(None, None)], method="highs"); return -r.fun
gr, gs, neg = [], [], 0
for _ in range(300):
    U = [rng.normal(size=(2, 2, 2)) for _ in range(3)]
    a, u = RV.ante_and_updated(rect_set(), [0], U); gr.append(a - u)
    S = segment(); a, u = RV.ante_and_updated(S, [0], U); gs.append(a - u); neg += u < noinfo(S, U) - 1e-9
gr, gs = np.array(gr), np.array(gs); OUT["RL7_8"] = dict(rect_max_gap=float(np.abs(gr).max()), segment_positive=int((gs > 1e-9).sum()), segment_max=float(gs.max()), negative_value=int(neg))
check("RL7 rectangular sets: ex-ante maxmin value = worst-case value of prior-by-prior updating (300 problems)", np.abs(gr).max() < 1e-9, f"max gap {np.abs(gr).max():.1e}")
check("RL8 the hiding segment is not rectangular: updating loses value in a positive fraction of problems, sometimes below ignoring X",
      (gs > 1e-9).sum() > 30 and gs.min() > -1e-9 and neg > 0, f"gap > 0 in {(gs > 1e-9).sum()}/300 (max {gs.max():.3f}); value of X negative after updating in {neg}")

# RL9
ok9 = True; nk = 0; err9 = 0.0
for _ in range(40):
    P0 = segment()[0]; P1 = segment()[0]; Pa, Pb = segment(); Ls = [rng.normal(size=(2, 2, 2)) for _ in range(3)]
    form = lambda P: min(float((P * l).sum()) for l in Ls) - sum(min(float((P[x] * l[x]).sum()) for l in Ls) for x in (0, 1))
    err9 = max(err9, abs(form(Pa) - RV.value(Pa, [0], Ls)))
    ss = np.linspace(0, 1, 4001); Ps = [(1 - t) * Pa + t * Pb for t in ss]; V = np.array([form(P) for P in Ps])
    # predicted kinks: changes of the active act for P and for each conditional block
    act = lambda P, blk: int(np.argmin([float((P[blk] * l[blk]).sum()) for l in Ls]))
    full = [int(np.argmin([float((P * l).sum()) for l in Ls])) for P in Ps]; c0 = [act(P, 0) for P in Ps]; c1 = [act(P, 1) for P in Ps]
    change = [i for i in range(1, len(ss)) if full[i] != full[i - 1] or c0[i] != c0[i - 1] or c1[i] != c1[i - 1]]
    d2 = np.abs(np.diff(V, 2)); near = set()
    for i in change: near.update(range(i - 2, i + 1))
    away = [j for j in range(len(d2)) if j not in near]; ok9 &= (d2[away].max() if away else 0) < 1e-12; nk += len(change)
    for i in change:                                           # jump of the slope equals the change of the active vectors along Pb - Pa
        D = Pb - Pa; g = lambda k: float((D * Ls[full[k]]).sum()) - float((D[0] * Ls[c0[k]][0]).sum()) - float((D[1] * Ls[c1[k]][1]).sum())
        slope_l = (V[i - 1] - V[i - 2]) / (ss[1] - ss[0]); slope_r = (V[i + 1] - V[i]) / (ss[1] - ss[0])
        if i + 1 < len(V) and i - 2 >= 0 and all(j not in change for j in (i - 1, i + 1)): ok9 &= abs((slope_r - slope_l) - (g(i) - g(i - 1))) < 1e-6
OUT["RL9"] = dict(kinks=nk, identity_error=err9)
check("RL9 wall-crossing for the value of information: exact piecewise-linear formula; kinks only on decision walls of P or a conditional; jumps match",
      ok9 and nk > 10 and err9 < 1e-12, f"{nk} kinks on 40 chords, identity error {err9:.1e}")

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "relevance.json"), "w"), indent=1, default=str)
print(f"{sum(results)}/{len(results)} passed")
