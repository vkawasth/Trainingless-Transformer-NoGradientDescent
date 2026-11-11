"""Checks for the history trace and the decision theorem (module pce/historytrace.py). Fixed state space X in {a,b,c,d} x Y in {0,1};
current data D: the tables of {X} and of {X', Y} (X' = class of X); H_D = constants + functions of x + functions of (x', y), dim 6, so
the fibre has dimension 2. Foreign contexts: aY = (1{x = a}, y), eY = (1{x in {a, c}}, y), XY = (x, y). Honest updater: refit of uniform.
  HT1 IDENTITY  every step is multiplicative and a refit onto data D_t multiplies by exp(h_t) with h_t in H_{D_t} (300 random refits)
  HT2 IDENTITY  the final law depends only on the class of tau modulo H_D: Pi_J(U e^tau) = Pi_J(U e^(tau + h)) for h in H_D, and equals
                the projection of the canonical representative (200 random histories mixing every kind of step)
  HT3 FINDING   path independence: histories whose edits and intermediate refits use only the current data's contexts, in any order and
                any number, end at the honest law (100 histories); one intermediate refit with a foreign context can make the end law depend
                on the order of the same steps, and does so in most random histories but not all. (A first version required order to matter
                in every one of 20 histories; it failed with a minimum of 0.0000, because when the other steps do not change the foreign
                table's margin, the foreign factor is the same in every order. Restated as a majority with a strictly positive maximum.)
  HT4 IDENTITY  the history coordinate: dim(F / H_D) = dim J; every positive law of the fibre is reached by some history (100 laws); distinct
                classes give distinct laws (300 pairs)
  HT5 IDENTITY  a merge-and-reconstruction is the multiplicative update theta(x | c, y) / P(x | c, y); from the honest law its class is the
                class of log theta, which vanishes iff the within-class odds do not depend on y (50 + 50 kernels): Prop. 142's criterion
                recovered from the trace theorem
  HT6 IDENTITY  audit and attribution: the trace log R - log R_honest recovers the injected class modulo H_D; an edit on aY lies in
                H_D + F_aY and not in H_D + F_eY; an edit on the full table XY is attributable to neither single context
  HT7 FINDING   mechanisms are indistinguishable and can cancel: the trace of an outcome-dependent split in class A lies in H_D + F_aY, the
                same class as a retracted edit on aY, and a likelihood factor on aY cancels it exactly though each alone moves the law
  HT8 IDENTITY  decisions: a functional in H_D never moves under any history (200); an open functional with a positive honest margin is
                reversed by a history through the full table; on the one-dimensional slice of edits on aY its range lies inside the fibre's
                range; every history moves an expected utility by at most osc(g) times the TV distance
  HC1 IDENTITY  path discrepancy is a pseudometric on kernels from the fibre (triangle inequality, 300 triples)
  HC2 IDENTITY  contraction: H(Ka M, Kb M) <= delta(M) H(Ka, Kb), delta the Dobrushin coefficient (300 random M)
  HC3 IDENTITY  concatenation: H_J(g1 g2, g1' g2') <= H_J'(g2, g2') + delta(g2') H_J(g1, g1') when J g1 lies in J' (200 cases)
  HC4 IDENTITY  invariance: a lossless re-encoding Phi (one with a left-inverse kernel) leaves H unchanged; a lossy one (a merge) can only
                lower it (200 cases)
"""
import os, sys, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from pce import historytrace as HT, kerneltransport as KT
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(31); F = KT.FINE; U = np.ones(8) / 8
P0 = np.array([0.10, 0.06, 0.04, 0.12, 0.14, 0.09, 0.15, 0.30]); P0 = P0 / P0.sum()
cX, cXcY = KT.ctx_X, KT.ctx_Xc_Y
caY = lambda w: (w[0] == 0, w[1]); ceY = lambda w: (w[0] in (0, 2), w[1]); cXY = KT.ctx_X_Y; cY = KT.ctx_Y
DATA = [(cX, KT.table(P0, cX, F)), (cXcY, KT.table(P0, cXcY, F))]
BD = HT.span([cX, cXcY], F); V = KT.vertices(DATA, F)
Rh = KT.refit(U, DATA, F)
fa = np.array([1.0 if (x == 0 and y == 1) else 0.0 for x, y in F])
BaY = np.column_stack([BD, HT.span([caY], F)]); BeY = np.column_stack([BD, HT.span([ceY], F)])
ALL_CTX = [cX, cXcY, caY, ceY, cXY, cY]

def rand_tables(ctxs):
    Pt = rng.dirichlet(np.ones(8) * 3); return [(c, KT.table(Pt, c, F)) for c in ctxs]
def rand_step(foreign=True):
    k = rng.integers(4)
    ctxs = ALL_CTX if foreign else [cX, cXcY, cY]
    if k == 0 and foreign:
        L = np.exp(rng.normal(0, 0.5, 8)); return lambda P: HT.step_likelihood(P, L)
    if k <= 1:
        c = ctxs[rng.integers(len(ctxs))]; (c, t), = rand_tables([c]); return lambda P: HT.step_jeffrey(P, c, t, F)
    if k == 2 or not foreign:
        sub = [c for c in ctxs if rng.random() < 0.5] or [ctxs[0]]; tabs = rand_tables(sub); return lambda P: HT.step_refit(P, tabs, F)
    t4 = rng.uniform(0.1, 0.9, 4); th = {(0, 0): t4[0], (0, 1): t4[1], (1, 0): t4[2], (1, 1): t4[3]}
    return lambda P: HT.step_mergesplit(P, th)

# HT1
ok = True; worst = 0.0
for _ in range(300):
    sub = [c for c in ALL_CTX if rng.random() < 0.4] or [cXY]; tabs = rand_tables(sub); Q = rng.dirichlet(np.ones(8))
    _, lf = HT.step_refit(Q, tabs, F); r = HT.residual(lf, HT.span(sub, F)); worst = max(worst, r)
ok = worst < 1e-6
check("HT1 IDENTITY a refit onto data D_t multiplies the law by exp(h_t) with h_t in H_{D_t}", ok, f"largest residual {worst:.1e}")

# HT2
ok = True; worst = 0.0
for _ in range(200):
    steps = [rand_step() for _ in range(rng.integers(3, 9))]; P, tau = HT.run(U, steps)
    ok &= np.abs(HT.normalise(U * np.exp(tau)) - P).max() < 1e-9
    R = KT.refit(P, DATA, F); h = BD @ rng.normal(0, 1, BD.shape[1])
    R2 = KT.refit(HT.normalise(U * np.exp(tau + h)), DATA, F); R3 = KT.refit(HT.normalise(U * np.exp(HT.project_out(tau, BD))), DATA, F)
    worst = max(worst, KT.tv(R, R2), KT.tv(R, R3))
ok &= worst < 1e-8
check("HT2 IDENTITY the end law depends on the history only through tau modulo H_D", ok, f"largest TV between representatives {worst:.1e}")

# HT3
ok = True
for _ in range(100):
    steps = [rand_step(foreign=False) for _ in range(rng.integers(2, 10))]; P, _ = HT.run(U, steps); ok &= KT.tv(KT.refit(P, DATA, F), Rh) < 1e-8
dep = []
for _ in range(20):
    base = [rand_step(foreign=False) for _ in range(3)]; tabs = rand_tables([caY]); fr = lambda P, tabs=tabs: HT.step_refit(P, tabs, F)
    ends = [KT.refit(HT.run(U, list(perm))[0], DATA, F) for perm in itertools.permutations(base + [fr])]
    dep.append(max(KT.tv(a, b) for a in ends for b in ends))
indep = ok; nd = int(sum(d > 1e-4 for d in dep)); ok &= nd > len(dep) / 2 and max(dep) > 1e-2
check("HT3 FINDING path independence for histories on the data's own contexts; a foreign intermediate table can make order matter", ok,
      f"own-context histories at the honest law: {indep}; order matters in {nd} of {len(dep)} histories with one foreign refit (max TV {max(dep):.4f})")

# HT4
dq = HT.quotient_dim(BD, F); A, b = KT.constraints(DATA, F); dJ = 8 - np.linalg.matrix_rank(A)
ok = dq == dJ; worst = 0.0
for w in rng.dirichlet(np.ones(len(V)), 100):
    R = 0.97 * (w @ V) + 0.03 * Rh; Rr = KT.refit(HT.normalise(U * np.exp(np.log(R / U))), DATA, F); worst = max(worst, KT.tv(Rr, R))
ok &= worst < 1e-8; sep = []
for _ in range(300):
    t1, t2 = rng.normal(0, 1, (2, 8)); d = HT.residual(t1 - t2, BD)
    if d < 1e-3: continue
    sep.append(KT.tv(KT.refit(HT.normalise(U * np.exp(t1)), DATA, F), KT.refit(HT.normalise(U * np.exp(t2)), DATA, F)))
ok &= min(sep) > 1e-6
check("HT4 IDENTITY the trace class is a coordinate on the fibre: dimensions agree, every positive law is reached, distinct classes differ",
      ok, f"dim F/H_D = {dq} = dim J = {dJ}; reach error {worst:.1e}; smallest TV between distinct classes {min(sep):.2e}")

# HT5
zero, pos, idok = [], [], True
for k in range(100):
    if k < 50: t = rng.uniform(0.05, 0.95, 2); th = {(0, 0): t[0], (0, 1): t[0], (1, 0): t[1], (1, 1): t[1]}
    else: t4 = rng.uniform(0.05, 0.95, 4); th = {(0, 0): t4[0], (0, 1): t4[1], (1, 0): t4[2], (1, 1): t4[3]}
    Q, lf = HT.step_mergesplit(Rh, th)
    theta = np.array([th[(KT.cls(x), y)] if x in (0, 2) else 1 - th[(KT.cls(x), y)] for x, y in F])
    cond = np.array([Rh[i] / sum(Rh[j] for j, (xx, yy) in enumerate(F) if KT.cls(xx) == KT.cls(x) and yy == y) for i, (x, y) in enumerate(F)])
    idok &= np.abs(lf - (np.log(theta) - np.log(cond))).max() < 1e-9 and HT.residual(lf - np.log(theta), BD) < 1e-9
    (zero if k < 50 else pos).append((HT.residual(np.log(theta), BD), KT.tv(KT.refit(Q, DATA, F), Rh)))
zero, pos = np.array(zero), np.array(pos)
ok = idok and zero[:, 0].max() < 1e-9 and zero[:, 1].max() < 1e-8 and pos[:, 0].min() > 1e-6 and pos[:, 1].min() > 1e-6
check("HT5 IDENTITY merge-and-reconstruction is a multiplicative update; its trace is the class of log theta (Prop. 142 recovered)", ok,
      f"outcome-free: residual max {zero[:, 0].max():.1e}, trace max {zero[:, 1].max():.1e}; outcome-dependent: residual min {pos[:, 0].min():.3f}, trace min {pos[:, 1].min():.4f}")

# HT6
ok = True; rec = 0.0; attr = []
for _ in range(50):
    (c, t), = rand_tables([caY]); P, tau = HT.run(U, [lambda P: HT.step_jeffrey(P, c, t, F)]); R = KT.refit(P, DATA, F)
    tr = HT.trace(R, Rh); rec = max(rec, HT.residual(tr - tau, BD)); attr.append((HT.residual(tr, BaY), HT.residual(tr, BeY), HT.residual(tr, BD)))
attr = np.array(attr); full = []
for _ in range(50):
    (c, t), = rand_tables([cXY]); P, tau = HT.run(U, [lambda P: HT.step_jeffrey(P, c, t, F)]); tr = HT.trace(KT.refit(P, DATA, F), Rh)
    full.append((HT.residual(tr, BaY), HT.residual(tr, BeY)))
full = np.array(full)
ok = rec < 1e-7 and attr[:, 0].max() < 1e-7 and attr[:, 1].min() > 1e-4 and full.min() > 1e-4
check("HT6 IDENTITY the audit recovers the injected class; attribution to a context is a span test", ok,
      f"recovery error {rec:.1e}; aY edits: residual from H_D+F_aY max {attr[:, 0].max():.1e}, from H_D+F_eY min {attr[:, 1].min():.3f}; XY edits: min residual {full.min():.3f}")

# HT7
th = {(0, 0): 0.2, (0, 1): 0.8, (1, 0): 0.5, (1, 1): 0.5}; Qs, lf = HT.step_mergesplit(Rh, th); Rs = KT.refit(Qs, DATA, F)
tr = HT.trace(Rs, Rh); coef = np.linalg.lstsq(np.column_stack([BD, fa]), tr, rcond=None)[0]; s = coef[-1]
in_aY = HT.residual(tr, BaY); Rl = KT.refit(HT.normalise(Rh * np.exp(s * fa)), DATA, F)
Rc = KT.refit(HT.normalise(Qs * np.exp(-s * fa)), DATA, F)
ok = in_aY < 1e-7 and KT.tv(Rs, Rh) > 0.05 and KT.tv(Rl, Rs) < 1e-8 and KT.tv(Rc, Rh) < 1e-8
check("HT7 FINDING an outcome-dependent split and a retracted edit on aY leave the same trace, and one cancels the other", ok,
      f"split alone TV {KT.tv(Rs, Rh):.4f} from honest; residual from H_D+F_aY {in_aY:.1e}; likelihood exp({s:.3f} 1{{a,1}}) reproduces it (TV {KT.tv(Rl, Rs):.1e}); "
      f"split then exp({-s:.3f} 1{{a,1}}) returns to honest (TV {KT.tv(Rc, Rh):.1e})")

# HT8
gdet = BD @ rng.normal(0, 1, BD.shape[1]); mv = 0.0; lip = True
for _ in range(200):
    steps = [rand_step() for _ in range(rng.integers(2, 7))]; R = KT.refit(HT.run(U, steps)[0], DATA, F); mv = max(mv, abs(gdet @ R - gdet @ Rh))
    g = rng.normal(0, 1, 8); lip &= abs(g @ R - g @ Rh) <= (g.max() - g.min()) * KT.tv(R, Rh) + 1e-12
while True:
    g = rng.normal(0, 1, 8); vals = V @ g
    if vals.min() < -0.05 and vals.max() > 0.05:
        if g @ Rh < 0: g = -g; vals = -vals
        if g @ Rh > 0.01: break
vmin = V[np.argmin(vals)]; Rt = 0.98 * vmin + 0.02 * Rh
Rrev = KT.refit(HT.normalise(U * (Rt / U)), DATA, F)
sl = [g @ KT.refit(HT.normalise(Rh * np.exp(s_ * fa)), DATA, F) for s_ in np.linspace(-15, 15, 61)]
ok = mv < 1e-8 and lip and g @ Rrev < 0 and vals.min() - 1e-9 <= min(sl) and max(sl) <= vals.max() + 1e-9
check("HT8 IDENTITY decisions: determined functionals never move; open ones are reversed through the full table; slices lie in the fibre range", ok,
      f"determined: max change {mv:.1e}; open: honest {g @ Rh:+.3f}, fibre range [{vals.min():+.3f}, {vals.max():+.3f}], reversed to {g @ Rrev:+.3f}; "
      f"aY slice range [{min(sl):+.3f}, {max(sl):+.3f}] ({'crosses' if min(sl) < 0 < max(sl) else 'does not cross'} zero)")

# HC1-HC4 on the fibre's vertices
def rk(n, m): return rng.dirichlet(np.ones(m) * 0.7, n)
ok = True
for _ in range(300):
    a, b, c = rk(8, 8), rk(8, 8), rk(8, 8); ok &= HT.pathdisc(a, c, V) <= HT.pathdisc(a, b, V) + HT.pathdisc(b, c, V) + 1e-12
check("HC1 IDENTITY path discrepancy satisfies the triangle inequality", ok)
ok = True; ratio = []
for _ in range(300):
    a, b, M = rk(8, 8), rk(8, 8), rk(8, 5); lhs = HT.pathdisc(a @ M, b @ M, V); rhs = HT.dobrushin(M) * HT.pathdisc(a, b, V)
    ok &= lhs <= rhs + 1e-12; ratio.append(lhs / rhs if rhs > 0 else 0)
check("HC2 IDENTITY contraction by the Dobrushin coefficient", ok, f"largest ratio {max(ratio):.3f}")
M = KT.merge(); cdata = [(KT.ctx_c, KT.table(P0 @ M, KT.ctx_c, KT.COARSE)), (KT.ctx_cY, KT.table(P0 @ M, KT.ctx_cY, KT.COARSE))]
Vc = KT.vertices(cdata, KT.COARSE); ok = True; slack = []
for _ in range(200):
    g1p = M @ rk(4, 4); g2, g2p = rk(4, 8), rk(4, 8)
    lhs = HT.pathdisc(M @ g2, g1p @ g2p, V); rhs = HT.pathdisc(g2, g2p, Vc) + HT.dobrushin(g2p) * HT.pathdisc(M, g1p, V)
    ok &= lhs <= rhs + 1e-12; slack.append(rhs - lhs)
check("HC3 IDENTITY concatenation bound through the transported fibre", ok, f"smallest slack {min(slack):.4f}")
ok = True; drop = []
for _ in range(200):
    a, b = rk(8, 8), rk(8, 8); r = rng.uniform(0.05, 0.95, 8)
    Phi = np.zeros((8, 16)); Psi = np.zeros((16, 8))
    for i in range(8): Phi[i, 2 * i] = r[i]; Phi[i, 2 * i + 1] = 1 - r[i]; Psi[2 * i, i] = Psi[2 * i + 1, i] = 1
    assert np.abs(Phi @ Psi - np.eye(8)).max() < 1e-12
    h, hphi = HT.pathdisc(a, b, V), HT.pathdisc(a @ Phi, b @ Phi, V); ok &= abs(h - hphi) < 1e-12
    hm = HT.pathdisc(a @ M, b @ M, V); ok &= hm <= h + 1e-12; drop.append(h - hm)
check("HC4 IDENTITY lossless re-encodings leave path discrepancy unchanged; a merge can only lower it", ok, f"largest drop under the merge {max(drop):.4f}")
json.dump(dict(dimF_HD=int(dq), dimJ=int(dJ), split_trace_tv=KT.tv(Rs, Rh), cancel_coef=float(s), honest=float(g @ Rh), reversed=float(g @ Rrev),
               fibre=[float(vals.min()), float(vals.max())], slice=[float(min(sl)), float(max(sl))]),
          open(os.path.join(HERE, "historytrace_summary.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
