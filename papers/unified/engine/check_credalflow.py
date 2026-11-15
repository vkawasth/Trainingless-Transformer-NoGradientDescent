"""Checks for credal sets under evidence flows (section sec:credalflow: lem:hilbertflow, lem:kahlerclass, thm:crossing, thm:viability,
prop:conserved). New script; it modifies nothing; uses pce/credalflow.py, pce/fsgram.py.
  CR1 IDENTITY  Hilbert metric: the tilt is an isometry (exact); restriction to a subset of states never increases it; a positive
                stochastic kernel contracts it by at least tanh(Delta / 4) (Birkhoff-Hopf), Delta the projective diameter of its rows
  CR2 IDENTITY  Kahler class: on random nondegenerate 2 x 3 margins, a change of the facet constants lambda moves the class (lambda modulo
                translations <n_c, u>) iff some circuit pairing <a, delta lambda> is non-zero; translations of the base point move nothing;
                margin changes move the class with rank reported
  CR3 IDENTITY  Descartes for exponential sums: along t in [-40, 40], <c, p_t> changes sign at most as often as the level coefficients
                C_s(p) do (5000 random cases with ties in a)
  CR4 IDENTITY  lexicographic limit: for large t the sign of <c, p_t> is the sign of the first non-zero C_s(p) (2000 cases, with ties
                and zero coefficients forced; exact rationals, 250-digit evaluation). First run: 112 violations, all in forced cases, because the float
                construction left a top coefficient of order 1e-17, which e^(60 gap) amplified; with rationals at 80 digits 3 remained (rounding of order 1e-80 times e^120 against a second term of order e^-120); now 250 digits
  CR5 IDENTITY  credal polytopes: the bracket of <c, p> over K_t is attained at tilted vertices (random interior points never leave it);
                when the top level is uniformly positive, the decision is determinate positive for every t beyond the explicit bound
                log+(Nbar / kappa) / Delta; the decision status changes at most 2 V (r - 1) times
  CR6 IDENTITY  adversarial control through an observation map R (likelihoods constant on cells): with directions b in d + E, (a) if
                every admissible direction has the same top cell and the conserved conditional bracket there is positive, random
                time-varying controls cannot delay determinacy beyond the uniform bound; (b) if some direction's top cell has a
                straddling conditional bracket, the constant control keeps the decision indeterminate at t = 50 and 200; (c) if K holds
                two laws with equal R-margins whose cell conditionals straddle on every cell, no direction and no time identifies it
  CR7 IDENTITY  conservation under any evidence through R: the cell conditionals of every member are unchanged; all Hilbert distances
                are unchanged; the number of vertices of K (3 states) is unchanged; the R-margins move
  CR8 IDENTITY  the long-run bracket equals the conserved conditional bracket on the top cell (exact at the vertices; t = 80)
  CR9 RECORD    dilation as a revealed conserved spread: a fair coin X and a signal Y with unknown correlation in [-0.8, 0.8]; the bracket
                of P(X = 1) - 1/2 is {0} before evidence and tends to [-0.4, 0.4] as evidence for Y = 1 accumulates
  CR10 IDENTITY valuation limit: with p_i(0) ~ e^(-N lambda_i) and a = N alpha, the time after which K_t is determinate positive tends to
                the tropical crossing time (the last switch of the dominant state to a positive payoff, over all vertices) at rate 1/N
"""
import os, sys, math, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import credalflow as CF
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(20261119)
def rlaw(m, conc=1.0): return rng.dirichlet(np.ones(m) * conc)
# CR1
iso = 0.0; res_bad = 0; bir_bad = 0
for _ in range(2000):
    m = int(rng.integers(3, 9)); p, q = rlaw(m), rlaw(m); a = rng.normal(size=m) * 2; t = float(rng.uniform(-5, 5))
    iso = max(iso, abs(CF.hilbert(CF.tilt(p, a, t), CF.tilt(q, a, t)) - CF.hilbert(p, q)))
    S = sorted(rng.choice(m, size=int(rng.integers(2, m + 1)), replace=False)); ps, qs = p[S] / p[S].sum(), q[S] / q[S].sum()
    res_bad += CF.hilbert(ps, qs) > CF.hilbert(p, q) + 1e-12
    Kk = rng.dirichlet(np.ones(m), size=m)                                  # rows: positive stochastic kernel
    D = max(CF.hilbert(Kk[i], Kk[j]) for i in range(m) for j in range(m))
    bir_bad += CF.hilbert(p @ Kk, q @ Kk) > math.tanh(D / 4) * CF.hilbert(p, q) + 1e-12
check("CR1 IDENTITY tilts are Hilbert isometries; restriction never expands; kernels contract by tanh(Delta/4)", iso < 1e-9 and res_bad == 0 and bir_bad == 0,
      f"2000 cases; max isometry error {iso:.1e}; restriction violations {res_bad}; Birkhoff violations {bir_bad}")
# CR2: Kahler class of the 2 x 3 fibre
from pce import fsgram as FS
bad = 0; ranks = []
for _ in range(300):
    f = rng.dirichlet([1, 1]); g = rng.dirichlet([1, 1, 1])
    if min(f.min(), g.min()) < 1e-3: continue
    fac = FS.cpl23_facets(f, g); Nn = np.array([FS.NORMALS[c] for c in fac], float)        # rows n_c
    circ = np.linalg.svd(Nn.T)[2][2:].T if len(fac) > 2 else np.zeros((len(fac), 0))     # basis of {a : sum a_c n_c = 0}
    dl = rng.normal(size=len(fac)); P_im = Nn @ np.linalg.pinv(Nn)                          # projector onto translations
    moved = np.linalg.norm(dl - P_im @ dl) > 1e-10; pair = np.abs(circ.T @ dl).max() > 1e-10 if circ.size else False
    tr = Nn @ rng.normal(size=2); bad += (moved != pair) or (np.abs(circ.T @ tr).max() > 1e-10 if circ.size else False)
    # margin changes: lambda_c = x0_c(f, g) for the facet cells; finite differences of the class (circuit pairings)
    J = []
    for k in range(3):
        e = np.zeros(3); e[k] = 1e-6; f2 = np.array([f[0] + e[0], f[1] - e[0]]); g2 = np.array([g[0] + e[1], g[1] + e[2], g[2] - e[1] - e[2]])
        x1 = FS._x0(f, g); x2 = FS._x0(f2, g2); d = np.array([x2[c] - x1[c] for c in fac]) / 1e-6; J.append(circ.T @ d)
    ranks.append(np.linalg.matrix_rank(np.array(J).T, tol=1e-4) if circ.size else 0)
from collections import Counter
check("CR2 IDENTITY a shift of the facet constants moves the Kahler class iff a circuit pairing changes; translations never do", bad == 0,
      f"{len(ranks)} margins; mismatches {bad}; rank of (margins -> class) by chamber: {dict(Counter(ranks))}")
# CR3 Descartes
bad = 0; tight = 0; tot = 0; ts = np.linspace(-40, 40, 8001)
for _ in range(5000):
    m = int(rng.integers(2, 8)); p = rlaw(m); c = rng.normal(size=m); a = rng.integers(-3, 4, size=m).astype(float)
    bound = CF.sign_changes(CF.level_coeffs(p, c, a))
    vals = [float(np.dot(c, CF.tilt(p, a, t))) for t in ts[::20]]; got = CF.sign_changes(vals); tot += 1
    bad += got > bound; tight += got == bound
check("CR3 IDENTITY Descartes: <c, p_t> changes sign at most as often as its level coefficients", bad == 0, f"{tot} cases; violations {bad}; bound attained in {tight / tot:.2f}")
# CR4 lexicographic limit (exact rationals, 250-digit evaluation: a vanishing top coefficient must vanish exactly)
from fractions import Fraction as Fr
import mpmath as mp
mp.mp.dps = 250
bad = 0; nforced = 0
for i in range(2000):
    m = int(rng.integers(2, 8)); k = rng.integers(1, 50, size=m); p = [Fr(int(x), int(k.sum())) for x in k]
    c = [Fr(int(x), 7) for x in rng.integers(-20, 21, size=m)]; a = rng.integers(-2, 3, size=m).astype(float)
    if i % 3 == 0:
        S = CF.levels(a)[0][1]
        if len(S) >= 2: c[S[-1]] = -sum(c[j] * p[j] for j in S[:-1]) / p[S[-1]]; nforced += 1
    C = [sum(c[j] * p[j] for j in S_) for _, S_ in CF.levels(a)]; nz = [x for x in C if x != 0]; ls = (1 if nz[0] > 0 else -1) if nz else 0
    t = mp.mpf(60); g = mp.fsum(mp.mpf(c[j].numerator) / c[j].denominator * mp.mpf(p[j].numerator) / p[j].denominator * mp.e ** (t * a[j]) for j in range(m))
    bad += ls != 0 and mp.sign(g) != ls
check("CR4 IDENTITY the eventual sign of <c, p_t> is the sign of the first non-zero level coefficient", bad == 0, f"2000 cases ({nforced} with a vanishing top coefficient); violations {bad}")
# CR5 credal polytopes
bad_ext = bad_bound = bad_status = 0; nb = 0; ratios = []
for _ in range(400):
    m = int(rng.integers(3, 7)); nv = int(rng.integers(2, 6)); V = [rlaw(m, 2.0) for _ in range(nv)]; a = rng.integers(-3, 4, size=m).astype(float)
    if len(CF.levels(a)) < 2: continue
    top = CF.levels(a)[0][1]; c = rng.normal(size=m); c[top] = np.abs(c[top]) + 0.2       # top level positive
    t = float(rng.uniform(0, 6)); lo, hi = CF.bracket(V, c, a, t)
    for _ in range(20):
        w = rng.dirichlet(np.ones(nv)); x = sum(wi * v for wi, v in zip(w, V)); gx = float(np.dot(c, CF.tilt(x, a, t)))
        bad_ext += gx < lo - 1e-12 or gx > hi + 1e-12
    B = CF.crossing_bound(V, c, a)
    if B is not None:
        nb += 1; T = B + 10; tl = CF.last_nonpositive(V, c, a, T)
        bad_bound += tl is not None and tl > B + 1e-9
        if tl is not None and B > 0: ratios.append(tl / B)
    tsg = np.linspace(-30, 30, 3001); st = []
    for tt in tsg:
        l_, h_ = CF.bracket(V, c, a, tt); st.append(1 if l_ > 0 else (-1 if h_ < 0 else 0))
    changes = sum(1 for i in range(1, len(st)) if st[i] != st[i - 1]); r = len(CF.levels(a)); bad_status += changes > 2 * nv * (r - 1)
check("CR5 IDENTITY brackets sit at tilted vertices; beyond the explicit bound every member is positive; status changes <= 2 V (r-1)", bad_ext == 0 and bad_bound == 0 and bad_status == 0 and nb > 100,
      f"{nb} bounded cases; extreme-point violations {bad_ext}; bound violations {bad_bound}; status violations {bad_status}; actual/bound median {np.median(ratios):.2f}")
# CR6 adversarial control through an observation map
def lift(b, R): return np.array([b[R[i]] for i in range(len(R))], float)
R = [0, 0, 1, 1, 2, 2]; m = 6
def cond_bracket(V, c, o):
    vals = [float(np.dot(c[[i for i in range(m) if R[i] == o]], CF.cell_conditionals(v, R)[o])) for v in V]; return min(vals), max(vals)
a_ok = b_ok = c_ok = 0; trials = 0
for _ in range(120):
    V = [rlaw(m, 2.0) for _ in range(4)]; c = rng.normal(size=m)
    # (a) forced: top cell 0 for every admissible direction; make the conditional bracket on cell 0 positive
    c[[0, 1]] = np.abs(c[[0, 1]]) + 0.1; d = np.array([2.0, 0.5, 0.0]); eps = 0.4
    corners = [d + np.array(s) * eps for s in itertools.product([-1, 1], repeat=3)]
    Delta = min(b[0] - max(b[1], b[2]) for b in corners); kappa = min(sum(c[i] * v[i] for i in (0, 1)) for v in V)
    Nbar = max(sum(max(-sum(c[i] * v[i] for i in range(m) if R[i] == o), 0) for o in (1, 2)) for v in V)
    Tb = max(0.0, math.log(Nbar / kappa)) / Delta if Nbar > 0 else 0.0
    ok = True
    for _ in range(10):                                                   # random piecewise-constant controls in d + E
        pos = np.zeros(3); t = 0.0
        for k in range(40):
            b = d + rng.uniform(-eps, eps, size=3); dt = float(rng.uniform(0.05, 0.5)); pos = pos + b * dt; t += dt
            if t > Tb + 1e-9:
                lo = min(float(np.dot(c, CF.tilt(v, lift(pos, R), 1.0))) for v in V); ok &= lo > 0
    a_ok += ok
    # (b) persistent: a direction whose top cell (cell 2) has a straddling conditional bracket
    c2 = rng.normal(size=m); V2 = [rlaw(m, 2.0) for _ in range(4)]
    cb = cond_bracket(V2, c2, 2)
    if cb[0] < 0 < cb[1]:
        b = np.array([0.0, 0.3, 1.0]); s50 = CF.bracket(V2, c2, lift(b, R), 50.0); s200 = CF.bracket(V2, c2, lift(b, R), 200.0)
        b_ok += s50[0] < 0 < s50[1] and s200[0] < 0 < s200[1]; trials += 1
    # (c) fibre impossibility: two laws with equal R-margins, cell conditionals straddling on every cell
    marg = rlaw(3); cc = np.array([1.0, -1.0] * 3)
    pA = np.concatenate([marg[o] * np.array([0.8, 0.2]) for o in range(3)]); pB = np.concatenate([marg[o] * np.array([0.2, 0.8]) for o in range(3)])
    okc = True
    for _ in range(50):
        b = rng.normal(size=3) * 5; t = float(rng.uniform(0, 100)); lo, hi = CF.bracket([pA, pB], cc, lift(b, R), t); okc &= lo < 0 < hi
    c_ok += okc
check("CR6 IDENTITY adversarial control: forced determinacy by the uniform bound; persistent ambiguity by a straddling top cell; fibre-straddle never identified",
      a_ok == 120 and b_ok == trials and trials > 10 and c_ok == 120, f"(a) {a_ok}/120; (b) {b_ok}/{trials}; (c) {c_ok}/120")
# CR7 conservation
bad = 0; moved = 0
for _ in range(500):
    V = [rlaw(m, 1.5) for _ in range(5)]; b = rng.normal(size=3) * 3; t = float(rng.uniform(0, 10)); a = lift(b, R)
    for v in V:
        cv, cw = CF.cell_conditionals(v, R), CF.cell_conditionals(CF.tilt(v, a, t), R)
        bad += max(np.abs(cv[o] - cw[o]).max() for o in cv) > 1e-12
        mv = np.array([v[[i for i in range(m) if R[i] == o]].sum() for o in range(3)]); mw = np.array([CF.tilt(v, a, t)[[i for i in range(m) if R[i] == o]].sum() for o in range(3)])
        moved += np.abs(mv - mw).max() > 1e-6
    for v, w in itertools.combinations(V, 2): bad += abs(CF.hilbert(v, w) - CF.hilbert(CF.tilt(v, a, t), CF.tilt(w, a, t))) > 1e-9
from scipy.spatial import ConvexHull
vb = 0
for _ in range(300):
    V = [rlaw(3, 1.0) for _ in range(int(rng.integers(3, 8)))]; a = rng.normal(size=3) * 3; t = float(rng.uniform(0, 3))
    h0 = len(ConvexHull(np.array(V)[:, :2]).vertices); h1 = len(ConvexHull(np.array([CF.tilt(v, a, t) for v in V])[:, :2]).vertices); vb += h0 != h1
check("CR7 IDENTITY conserved: cell conditionals, Hilbert distances, the combinatorial type of K; moved: the R-margins", bad == 0 and vb == 0 and moved > 2000,
      f"conditional/Hilbert violations {bad}; vertex-count changes {vb}/300; margins moved in {moved}/2500")
# CR8 long-run bracket = conserved conditional bracket on the top cell
bad = 0
for _ in range(300):
    V = [rlaw(m, 2.0) for _ in range(4)]; c = rng.normal(size=m); b = rng.normal(size=3)
    top = int(np.argmax(b)); cbk = cond_bracket(V, c, top); lr = CF.bracket(V, c, lift(b, R), 80.0 / max(1e-3, np.sort(b)[-1] - np.sort(b)[-2]))
    bad += abs(lr[0] - cbk[0]) > 1e-6 or abs(lr[1] - cbk[1]) > 1e-6
check("CR8 IDENTITY the long-run bracket is the conserved conditional bracket on the top observation cell", bad == 0, f"300 cases; mismatches {bad}")
# CR9 dilation record: states (x, y) in {0,1}^2 ordered 00, 01, 10, 11; observation y; fair X and Y, correlation rho in [-0.8, 0.8]
def law(rho): q = 0.25 * (1 + rho); r = 0.25 * (1 - rho); return np.array([q, r, r, q])   # P(x=y) = (1 + rho) / 2
Vd = [law(-0.8), law(0.8)]; cd = np.array([-0.5, -0.5, 0.5, 0.5]); Rd = [0, 1, 0, 1]; bd = np.array([0.0, 1.0])
rows = [(t, CF.bracket(Vd, cd, lift(bd, Rd), t)) for t in (0.0, 1.0, 3.0, 10.0, 40.0)]
ok = abs(rows[0][1][0]) < 1e-12 and abs(rows[0][1][1]) < 1e-12 and abs(rows[-1][1][0] + 0.4) < 1e-6 and abs(rows[-1][1][1] - 0.4) < 1e-6
check("CR9 RECORD dilation: the bracket of P(X=1) - 1/2 grows from {0} to the conserved conditional spread [-0.4, 0.4]", ok,
      "; ".join(f"t={t:g}: [{l:.3f}, {h:.3f}]" for t, (l, h) in rows))
# CR10 valuation limit of the crossing time
def trop_time(lams, alpha, c, T=50.0):
    """last t >= 0 at which some vertex's dominant state (argmax_i t alpha_i - lambda_i) has c_i <= 0."""
    last = 0.0
    for lam in lams:
        ts = np.linspace(0, T, 50001); dom = np.argmax(ts[:, None] * alpha[None, :] - lam[None, :], axis=1); badt = ts[c[dom] <= 0]
        if len(badt): last = max(last, float(badt.max()))
    return last
errs = []; okv = True
for _ in range(6):
    m4 = 4; alpha = rng.permutation(np.array([1.0, 0.6, 0.2, -0.4])); c = rng.normal(size=m4); c[int(np.argmax(alpha))] = abs(c[int(np.argmax(alpha))]) + 0.3
    lams = [rng.uniform(0, 1.5, size=m4) for _ in range(3)]; tt = trop_time(lams, alpha, c); row = []
    for Nn in (20, 80, 320):
        V = [np.exp(-Nn * l) / np.exp(-Nn * l).sum() for l in lams]; tl = CF.last_nonpositive(V, c, Nn * alpha, tt + 5, n=6000)
        row.append(abs((tl or 0.0) - tt))
    errs.append(row); okv &= row[-1] < row[0] or row[0] < 1e-3
errs = np.array(errs); sl = np.polyfit(np.log([20, 80, 320]), np.log(np.maximum(errs.mean(0), 1e-12)), 1)[0]
check("CR10 IDENTITY in the valuation limit the crossing time is the tropical transit time (error ~ 1/N)", okv and sl < -0.6,
      f"mean |tau_N - tau_trop| at N = 20, 80, 320: {', '.join(f'{x:.4f}' for x in errs.mean(0))}; slope {sl:.2f}")
print(f"{sum(results)}/{len(results)} passed")
