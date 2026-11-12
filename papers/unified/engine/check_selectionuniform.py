"""Checks for the uniform selection theorem (Lemma lem:lexunique, Theorem thm:selection, Corollary cor:selectlaw), module pce/selrate.py.
Configurations are random positively spanning sets of m = 6 normals in R^2 and R^3, some with duplicated normals (exact ties).
  SU1 IDENTITY  Lemma lem:lexunique: the recursion's output carries the level certificate (increasing levels, each level with a strictly
                positive relation modulo the earlier ones), and its sorted value vector beats, in the leximin order, every one of 400 random
                and 400 nearby points (150 configurations, a third with duplicated normals)
  SU2 IDENTITY  (b) uniform rate: L |u_L - u_lex| <= K(V, B) with the explicit constant of the proof, at points 1e-1 ... 1e-6 from bisected
                walls, for L = 20, 200, 2000, equal weights (B = 0) and weights in [1/2, 2] (B = log 2)
  SU3 IDENTITY  (c) on each chamber u_lex = A_P lam exactly; along lines the slope never exceeds max ||A_P||, also across bisected walls
                (no jumps: walls are kinks). A first version compared the two bracketing points of each bisected wall (5e-11 apart) with slack
                1e-9 and failed: at a wall two levels meet, the computed point is certified only to the residual tolerance 1e-7, and the
                differences seen were 1e-9 to 2.4e-7. The test now compares points 1e-4 outside the wall, where certificates are sharp, and
                reports the at-wall difference; the claim is unchanged.
  SU4 IDENTITY  (d) inside a chamber L (u_L - u_lex) = c_P(b) + O(exp(-L gamma)), gamma the least gap between levels: the ratio
                E(L gamma = 20) / E(L gamma = 10) is exp(-10) in the median; the column-binding fibre gives c = ln2/2 (Example ex:colbind);
                the Study 26 coupling gives c = 0, which is why its error fell faster than 1/L (XE9)
  SU5 IDENTITY  Corollary cor:selectlaw: on 2 x 3 coupling fibres the total variation between the toric law at L and the leximin law is at
                most (1/2) sum_k |v_k| K / L
  SU6 IDENTITY  the 1-D wall: l = (u, -u, -u + eps); u_L = -log(1 + e^{-L eps}) / (2L) exactly (solver vs formula); sup over eps of
                L |u_L - u_lex| is (log 2)/2, attained at the wall, and the coefficient is 0 on both sides of it
  SU7 IDENTITY  solver validation: the level-block soft-min agrees with a 60-digit Newton solution (mpmath) to 1e-9 relative to 1/L on
                20 cases with L times the least gap up to 60. For the record (not part of the pass condition): a plain double-precision
                Newton started at the leximin point agreed in those 20 cases too, but on the column-binding fibre at L = 800 it returns the
                leximin point itself (the upper level lies below rounding), which is why the block solver is used
"""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
import mpmath as mp
from pce import selrate as S, selectorlink as SLK
results = []; rec = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(2611)

def config(n, m=6, dup=False):
    while True:
        V = rng.normal(size=(m, n))
        if dup: V[m - 1] = V[0]
        if S.positive_relation(V) is not None and np.linalg.matrix_rank(V) == n: return V
def sortv(x): return np.sort(x)
def lexless(a, b, tol=1e-9):
    """True if sorted vector a is strictly below b in the leximin order"""
    for x, y in zip(a, b):
        if x < y - tol: return True
        if x > y + tol: return False
    return False

# SU1
ok = True; nconf = 0
for i in range(150):
    n = 2 + i % 2; dup = i % 3 == 0; V = config(n, dup=dup); lam = rng.normal(size=6)
    if dup: lam[5] = lam[0]
    u, lv = S.lex_levels(V, lam); parts = [I for _, I in lv]; t = np.array([x for x, _ in lv])
    ok &= S.valid_levels(V, lam, u, parts, t); best = sortv(V @ u - lam)
    for _ in range(400):
        w = u + rng.normal(size=n) * rng.choice([1e-3, 1e-1, 1.0])
        ok &= lexless(sortv(V @ w - lam), best) or np.linalg.norm(w - u) < 1e-9
    for _ in range(400):
        w = rng.normal(size=n) * 3
        ok &= lexless(sortv(V @ w - lam), best)
    nconf += 1
check("SU1 IDENTITY the recursion's point is certified and is the unique leximin maximiser of the sorted values (ties included)", ok,
      f"{nconf} configurations, 50 with duplicated normals, 800 competitors each")

# SU2, SU3
def walls_on_line(V, l0, d, ts):
    R = [S.lex_levels(V, l0 + t * d) for t in ts]; P = [tuple(I for _, I in r[1]) for r in R]; out = []
    for k in range(1, len(ts)):
        if P[k] != P[k - 1]:
            a, b = ts[k - 1], ts[k]
            for _ in range(30):
                mid = (a + b) / 2
                if tuple(I for _, I in S.lex_levels(V, l0 + mid * d)[1]) == P[k - 1]: a = mid
                else: b = mid
            out.append((a, b))
    return R, out
ratio = 0.0; nw = 0; atwall = 0.0; pl = 0.0; slope_ok = True; jump_ok = True; worst = 0.0; Ks = []
for i in range(10):
    n = 2 if i < 6 else 3; V = config(n); K0 = S.K_bound(V, 0.0); K1 = S.K_bound(V, math.log(2)); Ks.append((K0, K1))
    Amax = 0.0
    for _ in range(3):
        l0 = rng.normal(size=6); d = rng.normal(size=6); d /= np.linalg.norm(d); ts = np.linspace(-2, 2, 81)
        R, W = walls_on_line(V, l0, d, ts)
        for (u, lv), t in zip(R, ts):
            A = S.chamber_map(V, [I for _, I in lv]); Amax = max(Amax, np.linalg.norm(A, 2)); pl = max(pl, np.linalg.norm(A @ (l0 + t * d) - u))
        U = np.array([r[0] for r in R]); sl = np.linalg.norm(np.diff(U, axis=0), axis=1) / (ts[1] - ts[0])
        slope_ok &= bool(np.all(sl <= Amax * (1 + 1e-6)))
        for a, b in W[:3]:
            nw += 1; ua, ub = S.lex_levels(V, l0 + a * d)[0], S.lex_levels(V, l0 + b * d)[0]; atwall = max(atwall, np.linalg.norm(ua - ub))
            s1, s2 = a - 1e-4, b + 1e-4                     # just outside the band where the certificate is only good to 1e-7
            u1, u2 = S.lex_levels(V, l0 + s1 * d)[0], S.lex_levels(V, l0 + s2 * d)[0]
            jump_ok &= np.linalg.norm(u2 - u1) <= Amax * (s2 - s1) * (1 + 1e-6) + 1e-10
            for eps in (1e-1, 1e-2, 1e-4, 1e-6):
                for side in (a - eps, b + eps):
                    lam = l0 + side * d; ul = S.lex_levels(V, lam)[0]
                    for L in (20, 200, 2000):
                        e0 = L * np.linalg.norm(S.softmin(V, lam, L) - ul); ratio = max(ratio, e0 / K0)
                        a_w = np.exp(rng.uniform(-math.log(2), math.log(2), 6))
                        e1 = L * np.linalg.norm(S.softmin(V, lam, L, a=a_w) - ul); ratio = max(ratio, e1 / K1); worst = max(worst, e0, e1)
rec["SU2"] = dict(walls=nw, max_L_err=worst, max_ratio_to_K=ratio, K=[list(map(float, k)) for k in Ks])
check("SU2 IDENTITY uniform rate: L|u_L - u_lex| <= K(V,B) at every distance from the walls, for equal and bounded weights", ratio <= 1 and nw > 0,
      f"{nw} walls; max L|err| = {worst:.2f}; max ratio to the explicit constant = {ratio:.3f}; K(V,0) from {min(k[0] for k in Ks):.0f} to {max(k[0] for k in Ks):.0f}")
check("SU3 IDENTITY u_lex is piecewise linear on chambers, Lipschitz with constant max ||A_P||, continuous across walls", pl < 1e-10 and slope_ok and jump_ok,
      f"max |A_P lam - u_lex| = {pl:.1e}; slopes within max ||A_P|| on 30 lines; {nw} bisected walls, increments within the bound 1e-4 either side; at the wall itself |u(a) - u(b)| <= {atwall:.1e} with b - a = 5e-11, the certificate tolerance")

# SU4
R = []
while len(R) < 30:
    n = int(rng.integers(2, 4)); V = config(n); lam = rng.normal(size=6); b = rng.uniform(-1, 1, 6)
    u, lv = S.lex_levels(V, lam); ts_ = [t for t, _ in lv]
    if len(ts_) < 2: continue
    g = min(np.diff(ts_))
    if not 0.05 < g < 2: continue
    c = S.coefficient(V, [I for _, I in lv], b)
    R.append([np.linalg.norm((x / g) * (S.softmin(V, lam, x / g, a=np.exp(b)) - u) - c) for x in (5, 10, 20)])
R = np.array(R); med = float(np.median(R[:, 2] / R[:, 1]) * math.exp(10)); pref = float(np.max(R * np.exp([5, 10, 20])))
Vc, lc = SLK.cells((0.4, 0.6), (0.5, 0.3, 0.2)); idx = [0, 1, 2, 4, 5]
uc, lvc = S.lex_levels(Vc[idx], lc[idx]); cc = S.coefficient(Vc[idx], [I for _, I in lvc], np.zeros(5))
V26, l26 = SLK.cells((0.317, 0.683), (0.448, 0.351, 0.201)); u26, lv26 = S.lex_levels(V26, l26); c26 = S.coefficient(V26, [I for _, I in lv26], np.zeros(6))
e26 = [float(np.linalg.norm(L * (S.softmin(V26, l26, L) - u26))) for L in (100, 1000)]
rec["SU4"] = dict(median_ratio_times_e10=med, max_prefactor=pref, colbind_c=list(map(float, cc)), study26_c=list(map(float, c26)), study26_Lerr=e26)
check("SU4 IDENTITY inside a chamber L(u_L - u_lex) -> c_P(b) at rate exp(-L gamma); c = ln2/2 on the column-binding fibre, c = 0 on Study 26's",
      0.5 < med < 2 and abs(cc[0] - math.log(2) / 2) < 1e-12 and np.abs(c26).max() < 1e-12 and pref < 1e3,
      f"30 chambers: median E(20)/E(10) = {med:.3f} e^-10, max E e^(L gamma) = {pref:.0f}; colbind c_x = {cc[0]:.10f}; Study 26 c = 0, L|err| = {e26[0]:.3f}, {e26[1]:.1e} at L = 100, 1000")

# SU5
ok = True; mx = 0.0
for _ in range(60):
    f1 = rng.uniform(0.2, 0.8); g = rng.dirichlet([2, 2, 2]); V, lam = SLK.cells((f1, 1 - f1), tuple(g))
    if np.any(g < 0.03): continue
    K = S.K_bound(V, 0.0); u = S.lex_levels(V, lam)[0]
    for L in (30, 300):
        tv = 0.5 * np.abs(V @ (S.softmin(V, lam, L) - u)).sum(); bound = 0.5 * np.linalg.norm(V, axis=1).sum() * K / L
        ok &= tv <= bound; mx = max(mx, tv / bound)
check("SU5 IDENTITY law rate: TV(toric law at L, leximin law) <= (1/2) sum_k |v_k| K / L on 2 x 3 coupling fibres", ok, f"max TV / bound = {mx:.3f}")

# SU6
V1 = np.array([[1.0], [-1.0], [-1.0]]); ok = True; sup = 0.0; cs = []
for L in (10.0, 100.0, 1000.0):
    for eps in np.r_[-np.logspace(-4, 0, 9), 0.0, np.logspace(-4, 0, 9)]:
        lam = np.array([0.0, 0.0, -eps]); uL = S.softmin(V1, lam, L)[0]; ex = -math.log1p(math.exp(-L * eps)) / (2 * L) if eps >= 0 else eps / 2 - math.log1p(math.exp(L * eps)) / (2 * L)
        ok &= abs(uL - ex) < 1e-12; ul = S.lex_levels(V1, lam)[0][0]; sup = max(sup, L * abs(uL - ul))
for eps in (-0.3, 0.3):
    u, lv = S.lex_levels(V1, np.array([0.0, 0.0, -eps])); cs.append(float(S.coefficient(V1, [I for _, I in lv], np.zeros(3))[0]))
u0, lv0 = S.lex_levels(V1, np.zeros(3)); c0 = float(S.coefficient(V1, [I for _, I in lv0], np.zeros(3))[0])
check("SU6 IDENTITY the 1-D wall: exact formula, sup L|err| = (log 2)/2 at the wall, coefficient 0 on both sides", ok and abs(sup - math.log(2) / 2) < 1e-9 and max(map(abs, cs)) < 1e-12 and abs(c0 + math.log(2) / 2) < 1e-12,
      f"sup = {sup:.6f}; coefficient {cs[0]:.0e} | {c0:.4f} (wall) | {cs[1]:.0e}")

# SU7
mp.mp.dps = 60
def mp_softmin(V, lam, L, u0):
    u = mp.matrix([mp.mpf(x) for x in u0]); n = V.shape[1]
    for _ in range(200):
        z = [-L * (sum(mp.mpf(V[k, j]) * u[j] for j in range(n)) - mp.mpf(lam[k])) for k in range(len(lam))]; mx = max(z)
        w = [mp.e ** (zz - mx) for zz in z]; s = sum(w); w = [x / s for x in w]
        mu = [sum(w[k] * V[k, j] for k in range(len(w))) for j in range(n)]
        H = mp.matrix(n, n)
        for a in range(n):
            for c in range(n): H[a, c] = L * sum(w[k] * (V[k, a] - mu[a]) * (V[k, c] - mu[c]) for k in range(len(w)))
        d = mp.lu_solve(H, mp.matrix(mu))
        u = u + d
        if mp.norm(d) < mp.mpf(10) ** -50: break
    return np.array([float(x) for x in u])
ok = True; mxd = 0.0; stall = 0; n7 = 0
while n7 < 20:
    n = int(rng.integers(2, 4)); V = config(n); lam = rng.normal(size=6); u, lv = S.lex_levels(V, lam); ts_ = [t for t, _ in lv]
    if len(ts_) < 2: continue
    g = min(np.diff(ts_)); L = float(rng.uniform(5, 60)) / g
    if L > 3000: continue
    ref = mp_softmin(V, lam, L, u + S.coefficient(V, [I for _, I in lv], np.zeros(6)) / L)
    dd = L * np.linalg.norm(S.softmin(V, lam, L) - ref); mxd = max(mxd, dd); ok &= dd < 1e-9
    stall += L * np.linalg.norm(SLK.softmin(V, lam, L, u0=u) - ref) > 1e-6; n7 += 1
Vc5, lc5 = Vc[idx], lc[idx]; plain = 800 * (SLK.softmin(Vc5, lc5, 800, u0=uc) - uc); block = 800 * (S.softmin(Vc5, lc5, 800) - uc)
check("SU7 IDENTITY the level-block soft-min agrees with a 60-digit Newton solution", ok,
      f"20 cases, max L|diff| = {mxd:.1e}; plain double Newton off by more than 1e-6/L in {stall}; column-binding fibre at L = 800: L(u_L - u_lex) plain {plain[0]:.1e}, block {block[0]:.6f} (ln2/2 = 0.346574)")
rec["passed"] = f"{sum(results)}/{len(results)}"
os.makedirs(os.path.join(HERE, "selection"), exist_ok=True)
json.dump(rec, open(os.path.join(HERE, "selection", "results_selectionuniform.json"), "w"), indent=1)
print(f"{sum(results)}/{len(results)} passed")
