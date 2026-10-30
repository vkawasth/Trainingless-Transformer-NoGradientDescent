"""check_ehrhart.py -- Gap 3: the decision on the hiding segment and the ring of contingency tables (lab paper, "Theorem D and the ring of
contingency tables"; companion "Coupling algebras", Section 7). Writes lab/ehrhart.json.

  EH1  the fibre: integer 2x2x2 tables with the pair tables of k T* are the lattice points of a segment joined by the move chi; their
       number is k l + 1 (an Ehrhart polynomial, brute force for k = 1..3); the toric ideal is principal, one quartic binomial (Macaulay2)
  EH2  the hiding set is a strict sub-segment: R sits inside the margin fibre, its endpoints are where some a_ji reaches 0 or 1 while
       every cell stays positive; its lattice-point count along the ray is not a quasi-polynomial of period <= 12 (irrational ends)
  EH3  the worst case is a tropical limit, not a section: -(1/s) log sum_T exp(-s U_a(T)) lies in [min - log(#)/s, min] and tends to the
       lattice minimum; over the full fibre the lattice minimum equals the continuous one exactly (lattice ends), over R within |c_a|/n
  EH4  over the full fibre the worst case is an integer program solved by the Groebner basis (Conti-Traverso normal form = enumeration
       = linear programme); along a line of margins it is piecewise linear and breaks only where the binding cell changes, which is
       also where the fibre length (the leading term of the table count) breaks: the walls of the chamber complex
  EH5  settling is not a wall-crossing: along the ray n = k N the chamber is fixed. But the lattice is not negligible at n*: the truth sits
       (z tau)^2 |v_T| / d lattice steps from the decision wall (about ten in the Study 11b designs), and the conditional law of the
       position has sd tau |v_T| sqrt(n*) of about five steps, so exact settling is visibly discrete
  EH6  the exact (conditional) probe is the efficient one: tau_c = 1 / (|v_T| sqrt(sum 1/p)) <= tau_m on 300 random truths and on the
       Study 11b truths (Cramer-Rao along the fibre); the conditional MLE has sd sqrt(n) -> tau_c (simulation, 400 replicates)
  EH7  the exact conditional interval for the three-way log odds ratio has conditional coverage >= 0.95 for every psi (exact sums)
"""
import itertools, json, os, subprocess, sys, tempfile
from fractions import Fraction as Fr
from math import lcm
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "study11b_settle"))
from pce import ehrhart as EH, lab as L, segment_decision as SD
import design11b as D11

rng = np.random.default_rng(20261029); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
X = EH.X

# the worked truth, exact
w, a1, a2 = Fr(2, 5), [Fr(4, 5), Fr(3, 10), Fr(7, 10)], [Fr(1, 5), Fr(7, 10), Fr(3, 20)]
def pr(a, x): return np.prod([a[i] if x[i] else 1 - a[i] for i in range(3)])
PF = {x: w * pr(a1, x) + (1 - w) * pr(a2, x) for x in X}; N = lcm(*[p.denominator for p in PF.values()])
P = np.zeros((2, 2, 2)); T1 = np.zeros((2, 2, 2), dtype=object)
for x in X: P[x] = float(PF[x]); T1[x] = int(PF[x] * N)
mu, M, mS = L.moments(P); v = SD.v_closed(M); vT = float(np.prod(v)); m0 = mS((0, 1, 2)); lo_t, hi_t = L.segment(mu, v)
Rm = tuple(sorted([vT * L.r_of_t(lo_t), vT * L.r_of_t(hi_t)]))
jlo1, jhi1 = EH.fibre_int(T1); Fm = (m0 + jlo1 / N, m0 + jhi1 / N)
OUT["truth"] = dict(N=N, m0=m0, fibre_mT=Fm, R_mT=Rm, vT=vT, fibre_steps=jhi1 - jlo1)

# EH1
def brute(T):
    """all integer tables with the pair tables of T: choose the split of each (0,1)-cell over x_2, check the other margins"""
    M01 = T.sum(axis=2); M02 = T.sum(axis=1); M12 = T.sum(axis=0); cnt = 0
    for s in itertools.product(*[range(int(M01[a, b]) + 1) for a, b in itertools.product((0, 1), repeat=2)]):
        U = np.zeros((2, 2, 2), dtype=int)
        for k, (a, b) in enumerate(itertools.product((0, 1), repeat=2)): U[a, b, 0] = s[k]; U[a, b, 1] = M01[a, b] - s[k]
        cnt += np.array_equal(U.sum(axis=1), M02) and np.array_equal(U.sum(axis=0), M12)
    return cnt
ok = True; bf = {}
for trial in range(4):
    Ts = rng.integers(0, 4, size=(2, 2, 2)); Ts[0, 0, 0] += 1; l = -np.subtract(*EH.fibre_int(Ts))
    for k in (1, 2, 3):
        c = brute(k * Ts); ok &= c == k * l + 1 == EH.count_fibre(k * Ts); bf[f"T{trial} k{k}"] = int(c)
ok &= all(EH.count_fibre(k * T1) == k * (jhi1 - jlo1) + 1 for k in range(1, 30))
m2 = """R = QQ[p000,p001,p010,p011,p100,p101,p110,p111]; S = QQ[t_0..t_11];
X = {{0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1}};
pr = {{0,1},{0,2},{1,2}};
img = apply(X, x -> product(flatten apply(3, k -> (i := pr#k#0; j := pr#k#1; t_(4*k + 2*(x#i) + x#j)))));
I = ker map(S, R, img); G = flatten entries mingens I;
print("NGENS " | toString(#G)); print("DEG " | toString(apply(G, first @@ degree))); print("GEN " | toString G);"""
fd, path = tempfile.mkstemp(suffix=".m2"); open(path, "w").write(m2); os.close(fd)
out = subprocess.run(["M2", "--script", path], capture_output=True, text=True, timeout=300).stdout; os.unlink(path)
gens = [l for l in out.splitlines() if l.startswith(("NGENS", "DEG", "GEN"))]; OUT["toric_ideal"] = gens
ok &= "NGENS 1" in out and "DEG {4}" in out
check("EH1 the fibre is a lattice segment joined by chi; #tables = k l + 1 (brute force, 12 cases); the toric ideal is one quartic binomial",
      ok, f"worked truth: N = {N}, l = {jhi1 - jlo1} steps; " + "; ".join(gens[2:]))

# EH2
lo_e, hi_e = EH.eps_range(P); fib_cont = (m0 + lo_e, m0 + hi_e)
ends = [L.explanation(mu, v, t) for t in (lo_t, hi_t)]
wall = all(min(np.min(np.abs(e[1])), np.min(np.abs(1 - e[1])), np.min(np.abs(e[2])), np.min(np.abs(1 - e[2]))) < 1e-9 for e in ends)
pos = all(np.min(L.law(*e)) > 1e-4 for e in ends)
inside = fib_cont[0] < Rm[0] < Rm[1] < fib_cont[1] and np.allclose(fib_cont, Fm)
cR = [EH.lattice_in(k * N, m0, Rm) for k in range(1, 161)]; cF = [EH.count_fibre(k * T1) for k in range(1, 161)]
def quasi(c, p): return all(c[k + 2 * p] - 2 * c[k + p] + c[k] == 0 for k in range(len(c) - 2 * p))
noq = not any(quasi(cR, p) for p in range(1, 13)) and quasi(cF, 1)
OUT["R_counts"] = cR[:12]
check("EH2 R is a strict sub-segment of the margin fibre, ended by positivity walls of the explanation (a_ji in {0,1}) with every cell positive;"
      " its lattice count is no quasi-polynomial of period <= 12, the fibre's is a polynomial", inside and wall and pos and noq,
      f"fibre m_T in [{fib_cont[0]:.4f}, {fib_cont[1]:.4f}], R in [{Rm[0]:.4f}, {Rm[1]:.4f}] ({(Rm[1] - Rm[0]) / (fib_cont[1] - fib_cont[0]):.0%})")

# EH3
acts = {"A any harm": lambda x: 1.0 - float(sum(x) > 0), "B two harmed": lambda x: 0.5 - float(sum(x) >= 2), "C x0 harmed": lambda x: 1.0 - x[0]}
ok = True; trop = {}
U0 = {k: sum(P[x] * u(x) for x in X) for k, u in acts.items()}
for name, u in acts.items():
    c = EH.walsh(u); poly = SD.U_poly(mu, v, u, range(3)); ok &= abs((poly[1] if len(poly) > 1 else 0.0) - vT * c) < 1e-12
    # exact over the full fibre: lattice ends = continuous ends
    cF_ = Fr(int(round(c)))
    for k in (1, 3, 7):
        n = k * N; jl, jh = EH.fibre_int(k * T1)
        Uex = lambda j: sum(PF[x] * Fr(u(x)).limit_denominator(4) for x in X) + cF_ * Fr(j, n)
        e_lo = max(-PF[x] for x in X if EH.CHI[x] > 0); e_hi = min(PF[x] for x in X if EH.CHI[x] < 0)   # continuous ends (exact)
        ok &= e_lo * n == jl and e_hi * n == jh and min(Uex(j) for j in (jl, jh)) == min(Uex(e * n) for e in (e_lo, e_hi))
    for k in (1, 4, 16):
        n = k * N; jR = np.arange(int(np.ceil(n * (Rm[0] - m0))), int(np.floor(n * (Rm[1] - m0))) + 1)
        vals = U0[name] + c * jR / n; WR = min(U0[name] + c * (Rm[0] - m0), U0[name] + c * (Rm[1] - m0))
        ok &= vals.min() - WR <= abs(c) / n + 1e-12 and vals.min() >= WR - 1e-12
        for s in (1e2 * n, 1e3 * n, 1e5 * n):
            F = EH.tropical(vals, s); ok &= vals.min() - np.log(len(vals)) / s - 1e-12 <= F <= vals.min() + 1e-12
        trop[f"{name}, n={n}"] = dict(lattice_min=float(vals.min()), W_R=float(WR), tables=len(vals), F_1e3n=EH.tropical(vals, 1e3 * n))
OUT["tropical"] = trop
check("EH3 the worst case is the tropical limit of the weighted count over the lattice points (bounds [min - log#/s, min]); exact over the"
      " margin fibre at every k, within |c_a|/n over R; the slope in m_T is the Walsh coefficient c_a (beta_a = v_T c_a)", ok)

# EH4
ok = True
for name, u in acts.items():
    c = EH.walsh(u)
    for k in (1, 2, 5):
        T = k * T1; NF, steps = EH.conti_traverso(T, c); n = k * N
        jl, jh = EH.fibre_int(T); jmin = jl if c > 0 else (jh if c < 0 else 0)          # a flat act: every table minimises
        ok &= np.array_equal(np.array(NF, dtype=object), np.array(T + jmin * EH.CHI.astype(int), dtype=object))
        lpv, _ = EH.worst_lp(T.astype(float) / n, u); ok &= abs(lpv - (U0[name] + c * jmin / n)) < 1e-9
kinks = 0; tries = 0; agree = True
while kinks < 3 and tries < 50:
    tries += 1; Dd = rng.normal(size=(2, 2, 2)); Dd -= Dd.mean(); ts = np.linspace(-1, 1, 801) * 0.6 * P.min() / np.abs(Dd).max()
    Wv, bind, length = [], [], []
    for t in ts:
        Pt = P + t * Dd; lv, z = EH.worst_lp(Pt, acts["A any harm"]); Wv.append(lv); bind.append(tuple(z)); a_, b_ = EH.eps_range(Pt); length.append(b_ - a_)
    Wv, length = np.array(Wv), np.array(length); d2 = np.abs(Wv[2:] - 2 * Wv[1:-1] + Wv[:-2]); d2L = np.abs(length[2:] - 2 * length[1:-1] + length[:-2])
    for i in range(1, len(ts) - 1):
        same = bind[i - 1] == bind[i] == bind[i + 1]
        if same and d2[i - 1] > 1e-9: agree = False
        if not same and d2[i - 1] > 1e-9:
            kinks += 1; agree &= d2L[i - 1] > 1e-9 or d2L[max(0, i - 2)] > 1e-9 or d2L[min(len(d2L) - 1, i)] > 1e-9
ok &= agree and kinks >= 1
check("EH4 over the margin fibre the worst case is an integer programme solved by the Groebner binomial (Conti-Traverso = enumeration = LP);"
      " along lines of margins it is piecewise linear, breaking only where the binding cell changes, where the fibre length breaks too",
      ok, f"{kinks} kinks found on {tries} lines, each a break of the fibre length")

# EH5
ok = True; s5 = {}
for dn in ("D2", "D5"):
    sm = D11.summary(dn); Pd, mud, vd, taud, r0d, Rd = D11.truth(sm["truth"]); vTd = float(np.prod(vd)); md = r0d * vTd
    tc = EH.tau_c(Pd, vTd); rad = sm["radius"]; ns = (D11.Z * tc / rad) ** 2
    cnts = []
    for f in (0.25, 1, 4, 16, 64):
        n = int(f * ns); h = D11.Z * tc / np.sqrt(n) * abs(vTd); cnts.append(EH.lattice_in(n, md, (md - h, md + h)) / np.sqrt(n))
    ok &= abs(cnts[-1] - 2 * D11.Z * tc * abs(vTd)) < 0.05 * cnts[-1]
    gap = ns * rad * abs(vTd); ok &= abs(gap - (D11.Z * tc) ** 2 * abs(vTd) / rad) < 1e-9 and gap < 30
    s5[dn] = dict(nstar_c=ns, tables_per_sqrt_n=[float(c) for c in cnts], steps_truth_to_wall_at_nstar=float(gap), sd_of_j_at_nstar=float(tc * abs(vTd) * np.sqrt(ns)))
jl1, jh1 = EH.fibre_int(T1); ok &= all(EH.fibre_int(k * T1) == (k * jl1, k * jh1) for k in range(1, 50))
OUT["settling_lattice"] = s5
check("EH5 settling is not a wall-crossing: along n = k N the fibre's binding cells never change; but the lattice is not negligible at n*:"
      " the truth is (z tau)^2 |v_T| / d lattice steps from the decision wall, and the post-probe interval holds ~ 2 z tau |v_T| sqrt(n) tables",
      ok, "; ".join(f"{k}: n*_c = {v_['nstar_c']:.0f}, {v_['steps_truth_to_wall_at_nstar']:.1f} steps to the wall, sd of j {v_['sd_of_j_at_nstar']:.1f}" for k, v_ in s5.items()))

# EH6
ok = True; worst_ratio = 0.0
for _ in range(300):
    ww = rng.uniform(0.15, 0.85); b1 = rng.uniform(0.05, 0.95, 3); b2 = rng.uniform(0.05, 0.95, 3); Pr = L.law(ww, b1, b2)
    _, Mr, _ = L.moments(Pr); vr = SD.v_closed(Mr); vTr = float(np.prod(vr))
    if abs(vTr) < 1e-4: continue
    tc, tmr = EH.tau_c(Pr, vTr), EH.tau_m(Pr, vTr); ok &= tc <= tmr * (1 + 1e-10); worst_ratio = max(worst_ratio, tc / tmr)
ratios = {}
for tn in "GH":
    Pd, mud, vd, taud, r0d, Rd = D11.truth(tn); vTd = float(np.prod(vd)); ratios[tn] = (EH.tau_c(Pd, vTd), EH.tau_m(Pd, vTd)); ok &= ratios[tn][0] < ratios[tn][1]
# simulation: the conditional MLE of psi mapped to r has sd * sqrt(n) -> tau_c
Pd, mud, vd, taud, r0d, Rd = D11.truth("G"); vTd = float(np.prod(vd)); md = r0d * vTd; n = 4000; est = []
for _ in range(400):
    T = rng.multinomial(n, Pd.ravel()).reshape(2, 2, 2)
    if T.min() == 0: continue
    js, lw = EH.cond_logw(T); a, b = -30.0, 30.0
    for _ in range(100):
        m = 0.5 * (a + b); pm = np.exp(lw + m * js - np.max(lw + m * js)); pm /= pm.sum()
        if (pm * js).sum() < 0: a = m
        else: b = m
    est.append((md + EH.eps_of_psi(Pd, 0.5 * (a + b))) / vTd)
sd_sim = float(np.std(est) * np.sqrt(n)); ok &= abs(sd_sim / ratios["G"][0] - 1) < 0.12
OUT["efficiency"] = dict(tau_c_tau_m={k: list(v_) for k, v_ in ratios.items()}, max_ratio_random=worst_ratio, sim_sd_G=sd_sim)
check("EH6 the conditional (exact-test) probe is the efficient one: tau_c <= tau_m (300 random truths, G, H); its MLE has sd sqrt(n) ~ tau_c",
      ok, f"G: tau_c {ratios['G'][0]:.3f} vs tau_m {ratios['G'][1]:.3f} (simulated {sd_sim:.3f}); H: {ratios['H'][0]:.3f} vs {ratios['H'][1]:.3f}")

# EH7
ok = True; covs = []
for _ in range(3):
    T = rng.multinomial(40, P.ravel()).reshape(2, 2, 2)
    js, lw = EH.cond_logw(T); cis = {j: EH.exact_ci(T, 0.05, j) for j in js}
    for psi in np.linspace(-4, 4, 41):
        _, pm = EH.cond_pmf(T, psi); cov = sum(p for j, p in zip(js, pm) if cis[j][0] <= psi <= cis[j][1]); covs.append(cov)
ok &= min(covs) >= 0.95 - 1e-9 and max(covs) < 1 - 1e-6
OUT["exact_coverage"] = dict(min=min(covs), mean=float(np.mean(covs)))
check("EH7 the exact conditional interval for the three-way log odds ratio covers with conditional probability >= 0.95 at every psi",
      ok, f"min {min(covs):.4f}, mean {np.mean(covs):.4f} (conservative, as exact intervals are)")

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "ehrhart.json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, Fr)) else int(o) if isinstance(o, np.integer) else str(o))
print(f"{sum(results)}/{len(results)} passed")
