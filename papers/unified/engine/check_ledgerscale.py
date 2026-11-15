"""Study 28: scaling of the stratified ledger (sec:ledger). Pre-registered in study28_scale/PREREG_study28.md, frozen by SHA-256 in
study28_scale/FROZEN_PREREG_study28.txt before the first run. New script; it modifies nothing (it re-implements pce.stratledger.build
with timers so that LP time and certification time can be separated; the logic is identical).
  LS1 IDENTITY    every defined table certified at all four band ends, every shape and T; exact rational re-check of 300 random defined
                  tables per shape and end at T = 10000
  LS2 IDENTITY    linear control: solves per objective <= C(r + c - 2, r - 1) at T = 10000
  LS3 IDENTITY    solves <= regions used
  LS4 HYPOTHESIS  solves(T = 10000) < 5 x solves(T = 1000), every shape and end
  LS5 HYPOTHESIS  build time (all four ends) at T = 10000 < T x 4 x median direct LP time, every shape
  LS6 HYPOTHESIS  per-table membership check >= 10 times cheaper than one direct LP, every shape
  LS7 RECORD      regions against T and the tree bound; share of solves in the second half; memory ratio; log-log slope
First run (the only run): LS2 FAILED. One 2 x 8 objective (third of three) took 9 solves against C(8, 1) = 8. The objective is not
generic (tied row differences, so a dual-degenerate 2 x 2 cycle); prop:basiscount(i) assumes genericity and this check did not test
it. Threshold unchanged; diagnosis in check_ledgerscale_addendum.py (LS8, post hoc). All other checks passed.
Output: check_ledgerscale_output.txt, study28_scale/ledgerscale.json (aggregates only; synthetic data).
"""
import os, sys, math, json, time, itertools
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import stratledger as SL
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def nondeg(f, g):
    fs = {sum(c) for k in range(1, len(f)) for c in itertools.combinations(f, k)}; gs = {sum(c) for k in range(1, len(g)) for c in itertools.combinations(g, k)}
    return not (fs & gs)
def margins(rng, k, den):
    v = rng.dirichlet([2] * k) * den; x = np.maximum(1, np.floor(v)).astype(int); x[int(np.argmax(x))] += den - x.sum()
    return [Fr(int(y), den) for y in x] if x.min() >= 1 else None
def population(r, c, den, seed, n):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        f, g = margins(rng, r, den), margins(rng, c, den)
        if f and g and nondeg(f, g): out.append((f, g))
    return out
def timed_build(P, L, Bm, n, d, sense, order):
    T = Bm.shape[1]; assign = -np.ones(T, int); value = np.full(T, np.nan); solves = 0; tlp = 0.0; tcert = 0.0; growth = []
    for t in order:
        if assign[t] < 0:
            t0 = time.perf_counter(); x = SL.solve_ratio(P, Bm[:, t], n, d, sense); tlp += time.perf_counter() - t0; solves += 1
            if x is not None:
                t0 = time.perf_counter()
                for B in SL.bases_from_vertex(P, x):
                    k = L.add(B); ok, v = L.certify(k, Bm, n, d, sense); new = ok & (assign < 0); assign[new] = k; value[new] = v[new]
                    if assign[t] >= 0: break
                tcert += time.perf_counter() - t0
        growth.append(solves)
    return assign, value, solves, tlp, tcert, growth
SHAPES = [(2, n) for n in range(2, 9)] + [(3, n) for n in range(3, 8)] + [(4, n) for n in range(4, 7)]
TS = (1000, 3000, 10000); DEN = 1009
OUT = {}; ls1 = True; ls1_info = []; ls2 = True; ls2_info = []; ls3 = True; ls4 = True; ls4_bad = []; ls5 = True; ls5_bad = []; ls6 = True; ls6_bad = []
for si, (r, c) in enumerate(SHAPES):
    tag = f"{r}x{c}"; P = SL.Problem(r, c); pop = population(r, c, DEN, 20261200 + 10 * r + c, TS[-1])
    Bm_all = np.array([P.b(f, g) for f, g in pop]).T; cells = P.cells; ix = {cl: k for k, cl in enumerate(cells)}
    n0 = np.zeros(P.N); n0[ix[(0, 0)]] = 1; d0 = np.zeros(P.N)
    for cl in ((0, 0), (0, 1), (1, 0), (1, 1)): d0[ix[cl]] = 1
    Lk = np.array([1 + ((i + 2 * j) % 3) / 2 for (i, j) in cells]); n1 = n0 * Lk; d1 = d0 * Lk
    LkF = [Fr(1) + Fr((i + 2 * j) % 3, 2) for (i, j) in cells]
    ends = dict(hi0=(n0, d0, +1), lo0=(n0, d0, -1), hi1=(n1, d1, +1), lo1=(n1, d1, -1))
    endsF = dict(hi0=([Fr(int(x)) for x in n0], [Fr(int(x)) for x in d0]), lo0=([Fr(int(x)) for x in n0], [Fr(int(x)) for x in d0]),
                 hi1=([Fr(int(x)) * l for x, l in zip(n0, LkF)], [Fr(int(x)) * l for x, l in zip(d0, LkF)]))
    endsF["lo1"] = endsF["hi1"]
    defined_all = np.array([f[0] + f[1] + g[0] + g[1] > 1 for f, g in pop]); rec = dict(shape=tag, tree_bound=r ** (c - 1) * c ** (r - 1), per_T={})
    # linear control at T = 10000
    cap = math.comb(r + c - 2, r - 1); lin = []
    for o in range(3):
        nv = np.random.default_rng(7 + si * 10 + o).integers(-20, 21, size=P.N).astype(float); L = SL.Library(P)
        a, _, s, _, _, _ = timed_build(P, L, Bm_all, nv, np.zeros(P.N), +1, np.random.default_rng(1).permutation(TS[-1])); lin.append(s)
        ls1 &= bool((a >= 0).all())
    ls2 &= max(lin) <= cap; ls2_info.append(f"{tag} {max(lin)}/{cap}"); rec["linear_solves"] = lin; rec["linear_cap"] = cap
    for T in TS:
        idx = np.where(defined_all[:T])[0]; Bd = Bm_all[:, idx]; TD = len(idx); order = np.random.default_rng(1).permutation(TD)
        er = {}; tot_build = 0.0; libs = {}; asgs = {}; vals = {}
        for key, (nv, dv, sense) in ends.items():
            L = SL.Library(P); a, v, s, tlp, tcert, gr = timed_build(P, L, Bd, nv, dv, sense, order)
            used = len(set(a.tolist()) - {-1}); ok = bool((a >= 0).all()); ls1 &= ok; ls3 &= s <= used; tot_build += tlp + tcert
            half = (s - gr[TD // 2 - 1]) / s if s and TD > 1 else 0.0
            er[key] = dict(solves=s, library=len(L.bases), regions_used=used, t_lp=tlp, t_cert=tcert, all_certified=ok, second_half_share=half)
            libs[key] = L; asgs[key] = a; vals[key] = v
        rec["per_T"][T] = dict(defined=TD, ends=er, build_time=tot_build)
        if T == TS[-1]:
            # direct LP timing on 200 defined tables
            rng = np.random.default_rng(5); samp = rng.choice(TD, min(200, TD), replace=False); tl = []
            for t in samp:
                t0 = time.perf_counter(); SL.solve_ratio(P, Bd[:, t], n0, d0, +1); tl.append(time.perf_counter() - t0)
            lp_med = float(np.median(tl))
            # membership check: certify every library basis over all tables, per table and end
            tchk = []
            for key, (nv, dv, sense) in ends.items():
                t0 = time.perf_counter()
                for k in range(len(libs[key].bases)): libs[key].certify(k, Bd, nv, dv, sense)
                tchk.append((time.perf_counter() - t0) / TD)
            chk = float(np.mean(tchk))
            # exact re-check, 300 tables per end
            bad = 0
            for key, (nv, dv, sense) in ends.items():
                nF, dF = endsF[key]
                for t in rng.choice(TD, min(300, TD), replace=False):
                    okx, vx = SL.exact_check(P, libs[key], asgs[key][t], P.bF(*pop[idx[t]]), nF, dF, sense)
                    bad += (not okx) or abs(float(vx) - vals[key][t]) > 1e-9
            ls1 &= bad == 0
            mem_ledger = sum(len(libs[k].bases) * P.m * P.m * 8 for k in ends) + 4 * TD * 4; mem_bands = 8 * TD * 4
            rec.update(lp_median=lp_med, check_per_table=chk, exact_failures=bad, mem_ledger=mem_ledger, mem_bands=mem_bands,
                       direct_estimate=TD * 4 * lp_med)
            if not tot_build < TD * 4 * lp_med: ls5 = False; ls5_bad.append(tag)
            if not chk * 10 <= lp_med: ls6 = False; ls6_bad.append(tag)
    for key in ends:
        s1, s3 = rec["per_T"][TS[0]]["ends"][key]["solves"], rec["per_T"][TS[-1]]["ends"][key]["solves"]
        if not s3 < 5 * max(s1, 1): ls4 = False; ls4_bad.append(f"{tag}:{key} {s1}->{s3}")
    sl = []
    for key in ends:
        xs = [math.log(rec["per_T"][T]["defined"]) for T in TS]; ys = [math.log(max(rec["per_T"][T]["ends"][key]["solves"], 1)) for T in TS]
        sl.append(float(np.polyfit(xs, ys, 1)[0]))
    rec["loglog_slope"] = sl; OUT[tag] = rec
    e = rec["per_T"][TS[-1]]
    print(f"  {tag}: defined {e['defined']}; solves (hi0, lo0, hi1, lo1) {[e['ends'][k]['solves'] for k in ends]} at T=10000, "
          f"{[rec['per_T'][TS[0]]['ends'][k]['solves'] for k in ends]} at T=1000; tree bound {rec['tree_bound']}; build {e['build_time']:.1f}s vs direct "
          f"{rec['direct_estimate']:.1f}s; check/LP {rec['check_per_table'] / rec['lp_median']:.4f}; slopes {[round(x, 2) for x in sl]}", flush=True)
check("LS1 IDENTITY every defined table certified at every band end; exact re-check of 300 tables per shape and end", ls1,
      "exact failures " + str({k: OUT[k]["exact_failures"] for k in OUT}))
check("LS2 IDENTITY linear control: solves per objective <= C(r+c-2, r-1)", ls2, "; ".join(ls2_info))
check("LS3 IDENTITY solves <= regions used", ls3)
check("LS4 HYPOTHESIS tenfold more tables at most quintuple the solves, every shape and end", ls4, "violations " + (", ".join(ls4_bad) or "none"))
check("LS5 HYPOTHESIS ledger build cheaper than direct solving at T = 10000, every shape", ls5, "violations " + (", ".join(ls5_bad) or "none"))
check("LS6 HYPOTHESIS per-table membership check at least 10x cheaper than one LP, every shape", ls6, "violations " + (", ".join(ls6_bad) or "none"))
check("LS7 RECORD regions vs T and tree bound; second-half share; memory ratio; log-log slopes", True,
      "; ".join(f"{k}: regions(hi0) {[OUT[k]['per_T'][T]['ends']['hi0']['regions_used'] for T in TS]}, 2nd half {OUT[k]['per_T'][TS[-1]]['ends']['hi0']['second_half_share']:.2f}, "
                f"mem {OUT[k]['mem_ledger'] / OUT[k]['mem_bands']:.2f}" for k in OUT))
json.dump(OUT, open(os.path.join(HERE, "study28_scale", "ledgerscale.json"), "w"), indent=1, default=float)
print(f"{sum(results)}/{len(results)} passed")
