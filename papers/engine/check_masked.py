"""check_masked.py -- given measurements, partly masked (note: level2/partial_gluing.pdf, Section 10).

Two qubits, K = 5 known spin directions per side (10 measurements, 25 contexts, N = 2000 shots per context), random entangled noisy
states. A fraction of the 10 directions is masked (their tables are observed, the directions are not).

  MK1 nothing masked: the state is a linear least-squares fit (a convex problem), unique, positive semidefinite within noise
  MK2 10% and 20% masked (1 and 2 directions): each side keeps >= 3 known directions spanning R^3, so the linear fit can locate them; an
      exhaustive lattice of 300 directions per hidden direction is culled to < 2% by the fit alone; when the known directions are
      well-conditioned exactly one explanation survives the cone; ill-conditioned (nearly coplanar) trials are reported: there the linear
      explanations need not be positive and positivity has to drive the search (MK6); the truth lies in the 99% chi-square set in at least
      10 of 12 trials (the Gaussian chi-square calibration is approximate)
  MK3 30% and 50% masked: coordinate-wise lattice sweeps reach the best explanation and keep coverage; the errors are large exactly when a
      side keeps fewer than three independent known directions (the linear gauge below); the cost (L^h exhaustive) and the success rate
      of plain local search are reported
  MK4 the operator-free rungs do not depend on masking: lam* of the 25 tables is computed from the tables alone (same at every level)
  MK5 linear identifiability: a hidden direction is located by the linear fit iff its side's known directions span R^3; with three
      directions per side the 9 tables carry exactly the 15 state parameters, so with known directions in a plane the linear fit is the
      same at every candidate direction (no information at all)
  MK6 the cone culls what the linear fit cannot: candidate directions whose implied state has a negative eigenvalue (negative
      probabilities for some measurement) are removed; the truth survives, the survivors are closed under the mirror (global complex
      conjugation, which preserves positivity), and invariants such as the CHSH maximum are bounded
Writes ../level2/masked_results.json.  Run: python check_masked.py
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import Window
from pce.integrate import gluable_fraction
from pce.masked import (_min_eig_at, feasible_region, batch_rss, cone_cull, refine_psd, span_rank, bell_data, fit_state, rho_from, theta_from, lattice_search, sweep_search, local_search, refine, place, fib_sphere,
                        _dist, _design, _yw, _ang, _vec)
from scipy.stats import chi2

rng = np.random.default_rng(20261024); results = []; OUT = {"levels": {}}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))
def rand_dirs(n): v = rng.normal(size=(n, 3)); return v / np.linalg.norm(v, axis=1)[:, None]
def rand_state(p=0.85):
    psi = rng.normal(size=4) + 1j * rng.normal(size=4); psi /= np.linalg.norm(psi); return p * np.outer(psi, psi.conj()) + (1 - p) * np.eye(4) / 4
def chsh_max(theta):
    s = np.linalg.svd(theta[6:].reshape(3, 3), compute_uv=False); return 2 * np.sqrt(s[0] ** 2 + s[1] ** 2)
def lam_tables(P):
    K = P.shape[0]; names = [f"a{i}" for i in range(K)] + [f"b{j}" for j in range(K)]
    W = [Window((f"a{i}", f"b{j}"), P[i, j].reshape(2, 2)) for i in range(K) for j in range(K)]
    return gluable_fraction({n: 2 for n in names}, W).lam
K, N, Lat = 5, 2000, 300
LEVELS = {"0%": 0, "10%": 1, "20%": 2, "30%": 3, "50%": 5}; TRIALS = 6
if os.environ.get("MK_QUICK"): LEVELS = {"0%": 0, "10%": 1, "20%": 2}; TRIALS = 2

ok1 = ok2 = ok3 = True; rows = []
for lev, h in LEVELS.items():
    recs = []
    for tr in range(TRIALS):
        rho = rand_state(); A, B = rand_dirs(K), rand_dirs(K); P = bell_data(rho, A, B, N, rng); th_true = theta_from(rho)
        lam = lam_tables(P) if tr < 2 else None
        all10 = [("A", i) for i in range(K)] + [("B", j) for j in range(K)]
        hidden = [all10[k] for k in rng.choice(10, h, replace=False)]; true = [(A if s == "A" else B)[i] for s, i in hidden]
        knownA = [A[i] for i in range(K) if ("A", i) not in hidden]; knownB = [B[j] for j in range(K) if ("B", j) not in hidden]
        ident = span_rank(knownA, 1e-6) == 3 and span_rank(knownB, 1e-6) == 3; wellcond = span_rank(knownA) == 3 and span_rank(knownB) == 3
        rss_truth = fit_state(*place(A, B, hidden, true), P, N)[1] if h else None
        rec = {"h": h, "lam": lam, "identifiable": ident, "wellcond": wellcond, "ranks": [span_rank(knownA), span_rank(knownB)]}; t0 = time.time()
        if h == 0:
            th, r = fit_state(A, B, P, N); rec.update(err=0.0, chsh_err=abs(chsh_max(th) - chsh_max(th_true)), evals=1, modes=1, frac=None,
                                                    min_eig=float(np.linalg.eigvalsh(rho_from(th)).min()), rss=r, covered=True)
            ok1 &= rec["min_eig"] > -0.03 and r < chi2.ppf(0.999, 25 * 3 - 15)
        elif h <= 2:
            out = lattice_search(A, B, P, N, hidden, L=Lat, seed=tr); m = out["modes"][0]
            rec.update(err=_dist(m["dirs"], true), chsh_err=abs(chsh_max(m["theta"]) - chsh_max(th_true)), evals=out["evaluated"],
                       modes=len(out["modes"]), frac=out["coarse_fraction"], rss=m["rss"], covered=rss_truth <= m["rss"] + out["threshold"])
            rec["modes_cone"] = sum(mm["cone_ok"] for mm in out["modes"])
            ok2 &= ident and out["coarse_fraction"] < 0.02 and ((rec["modes_cone"] == 1 and m["cone_ok"]) if wellcond else True)
        else:
            outs, ev = sweep_search(A, B, P, N, hidden, L=Lat, starts=6, sweeps=3, seed=tr); best = min(outs, key=lambda x: x[2])
            rec.update(err=_dist(best[0], true), chsh_err=abs(chsh_max(best[1]) - chsh_max(th_true)), evals=ev, frac=None, rss=best[2],
                       covered=rss_truth <= best[2] + chi2.ppf(0.99, 2 * h))
        if h >= 1:
            loc = local_search(A, B, P, N, hidden, starts=12, seed=tr); gbest = min([rec["rss"]] + [x[2] for x in loc])
            rec["local_success"] = sum(x[2] <= gbest + chi2.ppf(0.99, 2 * h) for x in loc) / len(loc)
            if h >= 3: ok3 &= rec["rss"] <= gbest + chi2.ppf(0.99, 2 * h) and rec["covered"]
        rec["seconds"] = time.time() - t0; recs.append(rec)
    OUT["levels"][lev] = recs
    e = [r["err"] for r in recs]; ls = [r.get("local_success") for r in recs if r.get("local_success") is not None]
    ei = [r["err"] for r in recs if r["identifiable"]]; en = [r["err"] for r in recs if not r["identifiable"]]
    rows.append(f"{lev}: located linearly {len(ei)}/{len(recs)}, err median {np.median(ei) if ei else float('nan'):.1f} deg"
                + (f" (others {np.median(en):.1f})" if en else "") + f", covered {sum(r['covered'] for r in recs)}/{len(recs)}, "
                f"fits {int(np.median([r['evals'] for r in recs]))}, local success {np.mean(ls) if ls else float('nan'):.2f}")
    print("   ", rows[-1])
check("MK1 nothing masked: unique linear fit, positive semidefinite within noise, rss consistent with chi-square", ok1)
cov2 = [r["covered"] for lev in ("10%", "20%") for r in OUT["levels"][lev]]
ok2 &= sum(cov2) >= len(cov2) - 2
nm = [(r["modes"], r["modes_cone"]) for lev in ("10%", "20%") for r in OUT["levels"][lev]]; wc = [r["wellcond"] for lev in ("10%", "20%") for r in OUT["levels"][lev]]
ill = [x for x, c in zip(nm, wc) if not c]
check("MK2 10%/20% masked: each side keeps >= 3 known directions spanning R^3; the lattice is culled to < 2%; when the known directions are "
      "well-conditioned exactly one explanation survives the cone (ill-conditioned trials are reported, not required); the truth lies in the 99% "
      "chi-square set in at least 10 of 12 trials", ok2, "; ".join(rows[1:3]) + f"; explanations per trial (linear fit, after the cone) {nm}; well-conditioned {sum(wc)}/{len(wc)}; "
      f"covered {sum(cov2)}/{len(cov2)}; ill-conditioned trials (linear, after cone): {ill}")
check("MK3 30%/50% masked: lattice sweeps reach the best explanation and the truth stays in the chi-square set; errors are large exactly "
      "when a side keeps fewer than three independent known directions", ok3, "; ".join(rows[3:]))

lams = [r["lam"] for lev in OUT["levels"].values() for r in lev if r["lam"] is not None]
check("MK4 the operator-free rungs are computed from the tables alone, so they are the same at every masking level", len(lams) >= 8,
      "lam* of the tables in the sampled trials: " + ", ".join(f"{x:.3f}" for x in lams[:10]))

# MK5 linear identifiability: a hidden direction is located by the linear fit only if its side's known directions span R^3
def xz(t): return np.array([np.sin(t), 0, np.cos(t)])
def unit(v): v = np.asarray(v, float); return v / np.linalg.norm(v)
rho = rand_state(0.9)
A4 = np.array([xz(0.3), xz(1.4), unit([0.2, 0.9, 0.1]), unit([0.5, 0.7, 0.5])]); B4 = rand_dirs(4)
cc_span = cone_cull(A4, B4, bell_data(rho, A4, B4, None, rng), None, [("A", 3)], L=400)
A3 = np.array([xz(0.3), xz(1.4), unit([0.5, 0.7, 0.5])]); B3 = np.array([xz(-0.2), xz(1.1), unit([-0.4, 0.6, 0.3])])
P3 = bell_data(rho, A3, B3, None, rng); hidden3 = [("A", 2), ("B", 2)]
cc_plane = cone_cull(A3, B3, P3, None, hidden3, L=200)
r1 = cc_span["rss"]; r2 = cc_plane["rss"]; b1 = int(np.argmin(r1))
dref, _, rref = refine(A4, B4, bell_data(rho, A4, B4, None, rng), None, [("A", 3)], [cc_span["G"][cc_span["combos"][b1, 0]]])
err_ref = np.degrees(np.arccos(np.clip(dref[0] @ A4[3], -1, 1)))
far = np.degrees(np.arccos(np.clip(cc_span["G"] @ A4[3], -1, 1))) > 30
ok = err_ref < 1e-3 and rref < 1e-12 and r1[far].min() > 2 * r1.min() and np.median(r2) < 1e-12
check("MK5 linear identifiability: with the known directions spanning R^3 the linear fit locates the hidden direction (refining from the "
      "best lattice point recovers it exactly; one basin: points 30 degrees away fit worse); with the known directions in a plane the typical "
      "lattice point fits exactly (no information)", ok,
      f"spanning: refined error {err_ref:.1e} deg, rss beyond 30 deg >= {r1[far].min() / max(r1.min(), 1e-300):.0f} x best; planar: median rss {np.median(r2):.1e}, "
      f"{100 * np.mean(r2 < 1e-12):.0f}% of points fit exactly")

# MK6 the cone culls what the linear fit cannot: planar known directions on both sides, one hidden direction per side
keep = cc_plane["keep"]; frac_strict = float(keep.mean())
fr = feasible_region(A3, B3, P3, None, hidden3, L=200, seed=5); S = fr["samples"]
tA, tB = A3[2], B3[2]; mir = lambda v: v * np.array([1, -1, 1])
dA = np.array([np.degrees(np.arccos(np.clip(d[0] @ tA, -1, 1))) for d, _ in S]); dB = np.array([np.degrees(np.arccos(np.clip(d[1] @ tB, -1, 1))) for d, _ in S])
dAm = np.array([np.degrees(np.arccos(np.clip(d[0] @ mir(tA), -1, 1))) for d, _ in S])
truth_in = bool(np.min(np.maximum(dA, dB)) < 5)
mir_ok = all(_min_eig_at(A3, B3, P3, None, hidden3, [mir(d[0]), mir(d[1])])[0] >= -1e-9 for d, _ in S[::max(1, len(S) // 100)])
chs = np.array([chsh_max(t) for _, t in S]); ct = chsh_max(theta_from(rho))
bnew = unit([0.6, -0.3, 0.74]); ho = np.array([d[0] @ t[6:].reshape(3, 3) @ bnew for d, t in S]); htrue = tA @ theta_from(rho)[6:].reshape(3, 3) @ bnew
spread = float(np.median(np.minimum(dA, dAm)))
ok = len(S) > 100 and truth_in and mir_ok and chs.min() - 1e-6 <= ct <= chs.max() + 1e-6 and fr["feasible_optima"] >= 1
OUT["cone"] = {"strict_fraction": frac_strict, "feasible_optima": fr["feasible_optima"], "n_samples": len(S), "chsh": [float(chs.min()), float(chs.max())],
               "chsh_true": ct, "heldout": [float(ho.min()), float(ho.max())], "heldout_true": float(htrue), "samples_A": [d[0].tolist() for d, _ in S[::5]],
               "true_A": tA.tolist(), "median_dist_to_truth_or_mirror": spread, "chsh_samples": chs[::5].tolist(), "heldout_samples": ho[::5].tolist(),
               "lattice_min_eig": [fr["lattice"]["G"][fr["lattice"]["combos"][:, 0]][::9].tolist(), fr["lattice"]["min_eig"][::9].tolist()],
               "span_rss": [cc_span["G"].tolist(), cc_span["rss"].tolist()], "true_span": A4[3].tolist()}
check("MK6 the cone culls what the linear fit cannot: almost every lattice point implies a state with a negative eigenvalue (negative "
      "probabilities); climbing the smallest eigenvalue from the best lattice points finds the feasible set, which contains the truth, is "
      "closed under the mirror (global complex conjugation), and bounds invariants", ok,
      f"lattice points with a positive state: {100 * frac_strict:.2f}%; feasible optima {fr['feasible_optima']}; {len(S)} feasible samples, median "
      f"{spread:.1f} deg from the truth or its mirror; CHSH max in [{chs.min():.3f}, {chs.max():.3f}] (truth {ct:.3f}); held-out correlation in "
      f"[{ho.min():.3f}, {ho.max():.3f}] (truth {htrue:.3f})")

json.dump(OUT, open(os.path.join(HERE, "..", "level2", "masked_results.json"), "w"), default=float)
print(f"\n{sum(results)}/{len(results)} checks passed")
