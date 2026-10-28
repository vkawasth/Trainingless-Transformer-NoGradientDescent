"""Study 11b: when does a decision on the hiding segment settle? (lab paper, Theorem D(4)). See PREREG_study11b.md.

python run_study11b.py           the registered run (designs D1-D5, seed 1111)
python run_study11b.py --smoke   an invented design (not registered), few replicates: tests the code only
"""
import itertools, json, os, sys
import numpy as np
from numpy.polynomial import polynomial as Pn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from pce import segment_decision as SD
from pce.lab import law, moments
import design11b as D

REPS = 4000; MULTS = (0.5, 1, 2, 4, 8); D4_NS = (1000, 3000, 10000, 30000, 100000); SEED = 1111


def polys_of(mu, v, spec):
    return [SD.U_poly(mu, v, u, range(3)) for u in D._acts(spec)]


def constraint_stats(tabs, spec, u0, a0):
    """the constraint statistics at the estimate: g_j = U_a0(rhat) - u0 and U_a0(rhat) - U_b(rhat), from the tables"""
    e = SD.from_tables(tabs)
    if e is None: return None
    mu, v, rhat, tauhat, R = e; ps = polys_of(mu, v, spec); Ua = Pn.polyval(rhat, ps[a0])
    return np.array([Ua - u0] + [Ua - Pn.polyval(rhat, p) for b, p in enumerate(ps) if b != a0])


def gammas(mu, v, spec, a0):
    ps = polys_of(mu, v, spec); sl = [p[1] if len(p) > 1 else 0.0 for p in ps]
    return np.array([abs(sl[a0])] + [abs(sl[a0]) - abs(sl[b]) for b in range(len(ps)) if b != a0])


def delta_cov(P, spec, u0, a0, eps=1e-6):
    """asymptotic covariance (times n) of the constraint statistics: finite-difference Jacobian per window x multinomial covariance"""
    base = SD.exact_tables(P); Sig = 0.0
    for S in SD.WINDOWS:
        p = base[S].ravel(); J = []
        for k in range(len(p)):
            up = {K: T.copy() for K, T in base.items()}; dn = {K: T.copy() for K, T in base.items()}
            up[S] = (p + eps * np.eye(len(p))[k]).reshape(base[S].shape); dn[S] = (p - eps * np.eye(len(p))[k]).reshape(base[S].shape)
            J.append((constraint_stats(up, spec, u0, a0) - constraint_stats(dn, spec, u0, a0)) / (2 * eps))
        J = np.array(J).T; C = np.diag(p) - np.outer(p, p); Sig = Sig + J @ C @ J.T
    return Sig


def predict(P, spec, u0, a0, tau, ns, draws=400000, seed=7):
    """P(settled at n) = P(g_j sqrt(n) + xi_j >= gamma_j z tau for all j), xi ~ N(0, Sigma) (delta method)"""
    mu, M, mS = moments(P); v = SD.v_closed(M); g = constraint_stats(SD.exact_tables(P), spec, u0, a0); gam = gammas(mu, v, spec, a0)
    Sig = delta_cov(P, spec, u0, a0); xi = np.random.default_rng(seed).multivariate_normal(np.zeros(len(g)), Sig, size=draws)
    out = {int(n): float(np.mean(np.all(g * np.sqrt(n) + xi >= gam * D.Z * tau, axis=1))) for n in ns}
    return out, g, gam, Sig


def simulate(P, spec, u0, a0, ns, reps, rng):
    res = {}
    for n in ns:
        settled = 0; empty = 0; pre_hits = {}
        for _ in range(reps):
            tabs = SD.sample_tables(P, n, rng); e = SD.from_tables(tabs)
            if e is None: empty += 1; continue
            mu, v, rhat, tauhat, R = e; ps = polys_of(mu, v, spec); h = D.Z * tauhat / np.sqrt(n)
            settled += SD.set_rule(ps, SD.clip((rhat - h, rhat + h), R), u0) == a0
            pre = SD.set_rule(ps, R, u0); pre_hits[str(pre)] = pre_hits.get(str(pre), 0) + 1
        res[int(n)] = {"settled": settled / reps, "empty": empty / reps, "pre_probe": {k: c / reps for k, c in pre_hits.items()}}
    return res


def median_n(curve):
    """log-linear interpolation of the n at which the settling rate first crosses 1/2"""
    ns = sorted(curve); ps = [curve[n] for n in ns]
    for k in range(len(ns) - 1):
        if ps[k] < 0.5 <= ps[k + 1]:
            f = (0.5 - ps[k]) / (ps[k + 1] - ps[k]); return float(np.exp(np.log(ns[k]) + f * (np.log(ns[k + 1]) - np.log(ns[k]))))
    return None


def run(designs, reps, seed, smoke=False):
    rng = np.random.default_rng(seed); OUT = {}
    for name in designs:
        if smoke:
            P = law(0.5, np.array([0.75, 0.25, 0.70]), np.array([0.25, 0.70, 0.20])); mu, M, mS = moments(P); v = SD.v_closed(M)
            tau, r0, _, _ = SD.tau_r(P); R = SD.r_walls(mu, v); pA = polys_of(mu, v, [("A", 1.0)])[0]; UC = Pn.polyval(r0, pA) - abs(pA[1]) * 0.3
            spec = [("A", 1.0), ("C", float(UC + mu[0]))]; u0 = float(UC - 3 * 0.3 * abs(pA[1])); a0 = 0; rad = SD.settle_radius_affine(polys_of(mu, v, spec), r0, u0)
            nstar = (D.Z * tau / rad) ** 2; pre_pop = SD.set_rule(polys_of(mu, v, spec), R, u0); ns = [int(round(nstar * k)) for k in MULTS]
        else:
            s = D.summary(name); tn, spec, u0, a0, _ = D.build(name); P = D.truth(tn)[0]; tau = s["tau"]; nstar = s["nstar"]; pre_pop = s["pre_probe"]
            ns = list(D4_NS) if name == "D4" else [int(round(nstar * k)) for k in MULTS]
        pred, g, gam, Sig = predict(P, spec, u0, a0, tau, ns)
        sim = simulate(P, spec, u0, a0, ns, reps, rng)
        emp = {n: sim[n]["settled"] for n in ns}
        OUT[name] = {"ns": ns, "nstar": None if not np.isfinite(nstar) or nstar > 1e12 else float(nstar), "pre_probe_population": pre_pop,
                     "g": g.tolist(), "gamma": gam.tolist(), "Sigma": Sig.tolist(), "predicted": pred, "empirical": emp,
                     "sim": sim, "median_empirical": median_n(emp), "median_predicted": median_n(pred)}
        print(name, {"nstar": OUT[name]["nstar"], "pred": {k: round(x, 3) for k, x in pred.items()}, "emp": {k: round(x, 3) for k, x in emp.items()}}, flush=True)
    return OUT


def evaluate(OUT):
    E = {}
    main = ("D1", "D2", "D3", "D5"); cells = [(d, n) for d in main for n in OUT[d]["ns"]]
    dev = [abs(OUT[d]["empirical"][n] - OUT[d]["predicted"][n]) for d, n in cells]
    E["S1"] = bool(sum(x <= 0.05 for x in dev) >= 18 and max(dev) <= 0.08); E["S1_detail"] = {"within_0.05": int(sum(x <= 0.05 for x in dev)), "cells": len(dev), "max_dev": float(max(dev))}
    ratios = {}
    for d in main:
        me = OUT[d]["median_empirical"]; ratios[d] = None if me is None else me / OUT[d]["nstar"]
    E["S2"] = bool(all(r is not None and 1 / 1.25 <= r <= 1.25 for r in ratios.values())); E["S2_ratios"] = ratios
    d4 = OUT["D4"]; plateau = [abs(d4["empirical"][n] - d4["predicted"][n]) for n in d4["ns"]]
    E["S3"] = bool(max(plateau) <= 0.05 and abs(d4["empirical"][max(d4["ns"])] - d4["empirical"][3000]) <= 0.05)
    E["S3_detail"] = {"max_dev": float(max(plateau)), "emp": d4["empirical"], "pred": d4["predicted"]}
    s4 = {}
    for d in ("D1", "D2", "D3", "D5"):
        nmax = max(OUT[d]["ns"]); s4[d] = OUT[d]["sim"][nmax]["pre_probe"].get(str(OUT[d]["pre_probe_population"]), 0.0)
    E["S4"] = bool(all(x >= 0.95 for x in s4.values())); E["S4_detail"] = s4
    return E


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        OUT = run(["smoke"], int(os.environ.get("SMOKE_REPS", 300)), 5, smoke=True); print(json.dumps({k: OUT["smoke"][k] for k in ("nstar", "median_empirical", "median_predicted")}))
    else:
        OUT = run(D.DESIGNS, REPS, SEED); OUT["evaluation"] = evaluate(OUT)
        json.dump(OUT, open(os.path.join(HERE, "results_study11b.json"), "w"), indent=1, default=float)
        print(json.dumps(OUT["evaluation"], indent=1, default=float))
