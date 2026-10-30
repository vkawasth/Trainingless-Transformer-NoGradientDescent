"""Study 11c -- settling with the exact conditional probe (pre-registered; see PREREG_study11c.md).

Pair tables are known (population). One triple window of n draws. Two probes on the same samples:
  delta  rhat = mhat_T / v_T (centred moment with the window's own means), interval rhat +- z tauhat_m / sqrt(n), tauhat_m plug-in
  exact  the two-sided 95% exact conditional interval for the three-way log odds ratio psi given the window's own pair tables,
         mapped to r through the population margin fibre (P + eps chi has log odds ratio psi; m_T = m_T(P) + eps; r = m_T / v_T)
Both intervals are cut by the population segment R; the set rule on the result settles when it picks the oracle act.
Normal-theory predictions: SD.settle_probability with tau_m (delta) and tau_c (conditional), n*_m and n*_c = (z tau / d)^2.
Usage: python run_study11c.py [--smoke]
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(HERE, ".."); sys.path.insert(0, ENG); sys.path.insert(0, os.path.join(ENG, "study11b_settle"))
from pce import ehrhart as EH, segment_decision as SD, lab as L
import design11b as D

MULTS = (0.5, 1, 2, 4, 8); D4_NS = (1000, 3000, 10000, 30000, 100000); REPS = 2000; SEED = 1112
DESIGNS = ("D2", "D3", "D5", "D4")


def setup(name, smoke=False):
    if smoke:                                                          # invented, not registered
        P = L.law(0.5, np.array([.75, .25, .70]), np.array([.25, .70, .20])); mu, M, _ = L.moments(P); v = SD.v_closed(M)
        tau, r0, _, _ = SD.tau_r(P); R = SD.r_walls(mu, v); spec = [("A", 1.0)]
        pA = SD.U_poly(mu, v, D._acts(spec)[0], range(3)); u0 = float(np.polyval(pA[::-1], r0) - abs(pA[1]) * 0.35)
    else:
        tn, spec, u0, a0, d = D.build(name); P, mu, v, tau, r0, R = D.truth(tn)
    polys = [SD.U_poly(mu, v, u, range(3)) for u in D._acts(spec)]; vT = float(np.prod(v))
    a0, cons = SD.settle_constraints(polys, r0, u0); rad = SD.settle_radius_affine(polys, r0, u0)
    tc, tm = EH.tau_c(P, vT), EH.tau_m(P, vT)
    return dict(P=P, polys=polys, u0=u0, a0=a0, cons=cons, rad=rad, vT=vT, m0=r0 * vT, R=R, tc=tc, tm=tm,
                nstar_c=(D.Z * tc / rad) ** 2 if rad > 1e-9 else np.inf, nstar_m=(D.Z * tm / rad) ** 2 if rad > 1e-9 else np.inf)


def exact_interval(S, T):
    """the exact conditional interval for psi, mapped to r through the population fibre"""
    pl, pu = EH.exact_ci(T, 0.05, 0)
    lo_e, hi_e = EH.eps_range(S["P"])
    eL = lo_e if not np.isfinite(pl) else EH.eps_of_psi(S["P"], pl); eU = hi_e if not np.isfinite(pu) else EH.eps_of_psi(S["P"], pu)
    a, b = (S["m0"] + eL) / S["vT"], (S["m0"] + eU) / S["vT"]; return min(a, b), max(a, b)


def delta_interval(S, T, n):
    Q = T / n; mh = EH.m_hat(T); rh = mh / S["vT"]
    try: th = EH.tau_m(Q, S["vT"])
    except Exception: th = np.inf
    h = D.Z * th / np.sqrt(n); return rh - h, rh + h


def simulate(S, ns, reps, rng):
    out = {}
    for n in ns:
        se = sd = 0
        for _ in range(reps):
            T = rng.multinomial(n, S["P"].ravel()).reshape(2, 2, 2)
            se += SD.set_rule(S["polys"], SD.clip(exact_interval(S, T), S["R"]), S["u0"]) == S["a0"]
            sd += SD.set_rule(S["polys"], SD.clip(delta_interval(S, T, n), S["R"]), S["u0"]) == S["a0"]
        out[int(n)] = dict(exact=se / reps, delta=sd / reps,
                           pred_c=SD.settle_probability(S["cons"], S["tc"], D.Z, n), pred_m=SD.settle_probability(S["cons"], S["tm"], D.Z, n))
    return out


def median_n(curve):
    ns = sorted(curve); ps = [curve[n] for n in ns]
    for k in range(len(ns) - 1):
        if ps[k] < 0.5 <= ps[k + 1]:
            f = (0.5 - ps[k]) / (ps[k + 1] - ps[k]); return float(np.exp(np.log(ns[k]) + f * (np.log(ns[k + 1]) - np.log(ns[k]))))
    return None


def evaluate(OUT):
    ev = {}; main = [d for d in ("D2", "D3", "D5") if d in OUT]
    e1 = {d: OUT[d]["median_exact"] / OUT[d]["nstar_c"] if OUT[d]["median_exact"] else None for d in main}
    ev["E1"] = dict(ratios=e1, holds=all(r is not None and 1 / 1.25 <= r <= 1.25 for r in e1.values()))
    e2 = {d: (OUT[d]["median_exact"], OUT[d]["median_delta"]) for d in main}
    ev["E2"] = dict(medians=e2, holds=all(a is not None and b is not None and a < b for a, b in e2.values()))
    gaps = {d: max(abs(c["exact"] - c["pred_c"]) for n, c in OUT[d]["curve"].items()) for d in main}
    ev["E3"] = dict(max_gap=gaps, holds=all(g <= 0.06 for g in gaps.values()))
    if "D4" in OUT:
        c = OUT["D4"]["curve"]; a, b = c[3000]["exact"], c[100000]["exact"]
        ev["E4"] = dict(rate_3000=a, rate_100000=b, holds=abs(a - b) <= 0.05 and a < 0.5 and b < 0.5)
    return ev


def run(smoke=False):
    rng = np.random.default_rng(SEED); OUT = {}
    for name in (("smoke",) if smoke else DESIGNS):
        S = setup(name, smoke); t0 = time.time()
        ns = D4_NS if name == "D4" else [int(round(S["nstar_c"] * k)) for k in MULTS]
        curve = simulate(S, ns, 300 if smoke else REPS, rng)
        OUT[name] = dict(oracle=S["a0"], radius=S["rad"], tau_c=S["tc"], tau_m=S["tm"], nstar_c=S["nstar_c"], nstar_m=S["nstar_m"], curve=curve,
                         median_exact=median_n({n: c["exact"] for n, c in curve.items()}), median_delta=median_n({n: c["delta"] for n, c in curve.items()}),
                         seconds=time.time() - t0)
        print(name, json.dumps({k: v for k, v in OUT[name].items() if k != "curve"}, default=float), flush=True)
        for n, c in curve.items(): print("   ", n, c, flush=True)
    if not smoke: OUT["evaluation"] = evaluate(OUT)
    json.dump(OUT, open(os.path.join(HERE, "results_study11c_smoke.json" if smoke else "results_study11c.json"), "w"), indent=1, default=float)
    if not smoke: print(json.dumps(OUT["evaluation"], indent=1, default=float))


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
