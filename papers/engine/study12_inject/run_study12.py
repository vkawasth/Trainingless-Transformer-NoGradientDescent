"""Study 12: injected records that tilt a decision (lab paper, Part II). See PREREG_study12.md.

python run_study12.py            the registered run (seed 1212)
python run_study12.py --smoke    an invented truth and few replicates: tests the code only
python run_study12.py --predict  deterministic predictions only (no sampling)
"""
import itertools, json, os, sys
import numpy as np
from numpy.polynomial import polynomial as Pn
from scipy.stats import norm, chi2, ncx2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import design12 as D
from design12 import SD, IJ, law, moments, fit_v

REPS = 4000; SEED = 1212; ALPHA = 0.05


def marg(P, S): return P.sum(axis=tuple(j for j in range(P.ndim) if j not in S))


def sample_tabs(P, Q, N, eps, windows, rng):
    """each window: N records, round(eps N) of them injected from Q, the rest genuine from P (independent across windows)"""
    na = int(round(eps * N)); tabs = {}
    for S in windows:
        p = marg(P, S).ravel(); q = marg(Q, S).ravel()
        c = rng.multinomial(N - na, p / p.sum()) + (rng.multinomial(na, q / q.sum()) if na else 0)
        tabs[S] = (c / N).reshape((2,) * len(S))
    return tabs


def exact_tabs(P, windows): return {S: marg(P, S) for S in windows}


def strat_cov(P, Q, eps, S):
    p = marg(P, S).ravel(); q = marg(Q, S).ravel()
    return (1 - eps) * (np.diag(p) - np.outer(p, p)) + eps * (np.diag(q) - np.outer(q, q))


def jac(F, tabs, S, h=1e-6):
    base = tabs[S].ravel(); J = []
    for k in range(len(base)):
        up = dict(tabs); dn = dict(tabs)
        up[S] = (base + h * np.eye(len(base))[k]).reshape(tabs[S].shape); dn[S] = (base - h * np.eye(len(base))[k]).reshape(tabs[S].shape)
        J.append((np.atleast_1d(F(up)) - np.atleast_1d(F(dn))) / (2 * h))
    return np.array(J).T


def delta_cov(F, P, Q, eps, windows):
    Pe = IJ.mix(P, Q, eps); tabs = exact_tabs(Pe, windows); Sig = 0.0
    for S in windows:
        J = jac(F, tabs, S); Sig = Sig + J @ strat_cov(P, Q, eps, S) @ J.T
    return Sig


# ---- H1-H3: the hidden channel on design D2 ----
def g_stats(tabs, acts, u0, a0):
    e = SD.from_tables(tabs)
    if e is None: return None
    mu, v, rhat, tauhat, R = e; ps = [SD.U_poly(mu, v, u, range(3)) for u in acts]; Ua = Pn.polyval(rhat, ps[a0])
    return np.array([Ua - u0] + [Ua - Pn.polyval(rhat, p) for b, p in enumerate(ps) if b != a0])


def predict_hidden(h, eps_list, N, draws=400000, seed=7):
    out_A, out_flag = {}, {}; rng = np.random.default_rng(seed)
    for eps in eps_list:
        Pe = IJ.mix(h["P"], h["Q"], eps); tabs = exact_tabs(Pe, SD.WINDOWS); mu, v, rhat, tau, R = SD.from_tables(tabs)
        ps = [SD.U_poly(mu, v, u, range(3)) for u in h["acts"]]; sl = [p[1] if len(p) > 1 else 0.0 for p in ps]; a0 = h["a0"]
        gam = np.array([abs(sl[a0])] + [abs(sl[a0]) - abs(sl[b]) for b in range(len(ps)) if b != a0])
        F = lambda t: g_stats(t, h["acts"], h["u0"], a0); g = F(tabs); Sig = delta_cov(F, h["P"], h["Q"], eps, SD.WINDOWS)
        xi = rng.multivariate_normal(np.zeros(len(g)), Sig, size=draws)
        out_A[eps] = float(np.mean(np.all(g * np.sqrt(N) + xi >= gam * D.Z * tau, axis=1)))
        left = h["rc"] < h["r0"]
        Fd = lambda t: (lambda e: np.nan if e is None else (e[2] - e[4][0] if left else e[4][1] - e[2]))(SD.from_tables(t))
        dval = Fd(tabs); sd = np.sqrt(delta_cov(Fd, h["P"], h["Q"], eps, SD.WINDOWS)[0, 0])
        out_flag[eps] = float(norm.cdf(-dval * np.sqrt(N) / sd))
    return out_A, out_flag


def simulate_hidden(h, eps_list, N, reps, rng):
    res = {}
    for eps in eps_list:
        A = flag = 0
        for _ in range(reps):
            e = SD.from_tables(sample_tabs(h["P"], h["Q"], N, eps, SD.WINDOWS, rng))
            if e is None: flag += 1; continue
            mu, v, rhat, tauhat, R = e
            if not (R[0] <= rhat <= R[1]): flag += 1; continue
            ps = [SD.U_poly(mu, v, u, range(3)) for u in h["acts"]]; hh = D.Z * tauhat / np.sqrt(N)
            A += SD.set_rule(ps, SD.clip((rhat - hh, rhat + hh), R), h["u0"]) == h["a0"]
        res[eps] = {"A": A / reps, "flag": flag / reps}
    return res


def crossing(curve, descending):
    ks = sorted(curve)
    for a, b in zip(ks, ks[1:]):
        ya, yb = curve[a], curve[b]
        if (descending and ya >= 0.5 > yb) or (not descending and ya < 0.5 <= yb): return float(a + (0.5 - ya) * (b - a) / (yb - ya))
    return None


# ---- H4: the tetrad test at m = 4 ----
PAIRS4 = list(itertools.combinations(range(4), 2))

def tetrad_stat(tabs):
    M = {}; V = {}
    for (i, j) in PAIRS4:
        T = tabs[(i, j)]; pi, pj = T[1].sum(), T[:, 1].sum(); M[(i, j)] = T[1, 1] - pi * pj
        V[(i, j)] = sum(T[a, b] * ((a - pi) * (b - pj) - M[(i, j)]) ** 2 for a in (0, 1) for b in (0, 1))
    t = np.array([M[(0, 1)] * M[(2, 3)] - M[(0, 2)] * M[(1, 3)], M[(0, 1)] * M[(2, 3)] - M[(0, 3)] * M[(1, 2)]])
    G = np.zeros((2, 6)); idx = {p: k for k, p in enumerate(PAIRS4)}
    G[0, idx[(0, 1)]] = M[(2, 3)]; G[0, idx[(2, 3)]] = M[(0, 1)]; G[0, idx[(0, 2)]] = -M[(1, 3)]; G[0, idx[(1, 3)]] = -M[(0, 2)]
    G[1, idx[(0, 1)]] = M[(2, 3)]; G[1, idx[(2, 3)]] = M[(0, 1)]; G[1, idx[(0, 3)]] = -M[(1, 2)]; G[1, idx[(1, 2)]] = -M[(0, 3)]
    Sig = G @ np.diag([V[p] for p in PAIRS4]) @ G.T
    return t, Sig


def predict_tetrad(P, Q, eps, N):
    t, _ = tetrad_stat(exact_tabs(IJ.mix(P, Q, eps), PAIRS4))
    Sig = delta_cov(lambda tb: tetrad_stat(tb)[0], P, Q, eps, PAIRS4)
    lam = float(N * t @ np.linalg.solve(Sig, t)); crit = chi2.ppf(1 - ALPHA, 2)
    return float(ncx2.sf(crit, 2, lam)) if lam > 1e-12 else ALPHA, lam


def simulate_tetrad(P, Q, eps, N, reps, rng):
    crit = chi2.ppf(1 - ALPHA, 2); rej = 0
    for _ in range(reps):
        t, Sig = tetrad_stat(sample_tabs(P, Q, N, eps, PAIRS4, rng)); Sig = Sig / N
        rej += float(t @ np.linalg.solve(Sig, t)) > crit
    return rej / reps


def run(reps, seed, smoke=False, predict_only=False):
    rng = np.random.default_rng(seed); OUT = {}
    h = D.hidden()
    if smoke:                                                         # an invented truth, not registered
        Ps = law(0.5, np.array([0.75, 0.25, 0.70]), np.array([0.25, 0.70, 0.20])); base = IJ.pipeline(Ps, h["acts"], -10)
        lo, hi = IJ.stealth_range(Ps, (0, 1, 2)); ps = base["polys"]
        shift = Pn.polyval(base["rhat"] - 0.3, ps[0]) - Pn.polyval(0, ps[1])
        acts = [h["acts"][0], (lambda f, c: (lambda x: f(x) + c))(h["acts"][1], shift)]; ps = IJ.pipeline(Ps, acts, -10)["polys"]
        r0, R, vT = base["rhat"], base["R"], float(np.prod(base["v"])); rc = base["rhat"] - 0.3
        s = hi if np.sign(hi / vT) == np.sign(rc - r0) else lo
        h = dict(P=Ps, Q=IJ.as_law(IJ.as_vec(Ps) + s * IJ.chi(3, (0, 1, 2)), 3), acts=acts, u0=-10.0, a0=SD.oracle(ps, r0, -10.0), s=s, r0=r0, R=R,
                 vT=vT, rc=rc, eps_star=0.3 * abs(vT) / abs(s), eps_det=abs((R[0] if rc < r0 else R[1]) - r0) * abs(vT) / abs(s))
        ef = [round(h["eps_star"] * k, 4) for k in (0.5, 0.8, 1.0, 1.2)]; ed = [round(h["eps_det"] * k, 4) for k in (0.9, 1.0, 1.1)]
    else:
        ef, ed = list(D.EPS_FLIP), list(D.EPS_DET)
    predA, predF = predict_hidden(h, sorted(set(ef + ed)), D.N)
    OUT["hidden"] = {"eps_star": h["eps_star"], "eps_det": h["eps_det"], "s": h["s"], "vT": h["vT"], "r0": h["r0"], "rc": h["rc"],
                     "pred_A": predA, "pred_flag": predF, "eps_half_pred": crossing({e: predA[e] for e in ef}, True)}
    tau0 = SD.from_tables(exact_tabs(h["P"], SD.WINDOWS))[3]
    OUT["hidden"]["eps_half_closed_form"] = (abs(h["rc"] - h["r0"]) - D.Z * tau0 / np.sqrt(D.N)) * abs(h["vT"]) / abs(h["s"])
    t = D.tetrad_design(D.G4_SMOKE if smoke else None); OUT["tetrad"] = {"eps_s": t["eps_s"], "eps_v": t["eps_v"], "tilt_s": t["tilt_s"], "tilt_v": t["tilt_v"], "pred": {}}
    for name, Q, e in (("stealth", t["Qs"], t["eps_s"]), ("vertex", t["Qv"], t["eps_v"])):
        for n in D.N_TET: OUT["tetrad"]["pred"][f"{name}_{n}"] = predict_tetrad(t["P"], Q, e, n)
    print(json.dumps({"hidden": {k: v for k, v in OUT["hidden"].items()}, "tetrad": OUT["tetrad"]}, default=float), flush=True)
    if predict_only: return OUT
    sim = simulate_hidden(h, sorted(set(ef + ed)), D.N, reps, rng)
    OUT["hidden"]["emp_A"] = {e: sim[e]["A"] for e in sim}; OUT["hidden"]["emp_flag"] = {e: sim[e]["flag"] for e in sim}
    OUT["hidden"]["eps_half_emp"] = crossing({e: sim[e]["A"] for e in ef}, True)
    OUT["hidden"]["det_cross_emp"] = crossing({e: sim[e]["flag"] for e in ed}, False)
    OUT["tetrad"]["emp"] = {}
    for name, Q, e in (("stealth", t["Qs"], t["eps_s"]), ("vertex", t["Qv"], t["eps_v"])):
        for n in D.N_TET: OUT["tetrad"]["emp"][f"{name}_{n}"] = simulate_tetrad(t["P"], Q, e, n, reps, rng)
    print(json.dumps({"emp_A": OUT["hidden"]["emp_A"], "emp_flag": OUT["hidden"]["emp_flag"], "tet": OUT["tetrad"]["emp"]}, default=float), flush=True)
    return OUT, ef, ed


def evaluate(OUT, ef, ed):
    H = OUT["hidden"]; T = OUT["tetrad"]; E = {}
    d1 = [abs(H["emp_A"][e] - H["pred_A"][e]) for e in ef]
    E["H1"] = bool(sum(x <= 0.05 for x in d1) >= 9 and max(d1) <= 0.08 and H["eps_half_emp"] is not None and abs(H["eps_half_emp"] / H["eps_half_pred"] - 1) <= 0.10)
    E["H1_detail"] = {"max_dev": max(d1), "within": int(sum(x <= 0.05 for x in d1)), "eps_half_emp": H["eps_half_emp"], "eps_half_pred": H["eps_half_pred"]}
    d2 = [abs(H["emp_flag"][e] - H["pred_flag"][e]) for e in ed]
    E["H2"] = bool(sum(x <= 0.05 for x in d2) >= 6 and H["det_cross_emp"] is not None and abs(H["det_cross_emp"] / H["eps_det"] - 1) <= 0.05)
    E["H2_detail"] = {"max_dev": max(d2), "cross_emp": H["det_cross_emp"], "eps_det": H["eps_det"]}
    E["H3"] = bool(H["emp_A"][0.07] <= 0.5 and H["emp_flag"][0.07] <= 0.01); E["H3_detail"] = {"A_at_0.07": H["emp_A"][0.07], "flag_at_0.07": H["emp_flag"][0.07]}
    d4 = {k: abs(T["emp"][k] - T["pred"][k][0]) for k in T["emp"]}
    E["H4"] = bool(sum(x <= 0.04 for x in d4.values()) >= 5 and all(T["emp"][f"stealth_{n}"] <= 0.10 for n in D.N_TET) and T["emp"]["vertex_50000"] >= 0.80)
    E["H4_detail"] = {"dev": d4, "emp": T["emp"]}
    return E


if __name__ == "__main__":
    if "--predict" in sys.argv: run(0, SEED, predict_only=True)
    elif "--smoke" in sys.argv: run(int(os.environ.get("SMOKE_REPS", 300)), 5, smoke=True)
    else:
        OUT, ef, ed = run(REPS, SEED); OUT["evaluation"] = evaluate(OUT, ef, ed)
        json.dump(OUT, open(os.path.join(HERE, "results_study12.json"), "w"), indent=1, default=float)
        print(json.dumps(OUT["evaluation"], indent=1, default=float))
