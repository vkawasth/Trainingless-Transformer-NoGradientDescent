"""Study 92 helpers (calibration stage). 3x3 cloud-seeding design: rows (AgI-seeded, salt-seeded, unseeded storms), columns (below,
normal, above). Reuses lib91 (Sys, Walker, solve, analyse) with Sys(3, 3): 18 critical points off the escape loci (Study 79)."""
import numpy as np, lib91 as Q
RNG = np.random.default_rng
def gen33(seed, npaths=4, K=13):
    rng = RNG(seed); out = []
    for p in range(npaths):
        lf = np.log([0.3, 0.3, 0.4]) + 0.25 * rng.normal(size=3); lg = np.log([0.3, 0.4, 0.3]) + 0.25 * rng.normal(size=3); Fw, Gw = [], []
        for k in range(K + 1):
            Fw.append(np.exp(lf) / np.exp(lf).sum()); Gw.append(np.exp(lg) / np.exp(lg).sum())
            lf = lf + 0.15 * rng.normal(size=3); lg = lg + 0.15 * rng.normal(size=3)
        out.append((np.array(Fw), np.array(Gw)))
    return out
def base_point33(seed=9201):
    rng = RNG(seed); return rng.dirichlet([6] * 3), rng.dirichlet([6] * 3)
def start_solutions33(S, Fw0, Gw0, L, seed=1, bseed=9201, target=18):
    f0, g0 = base_point33(bseed); al, be = Q.params(f0, g0, L); x, res, n = Q.solve(S, al, be, seed=seed, target=target, max_loops=400)
    for imag in (0.2, -0.17, 0.31, -0.29):
        path = lambda t: Q.params(f0 + (t + imag * 1j * t * (1 - t)) * (Fw0 - f0), g0 + (t + imag * 1j * t * (1 - t)) * (Gw0 - g0), L)
        y, ok = S.track(x, path, n0=400); y, r = S.newton(y, *Q.params(Fw0, Gw0, L))
        d = np.abs(y[:, None] - y[None]).max(-1) + 9 * np.eye(len(y))
        if ok.all() and d.min() > 1e-6 and len(y) == target: break
    return y, r, n, ok
def analyse_fine(W, nbis=40, nfine=16):
    """lib91.analyse, except that a substep with a multiple caustic ("caustic?") is re-walked in nfine sub-substeps and analysed there
    (the Study 91 QF3/QF4 gap). Returns the event list."""
    ev = Q.analyse(W, nbis); out = []
    for e in ev:
        if e["kind"] != "caustic?": out.append(e); continue
        k = int(np.searchsorted(W.T, e["t"], side="right") - 1); t0, t1 = W.T[k], W.T[k + 1]
        sub = Q.Walker(W.S, W.Fw, W.Gw, W.L); Ts = [t0]; Xs = [W.X[k]]; x = W.X[k]
        for q in range(1, nfine + 1):
            tq = t0 + (t1 - t0) * q / nfine; x, _ = W.step(x, Ts[-1], tq); Ts.append(tq); Xs.append(x)
        sub.T, sub.X = np.array(Ts), np.array(Xs); sub.V = W.S.value(sub.X); sub.det = [False] * nfine
        sev = Q.analyse(sub, nbis)
        out += [dict(s, refined=True) for s in sev] if sev else [dict(e, refined="none found")]
    return out
