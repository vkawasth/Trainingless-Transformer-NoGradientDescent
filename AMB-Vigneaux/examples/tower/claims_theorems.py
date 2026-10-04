"""Numerical checks of the claim-learning theorems (three_layers, Section 'Obstructed claims cannot learn'):
  Lemma 1     graded distance d2W(x2) = Kantorovich distance from x2 to D(S1)   (transport LP)
  Theorem 2   limit of the updated prediction and the bound chain                (simulation + exact limit)
  Corollary 3 truth-free bounds  s/2 <= lim TV <= r / sqrt(2 sigma_min)
  Theorem 4   explicit finite-n bound (Hoeffding + union), violation rate <= delta
  Prop. 5a    E_{h*~pi} lim TV <= r_pi / sqrt(2 sigma_min) <= d2W / sqrt(2 sigma_min) for perturbed honest claims
  Prop. 5b    d2W alone cannot bound the error (d2W = 0, permanent error)
  Ties        two KL-minimisers: p_n does not converge, liminf TV can be 0 < s/2 <= limsup TV
All numbers are printed; tests/test_claims_theorems.py runs reduced versions."""
import numpy as np
from scipy.optimize import linprog


def l1(a, b): return float(np.abs(a - b).sum())
def tv(a, b): return 0.5 * l1(a, b)
def kl(p, q): return float(np.sum(p * np.log(p / q)))


def quantities(w, Sig, H):
    D = np.array([[l1(s, r) for r in H] for s in Sig])                 # atoms x hypotheses
    return dict(d2W=float(w @ D.min(1)), s=float(D.min()), r=float(D.min(0).max()), smin=float(Sig.min()))


def transport(w, Sig, H):
    """min over nu in D(S1) and couplings of x2 and nu of sum pi_ch ||sigma_c - rho_h||_1"""
    C, M = len(Sig), len(H); cost = np.array([[l1(s, r) for r in H] for s in Sig]).ravel()
    A = np.zeros((C, C * M))
    for c in range(C): A[c, c * M:(c + 1) * M] = 1                    # row marginals = w; column marginal free (nu)
    return linprog(cost, A_eq=A, b_eq=w, bounds=(0, None), method="highs").fun


def update(w, Sig, ys):
    lw = np.log(w) + np.log(Sig[:, ys]).sum(1); lw -= lw.max(); v = np.exp(lw); return v / v.sum()


def random_instance(rng, K=4, M=5, C=None):
    H = rng.dirichlet(np.ones(K) * 2, size=M); C = C or int(rng.integers(1, 5))
    Sig = np.abs(H[rng.integers(M, size=C)] + rng.uniform(0, 0.4) * rng.normal(size=(C, K))) + 0.02
    Sig /= Sig.sum(1, keepdims=True); w = rng.dirichlet(np.ones(C)); return H, Sig, w


def check_lemma1(trials=400, seed=0):
    rng = np.random.default_rng(seed); worst = 0.0
    for _ in range(trials):
        H, Sig, w = random_instance(rng); worst = max(worst, abs(transport(w, Sig, H) - quantities(w, Sig, H)["d2W"]))
    return worst


def check_thm2_cor3(trials=400, n=4000, seed=1):
    """exact limit sigma_{c+} checked against the chain; simulated p_n checked to be near the limit"""
    rng = np.random.default_rng(seed); viol = 0; far = 0; nchk = 0
    for _ in range(trials):
        H, Sig, w = random_instance(rng); Q = quantities(w, Sig, H)
        for hs in range(len(H)):                                        # uniformly over the truth (Cor. 3)
            rho = H[hs]; K = np.array([kl(rho, s) for s in Sig]); cd = int(np.argmin(K))
            if np.sort(K)[1:2].size and np.sort(K)[1] - K[cd] < 1e-3: continue        # near-ties: Theorem 2 needs uniqueness
            lim = tv(Sig[cd], rho)
            chain = [0.5 * Q["s"], 0.5 * min(l1(s, rho) for s in Sig), lim, np.sqrt(0.5 * K.min()),
                     min(l1(s, rho) for s in Sig) / np.sqrt(2 * Q["smin"])]
            viol += any(chain[i] > chain[i + 1] + 1e-12 for i in range(4)) or lim > Q["r"] / np.sqrt(2 * Q["smin"]) + 1e-12
            if len(K) > 1 and np.sort(K)[1] - K[cd] < 5e-3: continue                     # convergence check away from near-ties
            nchk += 1; ys = rng.choice(len(rho), size=n, p=rho); far += tv(update(w, Sig, ys) @ Sig, rho) - lim > 0.02
    return viol, far, nchk


def thm4_bound(w, Sig, rho, n, delta):
    K = np.array([kl(rho, s) for s in Sig]); cd = int(np.argmin(K)); ell = np.log(1 / Sig.min()); C = len(Sig)
    t = ell * np.sqrt(2 * n * np.log(max(C - 1, 1) / delta))
    return cd, sum(w[c] / w[cd] * np.exp(-n * (K[c] - K[cd]) + t) for c in range(C) if c != cd)


def check_thm4(trials=200, n=300, delta=0.1, reps=50, seed=2):
    rng = np.random.default_rng(seed); viol = tot = nonvac = 0
    for _ in range(trials):
        H, Sig, w = random_instance(rng, C=3); rho = H[rng.integers(len(H))]
        cd, B = thm4_bound(w, Sig, rho, n, delta); nonvac += B < 1
        for _ in range(reps):
            ys = rng.choice(len(rho), size=n, p=rho); viol += tv(update(w, Sig, ys) @ Sig, Sig[cd]) > B; tot += 1
    return viol / tot, nonvac / trials


def check_prop5a(trials=300, seed=3):
    """perturbed honest claims: atom sigma_h near rho_h, weight pi(h); E_pi lim TV <= r_pi/sqrt(2 smin) <= d2W/sqrt(2 smin)"""
    rng = np.random.default_rng(seed); viol = used = 0
    for _ in range(trials):
        M, K = 5, 4; H = rng.dirichlet(np.ones(K) * 2, size=M); pi = rng.dirichlet(np.ones(M))
        Sig = np.abs(H + rng.uniform(0, 0.15) * rng.normal(size=H.shape)) + 0.01; Sig /= Sig.sum(1, keepdims=True)
        D = np.array([[l1(s, r) for r in H] for s in Sig])
        if not all(D[i].argmin() == i for i in range(M)): continue      # hypothesis of Prop. 5a: nearest admissible law is its own
        used += 1; Q = quantities(pi, Sig, H); c = np.sqrt(2 * Q["smin"])
        lim = []
        for hs in range(M):
            Kl = np.array([kl(H[hs], s) for s in Sig]); lim.append(tv(Sig[int(np.argmin(Kl))], H[hs]))
        r_pi = float(pi @ D.min(0)); E = float(pi @ np.array(lim))
        viol += E > r_pi / c + 1e-12 or r_pi > Q["d2W"] + 1e-12
    return viol, used


def tie_example(eps=0.1, n=20000, seed=4):
    """S1 = {uniform}, sigma_{1,2} = uniform +- eps on two outcomes: equal KL, p_n hits rho* infinitely often"""
    rho = np.ones(3) / 3; Sig = np.array([[1 / 3 + eps, 1 / 3 - eps, 1 / 3], [1 / 3 - eps, 1 / 3 + eps, 1 / 3]])
    rng = np.random.default_rng(seed); ys = rng.choice(3, size=n, p=rho)
    L = np.cumsum(np.where(ys == 0, 1, np.where(ys == 1, -1, 0)))    # log-ratio in units of log(sigma1(0)/sigma2(0))
    a = np.log(Sig[0, 0] / Sig[1, 0]); t = 1 / (1 + np.exp(-a * L))  # weight of atom 1 (equal prior weights)
    err = np.array([tv(ti * Sig[0] + (1 - ti) * Sig[1], rho) for ti in t[n // 2:]])
    return dict(half_s=eps, kl_gap=abs(kl(rho, Sig[0]) - kl(rho, Sig[1])), zero_hits=int((L[n // 2:] == 0).sum()),
                min_err_late=float(err.min()), max_err_late=float(err.max()))


def collapsed_example(seed=5):
    """Prop. 5b: x2 = delta_{rho_h1} has d2W = 0 but error TV(rho_h1, rho*) forever when h* != h1"""
    rng = np.random.default_rng(seed); H = rng.dirichlet(np.ones(4) * 2, size=5); w = np.ones(1); Sig = H[:1]
    Q = quantities(w, Sig, H); ys = rng.choice(4, size=5000, p=H[2])
    return dict(d2W=Q["d2W"], r=Q["r"], err=tv(update(w, Sig, ys) @ Sig, H[2]), predicted=tv(H[0], H[2]))


if __name__ == "__main__":
    print("Lemma 1: max |transport LP - d2W| over 400 instances = %.2e" % check_lemma1())
    v, f, m = check_thm2_cor3(); print("Theorem 2 / Cor 3: chain violations %d; simulated p_n (n=4000) more than 0.02 from the limit: %d of %d" % (v, f, m))
    print("Theorem 4: violation rate of the explicit bound at delta = 0.1: %.4f (bound < 1 in %.2f of instances)" % check_thm4())
    v, u = check_prop5a(); print("Prop 5a: violations %d of %d instances" % (v, u))
    print("Prop 5b:", collapsed_example())
    print("Ties:", tie_example())
