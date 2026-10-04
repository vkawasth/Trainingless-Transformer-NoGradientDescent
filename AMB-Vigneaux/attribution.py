"""Context attribution for generated tokens: did a sequence come from context A, context B, a blend, a switching
alternation, or from something unmodeled? Finite-sample version of the decomposition / contamination results of the
posterior-tower paper (identifiability, Huber contamination, unmodeled weight), with probability bands. No gradients.

Inputs per token t: a_t = p_A(x_t | prefix), b_t = p_B(x_t | prefix), u_t = p_U(x_t | prefix) for a broad background
model U (the 'unknown' direction of the Huber model).

Level one (one state for the whole sequence):
    l(w) = sum_t log(w_A a_t + w_B b_t + w_U u_t),   w in the simplex   (concave; maximised by the EM fixed point)
    theta = w_A / (w_A + w_B)  : attribution between the declared contexts
    mu    = w_U               : unmodeled weight
  Bands are profile-likelihood intervals, with the chi-square threshold scaled by a design effect kappa estimated from
  block sums of the score (a sandwich correction for dependence and misspecification of the token stream).
Level two (one state vs alternating states): a two-state hidden Markov chain over {A, B} with free transition
  probabilities contains the i.i.d. blend as the memoryless case; the likelihood-ratio statistic tests 'switching'
  against 'blend' (threshold calibrated by simulation under the fitted blend).
Decision: unknown if the lower confidence bound of mu exceeds a declared budget; else switching if the level-two test
  rejects; else A / B / blend from the theta band.
"""
import re, math
import numpy as np
from collections import Counter, defaultdict

TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def tokenize(text):
    return TOK.findall(text.lower())


class BigramLM:
    """interpolated absolute-discount bigram model over a shared vocabulary (unseen words get add-k unigram mass)"""
    def __init__(self, docs, vocab, d=0.75, k=0.1):
        self.V = vocab; self.idx = {w: i for i, w in enumerate(vocab)}; self.d = d
        uni = Counter(); big = defaultdict(Counter)
        for toks in docs:
            prev = "<s>"
            for w in toks:
                uni[w] += 1; big[prev][w] += 1; prev = w
        tot = sum(uni.values()); self.puni = {w: (uni[w] + k) / (tot + k * len(vocab)) for w in vocab}
        self.pu_arr = np.array([self.puni[w] for w in vocab])
        self.big = big; self.ctot = {v: sum(c.values()) for v, c in big.items()}; self.ntypes = {v: len(c) for v, c in big.items()}
    def p(self, prev, w):
        pu = self.puni.get(w, self.puni.get("<unk>"))
        c = self.big.get(prev)
        if not c: return pu
        n = self.ctot[prev]; return max(c.get(w, 0) - self.d, 0) / n + self.d * self.ntypes[prev] / n * pu
    def seq(self, toks):
        prev, out = "<s>", []
        for w in toks:
            out.append(self.p(prev, w)); prev = w
        return np.array(out)
    def sample_next(self, prev, rng):
        # exact sampling from p(. | prev) over the vocabulary
        c = self.big.get(prev); probs = self.pu_arr.copy()
        if c:
            n = self.ctot[prev]; probs = self.d * self.ntypes[prev] / n * probs
            for w, cnt in c.items():
                probs[self.idx[w]] += max(cnt - self.d, 0) / n
        probs /= probs.sum(); return self.V[rng.choice(len(self.V), p=probs)], probs


def map_unk(toks, vocab_set):
    return [w if w in vocab_set else "<unk>" for w in toks]


def em_weights(P, fixed=None, iters=2000, tol=1e-12):
    """maximise sum_t log(P_t . w) over the simplex (columns of P = components); fixed = {col: value} pins weights"""
    k = P.shape[1]; free = [j for j in range(k) if not fixed or j not in fixed]
    w = np.full(k, 1.0 / k)
    if fixed:
        rest = 1 - sum(fixed.values())
        for j, v in fixed.items(): w[j] = v
        for j in free: w[j] = rest / len(free)
    for _ in range(iters):
        mix = P @ w; r = (P * w) / mix[:, None]; new = w.copy()
        if fixed:
            rest = 1 - sum(fixed.values()); tot = r[:, free].sum(0); new[free] = rest * tot / tot.sum() if tot.sum() > 0 else rest / len(free)
        else:
            new = r.mean(0)
        if np.abs(new - w).max() < tol: w = new; break
        w = new
    return w, float(np.log(P @ w).sum())


def design_effect(P, w, direction, block=None):
    """kappa = var of block sums of the score along `direction` / sum of squared per-token scores (>= ~1 if dependent).
    Default block length ceil(sqrt(n)), so the number of blocks grows with n; fewer than 4 blocks -> kappa = 1."""
    s = (P @ direction) / (P @ w)
    block = block or int(math.ceil(math.sqrt(len(s))))
    nb = len(s) // block
    if nb < 4: return 1.0
    bs = s[: nb * block].reshape(nb, block).sum(1)
    naive = float((s[: nb * block] - s[: nb * block].mean()) ** 2 @ np.ones(nb * block))
    return max(1.0, float(((bs - bs.mean()) ** 2).sum()) / naive) if naive > 0 else 1.0


def profile_band(P, ll_hat, which, grid, crit):
    """profile-likelihood set for a scalar parameter. which='mu': w_U fixed; which='theta': w_A/(w_A+w_B) fixed"""
    keep = []
    for g in grid:
        if which == "mu":
            _, l = em_weights(P, fixed={2: g}) if g < 1 else (None, float(np.log(P[:, 2]).sum()))
        else:
            # theta fixed: mixture of the fixed blend g*A + (1-g)*B with U, free weight on U
            Q = np.c_[g * P[:, 0] + (1 - g) * P[:, 1], P[:, 2]]; _, l = em_weights(Q)
        if 2 * (ll_hat - l) <= crit: keep.append(g)
    return (min(keep), max(keep)) if keep else (float("nan"), float("nan"))


def hmm_ll(a, b, pab, pba, pi=0.5):
    """forward log-likelihood of a two-state chain over {A, B} with emission probabilities a_t, b_t"""
    fa, fb = pi * a[0], (1 - pi) * b[0]; ll = 0.0
    for t in range(1, len(a)):
        z = fa + fb; ll += math.log(z); fa, fb = fa / z, fb / z
        fa, fb = (fa * (1 - pab) + fb * pba) * a[t], (fa * pab + fb * (1 - pba)) * b[t]
    return ll + math.log(fa + fb)


def switching_stat(a, b, grid=np.r_[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 0.995]):
    """max over the HMM grid minus the best i.i.d. blend (the blend is the memoryless chain pab = 1 - lam, pba = lam)"""
    _, lb = em_weights(np.c_[a, b]); best = lb
    for pab in grid:
        for pba in grid:
            pi = pba / (pab + pba)
            best = max(best, hmm_ll(a, b, pab, pba, pi))
    return 2 * (best - lb)


def attribute(a, b, u, mu_budget=0.10, switch_crit=None, level=0.95):
    P = np.c_[a, b, u]; w, ll = em_weights(P)
    crit2 = {0.95: 3.841, 0.9: 2.706}[level]
    kap_mu = design_effect(P, w, np.array([0.0, 0.0, 1.0]) - w)
    kap_th = design_effect(P, w, np.array([1.0, -1.0, 0.0]))
    mu_band = profile_band(P, ll, "mu", np.linspace(0, 1, 101), crit2 * kap_mu)
    mu_lower_1s = profile_band(P, ll, "mu", np.linspace(0, 1, 101), 2.706 * kap_mu)[0]   # one-sided 95% lower bound
    th_band = profile_band(P, ll, "theta", np.linspace(0, 1, 101), crit2 * kap_th)
    theta = w[0] / (w[0] + w[1]) if w[0] + w[1] > 0 else float("nan")
    sw = switching_stat(a, b) if switch_crit is not None else float("nan")
    if mu_lower_1s > mu_budget: dec = "unknown"
    elif switch_crit is not None and sw > switch_crit: dec = "switching"
    elif th_band[0] >= 0.9: dec = "A"
    elif th_band[1] <= 0.1: dec = "B"
    else: dec = "blend"
    return dict(w=w.tolist(), theta=theta, theta_band=th_band, mu=float(w[2]), mu_band=mu_band, mu_lower=mu_lower_1s,
                kappa=(kap_th, kap_mu), switch_stat=sw, decision=dec)


class ContextLM:
    """a generator conditioned on one context document: p(w | prev) = beta p_doc(w | prev) + (1 - beta) p_bg(w | prev)"""
    def __init__(self, doc, bg, vocab, beta=0.5):
        self.doc = BigramLM([doc], vocab, k=1e-6); self.bg = bg; self.beta = beta
    def seq(self, toks):
        return self.beta * self.doc.seq(toks) + (1 - self.beta) * self.bg.seq(toks)
    def sample_next(self, prev, rng):
        lm = self.doc if rng.random() < self.beta else self.bg
        return lm.sample_next(prev, rng)
