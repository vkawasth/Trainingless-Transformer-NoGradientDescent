"""Lifting: the decision maker's patchwork (lab paper, Theorem K and the propositions around it). New module; nothing frozen.

  we()                         the worked example of Section 17 (truth, mu, v, segment R, act polynomials, switch r_c, tau)
  arrangement(P)               the decision hyperplane on the pair line: T vs P depends only on the cell 001 (P(001) = 0.128)
  act_lift(a, b)               Kleisli lift of two acts: worst regrets a (of T) and b (of P) -> alpha* = b / (a + b), value ab / (a + b)
  lifted_rule(I)               the lifted (mixed) rule on an evidence interval I: continuous in the evidence
  block_radius(O, chamber)     the adversarial lift (P, Q, eps): the smallest share eps at which the block of explanations of the
                               observed law O reaches a genuine law in the other chamber (SLSQP multi-start over the two-type model)
  dilation(P)                  P(C | A, B) over the hiding segment, for each (A, B): intervals (pair-only disclosure) vs points (triple)
  evsi(n), savage_curve(n), wald_curve(n)   the value of the probe as a function of its size
  blowup_sd(mu, v, n, reps)    sampling sd of rho^2 = ||offdiag M|| and of |v| = rho at and away from the merge
  backtrack(deltas, retro)     sequential gluing (Gaussian model): forward-only acceptance vs backtracking (revoking past sections)
  rect_example()               a two-stage event tree: a non-rectangular set C, its rectangular hull (the sheafification), the
                               dynamic-inconsistency loss and the bound osc(u) d_H(C, rect C)
"""
from __future__ import annotations
import itertools
import numpy as np
from numpy.polynomial import polynomial as Pn
from scipy.optimize import minimize
from scipy.stats import norm
from .lab import law, moments, segment, explanation, r_of_t, t_of_r
from . import segment_decision as SD
from . import injection as IJ

XS = list(itertools.product((0, 1), repeat=3))


def we():
    P = law(0.6, np.array([.5, .1, .4]), np.array([.05, .5, .1])); mu, M, mS = moments(P); v = SD.v_closed(M); vT = float(np.prod(v))
    r0 = mS((0, 1, 2)) / vT; lo, hi = segment(mu, v); R = (r_of_t(lo), r_of_t(hi))
    acts = [lambda x: 1 - float(sum(x) > 0), lambda x: 0.872 - float(x[0] or x[1])]
    ps = [SD.U_poly(mu, v, u, range(3)) for u in acts]; UP = float(ps[1][0]); rc = (UP - ps[0][0]) / ps[0][1]
    return dict(P=P, mu=mu, v=v, vT=vT, r0=r0, R=R, acts=acts, ps=ps, UP=UP, rc=rc, tau=SD.tau_r(P)[0], u0=0.30)


def gapTP(Pq):
    """U_T - U_P for the worked example's acts: (1 - P(any harm)) - (0.872 - P(A or B)) = 0.128 - P(001)"""
    return float((Pq[0, 0, 0]) - (0.872 - (1 - Pq[0, 0, :].sum())))


# ------------------------------------------------------------------------------------------------ the act lift
def act_lift(a, b):
    if a + b <= 0: return 1.0, 0.0
    return b / (a + b), a * b / (a + b)


def regrets_on(I, ps, UP):
    """worst regrets of T and of P on an interval of r (affine utilities: endpoints suffice)"""
    d = [Pn.polyval(e, ps[0]) - UP for e in I]; return max(0.0, -min(d)), max(0.0, max(d))


def lifted_rule(I, ps, UP):
    a, b = regrets_on(I, ps, UP); return act_lift(a, b)[0]


# ------------------------------------------------------------------------------------------------ the adversarial lift
def block_radius(O, other, starts=200, seed=3):
    """min over two-type laws P' with other(P') <= 0 of max(0, 1 - min_x O(x)/P'(x)); also the share eps and the injected Q"""
    o = IJ.as_vec(O); m = O.ndim; rng = np.random.default_rng(seed); best = (0.0, None)
    lawv = lambda p: IJ.as_vec(law(p[0], p[1:1 + m], p[1 + m:1 + 2 * m]))
    for _ in range(starts):
        p0 = np.r_[rng.uniform(0.05, 0.95, 1 + 2 * m), 0.9]
        cons = [{"type": "ineq", "fun": lambda q: o - q[-1] * lawv(q[:-1])}, {"type": "ineq", "fun": lambda q: -other(IJ.as_law(lawv(q[:-1]), m))}]
        r = minimize(lambda q: -q[-1], p0, method="SLSQP", bounds=[(0, 1)] * (2 + 2 * m), constraints=cons, options={"maxiter": 500, "ftol": 1e-13})
        q = r.x; Pp = lawv(q[:-1])
        if r.success and np.all(o - q[-1] * Pp >= -1e-10) and other(IJ.as_law(Pp, m)) <= 1e-10 and q[-1] > best[0]: best = (q[-1], q)
    lam, q = best; Pp = lawv(q[:-1]); eps = 1 - lam; Q = (o - lam * Pp) / max(eps, 1e-300)
    return float(eps), IJ.as_law(Pp, m), IJ.as_law(Q, m)


# ------------------------------------------------------------------------------------------------ dilation
def dilation(P, n=4001):
    mu, M, mS = moments(P); v = SD.v_closed(M); lo, hi = segment(mu, v); ts = np.exp(np.linspace(np.log(lo), np.log(hi), n)); out = {}
    laws = [law(*explanation(mu, v, t)) for t in ts]
    for a, b in itertools.product((0, 1), repeat=2):
        c = np.array([L[a, b, 1] / L[a, b, :].sum() for L in laws]); out[(a, b)] = (float(c.min()), float(c.max()), float(P[a, b, 1] / P[a, b, :].sum()))
    pC = [float(L[:, :, 1].sum()) for L in laws]
    return out, (min(pC), max(pC))


# ------------------------------------------------------------------------------------------------ value of the probe in its size
def evsi(n, W, grid=2001):
    """Bayesian expected value of sample information, r ~ uniform on the hiding segment, rhat ~ N(r, tau^2 / n)"""
    rr = np.linspace(W["R"][0], W["R"][1], grid); w = np.ones_like(rr) / grid; d = Pn.polyval(rr, W["ps"][0]) - W["UP"]
    s = W["tau"] / np.sqrt(n); xs = np.linspace(W["R"][0] - 7 * s, W["R"][1] + 7 * s, 4001)
    L = norm.pdf((xs[:, None] - rr[None, :]) / s); joint = L * w; marg = joint.sum(1); pm = (joint @ d) / np.maximum(marg, 1e-300)
    marg = marg / marg.sum(); return float((marg * np.maximum(pm, 0)).sum() - max((w * d).sum(), 0))


def evpi(W, grid=2001):
    rr = np.linspace(W["R"][0], W["R"][1], grid); d = Pn.polyval(rr, W["ps"][0]) - W["UP"]; return float(np.mean(np.maximum(d, 0)) - max(np.mean(d), 0))


def _rule_probs(n, W, rr, z=1.96):
    s = W["tau"] / np.sqrt(n); pT = norm.cdf((rr - W["rc"]) / s - z); pP = norm.cdf((W["rc"] - rr) / s - z); return pT, pP


def savage_curve(n, W, grid=1601):
    """worst-case regret over the segment of the set rule with the Savage default (T when unsettled); its value is 0.00125 - this"""
    rr = np.linspace(W["R"][0], W["R"][1], grid); d = Pn.polyval(rr, W["ps"][0]) - W["UP"]; pT, pP = _rule_probs(n, W, rr)
    return float(np.max(np.where(d > 0, d * pP, -d * (1 - pP))))


def wald_curve(n, W, grid=1601):
    """worst-case expected utility over the segment of the set rule with the Wald default (P when unsettled)"""
    rr = np.linspace(W["R"][0], W["R"][1], grid); d = Pn.polyval(rr, W["ps"][0]) - W["UP"]; pT, pP = _rule_probs(n, W, rr)
    return float(np.min(W["UP"] + d * pT))


# ------------------------------------------------------------------------------------------------ the blow-up of the merge
def blowup_sd(mu, v, n, reps, rng):
    """sd over replicates of rho^2_hat = ||offdiag Mhat||_F (the lifted coordinate) and of rho_hat = sqrt of it (|v|-scale)"""
    P = law(0.5, mu + v, mu - v); r2 = []
    for _ in range(reps):
        tabs = SD.sample_tables(P, n, rng); M = []
        for (i, j) in SD.WINDOWS[:3]:
            T = tabs[(i, j)]; M.append(T[1, 1] - T[1].sum() * T[:, 1].sum())
        r2.append(np.sqrt(np.sum(np.square(M))))
    r2 = np.array(r2); return float(r2.std()), float(np.sqrt(r2).std()), float(r2.mean())


# ------------------------------------------------------------------------------------------------ backtracking (Gaussian model)
def backtrack(deltas, retro, draws=40000, seed=1, c_b=3.078, c_s=2.394):
    """T rounds, one batch per source per round; source 2 shifted by deltas[t] (in batch sd). Forward-only: a section accepted at an
    earlier round is final, a failed source is refused from then on. Backtracking: a failure revokes all of the source's sections.
    Returns the mean final pooled shift in pooled standard errors and the share of runs in which the source was ever flagged."""
    rng = np.random.default_rng(seed); T = len(deltas); E = rng.standard_normal((draws, T, 3)); E[:, :, 2] += np.asarray(deltas)[None, :]
    acc = np.ones((draws, T, 3), bool); refused = np.zeros(draws, bool)
    for t in range(T):
        sub = E[:, :t + 1, :]; A = acc[:, :t + 1, :]
        smean = np.stack([(sub[:, :, j] * A[:, :, j]).sum(1) / np.maximum(A[:, :, j].sum(1), 1) for j in range(3)], 1); ref = np.median(smean, 1)
        acc[:, t, :] &= np.abs(E[:, t, :] - ref[:, None]) <= c_b * np.sqrt(1 + 1 / (t + 1)); acc[refused, t, 2] = False
        A = acc[:, :t + 1, :]; cnt = A.sum(1); sm = (sub * A).sum(1)
        mj = sm[:, 2] / np.maximum(cnt[:, 2], 1); mo = sm[:, :2].sum(1) / np.maximum(cnt[:, :2].sum(1), 1)
        z = (mj - mo) / np.sqrt(1 / np.maximum(cnt[:, 2], 1) + 1 / np.maximum(cnt[:, :2].sum(1), 1))
        fail = (np.abs(z) > c_s) & (cnt[:, 2] > 0) & ~refused
        if retro: acc[fail, :t + 1, 2] = False
        else: acc[fail, t, 2] = False
        refused |= fail
    K = acc.sum((1, 2)); pooled = (E * acc).sum((1, 2)) / K
    return float((pooled * np.sqrt(3 * T)).mean()), float(refused.mean())


def worst_backtrack(retro, T=8, cs=np.linspace(0.25, 6, 24)):
    best = (-np.inf, None, None)
    for k in range(1, T + 1):
        for c in cs:
            sh, fl = backtrack([c] * k + [0] * (T - k), retro)
            if sh > best[0]: best = (sh, (k, float(c)), fl)
    return best


# ------------------------------------------------------------------------------------------------ rectangularity as gluing
def rect_example():
    """two stages: s in {1, 2} (P(s = 1) = 1/2 known), then y in {0, 1} with P(y = 1 | s) = q_s. C = the segment between
    mu1 = (q1, q2) = (0.8, 0.2) and mu2 = (0.2, 0.8) (correlated ambiguity). rect(C) = {1/2} x [0.2, 0.8]^2, the gluing of C's
    one-step conditional sets (its sheafification on the tree). Bet: pay 0.4, receive 1 if y = 1."""
    joint = lambda q1, q2: np.array([0.5 * (1 - q1), 0.5 * q1, 0.5 * (1 - q2), 0.5 * q2])       # cells (s1y0, s1y1, s2y0, s2y1)
    C = [joint(0.2 + 0.6 * a, 0.8 - 0.6 * a) for a in np.linspace(0, 1, 201)]
    Rq = np.linspace(0.2, 0.8, 61); Rc = [joint(q1, q2) for q1 in Rq for q2 in Rq]
    bet = np.array([-0.4, 0.6, -0.4, 0.6]); osc = 1.0
    V_C = lambda f: min(float(m @ f) for m in C); V_R = lambda f: min(float(m @ f) for m in Rc)
    ex_ante = max(V_C(bet), 0.0)                                       # accept or reject before seeing s
    # dynamic: after s the conditional set of q_s is [0.2, 0.8]; maxmin conditional value of the bet = 0.2 - 0.4 < 0: reject in each branch
    plan = np.array([0.0, 0.0, 0.0, 0.0]) if min(0.2, 0.8) - 0.4 < 0 else bet
    with_info = V_C(plan); voi = with_info - ex_ante
    dH = max(min(0.5 * np.abs(r - m).sum() for m in C) for r in Rc)
    recursive_equals_rect = abs(V_R(bet) - (0.5 * (0.2 - 0.4) + 0.5 * (0.2 - 0.4))) < 1e-12
    return dict(ex_ante=ex_ante, with_info=with_info, voi=voi, dH=dH, bound=-osc * dH, V_C_bet=V_C(bet), V_rect_bet=V_R(bet),
                recursive_equals_rect=recursive_equals_rect)
