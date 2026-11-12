"""Decision-maker memory: private relations accumulated over many decision episodes (companion note, 'Decision-maker memory'). New
module; it modifies nothing.

Episodes. In each episode e the decision maker estimates a quantity theta_e (threshold 1/2) from k sources. Every source gives an initial
report and revises it once; the order of the k revisions is a random permutation (honest sources are exchangeable). Honest reports are
theta_e + c_e + noise, where c_e is a common shock shared by all sources (the 'tail' of the de Finetti picture: what all honest sources
have in common) and the noise is independent N(0, SIGMA^2). The decision maker never sees theta_e or c_e.
Scenarios (deceptive sources are 1 = B and 2 = C, 0-based indices 1 and 2):
  honest       all honest; common shocks of size 0.05
  tilt         B adds -BETA to its revised report in every episode (towards decision X = 'reject')
  ordered      B and C both tilt; C revises right after B with probability ORDER_P, otherwise at a random position
  braidable    B and C both tilt; their revision times are random (the coalition's order is exchangeable)
  follower     no tilt; C revises right after B with probability ORDER_P and copies a fraction RHO of B's deviation
  shocks       all honest; common shocks of size 0.2 (four times larger)
Memory (built only from the reports). After each episode, with m_e the median of the revised reports (the decision maker's estimate of the
common part theta_e + c_e) and d_s = revised report of s minus m_e:
  tilt walk T_s      += sign(-d_s)                   (+1 when s pushed towards X)
  co-revision walk C_uv += sign(d_u) sign(d_v) + 1/(k-1)   (centred: half the deviations are positive in every episode)
  order walk O_uv    += +1 if u revised before v, else -1   (the signed area of the pair's count path, prop:ordrev(iii))
Under the honest model each increment has mean 0 (exchangeability of honest sources given the common part), so each walk is a mean-zero
random walk growing like sqrt(n); a source or pair that breaks the symmetry adds drift, growing like n.
Alarms: |walk| / sqrt(n) above a boundary, from episode N0 on; boundaries are calibrated per family (tilt, co-revision, order) on honest
simulations so that the probability of any alarm in E episodes is ALPHA. A primed memory watches only a declared list of relations
(prior suspicions), with its own, lower boundary.
"""
from __future__ import annotations
import itertools
import numpy as np

K = 6; SIGMA = 0.05; BETA = 0.012; ORDER_P = 0.7; RHO = 0.6; N0 = 10; ALPHA = 0.05
PAIRS = list(itertools.combinations(range(K), 2))
SCENARIOS = ["honest", "tilt", "ordered", "braidable", "follower", "shocks"]


def simulate(scenario, E, seed):
    """increments per episode: tilt (E, K), co-revision (E, P), order (E, P)"""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.3, 0.7, E); c = rng.normal(0, 0.2 if scenario == "shocks" else 0.05, E)
    noise = rng.normal(0, SIGMA, (E, K)); rev = theta[:, None] + c[:, None] + noise
    pos = np.array([rng.permutation(K) for _ in range(E)])               # pos[e, s] = position of s's revision
    if scenario in ("tilt", "ordered", "braidable"): rev[:, 1] -= BETA
    if scenario in ("ordered", "braidable"): rev[:, 2] -= BETA
    if scenario in ("ordered", "follower"):
        for e in range(E):
            if rng.random() < ORDER_P:                                  # C moves to the slot right after B
                order = list(np.argsort(pos[e])); order.remove(2); order.insert(order.index(1) + 1, 2)
                pos[e, order] = np.arange(K)
    if scenario == "follower":
        m0 = np.median(rev, axis=1); follow = pos[:, 2] == pos[:, 1] + 1
        rev[follow, 2] += RHO * (rev[follow, 1] - m0[follow])
    m = np.median(rev, axis=1); d = rev - m[:, None]
    tilt = np.sign(-d)
    corev = np.stack([np.sign(d[:, u]) * np.sign(d[:, v]) for u, v in PAIRS], axis=1) + 1.0 / (K - 1)   # centred: with an even k and the
    # median, exactly half the deviations are positive, so for exchangeable sources E[sign(d_u) sign(d_v)] = -1/(k-1)
    order = np.stack([np.where(pos[:, u] < pos[:, v], 1.0, -1.0) for u, v in PAIRS], axis=1)
    return dict(tilt=tilt, corev=corev, order=order)


def normalised_paths(inc):
    S = np.cumsum(inc, axis=0); n = np.arange(1, len(inc) + 1)[:, None]; return np.abs(S) / np.sqrt(n)


def calibrate(E, seeds, watch=None):
    """per family: boundary b with P(max over n >= N0 and watched relations of |walk|/sqrt(n) >= b) = ALPHA on honest histories"""
    mx = {f: [] for f in ("tilt", "corev", "order")}
    for sd in seeds:
        inc = simulate("honest", E, sd)
        for f in mx:
            Z = normalised_paths(inc[f])[N0 - 1:]
            if watch and f in watch: Z = Z[:, watch[f]]
            mx[f].append(Z.max())
    return {f: float(np.quantile(v, 1 - ALPHA)) for f, v in mx.items()}


class Memory:
    """the decision maker's private relations; update(increments of one episode) after each episode"""

    def __init__(self, bounds, watch=None):
        self.b = bounds; self.watch = watch or {}; self.S = None; self.n = 0; self.alarms = {}

    def update(self, inc_e):
        if self.S is None: self.S = {f: np.zeros_like(v) for f, v in inc_e.items()}
        self.n += 1
        for f, v in inc_e.items():
            self.S[f] += v
            if self.n < N0 or f in self.alarms: continue
            z = np.abs(self.S[f]) / np.sqrt(self.n)
            idx = self.watch.get(f, list(range(len(z))))
            hit = [i for i in idx if z[i] >= self.b[f]]
            if hit:
                j = max(hit, key=lambda i: z[i]); self.alarms[f] = (self.n, j, float(np.sign(self.S[f][j])))


def run(scenario, E, seed, bounds, watch=None):
    inc = simulate(scenario, E, seed); M = Memory(bounds, watch)
    for e in range(E): M.update({f: inc[f][e] for f in inc})
    return M.alarms, {f: M.S[f].copy() for f in M.S}
