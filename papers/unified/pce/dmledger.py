"""Decision-maker memory with an event ledger, a conditional lead-lag test, an anchor walk and forgetting (companion note, 'The memory
at work'). New module; pce/dmmemory.py is unchanged.

Episodes. Each episode e has a true value v_e = theta_e + c_e (theta_e ~ U(0.3, 0.7); c_e ~ N(0, 0.05), a change in the world that every
honest source can see). k = 6 sources give an initial report r0 = v + N(0, 0.08) and revise once, in a random order (a permutation). An
honest revision moves 60% of the way to a fresh measurement v + N(0, 0.04). In 25% of episodes an independent inspection reads v with
error N(0, 0.01) (the anchor). Decision X is 'low'; the decision maker never sees v.
Matched conditions (B = source 1, C = source 2; 'C follows B': in 70% of episodes C revises right after B and then moves halfway towards
B's revised report):
  honest            all honest
  bias              B's revision is shifted by -0.03 (towards X), independently of anyone
  news              all honest; in half the episodes the world moves by -0.06 and every source sees it (shared news, towards X)
  cascade           all honest; B measures more precisely (error 0.02); C follows B (honest social learning from a good source)
  lockstep          B and C both shift by -0.03 and revise next to each other, in random order
  staggered         B shifts by -0.03; C follows B (so C inherits B's lean); the coalition's pattern of 'cascade' with a bad leader
  late              'staggered', but only from episode 600 on
  exposure          all honest; in half the episodes the world moves by -0.06, and each source learns of it with its own probability
                    (B and C 0.9, the others 0.3): honest sources whose access to shared news differs (an imperfectly observed common
                    shock)
Ledger. One record per revision event: (episode, position in the order, source, initial report, revised report, anchor if any). Every
statistic below is recomputed from the ledger alone.
Walks (increments per episode; m = median of revised reports; d_s = revised - m; D_s = revised - initial):
  tilt_s         sign(-d_s)                                           individual lean towards X
  corev_uv       sign(d_u) sign(d_v) + 1/(k - 1)                      co-movement (centred)
  order_uv       +1 if u revised before v, else -1                    habitual order
  leadlag_uv     sign(D_v * (rev_u - init_v)) * (+1 if u before v else -1)
                 does v move towards u's revised report more when u revised first? (a conditional, Granger-type test: under an
                 exchangeable order independent of content it has mean 0, even though honest revisions all move towards the truth)
  anchor_s       sign(gain_s - median gain), gain = |init - anchor| - |revised - anchor|, anchored episodes only
                 did the source's revision help more than the typical revision?
Discounted walks: W_n = lam W_{n-1} + increment, normalised by sqrt(sum lam^(2j)); lam = 0.99.
Boundaries: per family, calibrated on honest histories (probability of any alarm in E episodes = 5%).
"""
from __future__ import annotations
import itertools
import numpy as np

K = 6; S0 = 0.08; S1 = 0.04; W = 0.6; BETA = 0.03; NEWS = 0.06; FOLLOW = 0.7; ANCHOR_P = 0.25; LAM = 0.99; N0 = 10; ALPHA = 0.05
PAIRS = list(itertools.combinations(range(K), 2)); OPAIRS = [(u, v) for u in range(K) for v in range(K) if u != v]
SCENARIOS = ["honest", "bias", "news", "exposure", "cascade", "lockstep", "staggered", "late"]
EXPOSE = np.array([0.3, 0.9, 0.9, 0.3, 0.3, 0.3])
FAMILIES = ["tilt", "corev", "order", "leadlag", "anchor"]


def episode(rng, scen, e):
    theta = rng.uniform(0.3, 0.7); c = rng.normal(0, 0.05)
    if scen == "news" and rng.random() < 0.5: c -= NEWS
    news = -NEWS if (scen == "exposure" and rng.random() < 0.5) else 0.0
    v = theta + c + news
    r0 = v - news + rng.normal(0, S0, K); meas = v + rng.normal(0, S1, K)
    if scen == "exposure": meas = meas - news * (rng.random(K) >= EXPOSE)   # sources that did not learn of the news miss it
    if scen == "cascade": meas[1] = v + rng.normal(0, 0.02)
    order = list(rng.permutation(K))
    active = scen in ("staggered", "lockstep", "bias") or (scen == "late" and e >= 600)
    follow = (scen == "cascade" or (scen in ("staggered", "late") and active)) and rng.random() < FOLLOW
    if follow: order.remove(2); order.insert(order.index(1) + 1, 2)
    if scen == "lockstep":
        order.remove(2); i = order.index(1)
        order.insert(i + 1 if rng.random() < 0.5 else i, 2)
    r1 = r0.copy()
    for s in order:                                                     # revisions in order
        r1[s] = (1 - W) * r0[s] + W * meas[s]
        if s == 1 and active and scen != "cascade": r1[s] -= BETA
        if s == 2 and scen == "lockstep": r1[s] -= BETA
        if s == 2 and follow: r1[s] = 0.5 * r1[s] + 0.5 * r1[1]
    anchor = v + rng.normal(0, 0.01) if rng.random() < ANCHOR_P else None
    return [(e, order.index(s), s, float(r0[s]), float(r1[s]), anchor) for s in range(K)]


def ledger(scen, E, seed):
    rng = np.random.default_rng(seed); out = []
    for e in range(E): out += episode(rng, scen, e)
    return out


def increments(L, E):
    """from the ledger alone: per family an (E, n) array of increments (NaN where undefined)"""
    inc = {f: np.full((E, K if f in ("tilt", "anchor") else (len(OPAIRS) if f == "leadlag" else len(PAIRS))), np.nan) for f in FAMILIES}
    by = {}
    for rec in L: by.setdefault(rec[0], []).append(rec)
    for e, recs in by.items():
        recs = sorted(recs, key=lambda r: r[2]); pos = np.array([r[1] for r in recs]); r0 = np.array([r[3] for r in recs]); r1 = np.array([r[4] for r in recs])
        d = r1 - np.median(r1); D = r1 - r0
        inc["tilt"][e] = np.sign(-d)
        inc["corev"][e] = [np.sign(d[u]) * np.sign(d[v]) + 1 / (K - 1) for u, v in PAIRS]
        inc["order"][e] = [1.0 if pos[u] < pos[v] else -1.0 for u, v in PAIRS]
        inc["leadlag"][e] = [np.sign(D[v] * (r1[u] - r0[v])) * (1.0 if pos[u] < pos[v] else -1.0) for u, v in OPAIRS]
        a = recs[0][5]
        if a is not None:
            gain = np.abs(r0 - a) - np.abs(r1 - a); inc["anchor"][e] = np.sign(gain - np.median(gain))
    return inc


def walks(inc, lam=None):
    """normalised walks |S|/sd per episode; lam = None for cumulative, else discounted"""
    out = {}
    for f, X in inc.items():
        Z = np.nan_to_num(X); cnt = (~np.isnan(X)).astype(float)
        if lam is None:
            S = np.cumsum(Z, axis=0); n = np.maximum(np.cumsum(cnt, axis=0), 1); out[f] = S / np.sqrt(n)
        else:
            S = np.zeros_like(Z); V = np.zeros_like(Z); s = np.zeros(Z.shape[1]); vv = np.zeros(Z.shape[1])
            for t in range(len(Z)):
                s = lam * s + Z[t]; vv = lam ** 2 * vv + cnt[t]; S[t] = s; V[t] = vv
            out[f] = S / np.sqrt(np.maximum(V, 1))
    return out


def calibrate(E, seeds, lam=None):
    mx = {f: [] for f in FAMILIES}
    for sd in seeds:
        Wk = walks(increments(ledger("honest", E, sd), E), lam)
        for f in FAMILIES: mx[f].append(np.abs(Wk[f][N0 - 1:]).max())
    return {f: float(np.quantile(v, 1 - ALPHA)) for f, v in mx.items()}


def alarms(Wk, bounds):
    """first alarm per family: (episode, index, sign)"""
    out = {}
    for f in FAMILIES:
        Z = Wk[f][N0 - 1:]; hit = np.argwhere(np.abs(Z) >= bounds[f])
        if len(hit):
            t = hit[:, 0].min(); j = max((h[1] for h in hit if h[0] == t), key=lambda j: abs(Z[t, j]))
            out[f] = (int(t + N0), int(j), float(np.sign(Z[t, j])))
    return out


def label(f, j):
    if f in ("tilt", "anchor"): return f"S{j + 1}"
    if f == "leadlag": u, v = OPAIRS[j]; return f"S{u + 1}->S{v + 1}"
    u, v = PAIRS[j]; return f"S{u + 1}-S{v + 1}"
