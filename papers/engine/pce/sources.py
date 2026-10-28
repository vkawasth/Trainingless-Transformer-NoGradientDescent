"""Sources, batches and a trusted source that is sometimes adversarial (lab paper, Part II, Theorem J and Study 13).

Three sources each send K_C batches. A batch is four windows (the three pair tables and the triple table), n records each, sampled
independently (as in Study 11b). The decision maker sees the label (source, batch) of every table and nothing else about provenance.
One source is compromised in some of its batches: there a share eps_b of the records is drawn from Q = P + s chi (the hidden channel of
Theorem F), which leaves every single and pair table unchanged and moves the triple moment m_T by eps_b s.

Sections and gluing. Every batch's triple table is a section over the triple context; all of them must glue (one population). The
hidden channel only moves m_T, so gluing of the triple sections is tested on m_T, on two covers, fine first:
  batch_filter     each batch's m_T against a robust reference, the median of the three source means (with one compromised source the
                   median lies between the two genuine ones, so a strongly shifted source cannot drag genuine batches out: no
                   swamping); triple sections with |z| > c_b are dropped
  source_filter    then each source's pooled m_T (kept batches) against the other sources; if the largest |z| exceeds c_s, that
                   source's remaining triple sections are dropped. With two sources left a failure cannot be localised, so at most one
                   source is dropped, and a source whose batches were all dropped is not tested.
Pair sections are always kept: they glue, since the hidden channel never touches them.

Rules compared (all on the same data):  naive (pool everything), glue (batch filter, then source filter), drop_C_triple (oracle that
knows the compromised source and drops its triple sections), write_off_C (drops every table of the compromised source).

Outcome of a rule: 'right' (interval [rhat +- h] cap R settles on the oracle act), 'wrong' (the other act is better on the whole
interval: a tilt), 'open' (the interval straddles the switch: a delay), 'flag' (rhat outside the segment: the walls detect).
"""
from __future__ import annotations
import itertools
import numpy as np
from numpy.polynomial import polynomial as Pn
from scipy.stats import norm
from . import segment_decision as SD
from . import injection as IJ

WINDOWS = SD.WINDOWS
XS = list(itertools.product((0, 1), repeat=3))
OTHER = ((1, 2), (0, 2), (0, 1))


def marg(P, S): return P.sum(axis=tuple(j for j in range(3) if j not in S))


def hidden_Q(P, s): return IJ.as_law(IJ.as_vec(P) + s * IJ.chi(3, (0, 1, 2)), 3)


# ------------------------------------------------------------------------------------------------------------ batches
def sample_batch(P, Q, eps, n, rng, rho=1):
    """four windows: the triple window has n records, each pair window rho n (pair tables come from large registries, the triple window
    is the probe); a share eps of every window's records is drawn from Q (whole-record injection, rounded), the rest from P"""
    tabs = {}
    for S in WINDOWS:
        nS = n if len(S) == 3 else rho * n; k = int(round(eps * nS))
        TP = marg(P, S).ravel(); c = rng.multinomial(nS - k, TP / TP.sum())
        if k:
            TQ = marg(Q, S).ravel(); c = c + rng.multinomial(k, TQ / TQ.sum())
        tabs[S] = (c / nS).reshape((2,) * len(S))
    return tabs


def mT_stats(T3):
    """centred triple moment of a triple table (own means) and the plug-in variance of its influence function (per record)"""
    m3 = [T3.sum(axis=tuple(j for j in range(3) if j != i))[1] for i in range(3)]
    mS = sum(T3[x] * np.prod([x[i] - m3[i] for i in range(3)]) for x in XS)
    Mt = {OTHER[i]: sum(T3[x] * (x[OTHER[i][0]] - m3[OTHER[i][0]]) * (x[OTHER[i][1]] - m3[OTHER[i][1]]) for x in XS) for i in range(3)}
    IF = lambda x: np.prod([x[i] - m3[i] for i in range(3)]) - mS - sum(Mt[OTHER[i]] * (x[i] - m3[i]) for i in range(3))
    return float(mS), float(sum(T3[x] * IF(x) ** 2 for x in XS))


def components(tabs):
    """from_tables, with the two parts of tau^2 kept apart: (mu, v, rhat, a2 = var_m / v_T^2, b2 = var_logv, R); None if no explanation"""
    e = SD.from_tables(tabs)
    if e is None: return None
    mu, v, rhat, _, R = e; vS = float(np.prod(v)); _, var_m = mT_stats(tabs[(0, 1, 2)]); b2 = 0.0
    for (i, j) in WINDOWS[:3]:
        T = tabs[(i, j)]; pi, pj = T[1].sum(), T[:, 1].sum(); M = T[1, 1] - pi * pj
        b2 += 0.25 * sum(T[a, b] * ((a - pi) * (b - pj) - M) ** 2 for a in (0, 1) for b in (0, 1)) / M ** 2
    return mu, v, rhat, var_m / vS ** 2, b2, R


def decide(pair_pool, trip_pool, n_pair, n_trip, acts, u0, a0, z=1.96):
    """the regulator's post-probe decision on pooled sections; pair windows carry n_pair records, the triple window n_trip"""
    tabs = dict(pair_pool); tabs[(0, 1, 2)] = trip_pool; c = components(tabs)
    if c is None: return "flag", np.nan
    mu, v, rhat, a2, b2, R = c
    if not (R[0] <= rhat <= R[1]): return "flag", rhat
    h = z * np.sqrt(a2 / n_trip + rhat ** 2 * b2 / n_pair); ps = [SD.U_poly(mu, v, u, range(3)) for u in acts]
    I = SD.clip((rhat - h, rhat + h), R)
    if SD.set_rule(ps, I, u0) == a0: return "right", rhat
    other = [k for k in range(len(ps)) if k != a0]
    worse = all(max(Pn.polyval(e, ps[a0]) - Pn.polyval(e, ps[b]) for e in I) < 0 for b in other)
    return ("wrong" if worse else "open"), rhat


# ------------------------------------------------------------------------------------------------------------ filters
def batch_filter(m, src, var_pool, n, c_b):
    """gluing test of every batch's m_T against the median of the source means; returns the keep mask and the z scores"""
    m = np.asarray(m, float); src = np.asarray(src); labs = sorted(set(src.tolist()))
    ref = float(np.median([m[src == j].mean() for j in labs])); kc = min((src == j).sum() for j in labs)
    zs = (m - ref) / np.sqrt(var_pool / n * (1 + 1 / kc)); return np.abs(zs) <= c_b, zs


def source_filter(m, src, var_pool, n, c_s, keep):
    """gluing test of each source's pooled m_T against the other sources (kept batches only); drops the worst source if |z| > c_s"""
    m = np.asarray(m, float); src = np.asarray(src); labs = sorted(set(src[keep].tolist())); zs = {}
    if len(labs) < 3: return keep.copy(), zs, None
    for j in labs:
        a = keep & (src == j); b = keep & (src != j)
        zs[j] = (m[a].mean() - m[b].mean()) / np.sqrt(var_pool / n * (1 / a.sum() + 1 / b.sum()))
    worst = max(zs, key=lambda j: abs(zs[j]))
    if abs(zs[worst]) > c_s: return keep & (src != worst), zs, worst
    return keep.copy(), zs, None


def pool(batches, mask, S):
    sel = [b[S] for b, k in zip(batches, mask) if k]; return sum(sel) / len(sel)


def run_rules(batches, src, n, acts, u0, a0, c_b, c_s, bad="C", z=1.96, rho=1):
    """all four rules on one draw; returns {rule: (outcome, rhat)} and the filter's diagnostics"""
    K = len(batches); ms = [mT_stats(b[(0, 1, 2)]) for b in batches]; m = np.array([x[0] for x in ms]); var_pool = float(np.mean([x[1] for x in ms]))
    src = np.asarray(src); allk = np.ones(K, bool); pairs = {S: pool(batches, allk, S) for S in WINDOWS[:3]}
    np_ = rho * n
    out = {"naive": decide(pairs, pool(batches, allk, (0, 1, 2)), np_ * K, n * K, acts, u0, a0, z)}
    kb, zb = batch_filter(m, src, var_pool, n, c_b)                           # fine cover: batches against the robust reference
    ks, zs, dropped = source_filter(m, src, var_pool, n, c_s, kb)              # coarse cover: sources, on the kept batches
    out["glue"] = decide(pairs, pool(batches, ks, (0, 1, 2)), np_ * K, n * int(ks.sum()), acts, u0, a0, z)
    kc = src != bad
    out["drop_C_triple"] = decide(pairs, pool(batches, kc, (0, 1, 2)), np_ * K, n * int(kc.sum()), acts, u0, a0, z)
    pc = {S: pool(batches, kc, S) for S in WINDOWS[:3]}
    out["write_off_C"] = decide(pc, pool(batches, kc, (0, 1, 2)), np_ * int(kc.sum()), n * int(kc.sum()), acts, u0, a0, z)
    diag = {"batch_dropped": int((~kb).sum()), "bad_batches_dropped": int(((~kb) & (src == bad)).sum()), "source_dropped": dropped,
            "z_source": {k: float(v) for k, v in zs.items()}, "z_batch_max": float(np.nanmax(np.abs(zb))), "kb": kb, "ks": ks}
    return out, diag


# ------------------------------------------------------------------------------------------------------------ the adversary
def tuned_shift(sigma_m, n, K_C, K, a):
    """mean shift D of the compromised source's m_T that puts the expected source z at a (other sources genuine, equal batch sizes)"""
    return a * sigma_m * np.sqrt(1 / (K_C * n) + 1 / ((K - K_C) * n))


def band(c_s, pi, q=0.0):
    """Theorem J: largest shift of the pooled m_T, in pooled standard errors, that a source of share pi can make while its expected
    gluing statistic stays at c_s - q (q = Phi^-1(1 - beta) for exposure risk beta)"""
    return (c_s - q) * np.sqrt(pi / (1 - pi))


def sigma_m(P):
    mu = np.array([marg(P, (i,))[1] for i in range(3)]); return np.sqrt(mT_stats(marg(P, (0, 1, 2)))[1])


# ------------------------------------------------------------------------------------------------------------ the delta-method model
def _ifs(P, F, h=1e-6):
    """influence functions of a functional F(tabs) of the four window tables, window by window"""
    base = SD.exact_tables(P); out = {}
    for S in WINDOWS:
        T = base[S]; rows = {}
        for x in itertools.product((0, 1), repeat=len(S)):
            D = np.zeros_like(T); D[x] = 1.0
            tp = dict(base); tp[S] = (1 - h) * T + h * D; tm = dict(base); tm[S] = (1 + h) * T - h * D
            rows[x] = (F(tp) - F(tm)) / (2 * h)
        mean = sum(T[x] * rows[x] for x in rows); out[S] = {x: rows[x] - mean for x in rows}
    return base, out


def gauss_model(P, acts, u0, a0):
    """per-record variances and covariances of the two statistics the rules use: G = s0 (rhat - rc_hat) (signed distance to the switch,
    positive on the oracle side) and rhat; split into the triple window part and the pair windows part"""
    mu, v, r0, a2, b2, R = components(SD.exact_tables(P))
    ps = [SD.U_poly(mu, v, u, range(3)) for u in acts]; b = 1 - a0
    rc = (ps[b][0] - ps[a0][0]) / (ps[a0][1] - (ps[b][1] if len(ps[b]) > 1 else 0.0)); s0 = np.sign(r0 - rc)
    def G(t):
        c = components(t); mu_, v_, rh, _, _, _ = c; q = [SD.U_poly(mu_, v_, u, range(3)) for u in acts]
        rc_ = (q[b][0] - q[a0][0]) / (q[a0][1] - (q[b][1] if len(q[b]) > 1 else 0.0)); return s0 * (rh - rc_)
    def Rh(t): return components(t)[2]
    base, IG = _ifs(P, G); _, IR = _ifs(P, Rh)
    cov = lambda A, B, Ss: sum(sum(base[S][x] * A[S][x] * B[S][x] for x in A[S]) for S in Ss)
    trip = [(0, 1, 2)]; pairs = list(WINDOWS[:3])
    return dict(r0=r0, rc=rc, s0=s0, R=R, d=abs(r0 - rc), a2=a2, b2=b2, sm=sigma_m(P), vT=float(np.prod(v)),
                GG_t=cov(IG, IG, trip), GG_p=cov(IG, IG, pairs), RR_p=cov(IR, IR, pairs), GR_p=cov(IG, IR, pairs), GR_t=cov(IG, IR, trip))


def predict(gm, n, K_C, shifts, c_b, c_s, z=1.96, draws=200000, seed=7, rules=("naive", "glue", "drop_C_triple", "write_off_C"), rho=1):
    """outcome probabilities of the four rules under the delta-method Gaussian model. shifts: per-batch shift of m_T for the K_C
    batches of the compromised source (source index 2); sources 0 and 1 are genuine; every source has K_C batches of n records."""
    rng = np.random.default_rng(seed); K = 3 * K_C; sm = gm["sm"]; vT = gm["vT"]; src = np.repeat([0, 1, 2], K_C)
    sh = np.concatenate([np.zeros(2 * K_C), np.asarray(shifts, float)])
    E = rng.standard_normal((draws, K)) * sm / np.sqrt(n) + sh                 # per-batch m_T errors (with the adversary's shift)
    # pair parts of G and rhat, per record, for all K batches and for the K - K_C batches without the compromised source
    Cp = np.array([[gm["GG_p"], gm["GR_p"]], [gm["GR_p"], gm["RR_p"]]]); L = np.linalg.cholesky(Cp + 1e-18 * np.eye(2))
    Zp = rng.standard_normal((draws, 2)) @ L.T; Zq = rng.standard_normal((draws, 2)) @ L.T
    out = {}
    # triple part of G and of rhat: both equal (m_T error) / v_T up to sign; G's coefficient is s0 / v_T, rhat's is 1 / v_T
    cg = gm["s0"] / vT; cr = 1.0 / vT
    kb = np.ones((draws, K), bool)
    if "glue" in rules:
        se = sm / np.sqrt(n); smean = np.stack([E[:, src == j].mean(1) for j in range(3)], 1); ref = np.median(smean, 1)
        kb = np.abs(E - ref[:, None]) <= c_b * se * np.sqrt(1 + 1 / K_C)        # fine cover: batches against the median of source means
        cnt = np.stack([(kb & (src == j)).sum(1) for j in range(3)], 1); sums = np.stack([(E * (kb & (src == j))).sum(1) for j in range(3)], 1)
        zs = np.zeros((draws, 3)); full = (cnt > 0).all(1)
        for j in range(3):                                                        # coarse cover: each source against the other two
            o = [k for k in range(3) if k != j]; cj = np.maximum(cnt[:, j], 1); co = np.maximum(cnt[:, o].sum(1), 1)
            zs[:, j] = (sums[:, j] / cj - sums[:, o].sum(1) / co) / (se * np.sqrt(1 / cj + 1 / co))
        w = np.argmax(np.abs(zs), 1); drop = full & (np.abs(zs[np.arange(draws), w]) > c_s)
        ks = kb.copy(); ks[drop] &= (src[None, :] != w[drop, None])
        out["glue_source_flag"] = {j: float(np.mean(drop & (w == j))) for j in range(3)}
        out["glue_bad_batches_dropped"] = float(np.mean((~kb[:, src == 2]).sum(1)))
        adv = np.where(sh != 0)[0]
        out["glue_adv_batches_dropped"] = float(np.mean(~kb[:, adv])) if len(adv) else 0.0
        out["glue_adv_removed"] = float(np.mean(~ks[:, adv])) if len(adv) else 0.0
    def res(mask, n_pair_rec, Zpair):
        kk = mask.sum(1); mbar = (E * mask).sum(1) / kk; n_trip = n * kk
        Gv = gm["s0"] * (gm["r0"] - gm["rc"]) + cg * mbar + Zpair[:, 0] / np.sqrt(n_pair_rec)
        Rv = gm["r0"] + cr * mbar + Zpair[:, 1] / np.sqrt(n_pair_rec)
        h = z * np.sqrt(gm["a2"] / n_trip + gm["r0"] ** 2 * gm["b2"] / n_pair_rec)
        flag = (Rv < gm["R"][0]) | (Rv > gm["R"][1])
        return {"right": float(np.mean(~flag & (Gv >= h))), "wrong": float(np.mean(~flag & (Gv <= -h))),
                "open": float(np.mean(~flag & (np.abs(Gv) < h))), "flag": float(np.mean(flag)),
                "shift_harm_se": float(np.mean(cr * mbar) / np.sqrt(gm["a2"] / (n * K)))}
    allk = np.ones((draws, K), bool); kc = np.broadcast_to(src != 2, (draws, K))
    if "naive" in rules: out["naive"] = res(allk, rho * n * K, Zp)
    if "glue" in rules: out["glue"] = res(ks, rho * n * K, Zp)
    if "drop_C_triple" in rules: out["drop_C_triple"] = res(kc, rho * n * K, Zp)
    if "write_off_C" in rules:
        # writing off C removes C's pair windows too: the pair part is that of the 2 K_C genuine batches
        out["write_off_C"] = res(kc, rho * n * (K - K_C), Zq)
    return out


def delay_factor(B, z=1.96): return ((z + B) / z) ** 2
