"""The decision maker's view (section sec:dmview): what he can see with his budget, and his private incoherence register.
New module; it modifies nothing. World: Study 26's law (outerrelay.W), cover 1 of Studies 27b-27d.
  fisher(reporters, law)          Fisher information of the context tables for the global law, on the tangent space of the simplex
  g_visibility(reporters, law)    (sigma_g, kernel_part): the best standard error of g from those reporters, and the part of grad g that
                                  no reporter sees (> 0 means g is not identified from them: unknowable at any sample size)
  spectrum(cover, N_A)            design-time visibility table: sigma_g and the witness cut-off kappa under each trust assumption
  gaussian_blind_risk(...)        the pre-registered Gaussian model of flip / detection / unseen flip at the final round
  register_entry(...)             one row of the private incoherence register (heterogeneity along g, explanations, attribution)
  fr_angle(p)                     Fisher-Rao (Bloch meridian) angle 2 arcsin sqrt(p); sampling noise is about 1/sqrt(n) everywhere
"""
from __future__ import annotations
import itertools, math
import numpy as np
from scipy.stats import chi2
from . import outerrelay as O
W = O.W; P = W.truth(); NS = 32
M1 = O.ev(lambda v: v["M"] == 1); MY1 = O.ev(lambda v: v["M"] == 1 and v["Y"] == 1); M0Y1 = O.ev(lambda v: v["M"] == 0 and v["Y"] == 1)
def g(p): return (MY1 @ p) / (M1 @ p) - (M0Y1 @ p) / ((1 - M1) @ p)
def grad_g(p):
    a, b, c = MY1 @ p, M1 @ p, M0Y1 @ p
    return MY1 / b - a * M1 / b ** 2 - (M0Y1 / (1 - b) + c * M1 / (1 - b) ** 2)
def _tangent():
    U = np.linalg.svd(np.ones((1, NS)))[2][1:].T; return U            # orthonormal basis of {x : sum x = 0}
TB = _tangent()
def fisher(reporters, law=P):
    I = np.zeros((NS, NS))
    for ctx, n in reporters:
        A = O.ctxmat(ctx); q = A @ law; I += n * A.T @ np.diag(1 / q) @ A
    return TB.T @ I @ TB
def g_visibility(reporters, law=P, tol=1e-9):
    if not reporters: return math.inf, 1.0
    It = fisher(reporters, law); gr = TB.T @ grad_g(law)
    w, V = np.linalg.eigh(It); big = w > tol * w.max()
    ker = V[:, ~big]; kpart = float(np.linalg.norm(ker.T @ gr) / np.linalg.norm(gr))
    if kpart > 1e-6: return math.inf, kpart
    s2 = float(sum((V[:, i] @ gr) ** 2 / w[i] for i in np.where(big)[0])); return math.sqrt(s2), kpart
def se_g(n, law=P):
    pm1 = M1 @ law; py1 = (MY1 @ law) / pm1; py0 = (M0Y1 @ law) / (1 - pm1)
    return math.sqrt(py1 * (1 - py1) / (n * pm1) + py0 * (1 - py0) / (n * (1 - pm1)))
COVER1 = [("M", "C"), ("M", "R"), ("C", "R"), ("M", "C", "Y"), ("M", "H"), ("R", "H", "Y")]
def spectrum(N_A, n_src=6000, z=2.00):
    """sigma_g from the reporters a decision maker can still trust, under four trust assumptions"""
    src = [(c, n_src) for c in COVER1]; anc = [(("M", "R", "Y"), N_A), (("M", "H", "Y"), N_A)]
    rows = {}
    for name, trusted in (("all honest", src + anc), ("S3 lies", [s for s in src if s[0] != ("M", "C", "Y")] + anc),
                          ("S3 and one anchor lie", [s for s in src if s[0] != ("M", "C", "Y")] + anc[1:]),
                          ("S3 and both anchors lie", [s for s in src if s[0] != ("M", "C", "Y")]),
                          ("local layer only", src)):
        sg, kp = g_visibility(trusted)
        rows[name] = dict(sigma_g=sg, kernel_part=kp, unknowable=bool(math.isinf(sg)))
    se3, seA = se_g(n_src), se_g(N_A)
    rows["kappa (witness cut-off, S3 vs an anchor)"] = z * math.sqrt(2) * math.sqrt(se3 ** 2 + seA ** 2)
    return rows
def gaussian_blind_risk(delta, N_A, rule, n_src=6000, tau_gap=0.04, z=2.00, nsim=200000, seed=27501):
    """final round, full shift: S3 reports g* + delta + e3; anchors report g* + e_A; witness pairs use changes from round 0.
    rule: 'R1' source point, 'R2' source cautious, 'R3' pooled cautious. Returns P(flip), P(detect), P(flip and not detect)."""
    rng = np.random.default_rng(seed); se3, seA = se_g(n_src), se_g(N_A)
    e3, e30 = rng.normal(0, se3, nsim), rng.normal(0, se3, nsim); eA = rng.normal(0, seA, (2, nsim)); eA0 = rng.normal(0, seA, (2, nsim))
    g3 = delta + e3
    if rule == "R1": flip = g3 >= tau_gap
    elif rule == "R2": flip = g3 - tau_gap > 2 * se3
    else:
        w = np.array([1 / se3 ** 2, 1 / seA ** 2, 1 / seA ** 2]); gp = (w[0] * g3 + w[1] * eA[0] + w[2] * eA[1]) / w.sum()
        flip = gp - tau_gap > 2 / math.sqrt(w.sum())
    kap = z * math.sqrt(2) * math.sqrt(se3 ** 2 + seA ** 2)
    det = np.zeros(nsim, bool)
    for j in range(2): det |= (delta + e3 - e30) - (eA[j] - eA0[j]) > kap
    return float(flip.mean()), float(det.mean()), float((flip & ~det).mean())
def register_entry(gvals, ses, alpha=0.01):
    """gvals, ses: dicts reporter -> (g estimate, standard error) for the reporters that see g. Heterogeneity Q along g and the
    explanations: subsets whose removal leaves a homogeneous remainder (Q below the chi-square cut at level alpha)."""
    names = sorted(gvals)
    def Q(sub):
        if len(sub) < 2: return 0.0, 0
        w = np.array([1 / ses[k] ** 2 for k in sub]); x = np.array([gvals[k] for k in sub]); m = (w @ x) / w.sum()
        return float(w @ (x - m) ** 2), len(sub) - 1
    q, df = Q(names); incoherent = df > 0 and q > chi2.ppf(1 - alpha, df)
    expl = []
    if incoherent:
        for r in range(1, len(names)):
            for S in itertools.combinations(names, r):
                if any(set(e) <= set(S) for e in expl): continue
                rest = [k for k in names if k not in S]; q2, d2 = Q(rest)
                if d2 == 0 or q2 <= chi2.ppf(1 - alpha, d2): expl.append(S)
    incl = sorted(set.intersection(*[set(e) for e in expl])) if expl else []
    mincard = min((len(e) for e in expl), default=0); card = [e for e in expl if len(e) == mincard]
    card_named = sorted(set.intersection(*[set(e) for e in card])) if card else []
    return dict(Q=q, df=df, incoherent=bool(incoherent), explanations=[list(e) for e in expl], named_inclusion=incl, named_cardinality=card_named)
def fr_angle(p): return 2 * math.asin(math.sqrt(min(max(p, 0.0), 1.0)))
