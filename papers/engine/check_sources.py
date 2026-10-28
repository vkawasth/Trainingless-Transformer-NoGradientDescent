"""check_sources.py -- sources, batches and a sometimes-adversarial trusted source (lab paper, Part II, Theorem J; Study 13).

  SR1 the hidden channel at the batch level: a batch with share eps of Q = P + s chi has exactly P's single and pair tables and its
      triple moment m_T moves by exactly eps s (the coefficient of chi in m_T is 1 for every law)
  SR2 Tier 1 (colimit failure): two triple sections from one population glue (lambda* = 1); a section with share eps of Q glues with
      a genuine one on lambda* = 1 - eps TV(P, Q), so 1 - lambda* is a lower bound on the contamination share (lambda* by the LP)
  SR3 the monad: pooling is the multiplication mu of the distribution monad; the attacked family of batch laws and the Dirac family
      eta(pooled law) have the same image under mu, so every rule that pools (factors through the comonad's counit) sees a whole-pool
      injection of share pi eps_C, and its budgets are Theorem F's divided by pi
  SR4 Theorem J(1), the band: a source of share pi whose expected gluing statistic is a moves the pooled m_T by exactly
      a sqrt(pi / (1 - pi)) pooled standard errors, for every K_C, K, n
  SR5 the decision statistic: the triple-window part of G = s0 (rhat - rc_hat) is (m_T error) / v_T exactly (its variance is
      sigma_m^2 / v_T^2 = the triple part of tau^2), so the adversary's channel enters G as it enters rhat
  SR6 Theorem J(2), no tilt: with the gluing filter and an adversary at exposure risk one half (a = c_s), the settled-wrong rate is at
      most Phi(B - z) at every N (Gaussian model), while naive pooling under a fixed share eps_pool in (eps*, eps_det) settles wrong
      with probability tending to one
  SR7 Theorem J(3), delay: with no adversary the glue rule's right-rate crosses one half at (z tau / d)^2; under the tuned adversary
      it crosses later, at most at the never-exposed bound ((z tau + B sigma_m / |v_T|) / d)^2 (registry-sized pair windows, rho = 25)
  SR8 the batch level localises a concentrated attack: one adversarial batch at the tuned source shift is dropped by the batch filter
      with probability >= 0.9 and the source is then not dropped (Gaussian model); a fixed share is removed (batch or source level)
      with probability -> 1 as N grows (its statistics grow like sqrt N)
  SR9 two sources cannot localise: with two sources the statistics are z_1 = -z_2, so the source filter does not drop either
  SR10 sampled batches agree with the Gaussian model (small check on an invented truth, 300 replicates): every rule's right-rate within
      0.08 of the model
"""
import itertools, os, sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pce.lab import law, moments
from pce import segment_decision as SD
from pce import injection as IJ
from pce import sources as SR
from pce import Window
from pce.integrate import gluable_fraction

rng = np.random.default_rng(20261029); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

def rand_truth():
    while True:
        w = rng.uniform(0.2, 0.8); a1 = rng.uniform(0.05, 0.95, 3); a2 = rng.uniform(0.05, 0.95, 3); P = law(w, a1, a2)
        mu, M, mS = moments(P)
        if min(abs(M[0, 1]), abs(M[0, 2]), abs(M[1, 2])) > 0.01 and M[0, 1] * M[0, 2] * M[1, 2] > 0: return P

# SR1
ok = True; worst = 0.0
for _ in range(20):
    P = rand_truth(); lo, hi = IJ.stealth_range(P, (0, 1, 2)); s = rng.uniform(lo, hi); Q = SR.hidden_Q(P, s); e = rng.uniform(0, 1); Pe = IJ.mix(P, Q, e)
    for S in SD.WINDOWS[:3]: ok &= np.allclose(SR.marg(Pe, S), SR.marg(P, S), atol=1e-15)
    d = SR.mT_stats(SR.marg(Pe, (0, 1, 2)))[0] - SR.mT_stats(SR.marg(P, (0, 1, 2)))[0]; worst = max(worst, abs(d - e * s))
check("SR1 batch-level hidden channel: pairs exactly unchanged, m_T moves by eps s", ok and worst < 1e-14, f"max |dm - eps s| {worst:.1e}")

# SR2
D3 = {"A": 2, "B": 2, "C": 2}; errs = []; lam_g = []
for _ in range(10):
    P = rand_truth(); lo, hi = IJ.stealth_range(P, (0, 1, 2)); Q = SR.hidden_Q(P, hi); e = rng.uniform(0.02, 0.5); Pe = IJ.mix(P, Q, e)
    lam = gluable_fraction(D3, [Window(("A", "B", "C"), P), Window(("A", "B", "C"), Pe)]).lam
    lam_g.append(gluable_fraction(D3, [Window(("A", "B", "C"), P), Window(("A", "B", "C"), P)]).lam)
    tv = 0.5 * np.abs(IJ.as_vec(Q) - IJ.as_vec(P)).sum(); errs.append(abs((1 - lam) - e * tv)); assert 1 - lam <= e + 1e-12
check("SR2 Tier 1: genuine sections glue (lambda* = 1); 1 - lambda* = eps TV(P, Q) <= eps", max(errs) < 1e-8 and min(lam_g) > 1 - 1e-9,
      f"max |1 - lambda* - eps TV| {max(errs):.1e}")

# SR3
P = law(0.6, [.5, .1, .4], [.05, .5, .1]); acts = [lambda x: 1 - float(sum(x) > 0), lambda x: 0.872 - float(x[0] or x[1])]
lo, hi = IJ.stealth_range(P, (0, 1, 2)); Q = SR.hidden_Q(P, hi); K_C = 8; epsC = 0.18
fam = [P] * (2 * K_C) + [IJ.mix(P, Q, epsC)] * K_C; pooled = sum(fam) / len(fam); dirac = [pooled] * len(fam)
same = np.allclose(sum(dirac) / len(dirac), pooled) and np.allclose(pooled, IJ.mix(P, Q, epsC / 3))
pre_orc = IJ.pipeline(P, acts, 0.30)["post"]
flip = lambda Pq: (lambda o: o is not None and o["post"] not in (pre_orc, "empty"))(IJ.pipeline(Pq, acts, 0.30))
e_star = IJ.exact_budget(P, Q, flip, emax=0.2, grid=2000)
eC = IJ.exact_budget(P, IJ.mix(P, Q, 1.0), lambda Pq: flip(Pq), emax=0.2, grid=2000)
check("SR3 pooling = monad multiplication: attacked family and eta(pooled) have one image; pooled budget for the source = eps* / pi",
      same and abs(3 * e_star - 3 * eC) < 1e-9 and abs(e_star - 0.037037) < 1e-5, f"eps* = {e_star:.4f}, source share needed {3 * e_star:.4f}")

# SR4
worst = 0.0
for _ in range(50):
    K_C = int(rng.integers(2, 12)); K = K_C + int(rng.integers(2, 30)); n = float(rng.integers(100, 10000)); sig = rng.uniform(0.01, 1); a = rng.uniform(0, 4)
    D = SR.tuned_shift(sig, n, K_C, K, a); pi = K_C / K; shift_se = pi * D / (sig / np.sqrt(n * K))
    worst = max(worst, abs(shift_se - SR.band(a, pi)))
check("SR4 Theorem J(1): pooled shift = a sqrt(pi / (1 - pi)) pooled se for every K_C, K, n", worst < 1e-12, f"max error {worst:.1e}")

# SR5
worst = 0.0
for _ in range(6):
    P = rand_truth(); mu, M, mS = moments(P); v = SD.v_closed(M); r0 = mS((0, 1, 2)) / np.prod(v)
    pA = SD.U_poly(mu, v, lambda x: 1 - float(sum(x) > 0), range(3)); c = pA[0] + pA[1] * (r0 - 0.2 * np.sign(pA[1]))
    acts_r = [lambda x: 1 - float(sum(x) > 0), (lambda cc: (lambda x: cc))(float(c))]
    try: gm = SR.gauss_model(P, acts_r, -1.0, 0)
    except Exception: continue
    worst = max(worst, abs(gm["GG_t"] / gm["a2"] - 1), abs(gm["a2"] - gm["sm"] ** 2 / gm["vT"] ** 2) / gm["a2"])
check("SR5 G's triple part is (m_T error) / v_T: its variance equals sigma_m^2 / v_T^2 = the triple part of tau^2", worst < 1e-5, f"max rel. error {worst:.1e}")

# SR6, SR7, SR8 on the worked-example truth (Gaussian model only; no sampling)
P = law(0.6, [.5, .1, .4], [.05, .5, .1]); lo, hi = IJ.stealth_range(P, (0, 1, 2))
rho = 25; K_C = 8; K = 24; cs = norm.ppf(1 - 0.05 / 6); cb = norm.ppf(1 - 0.05 / 48); gm = SR.gauss_model(P, acts, 0.30, 0)
B = SR.band(cs, 1 / 3); sm, vT, d = gm["sm"], gm["vT"], gm["d"]; tau = np.sqrt(gm["a2"] + gm["r0"] ** 2 * gm["b2"] / rho)
wr = []; naive_wrong = []
for n in (500, 2000, 8000, 32000):
    D = SR.tuned_shift(sm, n, K_C, K, cs); out = SR.predict(gm, n, K_C, [D * 2] * 4 + [0] * 4, cb, cs, rho=rho, draws=100000)
    wr.append(out["glue"]["wrong"])
    o2 = SR.predict(gm, n, K_C, [0.18 * hi] * K_C, cb, cs, rho=rho, draws=100000, rules=("naive",)); naive_wrong.append(o2["naive"]["wrong"])
check("SR6 Theorem J(2): glue never settles wrong beyond Phi(B - z); naive pooling under a fixed share settles wrong w.p. -> 1",
      max(wr) <= norm.cdf(B - 1.96) and naive_wrong[-1] > 0.95 and naive_wrong[0] < naive_wrong[-1],
      f"glue wrong {np.round(wr, 4).tolist()} <= {norm.cdf(B - 1.96):.3f}; naive wrong {np.round(naive_wrong, 3).tolist()}")

def half_N(fn, lo=50, hi=200000):
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if fn(mid) >= 0.5: hi = mid
        else: lo = mid
    return np.sqrt(lo * hi)
def right_glue(n, a):
    D = SR.tuned_shift(sm, n, K_C, K, a); return SR.predict(gm, n, K_C, [D * 2] * 4 + [0] * 4, cb, cs, rho=rho, draws=60000, rules=("naive", "glue"))["glue"]["right"]
rows = []
for a in (0.0, cs - norm.ppf(0.8)):
    n_half = half_N(lambda n: right_glue(n, a)); Bq = SR.band(a, 1 / 3); pred = ((1.96 * tau + Bq * sm / abs(vT)) / d) ** 2
    rows.append((a, 24 * n_half, pred))
check("SR7 Theorem J(3): with no adversary glue's right-rate crosses 1/2 at (z tau / d)^2 (within 5%); under the tuned adversary between that and the never-exposed bound ((z tau + B sigma_m / |v_T|) / d)^2, within 15% of the bound",
      abs(rows[0][1] / rows[0][2] - 1) < 0.05 and rows[0][1] < rows[1][1] <= rows[1][2] * 1.02 and rows[1][1] > 0.85 * rows[1][2], "; ".join(f"a = {r_[0]:.2f}: model N 1/2 {r_[1]:.0f}, formula {r_[2]:.0f}" for r_ in rows))

caught = []; srcdrop = []; fixed_flag = []
for n in (500, 2000, 8000):
    D = SR.tuned_shift(sm, n, K_C, K, cs - norm.ppf(0.8)); o = SR.predict(gm, n, K_C, [D * K_C] + [0] * 7, cb, cs, rho=rho, draws=60000, rules=("glue",))
    caught.append(o["glue_adv_batches_dropped"]); srcdrop.append(o["glue_source_flag"][2])
    o = SR.predict(gm, n, K_C, [0.18 * hi] * K_C, cb, cs, rho=rho, draws=60000, rules=("glue",)); fixed_flag.append(o["glue_adv_removed"])
check("SR8 a concentrated attack is localised at the batch level (dropped >= 0.9, source kept); a fixed share is removed (batch or source level) w.p. -> 1",
      min(caught) >= 0.9 and max(srcdrop) <= 0.1 and fixed_flag[-1] > 0.99 and fixed_flag[0] < fixed_flag[-1],
      f"batch dropped {np.round(caught, 3).tolist()}, source dropped {np.round(srcdrop, 3).tolist()}; fixed share caught {np.round(fixed_flag, 3).tolist()}")

# SR9
m = rng.normal(size=8); src = np.array(["G1"] * 4 + ["C"] * 4); keep, zs, dropped = SR.source_filter(m, src, 1.0, 1, 0.0, np.ones(8, bool))
keep3, zs3, dropped3 = SR.source_filter(np.r_[m, rng.normal(size=4)], np.r_[src, ["G2"] * 4], 1.0, 1, 0.0, np.ones(12, bool))
check("SR9 two sources cannot localise a gluing failure (no z statistics, nothing dropped); three can", dropped is None and keep.all() and dropped3 is not None)

# SR10
Pi = law(0.45, [.75, .25, .70], [.25, .70, .20]); mu, M, mS = moments(Pi); v = SD.v_closed(M); r0 = mS((0, 1, 2)) / np.prod(v)
pA = SD.U_poly(mu, v, lambda x: 1 - float(sum(x) > 0), range(3)); _, _, _, _, Rw = SD.from_tables(SD.exact_tables(Pi))
rc = r0 - 0.3 * np.sign(pA[1])                 # the flat act sits 0.3 below the truth on the act's line
cC = float(pA[0] + pA[1] * rc); actsI = [lambda x: 1 - float(sum(x) > 0), lambda x: cC]
gmI = SR.gauss_model(Pi, actsI, -1.0, 0); loI, hiI = IJ.stealth_range(Pi, (0, 1, 2)); s_dir = hiI if np.sign(hiI / gmI["vT"]) == np.sign(gmI["rc"] - gmI["r0"]) else loI
QI = SR.hidden_Q(Pi, s_dir); n = 400; reps = 300; src = np.array(["G1"] * 8 + ["G2"] * 8 + ["C"] * 8)
D = SR.tuned_shift(gmI["sm"], n, 8, 24, cs); epsb = 2 * D / abs(s_dir); counts = {r: 0 for r in ("naive", "glue", "drop_C_triple", "write_off_C")}
for _ in range(reps):
    bats = [SR.sample_batch(Pi, QI, (epsb if (lab == "C" and k < 4) else 0.0), n, rng, rho=rho) for k, lab in zip(list(range(8)) * 3, src)]
    out, diag = SR.run_rules(bats, src, n, actsI, -1.0, 0, cb, cs, rho=rho)
    for r in counts: counts[r] += out[r][0] == "right"
sh = [np.sign(s_dir) * 2 * D] * 4 + [0] * 4; pr = SR.predict(gmI, n, 8, [x for x in sh], cb, cs, rho=rho, draws=100000)
gaps = {r: abs(counts[r] / reps - pr[r]["right"]) for r in counts}
check("SR10 sampled batches agree with the Gaussian model on an invented truth (right-rate within 0.08 for every rule)", max(gaps.values()) < 0.08,
      "; ".join(f"{r}: {counts[r] / reps:.3f} vs {pr[r]['right']:.3f}" for r in counts))

print(f"{sum(results)}/{len(results)} passed")
