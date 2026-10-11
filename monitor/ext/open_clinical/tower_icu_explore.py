"""EXPLORATORY follow-ups to PREANALYSIS_TOWER_ICU.md (not pre-registered; written after tower_icu.json was frozen):
(1) is the toric type of the observed test system decided on its bands?  (2) the level-one obstruction of the observed death rate
against the tower's admissible predictions, with its certificate (Tower Thm. 10.8);  (3) a tower whose reliability floor is the base
rate of the evidence half instead of the fit sample's, in the shift arm.  Writes results/tower_icu_exploratory.json."""
import os, sys, json, contextlib, io
import numpy as np, pandas as pd
import tower_icu as TI
sys.path.insert(0, os.path.join(TI.H, '..', 'structure', 'programme_code'))
with contextlib.redirect_stdout(io.StringIO()): import check_lifting as CL
from monitor import bands, wall_signs
A = pd.read_csv(os.path.join(TI.CACHE, 'set_a.csv')); B = pd.read_csv(os.path.join(TI.CACHE, 'set_b.csv'))
B1, B2 = B.iloc[:len(B) // 2], B.iloc[len(B) // 2:]; Bs = B[B.ICUType.isin([1, 3])]; Bs1, Bs2 = Bs.iloc[:len(Bs) // 2], Bs.iloc[len(Bs) // 2:]
arms = {'same_population': (A, B1, B2), 'shift': (A[A.ICUType.isin([2, 4])], Bs1, Bs2)}
out = {}
for name, (fit, ev, te) in arms.items():
    te_h = te.dropna(subset=['H1']); n = len(te_h); d = te_h.death.mean(); g = np.bincount(te_h.H1.astype(int), minlength=3) / n
    signs = wall_signs(bands([d, 1 - d], n), bands(g, n))
    T = TI.Tower(fit); T.condition(ev); f_ka = T.rho_ka(te_h).mean(0)
    from fractions import Fraction as Fr
    S1 = [CL.Dist([(0, Fr(float(x)).limit_denominator(10 ** 9)), (1, 1 - Fr(float(x)).limit_denominator(10 ** 9))]) for x in np.unique(f_ka.round(9))]
    x1 = CL.Dist([(0, Fr(float(d)).limit_denominator(10 ** 9)), (1, 1 - Fr(float(d)).limit_denominator(10 ** 9))])
    d1, margin = CL.l1_distance(x1, S1, (0, 1)), CL.dual_margin(x1, S1, (0, 1))
    out[name] = dict(observed_rate=round(float(d), 4), g=[round(float(x), 4) for x in g], n=n, wall_signs_on_bands=signs.tolist(),
                     surface_observed_decided=bool((signs != 0).all()), admissible_rates=[round(float(f_ka.min()), 4), round(float(f_ka.max()), 4)],
                     level1_obstruction=dict(d1=round(d1, 5), certificate_margin=round(margin, 5), fails_at_level=1 if d1 > 1e-9 else None))
# (3) reliability floor learned from the evidence half (shift arm)
fit, ev, te = arms['shift']
class TowerEv(TI.Tower):
    pass
T = TowerEv(fit); T.base = float(ev.death.mean())   # the floor rho_0 is the evidence base rate; fitted models unchanged
te_h0 = te.dropna(subset=['H1']); fk0 = T.rho_ka(te_h0).mean(0)
tube = [dict(n=0, f1=round(float((fk0 * T.weights()).sum()), 5))] + [dict(n=n_, f1=round(float((fk0 * w_).sum()), 5)) for n_, w_ in T.condition(ev, record_every=50)]
for t_ in tube: t_['surface'] = TI.toric(t_['f1'], np.bincount(te_h0.H1.astype(int), minlength=3) / len(te_h0))[0]
Tf = TI.Tower(fit); Tf.condition(ev)
y = te.death.values; lt, lf = TI.logloss(T.predict(te), y), TI.logloss(Tf.predict(te, flat=True), y)
rng = np.random.default_rng(TI.SEED); idx = rng.integers(0, len(y), (2000, len(y))); dd = (lf[idx] - lt[idx]).mean(1)
te_h = te.dropna(subset=['H1']); g = np.bincount(te_h.H1.astype(int), minlength=3) / len(te_h); f_ka = T.rho_ka(te_h).mean(0); w = T.weights()
post = {}
for k in range(len(TI.MODELS)):
    for a in range(len(TI.ALPHAS)):
        s, _ = TI.toric(float(f_ka[k, a]), g); post[s] = post.get(s, 0.0) + float(w[k, a])
out['shift_evidence_floor'] = dict(floor=round(T.base, 4), logloss_tower=round(float(lt.mean()), 5), logloss_flat=round(float(lf.mean()), 5),
                                   delta=round(float(dd.mean()), 5), delta_ci=[round(float(np.percentile(dd, 2.5)), 5), round(float(np.percentile(dd, 97.5)), 5)],
                                   posterior_alpha={str(a): round(float(x), 4) for a, x in zip(TI.ALPHAS, w.sum(0))},
                                   predicted=round(float(T.predict(te).mean()), 4), observed=round(float(y.mean()), 4),
                                   surface_posterior={k: round(v, 4) for k, v in post.items()}, level1_rate=round(float((f_ka * w).sum()), 4), tube=tube,
                                   tears_of_tube=[(a['n'], b['n'], a['surface'], b['surface']) for a, b in zip(tube, tube[1:]) if a['surface'] != b['surface']],
                                   f_ka=[[round(float(x), 5) for x in r] for r in f_ka], weights=[[round(float(x), 6) for x in r] for r in w])
json.dump(out, open(os.path.join(TI.H, 'results', 'tower_icu_exploratory.json'), 'w'), indent=1); print(json.dumps(out, indent=1))
