"""WP4 (PREANALYSIS_PHASE4.md): a generalized reliability level. Level three carries (alpha, floor phi): rho' = alpha rho + (1-alpha) phi,
the Kleisli composite of rho with the binary channel P(1|1) = alpha + (1-alpha) phi, P(1|0) = (1-alpha) phi. Variants T-fit (phi = fit base
rate), T-ev (phi = evidence base rate, pre-registered here), T-fam (phi on a grid). Arms R0-R4 on PhysioNet 2012.
Usage: python3 wp4_tower.py synth | run  -> results/wp4_tower.json"""
import os, sys, json
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, '..', 'open_clinical'))
import tower_icu as TI
from monitor import bands, wall_signs
SEED = 20261010; ALPHAS = TI.ALPHAS; PHIS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

class TowerF(TI.Tower):
    def __init__(self, fit, floors):
        super().__init__(fit); self.floors = np.array(floors, float)
        self.logw = np.zeros((len(self.models), len(self.alphas), len(self.floors))) - np.log(len(self.models) * len(self.alphas) * len(self.floors))
    def rho_ka(self, df):
        r = self.rho(df); a = np.array(self.alphas)
        return a[None, None, :, None] * r[:, :, None, None] + (1 - a)[None, None, :, None] * self.floors[None, None, None, :]
    def condition(self, df, record_every=None):
        p = self.rho_ka(df); y = df.death.values; path = []
        for i in range(len(df)):
            self.logw += np.log(np.where(y[i] == 1, p[i], 1 - p[i]))
            if record_every and ((i + 1) % record_every == 0 or i == len(df) - 1): path.append((i + 1, self.weights().copy()))
        return path
    def predict(self, df, flat=False):
        p = self.rho_ka(df)
        if flat:
            lw = self.logw[:, -1, 0]; w1 = np.exp(lw - lw.max()); w1 /= w1.sum(); return p[:, :, -1, 0] @ w1
        return np.einsum('nkaf,kaf->n', p, self.weights())

def ll(p, y): return TI.logloss(p, y)
def arm(fit, ev, te, name, rng):
    y = te.death.values; flat = TowerF(fit, [fit.death.mean()]); flat.condition(ev); pf = flat.predict(te, flat=True); lf = ll(pf, y)
    idx = rng.integers(0, len(y), (2000, len(y))); te_h = te.dropna(subset=['H1']); g = np.bincount(te_h.H1.astype(int), minlength=3) / len(te_h)
    obs = float(te_h.death.mean()); sg = wall_signs(bands([obs, 1 - obs], len(te_h)), bands(g, len(te_h)))
    out = dict(arm=name, n=[len(fit), len(ev), len(te)], base_fit=round(float(fit.death.mean()), 4), base_evidence=round(float(ev.death.mean()), 4),
               observed_test=round(float(y.mean()), 4), logloss_flat=round(float(lf.mean()), 5), observed_surface=TI.toric(obs, g)[0],
               observed_surface_decided=bool((sg != 0).all()), variants={})
    for vname, floors in (('T-fit', [fit.death.mean()]), ('T-ev', [ev.death.mean()]), ('T-fam', PHIS)):
        T = TowerF(fit, floors); path = T.condition(ev, record_every=50); pt = T.predict(te); lt = ll(pt, y); d = (lf[idx] - lt[idx]).mean(1)
        w = T.weights(); fk = T.rho_ka(te_h).mean(0); post = {}
        for k in range(fk.shape[0]):
            for a in range(fk.shape[1]):
                for f in range(fk.shape[2]):
                    s = TI.toric(float(fk[k, a, f]), g)[0]; post[s] = post.get(s, 0.0) + float(w[k, a, f])
        tube = [(n_, round(float((fk * w_).sum()), 5)) for n_, w_ in path]
        out['variants'][vname] = dict(logloss=round(float(lt.mean()), 5), delta=round(float(d.mean()), 6),
                                      ci=[round(float(np.percentile(d, 2.5)), 6), round(float(np.percentile(d, 97.5)), 6)], earns_keep=bool(np.percentile(d, 2.5) > 0),
                                      harms=bool(np.percentile(d, 97.5) < 0), predicted=round(float(pt.mean()), 4),
                                      posterior_alpha={str(a): round(float(x), 4) for a, x in zip(ALPHAS, w.sum((0, 2)))},
                                      posterior_floor={str(f): round(float(x), 4) for f, x in zip(floors, w.sum((0, 1)))},
                                      posterior_models={m: round(float(x), 4) for m, x in zip(TI.MODELS, w.sum((1, 2)))},
                                      surface_posterior={k: round(v, 4) for k, v in post.items()},
                                      tube=tube, tube_crossings=int(sum(TI.toric(a_[1], g)[0] != TI.toric(b_[1], g)[0] for a_, b_ in zip(tube, tube[1:]))))
    return out

def arms(A, B):
    h = lambda D: (D.iloc[:len(D) // 2], D.iloc[len(D) // 2:]); S = lambda D, u: D[D.ICUType.isin(u)]
    return {'R0a same population (A -> B)': (A, *h(B)), 'R0b shift A surgical -> B medical/coronary': (S(A, [2, 4]), *h(S(B, [1, 3]))),
            'R1 same population reversed (B -> A)': (B, *h(A)), 'R2 reverse shift B surgical -> A medical/coronary': (S(B, [2, 4]), *h(S(A, [1, 3]))),
            'R3 opposite shift A medical/coronary -> B surgical': (S(A, [1, 3]), *h(S(B, [2, 4]))),
            'R4 cross shift A coronary+cardiac surgery -> B medical+surgical': (S(A, [1, 2]), *h(S(B, [3, 4])))}
def run():
    A = pd.read_csv(os.path.join(TI.CACHE, 'set_a.csv')); B = pd.read_csv(os.path.join(TI.CACHE, 'set_b.csv')); rng = np.random.default_rng(SEED)
    res = dict(plan='PREANALYSIS_PHASE4.md WP4', arms={})
    for name, (f, e, t) in arms(A, B).items():
        r = arm(f, e, t, name, rng); res['arms'][name] = r
        print(name, r['base_fit'], r['base_evidence'], r['observed_test'], {k: (v['delta'], v['ci'], v['predicted']) for k, v in r['variants'].items()}, flush=True)
    sh = [k for k in res['arms'] if k[:2] in ('R2', 'R3', 'R4')]
    res['H1_Tev_replicates'] = dict(arms_positive=sum(res['arms'][k]['variants']['T-ev']['earns_keep'] for k in sh), of=len(sh))
    res['H1_Tev_replicates']['holds'] = res['H1_Tev_replicates']['arms_positive'] >= 2
    res['H2_Tfam'] = dict(arms_positive=sum(res['arms'][k]['variants']['T-fam']['earns_keep'] for k in sh), of=len(sh))
    res['H2_Tfam']['holds'] = res['H2_Tfam']['arms_positive'] >= 2
    r1 = res['arms']['R1 same population reversed (B -> A)']['variants']
    res['H3_no_harm'] = dict(harms={k: v['harms'] for k, v in r1.items()}, holds=not any(v['harms'] for v in r1.values()))
    json.dump(res, open(os.path.join(H, 'results', 'wp4_tower.json'), 'w'), indent=1)
    print({k: res[k] for k in ('H1_Tev_replicates', 'H2_Tfam', 'H3_no_harm')})

def synth():
    """planted prevalence shift (log-odds +0.7 in evidence and test): T-ev and T-fam earn their keep; no shift: no variant harms"""
    import synth_tower_icu as ST
    rng = np.random.default_rng(8); ok = True
    fit = ST.patients(4000, rng); fit['death'] = (rng.random(len(fit)) < ST.truth(fit)).astype(int)
    for shift in (0.7, 0.0):
        ev, te = ST.patients(2000, rng), ST.patients(2000, rng)
        for d in (ev, te):
            p = ST.truth(d); lo = np.log(p / (1 - p)) + shift; d['death'] = (rng.random(len(d)) < 1 / (1 + np.exp(-lo))).astype(int)
        r = arm(fit, ev, te, f'planted shift {shift}', np.random.default_rng(1)); v = r['variants']
        print(shift, {k: (x['delta'], x['ci'], x['earns_keep'], x['harms']) for k, x in v.items()})
        ok &= (v['T-ev']['earns_keep'] and v['T-fam']['earns_keep']) if shift > 0 else not any(x['harms'] for x in v.values())
    open(os.path.join(H, 'results', 'wp4_tower_synth.txt'), 'w').write('PASS' if ok else 'FAIL'); print('PASS' if ok else 'FAIL')
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(H, '..', 'open_clinical')); {'synth': synth, 'run': run}[sys.argv[1]]()
