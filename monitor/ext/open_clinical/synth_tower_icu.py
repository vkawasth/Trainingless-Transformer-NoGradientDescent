"""Adapter checks of PREANALYSIS_TOWER_ICU.md on planted data: (1) the tower recovers a planted transport reliability and beats flat
averaging when alpha* = 1/2, and does not beat it beyond noise when alpha* = 1; (2) the interchange test with 17 raw states keeps its
size and has power (20 replicates each)."""
import numpy as np, pandas as pd, tower_icu as TI
def patients(n, rng):
    return pd.DataFrame(dict(RecordID=np.arange(n), H1=rng.choice([0, 1, 2, np.nan], n, p=[.15, .25, .55, .05]), R=rng.choice([0, 1, 2, np.nan], n, p=[.6, .2, .15, .05]),
                             vent24=rng.integers(0, 2, n), ICUType=rng.choice([1, 2, 3, 4], n), SOFAb=rng.choice([0, 1, 2, np.nan], n, p=[.4, .35, .2, .05])))
def truth(df):
    z = -2.6 + 0.9 * (df.H1.fillna(2) == 0) + 0.4 * (df.H1.fillna(2) == 1) + 0.7 * (df.R.fillna(0) == 2) + 0.6 * df.vent24 + 0.3 * (df.ICUType == 3)
    return 1 / (1 + np.exp(-z))
def main():
    rng = np.random.default_rng(3); ok = True; out = {}
    fit = patients(4000, rng); p = truth(fit); fit['death'] = (rng.random(len(fit)) < p).astype(int); base = fit.death.mean()
    for astar in (0.5, 1.0):
        ev, te = patients(2000, rng), patients(2000, rng)
        for d in (ev, te): d['death'] = (rng.random(len(d)) < astar * truth(d) + (1 - astar) * base).astype(int)
        r = TI.run_arm(fit, ev, te, f'planted alpha*={astar}')
        out[astar] = r; print(astar, r['posterior_alpha'], r['delta'], r['delta_ci'], r['level3_earns_keep'], 'condS err', r['condS_agreement_200'], 'coh', r['coherence_max_err'])
        top_alpha = max(r['posterior_alpha'].items(), key=lambda kv: kv[1])[0]
        ok &= (r['level3_earns_keep'] and abs(float(top_alpha) - 0.5) <= 0.25) if astar == 0.5 else (not r['level3_earns_keep'] and float(top_alpha) >= 0.75)
        ok &= r['condS_agreement_200'] < 1e-9 and r['coherence_max_err'] < 1e-12
    import banding_interchange as BI; BI.set_merge(17, [5, 7]); cal = {}
    for mem in (0.0, 0.3):
        prim = mon = 0
        for rep in range(20):
            n = 3000; x1 = np.clip(rng.normal(9, 3, n).astype(int), 0, 16)
            step = lambda v: np.clip(v + rng.integers(-2, 3, len(v)), 0, 16); x2 = step(x1); x3 = np.where(rng.random(n) < mem, step(x1), step(x2))
            res = BI.run(x1, x2, x3, np.random.default_rng(rep)); prim += res['decision_primary'].startswith('raw chain has memory'); mon += res['banded_gap_bits'] > res['banded_markov_null95']
        cal[mem] = dict(interchange=prim, monitor=mon); print('interchange calibration', mem, cal[mem])
    ok &= cal[0.0]['interchange'] <= 3 and cal[0.3]['interchange'] >= 14
    print('PASS' if ok else 'FAIL'); return 0 if ok else 1
