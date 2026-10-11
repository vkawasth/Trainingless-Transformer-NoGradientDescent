"""Diabetes 130-US Hospitals (UCI, 1999-2008): analyses of PREANALYSIS_AMENDMENT_OPEN_DATA.md, Part II.
Usage: python3 diabetes130.py [--zip PATH] [--out PATH] [--quick]"""
import os, sys, json, zipfile, argparse
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, 'lib'))
from analyses import geometry, composition, path_outcome, cbd_strata, selection, table
from sklearn.linear_model import LogisticRegression

def load(zpath):
    z = zipfile.ZipFile(zpath)
    name = [n for n in z.namelist() if n.endswith('diabetic_data.csv')][0]
    d = pd.read_csv(z.open(name), na_values=['?'], low_memory=False, keep_default_na=False)
    d['A1Cresult'] = d['A1Cresult'].replace({'': 'None'}).fillna('None')
    lo = d['age'].str.extract(r'\[(\d+)-')[0].astype(int)
    d['X'] = np.select([lo < 50, lo < 70], [0, 1], 2)
    d['Y'] = d['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})
    d['INS'] = d['insulin'].map({'No': 0, 'Steady': 1, 'Up': 2, 'Down': 2})
    d['A1C'] = d['A1Cresult'].map({'Norm': 0, '>7': 1, '>8': 2})          # NaN = not measured
    d['CHG'] = d['change'].map({'No': 0, 'Ch': 1})
    d['SRC'] = d['admission_source_id'].map({7: 'emergency room', 1: 'physician referral'})
    return d.sort_values('encounter_id').reset_index(drop=True)

def analyse(d, rng, quick=False):
    B = 50 if quick else 400; out = {}
    s = d[d.SRC.notna()]
    out['A_coupling_geometry'] = geometry({k: (g.INS, g.Y) for k, g in s.groupby('SRC')}, 3, 3, pairs='all')
    out['A_coupling_geometry']['toric_2x3_change_x_readmission'] = {k: __import__('analyses').toric_2x3(table(g.CHG, g.Y, 2, 3)) for k, g in s.groupby('SRC')}
    p = d.groupby('patient_nbr').filter(lambda g: len(g) >= 3).groupby('patient_nbr').head(3).copy()
    p['k'] = p.groupby('patient_nbr').cumcount()
    w = p.pivot(index='patient_nbr', columns='k', values='INS').dropna().astype(int)
    y3 = (p[p.k == 2].set_index('patient_nbr').Y.reindex(w.index) == 2).astype(int)
    out['B_composition'] = composition(w[0], w[1], w[2], rng, B=B, starts=5 if quick else 30)
    out['C_path_outcome'] = path_outcome(w[0], w[1], w[2], y3, rng, B=B * 5)
    strata = {f'age band {x}': (table(s[(s.X == x) & (s.SRC == 'emergency room')].CHG, s[(s.X == x) & (s.SRC == 'emergency room')].Y, 2, 3),
                                table(s[(s.X == x) & (s.SRC == 'physician referral')].CHG, s[(s.X == x) & (s.SRC == 'physician referral')].Y, 2, 3)) for x in (0, 1, 2)}
    out['D_contextuality_source'] = cbd_strata(strata, (2, 3), rng, B=B)
    # secondary weights: logistic model of A1C measurement on X, Y, admission source and admission type (one-hot)
    Xd = pd.get_dummies(d[['X', 'Y', 'admission_source_id', 'admission_type_id']].astype(str), drop_first=True)
    meas = d.A1C.notna().astype(int)
    lr = LogisticRegression(max_iter=2000).fit(Xd, meas); pr = np.clip(lr.predict_proba(Xd)[:, 1], 1e-3, 1)
    out['E_selection'] = selection(d.X, d.Y, d.A1C, rng, B=B, extra_weights=1 / pr)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--zip', default=os.path.join(H, 'data', 'uci', 'diabetes130.zip'))
    ap.add_argument('--out', default=os.path.join(H, 'results', 'results_diabetes130.json')); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args(); rng = np.random.default_rng(20261009); d = load(a.zip)
    res = dict(dataset='Diabetes 130-US Hospitals 1999-2008 (UCI)', plan='PREANALYSIS_AMENDMENT_OPEN_DATA.md (2026-10-10), Part II',
               cohort=dict(encounters=len(d), patients=int(d.patient_nbr.nunique()), patients_3plus=int((d.groupby('patient_nbr').size() >= 3).sum()),
                           a1c_measured=int(d.A1C.notna().sum()), by_source={k: int(v) for k, v in d.SRC.value_counts().items()}))
    res.update(analyse(d, rng, a.quick)); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)); print('wrote', a.out)
