"""PhysioNet/CinC Challenge 2012: analyses A-E of PREANALYSIS_AMENDMENT_OPEN_DATA.md, Part I.
Usage: python3 challenge2012.py a            (analysis sample: set A)
       python3 challenge2012.py b            (replication: set B; run only after results_2012_a.json is frozen)
       python3 challenge2012.py compare      (negative control A vs B, and the replication table)
       python3 challenge2012.py a --data DIR --zip set-a.zip --outcomes Outcomes-a.txt   (paths, e.g. for synthetic tests)"""
import os, sys, json, zipfile, io, argparse
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, 'lib'))
from analyses import geometry, composition, path_outcome, cbd_strata, selection, table
from monitor import State, regime

WINDOWS = [(0, 6), (21, 27), (42, 48)]
def band(v, cuts): return np.nan if v is None or np.isnan(v) else float(np.digitize(v, cuts))

def load(zpath, opath):
    z = zipfile.ZipFile(zpath); rows = []
    for name in z.namelist():
        if not name.endswith('.txt'): continue
        d = pd.read_csv(io.BytesIO(z.read(name)))
        hh = d.Time.str.split(':', expand=True).astype(float); d['t'] = hh[0] + hh[1] / 60
        desc = d[d.t == 0].drop_duplicates('Parameter').set_index('Parameter').Value
        rid = int(desc.get('RecordID', -1)); icu = desc.get('ICUType', np.nan)
        rec = dict(RecordID=rid, ICUType=icu)
        for k, (a, b) in enumerate(WINDOWS):
            w = d[(d.t >= a) & (d.t < b)]
            m = w[(w.Parameter == 'MAP') & w.Value.between(10, 250)].Value
            if m.empty: m = w[(w.Parameter == 'NIMAP') & w.Value.between(10, 250)].Value
            rec[f'H{k+1}'] = band(m.median() if not m.empty else np.nan, (65, 75))
        day = d[(d.t >= 0) & (d.t <= 24)]
        for par, col, cuts in (('Creatinine', 'R', (1.2, 2.0)), ('Lactate', 'L', (2.0, 4.0))):
            v = day[day.Parameter == par].sort_values('t').Value
            rec[col] = band(v.iloc[0] if not v.empty else np.nan, cuts)
        rec['vent24'] = int(((d.Parameter == 'MechVent') & (d.Value == 1) & (d.t < 24)).any())
        rows.append(rec)
    st = pd.DataFrame(rows)
    oc = pd.read_csv(opath).rename(columns={'In-hospital_death': 'death'})
    st = st.merge(oc[['RecordID', 'SOFA', 'death']], on='RecordID', how='left')
    return st[st.ICUType.notna() & (st.ICUType > 0)].reset_index(drop=True)

def analyse(st, rng, quick=False):
    B = 50 if quick else 400; out = {}
    ICU = {1: 'coronary', 2: 'cardiac surgery', 3: 'medical', 4: 'surgical'}
    a = st.dropna(subset=['H1', 'R'])
    out['A_coupling_geometry'] = geometry({ICU[k]: (a[a.ICUType == k].H1, a[a.ICUType == k].R) for k in (1, 2, 3, 4)}, 3, 3, pairs='all',
                                          binary_first=lambda x: (x != 0).astype(int))
    b = st.dropna(subset=['H1', 'H2', 'H3'])
    out['B_composition'] = composition(b.H1, b.H2, b.H3, rng, B=B, starts=5 if quick else 30)
    out['C_path_outcome'] = path_outcome(b.H1, b.H2, b.H3, b.death, rng, B=B * 5)
    c = st.dropna(subset=['H1', 'H2'])
    def strata_by(col, groups):
        return {g: (table(c[(c[col].isin(v)) & (c.ICUType == 3)].H2, c[(c[col].isin(v)) & (c.ICUType == 3)].vent24, 3, 2),
                    table(c[(c[col].isin(v)) & (c.ICUType == 4)].H2, c[(c[col].isin(v)) & (c.ICUType == 4)].vent24, 3, 2)) for g, v in groups.items()}
    out['D_contextuality_icu'] = cbd_strata(strata_by('H1', {'H1=shock': [0], 'H1=borderline': [1], 'H1=stable': [2]}), (3, 2), rng, B=B)
    c = c.assign(SOFAb=pd.cut(c.SOFA, [-1, 4, 8, 100], labels=[0, 1, 2]).astype(float))
    out['D_sensitivity_sofa'] = cbd_strata(strata_by('SOFAb', {'SOFA 0-4': [0], 'SOFA 5-8': [1], 'SOFA 9+': [2]}), (3, 2), rng, B=B)
    e = st.dropna(subset=['H1', 'R'])
    out['E_selection'] = selection(e.H1, e.R, e.L, rng, B=B)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('which'); ap.add_argument('--data', default=os.path.join(H, 'data', 'challenge2012'))
    ap.add_argument('--zip'); ap.add_argument('--outcomes'); ap.add_argument('--out'); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args(); rng = np.random.default_rng(20261009)
    if a.which in ('a', 'b'):
        if a.which == 'b' and not os.path.exists(os.path.join(H, 'results', 'results_2012_a.json')) and a.out is None:
            sys.exit('freeze set A first')
        st = load(a.zip or os.path.join(a.data, f'set-{a.which}.zip'), a.outcomes or os.path.join(a.data, f'Outcomes-{a.which}.txt'))
        res = dict(dataset=f'PhysioNet Challenge 2012, set {a.which.upper()}', plan='PREANALYSIS_AMENDMENT_OPEN_DATA.md (2026-10-10), Part I',
                   cohort=dict(records=len(st), with_H1_H2_H3=int(st[['H1', 'H2', 'H3']].notna().all(axis=1).sum()), with_creatinine=int(st.R.notna().sum()),
                               with_lactate=int(st.L.notna().sum()), deaths=int(st.death.sum()), by_icu={int(k): int(v) for k, v in st.ICUType.value_counts().sort_index().items()}))
        res.update(analyse(st, rng, a.quick))
        # margins kept for the negative control (aggregate only)
        T = table(st.dropna(subset=['H1', 'R']).H1, st.dropna(subset=['H1', 'R']).R, 3, 3); res['_HR_counts'] = T.astype(int).tolist()
        out = a.out or os.path.join(H, 'results', f'results_2012_{a.which}.json'); os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)); print('wrote', out)
    elif a.which == 'compare':
        A = json.load(open(os.path.join(H, 'results', 'results_2012_a.json'))); Bj = json.load(open(os.path.join(H, 'results', 'results_2012_b.json')))
        TA, TB = np.array(A['_HR_counts'], float), np.array(Bj['_HR_counts'], float)
        r = regime(State(TA + 1e-9, TA.sum()), State(TB + 1e-9, TB.sum()))
        dec = lambda R: dict(B_not_glued=R['B_composition']['decision']['answers_not_glued'], B_latent_markov=R['B_composition']['decision']['latent_markov_fits'],
                             C_multiset=R['C_path_outcome']['p_multiset_vs_x1'] < 0.05, C_order=R['C_path_outcome']['p_order_vs_multiset'] < 0.05,
                             D_contextual=R['D_contextuality_icu']['contextual'], D_sofa_contextual=R['D_sensitivity_sofa']['contextual'], E=R['E_selection']['decision'])
        dA, dB = dec(A), dec(Bj)
        res = dict(negative_control=dict(kind=r['kind'], stress=round(r['stress'], 3), expected='bend, stress < 1'),
                   replication={k: dict(A=dA[k], B=dB[k], replicated=dA[k] == dB[k]) for k in dA})
        json.dump(res, open(os.path.join(H, 'results', 'results_2012_compare.json'), 'w'), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
