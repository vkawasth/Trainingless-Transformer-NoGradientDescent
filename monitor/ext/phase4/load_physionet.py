"""Per-stay aggregates for phase 4 from the PhysioNet 2012 archives: hourly MAP x HR band tables for hours [0,24) and [24,48), and the
first-day lactate band. Writes a patient-level cache OUTSIDE the monitor folder ($PHYSIONET_DIR/cache/phase4_set_{a,b}.csv)."""
import os, io, zipfile, sys
import numpy as np, pandas as pd
DATA = os.environ.get('PHYSIONET_DIR', '/home/claude/phase2/data/challenge2012'); CACHE = os.path.join(DATA, 'cache')
def band_map(v): return np.digitize(v, (65, 75))
def band_hr(v): return np.digitize(v, (60, 100.0001))          # < 60, 60-100, > 100
def one(d):
    hh = d.Time.str.split(':', expand=True).astype(float); d = d.assign(t=hh[0] + hh[1] / 60, h=hh[0].astype(int))
    desc = d[d.t == 0].drop_duplicates('Parameter').set_index('Parameter').Value
    rec = dict(RecordID=int(desc.get('RecordID', -1)))
    m = d[(d.Parameter == 'MAP') & d.Value.between(10, 250)].groupby('h').Value.median()
    nm = d[(d.Parameter == 'NIMAP') & d.Value.between(10, 250)].groupby('h').Value.median()
    mp = m.combine_first(nm); hr = d[(d.Parameter == 'HR') & d.Value.between(10, 300)].groupby('h').Value.median()
    both = pd.concat([mp.rename('map'), hr.rename('hr')], axis=1).dropna()
    for w, (a, b) in enumerate(((0, 24), (24, 48)), 1):
        x = both[(both.index >= a) & (both.index < b)]; T = np.zeros((3, 3), int)
        np.add.at(T, (band_map(x['map'].values), band_hr(x['hr'].values)), 1)
        for i in range(3):
            for j in range(3): rec[f'W{w}_{i}{j}'] = int(T[i, j])
        rec[f'W{w}_n'] = int(T.sum())
    lac = d[(d.Parameter == 'Lactate') & (d.t <= 24)].sort_values('t').Value
    rec['L'] = float(np.digitize(lac.iloc[0], (2.0, 4.0))) if not lac.empty else np.nan
    return rec
if __name__ == '__main__':
    os.makedirs(CACHE, exist_ok=True)
    for which in ('a', 'b'):
        z = zipfile.ZipFile(os.path.join(DATA, f'set-{which}.zip'))
        rows = [one(pd.read_csv(io.BytesIO(z.read(n)))) for n in z.namelist() if n.endswith('.txt')]
        st = pd.DataFrame(rows).merge(pd.read_csv(os.path.join(CACHE, f'set_{which}.csv')), on='RecordID')     # tower cache: H1, R, ICUType, vent24, death, SOFA
        st.sort_values('RecordID').to_csv(os.path.join(CACHE, f'phase4_set_{which}.csv'), index=False); print(which, len(st))
