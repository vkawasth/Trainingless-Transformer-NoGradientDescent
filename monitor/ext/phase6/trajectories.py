"""Section 2 and 3 of PREANALYSIS_COLLAPSE.md: the five trajectories through the tracker, and the pre-registered forecasting test.
Usage: python3 trajectories.py   -> results/trajectories.json, results/forecast_test.json"""
import os, sys, io, json, zipfile, itertools
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); M = os.path.abspath(os.path.join(H, '..', '..')); EXT = os.path.join(M, 'ext')
for p in (M, os.path.join(EXT, 'phase4'), os.path.join(EXT, 'open_clinical'), os.path.join(EXT, 'c19')): sys.path.insert(0, p)
import fan_tracker as FT
SEED = 20261011; DATA = os.environ.get('PHYSIONET_DIR', '/home/claude/phase2/data/challenge2012')

def compact(tr):
    keep = ('step', 'n', 'f', 'g', 'tear_margin', 'break_margin', 'velocity', 'tear_time', 'break_time', 'drift', 'decided_flips', 'point_flips',
            'fan', 'fan_change', 'chamber_posterior_modal', 'decided')
    return [{k: r[k] for k in keep if k in r} for r in tr]

def gss(rng):
    from site_separation import ITEMS, YV, SHORT, wtable
    d = pd.read_csv(os.path.join(M, 'data', 'gss', 'gss_extract.csv'), low_memory=False); d = d[d.year >= 1985]; out = {}
    for w, cols in (('X', ITEMS), ('Y', [YV[i] for i in ITEMS])):
        for a, b in itertools.combinations(cols, 2):
            tabs, ne, yrs = [], [], []
            for yr, s in d.groupby('year'):
                if a not in s or b not in s: continue
                T, neff, n = wtable(s, a, b)
                if n >= 200: tabs.append(T); ne.append(neff); yrs.append(int(yr))
            inv = {v: k for k, v in YV.items()}; key = f"{w}|{SHORT[inv.get(a, a)]}|{SHORT[inv.get(b, b)]}"
            out[key] = compact(FT.track(tabs, neff=ne, rng=rng, labels=yrs))
    return out

def c19(rng):
    import c19 as C, tempfile
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        for strand, W in (('uk', C.load_uk(os.path.join(EXT, 'c19', 'raw', 'uk'))[0]), ('ie', C.load_ie(os.path.join(EXT, 'c19', 'raw', 'ie_archive.zip'), tmp)[0])):
            tabs = []
            for t in sorted(W):
                s = W[t].dropna(subset=['D', 'A']); T = np.zeros((3, 3)); np.add.at(T, (s.D.astype(int).values, s.A.astype(int).values), 1); tabs.append(T)
            res[strand] = compact(FT.track(tabs, rng=rng, labels=[f'W{t}' for t in sorted(W)]))
    return res

def tower(rng):
    import tower_icu as TI
    wp4 = json.load(open(os.path.join(EXT, 'phase4', 'results', 'wp4_tower.json')))['arms']
    A = pd.read_csv(os.path.join(TI.CACHE, 'set_a.csv')); B = pd.read_csv(os.path.join(TI.CACHE, 'set_b.csv'))
    import wp4_tower as W4
    out = {}
    for name, (f, e, t) in W4.arms(A, B).items():
        th = t.dropna(subset=['H1']); g = np.bincount(th.H1.astype(int), minlength=3) / len(th); path = wp4[name]['variants']['T-fam']['tube']
        recs = []
        for n_, f1 in path:
            W = FT.wall_table(np.array([f1, 1 - f1]), g, 1e9, len(th)); fan = FT.fan_type(np.array([f1, 1 - f1]), g)
            recs.append(dict(n=n_, f1=f1, wall_f1_g0=round(f1 - g[0], 5), wall_interval=[round(W[0, 1], 5), round(W[0, 2], 5)], decided=W[:, 3].astype(int).tolist(),
                             fan=fan, tear_margin=round(float(np.abs(W[:, 0]).min()), 5)))
        out[name] = dict(g=g.round(4).tolist(), observed=round(float(th.death.mean()), 4), path=recs,
                         fan_changes=sum(a['fan'] != b['fan'] for a, b in zip(recs, recs[1:])))
    return out

def icu_windows():
    """MAP band x HR band over all patient-hours in each 6-hour window, decedents and survivors (aggregate counts only)"""
    T = np.zeros((2, 8, 3, 3))
    for which in ('a', 'b'):
        oc = pd.read_csv(os.path.join(DATA, f'Outcomes-{which}.txt')).set_index('RecordID')['In-hospital_death']; z = zipfile.ZipFile(os.path.join(DATA, f'set-{which}.zip'))
        for name in z.namelist():
            if not name.endswith('.txt'): continue
            d = pd.read_csv(io.BytesIO(z.read(name))); hh = d.Time.str.split(':', expand=True).astype(float); d['h'] = hh[0].astype(int)
            rid = int(d[(d.Parameter == 'RecordID')].Value.iloc[0]) if (d.Parameter == 'RecordID').any() else -1
            if rid not in oc.index: continue
            mp = d[(d.Parameter == 'MAP') & d.Value.between(10, 250)].groupby('h').Value.median().combine_first(d[(d.Parameter == 'NIMAP') & d.Value.between(10, 250)].groupby('h').Value.median())
            hr = d[(d.Parameter == 'HR') & d.Value.between(10, 300)].groupby('h').Value.median(); both = pd.concat([mp.rename('m'), hr.rename('r')], axis=1).dropna()
            both = both[(both.index >= 0) & (both.index < 48)]
            np.add.at(T, (int(oc[rid]), (both.index.values // 6), np.digitize(both.m.values, (65, 75)), np.digitize(both.r.values, (60, 100.0001))), 1)
    return T
def icu(rng):
    T = icu_windows(); out = dict(counts={g: T[k].astype(int).tolist() for k, g in ((0, 'survivors'), (1, 'decedents'))})
    for k, g in ((0, 'survivors'), (1, 'decedents')): out[g] = compact(FT.track(list(T[k]), rng=rng, labels=[f'{6*w}-{6*w+6}h' for w in range(8)]))
    # between the two contexts at each window: the same information (MAP x HR), different outcome groups
    out['between_groups'] = [dict(window=f'{6*w}-{6*w+6}h', **FT.drift_split((T[0, w] + .5) / (T[0, w] + .5).sum(), (T[1, w] + .5) / (T[1, w] + .5).sum())) for w in range(8)]
    return out

def llm(rng):
    import llm_analysis as LA
    out = {}
    for task in LA.TASKS:
        w = LA.outcomes(task); tabs = [LA.table(w['35M'], w['4M']), LA.table(w['35J'], w['4J'])]
        out[task] = compact(FT.track(tabs, rng=rng, labels=['March', 'June']))
    return out

def forecast(G, rng):
    """section 3: does interaction drift or tear time forecast the next decided chamber change beyond margin velocity?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    rows = []
    for sysid, tr in G.items():
        for k in range(1, len(tr) - 1):
            r, nx = tr[k], tr[k + 1]
            rows.append(dict(system=sysid, year=r['step'], next_year=nx['step'], event=int(nx['decided_flips'] > 0), point_event=int(nx['point_flips'] > 0),
                             v=r['velocity'], mB=r['break_margin'], inter=r['drift']['interaction'], ltt=np.log(max(r['tear_time'], 1e-6))))
    F = pd.DataFrame(rows); tr_, te = F[F.next_year <= 2004], F[F.next_year >= 2006]
    def run(ev):
        P = {}
        for name, cols in (('B', ['v', 'mB']), ('B+inter', ['v', 'mB', 'inter']), ('B+tear', ['v', 'mB', 'ltt']), ('B+both', ['v', 'mB', 'inter', 'ltt'])):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(tr_[cols].values, tr_[ev].values)
            p = np.clip(m.predict_proba(te[cols].values)[:, 1], 1e-12, 1 - 1e-12); y = te[ev].values; P[name] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        groups = [np.where(te.system.values == s)[0] for s in te.system.unique()]; res = {}
        for name in ('B+inter', 'B+tear', 'B+both'):
            d = []
            for _ in range(2000):
                idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]); d.append((P['B'][idx] - P[name][idx]).mean())
            res[name] = dict(gain=round(float(np.mean(d)), 6), ci=[round(float(np.percentile(d, 2.5)), 6), round(float(np.percentile(d, 97.5)), 6)], forecasts=bool(np.percentile(d, 2.5) > 0))
        res.update(events_train=int(tr_[ev].sum()), events_test=int(te[ev].sum()), n_train=len(tr_), n_test=len(te)); return res
    out = dict(decided=run('event'))
    out['decided']['underpowered'] = out['decided']['events_test'] < 15
    if out['decided']['underpowered']: out['secondary_point_estimate'] = run('point_event')
    out['descriptive'] = dict(event_rate_by_tear_time_quartile={str(k): round(float(v), 4) for k, v in F.groupby(pd.qcut(F.ltt, 4, labels=False)).event.mean().items()},
                              event_rate_by_interaction_quartile={str(k): round(float(v), 4) for k, v in F.groupby(pd.qcut(F.inter, 4, labels=False)).event.mean().items()})
    return out

if __name__ == '__main__':
    rng = np.random.default_rng(SEED); res = dict(plan='PREANALYSIS_COLLAPSE.md sections 2-3')
    res['gss'] = gss(rng); print('gss', len(res['gss']), flush=True)
    res['c19'] = c19(rng); print('c19', flush=True)
    res['tower'] = tower(rng); print('tower', flush=True)
    res['icu'] = icu(rng); print('icu', flush=True)
    res['llm'] = llm(rng); print('llm', flush=True)
    os.makedirs(os.path.join(H, 'results'), exist_ok=True)
    json.dump(res, open(os.path.join(H, 'results', 'trajectories.json'), 'w'), default=lambda o: o.item() if hasattr(o, 'item') else (None if o is None else str(o)))
    fc = forecast(res['gss'], rng); json.dump(fc, open(os.path.join(H, 'results', 'forecast_test.json'), 'w'), indent=1); print(json.dumps(fc, indent=1))
