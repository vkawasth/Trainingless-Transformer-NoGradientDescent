"""C19PRC: analyses of PREANALYSIS_AMENDMENT_C19PRC_UK.md (UK primary) and PREANALYSIS_AMENDMENT_C19PRC_IE.md (Irish replication).
Usage: python3 c19.py uk|ie [--quick]      (reads raw/uk/*.sav or raw/ie_archive.zip; writes results/results_c19_<strand>.json)"""
import os, sys, json, re, zipfile, argparse, itertools
import numpy as np, pandas as pd, pyreadstat
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, '..', 'open_clinical', 'lib'))
from analyses import geometry, composition, path_outcome, cbd_strata, selection, table
from monitor import gluing_gap
import latentT
SEED = 20261009

def band(x):
    x = pd.to_numeric(x, errors='coerce'); x = x.where((x >= 0) & (x <= 27))
    return pd.Series(np.where(x.isna(), np.nan, np.digitize(x, [5, 10])), index=x.index, dtype=float)

def item_sum(df, cols, top):
    v = df[cols].apply(pd.to_numeric, errors='coerce'); v = v.where((v >= 0) & (v <= 3))
    return v.sum(axis=1, min_count=len(cols)).where(v.notna().all(axis=1))

def load_uk(folder):
    f = {w: os.path.join(folder, n) for w, n in (('12', 'C19PRC_UKW1W2_archive_final.sav'), ('3', 'C19PRC_UK_W3_archive_final.sav'),
                                                 ('4', 'C19PRC_UK_W4_archive_final.sav'), ('5', 'C19PRC_UKW5_archive_final.sav'), ('6', 'C19PRC_UK_W6_archive_final.sav'))}
    frames = {}
    for w, p in f.items():
        d, _ = pyreadstat.read_sav(p); d['pid'] = d['pid'].astype(str); frames[w] = d
    out = {}
    def tot(d, t, kind):
        names = {'D': [f'W{t}_Depression_Total', f'W{t}_Dep_Total'], 'A': [f'W{t}_GAD_Total']}[kind]
        for n in names:
            if n in d.columns: return band(d[n])
        pat = {'D': rf'^W{t}_Dep_?\d$', 'A': rf'^W{t}_GAD_?\d$'}[kind]; cols = [c for c in d.columns if re.match(pat, c)]
        return band(item_sum(d, cols, 3))
    W12 = frames['12']
    for t in (1, 2):
        o = pd.DataFrame({'pid': W12.pid, 'D': tot(W12, t, 'D'), 'A': tot(W12, t, 'A')})   # WAVE_1/WAVE_2 are empty in the archive file:
        out[t] = o[o.D.notna() | o.A.notna()].copy()                                          # a wave's respondents are those with its scores
    for t in (3, 4, 5, 6):
        d = frames[str(t)]; out[t] = pd.DataFrame({'pid': d.pid, 'D': tot(d, t, 'D'), 'A': tot(d, t, 'A')})
        tc = {3: 'W3_Type', 4: 'W4_Type', 6: 'W6_Type'}.get(t)
        if tc: out[t]['ctx'] = d[tc].map({3: {0: 'recontact', 1: 'fresh'}, 4: {1: 'recontact', 2: 'fresh'}, 6: {1: 'recontact', 0: 'fresh'}}[t])
    w3 = frames['3'][['pid', 'W3_weight']].rename(columns={'W3_weight': 'w2nd'})
    y = frames['6'][['pid', 'W6_Suicide_Ideation']].rename(columns={'W6_Suicide_Ideation': 'y'}); y['y'] = y.y.where(y.y.isin([0, 1]))
    return out, w3, y, dict(D_ctx_waves=(3, 4, 6), outcome='W6_Suicide_Ideation (lifetime item)')

def load_ie(zpath, tmp):
    z = zipfile.ZipFile(zpath); p = os.path.join(tmp, 'allw.sav'); open(p, 'wb').write(z.read('Mental Health ASAP Datasets/MHASAP.ALLWAVES.sav'))
    d, meta = pyreadstat.read_sav(p); d['pid'] = d['pid'].astype(str); out = {}
    pre = {1: '', 2: 'W2_', 3: 'W3_', 4: 'W4_', 5: 'W5_'}
    for t in range(1, 6):
        dep = [c for c in d.columns if re.fullmatch(rf'{pre[t]}DEP[1-9]', c)]
        gad = [c for c in d.columns if re.fullmatch(rf'{pre[t]}GAD_?[1-7]', c)]
        D = band(item_sum(d, dep, 3)); A = band(item_sum(d, gad, 3))
        o = pd.DataFrame({'pid': d.pid, 'D': D, 'A': A}); out[t] = o[o.D.notna() | o.A.notna()].copy()
    rc = {2: ('W2_Wave1_Recontact', None), 4: ('W4_Wave1_Recontact', {1: 'recontact', 0: 'fresh'})}
    vl = meta.variable_value_labels
    if 'W2_Wave1_Recontact' in d.columns:
        lab = vl.get('W2_Wave1_Recontact') or {}
        m = {k: ('recontact' if 'recontact' in str(v).lower() else 'fresh') for k, v in lab.items()} if lab else None
        s = d['W2_Wave1_Recontact']
        out[2]['ctx'] = (s.map(m) if m else s.astype(str).str.lower().map(lambda x: 'recontact' if 'recontact' in x else ('fresh' if 'fresh' in x else np.nan))).reindex(out[2].index)
    out[4]['ctx'] = d['W4_Wave1_Recontact'].map({1: 'recontact', 0: 'fresh'}).reindex(out[4].index)
    w1 = set(out[1].pid)                                                  # W5 has no recontact flag: recontact = took part at W1
    out[5]['ctx'] = np.where(out[5].pid.isin(w1), 'recontact', 'fresh')
    w3 = d[['pid', 'W3_Weight_Variable']].rename(columns={'W3_Weight_Variable': 'w2nd'})
    yv = d['W4_Suicidal_Ideation_1week']; lab = vl.get('W4_Suicidal_Ideation_1week') or {}
    no = [k for k, v in lab.items() if str(v).strip().lower().startswith('no')]
    y = pd.DataFrame({'pid': d.pid, 'y': np.where(yv.isna(), np.nan, (~yv.isin(no)).astype(float))})
    return out, w3, y, dict(D_ctx_waves=(2, 4, 5), outcome='W4_Suicidal_Ideation_1week', no_codes=no)

def analyse(W, w3, y, info, rng, quick=False):
    B = 50 if quick else 400; res = {'cohort': {f'W{t}': dict(n=int(len(W[t])), D=int(W[t].D.notna().sum()), A=int(W[t].A.notna().sum())) for t in W}}
    waves = sorted(W)
    ctx = {f'W{t}': (W[t].dropna(subset=['D', 'A']).D, W[t].dropna(subset=['D', 'A']).A) for t in waves}
    res['A_coupling_geometry'] = geometry(ctx, 3, 3, pairs='consecutive')
    wide = None
    for t in waves:
        x = W[t][['pid', 'D']].dropna().drop_duplicates('pid').rename(columns={'D': f'D{t}'})
        wide = x if wide is None else wide.merge(x, on='pid', how='outer')
    b1 = wide.dropna(subset=['D1', 'D2', 'D3'])
    res['B1_composition_W1_W3'] = composition(b1.D1, b1.D2, b1.D3, rng, B=B, starts=5 if quick else 30)
    runs = [tuple(range(s, s + 4)) for s in range(1, len(waves) - 2)] + ([(1, 2, 3, 5)] if len(waves) == 5 else [])
    res['B2_four_waves'] = 'not estimable: no four-wave run with 200 complete cases'
    for run in runs:
        c = wide.dropna(subset=[f'D{t}' for t in run])
        if len(c) >= 200:
            N = np.zeros((3,) * 4); np.add.at(N, tuple(c[f'D{t}'].astype(int).values for t in run), 1)
            r = latentT.test(N, 4, shared=True, B=60 if quick else 200, seed=1, starts=8 if quick else 20)
            s = latentT.test(N, 4, shared=False, B=30 if quick else 100, seed=2, starts=8 if quick else 20)
            res['B2_four_waves'] = dict(waves=list(run), n=len(c), latent_markov_shared=r, latent_markov_free_response=s,
                                        decision='latent Markov suffices' if r['p_boot'] > 0.05 else 'longer memory than a latent Markov chain')
            break
    allc = wide.dropna(subset=[f'D{t}' for t in waves])
    if len(allc) >= 200 and len(waves) >= 4:
        gaps = {}
        for t in waves[2:]:
            x1, x2, x3 = (allc[f'D{u}'].astype(int).values for u in (t - 2, t - 1, t)); g = gluing_gap(x1, x2, x3)
            r = composition(x1, x2, x3, rng, B=B, starts=3)
            gaps[f'I(D{t-2};D{t}|D{t-1})'] = dict(gap=round(g, 4), markov_null95=r['markov_null95'], p_markov=r['p_markov'])
        res['B3_all_waves_gaps'] = dict(n=len(allc), gaps=gaps)
    else: res['B3_all_waves_gaps'] = f'not estimable: {len(allc)} complete on all waves'
    cc = b1.merge(y.dropna().drop_duplicates('pid'), on='pid')
    res['C_path_outcome'] = (dict(path_outcome(cc.D1, cc.D2, cc.D3, cc.y.astype(int), rng, B=B * 5), outcome=info['outcome'])
                             if len(cc) >= 100 and cc.y.sum() >= 10 else f'not estimable: n={len(cc)}, yes={int(cc.y.sum())}')
    strata = {}
    for t in info['D_ctx_waves']:
        d = W[t].dropna(subset=['D', 'A']); d = d[d.ctx.isin(['recontact', 'fresh'])] if 'ctx' in d else d.iloc[0:0]
        strata[f'W{t}'] = (table(d[d.ctx == 'recontact'].D, d[d.ctx == 'recontact'].A, 3, 3), table(d[d.ctx == 'fresh'].D, d[d.ctx == 'fresh'].A, 3, 3))
    res['D_panel_conditioning'] = cbd_strata(strata, (3, 3), rng, B=B)
    e = W[1].dropna(subset=['D', 'A'])[['pid', 'D', 'A']].drop_duplicates('pid').merge(W[3][['pid', 'D']].dropna().drop_duplicates('pid').rename(columns={'D': 'D3'}), on='pid', how='left')
    e = e.merge(w3.dropna().drop_duplicates('pid'), on='pid', how='left')
    res['E_attrition'] = selection(e.D, e.A, e.D3, rng, B=B, extra_weights=e.w2nd.fillna(1.0).values)
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('strand'); ap.add_argument('--quick', action='store_true'); ap.add_argument('--out')
    ap.add_argument('--uk', default=os.path.join(H, 'raw', 'uk')); ap.add_argument('--ie', default=os.path.join(H, 'raw', 'ie_archive.zip'))
    a = ap.parse_args(); rng = np.random.default_rng(SEED)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        W, w3, y, info = load_uk(a.uk) if a.strand == 'uk' else load_ie(a.ie, tmp)
        res = dict(strand=a.strand, plan='PREANALYSIS_AMENDMENT_C19PRC_UK.md / _IE.md (2026-10-10)', info={k: str(v) for k, v in info.items()})
        res.update(analyse(W, w3, y, info, rng, a.quick))
    out = a.out or os.path.join(H, 'results', f'results_c19_{a.strand}.json'); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)); print('wrote', out)
