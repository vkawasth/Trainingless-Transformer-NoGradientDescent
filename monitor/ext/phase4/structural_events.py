"""WP3 (PREANALYSIS_PHASE4.md): a pipeline that turns a sequence of coupling systems into structural events -- blow-up / blow-down
(vertex count of the coupling polytope up / down: a change of toric type), chamber transition (wall-sign flips), break appearing or
healing (a zero margin), steepness change -- and tests their association with an external state by permutation.
Usage: python3 structural_events.py  -> results/wp3_events.json"""
import os, sys, json
import numpy as np, pandas as pd
import geom
H = os.path.dirname(os.path.abspath(__file__)); M = os.path.abspath(os.path.join(H, '..', '..')); SEED = 20261010

def events(f0, g0, f1, g1):
    """structural events between two systems with margins (f0, g0) -> (f1, g1)"""
    v0, v1 = geom.n_vertices(f0, g0), geom.n_vertices(f1, g1); b0, b1 = geom.breaks(f0, g0), geom.breaks(f1, g1)
    return dict(blow_up=int(v1 > v0), blow_down=int(v1 < v0), chamber_flip=int((geom.signs(f0, g0) != geom.signs(f1, g1)).any()),
                break_appears=int(b1 > b0), break_heals=int(b1 < b0))

def perm_assoc(event, y, strata=None, rng=None, B=2000):
    """difference in mean external state with and without the event; permutation p (two-sided), within strata if given"""
    event, y = np.asarray(event, int), np.asarray(y, float); obs = y[event == 1].mean() - y[event == 0].mean() if 0 < event.sum() < len(event) else np.nan
    if np.isnan(obs): return dict(n_event=int(event.sum()), diff=None, p=None)
    groups = [np.arange(len(y))] if strata is None else [np.where(strata == s)[0] for s in np.unique(strata)]
    null = []
    for _ in range(B):
        e = event.copy()
        for g in groups: e[g] = rng.permutation(e[g])
        null.append(y[e == 1].mean() - y[e == 0].mean())
    return dict(n_event=int(event.sum()), rate_with=round(float(y[event == 1].mean()), 4), rate_without=round(float(y[event == 0].mean()), 4),
                diff=round(float(obs), 4), p=float((np.sum(np.abs(null) >= abs(obs)) + 1) / (B + 1)))

def physionet(rng):
    import wp1_clinical as W
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    D = pd.concat([W.keep(pd.read_csv(f'{W.CACHE}/phase4_set_{s}.csv')) for s in ('a', 'b')], ignore_index=True)
    S = W.systems(D); E = pd.DataFrame([events(S[1][0][k], S[1][1][k], S[2][0][k], S[2][1][k]) for k in range(len(D))])
    F, _ = W.features(D); risk = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(F[W.MARG].values, D.death.values).predict_proba(F[W.MARG].values)[:, 1]
    dec = pd.qcut(risk, 10, labels=False).astype(int); y = D.death.values
    return {ev: dict(crude=perm_assoc(E[ev], y, rng=rng), within_margin_risk_deciles=perm_assoc(E[ev], y, strata=dec, rng=rng)) for ev in E.columns}, len(D)

def gss(rng):
    import wp1_gss as G
    sys.path.insert(0, M); from site_separation import ITEMS, YV, wtable
    d = pd.read_csv(os.path.join(M, 'data', 'gss', 'gss_extract.csv'), low_memory=False); d = d[d.year >= 1985]
    p = d[d.partyid.isin(range(7)) & d.wtssps.notna()]; rep = p.groupby('year').apply(lambda s: np.average(s.partyid.isin([4, 5, 6]), weights=s.wtssps))
    F = G.steps(); F['d_rep'] = [rep.get(b, np.nan) - rep.get(a, np.nan) for a, b in zip(F.year, F.next_year)]
    yr = F.groupby(['year', 'next_year']).agg(flip_share=('event', 'mean'), d_rep=('d_rep', 'first')).dropna().reset_index()
    from scipy.stats import spearmanr
    r = spearmanr(yr.flip_share, yr.d_rep.abs()).correlation; null = [spearmanr(rng.permutation(yr.flip_share.values), yr.d_rep.abs()).correlation for _ in range(2000)]
    return dict(year_steps=len(yr), spearman_flip_share_vs_abs_d_rep=round(float(r), 3), p=float((np.sum(np.abs(null) >= abs(r)) + 1) / 2001),
                table=yr.round(4).to_dict(orient='records'))

def llm():
    R = json.load(open(os.path.join(M, 'llm_results.json'))); rows = []
    for task, r in R.items():
        mg = r['margins']; d_inc = np.mean([mg[f'{m}J'][1] - mg[f'{m}M'][1] for m in ('35', '4')])
        d_inv = np.mean([mg[f'{m}J'][2] - mg[f'{m}M'][2] if len(mg[f'{m}M']) > 2 else 0 for m in ('35', '4')])
        rows.append(dict(task=task, regime=r['regime']['kind'], walls_crossed=len(r['regime'].get('walls_crossed', [])), d_incorrect=round(float(d_inc), 3), d_invalid=round(float(d_inv), 3)))
    return dict(note='8 tasks; descriptive only, no test', rows=rows)

if __name__ == '__main__':
    rng = np.random.default_rng(SEED); res = dict(plan='PREANALYSIS_PHASE4.md WP3')
    res['physionet_death'], res['physionet_n'] = physionet(rng); res['gss_partisan'] = gss(rng); res['llm_hallucination_proxy'] = llm()
    json.dump(res, open(os.path.join(H, 'results', 'wp3_events.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ('physionet_death', 'physionet_n')}, indent=1)); print({k: v for k, v in res['gss_partisan'].items() if k != 'table'}); print(res['llm_hallucination_proxy'])
