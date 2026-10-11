"""GSS: the spending items asked in two randomized wordings (X: 'welfare', 'solving the problems of the big cities', ...; Y: 'assistance
to the poor', 'assistance to big cities', ...), 1985-2024, and the 2006-2014 three-wave panels.
Responses: 1 too little, 2 about right, 3 too much. Weights: wtssps; effective sample size by Kish's formula.
Systems: contents (A, B) = two spending items; contexts = wording X / wording Y within a year, or years within a wording."""
import os, sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monitor import State, regime, cbd_rank2, cbd_split, churn, gluing_gap, bootstrap, tv
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'gss')
d = pd.read_csv(os.path.join(G, 'gss_extract.csv'), low_memory=False)
d = d[d.year >= 1985]
PAIRS = {'welfare-cities': (('natfare', 'natcity'), ('natfarey', 'natcityy')),
         'welfare-race': (('natfare', 'natrace'), ('natfarey', 'natracey')),
         'health-education': (('natheal', 'nateduc'), ('nathealy', 'nateducy'))}

def wtable(sub, a, b):
    m = sub[a].isin([1, 2, 3]) & sub[b].isin([1, 2, 3]) & sub['wtssps'].notna()
    s = sub[m]; w = s['wtssps'].values
    T = np.zeros((3, 3))
    np.add.at(T, (s[a].astype(int).values - 1, s[b].astype(int).values - 1), w)
    neff = w.sum() ** 2 / (w ** 2).sum() if len(w) else 0
    return T, neff, len(w)

out = {'years': sorted(int(y) for y in d.year.unique())}
for name, (X, Y) in PAIRS.items():
    res = {'X': {}, 'Y': {}, 'cbd': {}, 'steps': {'X': [], 'Y': []}}
    states = {'X': {}, 'Y': {}}
    for yr in out['years']:
        sub = d[d.year == yr]
        for ctx, (a, b) in (('X', X), ('Y', Y)):
            T, neff, n = wtable(sub, a, b)
            if n < 100: continue
            S = State(T + 1e-9, neff); states[ctx][yr] = S
            res[ctx][yr] = dict(n=n, neff=round(neff, 1), f=[round(x, 4) for x in S.f], g=[round(x, 4) for x in S.g], mT=round(S.mT, 4), mB=round(S.mB, 4))
        if yr in states['X'] and yr in states['Y']:
            TX, TY = states['X'][yr].T, states['Y'][yr].T
            bX = np.array([[TX[0, 0], TX[0, 1:].sum()], [TX[1:, 0].sum(), TX[1:, 1:].sum()]])     # 'too little' (0) vs not (1)
            bY = np.array([[TY[0, 0], TY[0, 1:].sum()], [TY[1:, 0].sum(), TY[1:, 1:].sum()]])
            cb = cbd_rank2(bX, bY)
            res['cbd'][yr] = dict(contextual=cb['contextual'], s=round(cb['s_odd'], 4), Delta=round(cb['Delta'], 4), CNTX=round(cb['CNTX'], 4),
                                  direct_A=round(cb['direct_A'], 4), direct_B=round(cb['direct_B'], 4))
            sp = cbd_split(TX, TY)
            so = cbd_split(TX, TY, ordinal=True)
            res['cbd'][yr].update(split_contextual=sp['contextual'], split_deficit=round(sp['deficit'], 5),
                                  cuts_contextual=so['contextual'], cuts_deficit=round(so['deficit'], 5))
    for ctx in ('X', 'Y'):
        yrs = sorted(states[ctx])
        for y0, y1 in zip(yrs, yrs[1:]):
            rg = regime(states[ctx][y0], states[ctx][y1])
            res['steps'][ctx].append(dict(frm=y0, to=y1, kind=rg['kind'], stress=round(rg['stress'], 2), change=round(rg['change'], 4),
                                          mT=round(rg['mT'], 4), crossed=[[list(I), list(J)] for I, J in rg['walls_crossed']]))
        rg = regime(states[ctx][yrs[0]], states[ctx][yrs[-1]])
        res['span_' + ctx] = dict(frm=yrs[0], to=yrs[-1], kind=rg['kind'], stress=round(rg['stress'], 2), crossed=[[list(I), list(J)] for I, J in rg['walls_crossed']])
    out[name] = res
    kinds = {c: pd.Series([s['kind'] for s in res['steps'][c]]).value_counts().to_dict() for c in ('X', 'Y')}
    tears = {c: [(s['frm'], s['to'], s['crossed']) for s in res['steps'][c] if s['kind'] == 'tear'] for c in ('X', 'Y')}
    nctx = sum(v['contextual'] for v in res['cbd'].values()); nsp = [y for y, v in res['cbd'].items() if v['split_contextual']]
    print(f"== {name}: step regimes X {kinds['X']} Y {kinds['Y']} | span X {res['span_X']['kind']} Y {res['span_Y']['kind']} | CbD contextual in {nctx} of {len(res['cbd'])} years ('too little' vs rest); all splits: {nsp}; order-respecting cuts: {[y for y, v in res['cbd'].items() if v['cuts_contextual']]}")
    for c in ('X', 'Y'):
        for t in tears[c]: print(f"   tear in wording {c}: {t[0]}->{t[1]} walls {t[2]}")
    print("   direct influence of wording on A ('too little'), median over years:", round(float(np.median([v['direct_A'] for v in res['cbd'].values()])), 3),
          "| on B:", round(float(np.median([v['direct_B'] for v in res['cbd'].values()])), 3),
          "| max s - Delta:", round(max(v['s'] - v['Delta'] for v in res['cbd'].values()), 3))

# ---------------------------------------------------------------- panels: churn, gluing gap, outcome
import pyreadr
pan = {}
for pnl in ('06', '08', '10'):
    p = list(pyreadr.read_r(os.path.join(G, f'gss_panel{pnl}_long.rda')).values())[0]
    CODE = {'Too Little': 1, 'About Right': 2, 'Too Much': 3}
    for c in ('natfare', 'natfarey'): p[c] = p[c].astype(object).map(CODE).astype(float)
    p['pres08'] = p['pres08'].astype(object).map({'Obama': 1, 'Mccain': 2}).astype(float)
    p['firstid'] = p['firstid'].astype(str)
    res = {}
    for item in ('natfare', 'natfarey'):
        w = p.pivot_table(index='firstid', columns='wave', values=item, aggfunc='first')
        w = w[w.isin([1, 2, 3]).all(axis=1)].astype(int)
        if len(w) < 100: continue
        x1, x2, x3 = w[1].values, w[2].values, w[3].values
        c12, c23, c13 = churn(x1, x2, [1, 2, 3]), churn(x2, x3, [1, 2, 3]), churn(x1, x3, [1, 2, 3])
        gg = gluing_gap(x1, x2, x3); ci = bootstrap(gluing_gap, [x1, x2, x3], B=400, seed=int(pnl))
        # null for the gap: the same three-wave data resampled under the glued (Markov) law
        rng = np.random.default_rng(7); nulls = []
        P12 = pd.crosstab(x1, x2, normalize='index'); P23 = pd.crosstab(x2, x3, normalize='index')
        for _ in range(400):
            a = rng.choice(x1, len(x1)); b = np.array([rng.choice(P12.columns, p=P12.loc[v].values) for v in a])
            c = np.array([rng.choice(P23.columns, p=P23.loc[v].values) for v in b]); nulls.append(gluing_gap(a, b, c))
        res[item] = dict(n=len(w), gap_p_markov=round(float(np.mean(np.array(nulls) >= gg)), 4), agree12=round(c12['agree'], 3), max12=round(c12['max_agree'], 3), excess12=round(c12['excess'], 3),
                         agree23=round(c23['agree'], 3), excess23=round(c23['excess'], 3), agree13=round(c13['agree'], 3), excess13=round(c13['excess'], 3),
                         tv12=round(c12['tv_margins'], 3), gap=round(gg, 4), gap_ci=[round(x, 4) for x in ci], gap_null95=round(float(np.percentile(nulls, 95)), 4))
    # outcome: 2008 presidential vote (1 Obama, 2 McCain), recorded in the panel; association with wave-1 welfare attitude
    # (unweighted; in the 2010 panel the vote was cast before wave 1, so this is an association, not a prediction)
    if 'pres08' in p.columns:
        q = p.pivot_table(index='firstid', columns='wave', values=['natfare', 'natfarey', 'pres08'], aggfunc='first')
        v = q['pres08'].bfill(axis=1).iloc[:, 0]
        res['outcome'] = {}
        for item in ('natfare', 'natfarey'):
            att = q[item][1]; ok = v.isin([1, 2]) & att.isin([1, 2, 3])
            tab = pd.crosstab(att[ok].astype(int), v[ok].astype(int), normalize='index')
            res['outcome'][item] = dict(n=int(ok.sum()), obama_share_by_wave1_attitude={int(k): round(float(tab.loc[k, 1]), 3) for k in tab.index})
    pan[pnl] = res
    print(f"== panel 20{pnl}: " + " | ".join(f"{it}: n={r['n']}, agree w1-w2 {r['agree12']} (max {r['max12']}, excess {r['excess12']}), gluing gap {r['gap']} CI {r['gap_ci']} null95 {r['gap_null95']} p {r['gap_p_markov']}"
                                               for it, r in res.items() if it.startswith('nat')) + (f" | outcome {res['outcome']}" if 'outcome' in res else ""))
out['panels'] = pan
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gss_results.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
