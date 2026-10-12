"""Part A of PREANALYSIS_CONTEXT_MAP.md: is the map from information A to outcome O the same in every context C?
The invariance model O _|_ C | A is a toric variety (binomials p_cao p_c'ao' - p_cao' p_c'ao); G2 = 2N * KL to it = 2N ln2 * I(O;C|A).
Toric decomposition: the (A, O) coupling polytope per context, observed and under the pooled (invariant) kernel.
Usage: python3 context_map.py synth | run  -> results/context_map.json"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import chi2
H = os.path.dirname(os.path.abspath(__file__)); M = os.path.abspath(os.path.join(H, '..', '..'))
sys.path.insert(0, M); sys.path.insert(0, os.path.join(H, '..', 'phase4')); sys.path.insert(0, os.path.join(H, '..', 'open_clinical'))
import geom
from monitor import bands, wall_signs
SEED = 20261011

def ci_fit(N):
    Nca, Nao, Na = N.sum(2), N.sum(0), N.sum((0, 2))
    return np.where(Na[None, :, None] > 0, Nca[:, :, None] * Nao[None, :, :] / np.maximum(Na, 1)[None, :, None], 0.0)
def g2(N, E): m = N > 0; return 2 * float((N[m] * np.log(N[m] / E[m])).sum())
def df_ci(N):
    d = 0
    for a in range(N.shape[1]):
        S = N[:, a, :]; r, s = (S.sum(1) > 0).sum(), (S.sum(0) > 0).sum(); d += max(r - 1, 0) * max(s - 1, 0)
    return d
def ci_test(N, rng, B=500):
    E = ci_fit(N); G = g2(N, E); d = df_ci(N); n = int(N.sum())
    res = dict(n=n, G2=round(G, 3), df=d, I_bits=round(G / (2 * n * np.log(2)), 5) if n else None,
               slice_share={str(a): round(g2(N[:, a:a + 1, :], E[:, a:a + 1, :]) / G, 3) if G > 0 else None for a in range(N.shape[1])})
    if d == 0: res['p'] = None; return res
    if E[E > 0].min() < 5:     # parametric bootstrap from the fitted invariant law
        P = (E / E.sum()).ravel(); null = [g2(Nb, ci_fit(Nb)) for Nb in (rng.multinomial(n, P).reshape(N.shape).astype(float) for _ in range(B))]
        res['p'] = float((np.sum(np.array(null) >= G) + 1) / (B + 1)); res['reference'] = 'bootstrap'
    else: res['p'] = float(chi2.sf(G, d)); res['reference'] = 'chi2'
    return res

def toric_type(a, o, kind):
    if kind == '3x3': return geom.n_vertices(np.asarray(a, float), np.asarray(o, float))
    import tower_icu as TI; return TI.toric(float(o[1]), np.asarray(a, float))[0]       # binary outcome x 3 information values
def decomposition(N, kind):
    """per context: observed toric type of (A, O), type under the invariant map, and whether the chamber is decided on the bands"""
    K = N.sum(0) / np.maximum(N.sum((0, 2)), 1)[:, None]; out = {}
    for c in range(N.shape[0]):
        n = N[c].sum()
        if n == 0: continue
        a, o = N[c].sum(1) / n, N[c].sum(0) / n; oinv = a @ K
        sg = wall_signs(bands(a, n), bands(o, n))
        out[c] = dict(n=int(n), type_observed=toric_type(a, o, kind), type_invariant=toric_type(a, oinv, kind), chamber_decided=bool((sg != 0).all()))
    return out

PARTY = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2}
def gss(rng):
    from site_separation import ITEMS, YV, SHORT
    d = pd.read_csv(os.path.join(M, 'data', 'gss', 'gss_extract.csv'), low_memory=False); d = d[(d.year >= 1985) & d.partyid.isin(range(7))]
    d['O'] = d.partyid.map(PARTY).astype(int); res = {}; fisher_ps = []
    for item in ITEMS:
        rows = {}; Gsum = dfsum = 0; types = dict(differ=0, explained=0, years=0, decided_both=0)
        for yr, s in d.groupby('year'):
            if item not in s or YV[item] not in s: continue
            N = np.zeros((2, 3, 3))
            for c, col in enumerate((item, YV[item])):
                t = s[s[col].isin([1, 2, 3])]; np.add.at(N, (c, t[col].astype(int).values - 1, t.O.values), 1)
            if (N.sum((1, 2)) < 100).any(): continue
            r = ci_test(N, rng); rows[int(yr)] = r; Gsum += r['G2']; dfsum += r['df']
            dec = decomposition(N, '3x3'); types['years'] += 1
            if dec[0]['type_observed'] != dec[1]['type_observed']:
                types['differ'] += 1; types['explained'] += int(all(v['type_invariant'] == v['type_observed'] for v in dec.values()))
            types['decided_both'] += int(dec[0]['chamber_decided'] and dec[1]['chamber_decided'])
            r['toric'] = {('X', 'Y')[c]: v for c, v in dec.items()}
        p = float(chi2.sf(Gsum, dfsum)) if dfsum else None; fisher_ps.append(p)
        res[SHORT[item]] = dict(years=len(rows), G2_sum=round(Gsum, 2), df_sum=dfsum, p=p, rejected_bonferroni=bool(p is not None and p < 0.05 / 11),
                                I_bits_mean=round(float(np.mean([r['I_bits'] for r in rows.values()])), 5), toric=types, by_year=rows)
    X2 = -2 * np.sum(np.log(np.maximum(fisher_ps, 1e-300))); res['_all_items_fisher'] = dict(X2=round(float(X2), 2), df=2 * len(fisher_ps), p=float(chi2.sf(X2, 2 * len(fisher_ps))))
    return res

def icu(rng):
    cache = os.environ.get('PHYSIONET_DIR', '/home/claude/phase2/data/challenge2012') + '/cache'; out = {}
    for w in ('a', 'b'):
        s = pd.read_csv(f'{cache}/set_{w}.csv').dropna(subset=['H1']); N = np.zeros((4, 3, 2))
        np.add.at(N, (s.ICUType.astype(int).values - 1, s.H1.astype(int).values, s.death.astype(int).values), 1)
        r = ci_test(N, rng); r['toric'] = {['coronary', 'cardiac surgery', 'medical', 'surgical'][c]: v for c, v in decomposition(N, '3x2').items()}
        Kc = (N / np.maximum(N.sum(2, keepdims=True), 1))[:, :, 1]; r['death_rate_by_unit_and_H1'] = np.round(Kc, 3).tolist(); out[f'set_{w}'] = r
    return out

def run():
    rng = np.random.default_rng(SEED); res = dict(plan='PREANALYSIS_CONTEXT_MAP.md Part A', gss=gss(rng), icu=icu(rng))
    os.makedirs(os.path.join(H, 'results'), exist_ok=True); json.dump(res, open(os.path.join(H, 'results', 'context_map.json'), 'w'), indent=1, default=str)
    for k, v in res['gss'].items(): print(k, {x: v[x] for x in v if x not in ('by_year',)})
    for k, v in res['icu'].items(): print(k, {x: v[x] for x in v})

def synth():
    """planted: invariant map with different information margins per context -> rejection rate near 5%; a context that changes
    P(O | A) at one answer level -> rejected, localized at that level"""
    rng = np.random.default_rng(3); K = np.array([[.6, .2, .2], [.35, .35, .3], [.15, .25, .6]]); rej = 0; ok = True
    for _ in range(200):
        N = np.zeros((2, 3, 3))
        for c, a in enumerate(([.5, .3, .2], [.25, .35, .4])):
            A = rng.choice(3, 700, p=a); O = np.array([rng.choice(3, p=K[x]) for x in A]); np.add.at(N, (c, A, O), 1)
        rej += ci_test(N, rng, B=100)['p'] < 0.05
    K2 = K.copy(); K2[2] = [.4, .25, .35]; N = np.zeros((2, 3, 3))
    for c, (a, KK) in enumerate((([.5, .3, .2], K), ([.25, .35, .4], K2))):
        A = rng.choice(3, 700, p=a); O = np.array([rng.choice(3, p=KK[x]) for x in A]); np.add.at(N, (c, A, O), 1)
    r = ci_test(N, rng); top = max(r['slice_share'], key=lambda k: r['slice_share'][k])
    ok = 0.02 <= rej / 200 <= 0.09 and r['p'] < 0.01 and top == '2'
    s = f"size {rej}/200; planted change at answer 2: p={r['p']:.2g}, largest slice share at answer {top}"; print(s, 'PASS' if ok else 'FAIL')
    os.makedirs(os.path.join(H, 'results'), exist_ok=True); open(os.path.join(H, 'results', 'context_map_synth.txt'), 'w').write(('PASS ' if ok else 'FAIL ') + s)
if __name__ == '__main__':
    {'synth': synth, 'run': run}[sys.argv[1]]()
