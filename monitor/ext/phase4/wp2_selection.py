"""WP2 (PREANALYSIS_PHASE4.md): restricted selection models whose weights are not built on the triangle cells, each with its own
parametric-bootstrap null (tables compatible AND selection follows the model).
Usage: python3 wp2_selection.py synth | run   -> results/wp2_selection.json"""
import os, sys, json, tempfile
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
H = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.abspath(os.path.join(H, '..'))
sys.path.insert(0, os.path.join(EXT, 'open_clinical', 'lib')); sys.path.insert(0, os.path.join(EXT, 'open_clinical')); sys.path.insert(0, os.path.join(EXT, 'c19'))
from mimic_monitor import cf_triangle, ipf3
SEED, B = 20261010, 200
CACHE = os.environ.get('PHYSIONET_DIR', '/home/claude/phase2/data/challenge2012') + '/cache'

def tab(a, b, w=None): T = np.zeros((3, 3)); np.add.at(T, (a, b), 1.0 if w is None else w); return T
def propensity(model, x, y, C, m):
    """fitted P(measured | model covariates) for every row"""
    if model == 'cell': key = x * 3 + y
    elif model == 'X-only': key = x
    elif model == 'Y-only': key = y
    else: key = None
    if key is not None:
        s = pd.Series(m).groupby(key).mean(); return s.reindex(key).values
    D = pd.get_dummies(C.astype(str), drop_first=True).astype(float)
    if model == 'X+external': D = pd.concat([D, pd.get_dummies(pd.Series(x).astype(str), prefix='x', drop_first=True).astype(float)], axis=1)
    lr = LogisticRegression(C=1e6, max_iter=3000).fit(D.values, m); return lr.predict_proba(D.values)[:, 1]
def weighted_cf(model, x, y, z, C, m):
    p = np.clip(propensity(model, x, y, C, m), 1e-3, 1); w = 1 / p[m]
    Txy, Wxz, Wyz = tab(x, y), tab(x[m], z[m].astype(int), w), tab(y[m], z[m].astype(int), w)
    return cf_triangle(Txy, Wxz, Wyz), (Txy, Wxz, Wyz), p
def analyse(x, y, z, C, rng, models=('cell', 'X-only', 'Y-only', 'external', 'X+external'), Bn=B):
    x, y = np.asarray(x, int), np.asarray(y, int); z = np.asarray(z, float); m = ~np.isnan(z); n = len(x); out = {}
    Txy, Txz, Tyz = tab(x, y), tab(x[m], z[m].astype(int)), tab(y[m], z[m].astype(int))
    out['available_case'] = dict(cf=round(cf_triangle(Txy, Txz, Tyz), 4), n=n, measured_share=round(float(m.mean()), 4))
    for model in models:
        cf, tabs, p = weighted_cf(model, x, y, z, C, m)
        P = ipf3(*tabs); Pz = P / np.maximum(P.sum(2, keepdims=True), 1e-300); null = []
        if model == 'cell': out[model] = dict(cf=round(cf, 4), note='0 by construction; not a test'); continue
        for _ in range(Bn):
            idx = rng.integers(0, n, n); xs, ys = x[idx], y[idx]; Cs = C.iloc[idx].reset_index(drop=True)
            zs = (rng.random(n)[:, None] > np.cumsum(Pz[xs, ys], 1)).sum(1).clip(0, 2).astype(float)
            ms = rng.random(n) < p[idx]; zs[~ms] = np.nan
            null.append(weighted_cf(model, xs, ys, zs, Cs, ms)[0])
        q = float(np.percentile(null, 95))
        out[model] = dict(cf=round(cf, 4), null95=round(q, 4), null_median=round(float(np.median(null)), 4),
                          p=float((np.sum(np.array(null) >= cf) + 1) / (Bn + 1)), consistent=bool(cf <= q))
    return out

def load_physionet(which):
    s = pd.read_csv(f'{CACHE}/phase4_set_{which}.csv').dropna(subset=['H1', 'R'])
    return s.H1.astype(int).values, s.R.astype(int).values, s.L.values, s[['ICUType', 'vent24']].reset_index(drop=True)
def load_diabetes():
    import diabetes130 as DB
    d = DB.load('/home/claude/phase2/data/uci/diabetes130.zip')
    return d.X.values, d.Y.values, d.A1C.values, d[['admission_source_id', 'admission_type_id']].reset_index(drop=True)
def load_uk():
    import c19, pyreadstat
    W, w3, y, info = c19.load_uk(os.path.join(EXT, 'c19', 'raw', 'uk'))
    e = W[1].dropna(subset=['D', 'A'])[['pid', 'D', 'A']].drop_duplicates('pid').merge(W[3][['pid', 'D']].dropna().drop_duplicates('pid').rename(columns={'D': 'D3'}), on='pid', how='left')
    cv, _ = pyreadstat.read_sav(os.path.join(EXT, 'c19', 'raw', 'uk', 'C19PRC_UKW1W2_archive_final.sav'), usecols=['pid', 'W1_Age_categories', 'W1_Gender', 'W1_Education_binary', 'W1_Urbanicity'])
    cv['pid'] = cv.pid.astype(str); cv['W1_Gender'] = cv.W1_Gender.where(cv.W1_Gender.isin([1, 2]), 3)
    e = e.merge(cv, on='pid', how='left').fillna({'W1_Age_categories': 0, 'W1_Education_binary': -1, 'W1_Urbanicity': -1})
    return e.D.astype(int).values, e.A.astype(int).values, e.D3.values, e[['W1_Age_categories', 'W1_Gender', 'W1_Education_binary', 'W1_Urbanicity']].reset_index(drop=True)

def run():
    rng = np.random.default_rng(SEED); res = dict(plan='PREANALYSIS_PHASE4.md WP2', B=B)
    for name, loader in (('physionet_set_a', lambda: load_physionet('a')), ('physionet_set_b', lambda: load_physionet('b')),
                         ('diabetes130', load_diabetes), ('c19prc_uk_attrition', load_uk)):
        x, y, z, C = loader(); res[name] = analyse(x, y, z, C, rng); print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(os.path.join(H, 'results', 'wp2_selection.json'), 'w'), indent=1)

def synth():
    """planted: (a) X-only selection with compatible tables -> X-only consistent, Y-only rejected;
               (b) selection on an external c that depends on (x, y) jointly -> external consistent, X-only and Y-only rejected.
    A first version of (b) let c drive z with c independent of (x, y); that selection leaves the triangle compatible, so no weight model
    can be tested on it (see the note in the paper). Changed before any real run."""
    rng = np.random.default_rng(4); n = 6000; ok = True; out = {}
    for kind in ('x_only', 'external'):
        c = rng.integers(0, 2, n); x = rng.choice(3, n, p=[.3, .4, .3]); y = (x + rng.choice([0, 1, 2], n, p=[.6, .25, .15])) % 3
        if kind == 'x_only':
            z = (x + rng.choice([0, 1], n, p=[.7, .3])) % 3; pm = np.array([.2, .5, .8])[x]
        else:   # c depends on (x, y) jointly (agreement), and selection depends on c only
            c = np.where(rng.random(n) < .85, (x == y).astype(int), 1 - (x == y).astype(int))
            z = (x + rng.choice([0, 1], n, p=[.7, .3])) % 3; pm = np.where(c == 1, .85, .15)
        zz = z.astype(float); zz[rng.random(n) >= pm] = np.nan
        r = analyse(x, y, zz, pd.DataFrame(dict(c=c)), np.random.default_rng(1), models=('X-only', 'Y-only', 'external'), Bn=100); out[kind] = r
        print(kind, {k: (v.get('cf'), v.get('null95'), v.get('consistent')) for k, v in r.items()})
        ok &= (r['X-only']['consistent'] and not r['Y-only']['consistent']) if kind == 'x_only' else (r['external']['consistent'] and not r['X-only']['consistent'] and not r['Y-only']['consistent'])
    json.dump(out, open(os.path.join(H, 'results', 'wp2_selection_synth.json'), 'w'), indent=1); print('PASS' if ok else 'FAIL')
if __name__ == '__main__':
    {'synth': synth, 'run': run}[sys.argv[1]]()
