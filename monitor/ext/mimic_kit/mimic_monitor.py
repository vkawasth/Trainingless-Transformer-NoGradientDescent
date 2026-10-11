"""mimic_monitor.py -- the structural-change monitor on MIMIC-IV, run locally where the credentialed data are held.

Implements the pre-analysis plan in PREANALYSIS_PLAN.md (Bands, Sec. 11). Reads the gzipped CSV files of the hosp/ and icu/ modules
with DuckDB, builds a per-stay table of clinical states IN MEMORY ONLY, and writes AGGREGATE results only (results.json, results.txt):
statistics, p-values, and proportion tables whose every cell count is at least the suppression threshold (default 10). No row-level
data are written, printed or sent anywhere.

Usage:   python3 mimic_monitor.py /path/to/mimiciv/3.1  [--out results] [--min-cell 10] [--threads 4] [--quick]
The path is the folder that contains hosp/ and icu/."""
import os, sys, json, argparse, itertools, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from monitor import State, regime, cbd_split, churn, gluing_gap, bootstrap, tear_margin
from panel_models import fit as lc_fit, G2 as lc_G2, gap as lc_gap, DF as LC_DF
from panel_paths import perm_test, loo_bits
from scipy.stats import chi2
from scipy.optimize import linprog

# ------------------------------------------------------------------ fixed definitions (PREANALYSIS_PLAN.md, Sec. 2)
MAP_IDS = {220052: 'mean', 220181: 'mean', 225312: 'mean'}
PRESSOR_IDS = {221906: 'norepinephrine', 221289: 'epinephrine', 221749: 'phenylephrine', 222315: 'vasopressin', 221662: 'dopamine'}
VENT_IDS = {225792: 'invasive ventilation'}
LAB_IDS = {50912: 'creatinine', 50813: 'lactate'}
WINDOWS = [(0, 6), (21, 27), (42, 48)]                 # hours after ICU admission
H_CUT = (65, 75)                                       # MAP: < 65 or any pressor = shock (0); 65-75 borderline (1); >= 75 stable (2)
R_CUT = (1.2, 2.0)                                     # creatinine mg/dL: < 1.2 (0); 1.2-2.0 (1); >= 2.0 (2)
L_CUT = (2.0, 4.0)                                     # lactate mmol/L: < 2 (0); 2-4 (1); >= 4 (2)
UNITS = ('Medical Intensive Care Unit (MICU)', 'Surgical Intensive Care Unit (SICU)')

def band(x, cuts): return np.where(np.isnan(x), np.nan, np.digitize(x, cuts)).astype(float)

# ------------------------------------------------------------------ extraction (DuckDB on the gzipped CSVs; nothing written to disk)
def extract(root, threads):
    import duckdb
    con = duckdb.connect(); con.execute(f"PRAGMA threads={threads}")
    f = lambda mod, t: os.path.join(root, mod, t + '.csv.gz')
    for mod, t in (('hosp', 'patients'), ('hosp', 'admissions'), ('hosp', 'labevents'), ('hosp', 'd_labitems'), ('icu', 'icustays'),
                   ('icu', 'chartevents'), ('icu', 'inputevents'), ('icu', 'procedureevents'), ('icu', 'd_items')):
        if not os.path.exists(f(mod, t)): sys.exit(f"missing {f(mod, t)}: the path must contain hosp/ and icu/ of MIMIC-IV")
    # verify the item identifiers against the dictionaries before using them
    di = con.execute(f"SELECT itemid, label FROM read_csv_auto('{f('icu', 'd_items')}')").df().set_index('itemid')['label'].str.lower()
    dl = con.execute(f"SELECT itemid, label FROM read_csv_auto('{f('hosp', 'd_labitems')}')").df().set_index('itemid')['label'].str.lower()
    for ids, dic in ((MAP_IDS, di), (PRESSOR_IDS, di), (VENT_IDS, di), (LAB_IDS, dl)):
        for k, kw in ids.items():
            if k not in dic.index or kw not in dic[k]: sys.exit(f"itemid {k}: expected a label containing '{kw}', found '{dic.get(k)}'. Stop: check the version.")
    con.execute(f"""CREATE TEMP TABLE cohort AS
        WITH s AS (SELECT i.subject_id, i.hadm_id, i.stay_id, i.first_careunit, i.intime, i.los,
                          ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime) AS k
                   FROM read_csv_auto('{f('icu', 'icustays')}') i)
        SELECT s.*, p.anchor_age, p.anchor_year_group, a.hospital_expire_flag, a.admission_type
        FROM s JOIN read_csv_auto('{f('hosp', 'patients')}') p USING (subject_id)
               JOIN read_csv_auto('{f('hosp', 'admissions')}') a USING (subject_id, hadm_id)
        WHERE s.k = 1 AND p.anchor_age >= 18 AND s.los >= 2""")
    n_all = con.execute(f"SELECT count(DISTINCT subject_id) FROM read_csv_auto('{f('icu', 'icustays')}')").fetchone()[0]
    st = con.execute("SELECT * FROM cohort").df()
    wsql = " UNION ALL ".join(f"SELECT {k} AS w, {a} AS a, {b} AS b" for k, (a, b) in enumerate(WINDOWS))
    con.execute(f"CREATE TEMP TABLE win AS {wsql}")
    ids = ','.join(map(str, MAP_IDS))
    mp = con.execute(f"""SELECT c.stay_id, w.w, median(c.valuenum) AS map
        FROM read_csv_auto('{f('icu', 'chartevents')}') c JOIN cohort USING (stay_id) CROSS JOIN win w
        WHERE c.itemid IN ({ids}) AND c.valuenum BETWEEN 10 AND 250
          AND date_diff('second', cohort.intime, c.charttime) / 3600.0 >= w.a AND date_diff('second', cohort.intime, c.charttime) / 3600.0 < w.b
        GROUP BY c.stay_id, w.w""").df()
    ids = ','.join(map(str, PRESSOR_IDS))
    pr = con.execute(f"""SELECT DISTINCT e.stay_id, w.w
        FROM read_csv_auto('{f('icu', 'inputevents')}') e JOIN cohort USING (stay_id) CROSS JOIN win w
        WHERE e.itemid IN ({ids})
          AND date_diff('second', cohort.intime, e.starttime) / 3600.0 < w.b AND date_diff('second', cohort.intime, e.endtime) / 3600.0 > w.a""").df()
    ve = con.execute(f"""SELECT DISTINCT e.stay_id FROM read_csv_auto('{f('icu', 'procedureevents')}') e JOIN cohort USING (stay_id)
        WHERE e.itemid IN ({','.join(map(str, VENT_IDS))}) AND date_diff('second', cohort.intime, e.starttime) / 3600.0 < 24""").df()
    lb = con.execute(f"""SELECT cohort.stay_id, l.itemid, arg_min(l.valuenum, l.charttime) AS v
        FROM read_csv_auto('{f('hosp', 'labevents')}') l JOIN cohort USING (hadm_id)
        WHERE l.itemid IN (50912, 50813) AND l.valuenum IS NOT NULL
          AND date_diff('second', cohort.intime, l.charttime) / 3600.0 BETWEEN -6 AND 24
        GROUP BY cohort.stay_id, l.itemid""").df()
    # per-stay states (in memory only)
    st = st.set_index('stay_id')
    for k in range(3):
        m = mp[mp.w == k].set_index('stay_id')['map'].reindex(st.index)
        p = st.index.isin(pr[pr.w == k].stay_id)
        h = band(m.values.astype(float), H_CUT); h[p] = 0
        st[f'H{k+1}'] = h
    st['vent24'] = st.index.isin(ve.stay_id).astype(int)
    st['R'] = band(lb[lb.itemid == 50912].set_index('stay_id')['v'].reindex(st.index).values.astype(float), R_CUT)
    st['L'] = band(lb[lb.itemid == 50813].set_index('stay_id')['v'].reindex(st.index).values.astype(float), L_CUT)
    st['death'] = st['hospital_expire_flag'].astype(int)
    return st, n_all

# ------------------------------------------------------------------ output helpers (aggregate only, with suppression)
def table(a, b, ka, kb):
    T = np.zeros((ka, kb)); np.add.at(T, (np.asarray(a, int), np.asarray(b, int)), 1); return T
def shown(T, mincell):
    return {'proportions': (T / T.sum()).round(4).tolist(), 'n': int(T.sum())} if T.min() >= mincell else {'suppressed': f'a cell count below {mincell}', 'n': int(T.sum())}

def cf_triangle(Txy, Txz, Tyz):
    """Contextual fraction of three pairwise tables (Abramsky-Barbosa-Mansfield): 1 - max weight of a sub-normalized joint of
    (X, Y, Z) whose pairwise margins lie below the tables."""
    kx, ky = Txy.shape; kz = Txz.shape[1]; cells = list(itertools.product(range(kx), range(ky), range(kz))); N = len(cells)
    A = []; b = []
    for (T, ix) in ((Txy / Txy.sum(), (0, 1)), (Txz / Txz.sum(), (0, 2)), (Tyz / Tyz.sum(), (1, 2))):
        for u, v in itertools.product(range(T.shape[0]), range(T.shape[1])):
            A.append([1.0 if (c[ix[0]], c[ix[1]]) == (u, v) else 0.0 for c in cells]); b.append(T[u, v])
    r = linprog(-np.ones(N), A_ub=np.array(A), b_ub=b, bounds=[(0, None)] * N, method='highs')
    return float(1 + r.fun)

def ipf3(Txy, Txz, Tyz, it=500):
    P = np.ones((Txy.shape[0], Txy.shape[1], Txz.shape[1])); P /= P.sum()
    for _ in range(it):
        P *= (Txy / Txy.sum() / np.maximum(P.sum(2), 1e-300))[:, :, None]
        P *= (Txz / Txz.sum() / np.maximum(P.sum(1), 1e-300))[:, None, :]
        P *= (Tyz / Tyz.sum() / np.maximum(P.sum(0), 1e-300))[None, :, :]
    return P / P.sum()

# ------------------------------------------------------------------ restricted selection models (phase-4 amendment, 2026-10-10)
def _logit_fit(D, m, it=50, ridge=1e-6):
    """logistic regression by Newton steps (no extra dependencies); returns fitted probabilities"""
    X = np.hstack([np.ones((len(D), 1)), D]); b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p) + 1e-9
        step = np.linalg.solve(X.T @ (W[:, None] * X) + ridge * np.eye(X.shape[1]), X.T @ (m - p)); b += step
        if np.abs(step).max() < 1e-8: break
    return 1 / (1 + np.exp(-X @ b))
def _dummies(C):
    return pd.get_dummies(C.astype(str), drop_first=True).astype(float).values if C.shape[1] else np.zeros((len(C), 0))
def _propensity(model, x, y, C, m):
    key = {'X-only': x, 'Y-only': y, 'cell': x * 3 + y}.get(model)
    if key is not None: return pd.Series(m.astype(float)).groupby(key).mean().reindex(key).values
    D = _dummies(C)
    if model == 'X+external': D = np.hstack([D, _dummies(pd.DataFrame(dict(x=x)))])
    return _logit_fit(D, m.astype(float))
def _weighted_cf(model, x, y, z, C, m):
    p = np.clip(_propensity(model, x, y, C, m), 1e-3, 1); w = 1 / p[m]
    Txy = table(x, y, 3, 3); Wxz = np.zeros((3, 3)); np.add.at(Wxz, (x[m], z[m].astype(int)), w); Wyz = np.zeros((3, 3)); np.add.at(Wyz, (y[m], z[m].astype(int)), w)
    return cf_triangle(Txy, Wxz, Wyz), (Txy, Wxz, Wyz), p
def restricted_selection(x, y, z, C, rng, B, models=('X-only', 'Y-only', 'external', 'X+external')):
    """For each weight model: the weighted contextual fraction and a null recomputed for that model by parametric bootstrap
    (tables compatible AND selection follows the model). Any weights that reproduce the full (x, y) table make the triangle compatible,
    so these tests check only what each model implies for the (x, y) composition of the measured; selection on z itself (not at random)
    is invisible to them."""
    x, y = np.asarray(x, int), np.asarray(y, int); z = np.asarray(z, float); m = ~np.isnan(z); n = len(x); out = {}
    for model in models:
        cf, tabs, p = _weighted_cf(model, x, y, z, C, m); P = ipf3(*tabs); Pz = P / np.maximum(P.sum(2, keepdims=True), 1e-300); null = []
        for _ in range(B):
            idx = rng.integers(0, n, n); xs, ys = x[idx], y[idx]; Cs = C.iloc[idx].reset_index(drop=True)
            zs = (rng.random(n)[:, None] > np.cumsum(Pz[xs, ys], 1)).sum(1).clip(0, 2).astype(float); ms = rng.random(n) < p[idx]; zs[~ms] = np.nan
            null.append(_weighted_cf(model, xs, ys, zs, Cs, ms)[0])
        q = float(np.percentile(null, 95))
        out[model] = dict(cf=round(cf, 4), null95=round(q, 4), p=float((np.sum(np.array(null) >= cf) + 1) / (B + 1)), consistent=bool(cf <= q))
    return out

def boundary_null_p(TX, TY, nX, nY, obs, rng, B):
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if cbd_split(TX, (1 - mid) * TY + mid * TX)['deficit'] > 1e-9: lo = mid
        else: hi = mid
    TYb = (1 - hi) * TY + hi * TX; ge = 0
    for _ in range(B):
        ge += cbd_split(rng.multinomial(int(nX), TX.ravel()).reshape(TX.shape) + 1e-12, rng.multinomial(int(nY), TYb.ravel()).reshape(TX.shape) + 1e-12)['deficit'] >= obs - 1e-12
    return (ge + 1) / (B + 1)

# ------------------------------------------------------------------ the analyses of the plan (Sec. 3)
def analyse(st, mincell, quick, rng):
    out = {}; B = 50 if quick else 400
    # A. coupling geometry over the calendar periods: system (H1, R)
    a = st.dropna(subset=['H1', 'R']); groups = sorted(a.anchor_year_group.unique()); S = {}
    for g in groups:
        s = a[a.anchor_year_group == g]; T = table(s.H1, s.R, 3, 3)
        if T.sum() >= 200: S[g] = State(T + 1e-9, T.sum())
    A_ = {'tables': {g: shown(table(a[a.anchor_year_group == g].H1, a[a.anchor_year_group == g].R, 3, 3), mincell) for g in S}, 'steps': []}
    gs = list(S)
    for g0, g1 in zip(gs, gs[1:]):
        r = regime(S[g0], S[g1]); A_['steps'].append(dict(frm=g0, to=g1, kind=r['kind'], stress=round(r['stress'], 2), tear_margin=round(r['mT'], 4),
                                                         walls_crossed=[[list(I), list(J)] for I, J in r['walls_crossed']]))
    A_['mutual_information_bits'] = {g: round(S[g].MI, 4) for g in S}
    # toric surface of the 2 x 3 system (pressor or not in window 1) x creatinine band, by period (Bridges, Prop. 8.1)
    tor = {}
    for g in gs:
        s = a[a.anchor_year_group == g]; T = table((s.H1 != 0).astype(int), s.R, 2, 3); f, gg = T.sum(1) / T.sum(), T.sum(0) / T.sum()
        fac = [(i, j) for i in range(2) for j in range(3) if f[i] + gg[j] < 1]
        cols = [j for j in range(3) if (0, j) in fac and (1, j) in fac]
        name = {3: 'CP^2', 5: 'Bl_2 CP^2', 6: 'Bl_3 CP^2'}.get(len(fac), 'CP^1 x CP^1' if len(fac) == 4 and len(cols) == 2 else 'F_1')
        tor[g] = dict(facets=[f'{i+1}{j+1}' for i, j in fac], surface=name, decided=bool((State(T + 1e-9, T.sum()).signs != 0).all()))
    A_['toric_2x3'] = tor; out['A_coupling_geometry'] = A_
    # B. composition of the hemodynamic path: gluing gap, Markov null, latent-class models
    b = st.dropna(subset=['H1', 'H2', 'H3']); x1, x2, x3 = (b[c].values.astype(int) for c in ('H1', 'H2', 'H3'))
    gg = gluing_gap(x1, x2, x3); ci = bootstrap(gluing_gap, [x1, x2, x3], B=B, seed=1)
    P12 = pd.crosstab(x1, x2, normalize='index'); P23 = pd.crosstab(x2, x3, normalize='index'); nulls = []
    K12c = np.cumsum(pd.crosstab(x1, x2, normalize='index').reindex(index=range(3), columns=range(3), fill_value=0).values, 1)
    K23c = np.cumsum(pd.crosstab(x2, x3, normalize='index').reindex(index=range(3), columns=range(3), fill_value=0).values, 1)
    step = lambda Kc, v: np.minimum((rng.random(len(v))[:, None] > Kc[v]).sum(1), 2)     # vectorized draw from row v of a kernel
    for _ in range(B):
        aa = rng.choice(x1, len(x1)); bb = step(K12c, aa); cc = step(K23c, bb); nulls.append(gluing_gap(aa, bb, cc))
    N3 = np.zeros((3, 3, 3)); np.add.at(N3, (x1, x2, x3), 1)
    models = {}
    for m in ('stable', 'drifting', 'markov'):
        th, P = lc_fit(N3, m, starts=5 if quick else 30); g2 = lc_G2(N3, P)
        models[m] = dict(G2=round(g2, 2), df=LC_DF[m], p=float(chi2.sf(g2, LC_DF[m])), gap_implied=round(lc_gap(P), 4))
    K12 = pd.crosstab(x1, x2, normalize='index').values; K23 = pd.crosstab(x2, x3, normalize='index').values; K13 = pd.crosstab(x1, x3, normalize='index').values
    out['B_composition'] = dict(n=len(b), gap_bits=round(gg, 4), gap_ci=[round(v, 4) for v in ci], markov_null95=round(float(np.percentile(nulls, 95)), 4),
                                p_markov=float((np.sum(np.array(nulls) >= gg) + 1) / (B + 1)), latent_models=models,
                                tv_K13_vs_K12K23=(0.5 * np.abs(K13 - K12 @ K23).sum(1)).round(4).tolist(),
                                churn_12=dict((k, round(v, 4)) for k, v in churn(x1, x2, [0, 1, 2]).items() if k in ('agree', 'max_agree', 'excess')))
    # C. path and outcome (in-hospital death): wave-1 state, multiset, ordered path
    y = b['death'].values; path = x1 * 9 + x2 * 3 + x3; ms = np.array([''.join(map(str, sorted(t))) for t in zip(x1, x2, x3)])
    g1, p1 = perm_test(y, ms, x1, 2, rng, B=B * 5); g2_, p2 = perm_test(y, path, ms, 2, rng, B=B * 5)
    base = loo_bits(y, np.zeros(len(y), int), 2)
    out['C_path_outcome'] = dict(n=len(y), bits_gained_x1=round(base - loo_bits(y, x1, 2), 4), bits_gained_multiset=round(base - loo_bits(y, ms, 2), 4),
                                 bits_gained_path=round(base - loo_bits(y, path, 2), 4), p_multiset_vs_x1=p1, p_order_vs_multiset=p2)
    # D. contextuality across care units within strata of the window-1 state: contents H2 (3) and ventilation in 24 h (2)
    c = st.dropna(subset=['H1', 'H2']); D = {}; ps = []
    for h in range(3):
        s = c[c.H1 == h]; TX = table(s[s.first_careunit == UNITS[0]].H2, s[s.first_careunit == UNITS[0]].vent24, 3, 2)
        TY = table(s[s.first_careunit == UNITS[1]].H2, s[s.first_careunit == UNITS[1]].vent24, 3, 2)
        if min(TX.sum(), TY.sum()) < 200: D[h] = 'too few stays'; continue
        sp = cbd_split(TX / TX.sum(), TY / TY.sum()); naive = 0.5 * float(np.abs(TX / TX.sum() - TY / TY.sum()).sum())
        p = boundary_null_p(TX / TX.sum(), TY / TY.sum(), TX.sum(), TY.sum(), sp['deficit'], rng, B) if sp['deficit'] > 0 else 1.0
        ps.append(p); D[h] = dict(naive_cf=round(naive, 4), deficit=round(sp['deficit'], 5), contextual_point=sp['contextual'], p_boundary=round(p, 4),
                                  MICU=shown(TX, mincell), SICU=shown(TY, mincell))
    X2 = -2 * np.log(ps).sum() if ps else 0.0
    out['D_contextuality_units'] = dict(strata=D, fisher=round(float(X2), 3), df=2 * len(ps), p=float(chi2.sf(X2, 2 * len(ps))) if ps else None)
    # E. selection: available-case incompatibility of (H1, R), (H1, L), (R, L), and after inverse-probability weighting on (H1, R)
    e = st.dropna(subset=['H1', 'R']); meas = e.L.notna().values
    Txy = table(e.H1, e.R, 3, 3); em = e[meas]; Txz = table(em.H1, em.L, 3, 3); Tyz = table(em.R, em.L, 3, 3)
    cf_avail = cf_triangle(Txy, Txz, Tyz)
    pm = e.assign(m=meas).groupby(['H1', 'R']).m.mean(); wts = 1 / pm.loc[list(zip(em.H1, em.R))].values
    Twz = np.zeros((3, 3)); np.add.at(Twz, (em.H1.astype(int).values, em.L.astype(int).values), wts)
    Twy = np.zeros((3, 3)); np.add.at(Twy, (em.R.astype(int).values, em.L.astype(int).values), wts)
    cf_ipw = cf_triangle(Txy, Twz, Twy)
    P = ipf3(Txy, Twz, Twy); null = []
    nxy, nm = int(Txy.sum()), int(len(em))
    for _ in range(B):
        Sxy = rng.multinomial(nxy, P.sum(2).ravel()).reshape(3, 3); Sxz = rng.multinomial(nm, P.sum(1).ravel()).reshape(3, 3); Syz = rng.multinomial(nm, P.sum(0).ravel()).reshape(3, 3)
        null.append(cf_triangle(Sxy + 1e-12, Sxz + 1e-12, Syz + 1e-12))
    out['E_selection'] = dict(n_all=nxy, n_lactate_measured=nm, cf_available_case=round(cf_avail, 4), cf_after_ipw=round(cf_ipw, 4),
                              null_median=round(float(np.median(null)), 4), null_95=round(float(np.percentile(null, 95)), 4),
                              p_available=float((np.sum(np.array(null) >= cf_avail) + 1) / (B + 1)), p_ipw=float((np.sum(np.array(null) >= cf_ipw) + 1) / (B + 1)),
                              note_ipw='weighting on the (H1, creatinine) cells makes the tables compatible by construction; only the available-case test and the restricted models below are evidence')
    C = pd.DataFrame(dict(admission_type=e['admission_type'].astype(str).values, age_band=np.digitize(e['anchor_age'].values, (50, 70))))
    out['E_selection']['restricted_models'] = restricted_selection(e.H1.astype(int).values, e.R.astype(int).values, e.L.values, C, rng, B=50 if quick else 200)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--out', default='results'); ap.add_argument('--min-cell', type=int, default=10)
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--quick', action='store_true'); ap.add_argument('--seed', type=int, default=20261009)
    a = ap.parse_args(); t0 = time.time(); rng = np.random.default_rng(a.seed)
    st, n_all = extract(a.root, a.threads)
    res = dict(plan='PREANALYSIS_PLAN.md (dated 2026-10-09)', cohort=dict(icu_patients=int(n_all), first_stays_adult_los_ge_2d=int(len(st)),
               with_H1_H2_H3=int(st[['H1', 'H2', 'H3']].notna().all(axis=1).sum()), with_creatinine=int(st.R.notna().sum()), with_lactate=int(st.L.notna().sum())))
    res.update(analyse(st, a.min_cell, a.quick, rng)); res['seconds'] = round(time.time() - t0, 1)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, 'results.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    with open(os.path.join(a.out, 'results.txt'), 'w') as fh:
        for k, v in res.items(): fh.write(f"{k}: {json.dumps(v, default=str)}\n")
    print(f"done in {res['seconds']} s; aggregate results in {a.out}/results.json (no row-level data written)")
