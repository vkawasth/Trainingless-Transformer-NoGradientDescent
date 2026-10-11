"""PREANALYSIS_TOWER_ICU.md: a three-level tower over PhysioNet 2012 (models x reliability), Fano tubes of its gradations, and the
interchange recheck of the ICU pressure bands.
Usage: python3 tower_icu.py load                 (reads set-a.zip, set-b.zip; patient-level cache OUTSIDE the monitor folder)
       python3 tower_icu.py tower|interchange    (writes results/tower_icu.json, results/interchange_icu.json)
       python3 tower_icu.py synth                (adapter checks on planted data)"""
import os, sys, io, json, zipfile
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, 'lib')); sys.path.insert(0, os.path.join(H, '..', 'c19'))
sys.path.insert(0, os.path.join(H, '..', '..')); sys.path.insert(0, os.path.join(H, '..', '..', '..', 'bridges', 'fukaya'))
DATA = os.environ.get('PHYSIONET_DIR', '/home/claude/phase2/data/challenge2012'); CACHE = os.path.join(DATA, 'cache')
SEED = 20261010; WINDOWS = [(0, 6), (21, 27), (42, 48)]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]; MODELS = ['M0', 'M1', 'M2', 'M3', 'M4']
FORMS = {'M0': [], 'M1': ['H1'], 'M2': ['H1', 'R'], 'M3': ['H1', 'R', 'vent24', 'ICUType'], 'M4': ['SOFAb']}

# ------------------------------------------------------------------ data
def load_set(which):
    z = zipfile.ZipFile(os.path.join(DATA, f'set-{which}.zip')); rows = []
    for name in z.namelist():
        if not name.endswith('.txt'): continue
        d = pd.read_csv(io.BytesIO(z.read(name))); hh = d.Time.str.split(':', expand=True).astype(float); d['t'] = hh[0] + hh[1] / 60
        desc = d[d.t == 0].drop_duplicates('Parameter').set_index('Parameter').Value
        rec = dict(RecordID=int(desc.get('RecordID', -1)), ICUType=desc.get('ICUType', np.nan))
        for k, (a, b) in enumerate(WINDOWS):
            w = d[(d.t >= a) & (d.t < b)]; m = w[(w.Parameter == 'MAP') & w.Value.between(10, 250)].Value
            if m.empty: m = w[(w.Parameter == 'NIMAP') & w.Value.between(10, 250)].Value
            rec[f'MAP{k+1}'] = m.median() if not m.empty else np.nan
        day = d[(d.t >= 0) & (d.t <= 24)]; v = day[day.Parameter == 'Creatinine'].sort_values('t').Value
        rec['creat'] = v.iloc[0] if not v.empty else np.nan
        rec['vent24'] = int(((d.Parameter == 'MechVent') & (d.Value == 1) & (d.t < 24)).any()); rows.append(rec)
    st = pd.DataFrame(rows); oc = pd.read_csv(os.path.join(DATA, f'Outcomes-{which}.txt')).rename(columns={'In-hospital_death': 'death'})
    st = st.merge(oc[['RecordID', 'SOFA', 'death']], on='RecordID', how='left'); st = st[st.ICUType.notna() & (st.ICUType > 0)]
    for k in (1, 2, 3): st[f'H{k}'] = np.where(st[f'MAP{k}'].isna(), np.nan, np.digitize(st[f'MAP{k}'], (65, 75)))
    st['R'] = np.where(st.creat.isna(), np.nan, np.digitize(st.creat, (1.2, 2.0)))
    st['SOFAb'] = np.where(st.SOFA.isna() | (st.SOFA < 0), np.nan, np.digitize(st.SOFA, (5, 9)))
    return st.sort_values('RecordID').reset_index(drop=True)

def design(df, cols):
    """one-hot with a 'missing' level for every categorical covariate"""
    parts = [pd.get_dummies(df[c].fillna(-1).astype(int).astype(str), prefix=c, dtype=float) for c in cols]
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)

# ------------------------------------------------------------------ the tower
class Tower:
    """hypotheses (k, alpha); rho_k^alpha(death | x) = alpha rho_k(x) + (1 - alpha) rho_0; log-space conditioning"""
    def __init__(self, fit, alphas=ALPHAS, models=MODELS):
        self.alphas, self.models, self.base = alphas, models, fit.death.mean(); self.fits = {}
        for m in models:
            if not FORMS[m]: self.fits[m] = None; continue
            X = design(fit, FORMS[m]); self.fits[m] = (LogisticRegression(C=np.inf, max_iter=5000).fit(X.values, fit.death.values), list(X.columns))
        self.logw = np.zeros((len(models), len(alphas))) - np.log(len(models) * len(alphas))
    def rho(self, df):
        """array (patients, models): fitted death probabilities"""
        out = []
        for m in self.models:
            if self.fits[m] is None: out.append(np.full(len(df), self.base)); continue
            lr, cols = self.fits[m]; X = design(df, FORMS[m]).reindex(columns=cols, fill_value=0.0); out.append(lr.predict_proba(X.values)[:, 1])
        return np.array(out).T
    def rho_ka(self, df):
        r = self.rho(df); a = np.array(self.alphas); return a[None, None, :] * r[:, :, None] + (1 - a)[None, None, :] * self.base   # (n, K, A)
    def weights(self): w = np.exp(self.logw - self.logw.max()); return w / w.sum()
    def condition(self, df, record_every=None):
        p = self.rho_ka(df); y = df.death.values; path = []
        for i in range(len(df)):
            self.logw += np.log(np.where(y[i] == 1, p[i], 1 - p[i]))
            if record_every and ((i + 1) % record_every == 0 or i == len(df) - 1): path.append((i + 1, self.weights().copy()))
        return path
    def predict(self, df, flat=False):
        w = self.weights(); p = self.rho_ka(df)
        if flat:   # flat Bayesian model averaging: alpha = 1 only, same evidence
            lw = self.logw[:, -1]; w1 = np.exp(lw - lw.max()); w1 /= w1.sum(); return p[:, :, -1] @ w1
        return np.einsum('nka,ka->n', p, w)
    def levels(self, df):
        """level 1 rho (per patient), level 2 Pi (weights over predictive points), level 3 Omega (weights over alpha, each with Pi_alpha)"""
        w = self.weights(); p = self.rho_ka(df); pa = w.sum(0); Pia = w / np.where(pa > 0, pa, 1)[None, :]
        return dict(rho=np.einsum('nka,ka->n', p, w), Pi=w, Omega=pa, Pi_alpha=Pia)

def logloss(p, y): p = np.clip(p, 1e-12, 1 - 1e-12); return -(y * np.log(p) + (1 - y) * np.log(1 - p))

# ------------------------------------------------------------------ toric type of the 2 x 3 system (death x H1 band)
from sweep23 import facets_of, self_int, surface, wallvals
from fractions import Fraction as Fr
def toric(f1, g, D=100000):
    f = Fr(int(round(f1 * D)), D); gg = [Fr(int(round(x * D)), D) for x in g[:2]]; gg.append(1 - gg[0] - gg[1])
    if any(v == 0 for v in wallvals(f, gg)): f += Fr(1, 10 * D)
    fac = facets_of(f, gg); return surface(fac, self_int(fac)), [1 if v > 0 else -1 for v in wallvals(f, gg)]
def polygon(f1, g):
    from toric_public import polygon as P; return P(f1, g)

def run_arm(fit, ev, te, name):
    T = Tower(fit); T0 = Tower(fit)                                       # T0: the prior tower (n = 0)
    te_h = te.dropna(subset=['H1']); g = np.bincount(te_h.H1.astype(int), minlength=3) / len(te_h)
    pk = T.rho_ka(te_h)                                                     # (n, K, A) on the test patients with H1
    f_ka = pk.mean(0)                                                       # each hypothesis's predicted death rate on them
    tube = []
    def snapshot(n, w):
        f1 = float((f_ka * w).sum()); s, walls = toric(f1, g)
        post = {}
        for k in range(len(MODELS)):
            for a in range(len(ALPHAS)):
                sk, _ = toric(float(f_ka[k, a]), g); post[sk] = post.get(sk, 0.0) + float(w[k, a])
        tube.append(dict(n=n, f1_level1=round(f1, 5), surface_level1=s, walls=walls, surface_posterior={k: round(v, 4) for k, v in post.items()},
                         level2_model_weights=[round(float(x), 4) for x in w.sum(1)], level3_alpha_weights=[round(float(x), 4) for x in w.sum(0)]))
    snapshot(0, T.weights())
    # the programme's tower code (condS of the base-relative tower) on the first 200 evidence patients, against the log-space path
    import contextlib, io as _io
    sys.path.insert(0, os.path.join(H, '..', 'structure', 'programme_code'))
    with contextlib.redirect_stdout(_io.StringIO()): import check_base_relative as CB
    p200 = T0.rho_ka(ev.iloc[:200]); S = CB.Dist([(((k, a), 'h'), 1.0 / (len(MODELS) * len(ALPHAS))) for k in range(len(MODELS)) for a in range(len(ALPHAS))])
    for i, yv in enumerate(ev.death.values[:200]):
        S, _ = CB.condS(S, {(k, a): (p200[i, k, a] if yv == 1 else 1 - p200[i, k, a]) for k in range(len(MODELS)) for a in range(len(ALPHAS))})
    T0.condition(ev.iloc[:200]); wS = np.array([[S.m.get(((k, a), 'h'), 0.0) for a in range(len(ALPHAS))] for k in range(len(MODELS))])
    condS_err = float(np.abs(wS - T0.weights()).max())
    for n, w in T.condition(ev, record_every=50): snapshot(n, w)
    y = te.death.values; pt, pf = T.predict(te), T.predict(te, flat=True); lt, lf = logloss(pt, y), logloss(pf, y)
    rng = np.random.default_rng(SEED); idx = rng.integers(0, len(y), (2000, len(y))); d = (lf[idx] - lt[idx]).mean(1)
    L = T.levels(te); coh = float(np.abs(L['Pi_alpha'] @ np.diag(L['Omega']) - L['Pi']).max())
    s_obs, walls_obs = toric(float(te_h.death.mean()), g)
    top = max(tube[-1]['surface_posterior'].items(), key=lambda kv: kv[1])
    tears = [(a['n'], b['n'], a['surface_level1'], b['surface_level1']) for a, b in zip(tube, tube[1:]) if a['surface_level1'] != b['surface_level1']]
    return dict(arm=name, n_fit=len(fit), n_evidence=len(ev), n_test=len(te), base_rate_fit=round(float(fit.death.mean()), 4),
                death_rate_evidence=round(float(ev.death.mean()), 4), death_rate_test=round(float(y.mean()), 4),
                logloss_tower=round(float(lt.mean()), 5), logloss_flat=round(float(lf.mean()), 5), logloss_each_model={m: round(float(logloss(T.rho(te)[:, k], y).mean()), 5) for k, m in enumerate(MODELS)},
                delta=round(float(d.mean()), 6), delta_ci=[round(float(np.percentile(d, 2.5)), 6), round(float(np.percentile(d, 97.5)), 6)],
                level3_earns_keep=bool(np.percentile(d, 2.5) > 0),
                posterior_models={m: round(float(x), 4) for m, x in zip(MODELS, T.weights().sum(1))}, posterior_alpha={str(a): round(float(x), 4) for a, x in zip(ALPHAS, T.weights().sum(0))},
                mass_alpha_below_1=round(float(T.weights().sum(0)[:-1].sum()), 4),
                calibration=dict(predicted=round(float(pt.mean()), 4), observed=round(float(y.mean()), 4)), coherence_max_err=coh, condS_agreement_200=condS_err,
                g_H1_test=[round(float(x), 4) for x in g], f_ka=[[round(float(x), 5) for x in r] for r in f_ka],
                surface_observed=s_obs, walls_observed=walls_obs, surface_decided=bool(top[1] >= 0.95), surface_top=dict(surface=top[0], prob=top[1]),
                tears_of_tube=tears, tube=tube)

# ------------------------------------------------------------------ interchange recheck
def interchange(st, which):
    import banding_interchange as BI
    BI.set_merge(17, [5, 7])                                                # 5-mmHg bins from 40: bin 5 starts at 65, bin 7 at 75
    s = st.dropna(subset=['MAP1', 'MAP2', 'MAP3'])
    b = [np.clip(((s[f'MAP{k}'].values - 40) // 5).astype(int), 0, 16) for k in (1, 2, 3)]
    assert all((BI.band(bb) == s[f'H{k}'].values).all() for k, bb in zip((1, 2, 3), b))
    return dict(set=which, **BI.run(*b, np.random.default_rng(SEED)))

if __name__ == '__main__':
    cmd = sys.argv[1]; os.makedirs(CACHE, exist_ok=True); os.makedirs(os.path.join(H, 'results'), exist_ok=True)
    if cmd == 'load':
        for w in ('a', 'b'): load_set(w).to_csv(os.path.join(CACHE, f'set_{w}.csv'), index=False); print('cached', w)
        sys.exit()
    if cmd == 'synth':
        import synth_tower_icu; sys.exit(synth_tower_icu.main())
    A = pd.read_csv(os.path.join(CACHE, 'set_a.csv')); B = pd.read_csv(os.path.join(CACHE, 'set_b.csv'))
    if cmd == 'tower':
        B1, B2 = B.iloc[:len(B) // 2], B.iloc[len(B) // 2:]
        Bs = B[B.ICUType.isin([1, 3])]; Bs1, Bs2 = Bs.iloc[:len(Bs) // 2], Bs.iloc[len(Bs) // 2:]
        res = dict(plan='PREANALYSIS_TOWER_ICU.md (2026-10-10)', same_population=run_arm(A, B1, B2, 'same population'),
                   shift=run_arm(A[A.ICUType.isin([2, 4])], Bs1, Bs2, 'shift: surgical -> medical/coronary'))
        json.dump(res, open(os.path.join(H, 'results', 'tower_icu.json'), 'w'), indent=1, default=str)
        for k in ('same_population', 'shift'):
            r = res[k]; print(k, {x: r[x] for x in ('logloss_tower', 'logloss_flat', 'delta', 'delta_ci', 'level3_earns_keep', 'posterior_models', 'posterior_alpha', 'surface_observed', 'surface_top', 'tears_of_tube', 'calibration')})
    if cmd == 'interchange':
        res = dict(plan='PREANALYSIS_TOWER_ICU.md Part 3 (2026-10-10)', set_a=interchange(A, 'a'), set_b=interchange(B, 'b'))
        json.dump(res, open(os.path.join(H, 'results', 'interchange_icu.json'), 'w'), indent=1, default=str)
        for k in ('set_a', 'set_b'):
            r = res[k]; print(k, {x: r[x] for x in ('n', 'banded_gap_bits', 'banded_markov_null95', 'raw_markov_null95_of_banded_gap', 'p_raw_markov', 'decision_primary')}, r['decomposition'], {d: r['sufficiency'][d]['fisher_p'] for d in ('D1', 'D3')})
