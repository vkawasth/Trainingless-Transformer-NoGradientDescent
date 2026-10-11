"""Synthetic data in the layout and column names of MIMIC-IV v3.1 (hosp/ and icu/ modules, gzipped CSV), for testing
mimic_monitor.py without credentialed data. Nothing here is real. Planted answers, checked by check_synthetic.py:

  * hemodynamic state Z_t over the three windows is a latent Markov chain; the observed state H_t is Z_t seen through MAP noise
    (so the observed answers are not glued, but a latent Markov chain fits);
  * in-hospital death depends on how many windows were in shock and on creatinine, not on the order of the windows;
  * lactate is measured with a probability that depends on the creatinine band only (missing at random given it), so the
    available-case tables are incompatible and inverse-probability weighting on (H_1, creatinine) removes the incompatibility;
  * care units differ only in their margins (direct influence): no contextuality across units.
Usage: python3 make_synthetic_mimic.py OUT_DIR [N_STAYS] [SEED]"""
import os, sys, numpy as np, pandas as pd
out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 6000; rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
os.makedirs(os.path.join(out, 'hosp'), exist_ok=True); os.makedirs(os.path.join(out, 'icu'), exist_ok=True)
w = lambda df, mod, name: df.to_csv(os.path.join(out, mod, name + '.csv.gz'), index=False, compression='gzip')
GROUPS = ['2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019', '2020 - 2022']
UNITS = ['Medical Intensive Care Unit (MICU)', 'Surgical Intensive Care Unit (SICU)', 'Cardiac Vascular Intensive Care Unit (CVICU)']
subj = 10000000 + np.arange(N); hadm = 20000000 + np.arange(N); stay = 30000000 + np.arange(N)
base = pd.Timestamp('2150-01-01')
intime = base + pd.to_timedelta(rng.integers(0, 3650 * 24, N), unit='h')
los = np.where(rng.random(N) < 0.15, rng.uniform(0.5, 1.9, N), rng.uniform(2.0, 9.0, N))
unit = rng.choice(3, N, p=[0.5, 0.3, 0.2]); grp = rng.choice(5, N)
# latent hemodynamic chain: 0 shock, 1 borderline, 2 stable; units shift the initial law (direct influence)
p0 = np.array([[0.30, 0.30, 0.40], [0.22, 0.33, 0.45], [0.35, 0.25, 0.40]])
T = np.array([[0.70, 0.20, 0.10], [0.15, 0.60, 0.25], [0.05, 0.15, 0.80]])
Z = np.zeros((N, 3), int)
Z[:, 0] = [rng.choice(3, p=p0[u]) for u in unit]
for t in (1, 2): Z[:, t] = [rng.choice(3, p=T[z]) for z in Z[:, t - 1]]
R = np.array([rng.choice(3, p=[[0.35, 0.35, 0.30], [0.55, 0.30, 0.15], [0.70, 0.22, 0.08]][z]) for z in Z[:, 0]])
L = np.array([rng.choice(3, p=[[0.25, 0.35, 0.40], [0.55, 0.30, 0.15], [0.75, 0.20, 0.05]][z]) for z in Z[:, 1]])
nshock = (Z == 0).sum(1)
pdeath = 1 / (1 + np.exp(-(-2.6 + 0.8 * nshock + 0.5 * R)))
dead = rng.random(N) < pdeath
age = rng.integers(18, 95, N)
w(pd.DataFrame(dict(subject_id=subj, gender=rng.choice(['F', 'M'], N), anchor_age=age, anchor_year=2150 + rng.integers(0, 10, N),
                    anchor_year_group=[GROUPS[g] for g in grp], dod=[(t + pd.Timedelta(days=5)).date() if d else None for t, d in zip(intime, dead)])), 'hosp', 'patients')
adm = intime - pd.to_timedelta(rng.uniform(1, 30, N), unit='h'); dis = intime + pd.to_timedelta(los + rng.uniform(1, 5, N), unit='D')
w(pd.DataFrame(dict(subject_id=subj, hadm_id=hadm, admittime=adm, dischtime=dis, deathtime=[x if d else None for x, d in zip(dis, dead)],
                    admission_type=np.random.default_rng(99).choice(['EW EMER.', 'URGENT', 'ELECTIVE'], N, p=[0.6, 0.25, 0.15]), hospital_expire_flag=dead.astype(int))), 'hosp', 'admissions')
w(pd.DataFrame(dict(subject_id=subj, hadm_id=hadm, stay_id=stay, first_careunit=[UNITS[u] for u in unit], last_careunit=[UNITS[u] for u in unit],
                    intime=intime, outtime=intime + pd.to_timedelta(los, unit='D'), los=los)), 'icu', 'icustays')
w(pd.DataFrame(dict(itemid=[220052, 220181, 225312, 221906, 221289, 221749, 222315, 221662, 225792, 220045],
                    label=['Arterial Blood Pressure mean', 'Non Invasive Blood Pressure mean', 'ART BP Mean', 'Norepinephrine', 'Epinephrine',
                           'Phenylephrine', 'Vasopressin', 'Dopamine', 'Invasive Ventilation', 'Heart Rate'],
                    abbreviation='', linksto=['chartevents'] * 3 + ['inputevents'] * 5 + ['procedureevents', 'chartevents'], category='', unitname='',
                    param_type='', lownormalvalue=None, highnormalvalue=None)), 'icu', 'd_items')
w(pd.DataFrame(dict(itemid=[50912, 50813, 50971], label=['Creatinine', 'Lactate', 'Potassium'], fluid='Blood', category='Chemistry')), 'hosp', 'd_labitems')
# chart: hourly MAP over the first 50 hours; window t covers hours [0,6), [21,27), [42,48); MAP drawn from the latent state with noise
WIN = [(0, 6), (21, 27), (42, 48)]; rows = []
mapmean = {0: 58, 1: 70, 2: 84}
for i in range(N):
    hrs = np.arange(0, min(50, int(los[i] * 24)))
    for h in hrs:
        t = 0 if h < 14 else (1 if h < 35 else 2); zt = Z[i, t]
        if zt == 0 and rng.random() < 0.6: continue                     # many shock hours are on pressors, MAP charted less
        rows.append((subj[i], hadm[i], stay[i], intime[i] + pd.Timedelta(hours=float(h) + 0.25), 220181 if rng.random() < 0.5 else 220052,
                     mapmean[zt] + rng.normal(0, 9)))
ce = pd.DataFrame(rows, columns=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'valuenum'])
ce['caregiver_id'] = 1; ce['storetime'] = ce['charttime'] + pd.Timedelta(minutes=10); ce['value'] = ce['valuenum'].round(0).astype(str); ce['valueuom'] = 'mmHg'; ce['warning'] = 0
hr = ce.sample(frac=0.3, random_state=1).assign(itemid=220045, valuenum=lambda x: rng.normal(90, 15, len(x)))
w(pd.concat([ce, hr])[['subject_id', 'hadm_id', 'stay_id', 'caregiver_id', 'charttime', 'storetime', 'itemid', 'value', 'valuenum', 'valueuom', 'warning']], 'icu', 'chartevents')
# inputs: a pressor infusion through each window spent in latent shock (70% of them)
rows = []
for i in range(N):
    for t, (a, b) in enumerate(WIN):
        if Z[i, t] == 0 and rng.random() < 0.7 and los[i] * 24 > b:
            rows.append((subj[i], hadm[i], stay[i], intime[i] + pd.Timedelta(hours=a - 2), intime[i] + pd.Timedelta(hours=b - 1), rng.choice([221906, 222315, 221749])))
ie = pd.DataFrame(rows, columns=['subject_id', 'hadm_id', 'stay_id', 'starttime', 'endtime', 'itemid'])
ie['caregiver_id'] = 1; ie['storetime'] = ie['starttime']; ie['amount'] = 1.0; ie['amountuom'] = 'mg'; ie['rate'] = 0.1; ie['rateuom'] = 'mcg/kg/min'
w(ie[['subject_id', 'hadm_id', 'stay_id', 'caregiver_id', 'starttime', 'endtime', 'storetime', 'itemid', 'amount', 'amountuom', 'rate', 'rateuom']], 'icu', 'inputevents')
vent = rng.random(N) < np.array([0.6, 0.35, 0.2])[Z[:, 0]] + 0.05 * (unit == 1)
pe = pd.DataFrame(dict(subject_id=subj[vent], hadm_id=hadm[vent], stay_id=stay[vent], caregiver_id=1, starttime=intime[vent] + pd.Timedelta(hours=2),
                       endtime=intime[vent] + pd.Timedelta(hours=30), storetime=intime[vent], itemid=225792, value=28.0, valueuom='hour'))
w(pe, 'icu', 'procedureevents')
# labs: creatinine always in the first day; lactate with probability depending on (H1 latent, creatinine band)
CR = {0: (0.6, 1.15), 1: (1.2, 1.95), 2: (2.0, 4.5)}; LA = {0: (0.6, 1.95), 1: (2.0, 3.95), 2: (4.0, 9.0)}
pm = np.array([0.25, 0.60, 0.90])                                  # depends on the creatinine band only
rows = []
for i in range(N):
    rows.append((i * 3, subj[i], hadm[i], 50912, intime[i] + pd.Timedelta(hours=3), rng.uniform(*CR[R[i]])))
    if rng.random() < pm[R[i]]:
        rows.append((i * 3 + 1, subj[i], hadm[i], 50813, intime[i] + pd.Timedelta(hours=4), rng.uniform(*LA[L[i]])))
    rows.append((i * 3 + 2, subj[i], hadm[i], 50971, intime[i] + pd.Timedelta(hours=3), rng.normal(4.1, 0.4)))
le = pd.DataFrame(rows, columns=['labevent_id', 'subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'])
le['specimen_id'] = le['labevent_id']; le['order_provider_id'] = None; le['storetime'] = le['charttime'] + pd.Timedelta(hours=1); le['value'] = le['valuenum'].round(2).astype(str)
le['valueuom'] = 'mg/dL'; le['ref_range_lower'] = None; le['ref_range_upper'] = None; le['flag'] = None; le['priority'] = 'STAT'; le['comments'] = None
w(le[['labevent_id', 'subject_id', 'hadm_id', 'specimen_id', 'itemid', 'order_provider_id', 'charttime', 'storetime', 'value', 'valuenum', 'valueuom',
      'ref_range_lower', 'ref_range_upper', 'flag', 'priority', 'comments']], 'hosp', 'labevents')
print(f"wrote synthetic MIMIC-IV-shaped data for {N} ICU stays to {out}")
