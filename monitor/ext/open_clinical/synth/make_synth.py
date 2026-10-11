"""Synthetic data in the layouts of the three open datasets, with planted answers, for testing the adapters before the real data are
analysed. Nothing here is real. Usage: python3 make_synth.py OUT_DIR"""
import os, sys, io, zipfile, numpy as np, pandas as pd
out = sys.argv[1]; os.makedirs(out, exist_ok=True); rng = np.random.default_rng(7)
T = np.array([[0.70, 0.20, 0.10], [0.15, 0.60, 0.25], [0.05, 0.15, 0.80]])
def chain(n, p0):
    Z = np.zeros((n, 3), int); Z[:, 0] = [rng.choice(3, p=p) for p in p0]
    for t in (1, 2): Z[:, t] = [rng.choice(3, p=T[z]) for z in Z[:, t - 1]]
    return Z
# ---- Challenge 2012 layout: one text file per record, Time,Parameter,Value; outcomes file
N = 4000; icu = rng.choice([1, 2, 3, 4], N, p=[0.15, 0.2, 0.35, 0.3])
p0 = {1: [0.25, 0.35, 0.40], 2: [0.30, 0.40, 0.30], 3: [0.30, 0.30, 0.40], 4: [0.22, 0.33, 0.45]}
Z = chain(N, [p0[i] for i in icu])
R = np.array([rng.choice(3, p=[[0.35, 0.35, 0.30], [0.55, 0.30, 0.15], [0.70, 0.22, 0.08]][z]) for z in Z[:, 0]])
L = np.array([rng.choice(3, p=[[0.25, 0.35, 0.40], [0.55, 0.30, 0.15], [0.75, 0.20, 0.05]][z]) for z in Z[:, 1]])
vent = rng.random(N) < np.array([0.6, 0.4, 0.25])[Z[:, 1]]
death = rng.random(N) < 1 / (1 + np.exp(-(-2.6 + 0.8 * (Z == 0).sum(1) + 0.4 * R)))
pm = np.array([0.25, 0.6, 0.9]); mapm = {0: 58, 1: 70, 2: 84}
CR = {0: (0.6, 1.15), 1: (1.2, 1.95), 2: (2.0, 4.5)}; LA = {0: (0.6, 1.95), 1: (2.0, 3.95), 2: (4.0, 9.0)}
buf = io.BytesIO(); zf = zipfile.ZipFile(buf, 'w')
for i in range(N):
    rid = 132000 + i; lines = ['Time,Parameter,Value', f'00:00,RecordID,{rid}', f'00:00,Age,{rng.integers(18, 90)}', '00:00,Gender,1', '00:00,Height,-1',
                               f'00:00,ICUType,{icu[i]}', '00:00,Weight,80']
    for h in range(0, 48):
        t = 0 if h < 14 else (1 if h < 35 else 2)
        lines.append(f'{h:02d}:30,{"MAP" if rng.random() < 0.6 else "NIMAP"},{mapm[Z[i, t]] + rng.normal(0, 9):.0f}')
    lines.append(f'03:00,Creatinine,{rng.uniform(*CR[R[i]]):.2f}')
    if rng.random() < pm[R[i]]: lines.append(f'04:00,Lactate,{rng.uniform(*LA[L[i]]):.2f}')
    lines.append(f'05:00,MechVent,{1 if vent[i] else 0}')
    zf.writestr(f'set-a/{rid}.txt', '\n'.join(lines) + '\n')
zf.close(); open(os.path.join(out, 'set-a.zip'), 'wb').write(buf.getvalue())
pd.DataFrame(dict(RecordID=132000 + np.arange(N), **{'SAPS-I': rng.integers(5, 25, N)}, SOFA=rng.integers(0, 15, N), Length_of_stay=rng.integers(2, 30, N),
                  Survival=-1, **{'In-hospital_death': death.astype(int)})).to_csv(os.path.join(out, 'Outcomes-a.txt'), index=False)
# ---- Diabetes 130 layout
P = 15000; n_enc = rng.choice([1, 2, 3, 4], P, p=[0.6, 0.2, 0.12, 0.08]); rows = []; eid = 1000
Zp = chain(P, [[0.45, 0.35, 0.20]] * P)
for pnum in range(P):
    age = rng.choice(10); src = rng.choice([7, 1, 4], p=[0.55, 0.35, 0.10])
    for k in range(n_enc[pnum]):
        zk = Zp[pnum, min(k, 2)]; X = 0 if age < 5 else (1 if age < 7 else 2)
        Y = rng.choice(3, p=[[0.6, 0.3, 0.1], [0.5, 0.35, 0.15], [0.45, 0.35, 0.2]][zk])
        meas = rng.random() < [[0.1, 0.15, 0.3], [0.2, 0.25, 0.4], [0.15, 0.2, 0.3]][X][Y]       # MAR on (X, Y)
        a1c = rng.choice(['Norm', '>7', '>8'], p=[[0.6, 0.25, 0.15], [0.4, 0.3, 0.3], [0.3, 0.3, 0.4]][Y]) if meas else 'None'
        ins = ['No', 'Steady', rng.choice(['Up', 'Down'])][zk] if rng.random() > 0.15 else rng.choice(['No', 'Steady', 'Up'])
        chg = 'Ch' if rng.random() < [0.3, 0.45, 0.7][zk] + 0.05 * (src == 1) else 'No'
        rows.append(dict(encounter_id=eid, patient_nbr=500000 + pnum, race='Caucasian', gender='Female', age=f'[{age*10}-{age*10+10})', weight='?',
                         admission_type_id=rng.choice([1, 2, 3]), discharge_disposition_id=1, admission_source_id=src, time_in_hospital=3,
                         max_glu_serum='None', A1Cresult=a1c, insulin=ins, change=chg, diabetesMed='Yes',
                         readmitted=['NO', '>30', '<30'][Y])); eid += rng.integers(1, 50)
dd = pd.DataFrame(rows); b = io.BytesIO(); zf = zipfile.ZipFile(b, 'w'); zf.writestr('diabetic_data.csv', dd.to_csv(index=False)); zf.writestr('IDS_mapping.csv', 'admission_type_id,description\n1,Emergency\n')
zf.close(); open(os.path.join(out, 'diabetes130.zip'), 'wb').write(b.getvalue())
# ---- Heart disease layout: four sites, same mechanism, different margins
b = io.BytesIO(); zf = zipfile.ZipFile(b, 'w')
for site, n, shift in (('cleveland', 303, 0.0), ('hungarian', 294, 0.1), ('switzerland', 123, 0.2), ('va', 200, 0.15)):
    lines = []
    for _ in range(n):
        dx = int(rng.random() < 0.45 + shift); cp = rng.choice([1, 2, 3, 4], p=[[0.15, 0.25, 0.35, 0.25], [0.05, 0.1, 0.2, 0.65]][dx])
        ex = int(rng.random() < [0.15, 0.55][dx] + (0.1 if cp == 4 else 0))
        thal = '?' if rng.random() < shift * 3 else str(rng.choice([3, 6, 7]))
        lines.append(f'{rng.integers(30, 77)},1,{cp},130,{"?" if site == "switzerland" else 240},0,0,150,{ex},1.0,2,0,{thal},{dx * rng.integers(1, 5)}')
    zf.writestr(f'processed.{site}.data', '\n'.join(lines) + '\n')
zf.close(); open(os.path.join(out, 'heartdisease.zip'), 'wb').write(b.getvalue())
print('wrote synthetic 2012, Diabetes 130 and Heart Disease layouts to', out)
