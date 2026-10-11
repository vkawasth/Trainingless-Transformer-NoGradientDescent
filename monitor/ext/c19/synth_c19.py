"""Synthetic data in the C19PRC UK and Irish SPSS layouts (variable names and codes as in the archive metadata), with planted answers:
latent first-order Markov depression over the waves; recontacted and fresh respondents differ only in their margins; attrition at W3
depends on (D1, A1) only; outcome depends on the number of waves above threshold. Usage: python3 synth_c19.py OUT_DIR"""
import os, sys, io, zipfile, numpy as np, pandas as pd, pyreadstat
out = sys.argv[1]; os.makedirs(os.path.join(out, 'uk'), exist_ok=True); rng = np.random.default_rng(11)
R = np.array([[0.85, 0.12, 0.03], [0.15, 0.7, 0.15], [0.03, 0.12, 0.85]]); T = np.array([[0.8, 0.15, 0.05], [0.15, 0.7, 0.15], [0.05, 0.2, 0.75]])
SC = {0: (0, 4), 1: (5, 9), 2: (10, 27)}
def score(c): lo, hi = SC[c]; return int(rng.integers(lo, hi + 1))
def items(total, k):          # split a total into k items 0-3
    v = np.zeros(k, int)
    for _ in range(min(total, 3 * k)):
        j = rng.choice([i for i in range(k) if v[i] < 3]); v[j] += 1
    return v
def cohort(n, waves, shift=0.0):
    Z = np.zeros((n, waves), int); Z[:, 0] = rng.choice(3, n, p=[0.5 - shift, 0.3, 0.2 + shift])
    for t in range(1, waves): Z[:, t] = [rng.choice(3, p=T[z]) for z in Z[:, t - 1]]
    D = np.vectorize(lambda z: rng.choice(3, p=R[z]))(Z); A = np.vectorize(lambda z: rng.choice(3, p=R[z]))(Z)
    return D, A
def pid(prefix, n): return [f'{prefix}{i:05d}' for i in range(n)]
# ---------- UK: W1 cohort of 2025 followed six waves with attrition; top-ups at W3, W4, W6
n = 2025; D, A = cohort(n, 6); ids = pid('uk', n)
stay = np.ones((n, 6), bool)
p3 = np.array([[0.75, 0.65, 0.55], [0.65, 0.55, 0.45], [0.55, 0.45, 0.35]])[D[:, 0], A[:, 0]]
stay[:, 1] = rng.random(n) < 0.7; stay[:, 2] = rng.random(n) < p3
for t in (3, 4, 5): stay[:, t] = stay[:, t - 1] & (rng.random(n) < 0.85) | (rng.random(n) < 0.05)
def wave_frame(t, sel, Dc, Ac, idl, extra=None):
    d = pd.DataFrame({'pid': np.array(idl)[sel]}); dt = np.array([score(c) for c in Dc[sel]]); at = np.array([score(c) for c in Ac[sel]])
    dn = {1: 'W1_Depression_Total', 2: 'W2_Depression_Total'}.get(t, f'W{t}_Dep_Total'); d[dn] = dt; d[f'W{t}_GAD_Total'] = at
    if extra: d = d.assign(**extra)
    return d
w1 = wave_frame(1, stay[:, 0], D[:, 0], A[:, 0], ids); w2 = wave_frame(2, stay[:, 1], D[:, 1], A[:, 1], ids)
w12 = w1.merge(w2, on='pid', how='left'); w12['WAVE_1'] = 1; w12['WAVE_2'] = w12['W2_GAD_Total'].notna().astype(int)
pyreadstat.write_sav(w12, os.path.join(out, 'uk', 'C19PRC_UKW1W2_archive_final.sav'))
yes = rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 0.7 * (D[:, :3] == 2).sum(1))))
for t, fn, tc, codes in ((3, 'C19PRC_UK_W3_archive_final.sav', 'W3_Type', (0, 1)), (4, 'C19PRC_UK_W4_archive_final.sav', 'W4_Type', (1, 2)),
                         (5, 'C19PRC_UKW5_archive_final.sav', None, None), (6, 'C19PRC_UK_W6_archive_final.sav', 'W6_Type', (1, 0))):
    d = wave_frame(t, stay[:, t - 1], D[:, t - 1], A[:, t - 1], ids)
    if tc:
        d[tc] = codes[0]; m = 800; Df, Af = cohort(m, 1, shift=0.08)
        f = wave_frame(t, np.ones(m, bool), Df[:, 0], Af[:, 0], pid(f'top{t}_', m)); f[tc] = codes[1]; d = pd.concat([d, f], ignore_index=True)
    if t == 3: d['W3_weight'] = 1.0
    if t == 6: d['W6_Suicide_Ideation'] = [int(yes[ids.index(p)]) if p in ids else 0 for p in d.pid]
    pyreadstat.write_sav(d, os.path.join(out, 'uk', fn))
# ---------- Ireland: one ALLWAVES file, items DEP1-9 / GAD_1-7 (W1), W2_DEP1.. / W2_GAD_1.., W3_GAD1.. etc.
n = 1041; D, A = cohort(n, 5); rows = []; fresh = {2: 1032, 4: 1098, 5: 1110}
stay = np.ones((n, 5), bool); stay[:, 1] = rng.random(n) < 0.6
stay[:, 2] = rng.random(n) < np.array([[0.65, 0.55, 0.45], [0.55, 0.45, 0.35], [0.45, 0.35, 0.3]])[D[:, 0], A[:, 0]]
stay[:, 3] = stay[:, 2] & (rng.random(n) < 0.8); stay[:, 4] = stay[:, 3] & (rng.random(n) < 0.85)
df = pd.DataFrame({'pid': pid('ie', n)})
def put(df, t, sel, Dc, Ac):
    dp = {1: 'DEP', 2: 'W2_DEP', 3: 'W3_DEP', 4: 'W4_DEP', 5: 'W5_DEP'}[t]; gp = {1: 'GAD_', 2: 'W2_GAD_', 3: 'W3_GAD', 4: 'W4_GAD', 5: 'W5_GAD'}[t]
    for i in np.where(sel)[0]:
        for j, v in enumerate(items(score(Dc[i]), 9), 1): df.loc[i, f'{dp}{j}'] = v
        for j, v in enumerate(items(score(Ac[i]), 7), 1): df.loc[i, f'{gp}{j}'] = v
    return df
for t in range(1, 6): df = put(df, t, stay[:, t - 1], D[:, t - 1], A[:, t - 1])
for t, m in fresh.items():
    Df, Af = cohort(m, 1, shift=0.08); f = pd.DataFrame({'pid': pid(f'iefresh{t}_', m)}); f = put(f, t, np.ones(m, bool), Df[:, 0], Af[:, 0]); df = pd.concat([df, f], ignore_index=True)
df['W2_Wave1_Recontact'] = np.where(df.pid.str.startswith('ie0') & df['W2_DEP1'].notna(), 1.0, np.where(df['W2_DEP1'].notna(), 2.0, np.nan))
df['W4_Wave1_Recontact'] = np.where(df.pid.str.startswith('ie0') & df['W4_DEP1'].notna(), 1.0, np.where(df['W4_DEP1'].notna(), 0.0, np.nan))
df['W3_Weight_Variable'] = 1.0
yes = rng.random(n) < 1 / (1 + np.exp(-(-2.0 + 0.8 * (D[:, :3] == 2).sum(1))))
df['W4_Suicidal_Ideation_1week'] = np.nan; idx = np.where(stay[:, 3])[0]; df.loc[idx, 'W4_Suicidal_Ideation_1week'] = np.where(yes[idx], 2.0, 1.0)
p = os.path.join(out, 'allw.sav')
pyreadstat.write_sav(df, p, variable_value_labels={'W2_Wave1_Recontact': {1.0: 'Recontact', 2.0: 'Fresh'}, 'W4_Suicidal_Ideation_1week': {1.0: 'No', 2.0: 'Yes'}})
with zipfile.ZipFile(os.path.join(out, 'ie_archive.zip'), 'w') as z: z.write(p, 'Mental Health ASAP Datasets/MHASAP.ALLWAVES.sav')
print('wrote synthetic UK and Irish layouts to', out)
