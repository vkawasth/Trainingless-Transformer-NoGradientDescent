"""Clinical slice. (1) NHANES 2009-10 -> 2011-12 (NHANES R package, raw with exam weights): contents A = diabetes (yes / no; self-reported diagnosis),
B = BMI class (under/normal, overweight, obese); contexts = survey cycles; overall and age 55+. (2) USMLE: from llm_results.json."""
import os, sys, json
import numpy as np, pandas as pd, pyreadr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monitor import State, regime, cbd_rank2, cbd_split
df = list(pyreadr.read_r(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nhanes', 'NHANESraw.rda')).values())[0]
df = df[(df.Age >= 20)]
df['diab'] = df['Diabetes'].astype(str).map({'Yes': 0, 'No': 1})
df['bmi3'] = pd.cut(df['BMI'], [0, 25, 30, 200], labels=[0, 1, 2], right=False).astype(float)
df['w'] = df['WTMEC2YR']
out = {}
for grp, sub in (('adults 20+', df), ('age 55+', df[df.Age >= 55])):
    st = {}
    for cyc in sorted(sub.SurveyYr.astype(str).unique()):
        s = sub[(sub.SurveyYr.astype(str) == cyc) & sub.diab.notna() & sub.bmi3.notna() & sub.w.notna()]
        T = np.zeros((2, 3)); np.add.at(T, (s.diab.astype(int).values, s.bmi3.astype(int).values), s.w.values)
        w = s.w.values; st[cyc] = (State(T + 1e-9, w.sum() ** 2 / (w ** 2).sum()), len(s))
    (c0, (A, nA)), (c1, (B, nB)) = sorted(st.items())
    rg = regime(A, B)
    bA = np.array([[A.T[0, 2], A.T[0, :2].sum()], [A.T[1, 2], A.T[1, :2].sum()]])   # diabetes x obese (yes / no)
    bB = np.array([[B.T[0, 2], B.T[0, :2].sum()], [B.T[1, 2], B.T[1, :2].sum()]])
    cb = cbd_rank2(bA, bB); sp = cbd_split(A.T, B.T)
    risk = {c: round(float(S.T[0, 2] / S.T[:, 2].sum()), 4) for c, (S, n) in st.items()}
    out[grp] = dict(n={c: n for c, (S, n) in st.items()}, diabetes={c: round(float(S.f[0]), 4) for c, (S, n) in st.items()},
                    bmi={c: [round(float(x), 4) for x in S.g] for c, (S, n) in st.items()}, regime=rg['kind'], stress=round(rg['stress'], 2),
                    tear_margin=round(rg['mT'], 4), break_margin=round(rg['mB'], 4), crossed=[[list(I), list(J)] for I, J in rg['walls_crossed']],
                    cbd_contextual=cb['contextual'], s=round(cb['s_odd'], 4), Delta=round(cb['Delta'], 4), split_contextual=sp['contextual'], split_deficit=round(sp['deficit'], 5), diabetes_risk_if_obese=risk)
    print(f"NHANES {grp}: {c0}->{c1} n {out[grp]['n']} | diabetes {out[grp]['diabetes']} | BMI {out[grp]['bmi']} | regime {rg['kind']} stress {rg['stress']:.2f} "
          f"(tear margin {rg['mT']:.4f}) | CbD contextual {cb['contextual']}, all splits {sp['contextual']} | P(diabetes | obese) {risk}")
llm = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_results.json')))['USMLE']
out['USMLE'] = llm
print("USMLE (LLM on medical licensing questions):", llm['regime'], "| churn", llm['churn'])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clinical_results.json'), 'w'), indent=1, default=str)
