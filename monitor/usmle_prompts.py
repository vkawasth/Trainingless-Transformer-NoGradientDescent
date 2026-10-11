"""USMLE in the LLM-drift data asks every question twice: once with the intact prompt and once with a damaged prompt in which a line of
the question, or the first option, is missing (99% of the questions; option (A) is absent in 10% of the damaged prompts).
This is a second change of information on the same items, beside the update: contexts intact / damaged for each model version.
Per version: accuracy under each prompt, the churn between them (Prop. excess churn), and the regime of the system
(GPT-3.5, GPT-4) from intact to damaged; and the March -> June change measured on the damaged prompts."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monitor import churn, State, regime
from llm_analysis import outcomes, MODELS, table
I, Dm = outcomes('USMLE', 'intact'), outcomes('USMLE', 'damaged')
ids = I.index.intersection(Dm.index); I, Dm = I.loc[ids], Dm.loc[ids]; n = len(ids)
out = {'n_items': n}
for m in MODELS:
    c = churn((I[m] != 0).astype(int).values, (Dm[m] != 0).astype(int).values, [0, 1]); T = c['T']
    out[m] = dict(acc_intact=round(float((I[m] == 0).mean()), 3), acc_damaged=round(float((Dm[m] == 0).mean()), 3),
                  lost=round(float(T[0, 1]), 3), gained=round(float(T[1, 0]), 3), excess=round(c['excess'], 3), disagree=round(1 - c['agree'], 3))
for v in ('M', 'J'):
    S0, S1 = State(table(I['35' + v], I['4' + v]) + 1e-9, n), State(table(Dm['35' + v], Dm['4' + v]) + 1e-9, n)
    rg = regime(S0, S1); out['system ' + v] = dict(kind=rg['kind'], stress=round(rg['stress'], 2), crossed=[[list(a), list(b)] for a, b in rg['walls_crossed']])
for fam in ('35', '4'):
    c = churn((Dm[fam + 'M'] != 0).astype(int).values, (Dm[fam + 'J'] != 0).astype(int).values, [0, 1]); T = c['T']
    out[fam + ' March->June damaged'] = dict(acc_M=round(float((Dm[fam + 'M'] == 0).mean()), 3), acc_J=round(float((Dm[fam + 'J'] == 0).mean()), 3),
                                            lost=round(float(T[0, 1]), 3), masked=round(float(min(T[0, 1], T[1, 0])), 3), excess=round(c['excess'], 3))
# did the update make the model more sensitive to the damaged prompt? difference of the damage cost, June minus March, paired bootstrap
rng = np.random.default_rng(3)
for fam in ('35', '4'):
    a = np.c_[(I[fam + 'M'] == 0), (Dm[fam + 'M'] == 0), (I[fam + 'J'] == 0), (Dm[fam + 'J'] == 0)].astype(float)
    dd = lambda x: (x[:, 2].mean() - x[:, 3].mean()) - (x[:, 0].mean() - x[:, 1].mean())
    bs = [dd(a[rng.integers(0, n, n)]) for _ in range(4000)]
    out[fam + ' damage cost June - March'] = dict(est=round(dd(a), 3), ci=[round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])])
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usmle_prompts.json'), 'w'), indent=1)
