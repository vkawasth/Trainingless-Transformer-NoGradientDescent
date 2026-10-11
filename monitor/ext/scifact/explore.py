"""Exploratory, after the frozen run: is the credal gate better than thresholding the naive-Bayes posterior at the same coverage?"""
import os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(os.path.join(H, 'results', 'scifact_dev_decisions.json')))
y = np.array([q['y'] for q in R]); r3 = np.array([q['r3'] for q in R]); conf = np.array([q['r3conf'] for q in R]); cid = np.array([q['cid'] for q in R])
out = {}
for name in ('r4', 'r5'):
    a = np.array([q[name] is not None for q in R]); k = int(a.sum()); top = np.argsort(-conf, kind='stable')[:k]
    out[name] = dict(acted=k, credal_accuracy=round(float(np.mean([R[i][name] == y[i] for i in np.where(a)[0]])), 4),
                     nb_same_coverage_accuracy=round(float((r3[top] == y[top]).mean()), 4), overlap=int(len(set(top) & set(np.where(a)[0]))))
    rng = np.random.default_rng(1); claims = np.unique(cid); groups = [np.where(cid == c)[0] for c in claims]; d = []
    for _ in range(2000):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]); aa = a[idx]; kk = int(aa.sum())
        if kk == 0: continue
        tp = idx[np.argsort(-conf[idx], kind='stable')[:kk]]; d.append(np.mean([R[i][name] == y[i] for i in idx[aa]]) - (r3[tp] == y[tp]).mean())
    out[name]['difference_ci'] = [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
# what makes the Frechet rule silent: how often all three verifiers agree, and the best possible Frechet lower likelihood
o = np.array([q['o'] for q in R]); out['all_three_agree'] = round(float(np.mean([len(set(t)) == 1 for t in o])), 4)
json.dump(out, open(os.path.join(H, 'results', 'scifact_explore.json'), 'w'), indent=1); print(json.dumps(out, indent=1))
# set-valued answers: is the gold label inside the hedge set, by hedge type
sv = {}
for q in R:
    if q['r5'] is None:
        k = '+'.join(['S', 'C', 'N'][z] for z in q['r5hedge']); sv.setdefault(k, [0, 0]); sv[k][0] += 1; sv[k][1] += int(q['y'] in q['r5hedge'])
out['r5_hedges_gold_inside'] = {k: dict(n=v[0], gold_inside=v[1], share=round(v[1] / v[0], 4)) for k, v in sv.items()}
out['r5_asserted_labels'] = sorted({['S', 'C', 'N'][q['r5']] for q in R if q['r5'] is not None})
json.dump(out, open(os.path.join(H, 'results', 'scifact_explore.json'), 'w'), indent=1); print(out['r5_hedges_gold_inside'], out['r5_asserted_labels'])
