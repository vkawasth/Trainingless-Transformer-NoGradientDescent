import json, sys, numpy as np
from prefix_post import load_true, prefix_posteriors
M = json.load(open('cH/rhm_meta.json'))
src, out = sys.argv[1], sys.argv[2]
if src == 'true': P = load_true(M)
else:
    Z = np.load(src); P = {l: Z[f'P{l}'] for l in range(1, M['depth'] + 1)}
tr = np.array(json.load(open('cH/train_ids.json'))).reshape(-1, 16)
va = np.array(json.load(open('cH/val_ids.json'))).reshape(-1, 16)
D = {'Xtr': tr[:19000], 'Xdv': tr[19000:20000], 'Xva': va}
for k in ('tr', 'dv', 'va'):
    pred, anc = prefix_posteriors(D['X' + k], P)
    D[f'{k}_pred'] = pred.astype(np.float32)
    for l in anc: D[f'{k}_anc{l}'] = anc[l].astype(np.float32)
X = D['Xva']; nll = -np.log(D['va_pred'][np.arange(len(X))[:, None], np.arange(16)[None], X])
np.save(out.replace('.npz', '_valnll.npy'), nll)
np.savez_compressed(out, **D); print(src, 'val nll/seq', nll.sum(1).mean())
