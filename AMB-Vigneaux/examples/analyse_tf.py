import json, glob, numpy as np
from prefix_post import boundary_level
floor = np.load('cH/feats_true_valnll.npy')
groups = {k: [t for t in range(16) if boundary_level(t) == k] for k in range(5)}
rows = {'fitted grammar (split-merge)': np.load('cH/feats_fit_valnll.npy')}
for f in sorted(glob.glob('runs/*.npy')): rows[f[5:-4]] = np.load(f)
def ci(d):  # paired bootstrap-free SE over sequences
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
print(f"{'model':32}{'total excess':>16}" + ''.join(f"{'lvl'+str(k):>14}" for k in range(5)))
out = {}
for n, v in rows.items():
    tot = ci((v - floor).sum(1)); per = [ci((v - floor)[:, g].sum(1)) for g in groups.values()]
    out[n] = {'total': tot, 'per_level': per}
    print(f"{n:32}{tot[0]:>9.3f}±{tot[1]:.3f}" + ''.join(f"{m:>8.3f}±{s:.3f}" for m, s in per))
print('floor per level (nats/seq):', [round(float(floor[:, g].sum(1).mean()), 3) for g in groups.values()])
json.dump(out, open('runs/summary.json', 'w'))
