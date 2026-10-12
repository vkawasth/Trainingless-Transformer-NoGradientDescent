"""EXPLORATORY (after the frozen run of recursive_training.py): are zeros absorbing with fresh data, and how do the mean accuracy and
certificate curves compare? Writes results/recursive_training_exploratory.json"""
import json, os, numpy as np
import recursive_training as R
rng = np.random.default_rng(R.SEED); Pxc, P0 = R.truth(rng); P0full = np.broadcast_to(P0, (R.C, R.X, R.Y)).copy(); out = {}
for lam in R.LAMS:
    Qs = np.array([R.run_chain(P0full, lam, None, np.random.default_rng(R.SEED + 10 * r + int(lam * 100))) for r in range(R.REPS)])
    pos = np.broadcast_to(P0 > 0, Qs.shape[2:]); z = Qs[:, :, pos] == 0          # (rep, gen, tracks)
    absorbed = np.array([[z[r, g:, t].all() for t in range(z.shape[2])] for r in range(R.REPS) for g in (10,)]).mean()
    recovered = (z[:, :-1] & ~z[:, 1:]).sum() / max(z[:, :-1].sum(), 1)
    out[str(lam)] = dict(share_zero_from_g10_to_g20=round(float(absorbed), 4), recovery_rate_per_zero_generation=round(float(recovered), 4))
json.dump(out, open(os.path.join(R.H, 'results', 'recursive_training_exploratory.json'), 'w'), indent=1); print(out)
