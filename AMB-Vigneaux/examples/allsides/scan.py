"""Where does the weekly left-right adversarial gap change regime? Scan the Slepian step test over all weeks
(26-week margins); null = moving-block bootstrap (block 8 weeks) of residuals from the trend-only fit."""
import sys, json, datetime
import numpy as np
sys.path.insert(0, "../..")
import load, events
from amb_vigneaux.spectral import slepian_step_test, slepian_trend_basis
R = load.load(); rate, den = events.weekly(R, "n_adv", "n_sent")
ok = (den[0] > 0) & (den[2] > 0); g = rate[0] - rate[2]
g = np.where(ok, g, np.interp(np.arange(len(g)), np.where(ok)[0], g[ok])); n = len(g)
taus = range(26, n - 26)
zs = np.array([slepian_step_test(g, t, NW=1)["z"] for t in taus]); i = int(np.argmax(abs(zs)))
X = np.column_stack([np.ones(n), np.linspace(-1, 1, n), slepian_trend_basis(n, 1)]); b = np.linalg.lstsq(X, g, rcond=None)[0]
fit, res = X @ b, g - X @ b
rng = np.random.default_rng(0); null = []
for _ in range(300):
    st = rng.integers(0, n - 8, size=n // 8 + 1); e = np.concatenate([res[s:s + 8] for s in st])[:n]
    y = fit + e; null.append(max(abs(slepian_step_test(y, t, NW=1)["z"]) for t in taus[::2]))
p = float(np.mean(np.array(null) >= abs(zs[i])))
wk = events.W0 + datetime.timedelta(weeks=list(taus)[i])
top = sorted(zip(abs(zs), taus), reverse=True)[:5]
print(f"max |z| {abs(zs[i]):.2f} at week of {wk} (step {zs[i]:+.2f} z); block-bootstrap scan p = {p:.3f}")
print("top weeks:", [(str(events.W0 + datetime.timedelta(weeks=t)), round(float(z), 2)) for z, t in top])
yr = {y: float(np.mean(g[[(events.W0 + datetime.timedelta(weeks=k)).year == y for k in range(n)]])) for y in range(2015, 2021)}
print("yearly mean gap L-R:", {k: round(v, 4) for k, v in yr.items()})
json.dump(dict(max_z=float(abs(zs[i])), week=str(wk), p=p, yearly=yr), open("scan_results.json", "w"), indent=1)
