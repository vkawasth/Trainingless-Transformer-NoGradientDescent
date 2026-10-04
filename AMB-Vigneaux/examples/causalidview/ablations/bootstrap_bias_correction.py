import os, sys, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
os.environ.setdefault("CAUSALIDVIEW_DIR", "/tmp/ds/CausalIDView")
import run as R
NB = int(os.environ.get("NB", 20)); res = {}
for seed in [int(s) for s in os.environ.get("SEEDS", "0 1 2 3 4 5").split()]:
    W = R.world(seed); Lmo, Umo = R.manski(W["q"].mean(1)); ivo = np.array([R.iv_bounds(W["q"][i]) for i in range(len(W["qry"]))])
    E = R.estimate_ours(W); rng = np.random.default_rng(100 + seed); Bs = []
    for _ in range(NB):
        Wb = dict(W); Wb["ctx"] = rng.choice(W["ctx"], len(W["ctx"]), replace=True); Bs.append(R.estimate_ours(Wb))
    o = {}
    mb = lambda k: np.mean([b[k] for b in Bs], 0)
    for name, f in {"raw": lambda k: E[k], "bag": mb, "BC": lambda k: 2 * E[k] - mb(k), "half-BC": lambda k: 1.5 * E[k] - 0.5 * mb(k)}.items():
        o[name] = (R.ep_rmse(f("Lm"), f("Um"), Lmo, Umo), R.ep_rmse(f("Li"), f("Ui"), ivo[:, 0], ivo[:, 1]))
    o["bias Li,Ui"] = (float(np.mean(E["Li"] - ivo[:, 0])), float(np.mean(E["Ui"] - ivo[:, 1])))
    o["bias Lm,Um"] = (float(np.mean(E["Lm"] - Lmo)), float(np.mean(E["Um"] - Umo)))
    for k, v in o.items(): res.setdefault(k, []).append(v)
    print(seed, {k: tuple(round(x, 4) for x in v) for k, v in o.items()}, flush=True)
print({k: tuple(np.round(np.mean(v, 0), 4)) for k, v in res.items()})
