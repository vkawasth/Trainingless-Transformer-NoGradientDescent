"""How much can a support prior buy? Ridge (closed-form LOO) on the TRUE support vs all 50 coordinates."""
import os, sys, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R
S = R.S
def supports(seed):
    fam = S.CATE_FAMILIES[seed % len(S.CATE_FAMILIES)]; rng = S._rng(seed, "nuisance_parameters")
    mu = S._unit_vector(rng, R.D, 14); vT = S._unit_vector(rng, R.D, 14); cp = S._cate_parameters(seed, R.D, fam)
    tau = (cp["a"] != 0) | (cp["b"] != 0)
    if "hidden_w" in cp: tau |= (cp["hidden_w"] != 0).any(0)
    return dict(T=vT != 0, b=mu != 0, tau=tau)
if __name__ == "__main__":
    res = []
    for seed in range(int(os.environ.get("NW", 10))):
        W = R.world(seed); c, q = W["ctx"], W["qry"]; X = W["X"]; y = (W["Y"][c] == W["T"][c]).astype(float)
        Lo, Uo = R.manski(W["q"].mean(1)); sp = supports(seed); U = sp["T"] | sp["b"] | sp["tau"]
        rm = lambda s: float(np.sqrt(np.mean((s - Uo) ** 2)))
        a = rm(R._krr_linear_loo(X[c], y, X[q])); b = rm(R._krr_linear_loo(X[c][:, U], y, X[q][:, U]))
        res.append((a, b, U.sum())); print(seed, W["family"][:10], f"all50 {a:.3f}  true-support({U.sum()}) {b:.3f}", flush=True)
    print("mean", np.round(np.mean(res, 0), 4))
