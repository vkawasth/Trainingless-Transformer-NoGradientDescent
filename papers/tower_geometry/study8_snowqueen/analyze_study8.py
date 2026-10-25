"""Study 8 analysis (frozen with the pre-registration). Reads data_study8.json:
{"n": {"11": int, "12": int, "21": int, "22": int},
 "D2": {"11": [[p(+,+), p(+,-)], [p(-,+), p(-,-)]], ...}, "D3": {...}}
with rows = A (+1 first), columns = B (+1 first). Writes results_study8.json and prints the predictions' outcomes."""
import json, os, sys, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from pce import Window
from pce.fingerprint import fingerprint
from pce.cbd import kd_cyclic, signalling, cbd
from pce.npa import npa_fraction
from pce.integrate import gluable_fraction
KEYS = ["11", "12", "21", "22"]
def windows(T):
    # copies are distinguished by context: A_x^y and B_y^x (Contextuality-by-Default); for lam* and lam_Q the shared variables are A_x, B_y
    return [Window((f"A{k[0]}", f"B{k[1]}"), np.asarray(T[k], float)) for k in KEYS]
DOM = {"A1": 2, "A2": 2, "B1": 2, "B2": 2}
def tables(T):
    P = np.zeros((2, 2, 2, 2))
    for k in KEYS: P[int(k[0]) - 1, int(k[1]) - 1] = np.asarray(T[k], float)
    return P
def quantities(T):
    W = windows(T); s, _ = signalling(W); c = cbd(W)["c"]
    kd = kd_cyclic([W[0], W[1], W[3], W[2]])            # cycle A1-B1-A2? order: (1,1),(1,2),(2,2),(2,1)
    lam = gluable_fraction(DOM, W).lam; lq = npa_fraction(tables(T), "1+AB")
    f = fingerprint(DOM, W)
    return {"s": {k: float(v) for k, v in s.items()}, "Delta": float(sum(s.values())), "linear_gap": f["linear_gap"], "c": float(c),
            "kd_contextual": bool(kd[0]), "kd_stat": [float(kd[1]), float(kd[2])], "lam_classical": float(lam), "lam_quantum": float(lq),
            "origin": "contextual beyond direct influence" if c > 1e-6 and f["linear_gap"] > 1e-6 else f["origin"]}
def bootstrap(T, n, B=2000, seed=8):
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        Tb = {k: (rng.multinomial(n[k], np.clip(np.asarray(T[k], float).ravel(), 0, None) / np.clip(np.asarray(T[k], float).sum(), 1e-12, None)) / n[k]).reshape(2, 2) for k in KEYS}
        q = quantities(Tb); out.append([q["Delta"], q["c"], q["lam_classical"], q["lam_quantum"], q["lam_quantum"] - q["lam_classical"]])
    a = np.array(out); return {nm: [float(np.percentile(a[:, i], 2.5)), float(np.percentile(a[:, i], 97.5))] for i, nm in enumerate(["Delta", "c", "lam_classical", "lam_quantum", "gap_q_minus_c"])}
def recode(T, var):
    T2 = {}
    for k in KEYS:
        M = np.asarray(T[k], float).copy()
        if var == "A" + k[0]: M = M[::-1, :]
        if var == "B" + k[1]: M = M[:, ::-1]
        T2[k] = M
    return T2
if __name__ == "__main__":
    D = json.load(open(os.path.join(HERE, "data_study8.json"))); res = {}
    for ds in ("D2", "D3"):
        T = D[ds]; sums = {k: float(np.sum(T[k])) for k in KEYS}
        assert all(abs(v - 1) <= 0.01 for v in sums.values()), f"{ds} tables do not sum to 1: {sums}"
        T = {k: (np.asarray(T[k], float) / np.sum(T[k])).tolist() for k in KEYS}
        q = quantities(T); ci = bootstrap(T, D.get("n_" + ds, D["n"]), B=int(os.environ.get("S8_B", 2000)))
        inv = all(abs(quantities(recode(T, v))["c"] - q["c"]) < 1e-6 and abs(quantities(recode(T, v))["lam_classical"] - q["lam_classical"]) < 1e-6
                  and abs(quantities(recode(T, v))["lam_quantum"] - q["lam_quantum"]) < 1e-5 for v in DOM)
        res[ds] = {"point": q, "ci95": ci, "recoding_invariant": bool(inv)}
    r2, r3 = res["D2"], res["D3"]
    res["predictions"] = {
        "P1_D2_fractions_zero": r2["point"]["lam_classical"] < 1e-6 and r2["point"]["lam_quantum"] < 1e-5,
        "P2_D2_direct_influence": r2["ci95"]["Delta"][0] > 0,
        "P3_D2_cbd_kd_agree_and_contextual": (r2["point"]["c"] > 1e-6) == r2["point"]["kd_contextual"] and r2["point"]["c"] > 1e-6,
        "P4_D3_contextual_beyond_DI": r3["ci95"]["c"][0] > 0,
        "P5_D3_quantum_exceeds_classical": r3["ci95"]["gap_q_minus_c"][0] > 0,
        "P6_recoding_invariant": r2["recoding_invariant"] and r3["recoding_invariant"]}
    json.dump(res, open(os.path.join(HERE, "results_study8.json"), "w"), indent=1)
    for k, v in res["predictions"].items(): print(k, v)
    for ds in ("D2", "D3"):
        p = res[ds]["point"]; c = res[ds]["ci95"]
        print(ds, f"Delta {p['Delta']:.3f} {np.round(c['Delta'], 3)}; c {p['c']:.3f} {np.round(c['c'], 3)}; KD {p['kd_contextual']}; lam* {p['lam_classical']:.3f}; lam_Q {p['lam_quantum']:.3f}; origin {p['origin']}")
