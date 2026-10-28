"""Study 13: a trusted source that is sometimes adversarial. Run: python3 run_study13.py [--smoke | --predict]
Writes results_study13.json (registered run) or smoke_study13.json. See PREREG_study13.md."""
import json, os, sys, time
import numpy as np
from numpy.polynomial import polynomial as Pn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import design13 as D
from pce import sources as SR

RULES = ("naive", "glue", "drop_C_triple", "write_off_C"); OUTS = ("right", "open", "wrong", "flag")

def predictions(d):
    gm = d["gm"]; out = {}
    for adv in D.ADVERSARIES:
        for n in D.NB:
            e, sh = D.shifts(adv, n, gm["sm"], d["s"], d["a2_share"])
            out[f"{adv}|{n}"] = SR.predict(gm, n, D.K_C, sh, D.C_B, D.C_S, z=D.Z, draws=200000, seed=7, rho=D.RHO)
    return out

def simulate(d, reps, rng):
    gm = d["gm"]; mu, v = SR.components(SR.SD.exact_tables(d["P"]))[:2]; pany = SR.SD.U_poly(mu, v, lambda x: float(sum(x) > 0), range(3))
    res = {}
    for adv in D.ADVERSARIES:
        for n in D.NB:
            e, sh = D.shifts(adv, n, gm["sm"], d["s"], d["a2_share"]); eps = [0.0] * (2 * D.K_C) + e; t0 = time.time()
            cnt = {r: {o: 0 for o in OUTS} for r in RULES}; rh = {r: [] for r in RULES}; zC = []; srcdrop = {"G1": 0, "G2": 0, "C": 0}; advdrop = 0; advrem = 0; advtot = 0
            for _ in range(reps):
                bats = [SR.sample_batch(d["P"], d["Q"], eps[k], n, rng, rho=D.RHO) for k in range(D.K)]
                out, diag = SR.run_rules(bats, D.SRC, n, d["acts"], d["u0"], d["a0"], D.C_B, D.C_S, bad="C", z=D.Z, rho=D.RHO)
                for r in RULES: cnt[r][out[r][0]] += 1; rh[r].append(out[r][1])
                zC.append(diag["z_source"].get("C", np.nan))
                if diag["source_dropped"]: srcdrop[diag["source_dropped"]] += 1
                adv_idx = [2 * D.K_C + k for k in range(D.K_C) if e[k] > 0]
                advdrop += int((~diag["kb"][adv_idx]).sum()); advrem += int((~diag["ks"][adv_idx]).sum()); advtot += len(adv_idx)
            q = lambda a: [float(np.nanpercentile(a, p)) for p in (2.5, 25, 50, 75, 97.5)]
            res[f"{adv}|{n}"] = {"eps_b": e, "rates": {r: {o: cnt[r][o] / reps for o in OUTS} for r in RULES},
                                 "rhat_q": {r: q(rh[r]) for r in RULES}, "harm_q": {r: q(Pn.polyval(np.array(rh[r]), pany)) for r in RULES},
                                 "zC_q": q(zC), "source_dropped": {k: v_ / reps for k, v_ in srcdrop.items()},
                                 "adv_batches_dropped_batch_level": (advdrop / advtot) if advtot else None,
                                 "adv_batches_removed": (advrem / advtot) if advtot else None}
            print(adv, n, {r: res[f"{adv}|{n}"]["rates"][r]["right"] for r in RULES}, f"{time.time() - t0:.0f}s", flush=True)
    return res

def n_half(curve):
    """N (total triple records) where the right-rate crosses 1/2, by log-linear interpolation over the N grid; None if it never does"""
    Ns = [D.K * n for n in D.NB]; ys = [curve[n] for n in D.NB]
    for i in range(len(Ns) - 1):
        if ys[i] < 0.5 <= ys[i + 1]:
            t = (0.5 - ys[i]) / (ys[i + 1] - ys[i]); return float(np.exp(np.log(Ns[i]) + t * (np.log(Ns[i + 1]) - np.log(Ns[i]))))
    return float(Ns[0]) if ys[0] >= 0.5 else None

def evaluate(pred, sim):
    E = {}; g1 = []
    for key in sim:
        for r in RULES: g1.append(abs(sim[key]["rates"][r]["right"] - pred[key][r]["right"]))
    E["H1"] = bool(np.mean(np.array(g1) <= 0.05) >= 0.90 and max(g1) <= 0.08)
    E["H1_detail"] = {"share_within_0.05": float(np.mean(np.array(g1) <= 0.05)), "max_gap": float(max(g1)), "cells": len(g1)}
    glue_wrong = max(sim[k]["rates"]["glue"]["wrong"] for k in sim); nb = D.NB[-1]
    naive_A2 = sim[f"A2|{nb}"]["rates"]["naive"]["wrong"]
    rec = [abs(sim[f"A2|{n}"]["rates"]["glue"]["right"] - sim[f"A2|{n}"]["rates"]["drop_C_triple"]["right"]) for n in D.NB if D.K * n >= 48000]
    E["H2"] = bool(glue_wrong <= 0.05 and naive_A2 >= 0.75 and max(rec) <= 0.05)
    E["H2_detail"] = {"glue_wrong_max": glue_wrong, "naive_wrong_A2_at_largest_N": naive_A2, "glue_vs_oracle_under_A2_right_gap_N>=48k": max(rec)}
    h3 = {}
    for adv in ("A0", "A1", "A1b"):
        emp = n_half({n: sim[f"{adv}|{n}"]["rates"]["glue"]["right"] for n in D.NB}); prd = n_half({n: pred[f"{adv}|{n}"]["glue"]["right"] for n in D.NB})
        h3[adv] = {"emp": emp, "pred": prd, "ratio": (emp / prd) if (emp and prd) else None}
    dom = all(sim[f"A1b|{n}"]["rates"]["drop_C_triple"]["right"] >= sim[f"A1b|{n}"]["rates"]["glue"]["right"] for n in D.NB if D.K * n >= 24000)
    E["H3"] = bool(all(h3[a]["ratio"] is not None and abs(h3[a]["ratio"] - 1) <= 0.15 for a in h3) and dom)
    E["H3_detail"] = {"n_half": h3, "oracle_beats_glue_under_A1b": dom}
    a3 = [sim[f"A3|{n}"]["adv_batches_removed"] for n in D.NB]; a3src = [sim[f"A3|{n}"]["source_dropped"]["C"] for n in D.NB]
    a3gap = [abs(sim[f"A3|{n}"]["rates"]["glue"]["right"] - sim[f"A0|{n}"]["rates"]["glue"]["right"]) for n in D.NB]
    a0drop = [sum(sim[f"A0|{n}"]["source_dropped"].values()) for n in D.NB]
    a0gap = [abs(sim[f"A0|{n}"]["rates"]["glue"]["right"] - sim[f"A0|{n}"]["rates"]["naive"]["right"]) for n in D.NB]
    E["H4"] = bool(min(a3) >= 0.90 and max(a3src) <= 0.10 and max(a3gap) <= 0.05 and max(a0drop) <= 0.08 and max(a0gap) <= 0.04)
    E["H4_detail"] = {"A3_adv_batch_removed": a3, "A3_C_dropped": a3src, "A3_vs_A0_glue_right_gap": a3gap, "A0_any_source_dropped": a0drop,
                      "A0_glue_vs_naive_right_gap": a0gap}
    worst = {r: min(sim[f"{a}|{nb}"]["rates"][r]["right"] for a in D.ADVERSARIES) for r in RULES}
    E["H5"] = bool(worst["glue"] >= 0.85 and worst["naive"] <= 0.05)
    E["H5_detail"] = {"worst_case_right_at_largest_N": worst}
    return E

def main():
    smoke = "--smoke" in sys.argv; only = "--predict" in sys.argv; name = "I" if smoke else "W"; d = D.setup(name)
    pred = predictions(d); OUT = {"truth": name, "design": {"K_C": D.K_C, "rho": D.RHO, "c_b": D.C_B, "c_s": D.C_S, "z": D.Z, "NB": D.NB, "s": d["s"],
           "eps_star": float(d["eps_star"]), "eps_det": float(d["eps_det"]), "a2_share": float(d["a2_share"]), "a_risk20": D.A_RISK20, "B_risk20": float(SR.band(D.A_RISK20, 1 / 3)),
           "B_risk50": float(SR.band(D.C_S, 1 / 3)), "gm": {k: (float(v) if np.ndim(v) == 0 else [float(x) for x in v]) for k, v in d["gm"].items()}},
           "pred": pred}
    if only:
        print(json.dumps({k: {r: pred[k][r]["right"] for r in RULES} for k in pred}, indent=0)); return
    reps = 600 if smoke else 2000; rng = np.random.default_rng(99 if smoke else 1313)
    OUT["reps"] = reps; OUT["sim"] = simulate(d, reps, rng); OUT["evaluation"] = evaluate(pred, OUT["sim"])
    fn = "smoke_study13.json" if smoke else "results_study13.json"
    json.dump(OUT, open(os.path.join(HERE, fn), "w"), indent=1, default=float); print(json.dumps(OUT["evaluation"], indent=1, default=float))

if __name__ == "__main__":
    main()
