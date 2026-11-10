"""Eight scenarios on the bridge with redundant reporters of B's context, for comparing every signal the decision maker can compute
(companion note, 'Watching for coordinated deformation'). New module.
World: the bridge law P (pce/bridgetoy), decision B = P(X3 = X4) with threshold 1/2, reported by S3 and by two redundant reporters R1, R2.
At t = 0 every report is honest. Steps t = 1..20. An inspection of (X3, X5) of the true world arrives at t = 15 (in the contextual world,
which has no joint law, the inspection re-measures the context (X3, X4)).
  1 redundant_liar   R2 reports agreement 0.35 (true marginals) from t = 5
  2 framing          as 1, plus planted reports: F1 on (X3, X5) and F2 on (X2, X4) from the coalition's false law (t = 5, 6), and
                     F3, F4 duplicating R2's false report on (X3, X4) (t = 6, 7), so that the honest S3 and R1 become the minority
  3 efficient        S2, S3, R1, R2 move together on the straight path to the least-change false law (B = 0.35), reached at t = 10
  4 stealthy         the same reporters move to the maximum-entropy false law (B = 0.35), reached at t = 20
  5 contextual       an honest contextual world: pair correlations (0.8, 0.8, 0.8, 0.8, -0.8) on the cycle, uniform margins, constant
  6 drift            honest drift: the coupling on (X3, X4) moves linearly to the value giving B = 0.35, reached at t = 20
  7 shock            the same honest change, all at once at t = 8
  8 matched          honest drift of the world along the efficient coalition's path (scenario 3); every report identical to 3
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from . import tamper as T, bridgetoy as BT, miscal as MC, stalkbank as SB

STEPS = 20; TAU = 0.5; TOL = 0.02
BASE = ["S1", "S2", "S3", "S4", "S5", "R1", "R2"]
CTXS = dict(BT.CTX, R1=(2, 3), R2=(2, 3))
SCENARIOS = ["redundant_liar", "framing", "efficient", "stealthy", "contextual", "drift", "shock", "matched"]


def setup():
    J0 = np.full(5, 0.7); field = np.array([0.15, 0, 0, 0, 0]); P = MC.ising(J0, field); _, gA, gB = BT.world()
    hon = MC.reports(P); anch = [hon["S1"], hon["S4"], hon["S5"]]
    p1 = MC.coalition_endpoint(P, ["S2", "S3"], anch, gB, 0.35)
    z = np.full(32, 1 / 32); groups = []
    for c in anch:
        M, keys = T.marg(5, c[1])
        for i, k in enumerate(keys): groups.append((M[i] > 0, c[2][k]))
    groups += [(gB > 0.5, 0.35), (gB < 0.5, 0.65)]
    for _ in range(5000):
        for m, v in groups: z[m] *= v / z[m].sum()
    j1 = brentq(lambda j: gB @ MC.ising(np.r_[0.7, 0.7, j, 0.7, 0.7], field) - 0.35, -3, 0.7)
    return dict(P=P, gB=gB, p1=p1, p2=z, J0=J0, J1=np.r_[0.7, 0.7, j1, 0.7, 0.7], field=field)


def contextual_tables():
    out = {}
    for i, e in enumerate([0.8, 0.8, 0.8, 0.8, -0.8]):
        a = (1 + e) / 4; d = (1 - e) / 4; out[f"S{i+1}"] = {(1, 1): a, (-1, -1): a, (1, -1): d, (-1, 1): d}
    return out


def history(name, W):
    """for each t: (current report cells, truth law or None for the contextual world, truth value of B)"""
    P, gB = W["P"], W["gB"]; out = []
    ctab = contextual_tables()
    for t in range(STEPS + 1):
        truth = P; rep = {}
        if name == "contextual":
            for s in BT.LABELS: rep[s] = ctab[s]
            rep["R1"] = rep["R2"] = ctab["S3"]; tv = ctab["S3"][(1, 1)] + ctab["S3"][(-1, -1)]
        else:
            if name == "drift": truth = MC.ising(W["J0"] + min(t / 20, 1) * (W["J1"] - W["J0"]), W["field"])
            if name == "shock": truth = MC.ising(W["J1"] if t >= 8 else W["J0"], W["field"])
            if name == "matched": lam = min(t / 10, 1); truth = (1 - lam) * P + lam * W["p1"]
            for s in BASE: rep[s] = BT.table(truth, CTXS[s])
            if name in ("efficient", "stealthy"):
                lam = min(t / 10, 1) if name == "efficient" else min(t / 20, 1); tgt = W["p1"] if name == "efficient" else W["p2"]
                pl = (1 - lam) * P + lam * tgt
                for s in ("S2", "S3", "R1", "R2"): rep[s] = BT.table(pl, CTXS[s])
            if name in ("redundant_liar", "framing") and t >= 5: rep["R2"] = BT.pair_table(P, (2, 3), 0.35)
            if name == "framing":
                if t >= 5: rep["F1"] = BT.table(W["p1"], (2, 4))
                if t >= 6: rep["F2"] = BT.table(W["p1"], (1, 3))
                if t >= 6: rep["F3"] = BT.pair_table(P, (2, 3), 0.35)
                if t >= 7: rep["F4"] = BT.pair_table(P, (2, 3), 0.35)
            tv = gB @ truth
        cells = [(s, CTXS.get(s, {"F1": (2, 4), "F2": (1, 3), "F3": (2, 3), "F4": (2, 3)}.get(s)), rep[s]) for s in rep]
        if t >= 15:
            qt = truth if truth is not None else P
            if name == "contextual": cells.append(("Q", (2, 3), ctab["S3"]))   # a contextual world has no joint law: re-measure a reported context
            else: cells.append(("Q", (2, 4), BT.table(qt, (2, 4))))
        out.append((cells, tv))
    return out


def run(name, W):
    H = history(name, W); gB = W["gB"]; rows = []; hist = []; last = {}
    prev_vec = None; prev_disp = None; path_len = 0.0; naive0 = None
    for t, (cells, tv) in enumerate(H):
        for c in cells:
            key = c[0]
            if key not in last or any(abs(last[key][k] - c[2][k]) > 1e-12 for k in c[2]): hist.append((f"{key}@{t}", c[1], c[2])); last[key] = c[2]
        s, dec = SB.stalks(cells, hist, gB, TAU, TOL)
        vec = np.concatenate([[c[2][k] for k in sorted(c[2])] for c in sorted(cells, key=lambda c: c[0]) if c[0] in BASE])
        disp = None if prev_vec is None else vec - prev_vec
        s["velocity"] = 0.0 if disp is None else float(np.abs(disp).sum())
        s["acceleration"] = abs(s["velocity"] - (rows[-1]["s"]["velocity"] if rows else 0.0))
        if disp is not None and prev_disp is not None and np.linalg.norm(disp) > 1e-12 and np.linalg.norm(prev_disp) > 1e-12:
            s["curvature"] = float(np.arccos(np.clip(disp @ prev_disp / np.linalg.norm(disp) / np.linalg.norm(prev_disp), -1, 1)))
        else: s["curvature"] = 0.0
        path_len += s["velocity"]; naive0 = s["naive_value"] if naive0 is None else naive0
        s["efficiency"] = abs(s["naive_value"] - naive0) / path_len if path_len > 1e-12 else 0.0
        s["synchrony"] = 0.0 if disp is None else float(sum(np.abs(disp[4 * i:4 * i + 4]).sum() > 1e-9 for i in range(len(BASE))))
        views = {}
        for x in BASE:
            I, _ = SB.tol_interval([c for c in cells if c[0] != x], gB); views[x] = 0.5 * (I[0] + I[1])
        rows.append(dict(t=t, truth=tv, s=s, dec=dec, views=views, disp=None if disp is None else disp.tolist()))
        if disp is not None and np.linalg.norm(disp) > 1e-12: prev_disp = disp
        prev_vec = vec
    order_prev = None; crossings = []
    for r in rows:                                                      # decorated braid on the source-exclusion strands, tolerance band 0.01
        o = sorted(BASE, key=lambda x: r["views"][x] + 1e-6 * BASE.index(x))
        if order_prev is not None:
            cr = 0
            for a in BASE:
                for b in BASE:
                    if BASE.index(a) < BASE.index(b):
                        before = order_prev.index(a) < order_prev.index(b); after = o.index(a) < o.index(b)
                        if before != after and abs(r["views"][a] - r["views"][b]) > 0.01: cr += 1
            crossings.append(cr)
        else: crossings.append(0)
        order_prev = o
    for i, r in enumerate(rows):
        r["s"]["crossings"] = float(crossings[i]); r["s"]["crossing_burst"] = float(sum(crossings[max(0, i - 2):i + 1]))
    return rows
