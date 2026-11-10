"""Checks for 'Watching for coordinated deformation' (companion note; modules pce/stalkbank.py, pce/watchtoy.py; scenario runs saved by
the toy in watchtoy_*.json). IDENTITY = an exact instance of a statement proved in the text; FINDING = a computed fact about the toy.
  WC1  IDENTITY  report-only indistinguishability: the efficient coalition and the matched honest drift give identical values of every
                 report-derived signal at every step (only the independent inspection differs)
  WC2  IDENTITY  hidden movement that keeps the reports exact: moving the true law along a hidden direction inside the fibre keeps its
                 decision value inside the decision maker's range at every step
  WC3  IDENTITY  observability: an inspection whose map annihilates the coalition's movement (sigma = 0) never disagrees with the coalition
                 path; an inspection whose true table leaves the range allowed by the reports disagrees (sufficient condition)
  WC4  IDENTITY  minimal flipping distance and time: the least L1 change of {S2, S3}'s reports that brings B to the threshold coherently
                 is D = 0.683 (one LP); at per-step speed delta the straight path reaches the threshold in ceil(D / delta) steps, and
                 no path can do it faster
  WC5  IDENTITY  within-context alignment: a coalition that changes a context's table beyond the tolerance without moving every reporter of
                 that context identically triggers the ladder at its first change (one reporter not recruited, or one lagging); full
                 lockstep coverage does not
  WC6  FINDING   framing: majority, reputation and simplest-explanation rules are wrong in 70% of steps; the robust rule is never wrong
  WC7  FINDING   first-fire steps of the main signals in the eight scenarios, as reported in the note
"""
import os, sys, json, math
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import watchtoy as WT, bridgetoy as BT, tamper as T, miscal as MC, stalkbank as SB
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
R = {n: json.load(open(os.path.join(HERE, f"watchtoy_{n}.json"))) for n in WT.SCENARIOS}
V = {w: json.load(open(os.path.join(HERE, f"watchtoy_variant_{w}.json"))) for w in ("partial", "lagged")}
W = WT.setup(); P, gB = W["P"], W["gB"]; hon = MC.reports(P); anch = [hon["S1"], hon["S4"], hon["S5"]]

# WC1
report_sigs = [k for k in R["efficient"][0]["s"] if k not in ("anchor_residual",)]
diff = 0.0
for a, b in zip(R["efficient"][:15], R["matched"][:15]):                 # before the inspection arrives, every signal is report-only
    for k in report_sigs:
        x, y = a["s"][k], b["s"][k]
        if x is None or y is None or (isinstance(x, float) and math.isnan(x)): continue
        diff = max(diff, abs(x - y))
anc = max(abs(a["s"]["anchor_residual"] - b["s"]["anchor_residual"]) for a, b in zip(R["efficient"], R["matched"]))
check("WC1 IDENTITY identical reports give identical report-derived signals; only the inspection differs", diff < 1e-9 and anc > 0.1,
      f"largest difference over {len(report_sigs)} signals, steps 0-14 (before the inspection): {diff:.1e}; anchor residual difference {anc:.3f}")

# WC2
cells = [hon[k] for k in ("S1", "S2", "S4", "S5")]; I = T.interval(5, cells, gB); ok = True; moved = 0.0   # B not identified
M_, b_ = T.system(5, cells); vmin = linprog(gB, A_eq=M_, b_eq=b_, bounds=[(0, None)] * 32, method="highs").x
for s_ in np.linspace(0, 1, 11):                                        # the true law slides to the fibre's vertex of least B
    q = (1 - s_) * P + s_ * vmin; moved = max(moved, abs(gB @ q - gB @ P))
    ok &= q.min() >= -1e-12 and I[0] - 1e-9 <= gB @ q <= I[1] + 1e-9 and all(abs(BT.table(q, c[1])[k] - c[2][k]) < 1e-9 for c in cells for k in c[2])
check("WC2 IDENTITY truth moved inside the fibre keeps the reports and stays in the decision range", ok and moved > 0.4,
      f"reports S1, S2, S4, S5 unchanged while B slides from {gB @ P:.3f} by up to {moved:.3f}, always inside [{I[0]:.3f}, {I[1]:.3f}]")

# WC3
p1 = MC.coalition_endpoint(P, ["S2", "S3"], anch, gB, 0.35); path, _ = MC.coalition_path(P, ["S2", "S3"], anch, gB, 0.35)
out = {}
for ch in [(0, 2), (0, 3), (1, 3), (1, 4), (2, 4)]:
    M, _k = T.marg(5, ch); sigma = float(np.abs(M @ (p1 - P)).sum())
    res = max(T.residual(5, list(r.values()) + [("Q", ch, BT.table(P, ch))]) for r in path)
    rng = []
    for r in path:
        lo_hi = [T.interval(5, list(r.values()), M[i]) for i in range(M.shape[0])]
        outside = any(not (lh[0] - 1e-9 <= (M[i] @ P) <= lh[1] + 1e-9) for i, lh in enumerate(lo_hi)); rng.append(outside)
    out[ch] = (round(sigma, 3), round(res, 3), any(rng))
ok = all((v[0] < 1e-9) <= (v[1] < 1e-9) for v in out.values()) and all((not v[2]) or v[1] > 1e-6 for v in out.values())
check("WC3 IDENTITY sigma = 0 implies no disagreement; a true table outside the reports' range implies disagreement", ok,
      f"chord: (sigma, largest residual with the inspection, true table leaves the range at some step) {out}")

# WC4
pth = MC.coalition_endpoint(P, ["S2", "S3"], anch, gB, 0.5)
D = sum(sum(abs(BT.table(pth, BT.CTX[s])[k] - BT.table(P, BT.CTX[s])[k]) for k in BT.table(P, BT.CTX[s])) for s in ("S2", "S3"))
delta = D / 7.0; steps = None
for t in range(1, 20):
    lam = min(t * delta / D, 1.0); q = (1 - lam) * P + lam * pth
    if gB @ q <= 0.5 + 1e-9: steps = t; break
check("WC4 IDENTITY minimal flipping distance D and the straight path's time ceil(D / delta)", abs(D - 0.6827329393) < 1e-6 and steps == math.ceil(D / delta - 1e-9),
      f"D = {D:.4f}; delta = D/7; straight path reaches the threshold at step {steps}; no path with steps of size <= delta can arrive earlier, since it must travel at least D")

# WC5
first = lambda rows: next((r["t"] for r in rows if r["s"]["ladder_flag"] > 0), None)
check("WC5 IDENTITY imperfect coordination triggers the ladder at once; full lockstep does not",
      first(V["partial"]) == 1 and first(V["lagged"]) == 1 and first(R["efficient"]) is None,
      f"first ladder flag: partial {first(V['partial'])}, lagged {first(V['lagged'])}, efficient (lockstep, full coverage) {first(R['efficient'])}")

# WC6
wr = {rule: np.mean([(r["dec"][rule] == "accept") != (r["truth"] >= 0.5) and r["dec"][rule] != "investigate" for r in R["framing"][1:]])
      for rule in ("naive", "majority", "reputation", "occam", "robust")}
check("WC6 FINDING framing defeats majority, reputation and simplest explanation; the robust rule is never wrong",
      all(abs(wr[k] - 0.7) < 1e-9 for k in ("majority", "reputation", "occam")) and wr["robust"] == 0, f"{ {k: round(v, 2) for k, v in wr.items()} }")

# WC7
lines = open(os.path.join(HERE, "watchtoy_summary.txt")).read()
ok = ("within_max                    5          5" in lines and "hid_width                     -          -          5" in lines
      and "efficiency                    -          6          1" in lines and "naive wrong from              -          7          7         14" in lines)
check("WC7 FINDING the first-fire table of the note", ok, "watchtoy_summary.txt")
print(f"{sum(results)}/{len(results)} passed")
