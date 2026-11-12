"""Checks for Study 27b (specification study27b/SPEC_study27b.md, frozen before the run; results from run_study27b.py, run once).
Identities must hold; hypotheses are recorded with their pre-registered verdict (a failed hypothesis is a finding, not a failed check:
the check passes when the verdict is computed exactly as pre-registered and recorded).
  ZB1 IDENTITY    the specification is unchanged since the freeze (SHA-256 in FROZEN_SPEC_study27b.txt equals the file and the results)
  ZB2 IDENTITY    transformations: R1 unchanged under H_comm; R2 unchanged under H_comm, H_pad_safe, H_mixed and H_pad_op; activation
                  unchanged under H_comm and H_pad_safe and changed under every H_pad_op (Theorem 138, verified per history)
  ZB3 IDENTITY    exact witness toy: WB3 (timely anchor separates, Phi 0 vs 4, same source-source crossings), WB4 (lagging anchor: same
                  braid word as the relay), WB5 (no anchor: same source-only braid), at s = 0.0025, 0.005, 0.01
  ZB4 FINDING     WB7b: anchor placed on the far side of the move -> the relay makes no crossing (base order is part of the definition)
  ZB5-ZB9 RECORD  H1-H5: the pre-registered verdict, recomputed here from the stored statistics with the frozen rule, equals the verdict in
                  the results, and a failed prediction is recorded as FAILED in RESULTS_study27b.md (as XS4 does for Study 26). The
                  verdict itself (HELD or FAILED) is printed in the bracket; a failed prediction is a finding, not a failed check.
  H1 trace compression: CI of err(R1) - err(R0) on E2 inside (-0.02, 0.02). H2: E3 CI lower end > 0.02 and E2 CI inside the margin.
  H3: comm CI (dR0 - dR1) and pad_safe CI (dR0 - dR2) lower ends > 0.02, with the identities. H4: CI lower end > 0.02 with the identity.
  H5: balanced accuracy >= 0.90 in configuration B and < 0.90 in configuration A.
"""
import os, sys, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "study27b")
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
R = json.load(open(os.path.join(D, "results_study27b.json")))
h = hashlib.sha256(open(os.path.join(D, "SPEC_study27b.md"), "rb").read()).hexdigest()
check("ZB1 IDENTITY specification unchanged since the freeze", h == open(os.path.join(D, "FROZEN_SPEC_study27b.txt")).read().split()[0] == R["spec_sha256"], h[:12])
idn = R["representation"]["H3"]["identities"]
check("ZB2 IDENTITY transformations preserve exactly what Theorem 138 says", all(idn.values()), ", ".join(k for k, v in idn.items() if v))
WB = R["toy"]["WB"]
check("ZB3 IDENTITY exact witness toy: timely anchor separates; lagging anchor and no anchor do not", WB["WB3"] and WB["WB4"] and WB["WB5"],
      f"WB3 {WB['WB3']}, WB4 {WB['WB4']}, WB5 {WB['WB5']}")
check("ZB4 FINDING anchor on the far side: relay Phi", True, str(WB["WB7b_anchor_first_C2_Phi"]))
r = R["representation"]; H3 = r["H3"]["results"]; H4 = r["H4"]; H5 = R["witness"]["H5"]
res_md = open(os.path.join(D, "RESULTS_study27b.md")).read()
def m(lo, hi, e=0.02): return -e < lo and hi < e
ver = {"H1": m(*r["H1"]["ci"]),
       "H2": r["H2"]["E3_ci"][0] > 0.02 and m(*r["H2"]["E2_ci"]),
       "H3": all(idn[k] for k in ("R1_comm", "R2_comm", "R2_pad_safe", "R2_mixed")) and H3["comm_E2"]["R0_minus_R1_ci"][0] > 0.02 and H3["pad_safe_E2"]["R0_minus_R2_ci"][0] > 0.02,
       "H4": H4["ci"][0] > 0.02 and H4["identity"],
       "H5": H5["B"]["balanced_accuracy"] >= 0.90 and H5["A"]["balanced_accuracy"] < 0.90}
rec = {"H1": r["H1"]["verdict"], "H2": r["H2"]["verdict"], "H3": r["H3"]["verdict"], "H4": H4["verdict"], "H5": H5["verdict"]}
info = {"H1": f"err R1 - R0 = {r['H1']['diff']:+.4f}, CI {r['H1']['ci'][0]:+.4f}..{r['H1']['ci'][1]:+.4f}",
        "H2": f"E3: {r['H2']['E3_diff']:+.4f} CI {r['H2']['E3_ci'][0]:+.4f}..{r['H2']['E3_ci'][1]:+.4f}; E2 CI {r['H2']['E2_ci'][0]:+.4f}..{r['H2']['E2_ci'][1]:+.4f}",
        "H3": f"comm CI {H3['comm_E2']['R0_minus_R1_ci'][0]:+.4f}..{H3['comm_E2']['R0_minus_R1_ci'][1]:+.4f}; pad_safe CI {H3['pad_safe_E2']['R0_minus_R2_ci'][0]:+.4f}..{H3['pad_safe_E2']['R0_minus_R2_ci'][1]:+.4f}",
        "H4": f"E3 err R1 {H4['err_R1']:.4f}, R2 {H4['err_R2']:.4f}, CI {H4['ci'][0]:+.4f}..{H4['ci'][1]:+.4f}",
        "H5": f"balanced accuracy A {H5['A']['balanced_accuracy']:.3f}, B {H5['B']['balanced_accuracy']:.3f}"}
for n_, h_ in zip(range(5, 10), ("H1", "H2", "H3", "H4", "H5")):
    ok = ver[h_] == rec[h_] and (ver[h_] or f"{h_} FAILED" in res_md)
    check(f"ZB{n_} RECORD {h_} verdict recomputed and recorded", ok, f"{'HELD' if ver[h_] else 'FAILED'}: {info[h_]}")
print(f"{sum(results)}/{len(results)} passed")
