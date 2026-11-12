"""Checks for Study 27c (study27c/SPEC_study27c.md, frozen before the run; results of run_study27c.py, run once).
  ZC1 IDENTITY  the specification is unchanged since the freeze (hash in FROZEN_SPEC_study27c.txt equals the file and the results)
  ZC2 IDENTITY  calibration followed the frozen rules: z is the smallest grid value with honest rate <= 0.05 (recorded), and each value-of-
                information step switched a component off exactly when its calibration loss did not rise
  ZC3-ZC8 RECORD  P1-P6: the verdict recomputed from the stored test rates with the frozen rule equals the recorded verdict; the bracket
                says HELD or FAILED (a failed prediction is a finding, not a failed check)
"""
import os, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "study27c")
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
R = json.load(open(os.path.join(D, "results_study27c.json")))
h = hashlib.sha256(open(os.path.join(D, "SPEC_study27c.md"), "rb").read()).hexdigest()
check("ZC1 IDENTITY specification unchanged since the freeze", h == open(os.path.join(D, "FROZEN_SPEC_study27c.txt")).read().split()[0] == R["spec_sha256"], h[:12])
ok = True; info = []
for c, v in R["calibration"].items():
    ok &= 2.0 <= v["z"] <= 4.0 and (v["honest_rate_at_z"] <= 0.05)
    for s in v["voi_steps"]: ok &= s["switched_off"] == (s["loss_off"] <= s["loss_on"])
    info.append(f"{c}: z = {v['z']}, kept {[k for k, x in v['components'].items() if x]}")
check("ZC2 IDENTITY calibration and value-of-information steps follow the frozen rules", ok, "; ".join(info))
T = R["test"]
rule = {"P1": all(T[c]["rates"]["honest"] <= 0.065 for c in T), "P2": all(T[c]["rates"]["shock"] <= 0.10 for c in T),
        "P3": T["B"]["balanced_accuracy"] >= 0.90, "P4": T["B"]["balanced_accuracy"] >= T["A"]["balanced_accuracy"] - 0.01,
        "P5": all(T[c]["rates"]["cascade_lagging"] >= 0.90 for c in T), "P6": all(T[c]["guard_activations"] == 0 for c in T)}
txt = {"P1": f"honest false-relay A {T['A']['rates']['honest']:.3f}, B {T['B']['rates']['honest']:.3f} (<= 0.065)",
       "P2": f"shock A {T['A']['rates']['shock']:.3f}, B {T['B']['rates']['shock']:.3f} (<= 0.10)",
       "P3": f"balanced accuracy B {T['B']['balanced_accuracy']:.3f} (>= 0.90)",
       "P4": f"B {T['B']['balanced_accuracy']:.3f} vs A {T['A']['balanced_accuracy']:.3f}",
       "P5": f"lagging cascade relay-like A {T['A']['rates']['cascade_lagging']:.3f}, B {T['B']['rates']['cascade_lagging']:.3f}",
       "P6": f"guard activations A {T['A']['guard_activations']}, B {T['B']['guard_activations']}"}
for n, p in zip(range(3, 9), ("P1", "P2", "P3", "P4", "P5", "P6")):
    check(f"ZC{n} RECORD {p} verdict recomputed equals the recorded verdict", rule[p] == R["predictions"][p], f"{'HELD' if rule[p] else 'FAILED'}: {txt[p]}")
print(f"{sum(results)}/{len(results)} passed")
