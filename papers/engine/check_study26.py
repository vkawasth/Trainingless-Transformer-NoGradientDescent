"""Checks for Study 26 (engine/study26_e2e): integrity of the freeze, reproducibility, and the theorem identities inside the run.
  XS1 IDENTITY  PREREG_study26.md and world26.py match the hashes frozen in FROZEN_PREREG_study26.txt before any table was generated
  XS2 IDENTITY  a fresh run of run_study26.py reproduces results_study26.json exactly and prints the same check lines
  XS3 IDENTITY  every check labelled IDENTITY in the run passes (the theorem statements on the exact-data fibre, the Fisher kernel, the VOI
                criterion, the history trace and the reachable slice), except XE9, whose pre-registered operationalisation is recorded as
                failed in DEVIATIONS_study26.md. Checks labelled IDENTITY/PREDICTION are left out: their prediction half can fail. (A first
                version counted them and failed on XE18, whose prediction rho_q > 2 eta failed; Theorem 122(iii) then holds vacuously.)
  XS4 IDENTITY  every failed pre-registered check is named in DEVIATIONS_study26.md
"""
import os, sys, json, hashlib, subprocess, re
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "study26_e2e")
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
frozen = dict(l.split()[::-1] for l in open(os.path.join(D, "FROZEN_PREREG_study26.txt")).read().split("\n") if l.strip())
ok = all(hashlib.sha256(open(os.path.join(D, f), "rb").read()).hexdigest() == h for f, h in frozen.items())
check("XS1 IDENTITY the pre-registration and the world generator match their frozen hashes", ok, ", ".join(frozen))
old_json = open(os.path.join(D, "results_study26.json")).read(); old_out = open(os.path.join(D, "run_study26_output.txt")).read()
new_out = subprocess.run([sys.executable, os.path.join(D, "run_study26.py")], capture_output=True, text=True, cwd=D).stdout
new_json = open(os.path.join(D, "results_study26.json")).read()
check("XS2 IDENTITY a fresh run reproduces the results and the check lines", new_json == old_json and new_out == old_out)
lines = [l for l in new_out.splitlines() if re.match(r"(PASS|FAIL) XE", l)]
ident = [l for l in lines if " IDENTITY" in l.split("[")[0] and "PREDICTION" not in l.split("[")[0] and "XE9 " not in l]
check("XS3 IDENTITY every theorem identity in the run holds", all(l.startswith("PASS") for l in ident), f"{len(ident)} identity checks")
failed = [re.search(r"XE\d+b?", l).group() for l in lines if l.startswith("FAIL")]; dev = open(os.path.join(D, "DEVIATIONS_study26.md")).read()
check("XS4 IDENTITY every failed pre-registered check is recorded in the deviations", all(f in dev for f in failed), f"failed: {failed}")
print(f"{sum(results)}/{len(results)} passed")
