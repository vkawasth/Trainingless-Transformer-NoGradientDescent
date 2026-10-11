"""Test mimic_monitor.py end to end on synthetic MIMIC-IV-shaped data with planted answers (make_synthetic_mimic.py).
No credentialed data are needed or touched. Usage: python3 check_synthetic.py [--full]"""
import os, sys, json, subprocess, tempfile
H = os.path.dirname(os.path.abspath(__file__)); full = '--full' in sys.argv
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run([sys.executable, os.path.join(H, 'make_synthetic_mimic.py'), os.path.join(tmp, 'mimic'), '6000', '1'], check=True)
    subprocess.run([sys.executable, os.path.join(H, 'mimic_monitor.py'), os.path.join(tmp, 'mimic'), '--out', os.path.join(tmp, 'res')] + ([] if full else ['--quick']), check=True)
    r = json.load(open(os.path.join(tmp, 'res', 'results.json')))
ok = []
def check(name, cond): ok.append(bool(cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name}")
B, C, D, E = r['B_composition'], r['C_path_outcome'], r['D_contextuality_units'], r['E_selection']
check("cohort: first adult stays with ICU stay >= 2 days (about 85% of 6000)", 4900 <= r['cohort']['first_stays_adult_los_ge_2d'] <= 5300)
check("planted latent Markov chain: observed answers not glued (gap above the Markov null)", B['gap_bits'] > B['markov_null95'])
check("planted latent Markov chain: stable-attitude model rejected, latent Markov model fits", B['latent_models']['stable']['p'] < 0.01 and B['latent_models']['markov']['p'] > 0.05)
check("planted: death depends on the number of shock windows, so the three states add to the first (p < 0.05)", C['p_multiset_vs_x1'] < 0.05)
check("planted: care units differ only in margins, no contextuality across units (combined p > 0.05)", D['p'] is None or D['p'] > 0.05)
check("planted MAR selection of lactate: available-case tables incompatible beyond sampling", E['cf_available_case'] > E['null_95'])
check("inverse-probability weighting on (H1, creatinine) removes the incompatibility", E['cf_after_ipw'] <= E['null_95'])
RM = E['restricted_models']
check("restricted models: the planted creatinine-only (Y-only) selection is consistent; H1-only and external weights are rejected",
      RM['Y-only']['consistent'] and not RM['X-only']['consistent'] and not RM['external']['consistent'])
check("every reported proportion table has all cells >= 10 or is marked suppressed",
      all(('suppressed' in t) or t['n'] >= 10 for t in r['A_coupling_geometry']['tables'].values()))
print(f"{sum(ok)}/{len(ok)} synthetic checks pass"); sys.exit(0 if all(ok) else 1)
