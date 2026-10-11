"""Run the three adapters on synthetic data with planted answers (make_synth.py) and check that the planted answers are recovered."""
import os, sys, json, subprocess, tempfile
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ok = []
def check(n, c): ok.append(bool(c)); print(f"[{'PASS' if c else 'FAIL'}] {n}")
with tempfile.TemporaryDirectory() as t:
    run = lambda *a: subprocess.run([sys.executable, *a, '--quick'], check=True, capture_output=True)
    subprocess.run([sys.executable, os.path.join(H, 'synth', 'make_synth.py'), t], check=True, capture_output=True)
    run(os.path.join(H, 'challenge2012.py'), 'a', '--zip', f'{t}/set-a.zip', '--outcomes', f'{t}/Outcomes-a.txt', '--out', f'{t}/a.json')
    run(os.path.join(H, 'diabetes130.py'), '--zip', f'{t}/diabetes130.zip', '--out', f'{t}/d.json')
    run(os.path.join(H, 'heart.py'), '--zip', f'{t}/heartdisease.zip', '--out', f'{t}/h.json')
    A, Dd, Hh = (json.load(open(f'{t}/{x}.json')) for x in 'adh')
for name, R, dkey in (('2012', A, 'D_contextuality_icu'), ('diabetes', Dd, 'D_contextuality_source')):
    check(f"{name}: planted latent Markov chain: answers not glued, latent Markov fits, stable rejected",
          R['B_composition']['decision'] == {'answers_not_glued': True, 'latent_markov_fits': True} and R['B_composition']['latent_models']['stable']['p'] < 0.01)
    check(f"{name}: planted MAR selection removed by weighting", R['E_selection']['decision'] == 'selection explains the incompatibility')
    check(f"{name}: planted direct influence only: not contextual", not R[dkey]['contextual'])
check("2012: death depends on the shock count: multiset adds to the first state", A['C_path_outcome']['p_multiset_vs_x1'] < 0.05)
check("heart: planted margins-only site differences: no pair contextual after Bonferroni", not any(v['bonferroni_6'] for v in Hh['D_contextuality_sites'].values()))
print(f"{sum(ok)}/{len(ok)} synthetic checks pass"); sys.exit(0 if all(ok) else 1)
