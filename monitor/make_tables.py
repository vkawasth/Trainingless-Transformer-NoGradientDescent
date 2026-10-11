"""LaTeX tables for the validation paper, generated from the saved results (no hand transcription)."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(os.path.join(H, 'llm_results.json'))); G = json.load(open(os.path.join(H, 'gss_results.json')))
C = json.load(open(os.path.join(H, 'clinical_results.json'))); B = json.load(open(os.path.join(H, 'cbd_boot.json'))); N = json.load(open(os.path.join(H, 'usmle_prompts.json')))
NAMES = {'USMLE': 'USMLE (medical)', 'OPINIONQA': 'OpinionQA', 'SENSITIVEQ': 'sensitive questions', 'PRIME_FULL': 'primes',
         'COUNTAHAPPYNUMBER': 'happy numbers', 'ARC': 'ARC', 'LEETCODE_EASY': 'LeetCode (easy)', 'HOTPOTQA': 'HotpotQA (agent)'}
def pct(x): return f"{100 * x:.1f}"
rows = []
for t, r in L.items():
    c3, c4 = r['churn']['35'], r['churn']['4']; rg = r.get('regime', {}); cb = r['cbd']
    rows.append(f"{NAMES[t]} & {r['n_items']} & {pct(c3['acc_M'])}$\\to${pct(c3['acc_J'])} & {pct(c4['acc_M'])}$\\to${pct(c4['acc_J'])} & "
                f"{rg.get('kind', '--')} & {rg.get('stress', 0):.1f} & {cb['s_odd']:.2f}/{cb['Delta']:.2f} & "
                f"{pct(c3['lost'])}/{pct(c3['masked'])} & {pct(c4['lost'])}/{pct(c4['masked'])}\\\\")
open(os.path.join(H, 'tab_llm.tex'), 'w').write("\n".join(rows) + "\n\\bottomrule\n")
# GSS: wording effects, contextuality, regimes
rows = []
for name in ('welfare-cities', 'welfare-race', 'health-education'):
    g = G[name]; import numpy as np
    dA = np.median([v['direct_A'] for v in g['cbd'].values()]); dB = np.median([v['direct_B'] for v in g['cbd'].values()])
    flags = sum(v['contextual'] for v in g['cbd'].values()); fsp = sum(v['split_contextual'] for v in g['cbd'].values()); fct = sum(v['cuts_contextual'] for v in g['cbd'].values())
    bmax = max(B[name]['post_contextual'].values()); bcut = max(B[name]['post_contextual_cuts'].values())
    kX = {k: sum(1 for s in g['steps']['X'] if s['kind'] == k) for k in ('bend', 'tear?', 'tear', 'break')}
    kY = {k: sum(1 for s in g['steps']['Y'] if s['kind'] == k) for k in ('bend', 'tear?', 'tear', 'break')}
    rows.append(f"{name} & {pct(dA)} & {pct(dB)} & {flags} / {fsp} / {fct} & {bmax:.2f} / {bcut:.2f} & "
                f"{kX['bend']}/{kX['tear?']}/{kX['tear']}/{kX['break']} & {kY['bend']}/{kY['tear?']}/{kY['tear']}/{kY['break']}\\\\")
open(os.path.join(H, 'tab_gss.tex'), 'w').write("\n".join(rows) + "\n\\bottomrule\n")
rows = []
for p, r in G['panels'].items():
    for it in ('natfare', 'natfarey'):
        if it not in r: continue
        q = r[it]; lab = 'welfare' if it == 'natfare' else 'the poor'
        rows.append(f"20{p}--{int(p) + 4 + 2000} & {lab} & {q['n']} & {q['agree12']:.3f} & {q['max12']:.3f} & {q['excess12']:.3f} & {q['gap']:.3f} [{q['gap_ci'][0]:.3f}, {q['gap_ci'][1]:.3f}] & {q['gap_null95']:.3f}\\\\")
open(os.path.join(H, 'tab_panels.tex'), 'w').write("\n".join(rows) + "\n\\bottomrule\n")
M = json.load(open(os.path.join(H, 'panel_models.json'))); rows = []
pv = lambda x: f"{x:.3f}" if x >= 0.001 else "$<0.001$"
for key, r in M.items():
    p, it = key.split('/'); lab = 'welfare' if it == 'natfare' else 'the poor'
    rows.append(f"20{p}--{int(p) + 2004} & {lab} & {r['n']} & {r['gap_observed']:.3f} & " + " & ".join(f"{r[m]['G2']:.1f} ({pv(r[m]['p'])}) & {r[m]['gap_implied']:.3f}" for m in ('stable', 'drifting', 'markov')) + "\\\\")
open(os.path.join(H, 'tab_panelmodels.tex'), 'w').write("\n".join(rows) + "\n\\bottomrule\n")
rows = []
for fam, lab in (('35', 'GPT-3.5'), ('4', 'GPT-4')):
    for v, vl in (('M', 'March'), ('J', 'June')):
        q = N[fam + v]; rows.append(f"{lab} & {vl} & {pct(q['acc_intact'])} & {pct(q['acc_damaged'])} & {pct(q['lost'])} & {pct(q['gained'])} & {q['excess']:.3f}\\\\")
    dc = N[fam + ' damage cost June - March']
    rows.append(f"\\multicolumn{{7}}{{r}}{{\\emph{{{lab}: change of the damage cost, June minus March: {100 * dc['est']:+.1f} points, 95\\% interval [{100 * dc['ci'][0]:+.1f}, {100 * dc['ci'][1]:+.1f}]}}}}\\\\")
open(os.path.join(H, 'tab_usmle.tex'), 'w').write("\n".join(rows) + "\n\\bottomrule\n")
for t in ('tab_llm', 'tab_usmle', 'tab_gss', 'tab_panels', 'tab_panelmodels'): print(open(os.path.join(H, t + '.tex')).read())
