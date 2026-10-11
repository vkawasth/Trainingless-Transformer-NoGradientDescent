"""check_monitor.py -- reproducibility and claim checks for the validation paper (validation.tex).

1. reruns the unit tests of monitor.py;
2. reruns the fast analyses (LLM drift, USMLE prompts, NHANES) and checks they reproduce the saved results exactly;
3. checks the saved results of the slow analyses (GSS 1985-2024 and panels, CbD bootstrap) for internal consistency;
4. checks Prop. 'tears need stress' (every point-estimate wall crossing has stress >= 1) and Prop. 'excess churn' (binary excess churn
   equals twice the masked damage) on every reported change;
5. pins every number quoted in the text;
6. regenerates the tables and checks that the tables in the paper are the generated ones.
Run from this folder: python3 check_monitor.py  (about one minute; the GSS and bootstrap scripts are not rerun, see README)."""
import json, os, subprocess, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); os.chdir(H); sys.path.insert(0, H)
from monitor import bands, regime, State, walls, wall_values
sys.path.insert(0, os.path.join(H, '..', 'bridges', 'fukaya'))

results = []
def check(name, cond, detail=""):
    results.append(bool(cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
def run(script):
    r = subprocess.run([sys.executable, script], capture_output=True, text=True); return r.returncode, r.stdout + r.stderr
def load(f): return json.load(open(f))
def close(a, b, tol): return abs(a - b) <= tol

# 1. unit tests --------------------------------------------------------------------------------------------------
rc, out = run('monitor.py'); check("monitor.py unit tests", rc == 0 and "ALL UNIT TESTS PASS" in out, out.strip().splitlines()[-1])

# 2. fast analyses reproduce the saved results -------------------------------------------------------------------
for script, outs in (('llm_analysis.py', ['llm_results.json']), ('usmle_prompts.py', ['usmle_prompts.json']), ('clinical_analysis.py', ['clinical_results.json'])):
    saved = {f: load(f) for f in outs}; rc, out = run(script)
    check(f"{script} reruns and reproduces {', '.join(outs)}", rc == 0 and all(load(f) == saved[f] for f in outs), "" if rc == 0 else out[-300:])

L, G, C, B, N = load('llm_results.json'), load('gss_results.json'), load('clinical_results.json'), load('cbd_boot.json'), load('usmle_prompts.json')

# 3. slow analyses: internal consistency -------------------------------------------------------------------------
pairs = ('welfare-cities', 'welfare-race', 'health-education')
check("GSS: 24 survey years 1985-2024 with both wordings for every pair", all(len(G[p]['cbd']) == 24 for p in pairs))
check("GSS: every saved year state is consistent (margins sum to 1, tear margin recomputes)",
      all(close(sum(s['f']), 1, 2e-3) and close(sum(s['g']), 1, 2e-3) and close(min(abs(wall_values(s['f'], s['g']))), s['mT'], 2e-3)
          for p in pairs for c in 'XY' for s in G[p][c].values()))
check("GSS: rank-2 contextual years are a subset of the canonical (split) contextual years",
      all(not v['contextual'] or v['split_contextual'] for p in pairs for v in G[p]['cbd'].values()))
check("CbD posterior: flagged years are exactly the canonical point-estimate contextual years",
      all(sorted(B[p]['flagged']) == sorted(int(y) for y, v in G[p]['cbd'].items() if v['split_contextual']) for p in pairs))
check("panels: excess churn = max - observed agreement; gap above every Markov resample (p = 0)",
      all(close(r[it]['max12'] - r[it]['agree12'], r[it]['excess12'], 2e-3) and r[it]['gap_p_markov'] == 0
          for r in G['panels'].values() for it in ('natfare', 'natfarey')))
# spot-check one saved year with the split LP
from monitor import cbd_split
import pandas as pd
d = pd.read_csv(os.path.join('data', 'gss', 'gss_extract.csv'), low_memory=False)
def wt(sub, a, b):
    m = sub[a].isin([1, 2, 3]) & sub[b].isin([1, 2, 3]) & sub.wtssps.notna(); s = sub[m]; w = s.wtssps.values
    T = np.zeros((3, 3)); np.add.at(T, (s[a].astype(int).values - 1, s[b].astype(int).values - 1), w); return T, w.sum() ** 2 / (w ** 2).sum()
sub = d[d.year == 1988]; sp = cbd_split(wt(sub, 'natheal', 'nateduc')[0], wt(sub, 'nathealy', 'nateducy')[0])
check("GSS health-education 1988: canonical deficit recomputes", sp['contextual'] and close(sp['deficit'], G['health-education']['cbd']['1988']['split_deficit'], 1e-4), f"{sp['deficit']:.5f}")

# 4. the two propositions on every reported change ---------------------------------------------------------------
steps = [(t, r['regime']) for t, r in L.items() if 'regime' in r] + \
        [(f"{p}/{c} {s['frm']}-{s['to']}", s) for p in pairs for c in 'XY' for s in G[p]['steps'][c]] + \
        [(f"NHANES {k}", dict(kind=v['regime'], stress=v['stress'])) for k, v in C.items() if isinstance(v, dict) and isinstance(v.get('regime'), str)] + \
        [(f"USMLE prompt {v}", N['system ' + v]) for v in 'MJ']
tears = [(n, s) for n, s in steps if s['kind'] in ('tear', 'tear?')]
check(f"Prop (tears need stress): all {len(tears)} tears and possible tears have stress >= 1", all(s['stress'] >= 1 for _, s in tears),
      f"min stress {min(s['stress'] for _, s in tears):.2f}")
rng = np.random.default_rng(0); viol = 0; crossings = 0
for _ in range(3000):
    n, m = rng.integers(2, 5), rng.integers(2, 5)
    A = State(rng.dirichlet(np.ones(n * m)).reshape(n, m), 1000); Bs = State(rng.dirichlet(np.ones(n * m)).reshape(n, m), 1000)
    if np.any(np.sign(wall_values(A.f, A.g)) * np.sign(wall_values(Bs.f, Bs.g)) < 0):
        crossings += 1; viol += regime(A, Bs)['stress'] < 1 - 1e-12
check("Prop (tears need stress) on 3000 random changes", viol == 0, f"{crossings} crossings, {viol} violations")
check("Prop (excess churn): binary excess churn = 2 x masked in every task and family",
      all(close(v['excess_binary'], 2 * v['masked'], 2e-3) for r in L.values() for v in r['churn'].values()))
check("Prop (excess churn) on the prompt change: excess = 2 min(lost, gained)",
      all(close(N[m]['excess'], 2 * min(N[m]['lost'], N[m]['gained']), 2e-3) for m in ('35M', '35J', '4M', '4J')))
check("excess churn >= 0 everywhere", all(v['excess'] >= -1e-9 for r in L.values() for v in r['churn'].values()))

# 5. numbers quoted in the text ----------------------------------------------------------------------------------
P = lambda x: round(100 * x, 1)
c = L['PRIME_FULL']['churn']
check("primes: GPT-3.5 52.9->76.0, GPT-4 84.0->51.1, GPT-3.5 lost 17.3 (a third of its March correct)",
      (P(c['35']['acc_M']), P(c['35']['acc_J']), P(c['4']['acc_M']), P(c['4']['acc_J']), P(c['35']['lost'])) == (52.9, 76.0, 84.0, 51.1, 17.3)
      and round(c['35']['lost'] / c['35']['acc_M'], 2) == 0.33)
check("sensitive questions: GPT-3.5 2->8, GPT-4 21->5, possible tear with stress 1.2",
      tuple(P(L['SENSITIVEQ']['churn'][f][k]) for f in ('35', '4') for k in ('acc_M', 'acc_J')) == (2.0, 8.0, 21.0, 5.0)
      and L['SENSITIVEQ']['regime']['kind'] == 'tear?' and round(L['SENSITIVEQ']['regime']['stress'], 1) == 1.2)
kinds = {t: r['regime']['kind'] for t, r in L.items()}
check("tears: primes, happy numbers, HotpotQA; possible tears: USMLE, sensitive; bends: OpinionQA, ARC, LeetCode",
      {t for t, k in kinds.items() if k == 'tear'} == {'PRIME_FULL', 'COUNTAHAPPYNUMBER', 'HOTPOTQA'}
      and {t for t, k in kinds.items() if k == 'tear?'} == {'USMLE', 'SENSITIVEQ'} and {t for t, k in kinds.items() if k == 'bend'} == {'OPINIONQA', 'ARC', 'LEETCODE_EASY'})
trade = {t for t, r in L.items() if np.sign(r['churn']['35']['acc_M'] - r['churn']['4']['acc_M']) * np.sign(r['churn']['35']['acc_J'] - r['churn']['4']['acc_J']) < 0}
check("GPT-3.5 and GPT-4 trade places exactly on the three tears and the sensitive questions", trade == {'PRIME_FULL', 'COUNTAHAPPYNUMBER', 'HOTPOTQA', 'SENSITIVEQ'}, str(sorted(trade)))
check("primes and happy numbers cross the wall f_correct = g_correct", all([[0], [0]] in L[t]['regime']['walls_crossed'] for t in ('PRIME_FULL', 'COUNTAHAPPYNUMBER')))
mu = L['USMLE']['margins']
check("USMLE possible tear is on the wall of the invalid-answer shares, both below 1%, stress 19",
      L['USMLE']['regime']['walls_maybe'] == [[[0, 1], [0, 1]]] and all(mu[m][2] < 0.01 for m in mu) and L['USMLE']['regime']['stress'] == 19.0)
o = L['OPINIONQA']
check("OpinionQA: GPT-3.5 answers all in both months, GPT-4 97.9 -> 24.7, bend, s = Delta",
      (P(o['churn']['35']['acc_M']), P(o['churn']['35']['acc_J']), P(o['churn']['4']['acc_M']), P(o['churn']['4']['acc_J'])) == (100.0, 100.0, 97.9, 24.7)
      and o['regime']['kind'] == 'bend' and close(o['cbd']['s_odd'], o['cbd']['Delta'], 1e-4))
gaps = {t: r['cbd']['Delta'] - r['cbd']['s_odd'] for t, r in L.items() if t != 'OPINIONQA'}
check("no task contextual (rank 2 and canonical); Delta - s from 0.08 (ARC) to 1.1 (primes, happy numbers)",
      not any(r['cbd']['contextual'] or r['cbd']['lp_contextual'] or r['cbd_split']['contextual'] for r in L.values()) and min(gaps, key=gaps.get) == 'ARC'
      and round(min(gaps.values()), 2) == 0.08 and round(gaps['PRIME_FULL'], 1) == 1.1 and round(gaps['COUNTAHAPPYNUMBER'], 1) == 1.1)
u = L['USMLE']['churn']
check("USMLE intact: GPT-4 88.2->84.7, lost 4.7 (5.3% of correct), masked 1.2; GPT-3.5 63.5->65.0, gained 9.1, lost 7.6 all masked",
      (P(u['4']['acc_M']), P(u['4']['acc_J']), P(u['4']['lost']), P(u['4']['masked']), P(u['35']['acc_M']), P(u['35']['acc_J']), P(u['35']['gained']), P(u['35']['lost']), P(u['35']['masked']))
      == (88.2, 84.7, 4.7, 1.2, 63.5, 65.0, 9.1, 7.6, 7.6) and round(100 * u['4']['lost'] / u['4']['acc_M'], 1) == 5.3)
check("pooling intact and damaged prompts reproduces Chen et al. GPT-4 USMLE 86.6 -> 82.1",
      round(50 * (N['4M']['acc_intact'] + N['4M']['acc_damaged']), 1) == 86.6 and round(50 * (N['4J']['acc_intact'] + N['4J']['acc_damaged']) + 1e-9, 1) == 82.1)
cost = {m: round(100 * (N[m]['acc_intact'] - N[m]['acc_damaged']), 1) for m in ('35M', '35J', '4M', '4J')}
check("prompt damage costs 1.1-5.3 points (GPT-4 3.2 -> 5.3, GPT-3.5 1.1 -> 4.4) and flips 10-17% of questions; intervals include 0",
      cost == {'35M': 1.1, '35J': 4.4, '4M': 3.2, '4J': 5.3} and (round(100 * min(N[m]['disagree'] for m in cost)), round(100 * max(N[m]['disagree'] for m in cost))) == (10, 17)
      and all(N[f + ' damage cost June - March']['ci'][0] < 0 < N[f + ' damage cost June - March']['ci'][1] for f in ('35', '4')))
m4 = L['COUNTAHAPPYNUMBER']['margins']; lo, hi = bands(m4['4J'], L['COUNTAHAPPYNUMBER']['n_items'])
check("happy numbers: GPT-4 'no valid answer' 9.4% -> 0, June upper band 1.1%, no break anywhere",
      round(100 * m4['4M'][2], 1) == 9.4 and m4['4J'][2] == 0 and round(100 * hi[2], 1) == 1.1
      and not any(b for r in L.values() for b in r['breaks'].values()) and not any(s['kind'] == 'break' for _, s in steps))
med = lambda p, k: np.median([v[k] for v in G[p]['cbd'].values()])
check("GSS wording effect: welfare 44-45 points, cities 27 points (medians)",
      (round(100 * med('welfare-cities', 'direct_A')), round(100 * med('welfare-race', 'direct_A')), round(100 * med('welfare-cities', 'direct_B'))) == (44, 45, 27))
fl = {p: (sum(v['contextual'] for v in G[p]['cbd'].values()), sum(v['split_contextual'] for v in G[p]['cbd'].values())) for p in pairs}
post = {p: {int(y): v for y, v in B[p]['post_contextual'].items()} for p in pairs}
he = [post['health-education'][y] for y in B['health-education']['flagged']]
check("GSS contextual years (rank 2 / canonical): 0/0, 0/2, 3/10; posteriors max 0.06, 0.80, 0.97; health-education 0.37-0.97 in flagged years; only 1988 > 0.95; none > 1 - 0.05/72",
      fl == {'welfare-cities': (0, 0), 'welfare-race': (0, 2), 'health-education': (3, 10)}
      and [round(max(post[p].values()), 2) for p in pairs] == [0.06, 0.80, 0.97] and (round(min(he), 2), round(max(he), 2)) == (0.37, 0.97)
      and [(p, y) for p in pairs for y, v in post[p].items() if v > 0.95] == [('health-education', 1988)] and post['health-education'][1988] == 0.968
      and max(v for p in pairs for v in post[p].values()) < 1 - 0.05 / 72)
gtears = [(n, s) for n, s in steps if '/' in n and s['kind'] == 'tear']
check("GSS: exactly one decided tear, welfare-race wording Y 2014-2016, wall f(too little) = g(about right, too much)",
      len(gtears) == 1 and gtears[0][0] == 'welfare-race/Y 2014-2016' and gtears[0][1]['crossed'] == [[[0], [1, 2]]])
y14, y16 = G['welfare-race']['Y']['2014'], G['welfare-race']['Y']['2016']
check("GSS tear: 62.5% + 26.9% = 0.89 in 2014, 1.13 in 2016, forcing >= 13% 'too little' to both",
      (round(100 * y14['f'][0], 1), round(100 * y14['g'][0], 1), round(y14['f'][0] + y14['g'][0], 2), round(y16['f'][0] + y16['g'][0], 2), round(100 * (y16['f'][0] + y16['g'][0] - 1)))
      == (62.5, 26.9, 0.89, 1.13, 13))
nst = sum(len(G[p]['steps'][c]) for p in pairs for c in 'XY')
A14 = State(*wt(d[d.year == 2014], 'natfarey', 'natracey'), alpha=0.05 / nst); A16 = State(*wt(d[d.year == 2016], 'natfarey', 'natracey'), alpha=0.05 / nst)
check(f"GSS tear is only a possible tear under Bonferroni over the {nst} steps", nst == 138 and regime(A14, A16)['kind'] == 'tear?')
top3 = sum(any(s['frm'] == 2018 and s['to'] == 2021 for s in sorted(G[p]['steps'][c], key=lambda s: -s['change'])[:3]) for p in pairs for c in 'XY')
top3py = sum(any(s['frm'] == 2018 for s in sorted(G[p]['steps'][c], key=lambda s: -s['change'] / (s['to'] - s['frm']))[:3]) for p in pairs for c in 'XY')
check("GSS 2018-2021 is among the three largest changes in 3 of 6 series, and in none per year", top3 == 3 and top3py == 0, f"{top3}, {top3py}")
pn = [G['panels'][k][it] for k in ('06', '08', '10') for it in ('natfare', 'natfarey')]
check("panels: max agreement 94-96%, observed 62-74%, excess a fifth to a third, change 26-38%",
      (round(100 * min(q['max12'] for q in pn)), round(100 * max(q['max12'] for q in pn)), round(100 * min(q['agree12'] for q in pn)), round(100 * max(q['agree12'] for q in pn)),
       round(100 * (1 - max(q['agree12'] for q in pn))), round(100 * (1 - min(q['agree12'] for q in pn)))) == (94, 96, 62, 74, 26, 38)
      and 0.2 <= min(q['excess12'] for q in pn) and max(q['excess12'] for q in pn) <= 0.34)
check("panels: gluing gap 0.055-0.082 bits; Markov null 95th percentile 0.024-0.030",
      (round(min(q['gap'] for q in pn), 3), round(max(q['gap'] for q in pn), 3), round(min(q['gap_null95'] for q in pn), 3), round(max(q['gap_null95'] for q in pn), 3))
      == (0.055, 0.082, 0.024, 0.030))
M = load('panel_models.json')
check("panel models: observed gap in panel_models.json equals the gluing gap of gss_analysis",
      all(close(M[f'{k}/{it}']['gap_observed'], G['panels'][k][it]['gap'], 1e-4) for k in ('06', '08', '10') for it in ('natfare', 'natfarey')))
st = {k: v['stable'] for k, v in M.items()}
check("stable model: rejected in 2006 and 2008 (p <= 0.02, bootstrap <= 0.04), not in 2010 (p 0.10, 0.46); implied gaps 0.067-0.105",
      all(st[f'{k}/{it}']['p'] <= 0.021 and st[f'{k}/{it}']['p_boot'] <= 0.04 for k in ('06', '08') for it in ('natfare', 'natfarey'))
      and (round(st['10/natfare']['p'], 2), round(st['10/natfarey']['p'], 2)) == (0.10, 0.46)
      and (round(min(v['gap_implied'] for v in st.values()), 3), round(max(v['gap_implied'] for v in st.values()), 3)) == (0.067, 0.105))
check("drifting and latent Markov models fit all six series (p >= 0.15 chi-square, bootstrap >= 0.05), df 6, interior fits",
      all(M[k][m]['p'] >= 0.145 and M[k][m]['p_boot'] >= 0.05 and M[k][m]['df'] == 6 and M[k][m]['min_class'] > 0.01 for k in M for m in ('drifting', 'markov')))
cut = {p: sum(v['cuts_contextual'] for v in G[p]['cbd'].values()) for p in pairs}
check("ordinal cuts: welfare-race flags vanish, health-education keeps 8 of 10, posterior max 0.93; cut flags subset of split flags",
      cut == {'welfare-cities': 0, 'welfare-race': 0, 'health-education': 8} and round(max(B['health-education']['post_contextual_cuts'].values()), 2) == 0.93
      and all(not v['cuts_contextual'] or v['split_contextual'] for p in pairs for v in G[p]['cbd'].values()))
ob = {it: [G['panels'][k]['outcome'][it]['obama_share_by_wave1_attitude'] for k in ('06', '08', '10')] for it in ('natfare', 'natfarey')}
rng_ = lambda it, c: (min(o[c] for o in ob[it]), max(o[c] for o in ob[it]))
check("outcome: welfare 80.6-81.4 vs 38.1-47.4; assistance to the poor 68.6-75.0 vs 21.3-31.9",
      (rng_('natfare', '1'), rng_('natfare', '3'), rng_('natfarey', '1'), rng_('natfarey', '3')) == ((0.806, 0.814), (0.381, 0.474), (0.686, 0.75), (0.213, 0.319)))
a, o = C['adults 20+'], C['age 55+']
check("NHANES: diabetes 10.2 -> 11.1%; bend stress 0.08 (adults), 0.90 (55+); P(diabetes|obese) 18% and 31% in both cycles; not contextual in either test",
      (P(a['diabetes']['2009_10']), P(a['diabetes']['2011_12'])) == (10.2, 11.1) and (a['regime'], a['stress'], o['regime'], o['stress']) == ('bend', 0.08, 'bend', 0.9)
      and all(round(100 * v) == 18 for v in a['diabetes_risk_if_obese'].values()) and all(round(100 * v) == 31 for v in o['diabetes_risk_if_obese'].values())
      and not any(x[k] for x in (a, o) for k in ('cbd_contextual', 'split_contextual')))
rc, out = run('monitor.py')
check("unit test: 219 of 400 random 3x3 systems are contextual only in their splits", "219 of 400" in out)

# 5b. site separation, toric geometry, panels' paths (saved outputs of site_separation.py, ctx_allpairs.py, boundary_tests.py,
#     toric_public.py, panel_paths.py; figures and their numbers from make_figures.py) --------------------------------------
SS, CXa, BT, TP, PPa = load('site_separation.json'), load('ctx_allpairs.json'), load('boundary_tests.json'), load('toric_public.json'), load('panel_paths.json')
L3s = SS['L3']; ncf = [r['naive_cf'] for r in L3s]
d85 = d[d.year >= 1985]
sh = lambda v: [np.average(g[v] == 1, weights=g.wtssps) for _, g in d85[d85[v].isin([1, 2, 3])].groupby('year')]
check("GSS welfare wordings 1985-2024: 55,169 respondents (27,047 X, 28,122 Y, none both); 'too little' 13.4-33.4% (welfare), 54.6-73.5% (the poor)",
      (int((d85.natfare.isin([1, 2, 3]) | d85.natfarey.isin([1, 2, 3])).sum()), int(d85.natfare.isin([1, 2, 3]).sum()), int(d85.natfarey.isin([1, 2, 3]).sum()),
       int((d85.natfare.isin([1, 2, 3]) & d85.natfarey.isin([1, 2, 3])).sum())) == (55169, 27047, 28122, 0)
      and [round(100 * x, 1) for x in (min(sh('natfare')), max(sh('natfare')), min(sh('natfarey')), max(sh('natfarey')))] == [13.4, 33.4, 54.6, 73.5])
check("L3: 1320 system-years; naive contextual fraction = TV, positive everywhere, 0.022-0.576, median 0.119",
      len(L3s) == 1320 and min(ncf) > 0 and (round(min(ncf), 3), round(max(ncf), 3), round(float(np.median(ncf)), 3)) == (0.022, 0.576, 0.119))
check("L3: canonical flags 523, cuts 307, rank 2 93; 25 posteriors above 0.95",
      (sum(r['contextual'] for r in L3s), sum(r['cuts_contextual'] for r in L3s), sum(r['rank2'] for r in L3s), SS['L3_screen']['over95']) == (523, 307, 93, 25))
pp = sorted((v['deff1']['p'], k) for k, v in CXa.items() if not k.startswith('_'))
check("L3: boundary null over 55 pairs: smallest p 0.0073 (crime x drugs, flagged 20 of 24 years), next 0.42; design effect 2: 0.15; none passes 0.05/55",
      len(pp) == 55 and pp[0][1] == 'natcrime|natdrug' and round(pp[0][0], 4) == 0.0073 and round(pp[1][0], 2) == 0.42
      and round(CXa['natcrime|natdrug']['deff2']['p'], 2) == 0.15 and CXa['natcrime|natdrug']['years_flagged'] == 20
      and CXa['_summary']['deff1']['surviving'] == [] and CXa['_summary']['deff2']['surviving'] == [] and BT['flags_surviving_bonferroni'] == [])
# the naive contextual fraction equals TV(T_X, T_Y): check the LP definition on one system-year
from scipy.optimize import linprog
sub = d85[d85.year == 2024]
def wt3(a, b):
    m = sub[a].isin([1, 2, 3]) & sub[b].isin([1, 2, 3]) & sub.wtssps.notna(); s_ = sub[m]; T_ = np.zeros((3, 3))
    np.add.at(T_, (s_[a].astype(int).values - 1, s_[b].astype(int).values - 1), s_.wtssps.values); return T_ / T_.sum()
TX_, TY_ = wt3('natfare', 'natcity'), wt3('natfarey', 'natcityy')
r_ = linprog(np.r_[np.zeros(9), -1], A_ub=np.c_[np.eye(9), np.zeros(9)], b_ub=np.minimum(TX_, TY_).ravel(), A_eq=[np.r_[np.ones(9), -1]], b_eq=[0], bounds=[(0, None)] * 10, method='highs')
check("naive contextual fraction = 1 - max lambda (LP) = TV(T_X, T_Y), welfare x cities 2024", close(1 + r_.fun, 0.5 * np.abs(TX_ - TY_).sum(), 1e-9)
      and close(1 + r_.fun, [x['naive_cf'] for x in L3s if x['year'] == 2024 and x['a'] == 'natfare' and x['b'] == 'natcity'][0], 1e-4))
dp = dict(zip(SS['items'], SS['L1_pooled']['d']))
check("L1: connection non-flat in all 24 years (chi2_10 >= 333); welfare -1.96, cities +1.29, crime 0.51, race 0.34; space, defense, environment, health within 0.07",
      SS['L1_pooled']['years_flat_rejected'] == 24 and min(SS['L1'][y]['flat_Q'] for y in SS['L1']) >= 333
      and [round(dp[i], 2) for i in ('natfare', 'natcity', 'natcrime', 'natrace')] == [-1.96, 1.29, 0.51, 0.34] and all(abs(dp[i]) <= 0.07 for i in ('natspac', 'natarms', 'natenvir', 'natheal')))
mis = [r['mi'] for r in SS['L2']]
from scipy.stats import chi2 as _chi2
check("L2: 2640 tables, MI median 0.024 bits, max 0.23; independence rejected at Bonferroni in 1140",
      len(mis) == 2640 and round(float(np.median(mis)), 3) == 0.024 and round(max(mis), 2) == 0.23 and sum(_chi2.sf(r['G'], 4) < 0.05 / 2640 for r in SS['L2']) == 1140)
dis = SS['dissociation']; DC = SS['dissociation_counts']
check("separation: Bonferroni |z| > 4.12 with bootstrap covariance; loop non-flat 493, dependent 733, both 340 (261 without a contextual point estimate); Spearman 0.85, 0.24, 0.14",
      round(SS['z_bonferroni'], 2) == 4.12 and (DC['L1'], DC['L2'], DC['L1_and_L2'], DC['L1_and_L2_not_flagged']) == (493, 733, 340, 261)
      and (round(SS['rank_corr']['naiveCF_absPhi'], 2), round(SS['rank_corr']['naiveCF_MI'], 2), round(SS['rank_corr']['absPhi_MI'], 2)) == (0.85, 0.24, 0.14))
st = [r['steep'] for r in SS['L2']]; stm = [r['steep_min'] for r in SS['L2']]; top = max(SS['L2'], key=lambda r: r['steep'])
check("entropy brane: smallest Fisher eigenvalue positive (4.9-12.2); steepness (largest) 10-614, median 50, maximal for education x health 1990 (X), min f_i g_j 0.0008",
      min(stm) > 0 and (round(min(stm), 1), round(max(stm), 1)) == (4.9, 12.2) and (round(min(st)), round(max(st)), round(float(np.median(st)))) == (10, 614, 50)
      and (top['year'], top['a'], top['b'], top['w'], round(top['min_fg'], 4)) == (1990, 'nateduc', 'natheal', 'X', 0.0008))
# toric
from fractions import Fraction as _F
def fr_(x): a_, b_ = (x.split('/') + ['1'])[:2]; return _F(int(a_), int(b_))
two = [TP['NHANES 2009_10'], TP['NHANES 2011_12']] + list(TP['health x education (X)'].values()) + list(TP['poor x cities (Y)'].values())
def fac_(r): f1 = fr_(r['f1']); g = [fr_(x) for x in r['g']]; f = (f1, 1 - f1); return sorted(f'{i+1}{j+1}' for i in range(2) for j in range(3) if f[i] + g[j] < 1)
check("toric: facets recompute from the saved margins (Prop fano23) for all 50 systems; vertices = facets = number of branes",
      len(two) == 50 and all(fac_(r) == sorted(r['facets']) and len(r['branes']) == r['n_vertices'] == len(r['facets']) for r in two))
from collections import Counter as _C
he, pc = TP['health x education (X)'], TP['poor x cities (Y)']
check("toric: NHANES CP^2 both cycles, decided, 3 branes at leximin = f1/3; health x education 15/6/3, decided 2010-16 and 2021; poor x cities 18/4/2 (Bl3 1994, 1996), never decided",
      all(TP[k]['surface'] == 'CP^2' and TP[k]['chamber_decided'] and TP[k]['branes_at_leximin'] == 3 and close(TP[k]['leximin'][0], float(fr_(TP[k]['f1'])) / 3, 1e-9) for k in ('NHANES 2009_10', 'NHANES 2011_12'))
      and _C(r['surface'] for r in he.values()) == _C({'CP^1 x CP^1': 15, 'Bl_2 CP^2': 6, 'F_1 = Bl_1 CP^2': 3}) and sorted(y for y, r in he.items() if r['chamber_decided']) == ['2010', '2012', '2014', '2016', '2021']
      and _C(r['surface'] for r in pc.values()) == _C({'F_1 = Bl_1 CP^2': 18, 'Bl_2 CP^2': 4, 'Bl_3 CP^2 (dP6)': 2}) and sorted(y for y, r in pc.items() if r['surface'].startswith('Bl_3')) == ['1994', '1996']
      and not any(r['chamber_decided'] for r in pc.values()))
check("toric: product carries no brane in all 50; leximin carries at least half in all 50; 4 every year for health x education (4 of 5 in the 6 Bl2 years); 3 of 4 in 12 years and 3 of 6 in one for poor x cities",
      all(r['branes_at_product'] == 0 for r in two) and all(2 * r['branes_at_leximin'] >= r['n_vertices'] for r in two) and all(r['branes_at_leximin'] == 4 for r in he.values())
      and _C((r['n_vertices'], r['branes_at_leximin']) for r in he.values()) == _C({(4, 4): 18, (5, 4): 6})
      and _C((r['n_vertices'], r['branes_at_leximin']) for r in pc.values())[(4, 3)] == 12 and _C((r['n_vertices'], r['branes_at_leximin']) for r in pc.values())[(6, 3)] == 1)
check("NHANES 2009-10 observed a11 = 0.012 against 0.032 under independence", (round(TP['NHANES 2009_10']['observed'][0], 3), round(TP['NHANES 2009_10']['product'][0], 3)) == (0.012, 0.032))
check("3x3 polytopes: 9/12/15/18 vertices in 54/1026/1070/490 of 2640 tables; welfare x race (Y) 15 -> 12 at the 2014 -> 2016 tear",
      TP['vertices33'] == {'9': 54, '12': 1026, '15': 1070, '18': 490} and (TP['vertices33_series']['2014|natfare|natrace|Y'], TP['vertices33_series']['2016|natfare|natrace|Y']) == (15, 12))
from nef33 import vertices as _v33
def _t33(yr):
    s_ = d85[(d85.year == yr) & d85.natfarey.isin([1, 2, 3]) & d85.natracey.isin([1, 2, 3]) & d85.wtssps.notna()]; T_ = np.zeros((3, 3))
    np.add.at(T_, (s_.natfarey.astype(int).values - 1, s_.natracey.astype(int).values - 1), s_.wtssps.values); return T_ / T_.sum()
A_, B_ = _t33(2014), _t33(2016); VA, VB = _v33(A_.sum(1), A_.sum(0)), _v33(B_.sum(1), B_.sum(0))
FA, FB = set().union(*VA.values()), set().union(*VB.values())
check("tear 2014 -> 2016 is one blow-down: exactly one wall changes side; the facet a11 = 0 (4 vertices) disappears; 15 -> 12 vertices",
      int((np.sign(wall_values(A_.sum(1), A_.sum(0))) != np.sign(wall_values(B_.sum(1), B_.sum(0)))).sum()) == 1 and FA - FB == {0} and FB <= FA
      and sum(1 for z in VA.values() if 0 in z) == 4 and (len(VA), len(VB)) == (15, 12))
check("3x3 vertex count recomputes for welfare x race (Y) 2016", len(_v33(*(lambda T_: (T_.sum(1), T_.sum(0)))((lambda s_: (lambda T_: T_ / T_.sum())(np.histogram2d(s_.natfarey - 1, s_.natracey - 1, bins=[3, 3], range=[[-.5, 2.5]] * 2, weights=s_.wtssps)[0]))(d85[(d85.year == 2016) & d85.natfarey.isin([1, 2, 3]) & d85.natracey.isin([1, 2, 3]) & d85.wtssps.notna()])))) == 12)
FN = load('figure_numbers.json')['fisher_bregman']
check("Fisher-Bregman: KL(p||q) = KL(p||p*) + KL(p*||q): 0.437 = 0.024 + 0.413 bits (94% in the shift of 'too little')",
      close(FN['KLpq'], FN['KLpps'] + FN['KLpsq'], 1e-12) and (round(FN['KLpq'], 3), round(FN['KLpps'], 3), round(FN['KLpsq'], 3)) == (0.437, 0.024, 0.413) and round(100 * FN['KLpsq'] / FN['KLpq']) == 94)
tvr = [x for k, v in PPa.items() if '/' in k for x in v['tv_rows']]
pv_ = [PPa[k][o]['p_path_vs_multiset'] for k in PPa if '/' in k for o in ('vote', 'party')]
check("panels: TV(K13, K12K23) rows 0.04-0.19; order: pooled p 0.16 (vote), 0.54 (party), multiset vs x1 p < 0.001; bits gained 0.098/0.115 (vote), 0.049/0.079 (party); 2 of 12 series p < 0.05",
      (round(min(tvr), 2), round(max(tvr), 2)) == (0.04, 0.19) and (round(PPa['pooled_vote']['p_path_vs_multiset'], 2), round(PPa['pooled_party']['p_path_vs_multiset'], 2)) == (0.16, 0.54)
      and PPa['pooled_vote']['p_multiset_vs_x1'] < 0.001 and PPa['pooled_party']['p_multiset_vs_x1'] < 0.001
      and [round(PPa['pooled_' + o]['bits_series'] - PPa['pooled_' + o][m], 3) for o in ('vote', 'party') for m in ('bits_path', 'bits_multiset')] == [0.098, 0.115, 0.049, 0.079]
      and sum(p < 0.05 for p in pv_) == 2 and len(pv_) == 12)
check("figures present", all(os.path.exists(os.path.join('figs', f)) for f in ('fig_toric.pdf', 'fig_toric_years.pdf', 'fig_entropy_brane.pdf', 'fig_fisher_bregman.pdf', 'fig_tearing.pdf', 'fig_cbd.pdf', 'fig_nerve.pdf', 'fig_panels.pdf')))

# 5c. positive control, pre-registered clinical analyses and six-wave panels (ext/) -------------------------------------
E = lambda *p: load(os.path.join('ext', *p))
SQ = E('snowqueen', 'results_snowqueen.json'); cr, al = SQ['correct_responses'], SQ['all_responses']
sys.path.insert(0, os.path.join(H, 'ext', 'snowqueen')); import importlib; sqm = importlib.import_module('snowqueen')
check("Snow Queen: statistic 0.452 [0.226, 0.668] (correct), 0.276 [0.012, 0.494] (all); contextual by criterion and LP; recomputes",
      (cr['LHS'], cr['bootstrap_CI_99_99'], round(al['LHS'], 3), al['bootstrap_CI_99_99'], round(cr['Delta'], 3), round(al['s_odd'], 3), round(al['Delta'], 2)) == (0.452, [0.226, 0.668], 0.276, [0.012, 0.494], 1.548, 3.346, 1.07)
      and all(x['contextual_criterion'] and x['contextual_LP'] for x in (cr, al)) and abs(sqm.lhs(sqm.cyc(sqm.T2)) - 0.452) < 5e-4 and abs(sqm.lhs(sqm.cyc(sqm.T3)) - 0.2764) < 5e-4)
from hashlib import sha256
def frozen_ok(folder, listing):
    lines = open(os.path.join('ext', folder, 'results', listing)).read().split('\n')
    return all(sha256(open(os.path.join('ext', folder, l.split()[1]), 'rb').read()).hexdigest() == l.split()[0] for l in lines if 'frozen/' in l)
check("frozen results match their recorded SHA-256 hashes (open clinical, C19PRC)", frozen_ok('open_clinical', 'FROZEN.txt') and frozen_ok('c19', 'FROZEN_UK.txt'))
A, Bb, CC = E('open_clinical', 'results', 'frozen', 'results_2012_a.json'), E('open_clinical', 'results', 'results_2012_b.json'), E('open_clinical', 'results', 'results_2012_compare.json')
m = lambda R: R['B_composition']['latent_models']
check("PhysioNet 2012: gaps 0.0165/0.0039 (A), 0.0175/0.0042 (B); latent Markov p 0.089 (A) fits, 0.008 (B) rejected; negative control bend 0.83",
      (A['B_composition']['gap_bits'], A['B_composition']['markov_null95'], Bb['B_composition']['gap_bits'], Bb['B_composition']['markov_null95']) == (0.0165, 0.0039, 0.0175, 0.0042)
      and round(m(A)['markov']['p'], 3) == 0.089 and round(m(Bb)['markov']['p'], 3) == 0.008 and CC['negative_control']['kind'] == 'bend' and round(CC['negative_control']['stress'], 2) == 0.83)
check("PhysioNet 2012: multiset p 0.023/0.016, order 0.97/0.21; ICU contextual p 1.0 both (and SOFA); lactate 0.0453 (null 0.0365), 0.0391 (0.0364), 0 after cell IPW by construction; 6 of 7 replicated",
      (round(A['C_path_outcome']['p_multiset_vs_x1'], 3), round(Bb['C_path_outcome']['p_multiset_vs_x1'], 3), round(A['C_path_outcome']['p_order_vs_multiset'], 2), round(Bb['C_path_outcome']['p_order_vs_multiset'], 2)) == (0.023, 0.016, 0.97, 0.21)
      and all(R[k]['p'] == 1.0 for R in (A, Bb) for k in ('D_contextuality_icu', 'D_sensitivity_sofa'))
      and (A['E_selection']['cf_available_case'], A['E_selection']['null_95'], Bb['E_selection']['cf_available_case'], Bb['E_selection']['null_95']) == (0.0453, 0.0365, 0.0391, 0.0364)
      and abs(A['E_selection']['cf_after_ipw']) < 1e-9 and abs(Bb['E_selection']['cf_after_ipw']) < 1e-9 and sum(v['replicated'] for v in CC['replication'].values()) == 6)
st = lambda R: {(s['frm'], s['to']): s['kind'] for s in R['A_coupling_geometry']['steps']}
check("PhysioNet 2012: decided tears coronary-cardiac surgery and cardiac surgery-medical in both sets",
      all(st(R)[('coronary', 'cardiac surgery')] == 'tear' and st(R)[('cardiac surgery', 'medical')] == 'tear' for R in (A, Bb)))
DB = E('open_clinical', 'results', 'frozen', 'results_diabetes130.json'); dbm = DB['B_composition']['latent_models']
check("Diabetes 130: 101,766 encounters, 71,518 patients, 6,339 with 3+; HbA1c 16.7%; 0.0937 available case (null 0.0135), after cell IPW 0 by construction, secondary 0.0081; gap 0.073 vs 0.0025; latent p 0.042/0.043; multiset p 0.72; order p 0.035 with -0.0016 bits; source p 1.0; decided tear and surfaces",
      (DB['cohort']['encounters'], DB['cohort']['patients'], DB['cohort']['patients_3plus'], round(100 * DB['E_selection']['measured_share'], 1)) == (101766, 71518, 6339, 16.7)
      and (DB['E_selection']['cf_available_case'], DB['E_selection']['null_95'], DB['E_selection']['cf_after_secondary_weights']) == (0.0937, 0.0135, 0.0081) and abs(DB['E_selection']['cf_after_ipw']) < 1e-9
      and (round(DB['B_composition']['gap_bits'], 3), DB['B_composition']['markov_null95'], round(dbm['markov']['p'], 3), round(dbm['drifting']['p'], 3)) == (0.073, 0.0025, 0.042, 0.043)
      and round(DB['C_path_outcome']['p_order_vs_multiset'], 3) == 0.035 and DB['C_path_outcome']['bits_gained_path'] == -0.0016 and round(DB['C_path_outcome']['p_multiset_vs_x1'], 2) == 0.72 and DB['D_contextuality_source']['p'] == 1.0
      and DB['A_coupling_geometry']['steps'][0]['kind'] == 'tear' and {k: (v['surface'], v['decided']) for k, v in DB['A_coupling_geometry']['toric_2x3_change_x_readmission'].items()}
      == {'emergency room': ('Bl_2 CP^2', True), 'physician referral': ('CP^1 x CP^1', True)})
HT = E('open_clinical', 'results', 'frozen', 'results_heart.json'); pp = {k: v['p'] for k, v in HT['D_contextuality_sites'].items()}
ms = HT['E_missing_share_by_site']
check("Heart: no pair contextual, smallest p 0.11 (Cleveland vs Switzerland, 8 without disease); ca missing 99% (Hungary, VA), 96% Switzerland; thal 90.5% Hungary, 83% VA, 42% Switzerland; available-case selection 0.0597",
      sum(HT['cohort'].values()) == 920 and not any(v['bonferroni_6'] or v['contextual'] for v in HT['D_contextuality_sites'].values()) and min(pp, key=pp.get) == 'cleveland vs switzerland'
      and round(min(pp.values()), 2) == 0.11 and HT['D_contextuality_sites']['cleveland vs switzerland']['strata']['diagnosis 0']['nY'] == 8
      and (round(100 * ms['hungarian']['ca']), round(100 * ms['va']['ca'])) == (99, 99) and sorted(round(100 * ms[s]['thal']) for s in ('hungarian', 'va')) == [83, 90] and round(100 * ms['switzerland']['thal']) == 42
      and round(100 * ms['switzerland']['ca']) == 96 and ms['hungarian']['thal'] == 0.905 and HT['E_selection']['cf_available_case'] == 0.0597)
UK, IE = E('c19', 'results', 'frozen', 'results_c19_uk.json'), E('c19', 'results', 'frozen', 'results_c19_ie.json')
u1, i1, u2, i2 = UK['B1_composition_W1_W3'], IE['B1_composition_W1_W3'], UK['B2_four_waves'], IE['B2_four_waves']
check("C19PRC B1: UK n 950 gap 0.057 vs 0.016, latent Markov p 0.50; Ireland n 418 gap 0.062 vs 0.037, p 0.042",
      (u1['n'], round(u1['gap_bits'], 3), round(u1['markov_null95'], 3), round(u1['latent_models']['markov']['p'], 2), i1['n'], round(i1['gap_bits'], 3), round(i1['markov_null95'], 3), round(i1['latent_models']['markov']['p'], 3))
      == (950, 0.057, 0.016, 0.5, 418, 0.062, 0.037, 0.042))
check("C19PRC B2 (W1-W4): UK n 771, G2 50.6, 54 df, p 0.89, free 0.72; Ireland n 327, p 0.45, free 0.32; both 'latent Markov suffices'",
      (u2['waves'], u2['n'], round(u2['latent_markov_shared']['G2'], 1), u2['latent_markov_shared']['df'], round(u2['latent_markov_shared']['p_boot'], 2), round(u2['latent_markov_free_response']['p_boot'], 2),
       i2['n'], round(i2['latent_markov_shared']['p_boot'], 2), round(i2['latent_markov_free_response']['p_boot'], 2)) == ([1, 2, 3, 4], 771, 50.6, 54, 0.89, 0.72, 327, 0.45, 0.32)
      and u2['decision'] == i2['decision'] == 'latent Markov suffices')
g3 = lambda R: [v['gap'] for v in R['B3_all_waves_gaps']['gaps'].values()]
check("C19PRC B3: UK n 582 gaps 0.057-0.099 increasing, Ireland n 271 gaps 0.066-0.148, all above the Markov null",
      (UK['B3_all_waves_gaps']['n'], round(min(g3(UK)), 3), round(max(g3(UK)), 3), IE['B3_all_waves_gaps']['n'], round(min(g3(IE)), 3), round(max(g3(IE)), 3)) == (582, 0.057, 0.099, 271, 0.066, 0.148)
      and g3(UK) == sorted(g3(UK)) and all(v['gap'] > v['markov_null95'] for R in (UK, IE) for v in R['B3_all_waves_gaps']['gaps'].values()))
check("C19PRC C/D/E: UK multiset p < 0.001, order 0.57; Ireland outcome not estimable; recontact vs fresh p 1.0 (UK), 0.48 (IE); UK attrition 0.0546 available case (null 0.0473), secondary 0.0125, 0 after cell IPW by construction; Ireland 0.031 vs 0.069, none detected",
      UK['C_path_outcome']['p_multiset_vs_x1'] < 0.001 and round(UK['C_path_outcome']['p_order_vs_multiset'], 2) == 0.57 and isinstance(IE['C_path_outcome'], str)
      and UK['D_panel_conditioning']['p'] == 1.0 and round(IE['D_panel_conditioning']['p'], 2) == 0.48
      and (UK['E_attrition']['cf_available_case'], UK['E_attrition']['null_95'], UK['E_attrition']['cf_after_secondary_weights']) == (0.0546, 0.0473, 0.0125)
      and (round(IE['E_attrition']['cf_available_case'], 3), round(IE['E_attrition']['null_95'], 3)) == (0.031, 0.069) and abs(UK['E_attrition']['cf_after_ipw']) < 1e-9
      and IE['E_attrition']['decision'] == 'no incompatibility beyond sampling' and (UK['cohort']['W1']['n'], IE['cohort']['W1']['n']) == (2025, 1041))

PW = E('snowqueen', 'results_cbd_split_power.json'); pw = {(r['nX'], r['margins_shifted'], r['eps']): r for r in PW}
check("planted CbD (cbd_split + boundary null): size <= 0.075 (0 with shifted margins); power <= 0.10 at deficit <= 0.04; 0.40-0.575 at the largest deficits",
      max(r['rate'] for r in PW if r['eps'] == 0) == 0.075 and all(r['rate'] == 0 for r in PW if r['eps'] == 0 and r['margins_shifted'])
      and max(r['rate'] for r in PW if r['population_deficit'] <= 0.04) == 0.1 and [pw[(n, sh, 0.02)]['rate'] for n in (500, 1200) for sh in (False, True)] == [0.55, 0.4, 0.575, 0.525]
      and [pw[(500, sh, e)]['population_deficit'] for sh in (False, True) for e in (0.01, 0.02)] == [0.08, 0.16, 0.0359, 0.0759])

q = {k: E('c19', 'results', f'b2_equal_starts_{k}.json') for k in ('uk', 'ie', 'power')}
check("C19PRC B2 rerun with 8 EM starts for observed and bootstrap fits: p unchanged (UK 0.886/0.723, Ireland 0.448/0.317); power vs planted memory 3/8 (n 771), 0/8 (n 327)",
      [round(q[k][m]['p_boot'], 3) for k in ('uk', 'ie') for m in ('shared', 'free')] == [0.886, 0.723, 0.448, 0.317] and q['ie']['shared']['G2'] == 64.2
      and (q['power']['power_n771']['rejections'], q['power']['power_n327']['rejections'], q['power']['power_n771']['reps']) == (3, 0, 8))
txt = open('validation.tex').read()
check("retired claims are gone from the paper (settles, explained by weighting, weighting removes, holds in the latent states)",
      not any(w in txt for w in ('settles', 'settle what', 'explained by weighting', 'weighting removes', 'holds in the latent', 'never contextually')))

# 5d. structural tests on synthetic data (ext/structure) ---------------------------------------------------------------
def frozen_list(folder, listing):
    ls = [l for l in open(os.path.join('ext', folder, listing)).read().split('\n') if l and not l.startswith('#')]
    return all(sha256(open(os.path.join('ext', folder, l.split()[1]), 'rb').read()).hexdigest() == l.split()[0] for l in ls)
check("structure: frozen data and results match their SHA-256 hashes", frozen_list('structure', 'FROZEN_DATA.txt') and frozen_list('structure', 'FROZEN_RESULTS.txt'))
ST = {k: E('structure', 'results', f'{k}_results.json') for k in ('t1', 't2', 't3', 't4')}
check("structure: 40 of 41 pre-registered expectations hold (T3 21/21, T1 13/13, T2 3/4, T4 3/3)",
      [(ST[k]['passed'], ST[k]['n']) for k in ('t3', 't1', 't2', 't4')] == [(21, 21), (13, 13), (3, 4), (3, 3)])
it = lambda k, frag: next(i for i in ST[k]['items'] if frag in i['test'])
t2f = it('t2', 'planted law lambda^B_eps: (U)')
check("structure T2: the one failed expectation is the random-instance check at eps = 1 (0 violations of (M)); targeted defect eps/126 exact",
      not t2f['passed'] and [s['violations']['M'] for s in t2f['observed'] if s['eps'] == '1'] == [0]
      and [s['defect'] for s in it('t2', 'level-one (M) defect')['observed']] == ['0', '1/12600', '1/1260', '1/504', '1/252', '1/126'])
lv = lambda frag: it('t1', frag)['observed']
check("structure T1: planted claims fail first at levels 1, 2, 3 with d1 = 0.4, d2 = 1/2, d3 = 2; Bands Ex. 9.1 reproduced",
      (lv('level 1:')['level'], lv('level 1:')['d1'], lv('level 2:')['level'], lv('level 2:')['d2'], lv('level 3:')['level'], lv('level 3:')['d3']) == (1, 0.4, 2, '1/2', 3, '2')
      and it('t1', 'Bands Ex. 9.1')['passed'])
check("structure T3: functoriality defect -2e^2(1-e)^2; t(1/10) = 369/2500; integral kernels are the 10 predicted",
      it('t3', 'sympy formula')['expected'] == 'entry (0,1) = -2*e**2*(e - 1)**2' and it('t3', 'F2(beta_r')['observed']['t_0_1'] == '369/2500'
      and len(it('t3', 'integral on circulations')['observed']['integral']) == 10)

bi = {k: E('c19', 'results', f'banding_{k}.json') for k in ('uk', 'ie')}; cal = E('c19', 'results', 'banding_calibration.json')
check("banding interchange: identity exact, defect a circulation; corrected null 0.0231/0.0554 vs monitor 0.0164/0.0367; p 0.0025/0.022; share 0.21/0.34; bands not sufficient",
      all(bi[k]['decomposition']['identity_err'] < 1e-12 and bi[k]['decomposition']['interchange_is_circulation'] for k in bi)
      and [(bi[k]['raw_markov_null95_of_banded_gap'], bi[k]['banded_markov_null95'], round(bi[k]['p_raw_markov'], 4), round(bi[k]['decomposition']['share_interchange'], 2)) for k in ('uk', 'ie')]
      == [(0.0231, 0.0164, 0.0025, 0.21), (0.0554, 0.0367, 0.0224, 0.34)]
      and all(bi[k]['sufficiency'][d]['fisher_p'] < 1e-4 for k in bi for d in ('D1', 'D3')))
check("banding interchange: plan and results match their frozen hashes", frozen_list('c19', 'results/FROZEN_BANDING.txt'))
check("banding calibration: monitor banded null rejects 20/20 first-order raw chains; interchange test 1/20, power 17/20 and 20/20",
      (cal['0.0']['monitor_banded_null'], cal['0.0']['primary'], cal['0.2']['primary'], cal['0.6']['primary']) == (20, 1, 17, 20))

TW = E('open_clinical', 'results', 'tower_icu.json'); TX = E('open_clinical', 'results', 'tower_icu_exploratory.json'); IC = E('open_clinical', 'results', 'interchange_icu.json')
check("tower and ICU interchange: plan, adapter checks and results match their frozen hashes", frozen_list('open_clinical', 'results/FROZEN_TOWER_ICU.txt'))
sp, sh = TW['same_population'], TW['shift']
check("PhysioNet tower: same population idle (Delta CI spans 0, alpha=1 0.95, calibrated 0.141/0.141, F1 at 1.0); shift: alpha<1 mass 0.56, Delta CI spans 0, 0.132 vs 0.184, CP^2 at 1.0; condS agreement < 1e-12",
      sp['delta_ci'][0] < 0 < sp['delta_ci'][1] and round(sp['posterior_alpha']['1.0'], 2) == 0.95 and (round(sp['calibration']['predicted'], 3), round(sp['calibration']['observed'], 3)) == (0.141, 0.141)
      and sp['surface_top'] == {'surface': 'F_1 = Bl_1 CP^2', 'prob': 1.0} and not sh['level3_earns_keep'] and round(sh['mass_alpha_below_1'], 2) == 0.56
      and (round(sh['calibration']['predicted'], 3), round(sh['calibration']['observed'], 3)) == (0.132, 0.184) and sh['surface_top'] == {'surface': 'CP^2', 'prob': 1.0}
      and max(sp['condS_agreement_200'], sh['condS_agreement_200']) < 1e-12 and not sp['tears_of_tube'] and not sh['tears_of_tube'])
ef = TX['shift_evidence_floor']
check("PhysioNet tower exploratory: level-1 obstruction d1 0.107 = certificate; observed chambers undecided; evidence floor Delta 0.0091 (0.0025, 0.0161), surfaces 0.53/0.47, six crossings",
      TX['shift']['level1_obstruction']['d1'] == TX['shift']['level1_obstruction']['certificate_margin'] and round(TX['shift']['level1_obstruction']['d1'], 3) == 0.107
      and not TX['shift']['surface_observed_decided'] and not TX['same_population']['surface_observed_decided']
      and (round(ef['delta'], 4), round(ef['delta_ci'][0], 4), round(ef['delta_ci'][1], 4)) == (0.0091, 0.0025, 0.0161) and len(ef['tears_of_tube']) == 6
      and (round(ef['surface_posterior']['F_1 = Bl_1 CP^2'], 2), round(ef['surface_posterior']['CP^2'], 2)) == (0.53, 0.47))
check("ICU interchange: corrected null 0.0078/0.0068 vs monitor 0.0039; gaps 0.0165/0.0175 exceed it (p 0.0025); share 0.38/0.24; bands not sufficient",
      [(IC[k]['raw_markov_null95_of_banded_gap'], IC[k]['banded_markov_null95'], round(IC[k]['banded_gap_bits'], 4), round(IC[k]['p_raw_markov'], 4), round(IC[k]['decomposition']['share_interchange'], 2)) for k in ('set_a', 'set_b')]
      == [(0.0078, 0.0039, 0.0165, 0.0025, 0.38), (0.0068, 0.0039, 0.0175, 0.0025, 0.24)] and all(IC[k]['sufficiency'][d]['fisher_p'] < 1e-18 for k in IC if k.startswith('set') for d in ('D1', 'D3')))

P4 = lambda f: E('phase4', 'results', f)
check("phase 4: plan and results match their frozen hashes", frozen_list('phase4', 'FROZEN_RESULTS.txt'))
w1c, w1g = P4('wp1_clinical.json'), P4('wp1_gss.json')['result']
check("WP1: ICU geometry adds nothing (death gain 0.0004, CI spans 0; shock onset spans 0; 98.8% broken); GSS chamber change gain 0.041 (0.020, 0.062)",
      not w1c['primary_death']['geometry_forecasts_beyond_margins'] and abs(w1c['primary_death']['delta_B2_S2'] - 0.00035) < 1e-6 and not w1c['secondary_shock_onset']['geometry_forecasts_beyond_margins']
      and w1c['descriptive']['broken_share_W2_setA'] == 0.988 and w1g['geometry_forecasts_beyond_margins'] and (round(w1g['delta_B2_S2'], 3), round(w1g['ci_B2_S2'][0], 3), round(w1g['ci_B2_S2'][1], 3)) == (0.041, 0.02, 0.062)
      and (w1c['primary_death']['n_train'], w1c['primary_death']['n_test'], w1c['primary_death']['events_test'], w1g['n_train'], w1g['n_test']) == (3758, 3802, 549, 1320, 1100))
w2 = P4('wp2_selection.json'); fits = {k: [m for m in ('X-only', 'Y-only', 'external', 'X+external') if w2[k][m]['consistent']] for k in ('physionet_set_a', 'physionet_set_b', 'diabetes130', 'c19prc_uk_attrition')}
check("WP2: restricted models fit UK attrition (X-only, external, X+external) and nothing for ICU lactate (both sets) or diabetes HbA1c",
      fits == {'physionet_set_a': [], 'physionet_set_b': [], 'diabetes130': [], 'c19prc_uk_attrition': ['X-only', 'external', 'X+external']}
      and (w2['diabetes130']['X-only']['cf'], w2['c19prc_uk_attrition']['X-only']['cf'], w2['c19prc_uk_attrition']['X-only']['null95']) == (0.0163, 0.0118, 0.017))
w3 = P4('wp3_events.json')
check("WP3: no ICU structural event associated with death (all p > 0.2 crude and within deciles); GSS Spearman 0.346, p 0.11; 8 LLM tasks tabulated",
      all(v['crude']['p'] > 0.2 and v['within_margin_risk_deciles']['p'] > 0.2 for v in w3['physionet_death'].values()) and w3['physionet_n'] == 7560
      and w3['gss_partisan']['spearman_flip_share_vs_abs_d_rep'] == 0.346 and round(w3['gss_partisan']['p'], 2) == 0.11 and len(w3['llm_hallucination_proxy']['rows']) == 8)
w4 = P4('wp4_tower.json'); ar = w4['arms']; vf = lambda k, v: ar[k]['variants'][v]
check("WP4: H1 (evidence floor 2/3), H2 (family 3/3), H3 (no harm) hold; family predicts 0.1764 vs 0.175 (R4), puts 0.895 on F1 in R0b",
      w4['H1_Tev_replicates'] == {'arms_positive': 2, 'of': 3, 'holds': True} and w4['H2_Tfam'] == {'arms_positive': 3, 'of': 3, 'holds': True} and w4['H3_no_harm']['holds']
      and vf('R4 cross shift A coronary+cardiac surgery -> B medical+surgical', 'T-fam')['predicted'] == 0.1764
      and round(vf('R0b shift A surgical -> B medical/coronary', 'T-fam')['surface_posterior']['F_1 = Bl_1 CP^2'], 3) == 0.895)
check("phase 4 planted checks all PASS", all(open(os.path.join('ext', 'phase4', 'results', f)).read().startswith('PASS') for f in ('wp1_gss_synth.txt', 'wp3_synth.txt', 'wp4_tower_synth.txt')))

# 6. tables -----------------------------------------------------------------------------------------------------
saved = {t: open(t).read() for t in ('tab_llm.tex', 'tab_gss.tex', 'tab_panels.tex', 'tab_usmle.tex', 'tab_panelmodels.tex')}
rc, _ = run('make_tables.py')
check("tables regenerate identically from the results", rc == 0 and all(open(t).read() == s for t, s in saved.items()))
tex = open('validation.tex').read()
check("paper inputs the generated tables", all(f"\\input{{{t[:-4]}.tex}}" in tex for t in saved))

print(f"\n{sum(results)}/{len(results)} checks pass")
sys.exit(0 if all(results) else 1)
