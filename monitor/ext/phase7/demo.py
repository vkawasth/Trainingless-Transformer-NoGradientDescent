"""Worked example (illustrative, synthetic): three possible answers to one question, four sources of known reliability,
plus a decision with a utility matrix. Output: results/demo.json"""
import json, numpy as np, decide as D
rng = np.random.default_rng(5); q = np.array([.55, .30, .15])            # truth over answers a, b, c
O = ['a', 'b', 'c']
S = [D.Source('reference text', rng.multinomial(400, .9 * q + .1 * np.eye(3)[1]), .9),
     D.Source('forum', rng.multinomial(3000, .3 * q + .7 * np.eye(3)[2]), .3),          # 70% wrong, pushes c
     D.Source('paraphrase set', rng.multinomial(1500, .6 * q + .4 * np.eye(3)[1]), (.6, .7)),  # reliability interval, pushes b
     D.Source('self-generated', rng.multinomial(2000, [.2, .8, 0]), .8)]              # overclaimed: conflicts with the rest
out = {}
U, acts = D.assert_actions(O)
for k in (1.0, 1.1, 1.5, 2.0):
    out[f'assert, kappa={k}'] = D.decide(S, U, acts, kappa=k)
# a decision with consequences: act on a, act on b, or gather more (constant utility)
Uact = np.array([[1.0, 0.2, 0.0], [0.3, 1.0, 0.1], [0.45, 0.45, 0.45]]); out['action, kappa=1.1'] = D.decide(S, Uact, ['act for a', 'act for b', 'defer'], kappa=1.1)
json.dump(out, open('results/demo.json', 'w'), indent=1, default=float)
for k, d in out.items():
    print(f"\n== {k}: {d['status']}  action={d['action']}  hedge={d['hedge_set']}  robust={d['robust_choice']}")
    print('   strength kappa* vs rivals:', d['strength_kappa_star'], ' margin vs', d['closest_rival'], d['margin_vs_closest'])
    for r in d['trace']['repairs']: print('   repair:', r['repaired_source'], 'stated', r['reliability_stated'], 'certified max', r['reliability_certified_max'], 'alternatives', r['resolvable_by_excluding'], 'conflicts', r['conflicts'])
    print('   contributions:', d['trace'].get('contributions'))
    print('   binding:', [(b['source'], b['kind'], O[b['outcome']], b['dual']) for b in d['trace'].get('binding', [])])
    print('   weakest link:', d['weakest_link']['source'], {s: (v['r'], v['r_critical']) for s, v in d['weakest_link']['by_source'].items()})
    print('   thinnest cell:', d['thinnest_cell'] and (d['thinnest_cell']['source'], O[d['thinnest_cell']['outcome']], d['thinnest_cell']['counts']), ' VOI best:', d['value_of_information']['best'], d['value_of_information']['margin_gain'])
