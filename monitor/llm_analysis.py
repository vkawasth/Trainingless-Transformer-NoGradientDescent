"""LLM retraining: GPT-3.5 and GPT-4, March (0301/0314) and June (0613) 2023, same questions (Chen, Zaharia, Zou 2023, LLMDrift).
Each item gets an outcome in {0: correct (or answered), 1: wrong, 2: no valid answer (refusal or unparseable)}.
System per task: contents A = GPT-3.5, B = GPT-4, measured jointly on the same items; contexts = March, June."""
import re, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monitor import State, regime, cbd_rank2, cbd_split, churn, bands
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'llmdrift')
MODELS = {'35M': 'gpt-3.5-turbo-0301', '35J': 'gpt-3.5-turbo-0613', '4M': 'gpt-4-0314', '4J': 'gpt-4-0613'}

def letter(s):
    """The final answer letter: the last 'answer is (X)'; otherwise the parenthesized letter if only one is given, else the last one."""
    s = str(s)
    m = re.findall(r'answer is:?\s*\(?([A-J])\b', s)
    if m: return m[-1]
    m = re.findall(r'\(([A-J])\)', s)
    if m: return m[0] if len(set(m)) == 1 else m[-1]
    m = re.match(r'\s*([A-J])[\.\)]', s); return m.group(1) if m else None
def yesno(s):
    """The final [Yes]/[No] (the last bracketed one), else a leading yes/no."""
    s = str(s); m = re.findall(r'\[(yes|no)\]', s, re.I)
    if m: return m[-1].lower()
    m = re.match(r'\s*(yes|no)\b', s, re.I); return m.group(1).lower() if m else None
def boxed(s):
    m = re.findall(r'\\boxed\{\s*(-?\d+)\s*\}', str(s)); return m[-1] if m else None
def norm_txt(s): return re.sub(r'[^a-z0-9 ]', '', str(s).lower()).strip()

def outcomes(task, variant='intact'):
    """Outcome per (item, model). Items are keyed by id, except: OpinionQA reuses a few ids for different questions, so its items are
    keyed by (id, query); USMLE asks every question twice, once with an intact prompt and once with a damaged one (a line of the
    question or the first option is missing), so variant='intact' keeps the longer prompt and variant='damaged' the shorter."""
    d = pd.read_csv(os.path.join(D, f'{task}_EVAL.csv'), low_memory=False)
    d['mshort'] = d.model.str.split('/').str[-1]
    if task == 'OPINIONQA':
        d['item'] = d['id'].astype(str) + '|' + d['query'].astype(str)
    elif task == 'USMLE':
        d['qlen'] = d['query'].str.len()
        d = d.sort_values(['mshort', 'id', 'qlen', 'Unnamed: 0'])
        d = d.groupby(['mshort', 'id'], as_index=False).nth(-1 if variant == 'intact' else 0)
        d['item'] = d['id']
    else:
        d['item'] = d['id']
    assert not d.duplicated(['mshort', 'item']).any()
    def code(r):
        if task == 'USMLE':
            a, ref = letter(r.answer), letter(r.ref_answer)
            return 2 if a is None else (0 if a == ref else 1)
        if task == 'OPINIONQA':                          # no correct answer exists: answered (a valid option) or not
            return 2 if letter(r.answer) is None else 0
        if task == 'PRIME_FULL':
            a = yesno(r.answer); return 2 if a is None else (0 if a == str(r.ref_answer).strip().lower() else 1)
        if task == 'COUNTAHAPPYNUMBER':
            a, ref = boxed(r.answer), boxed(r.ref_answer); return 2 if a is None else (0 if a == ref else 1)
        if task == 'ARC':
            a = str(r.answer).strip(); return 2 if not a.startswith('[[') else (0 if a.replace(' ', '') == str(r.ref_answer).replace(' ', '') else 1)
        if task == 'SENSITIVEQ':
            return 0 if int(r['Response Rate']) == 1 else 2
        if task == 'LEETCODE_EASY':
            v = str(r['Directly Usable']).strip().lower(); return 0 if v in ('true', '1', 'yes', 'accepted') else 1
        if task == 'HOTPOTQA':
            a = str(r.answer)
            if a.startswith('Could not parse') or a.lower().startswith('agent stopped'): return 2
            return 0 if norm_txt(a) == norm_txt(r.ref_answer) else 1
    d['y'] = d.apply(code, axis=1)
    w = d.pivot(index='item', columns='mshort', values='y').dropna()
    return w.rename(columns={v: k for k, v in MODELS.items()}).astype(int)

def table(a, b, k=3):
    T = np.zeros((k, k))
    for x, y in zip(a, b): T[x, y] += 1
    return T

TASKS = ['USMLE', 'OPINIONQA', 'SENSITIVEQ', 'PRIME_FULL', 'COUNTAHAPPYNUMBER', 'ARC', 'LEETCODE_EASY', 'HOTPOTQA']
LABEL = {0: 'correct', 1: 'wrong', 2: 'no answer'}
if __name__ == '__main__':
    out = {}
    for task in TASKS:
        w = outcomes(task); n = len(w); k = 3
        cats_used = sorted(set(w.values.ravel()))
        res = dict(n_items=n, margins={m: [float((w[m] == c).mean()) for c in range(k)] for m in MODELS})
        # drop categories that never occur in any model, so that margins live in the open simplex where possible
        keep = [c for c in range(k) if (w.values == c).any()]
        remap = {c: i for i, c in enumerate(keep)}; wk = w.replace(remap); kk = len(keep)
        TM, TJ = table(wk['35M'], wk['4M'], kk), table(wk['35J'], wk['4J'], kk)
        if kk >= 2:
            SM, SJ = State(TM + 1e-9, n), State(TJ + 1e-9, n)
            rg = regime(SM, SJ)
            res['regime'] = dict(kind=rg['kind'], stress=round(rg['stress'], 2), change=round(rg['change'], 3), tear_margin_March=round(rg['mT'], 3),
                                 walls_crossed=[[[keep[i] for i in I], [keep[j] for j in J]] for I, J in rg['walls_crossed']],
                                 walls_maybe=[[[keep[i] for i in I], [keep[j] for j in J]] for I, J in rg['walls_maybe']])
        # break, per model family: a category with band above 1% in March and below 1% in June
        res['breaks'] = {}
        for fam in ('35', '4'):
            pM = np.array(res['margins'][fam + 'M']); pJ = np.array(res['margins'][fam + 'J'])
            loM, hiM = bands(pM, n); loJ, hiJ = bands(pJ, n)
            res['breaks'][fam] = [LABEL[c] for c in range(k) if loM[c] > 0.01 and hiJ[c] < 0.01]
        # CbD: canonical (all dichotomizations) on the full tables, and the rank-2 reduction 'first outcome vs rest'
        bM = table((w['35M'] != 0).astype(int), (w['4M'] != 0).astype(int), 2); bJ = table((w['35J'] != 0).astype(int), (w['4J'] != 0).astype(int), 2)
        cb = cbd_rank2(bM / n, bJ / n)
        res['cbd'] = {kk2: (round(v, 4) if isinstance(v, float) else v) for kk2, v in cb.items()}
        sp = cbd_split(TM / n, TJ / n)
        res['cbd_split'] = dict(contextual=sp['contextual'], deficit=round(sp['deficit'], 4), n_splits=sp['n_splits'])
        # churn per family: observed agreement of March and June outcomes on the same items vs the maximal coupling
        res['churn'] = {}
        for fam in ('35', '4'):
            c = churn(w[fam + 'M'].values, w[fam + 'J'].values, list(range(k)))
            T = c['T']
            res['churn'][fam] = dict(agree=round(c['agree'], 3), max_agree=round(c['max_agree'], 3), excess=round(c['excess'], 3),
                                     acc_M=round(float((w[fam + 'M'] == 0).mean()), 3), acc_J=round(float((w[fam + 'J'] == 0).mean()), 3),
                                     lost=round(float(T[0, 1:].sum()), 3), gained=round(float(T[1:, 0].sum()), 3),
                                     masked=round(float(min(T[0, 1:].sum(), T[1:, 0].sum())), 3),
                                     excess_binary=round(churn((w[fam + 'M'] != 0).astype(int).values, (w[fam + 'J'] != 0).astype(int).values, [0, 1])['excess'], 3))
        out[task] = res
        r = res.get('regime', {})
        print(f"{task:18s} n={n:5d} | regime {r.get('kind','-'):6s} stress {r.get('stress','-')!s:6s} crossed {r.get('walls_crossed')} maybe {r.get('walls_maybe')} | breaks {res['breaks']} | "
              f"CbD contextual {cb['contextual']} (s={cb['s_odd']:.3f}, Delta={cb['Delta']:.3f}), split {sp['contextual']} deficit {sp['deficit']:.4f} | "
              + " ; ".join(f"GPT-{f}: acc {v['acc_M']}->{v['acc_J']}, lost {v['lost']}, gained {v['gained']}, masked {v['masked']}, excess churn {v['excess']} (binary {v['excess_binary']})" for f, v in res['churn'].items()))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_results.json'), 'w'), indent=1)
