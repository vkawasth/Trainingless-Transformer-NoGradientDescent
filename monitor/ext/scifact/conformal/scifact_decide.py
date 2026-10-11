"""PREANALYSIS_SCIFACT.md: the decision layer on SciFact (oracle abstracts). Run once. Output: results/scifact_results.json"""
import os, sys, json, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.sparse import hstack, csr_matrix
from scipy.stats import beta
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, '..', '..', 'phase7'))
import decide as DC
SEED = 20261013; rng = np.random.default_rng(SEED); D = os.environ.get('SCIFACT_DIR', '/home/claude/monitor_data_sf/data'); LAB = ['SUPPORT', 'CONTRADICT', 'NEI']
ALPHA = 0.05 / 3; KAPPA = 1.1; TAU = 0.5

corpus = {d['doc_id']: d for d in map(json.loads, open(os.path.join(D, 'corpus.jsonl')))}
def pairs(path):
    out = []
    for c in map(json.loads, open(path)):
        for d in c['cited_doc_ids']:
            ev = c['evidence'].get(str(d)); out.append(dict(claim_id=c['id'], claim=c['claim'], doc=int(d), y=LAB.index(ev[0]['label']) if ev else 2))
    return out
train, dev = pairs(os.path.join(D, 'claims_train.jsonl')), pairs(os.path.join(D, 'claims_dev.jsonl'))
DRY = bool(os.environ.get('DRYRUN'))   # debugging only: replaces dev by proper-train pairs; output not kept

# ---- features
tf = TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=2).fit([s for d in corpus.values() for s in d['abstract']] + [d['title'] for d in corpus.values()])
CUE = dict(neg=r"\b(not|no|none|never|neither|nor|without|fail(s|ed)?|lack(s|ed)?|absen(ce|t)|cannot|unable|ineffective|unrelated|independent)\b",
           dec=r"\b(decreas\w*|reduc\w*|lower(s|ed)?|inhibit\w*|suppress\w*|less|loss|impair\w*|attenuat\w*|down\w*)\b",
           inc=r"\b(increas\w*|elevat\w*|higher|enhanc\w*|promot\w*|induc\w*|more|greater|up\w*|activat\w*)\b")
def cues(t): t = t.lower(); return np.array([len(re.findall(CUE[k], t)) for k in ('neg', 'dec', 'inc')], float)
def views(p):
    d = corpus[p['doc']]; S = d['abstract']; cv = tf.transform([p['claim']]); sv = tf.transform(S); cos = (sv @ cv.T).toarray().ravel()
    o = np.argsort(-cos); top = [S[i] for i in o[:2]]; best = cos[o[0]]
    tcos = float((tf.transform([d['title']]) @ cv.T).toarray()[0, 0]); cw = set(re.findall(r'\w+', p['claim'].lower())); aw = set(re.findall(r'\w+', ' '.join(S).lower()))
    A = [best, cos.mean(), cos[o[:2]].mean(), o[0] / max(len(S) - 1, 1), tcos, len(cw & aw), len(cw & aw) / max(len(cw), 1), len(S)]
    cc, sc = cues(p['claim']), cues(' '.join(top))
    B = list(cc) + list(sc) + list(cc * best) + list(sc * best) + [float((cc[0] > 0) != (sc[0] > 0)), float(cc[1] > 0 and sc[2] > 0), float(cc[2] > 0 and sc[1] > 0)]
    return np.array(A), np.array(B), p['claim'], ' '.join(top)
def build(P):
    V = [views(p) for p in P]; return np.array([v[0] for v in V]), np.array([v[1] for v in V]), [v[2] for v in V], [v[3] for v in V], np.array([p['y'] for p in P])

claims = sorted({p['claim_id'] for p in train}); cal_ids = set(rng.choice(claims, int(round(0.25 * len(claims))), replace=False).tolist())
ptr = [p for p in train if p['claim_id'] not in cal_ids]; pcal = [p for p in train if p['claim_id'] in cal_ids]
if DRY: dev = ptr[:120]
Atr, Btr, Ctr, Str, ytr = build(ptr); Aca, Bca, Cca, Sca, yca = build(pcal); Adv, Bdv, Cdv, Sdv, ydv = build(dev)
hc, hs = HashingVectorizer(n_features=2 ** 14, alternate_sign=False, norm=None), HashingVectorizer(n_features=2 ** 14, alternate_sign=False, norm=None)
tc, ts = TfidfTransformer().fit(hc.transform(Ctr)), TfidfTransformer().fit(hs.transform(Str))
def Xc(C, S): return hstack([tc.transform(hc.transform(C)), ts.transform(hs.transform(S))]).tocsr()
M = dict(A=make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Atr, ytr),
         B=make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Btr, ytr),
         C=make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=1.0, max_iter=5000)).fit(Xc(Ctr, Str), ytr))
def proba(k, A, B, C, S): return M[k].predict_proba(A if k == 'A' else (B if k == 'B' else Xc(C, S)))
Pca = {k: proba(k, Aca, Bca, Cca, Sca) for k in M}; Pdv = {k: proba(k, Adv, Bdv, Cdv, Sdv) for k in M}
def conf_q(v, a): v = np.sort(v); j = int(np.floor((len(v) + 1) * a)); return float(v[max(j - 1, 0)]) if j >= 1 else 0.0
r = {k: conf_q(Pca[k][np.arange(len(yca)), yca], ALPHA) for k in M}
cal_acc = {k: float((Pca[k].argmax(1) == yca).mean()) for k in M}; best = max(cal_acc, key=cal_acc.get); r_single = conf_q(Pca[best][np.arange(len(yca)), yca], 0.05)
U, acts = DC.assert_actions(LAB)

def run(i, keys, rr, floors=None):
    S = [DC.Source(k, Pdv[k][i] * 1e6, rr[k]) for k in keys]; return DC.decide(S, U, acts, kappa=KAPPA, floors=floors)
recs = []
for i in range(len(dev)):
    d = run(i, list(M), r); ds = run(i, [best], {best: r_single})
    act = d['action']; fl_ok = None
    if act is not None:
        l = acts.index(act); lo = DC.margin([DC.Source(k, Pdv[k][i] * 1e6, r[k]) for k in M if k not in [x['repaired_source'] for x in d['trace']['repairs']]], 3, np.eye(3)[l])
        fl_ok = lo is not None and lo >= TAU
    wl = d['weakest_link']
    recs.append(dict(claim_id=dev[i]['claim_id'], y=int(ydv[i]), act=None if act is None else acts.index(act), single_act=None if ds['action'] is None else acts.index(ds['action']),
                     floor_act=bool(fl_ok) if act is not None else False, evid=d['obstructions']['evid'] is not None, miss=bool(any(Pdv[k][i][ydv[i]] < r[k] for k in M)),
                     weakest=(wl or {}).get('source'), slack=((wl or {}).get('by_source', {}).get((wl or {}).get('source'), {}) or {}).get('slack'),
                     best_pred=int(Pdv[best][i].argmax()), best_conf=float(Pdv[best][i].max())))
y = ydv; n = len(recs); acted = [x for x in recs if x['act'] is not None]; wrong = sum(x['act'] != x['y'] for x in acted)
cp_lo = float(beta.ppf(0.025, wrong, n - wrong + 1)) if wrong > 0 else 0.0
res = dict(plan='PREANALYSIS_SCIFACT.md', n_train_pairs=len(ptr), n_cal_pairs=len(pcal), n_dev_pairs=n, reliabilities={k: round(v, 4) for k, v in r.items()},
           calibration_accuracy={k: round(v, 4) for k, v in cal_acc.items()}, best_single=best, r_single=round(r_single, 4),
           dev_accuracy_argmax={k: round(float((Pdv[k].argmax(1) == y).mean()), 4) for k in M})
res['P1'] = dict(acted=len(acted), act_and_wrong=wrong, rate=round(wrong / n, 4), cp_lower=round(cp_lo, 4), some_source_misses=round(float(np.mean([x['miss'] for x in recs])), 4), holds=bool(cp_lo < 0.05))
res['P2'] = dict(coverage=round(len(acted) / n, 4), holds=bool(len(acted) / n >= 0.25))
sa = [x for x in recs if x['single_act'] is not None]
res['P3'] = dict(three_source_acts=len(acted), single_source_acts=len(sa), single_wrong=sum(x['single_act'] != x['y'] for x in sa), holds=bool(len(acted) >= len(sa)))
# P4: matched-coverage confidence thresholding of the best source
order = np.argsort([-x['best_conf'] for x in recs]); keep_base = set(order[:len(acted)].tolist()); keep_lay = {i for i, x in enumerate(recs) if x['act'] is not None}
cl = np.array([x['claim_id'] for x in recs]); uc = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in uc}
def acc_on(keep, pred):
    k = list(keep); return np.mean([pred(i) == recs[i]['y'] for i in k]) if k else np.nan
def p4_diff(sample):
    rows = np.concatenate([idx[c] for c in sample]); kb = [i for i in rows if i in keep_base]; kl = [i for i in rows if i in keep_lay]
    if not kb or not kl: return np.nan
    return np.mean([recs[i]['act'] == recs[i]['y'] for i in kl]) - np.mean([recs[i]['best_pred'] == recs[i]['y'] for i in kb])
d0 = p4_diff(uc); boots = [p4_diff(rng.choice(uc, len(uc), replace=True)) for _ in range(2000)]; boots = np.array([b for b in boots if not np.isnan(b)])
res['P4'] = dict(layer_acc=round(float(acc_on(keep_lay, lambda i: recs[i]['act'])), 4) if keep_lay else None, baseline_acc=round(float(acc_on(keep_base, lambda i: recs[i]['best_pred'])), 4) if keep_base else None,
                 diff=None if np.isnan(d0) else round(float(d0), 4), ci=[round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)] if len(boots) else None,
                 holds=bool(len(boots) and np.percentile(boots, 2.5) > -0.05))
def p5_diff(sample):
    rows = np.concatenate([idx[c] for c in sample]); a = [recs[i]['best_pred'] == recs[i]['y'] for i in rows if recs[i]['act'] is not None]; h = [recs[i]['best_pred'] == recs[i]['y'] for i in rows if recs[i]['act'] is None]
    return np.mean(a) - np.mean(h) if a and h else np.nan
d5 = p5_diff(uc); b5 = np.array([b for b in (p5_diff(rng.choice(uc, len(uc), replace=True)) for _ in range(2000)) if not np.isnan(b)])
res['P5'] = dict(diff=None if np.isnan(d5) else round(float(d5), 4), ci=[round(float(np.percentile(b5, 2.5)), 4), round(float(np.percentile(b5, 97.5)), 4)] if len(b5) else None,
                 holds=bool(len(b5) and np.percentile(b5, 2.5) > 0))
fa = [x for x in recs if x['floor_act']]
res['descriptive'] = dict(evidence_obstruction_share=round(float(np.mean([x['evid'] for x in recs])), 4),
                          floor_variant=dict(acted=len(fa), wrong=sum(x['act'] != x['y'] for x in fa)),
                          weakest_link_share={k: round(sum(x['weakest'] == k for x in acted) / max(len(acted), 1), 4) for k in M},
                          weakest_link_median_slack=None if not acted else round(float(np.median([x['slack'] for x in acted if x['slack'] is not None] or [np.nan])), 4),
                          by_gold={LAB[g]: dict(n=sum(x['y'] == g for x in recs), acted=sum(x['y'] == g and x['act'] is not None for x in recs),
                                                correct=sum(x['y'] == g and x['act'] == g for x in recs)) for g in range(3)},
                          act_label_counts={LAB[g]: sum(x['act'] == g for x in acted) for g in range(3)})
# ---- amendment 1: descriptive risk-coverage curve
curve = []
for at in (0.05, 0.10, 0.20, 0.30):
    ra = {k: conf_q(Pca[k][np.arange(len(yca)), yca], at / 3) for k in M}; rs = conf_q(Pca[best][np.arange(len(yca)), yca], at)
    a3 = [run(i, list(M), ra)['action'] for i in range(len(dev))]; a1 = [run(i, [best], {best: rs})['action'] for i in range(len(dev))]
    def summ(A): ac = [(acts.index(a), int(y[i])) for i, a in enumerate(A) if a is not None]; return dict(acted=len(ac), coverage=round(len(ac) / n, 4), wrong=sum(p != t for p, t in ac), act_and_wrong_rate=round(sum(p != t for p, t in ac) / n, 4))
    curve.append(dict(alpha_total=at, reliabilities={k: round(v, 4) for k, v in ra.items()}, three_sources=summ(a3), best_single=dict(r=round(rs, 4), **summ(a1))))
res['amendment1_risk_coverage'] = curve
if DRY: print(json.dumps(res, indent=1)); sys.exit(0)
os.makedirs(os.path.join(H, 'results'), exist_ok=True)
json.dump(res, open(os.path.join(H, 'results', 'scifact_results.json'), 'w'), indent=1)
json.dump(recs, open(os.path.join(H, 'results', 'scifact_decisions.json'), 'w'))
print(json.dumps(res, indent=1))
