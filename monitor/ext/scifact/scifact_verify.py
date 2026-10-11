"""PREANALYSIS_SCIFACT.md: deciding SciFact claim-abstract labels from three dependent noisy verifiers with credal rules.
Usage: python3 scifact_verify.py checks   -> results/scifact_checks.txt  (planted checks on synthetic channels)
       python3 scifact_verify.py run      -> results/scifact_results.json (run once)
Data: SCIFACT_DIR (default /home/claude/monitor_data_sf/data); not packaged."""
import os, sys, json, re, itertools
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(H, '..', '..')))
from monitor import bands
SEED = 20261013; DATA = os.environ.get('SCIFACT_DIR', '/home/claude/monitor_data_sf/data'); LAB = ['SUPPORT', 'CONTRADICT', 'NEI']; KAPPA, TAU = 1.1, 0.5
NEG = {'no', 'not', 'never', 'without', 'none', 'nor', 'cannot', 'fail', 'fails', 'failed', 'lack', 'lacks', 'absence', 'absent', 'neither', 'unable', "n't"}
UP = {'increase', 'increases', 'increased', 'increasing', 'higher', 'elevated', 'enhance', 'enhances', 'enhanced', 'promote', 'promotes', 'improve', 'improves',
      'improved', 'more', 'greater', 'raise', 'raises', 'upregulate', 'upregulates', 'upregulated', 'induce', 'induces', 'induced', 'activate', 'activates', 'positive'}
DOWN = {'decrease', 'decreases', 'decreased', 'reduce', 'reduces', 'reduced', 'reduction', 'lower', 'inhibit', 'inhibits', 'inhibited', 'suppress', 'suppresses',
        'suppressed', 'less', 'decline', 'declines', 'downregulate', 'downregulated', 'impair', 'impairs', 'impaired', 'prevent', 'prevents', 'prevented', 'negative', 'loss'}
tok = lambda s: re.findall(r"[a-z0-9']+", s.lower())
nums = lambda s: set(re.findall(r'\d+(?:\.\d+)?', s))

# ------------------------------------------------------------------------------------------- data and verifiers
def load():
    corpus = {d['doc_id']: d for d in map(json.loads, open(os.path.join(DATA, 'corpus.jsonl')))}
    def pairs(split):
        out = []
        for c in map(json.loads, open(os.path.join(DATA, f'claims_{split}.jsonl'))):
            for d in c['cited_doc_ids']:
                ev = c['evidence'].get(str(d)); out.append(dict(cid=c['id'], claim=c['claim'], doc=d, y=LAB.index(ev[0]['label']) if ev else 2))
        return out
    return corpus, pairs('train'), pairs('dev')

def build_features(corpus, P):
    from sklearn.feature_extraction.text import TfidfVectorizer
    sents = [s for d in corpus.values() for s in d['abstract']] + [d['title'] for d in corpus.values()]
    tv = TfidfVectorizer(sublinear_tf=True, stop_words='english').fit(sents)            # unsupervised, corpus text only
    def cues(t):
        w = tok(t); return np.array([sum(x in NEG for x in w), sum(x in UP for x in w), sum(x in DOWN for x in w)], float)
    F1, F3, T2 = [], [], []
    for p in P:
        d = corpus[p['doc']]; S = d['abstract']; cv = tv.transform([p['claim']]); sv = tv.transform(S); cos = (sv @ cv.T).toarray().ravel()
        top = np.argsort(-cos)[:2]; tops = ' '.join(S[i] for i in top); cc, sc, tc = cues(p['claim']), cues(tops), cues(d['title'])
        tcos = float((tv.transform([d['title']]) @ cv.T).toarray()[0, 0]); nov = len(nums(p['claim']) & nums(tops)); ncl = len(nums(p['claim']))
        mism = cc[1] * sc[2] + cc[2] * sc[1]; par = float((cc[0] % 2) != (sc[0] % 2))
        F1.append([cos[top[0]], cos[top].mean(), cos.mean(), *cc, *sc, mism, par, nov, ncl - nov])
        F3.append([tcos, *cc, *tc, cc[1] * tc[2] + cc[2] * tc[1], float((cc[0] % 2) != (tc[0] % 2))])
        T2.append(p['claim'] + ' [SEP] ' + tops)
    return np.array(F1), np.array(F3), T2

def make_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    return [make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
            make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(C=1.0, max_iter=5000)),
            make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))]

# ------------------------------------------------------------------------------------------- channels and rules
def channel_bands(counts):
    """rows: true label; returns point, lo, hi per row (monitor.bands with n = row count)"""
    P, LO, HI = np.zeros(counts.shape), np.zeros(counts.shape), np.zeros(counts.shape)
    for y in range(counts.shape[0]):
        n = counts[y].sum(); P[y] = counts[y] / n; LO[y], HI[y] = bands(P[y], n)
    return P, LO, HI

def credal_decide(plo, phi, Llo, Lhi, kappa=KAPPA, tau=None):
    """plo/phi: prior band; Llo/Lhi: likelihood bounds per label for the observed outputs. Returns asserted label or None, best
    worst-case label, kappa*, hedge set, lower posterior of the best label."""
    lo, hi = plo * Llo, phi * Lhi; best = int(np.argmax(lo)); ks = min((lo[best] / hi[z] if hi[z] > 0 else np.inf) for z in range(3) if z != best)
    lpost = lo[best] / (lo[best] + sum(hi[z] for z in range(3) if z != best)) if lo[best] > 0 else 0.0
    act = ks >= kappa and (tau is None or lpost >= tau)
    hedge = [z for z in range(3) if not any(w != z and lo[w] >= kappa * hi[z] for w in range(3))]
    return (best if act else None), best, float(ks), hedge, float(lpost)

def rules(o, ch, prior):
    """o: tuple of three outputs; ch: dict with per-verifier (P, LO, HI) and joint (P, LO, HI) over 27 cells"""
    cell = o[0] * 9 + o[1] * 3 + o[2]; (pp, plo, phi) = prior
    nb = pp * np.prod([ch['v'][v][0][:, o[v]] for v in range(3)], axis=0); post = nb / nb.sum()
    L4lo = np.prod([ch['v'][v][1][:, o[v]] for v in range(3)], axis=0); L4hi = np.prod([ch['v'][v][2][:, o[v]] for v in range(3)], axis=0)
    L5lo, L5hi = ch['j'][1][:, cell], ch['j'][2][:, cell]
    L6lo = np.maximum(0, sum(ch['v'][v][1][:, o[v]] for v in range(3)) - 2); L6hi = np.min([ch['v'][v][2][:, o[v]] for v in range(3)], axis=0)
    return post, (L4lo, L4hi), (L5lo, L5hi), (L6lo, L6hi), cell

def aurc(conf, correct):
    idx = np.argsort(-np.asarray(conf), kind='stable'); c = np.asarray(correct)[idx]; risk = np.cumsum(1 - c) / np.arange(1, len(c) + 1); return float(risk.mean())
def auc(score, correct):
    from scipy.stats import rankdata
    s, c = np.asarray(score, float), np.asarray(correct, bool)
    if c.all() or (~c).all(): return np.nan
    r = rankdata(s); return float((r[c].sum() - c.sum() * (c.sum() + 1) / 2) / (c.sum() * (~c).sum()))

# ------------------------------------------------------------------------------------------- planted checks
def checks():
    rng = np.random.default_rng(1); ok = {}
    # (a) Frechet bounds contain the true joint likelihood for random couplings with the given margins
    viol = 0
    for _ in range(300):
        J = rng.dirichlet(np.ones(27) * .3).reshape(3, 3, 3); M = [J.sum((1, 2)), J.sum((0, 2)), J.sum((0, 1))]
        for o in itertools.product(range(3), repeat=3):
            lo = max(0, sum(M[v][o[v]] for v in range(3)) - 2); hi = min(M[v][o[v]] for v in range(3)); viol += int(not (lo - 1e-12 <= J[o] <= hi + 1e-12))
    ok['frechet_contains_truth'] = viol == 0
    # (b) with point bands (huge n) the credal rule acts exactly when the naive-Bayes posterior ratio clears kappa
    agree = 0
    for _ in range(300):
        pr = rng.dirichlet(np.ones(3)); L = rng.uniform(.01, 1, 3); post = pr * L; b = int(np.argmax(post)); r = min(post[b] / post[z] for z in range(3) if z != b)
        a, *_ = credal_decide(pr, pr, L, L, kappa=1.1); agree += int((a is not None) == (r >= 1.1))
    ok['credal_reduces_to_bayes'] = agree == 300
    # (c) bands contain the generating channel row in >= 90% of simulated rows (n = 200)
    cov = 0
    for _ in range(300):
        p = rng.dirichlet(np.ones(3)); c = rng.multinomial(200, p); lo, hi = bands(c / 200, 200); cov += int(((lo <= p) & (p <= hi)).all())
    ok['band_coverage'] = cov / 300 >= 0.9
    # (d) independent verifiers: the independence product lies inside the Frechet bounds
    viol = 0
    for _ in range(300):
        M = [rng.dirichlet(np.ones(3)) for _ in range(3)]
        for o in itertools.product(range(3), repeat=3):
            p = np.prod([M[v][o[v]] for v in range(3)]); viol += int(not (max(0, sum(M[v][o[v]] for v in range(3)) - 2) - 1e-12 <= p <= min(M[v][o[v]] for v in range(3)) + 1e-12))
    ok['product_inside_frechet'] = viol == 0
    os.makedirs(os.path.join(H, 'results'), exist_ok=True)
    open(os.path.join(H, 'results', 'scifact_checks.txt'), 'w').write(('PASS ' if all(ok.values()) else 'FAIL ') + json.dumps(ok)); print(ok)

# ------------------------------------------------------------------------------------------- the run
def run():
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(SEED); corpus, TR, DV = load()
    F1t, F3t, T2t = build_features(corpus, TR); F1d, F3d, T2d = build_features(corpus, DV)
    yt, yd = np.array([p['y'] for p in TR]), np.array([p['y'] for p in DV]); gt = np.array([p['cid'] for p in TR])
    Xt, Xd = [F1t, T2t, F3t], [F1d, T2d, F3d]; cvp = np.zeros((3, len(TR)), int)
    for tr, te in GroupKFold(5).split(F1t, yt, gt):
        for v, m in enumerate(make_models()):
            Xa = Xt[v]; m.fit(Xa[tr] if v != 1 else [Xa[i] for i in tr], yt[tr]); cvp[v, te] = m.predict(Xa[te] if v != 1 else [Xa[i] for i in te])
    dvp = np.zeros((3, len(DV)), int)
    for v, m in enumerate(make_models()): m.fit(Xt[v], yt); dvp[v] = m.predict(Xd[v])
    cvacc = [float((cvp[v] == yt).mean()) for v in range(3)]; best_v = int(np.argmax(cvacc))
    C = [np.array([[np.sum((yt == y) & (cvp[v] == o)) for o in range(3)] for y in range(3)], float) for v in range(3)]
    cells = cvp[0] * 9 + cvp[1] * 3 + cvp[2]; JC = np.array([[np.sum((yt == y) & (cells == k)) for k in range(27)] for y in range(3)], float)
    ch = dict(v=[channel_bands(c) for c in C], j=channel_bands(JC))
    pc = np.bincount(yt, minlength=3).astype(float); plo, phi = bands(pc / pc.sum(), pc.sum()); prior = (pc / pc.sum(), plo, phi)
    rows = []
    for i, p in enumerate(DV):
        o = tuple(int(dvp[v, i]) for v in range(3)); post, b4, b5, b6, cell = rules(o, ch, prior)
        vote = np.bincount(list(o), minlength=3); r2 = int(np.argmax(vote)) if vote.max() >= 2 else o[best_v]
        rec = dict(cid=p['cid'], doc=p['doc'], y=int(p['y']), o=o, r1=o[best_v], r2=r2, r3=int(np.argmax(post)), r3conf=float(post.max()))
        for name, (Llo, Lhi) in (('r4', b4), ('r5', b5), ('r6', b6)):
            a, best, ks, hedge, lp = credal_decide(plo, phi, Llo, Lhi); af, *_ = credal_decide(plo, phi, Llo, Lhi, tau=TAU)
            rec.update({name: a, name + 'best': best, name + 'k': ks, name + 'hedge': hedge, name + 'floor': af})
        # weakest link for R4: per-source log-likelihood-ratio contribution of the asserted label against its closest rival
        if rec['r4'] is not None:
            y4 = rec['r4']; lo4, hi4 = plo * b4[0], phi * b4[1]; rv = min((z for z in range(3) if z != y4), key=lambda z: lo4[y4] / hi4[z] if hi4[z] > 0 else np.inf)
            contrib = [np.log(max(ch['v'][v][1][y4, o[v]], 1e-12) / max(ch['v'][v][2][rv, o[v]], 1e-12)) for v in range(3)]; rec['r4weak'] = int(np.argmin(contrib))
        rec['thin'] = bool(JC[rec['r5best'], cell] < 5)
        rows.append(rec)
    # ---------------- metrics
    y = np.array([r['y'] for r in rows]); cid = np.array([r['cid'] for r in rows]); claims = np.unique(cid); groups = [np.where(cid == c)[0] for c in claims]
    def macro_f1(pred):
        f = []
        for k in range(3):
            tp = np.sum((pred == k) & (y == k)); fp = np.sum((pred == k) & (y != k)); fn = np.sum((pred != k) & (y == k)); f.append(2 * tp / max(2 * tp + fp + fn, 1))
        return float(np.mean(f))
    base = {}
    for r in ('r1', 'r2', 'r3'):
        pr = np.array([q[r] for q in rows]); base[r] = dict(accuracy=round(float((pr == y).mean()), 4), macro_f1=round(macro_f1(pr), 4))
    def acted(name, key=None):
        key = key or name; a = np.array([q[key] is not None for q in rows]); c = np.array([q[key] is not None and q[key] == q['y'] for q in rows]); return a, c
    cred = {}
    for name in ('r4', 'r5', 'r6'):
        for key, lab in ((name, 'kappa'), (name + 'floor', 'kappa+floor')):
            a, c = acted(name, key); cred[f'{name} {lab}'] = dict(coverage=round(float(a.mean()), 4), acted=int(a.sum()), accuracy_when_acting=round(float(c[a].mean()), 4) if a.any() else None)
    r3c = np.array([q['r3'] == q['y'] for q in rows]); a5, c5 = acted('r5'); a4, c4 = acted('r4')
    k5 = np.array([q['r5k'] for q in rows]); c5best = np.array([q['r5best'] == q['y'] for q in rows]); r3conf = np.array([q['r3conf'] for q in rows])
    only4 = a4 & ~a5
    def stats(idx):
        s = {}
        s['H1'] = c5[idx][a5[idx]].mean() - r3c[idx].mean() if a5[idx].any() else np.nan
        s['H2'] = (c5[idx][a5[idx]].mean() - c4[idx][only4[idx]].mean()) if (a5[idx].any() and only4[idx].any()) else np.nan
        s['H3'] = auc(k5[idx], c5best[idx]); s['H4'] = aurc(r3conf[idx], r3c[idx]) - aurc(k5[idx], c5best[idx]); return s
    point = stats(np.arange(len(rows))); boots = {k: [] for k in point}
    for _ in range(2000):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]); s = stats(idx)
        for k in s: boots[k].append(s[k])
    ci = {k: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)] for k, v in boots.items()}
    hyp = dict(H1=dict(value=round(float(point['H1']), 4), ci=ci['H1'], holds=bool(ci['H1'][0] > 0)),
               H2=dict(value=None if np.isnan(point['H2']) else round(float(point['H2']), 4), ci=ci['H2'], n_r4_only=int(only4.sum()),
                       underpowered=bool(only4.sum() < 20), holds=bool(only4.sum() >= 20 and ci['H2'][0] > 0)),
               H3=dict(value=round(float(point['H3']), 4), ci=ci['H3'], holds=bool(ci['H3'][0] > 0.5)),
               H4=dict(value=round(float(point['H4']), 4), aurc_r3=round(aurc(r3conf, r3c), 4), aurc_r5=round(aurc(k5, c5best), 4), ci=ci['H4'], holds=bool(ci['H4'][0] > 0)))
    # descriptive: gluing conflicts across abstracts of the same claim; weakest link; thin support
    conf = 0; multi = 0
    for c in claims:
        g = [rows[i] for i in np.where(cid == c)[0]]
        if len(g) > 1:
            multi += 1; d = {q['r5'] for q in g if q['r5'] is not None}; conf += int(0 in d and 1 in d)
    wl = {}
    for q in rows:
        if q['r4'] is not None: k = ('correct' if q['r4'] == q['y'] else 'incorrect'); wl.setdefault(k, [0, 0, 0]); wl[k][q['r4weak']] += 1
    desc = dict(train_cv_accuracy=[round(a, 4) for a in cvacc], best_single=['V1 sentence cues', 'V2 lexical', 'V3 title'][best_v],
                verifier_agreement_dev=round(float(np.mean([len(set(q['o'])) == 1 for q in rows])), 4),
                multi_abstract_claims=multi, decided_support_contradict_conflicts=conf, r4_weakest_link_by_verifier=wl,
                thin_support_share=round(float(np.mean([q['thin'] for q in rows])), 4), dev_pairs=len(rows), dev_label_counts=np.bincount(y, minlength=3).tolist())
    res = dict(plan='PREANALYSIS_SCIFACT.md', baselines=base, credal=cred, hypotheses=hyp, descriptive=desc,
               channels=dict(per_verifier=[c.astype(int).tolist() for c in C], prior_counts=pc.astype(int).tolist()))
    json.dump(res, open(os.path.join(H, 'results', 'scifact_results.json'), 'w'), indent=1)
    json.dump([{k: (list(v) if isinstance(v, tuple) else v) for k, v in q.items()} for q in rows], open(os.path.join(H, 'results', 'scifact_dev_decisions.json'), 'w'))
    print(json.dumps({k: res[k] for k in ('baselines', 'credal', 'hypotheses', 'descriptive')}, indent=1))

if __name__ == '__main__':
    checks() if sys.argv[1:] == ['checks'] else run()
