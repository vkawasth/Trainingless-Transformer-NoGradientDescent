"""Layer 1 on the GSS panels: composition of kernels and path dependence of outcomes.

For the welfare item (each wording), complete three-wave cases, unweighted:
  * the kernels K12, K23, K13 (row-stochastic transition matrices of the answer) and the composite K12 K23; their difference row by row
    in total variation. K13 = K12 K23 holds exactly when ... it is implied by the gluing (Markov) law, and the gluing gap measures the
    joint failure; the kernel difference is the failure visible in the wave-1 -> wave-3 transition alone.
  * path dependence of outcomes: does the ordered path (x1, x2, x3) predict the 2008 vote (Obama / McCain) and wave-3 party
    identification (Democrat / Independent / Republican, leaners with their party) beyond (a) the static wave-1 answer and (b) the
    order-free summary of the same three answers (the multiset {x1, x2, x3})? Cell-mean models, exact conditional permutation tests
    (outcomes permuted within strata of the smaller model's cells; 2000 permutations), and leave-one-out log loss in bits with
    Jeffreys-smoothed cell means. Comparing (path) with (multiset) isolates the order of the answers, holding fixed how many times each
    answer was given, so measurement error that averages out cannot masquerade as path dependence.
Writes panel_paths.json."""
import os, sys, json
import numpy as np, pandas as pd, pyreadr
H = os.path.dirname(os.path.abspath(__file__)); G = os.path.join(H, 'data', 'gss')
CODE = {'Too Little': 1, 'About Right': 2, 'Too Much': 3}
PARTY = {'Strong Democrat': 0, 'Not Str Democrat': 0, 'Ind,near Dem': 0, 'Independent': 1, 'Ind,near Rep': 2, 'Not Str Republican': 2, 'Strong Republican': 2}

def G2(y, cells, K):
    """Likelihood-ratio statistic of the cell-mean model against the intercept model."""
    tab = pd.crosstab(cells, y).reindex(columns=range(K), fill_value=0).values.astype(float)
    n = tab.sum(); rows = tab.sum(1, keepdims=True); cols = tab.sum(0, keepdims=True); m = tab > 0
    return 2 * float((tab[m] * np.log(tab[m] * n / (rows @ cols)[m])).sum())

def loo_bits(y, cells, K, a=0.5):
    """Leave-one-out log loss in bits of the cell-mean model with Dirichlet(a) smoothing."""
    df = pd.DataFrame({'c': cells, 'y': y}); cnt = df.groupby(['c', 'y']).size().unstack(fill_value=0).reindex(columns=range(K), fill_value=0)
    tot = cnt.sum(axis=1); ll = 0.0
    for c, yy in zip(cells, y):
        ll += np.log2((cnt.loc[c, yy] - 1 + a) / (tot[c] - 1 + K * a))
    return -ll / len(y)

def perm_test(y, big, small, K, rng, B=2000):
    obs = G2(y, big, K) - G2(y, small, K); y = np.asarray(y); small = np.asarray(small); cnt = 0
    strata = [np.where(small == s)[0] for s in np.unique(small)]
    for _ in range(B):
        yp = y.copy()
        for ix in strata: yp[ix] = y[rng.permutation(ix)]
        cnt += (G2(yp, big, K) - G2(yp, small, K)) >= obs - 1e-12
    return obs, (cnt + 1) / (B + 1)

if __name__ == '__main__':
    rng = np.random.default_rng(5); out = {}; pooled = {o: [] for o in ('vote', 'party')}
    for pnl in ('06', '08', '10'):
        p = list(pyreadr.read_r(os.path.join(G, f'gss_panel{pnl}_long.rda')).values())[0]
        p['firstid'] = p['firstid'].astype(str)
        p['vote'] = p['pres08'].astype(object).map({'Obama': 0, 'Mccain': 1})
        p['party'] = p['partyid'].astype(object).map(PARTY)
        q = p.pivot_table(index='firstid', columns='wave', values=['vote', 'party'], aggfunc='first')
        vote = q['vote'].bfill(axis=1).iloc[:, 0]; party3 = q['party'][3] if 3 in q['party'].columns else None
        for item in ('natfare', 'natfarey'):
            p[item + '_c'] = p[item].astype(object).map(CODE)
            w = p.pivot_table(index='firstid', columns='wave', values=item + '_c', aggfunc='first')
            w = w[w.isin([1, 2, 3]).all(axis=1)].astype(int)
            x1, x2, x3 = w[1].values - 1, w[2].values - 1, w[3].values - 1; n = len(w)
            K12 = pd.crosstab(x1, x2, normalize='index').values; K23 = pd.crosstab(x2, x3, normalize='index').values
            K13 = pd.crosstab(x1, x3, normalize='index').values; comp = K12 @ K23
            res = dict(n=n, K12=K12.round(4).tolist(), K23=K23.round(4).tolist(), K13=K13.round(4).tolist(), K12K23=comp.round(4).tolist(),
                       tv_rows=(0.5 * np.abs(K13 - comp).sum(1)).round(4).tolist(),
                       stay_direct=np.diag(K13).round(4).tolist(), stay_composite=np.diag(comp).round(4).tolist())
            path = x1 * 9 + x2 * 3 + x3
            mset = np.array([''.join(map(str, sorted(t))) for t in zip(x1, x2, x3)])
            for oname, ys in (('vote', vote), ('party', party3)):
                ys = ys.reindex(w.index); ok = ys.notna().values
                if ok.sum() < 100: continue
                y = ys.values[ok].astype(int); K = int(y.max()) + 1
                c_path, c_set, c_x1 = path[ok], mset[ok], x1[ok]
                g_set_x1, p_set_x1 = perm_test(y, c_set, c_x1, K, rng)
                g_path_set, p_path_set = perm_test(y, c_path, c_set, K, rng)
                base = loo_bits(y, np.zeros(len(y), int), K)
                res[oname] = dict(n=int(ok.sum()), bits_intercept=round(base, 4), bits_x1=round(loo_bits(y, c_x1, K), 4),
                                  bits_multiset=round(loo_bits(y, c_set, K), 4), bits_path=round(loo_bits(y, c_path, K), 4),
                                  G_multiset_vs_x1=round(g_set_x1, 2), p_multiset_vs_x1=round(p_set_x1, 4),
                                  G_path_vs_multiset=round(g_path_set, 2), p_path_vs_multiset=round(p_path_set, 4))
                pooled[oname].append((y, c_path, c_set, c_x1, f'{pnl}{item}'))
            out[f'{pnl}/{item}'] = res
            print(pnl, item, n, 'TV(K13, K12K23) by row', res['tv_rows'],
                  '|', ' | '.join(f"{o}: bits x1 {res[o]['bits_x1']} set {res[o]['bits_multiset']} path {res[o]['bits_path']} (intercept {res[o]['bits_intercept']}); "
                                  f"p(set|x1) {res[o]['p_multiset_vs_x1']} p(path|set) {res[o]['p_path_vs_multiset']}" for o in ('vote', 'party') if o in res), flush=True)
    # pooled tests: strata are (series, smaller-model cell)
    for oname, parts in pooled.items():
        y = np.concatenate([a[0] for a in parts]); K = int(y.max()) + 1
        tag = np.concatenate([[a[4]] * len(a[0]) for a in parts])
        cp = np.char.add(tag.astype(str), np.concatenate([a[1] for a in parts]).astype(str))
        cs = np.char.add(tag.astype(str), np.concatenate([a[2] for a in parts]).astype(str))
        c1 = np.char.add(tag.astype(str), np.concatenate([a[3] for a in parts]).astype(str))
        g1, p1 = perm_test(y, cs, c1, K, rng); g2, p2 = perm_test(y, cp, cs, K, rng)
        out['pooled_' + oname] = dict(n=len(y), G_multiset_vs_x1=round(g1, 2), p_multiset_vs_x1=round(p1, 4), G_path_vs_multiset=round(g2, 2), p_path_vs_multiset=round(p2, 4),
                                      bits_x1=round(loo_bits(y, c1, K), 4), bits_multiset=round(loo_bits(y, cs, K), 4), bits_path=round(loo_bits(y, cp, K), 4),
                                      bits_series=round(loo_bits(y, tag, K), 4))
        print('pooled', oname, out['pooled_' + oname])
    json.dump(out, open(os.path.join(H, 'panel_paths.json'), 'w'), indent=1)
