"""Interchange test of PREANALYSIS_BANDING_INTERCHANGE.md: does banding the PHQ-9 total create the composition failure?
Algebras Prop. 6.1 with the banding merge b as the middle kernel. Usage: python3 banding_interchange.py uk|ie -> results/banding_<strand>.json"""
import os, sys, json, tempfile, numpy as np, pandas as pd
from scipy.stats import chi2
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, '..', 'open_clinical', 'lib'))
import c19
from monitor import gluing_gap
SEED, B = 20261010, 400
c19.band = lambda x: pd.to_numeric(x, errors='coerce').where(lambda v: (v >= 0) & (v <= 27))   # raw totals: same range rule, no banding

def wide_raw(W):
    wide = None
    for t in (1, 2, 3):
        x = W[t][['pid', 'D']].dropna().drop_duplicates('pid').rename(columns={'D': f'D{t}'}); wide = x if wide is None else wide.merge(x, on='pid', how='outer')
    w = wide.dropna(subset=['D1', 'D2', 'D3']); return [w[f'D{t}'].astype(int).values for t in (1, 2, 3)]

NRAW = 28                                                                  # raw states 0..27 (set_merge changes it, e.g. for ICU pressure bins)
band = lambda v: np.digitize(v, [5, 10])
bmat = np.array([[1.0 if band(np.array([s]))[0] == j else 0.0 for j in range(3)] for s in range(NRAW)])   # the merge b: raw -> 3
def set_merge(nraw, cuts):
    global NRAW, band, bmat
    NRAW = nraw; band = lambda v: np.digitize(v, cuts)
    bmat = np.array([[1.0 if band(np.array([s]))[0] == j else 0.0 for j in range(3)] for s in range(NRAW)])
def joint(a, b_, na, nb): T = np.zeros((na, nb)); np.add.at(T, (a, b_), 1); return T / T.sum()
def glue(al, be):
    mid = al.sum(0); inv = np.where(mid > 0, 1 / np.where(mid > 0, mid, 1), 0); return al @ np.diag(inv) @ be

def decompose(x1, x2, x3):
    al, be, ga = joint(x1, x2, NRAW, NRAW), joint(x2, x3, NRAW, NRAW), joint(x1, x3, NRAW, NRAW)
    O = bmat.T @ ga @ bmat; Lm = bmat.T @ glue(al, be) @ bmat; R = glue(bmat.T @ al @ bmat, bmat.T @ be @ bmat)
    tot, raw, inter = O - R, O - Lm, Lm - R
    return dict(identity_err=float(np.abs(tot - raw - inter).max()), tv_total=0.5 * np.abs(tot).sum(), tv_raw=0.5 * np.abs(raw).sum(),
                tv_interchange=0.5 * np.abs(inter).sum(), share_interchange=float((tot * inter).sum() / max((tot * tot).sum(), 1e-300)),
                interchange_is_circulation=bool(np.abs(inter.sum(0)).max() < 1e-12 and np.abs(inter.sum(1)).max() < 1e-12))

def sufficiency(x1, x2, x3):
    """within each band of D2, homogeneity of the law of D1 (and of D3) across the raw D2 values: chi-square per band, Fisher-combined"""
    out = {}
    for name, y in (('D1', x1), ('D3', x3)):
        ps = []
        for bb in range(3):
            m = band(x2) == bb; t = pd.crosstab(x2[m], band(y[m]))
            t = t.loc[t.sum(axis=1) > 0, t.sum(axis=0) > 0]
            if t.shape[0] > 1 and t.shape[1] > 1:
                e = np.outer(t.sum(axis=1), t.sum(axis=0)) / t.values.sum(); X2 = ((t.values - e) ** 2 / e).sum(); ps.append(chi2.sf(X2, (t.shape[0] - 1) * (t.shape[1] - 1)))
        F = -2 * np.log(ps).sum(); out[name] = dict(per_band_p=[round(float(p), 4) for p in ps], fisher_p=float(chi2.sf(F, 2 * len(ps))))
    return out

def run(x1, x2, x3, rng):
    n = len(x1); obs = decompose(x1, x2, x3); gb = gluing_gap(band(x1), band(x2), band(x3)); gr = gluing_gap(x1, x2, x3)
    K = lambda a, c, na, nc: np.nan_to_num(pd.crosstab(a, c).reindex(index=range(na), columns=range(nc), fill_value=0).pipe(lambda t: t.div(t.sum(axis=1), axis=0)).values)
    K12, K23 = K(x1, x2, NRAW, NRAW), K(x2, x3, NRAW, NRAW)
    b1, b2_, b3 = band(x1), band(x2), band(x3); K12b, K23b = K(b1, b2_, 3, 3), K(b2_, b3, 3, 3)
    def draw(Kr, v):
        cum = Kr.cumsum(1); return np.minimum((rng.random(len(v))[:, None] > cum[v]).sum(1), Kr.shape[1] - 1)
    null_raw_gap, null_raw_tv, null_band_gap = [], [], []
    for _ in range(B):
        a = rng.choice(x1, n); b_ = draw(K12, a); c = draw(K23, b_)
        null_raw_gap.append(gluing_gap(band(a), band(b_), band(c))); null_raw_tv.append(decompose(a, b_, c)['tv_raw'])
        a = rng.choice(b1, n); b_ = draw(K12b, a); c = draw(K23b, b_); null_band_gap.append(gluing_gap(a, b_, c))
    q = lambda v: round(float(np.percentile(v, 95)), 4); p = lambda v, o: float((np.sum(np.array(v) >= o) + 1) / (B + 1))
    res = dict(n=n, banded_gap_bits=round(gb, 4), raw_gap_bits=round(gr, 4),
               banded_markov_null95=q(null_band_gap), p_banded_markov=p(null_band_gap, gb),
               raw_markov_null95_of_banded_gap=q(null_raw_gap), raw_markov_median_of_banded_gap=round(float(np.median(null_raw_gap)), 4), p_raw_markov=p(null_raw_gap, gb),
               decomposition={k: (round(v, 4) if isinstance(v, float) else v) for k, v in obs.items()},
               tv_raw_null95=q(null_raw_tv), p_tv_raw=p(null_raw_tv, obs['tv_raw']), sufficiency=sufficiency(x1, x2, x3))
    res['decision_primary'] = ('banded failure explained by banding a raw Markov chain (interchange defect)' if gb <= np.percentile(null_raw_gap, 95)
                               else 'raw chain has memory that banding does not explain')
    res['decision_secondary'] = 'raw composition failure not detected' if obs['tv_raw'] <= np.percentile(null_raw_tv, 95) else 'raw composition failure detected'
    return res

if __name__ == '__main__':
    strand = sys.argv[1]; rng = np.random.default_rng(SEED)
    with tempfile.TemporaryDirectory() as tmp:
        W = c19.load_uk(os.path.join(H, 'raw', 'uk'))[0] if strand == 'uk' else c19.load_ie(os.path.join(H, 'raw', 'ie_archive.zip'), tmp)[0]
    x1, x2, x3 = wide_raw(W); res = dict(strand=strand, plan='PREANALYSIS_BANDING_INTERCHANGE.md (2026-10-10)', **run(x1, x2, x3, rng))
    json.dump(res, open(os.path.join(H, 'results', f'banding_{strand}.json'), 'w'), indent=1); print(json.dumps(res, indent=1))
