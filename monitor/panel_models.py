"""Measurement models for the GSS three-wave panels (welfare item, each wording; complete cases, unweighted).

A positive gluing gap I(X1; X3 | X2) of the observed answers can come from real change that is not Markov, or from measurement error
around a latent attitude (Converse 1964; Wiley and Wiley 1970; Achen 1975). Three latent-class models, K = 3 classes, are fitted by EM
(30 random starts) to the 27-cell table and tested by G2 against the saturated model (26 free cells):

  stable      the latent class is fixed over the waves; answers come from one response law P(X | Z) at every wave     8 params, df 18
  drifting    the latent class is fixed; each wave has its own response law P(X_t | Z)                                20 params, df 6
  markov      the latent class follows a Markov chain Z1 -> Z2 -> Z3; one response law at every wave (latent Markov)   20 params, df 6

All three are generically identified with three waves of three-valued answers (Kruskal's condition 3 + 3 + 3 >= 2 * 3 + 2 for the
conditionally independent models). For each fit we report G2, df, the chi-square p-value, a parametric-bootstrap p-value (100 draws,
because some expected counts are below 5), and the gluing gap the fitted law implies. A 'markov' fit means the panel is consistent with
latent change that composes (is glued at the latent level) seen through unreliable answers.
"""
import os, sys, json
import numpy as np, pandas as pd, pyreadr
from scipy.stats import chi2
H = os.path.dirname(os.path.abspath(__file__)); G = os.path.join(H, 'data', 'gss'); K = 3

def joint(model, th):
    if model == 'stable':
        pi, R = th; return np.einsum('z,za,zb,zc->abc', pi, R, R, R)
    if model == 'drifting':
        pi, R1, R2, R3 = th; return np.einsum('z,za,zb,zc->abc', pi, R1, R2, R3)
    pi, T1, T2, R = th; return np.einsum('x,xy,yz,xa,yb,zc->abc', pi, T1, T2, R, R, R)

def em(N, model, rng, iters=4000):
    n = N.sum(); D = lambda *s: rng.dirichlet(np.ones(s[-1]), size=s[:-1]) if len(s) > 1 else rng.dirichlet(np.ones(s[0]))
    th = {'stable': lambda: [D(K), D(K, 3)], 'drifting': lambda: [D(K), D(K, 3), D(K, 3), D(K, 3)],
          'markov': lambda: [D(K), D(K, K), D(K, K), D(K, 3)]}[model]()
    old = -np.inf
    for _ in range(iters):
        if model == 'markov':
            pi, T1, T2, R = th
            J = np.einsum('x,xy,yz,xa,yb,zc->xyzabc', pi, T1, T2, R, R, R); P = J.sum((0, 1, 2))
            W = J / np.maximum(P, 1e-300) * N                                   # expected counts over (z1, z2, z3, a, b, c)
            pi = W.sum((1, 2, 3, 4, 5)) / n
            T1 = W.sum((2, 3, 4, 5)); T1 /= T1.sum(1, keepdims=True)
            T2 = W.sum((0, 3, 4, 5)); T2 /= T2.sum(1, keepdims=True)
            R = W.sum((1, 2, 4, 5)) + W.sum((0, 2, 3, 5)) + W.sum((0, 1, 3, 4)); R /= R.sum(1, keepdims=True)
            th = [pi, T1, T2, R]
        else:
            Rs = [th[1]] * 3 if model == 'stable' else th[1:]
            J = np.einsum('z,za,zb,zc->zabc', th[0], *Rs); P = J.sum(0); W = J / np.maximum(P, 1e-300) * N
            pi = W.sum((1, 2, 3)) / n; R1, R2, R3 = W.sum((2, 3)), W.sum((1, 3)), W.sum((1, 2))
            if model == 'stable':
                R = R1 + R2 + R3; th = [pi, R / R.sum(1, keepdims=True)]
            else:
                th = [pi] + [r / r.sum(1, keepdims=True) for r in (R1, R2, R3)]
        P = joint(model, th); ll = float((N[N > 0] * np.log(P[N > 0])).sum())
        if ll - old < 1e-10: break
        old = ll
    return ll, th

def fit(N, model, seed=0, starts=30):
    rng = np.random.default_rng(seed); best = max((em(N, model, rng) for _ in range(starts)), key=lambda r: r[0])
    return best[1], joint(model, best[1])

def G2(N, P):
    m = N > 0; return 2 * float((N[m] * np.log(N[m] / (N.sum() * P[m]))).sum())

def gap(P):
    P2 = P.sum((0, 2)); g = 0.0
    for b in range(3):
        Pb = P[:, b, :] / P2[b]; pa, pc = Pb.sum(1), Pb.sum(0)
        g += P2[b] * sum(Pb[a, c] * np.log2(Pb[a, c] / (pa[a] * pc[c])) for a in range(3) for c in range(3) if Pb[a, c] > 0)
    return float(g)

DF = {'stable': 26 - 8, 'drifting': 26 - 20, 'markov': 26 - 20}   # valid for interior fits (all class shares and transitions > 0)
CODE = {'Too Little': 1, 'About Right': 2, 'Too Much': 3}
if __name__ == '__main__':
    out = {}
    for pnl in ('06', '08', '10'):
        p = list(pyreadr.read_r(os.path.join(G, f'gss_panel{pnl}_long.rda')).values())[0]
        p['firstid'] = p['firstid'].astype(str)
        for item in ('natfare', 'natfarey'):
            p[item] = p[item].astype(object).map(CODE).astype(float)
            w = p.pivot_table(index='firstid', columns='wave', values=item, aggfunc='first')
            w = w[w.isin([1, 2, 3]).all(axis=1)].astype(int)
            N = np.zeros((3, 3, 3)); np.add.at(N, (w[1].values - 1, w[2].values - 1, w[3].values - 1), 1); n = int(N.sum())
            res = dict(n=n, margins=[[round(float(x), 3) for x in N.sum(tuple(j for j in range(3) if j != t)) / n] for t in range(3)], gap_observed=round(gap(N / n), 4))
            for model in ('stable', 'drifting', 'markov'):
                th, P = fit(N, model, seed=int(pnl)); g2 = G2(N, P)
                rng = np.random.default_rng(100 + int(pnl)); bs = []
                for _ in range(100):
                    Nb = rng.multinomial(n, P.ravel()).reshape(3, 3, 3); bs.append(G2(Nb, fit(Nb, model, seed=int(rng.integers(1e9)), starts=4)[1]))
                res[model] = dict(G2=round(g2, 2), df=DF[model], p=round(float(chi2.sf(g2, DF[model])), 4), p_boot=round(float(np.mean(np.array(bs) >= g2)), 3),
                                  gap_implied=round(gap(P), 4), min_class=round(float(min(th[0])), 3))
            out[f'{pnl}/{item}'] = res
            print(pnl, item, n, 'gap', res['gap_observed'], ' | '.join(f"{m}: G2 {res[m]['G2']} df {res[m]['df']} p {res[m]['p']} boot {res[m]['p_boot']} gap {res[m]['gap_implied']}" for m in ('stable', 'drifting', 'markov')), flush=True)
    json.dump(out, open(os.path.join(H, 'panel_models.json'), 'w'), indent=1)
