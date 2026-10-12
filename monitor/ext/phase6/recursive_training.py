"""Section 4 of PREANALYSIS_COLLAPSE.md: recursive training with known truth. A conditional categorical model per (context, information)
cell, retrained on its own samples for 20 generations, with fresh real data shares 0, 5, 10, 25%, optionally through a paraphrase
kernel. Tests predictions 1-5. Usage: python3 recursive_training.py  -> results/recursive_training.json"""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr, rankdata
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(H, '..', '..')))
from monitor import bands
SEED = 20261011; C, X, Y, NCELL, GEN, REPS, NHOLD = 3, 6, 8, 200, 20, 50, 500
LAMS = [0.0, 0.05, 0.10, 0.25]; LOG2 = np.log(2)

def truth(rng):
    Pxc = rng.dirichlet(np.ones(X) * 2, C); P0 = rng.dirichlet(np.ones(Y) * 0.4, X); return Pxc, P0
def kernel(tau):
    K = np.eye(Y) * (1 - tau)
    for y in range(Y): K[y, (y + 1) % Y] += tau / 2; K[y, (y - 1) % Y] += tau / 2
    return K
def dobrushin(K): return max(0.5 * np.abs(K[a] - K[b]).sum() for a in range(len(K)) for b in range(len(K)))
def fit(counts): return counts / counts.sum(-1, keepdims=True)
def sample(P, n, rng):          # P: (C, X, Y) laws; n samples per cell
    return np.array([[rng.multinomial(n, P[c, x]) for x in range(X)] for c in range(C)]).astype(float)
def distinguish(Q): return float(np.mean([0.5 * np.abs(Q[c, a] - Q[c, b]).sum() for c in range(C) for a in range(X) for b in range(a + 1, X)]))
def cmi_bits(Q, Pxc):
    """I(Y; C | X) of the model's law with the true P(x | c) and uniform P(c): distance to the toric variety Y _|_ C | X"""
    Pc = np.ones(C) / C; Pcx = Pc[:, None] * Pxc; Px = Pcx.sum(0); Pc_x = Pcx / Px[None, :]
    Qbar = np.einsum('cx,cxy->xy', Pc_x, Q); I = 0.0
    for c in range(C):
        for x in range(X):
            m = Q[c, x] > 0; I += Pcx[c, x] * (Q[c, x][m] * np.log(Q[c, x][m] / Qbar[x][m])).sum()
    return I / LOG2

def run_chain(P0full, lam, K, rng):
    """returns Q_g for g = 0..GEN, shape (GEN+1, C, X, Y)"""
    Q = [fit(sample(P0full, NCELL, rng))]; nf = int(round(lam * NCELL)); ns = NCELL - nf
    for g in range(1, GEN + 1):
        syn = sample(Q[-1] @ K if K is not None else Q[-1], ns, rng) if ns else 0
        fresh = sample(P0full, nf, rng) if nf else 0
        Q.append(fit(syn + fresh))
    return np.array(Q)

def certificate(Q, lo, hi, held):
    """mean over cells of: l1 distance from the model law to the held-out band box, plus held-out mass on categories the model gives 0"""
    out = np.maximum(lo - Q, 0) + np.maximum(Q - hi, 0); crom = (held * (Q == 0)).sum(-1); return float((out.sum(-1) + crom).mean())

def main():
    rng = np.random.default_rng(SEED); Pxc, P0 = truth(rng); P0full = np.broadcast_to(P0, (C, X, Y)).copy()
    tails = [(x, y) for x in range(X) for y in range(Y) if 0 < P0[x, y] <= 0.02]
    K = kernel(0.2); eta = dobrushin(K); etag = [dobrushin(np.linalg.matrix_power(K, g)) for g in range(GEN + 1)]
    res = dict(plan='PREANALYSIS_COLLAPSE.md section 4', settings=dict(C=C, X=X, Y=Y, n_per_cell=NCELL, generations=GEN, replicates=REPS, held_out=NHOLD,
               tails=len(tails), tail_probs=sorted(round(float(P0[x, y]), 4) for x, y in tails), eta_K=eta, eta_K_power=[round(e, 4) for e in etag]))
    chains = {lam: [] for lam in LAMS}; chainsK = []; held = []
    for r in range(REPS):
        rr = np.random.default_rng(SEED + 1000 + r); H0 = sample(P0full, NHOLD, rr) / NHOLD; held.append(H0)
        for lam in LAMS: chains[lam].append(run_chain(P0full, lam, None, np.random.default_rng(SEED + 10 * r + int(lam * 100))))
        chainsK.append(run_chain(P0full, 0.0, K, np.random.default_rng(SEED + 7777 + r)))
    # ---------- prediction 1: contraction (regime K, lambda 0)
    D = np.array([[distinguish(Q[g]) for g in range(GEN + 1)] for Q in chainsK]); Dm = D.mean(0)
    flat = np.broadcast_to(P0.mean(0), (C, X, Y)).copy(); Dn = np.array([[distinguish(Q[g]) for g in range(GEN + 1)] for Q in
                                                                          (run_chain(flat, 0.0, K, np.random.default_rng(SEED + 9000 + r)) for r in range(REPS))])
    s = np.percentile(Dn, 95, axis=0); D0 = distinguish(P0full)
    b_plan = eta ** np.arange(GEN + 1) * D0 + s; b_sharp = np.array(etag) * D0 + s
    res['P1_contraction'] = dict(eta=eta, D_true=round(D0, 4), mean_D=[round(v, 4) for v in Dm], bound_plan=[round(v, 4) for v in b_plan], null95=[round(v, 4) for v in s],
                                 holds=bool((Dm <= b_plan + 1e-12).all()), note='eta(K) = 1 for this cyclic kernel, so the planned bound reduces to D_g <= D_0 + s_g',
                                 exploratory_sharper_bound=dict(bound=[round(v, 4) for v in b_sharp], holds=bool((Dm <= b_sharp + 1e-12).all())))
    # ---------- prediction 2: breaks in finite time, forecast by break time (regime S, lambda 0)
    tracks = []; obs = []
    for Q in chains[0.0]:
        for c in range(C):
            for (x, y) in tails:
                q = Q[:, c, x, y]; z = np.where(q == 0)[0]; tb = int(z[0]) if len(z) else None
                absorbing = tb is not None and bool((q[tb:] == 0).all()); tracks.append(dict(tb=tb, absorbing=absorbing))
                if tb is not None:
                    for g in range(2, tb):
                        v = 0.5 * (abs(q[g] - q[g - 1]) + abs(q[g - 1] - q[g - 2]))
                        if v > 0: obs.append((q[g] / v, q[g], tb - g))
    broke = np.mean([t['tb'] is not None and t['absorbing'] for t in tracks]); O = np.array(obs)
    rho = spearmanr(O[:, 0], O[:, 2]); rx, rz, ry = rankdata(O[:, 0]), rankdata(O[:, 1]), rankdata(O[:, 2])
    res_x = rx - np.polyval(np.polyfit(rz, rx, 1), rz); res_y = ry - np.polyval(np.polyfit(rz, ry, 1), rz); pr = spearmanr(res_x, res_y)
    res['P2_breaks'] = dict(tracks=len(tracks), share_broken_absorbing_by_g20=round(float(broke), 4), share_broken_at_g0=round(float(np.mean([t['tb'] == 0 for t in tracks])), 4),
                            forecast_obs=len(O), spearman_breaktime_remaining=round(float(rho.correlation), 3), p=float(rho.pvalue),
                            spearman_level_remaining=round(float(spearmanr(O[:, 1], O[:, 2]).correlation), 3),
                            partial_spearman_given_level=round(float(pr.correlation), 3), partial_p=float(pr.pvalue),
                            holds=bool(broke >= 0.9 and rho.correlation > 0 and rho.pvalue < 0.01 and pr.correlation > 0 and pr.pvalue < 0.01))
    # ---------- prediction 3: fresh data prevents absorbing breaks
    P3 = {}
    for lam in LAMS:
        Qs = np.array(chains[lam]); z = (Qs == 0); run3 = (z[:, 2:] & z[:, 1:-1] & z[:, :-2]).any(1)      # (rep, C, X, Y)
        pos = np.broadcast_to(P0 > 0, run3.shape[1:]); share = float(run3[:, pos].mean())
        mean, se = Qs.mean(0), Qs.std(0, ddof=1) / np.sqrt(REPS); floor_ok = bool((mean >= lam * P0full[None] - 2 * se - 1e-12).all())
        P3[str(lam)] = dict(share_tracks_zero_3_consecutive=round(share, 4), floor_held=floor_ok,
                            share_among_categories_above_005=round(float(run3[:, np.broadcast_to(P0 > 0.005, run3.shape[1:])].mean()), 4))
    res['P3_fresh_data'] = dict(by_lambda=P3, holds=bool(all(P3[str(l)]['share_tracks_zero_3_consecutive'] <= 0.01 and P3[str(l)]['floor_held'] for l in LAMS if l >= 0.05)
                                                       and P3['0.0']['share_tracks_zero_3_consecutive'] > 0.01))
    # ---------- prediction 4: contexts drift off the invariance variety
    I = {lam: np.array([[cmi_bits(Q[g], Pxc) for g in range(GEN + 1)] for Q in chains[lam]]) for lam in LAMS}
    m0 = I[0.0].mean(0); tr = spearmanr(np.arange(GEN + 1), m0)
    res['P4_invariance'] = dict(mean_cmi_bits={str(l): [round(v, 4) for v in I[l].mean(0)] for l in LAMS}, trend_spearman=round(float(tr.correlation), 3),
                                share_g20_above_g1=round(float((I[0.0][:, 20] > I[0.0][:, 1]).mean()), 3),
                                share_lam25_below_lam0_at_g20=round(float((I[0.25][:, 20] < I[0.0][:, 20]).mean()), 3),
                                holds=bool(tr.correlation > 0 and (I[0.0][:, 20] > I[0.0][:, 1]).mean() >= 0.9 and (I[0.25][:, 20] < I[0.0][:, 20]).mean() >= 0.9))
    # ---------- prediction 5: certificates warn before accuracy drops (regime S, lambda 0)
    amax = P0full.argmax(-1); out5 = []
    for r, Q in enumerate(chains[0.0]):
        H0 = held[r]; lo, hi = np.zeros_like(H0), np.zeros_like(H0)
        for c in range(C):
            for x in range(X): lo[c, x], hi[c, x] = bands(H0[c, x], NHOLD)
        acc = np.array([(Q[g].argmax(-1) == amax).mean() for g in range(GEN + 1)]); cert = np.array([certificate(Q[g], lo, hi, H0) for g in range(GEN + 1)])
        nrng = np.random.default_rng(SEED + 5000 + r); null = [certificate(fit(sample(P0full, NCELL, nrng)), lo, hi, H0) for _ in range(200)]; thr = np.percentile(null, 95)
        drop = np.where(acc <= acc[0] - 0.05)[0]; warn = np.where(cert > thr)[0]
        out5.append(dict(drop=int(drop[0]) if len(drop) else None, warn=int(warn[0]) if len(warn) else None))
    with_drop = [o for o in out5 if o['drop'] is not None]; early = [o for o in with_drop if o['warn'] is not None and o['warn'] < o['drop']]
    res['P5_early_warning'] = dict(replicates_with_drop=len(with_drop), warned_strictly_before=len(early),
                                   share_among_drops=round(len(early) / max(len(with_drop), 1), 3), share_all=round(len(early) / REPS, 3),
                                   median_lead=float(np.median([o['drop'] - o['warn'] for o in early])) if early else None, per_replicate=out5,
                                   holds=bool(len(early) / max(len(with_drop), 1) >= 0.8 and len(early) / REPS >= 0.5))
    os.makedirs(os.path.join(H, 'results'), exist_ok=True); json.dump(res, open(os.path.join(H, 'results', 'recursive_training.json'), 'w'), indent=1)
    for k in ('P1_contraction', 'P2_breaks', 'P3_fresh_data', 'P4_invariance', 'P5_early_warning'):
        print(k, {a: b for a, b in res[k].items() if a not in ('per_replicate', 'mean_D', 'bound_plan', 'null95')})
if __name__ == '__main__':
    main()
