"""PREANALYSIS_SUPPORT.md experiments E1-E6, run once. Output: results/support_experiments.json"""
import os, sys, json, itertools
import numpy as np
from scipy.optimize import linprog, minimize
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, '..', 'open_clinical'))
import support as SP, decide as DC
SEED = 20261012; rng = np.random.default_rng(SEED); R = {}

def rand_D(n, m):
    cells = [(x, y) for x in range(n) for y in range(m)]; k = rng.integers(1, len(cells) + 1); return [cells[i] for i in rng.choice(len(cells), k, replace=False)]
def lp_feasible(f, g, lower):
    n, m = len(f), len(g); r = linprog(np.zeros(n * m), A_eq=SP._transport_eq(n, m), b_eq=np.concatenate([f, g]), bounds=[(l, None) for l in np.asarray(lower).ravel()], method='highs'); return r.status == 0

# ---------------- E1
import tower_icu as TI
nmap = {'CP^2': 3, 'F_1 = Bl_1 CP^2': 4, 'CP^1 x CP^1': 4, 'Bl_2 CP^2': 5, 'Bl_3 CP^2 (dP6)': 6}
eq, tor, tor_n = 0, 0, 0; done = 0
while done < 500:
    n, m = int(rng.integers(2, 4)), int(rng.integers(2, 5)); f, g = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(m)); D = rand_D(n, m)
    tau = np.zeros((n, m))
    for (x, y) in D: tau[x, y] = rng.uniform(0, 0.5 * min(f[x], g[y]) / max(n, m))
    fr, gr = SP.reduced_margins(f, g, tau, D)
    if min(fr.min(), gr.min()) <= 0: continue
    lower = tau * SP.mask(D, n, m); V1 = SP.vertices_bounded(f, g, lower); V2 = SP.vertices_bounded(fr, gr, np.zeros((n, m))) + lower.ravel()
    A1, A2 = np.unique(np.round(V1, 7), axis=0), np.unique(np.round(V2, 7), axis=0); eq += int(A1.shape == A2.shape and np.abs(A1 - A2).max() < 1e-7)
    if (n, m) == (2, 3):
        s = fr.sum(); tor_n += 1; tor += int(nmap[TI.toric(fr[0] / s, gr / s)[0]] == len(A1))
    done += 1
inf_ok, k = 0, 0
while k < 200:
    n, m = int(rng.integers(2, 4)), int(rng.integers(2, 5)); f, g = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(m)); D = rand_D(n, m)
    tau = np.zeros((n, m))
    for (x, y) in D: tau[x, y] = rng.uniform(0, 0.6)
    fr, gr = SP.reduced_margins(f, g, tau, D)
    if min(fr.min(), gr.min()) >= 0: continue
    inf_ok += int(not lp_feasible(f, g, tau * SP.mask(D, n, m))); k += 1
R['E1'] = dict(vertex_sets_equal=eq, of=500, toric_type_2x3=tor, of_2x3=tor_n, infeasible_by_lp=inf_ok, of_infeasible=200,
               holds=bool(eq == 500 and tor == tor_n and inf_ok == 200))
print('E1', R['E1'], flush=True)

# ---------------- E2
a_ok, a_n, b_ok, b_n, c_dis, c_n, bfloor_ok, bfloor_n = 0, 0, 0, 0, 0, 0, 0, 0
for _ in range(1000):
    n, m = int(rng.integers(2, 5)), int(rng.integers(2, 5)); f, g = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(m)); D = rand_D(n, m)
    tau, w = rng.uniform(0, .1, (n, m)), rng.uniform(.5, 2, (n, m)); h = SP.headroom(f, g, tau, w, D); dl, duals = SP.headroom_lp(f, g, tau, w, D)
    if h['delta'] >= 0:
        a_n += 1; a_ok += int(abs(dl - h['delta']) < 1e-9)
        if len(h['ties']) == 1:
            b_n += 1; allowed = set(h['ties']); nz = [('row', i) if i < n else ('col', i - n) for i in range(n + m) if abs(duals[i]) > 1e-9]
            b_ok += int(all(z in allowed for z in nz))
    else:
        c_n += 1; c_dis += int(abs(dl - h['delta']) > 1e-9)
R['E2'] = dict(a_exact=a_ok, a_of=a_n, b_duals_on_bottleneck=b_ok, b_of=b_n, b_share=round(b_ok / max(b_n, 1), 4), c_negative_cases=c_n, c_share_differ=round(c_dis / max(c_n, 1), 4),
               holds_a=bool(a_ok == a_n), holds_b=bool(b_ok / max(b_n, 1) >= 0.99))
print('E2', R['E2'], flush=True)

# ---------------- E3
def Hb(p): p = np.asarray(p).ravel(); p = p[p > 1e-15]; return float(-(p * np.log2(p)).sum())
a_ok, n_pairs, b_ok, c_ok, c_n = 0, 0, 0, 0, 0
for _ in range(200):
    pxy = rng.dirichlet(np.ones(4)).reshape(2, 2); py = pxy.sum(0); pyz = np.array([py[y] * rng.dirichlet(np.ones(2)) for y in range(2)])
    cs = SP.chain_statuses(pxy, pyz, 0.0); b_ok += int(cs['tau_ci'] <= cs['tau_any'] + 1e-12); bound = Hb(pxy) + Hb(pyz) - Hb(py)
    A = np.vstack([SP._marg_matrix((2, 2, 2), (0, 1)), SP._marg_matrix((2, 2, 2), (1, 2))]); b = np.concatenate([pxy.ravel(), pyz.ravel()])
    for tau in np.linspace(0, 1.5 * cs['tau_any'], 20):
        n_pairs += 1; lpf = SP.glue((2, 2, 2), [((0, 1), pxy), ((1, 2), pyz)], tau)['status'] == 'glues'; cf = SP.chain_statuses(pxy, pyz, tau)
        a_ok += int(lpf == cf['floored_filler'])
        if lpf:
            ci = np.array([[[pxy[x, y] * pyz[y, z] / py[y] for z in range(2)] for y in range(2)] for x in range(2)]).ravel()
            r0 = linprog(np.zeros(8), A_eq=A, b_eq=b, bounds=[(tau, None)] * 8, method='highs'); x0 = ci if ci.min() >= tau else r0.x
            res = minimize(lambda p: -Hb(np.clip(p, 1e-300, 1)), x0, method='SLSQP', bounds=[(tau, 1)] * 8, constraints=[dict(type='eq', fun=lambda p: A @ p - b)], options=dict(ftol=1e-14, maxiter=1000))
            hmax = max(Hb(res.x) if np.abs(A @ res.x - b).max() < 1e-8 and res.x.min() >= tau - 1e-9 else -1, Hb(x0))
            c_n += 1; c_ok += int((abs(hmax - bound) < 1e-6) == cf['ci_filler_meets_floor'])
R['E3'] = dict(a_lp_vs_closed_form=a_ok, a_of=n_pairs, b_tci_le_tany=b_ok, b_of=200, c_entropy_equality_iff_ci=c_ok, c_of=c_n, c_share=round(c_ok / c_n, 4),
               holds=bool(a_ok == n_pairs and b_ok == 200 and c_ok / c_n >= 0.99))
print('E3', R['E3'], flush=True)

# ---------------- E4
def coupling(a, b):
    lo, hi = max(0, a + b - 1), min(a, b); p11 = rng.uniform(lo, hi); return np.array([[1 - a - b + p11, b - p11], [a - p11, p11]])   # index [0/1, 0/1], P(first=1)=a
def E_pm(t): return float(t[0, 0] + t[1, 1] - t[0, 1] - t[1, 0])
agree, cert_ok, cert_n, n_ctx = 0, 0, 0, 0; fams_glue = []
for kind in ('local', 'global'):
    for _ in range(300):
        if kind == 'local':
            px, py, pz = rng.uniform(.1, .9, 3); txy, tyz, tzx = coupling(px, py), coupling(py, pz), coupling(pz, px)
        else:
            J = rng.dirichlet(np.ones(8)).reshape(2, 2, 2); txy, tyz, tzx = J.sum(2), J.sum(0), J.sum(1).T
        fam = [((0, 1), txy), ((1, 2), tyz), ((2, 0), tzx)]; gl = SP.glue((2, 2, 2), fam, 0.0)
        agree += int((gl['status'] == 'glues') == SP.three_cycle_noncontextual(E_pm(txy), E_pm(tyz), E_pm(tzx)))
        if gl['status'] == 'fails': n_ctx += 1; cert_n += 1; cert_ok += int(gl['verified'])
        else: fams_glue.append(fam)
sc, sc_inst = 0, []
for fam in fams_glue:
    tloc = min(t.min() for _, t in fam) / 2; hit = None
    for tau in np.linspace(0, tloc, 6)[1:]:
        if SP.glue((2, 2, 2), fam, tau)['status'] == 'fails': hit = float(tau); break
    if hit is not None: sc += 1; sc_inst.append(dict(tau=round(hit, 5), tau_local_max=round(float(tloc), 5)))
R['E4'] = dict(a_lp_vs_3cycle=agree, a_of=600, contextual=n_ctx, b_certificates_verified=cert_ok, b_of=cert_n, c_gluable=len(fams_glue),
               c_support_contextual=sc, c_share=round(sc / len(fams_glue), 4), c_examples=sc_inst[:5],
               holds=bool(agree == 600 and cert_ok == cert_n and sc >= 1))
print('E4', {k: v for k, v in R['E4'].items() if k != 'c_examples'}, flush=True)

# ---------------- E5
a_ok, b_ok, b_n, c_ok, c_n, shrink, tight, tight_n = 0, 0, 0, 0, 0, 0, 0, 0
for i in range(500):
    n, m = int(rng.integers(2, 4)), int(rng.integers(2, 4)); f, g = rng.dirichlet(np.ones(n) * 2), rng.dirichlet(np.ones(m) * 2); N = n * m
    tau = np.full((n, m), rng.uniform(0, .5) * min(f.min() / m, g.min() / n)); fr, gr = SP.reduced_margins(f, g, tau, [(x, y) for x in range(n) for y in range(m)])
    vr = linprog(rng.normal(size=N), A_eq=SP._transport_eq(n, m), b_eq=np.concatenate([fr, gr]), bounds=[(0, None)] * N, method='highs').x
    lam = rng.uniform(); pi = tau.ravel() + lam * np.outer(fr, gr).ravel() / fr.sum() + (1 - lam) * vr
    k = int(rng.integers(2, N + 1))
    if i % 2 == 0:
        tgt = np.concatenate([rng.permutation(k), rng.integers(0, k, N - k)])[:N]; K = np.zeros((k, N)); K[tgt, np.arange(N)] = 1
    else:
        K = rng.dirichlet(np.ones(k), N).T
    Kp, Kt = SP.push(pi, tau, K); s0, s1 = (pi - tau.ravel()).min(), (Kp - Kt).min()
    a_ok += int((Kp >= Kt - 1e-12).all())
    if i % 2 == 0: b_n += 1; b_ok += int(s1 >= s0 - 1e-12)
    else: c_n += 1; c_ok += int(s1 >= K.sum(1).min() * s0 - 1e-12); shrink += int(s1 < s0 - 1e-12)
    for t in range(k):
        r = linprog(K[t], A_eq=SP._transport_eq(n, m), b_eq=np.concatenate([f, g]), bounds=[(v, None) for v in tau.ravel()], method='highs')
        tight_n += 1; tight += int(abs(r.fun - Kt[t]) < 1e-9)
R['E5'] = dict(a_floor_preserved=a_ok, a_of=500, b_coarsening_no_shrink=b_ok, b_of=b_n, c_stochastic_bound=c_ok, c_of=c_n, c_share_shrinks=round(shrink / c_n, 4),
               tightness_share=round(tight / tight_n, 4), holds=bool(a_ok == 500 and b_ok == b_n and c_ok == c_n))
print('E5', R['E5'], flush=True)

# ---------------- E6
U, acts = DC.assert_actions('abc'); rep_ok, rep_n = 0, 0
for _ in range(100):
    q = rng.dirichlet(np.ones(3)); src = [dict(name=f's{j}', counts=rng.multinomial(int(rng.integers(200, 2000)), q).tolist(), r=round(float(rng.uniform(.4, 1)), 3)) for j in range(int(rng.integers(2, 5)))]
    prob = dict(decision=dict(sources=src, U=U.tolist(), actions=acts, kappa=1.1, floors=None)); versions = [prob]
    p2 = json.loads(json.dumps(prob)); p2['decision']['floors'] = {int(rng.integers(0, 3)): round(float(rng.uniform(0, .2)), 3)}; versions.append(p2)
    p3 = json.loads(json.dumps(p2)); j = int(rng.integers(0, len(src))); p3['decision']['sources'][j]['counts'] = (np.array(src[j]['counts']) + rng.multinomial(300, q)).tolist(); versions.append(p3)
    p4 = json.loads(json.dumps(p3)); p4['decision']['kappa'] = round(float(rng.uniform(1, 2)), 3); versions.append(p4)
    for v, p in enumerate(versions): rec = SP.record(p, f'v{v}'); rep_n += 1; rep_ok += int(SP.replay(rec))
cls = {t: 0 for t in ('supp', 'glue', 'comp', 'evid')}
def only(O, t): return O[t] is not None and all(O[u] is None for u in O if u != t)
for _ in range(20):
    q = rng.dirichlet(np.ones(3)); c = rng.multinomial(2000, q); o = int(np.argmin(q))
    O = SP.audit(dict(decision=dict(sources=[dict(name='a', counts=c.tolist(), r=1.0)], U=U.tolist(), actions=acts, floors={o: float(c[o] / 2000 + .15)})))['obstructions']; cls['supp'] += int(only(O, 'supp'))
    e = rng.uniform(0, .5); PR = [np.array([[.5, 0], [0, .5]]), np.array([[.5, 0], [0, .5]]), np.array([[0, .5], [.5, 0]])]
    fam = [[list(k), ((1 - e) * t + e * .25).tolist()] for k, t in zip(((0, 1), (1, 2), (2, 0)), PR)]
    O = SP.audit(dict(cover=dict(card=[2, 2, 2], family=fam, tau=0.0)))['obstructions']; cls['glue'] += int(only(O, 'glue'))
    ks = [rng.dirichlet(np.ones(3), 3).T.tolist() for _ in range(int(rng.integers(2, 4)))]
    O = SP.audit(dict(loop=dict(kernels=ks, tolerance=1e-6)))['obstructions']; cls['comp'] += int(only(O, 'comp'))
    q4 = np.array([.5, .3, .15, .05]); honest = [dict(name='a', counts=rng.multinomial(3000, .7 * q4 + .3 * np.eye(4)[1]).tolist(), r=.7), dict(name='b', counts=rng.multinomial(3000, .7 * q4 + .3 * np.eye(4)[2]).tolist(), r=.7)]
    U4, a4 = DC.assert_actions('ABCD'); O = SP.audit(dict(decision=dict(sources=honest + [dict(name='liar', counts=[0, 0, 0, 3000], r=.9)], U=U4.tolist(), actions=a4)))['obstructions']; cls['evid'] += int(only(O, 'evid'))
R['E6'] = dict(a_replay_identical=rep_ok, a_of=rep_n, b_classified=cls, b_of_each=20, holds=bool(rep_ok == rep_n and all(v == 20 for v in cls.values())))
print('E6', R['E6'], flush=True)
json.dump(R, open(os.path.join(H, 'results', 'support_experiments.json'), 'w'), indent=1)
