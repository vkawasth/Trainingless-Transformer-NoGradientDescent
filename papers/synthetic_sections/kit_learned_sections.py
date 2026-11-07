"""kit_learned_sections.py -- Study 17 of PREREG_study17.md (DRAFT): small transformers trained only on context marginals of synthetic
worlds; their implied joint laws measured against the marginal projection of the companion note (sections 12.13-12.19).

  python kit_learned_sections.py plan                      # list the conditions (no training)
  python kit_learned_sections.py train --out runs17.jsonl [--only K] [--resume]      # train every model, write implied joints
  python kit_learned_sections.py score --runs runs17.jsonl --out study17_final.json  # the pre-registered analysis, once
  python kit_learned_sections.py smoke                      # non-neural responders + a 60-step run on a dummy world outside the study

Worlds (all exact rational laws, seed 20261107):
  G1  latent-class 4-cycle worlds: hidden H with 3 classes, A0, A1, B0, B1 independent given H; parameters in (1/20)Z
  G2  cohort worlds: each of the four cycle pairs is recorded in its own G1 world (separate studies of different populations)
  C   cooked theorem paths: P_eps = (1 - eps) P_CH + eps P_int and (1 - eps) P_0 + eps P_int; the symmetric CHSH path at lam = 3/5, 4/5, 1
  W2  chains X - Y - Z with a hidden binary confounder (true law not Markov); pairs of chains and their product
"""
import os, sys, json, math, time, argparse, itertools, random
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import margsec as G
from pce import selcomp as S
SEED = 20261107
CELLS4 = list(itertools.product((0, 1), repeat=4)); CTX4 = [(0, 2), (0, 3), (1, 2), (1, 3)]          # (A0,B0), (A0,B1), (A1,B0), (A1,B1)
STEPS, BATCH, LR, WD, DM, NL, NH, FF = 4000, 256, 3e-3, 0.01, 64, 2, 4, 128

# ------------------------------------------------------------------ worlds
def latent_world(rng, nvar=4, k=3):
    """exact joint over {0,1}^nvar: H ~ pi (multiples of 1/20, all >= 1/20), X_v | H ~ Bernoulli(theta) with theta in {1..19}/20"""
    while True:
        w = [rng.randint(1, 18) for _ in range(k)]
        if sum(w) == 20: break
        if rng.random() < 0.02: w = [7, 7, 6]; break
    pi = [Fr(x, 20) for x in w]; th = [[Fr(rng.randint(1, 19), 20) for _ in range(nvar)] for _ in range(k)]
    out = {}
    for c in itertools.product((0, 1), repeat=nvar):
        out[c] = sum(pi[h] * math.prod((th[h][v] if c[v] == 0 else 1 - th[h][v]) for v in range(nvar)) for h in range(k))
    return [out[c] for c in itertools.product((0, 1), repeat=nvar)]

def tables_of(P, ctx=CTX4, nvar=4):
    cells = list(itertools.product((0, 1), repeat=nvar)); out = []
    for (i, j) in ctx:
        t = [[Fr(0), Fr(0)], [Fr(0), Fr(0)]]
        for c, p in zip(cells, P): t[c[i]][c[j]] += p
        out.append(t)
    return out

def cooked():
    rng = random.Random(SEED + 7)
    def weights(support, total=240):
        w = [rng.randint(5, 25) for _ in support]; s = sum(w); w = [x * total // s for x in w]; w[0] += total - sum(w)
        P = [Fr(0)] * 16
        for k, x in zip(support, w): P[k] = Fr(x, total)
        return P
    F = G.facets_D(); Lset = {(tuple(c), c0) for c, c0 in G.local_ineqs()}
    ch = [f for f in F if (tuple(map(Fr, f[0])), Fr(f[1])) not in Lset][0]; pos = [f for f in F if (tuple(map(Fr, f[0])), Fr(f[1])) in Lset][0]
    Pint = weights(range(16)); Pch = weights(ch[2]); P0 = weights(pos[2])
    return Pint, Pch, P0

def conditions():
    """the full list of training conditions (dicts); deterministic"""
    rng = random.Random(SEED); conds = []
    G1 = [latent_world(rng) for _ in range(8)]
    for w, P in enumerate(G1):
        for N in (300, 3000):
            for s in range(3): conds.append(dict(kind="G1", world=w, N=N, rep=s))
    G2 = [[latent_world(rng) for _ in range(4)] for _ in range(8)]
    for w in range(8):
        for N in (300, 3000):
            for s in range(3): conds.append(dict(kind="G2", world=w, N=N, rep=s))
    for eps in ("1/5", "1/20", "1/100", "0"):
        for path in ("CH", "zero"):
            for s in range(2): conds.append(dict(kind="C", path=path, eps=eps, N=None, rep=s))
    for lam in ("3/5", "4/5", "1"):
        for s in range(2): conds.append(dict(kind="C", path="PR", eps=lam, N=None, rep=s))
    W2 = [(chain_world(rng), chain_world(rng)) for _ in range(6)]
    for w in range(6):
        for part in ("c1", "c2", "prod"):
            for s in range(2): conds.append(dict(kind="W2", world=w, part=part, N=None, rep=s))
    for i, c in enumerate(conds): c["idx"] = i; c["seed"] = SEED + 100 * i
    return conds, dict(G1=G1, G2=G2, W2=W2, cooked=cooked())

def chain_world(rng):
    """X, Y, Z binary; hidden binary U with X, Y depending on U, Z depending on Y and U (so the law is not Markov X - Y - Z)"""
    pu = Fr(rng.randint(4, 16), 20); out = {}
    tx = [Fr(rng.randint(2, 18), 20) for _ in range(2)]; ty = [[Fr(rng.randint(2, 18), 20) for _ in range(2)] for _ in range(2)]
    tz = [[Fr(rng.randint(2, 18), 20) for _ in range(2)] for _ in range(2)]
    for x, y, z in itertools.product((0, 1), repeat=3):
        v = Fr(0)
        for u in (0, 1):
            pu_ = pu if u == 0 else 1 - pu; px = tx[u] if x == 0 else 1 - tx[u]; py = ty[u][x] if y == 0 else 1 - ty[u][x]; pz = tz[u][y] if z == 0 else 1 - tz[u][y]
            v += pu_ * px * py * pz
        out[(x, y, z)] = v
    return out

def condition_data(c, W):
    """returns (nvar, nval, contexts [(i, j)], target tables (exact) the data come from, true joint or None, training tables)"""
    if c["kind"] == "G1":
        P = W["G1"][c["world"]]; T = tables_of(P); return 4, 2, CTX4, T, P
    if c["kind"] == "G2":
        Ws = W["G2"][c["world"]]; T = [tables_of(Ws[k])[k] for k in range(4)]; return 4, 2, CTX4, T, None
    if c["kind"] == "C":
        Pint, Pch, P0 = W["cooked"]
        if c["path"] == "PR":
            lam = Fr(c["eps"]); T = [[[(1 + lam) / 4, (1 - lam) / 4], [(1 - lam) / 4, (1 + lam) / 4]]] * 3 + [[[(1 - lam) / 4, (1 + lam) / 4], [(1 + lam) / 4, (1 - lam) / 4]]]
            return 4, 2, CTX4, T, None
        e = Fr(c["eps"]); base = Pch if c["path"] == "CH" else P0; P = [(1 - e) * a + e * b for a, b in zip(base, Pint)]
        return 4, 2, CTX4, tables_of(P), P
    c1, c2 = W["W2"][c["world"]]
    if c["part"] in ("c1", "c2"):
        L = c1 if c["part"] == "c1" else c2; P = [L[k] for k in itertools.product((0, 1), repeat=3)]
        return 3, 2, [(0, 1), (1, 2)], tables_of(P, [(0, 1), (1, 2)], 3), P
    P = {}
    for a in itertools.product((0, 1), repeat=3):
        for b in itertools.product((0, 1), repeat=3): P[tuple(2 * x + y for x, y in zip(a, b))] = c1[a] * c2[b]
    cells = list(itertools.product(range(4), repeat=3)); Pv = [P[k] for k in cells]
    T = []
    for (i, j) in [(0, 1), (1, 2)]:
        t = [[Fr(0)] * 4 for _ in range(4)]
        for k, p in zip(cells, Pv): t[k[i]][k[j]] += p
        T.append(t)
    return 3, 4, [(0, 1), (1, 2)], T, Pv

# ------------------------------------------------------------------ the model
def make_model(nvar, nval):
    import torch
    V = 2 + nvar + nval                                   # BOS, EOS, variable tokens, value tokens
    class Net(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.e = torch.nn.Embedding(V, DM)
            layer = torch.nn.TransformerEncoderLayer(DM, NH, FF, 0.0, batch_first=True); s.t = torch.nn.TransformerEncoder(layer, NL, enable_nested_tensor=False)
            s.o = torch.nn.Linear(DM, V)
        def forward(s, x):                                # no positional embedding (NoPE); causal mask
            m = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1])
            return s.o(s.t(s.e(x), mask=m, is_causal=True))
    return Net(), V

def train_one(c, W, steps=STEPS, log_every=500):
    import torch
    torch.manual_seed(c["seed"]); rng = np.random.default_rng(c["seed"]); torch.set_num_threads(1)
    nvar, nval, ctx, T, P = condition_data(c, W); net, V = make_model(nvar, nval)
    BOS, EOS = 0, 1; var = lambda v: 2 + v; val = lambda a: 2 + nvar + a
    # the data: exact context tables (N = None) or a fixed finite sample of N records per context
    probs = [np.array([float(x) for row in t for x in row]) for t in T]
    if c.get("N"):
        data = [rng.choice(len(p), size=c["N"], p=p / p.sum()) for p in probs]
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=WD); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    traj = []
    for step in range(steps):
        ci = rng.integers(0, len(ctx), BATCH)
        if c.get("N"): cell = np.array([data[k][j] for k, j in zip(ci, rng.integers(0, c["N"], BATCH))])
        else: cell = np.array([rng.choice(len(probs[k]), p=probs[k] / probs[k].sum()) for k in ci])
        flip = rng.random(BATCH) < 0.5; ci_ = np.array([ctx[k][0] for k in ci]); cj_ = np.array([ctx[k][1] for k in ci])
        av, bv = cell // nval, cell % nval
        v1 = np.where(flip, cj_, ci_); x1 = np.where(flip, bv, av); v2 = np.where(flip, ci_, cj_); x2 = np.where(flip, av, bv)
        seq = np.stack([np.full(BATCH, BOS), 2 + v1, 2 + nvar + x1, 2 + v2, 2 + nvar + x2, np.full(BATCH, EOS)], axis=1).astype(np.int64)
        x = torch.tensor(seq)
        loss = torch.nn.functional.cross_entropy(net(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % log_every == 0: traj.append(dict(step=step + 1, joint=implied(net, nvar, nval)))
    return net, nvar, nval, traj

def implied(net, nvar, nval, orders=None):
    """exact implied joint: for each variable order, prod_t q(v_t | prefix) with the variable tokens forced, value tokens renormalised;
    returns {"mean": joint averaged over orders, "orders": [joint per order]} (joints as lists over itertools.product(range(nval), repeat=nvar))"""
    import torch
    orders = orders or list(itertools.permutations(range(nvar))); cells = list(itertools.product(range(nval), repeat=nvar)); per = []
    with torch.no_grad():
        for o in orders:
            J = np.zeros(len(cells))
            for k, cvals in enumerate(cells):
                seq = [0]; lp = 0.0
                for v in o:
                    seq.append(2 + v); logits = net(torch.tensor([seq]))[0, -1, 2 + nvar: 2 + nvar + nval]
                    lp += float(torch.log_softmax(logits, -1)[cvals[v]]); seq.append(2 + nvar + cvals[v])
                J[k] = math.exp(lp)
            per.append((J / J.sum()).tolist())
    return dict(mean=np.mean(per, axis=0).tolist(), orders=per)

# ------------------------------------------------------------------ metrics
def fibre_support(T, ctx, nvar, nval):
    """cells that some joint law with context marginals T can make positive (the others are forced to 0), by linear programming"""
    cells = np.array(list(itertools.product(range(nval), repeat=nvar))); A = []; b = []
    for k, (i, j) in enumerate(ctx):
        for a in range(nval):
            for bb in range(nval): A.append(((cells[:, i] == a) & (cells[:, j] == bb)).astype(float)); b.append(float(T[k][a][bb]))
    A.append(np.ones(len(cells))); b.append(1.0)
    sup = S._support(np.array(A), np.array(b)); return sup

def ipf(Q, T, ctx, nvar, nval, iters=20000, tol=1e-13):
    """I-projection of Q onto {R : R's context marginals = T}: iterative proportional fitting started at Q restricted to the cells the
    data do not force to zero (without the restriction IPF only creeps toward boundary points such as a CH-facet fibre)"""
    cells = np.array(list(itertools.product(range(nval), repeat=nvar))); R = np.array(Q, float).copy()
    sup = fibre_support(T, ctx, nvar, nval)
    if sup is not None:
        mask = np.zeros(len(R), bool); mask[list(sup)] = True; R[~mask] = 0.0
    for _ in range(iters):
        err = 0.0
        for k, (i, j) in enumerate(ctx):
            for a in range(nval):
                for b in range(nval):
                    m = (cells[:, i] == a) & (cells[:, j] == b); cur = R[m].sum(); tg = float(T[k][a][b]); err = max(err, abs(cur - tg))
                    if cur > 0: R[m] *= tg / cur
        if err < tol: break
    return R

def kl(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float); m = p > 0; return float((p[m] * np.log(p[m] / q[m])).sum())
def tv(p, q): return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())

def marg_mismatch(Q, T, ctx, nvar, nval):
    cells = np.array(list(itertools.product(range(nval), repeat=nvar))); tot = 0.0
    for k, (i, j) in enumerate(ctx):
        Qc = np.array([[Q[(cells[:, i] == a) & (cells[:, j] == b)].sum() for b in range(nval)] for a in range(nval)]).ravel()
        tot += kl(np.array([float(x) for r in T[k] for x in r]), Qc)
    return tot

def mstar(T, ctx, nvar, nval, starts=20, iters=3000, seed=SEED):
    """inf over joint laws R of sum_c KL(d_c || R_c), by EM from several random starts"""
    rng = np.random.default_rng(seed); cells = np.array(list(itertools.product(range(nval), repeat=nvar))); best = np.inf
    masks = [[[(cells[:, i] == a) & (cells[:, j] == b) for b in range(nval)] for a in range(nval)] for (i, j) in ctx]
    for _ in range(starts):
        R = rng.dirichlet(np.ones(len(cells)))
        for _ in range(iters):
            new = np.zeros_like(R)
            for k in range(len(ctx)):
                for a in range(nval):
                    for b in range(nval):
                        m = masks[k][a][b]; s = R[m].sum()
                        if s > 0: new[m] += float(T[k][a][b]) * R[m] / s
            R = new / len(ctx)
        best = min(best, marg_mismatch(R, T, ctx, nvar, nval))
    return best

def sections_4cycle(T):
    """maximum-entropy and leximin sections of exact 4-cycle data (data vector of margsec)"""
    d = [T[0][0][0], T[1][0][0], T[2][0][0], T[3][0][0], T[0][0][0] + T[0][0][1], T[2][0][0] + T[2][0][1], T[0][0][0] + T[0][1][0], T[1][0][0] + T[1][1][0]]
    return np.array(G.maxent(d)), np.array([float(x) for x in G.leximin(d)]), d

def q_equal(J, i=0, j=1, nvar=4):
    cells = list(itertools.product((0, 1), repeat=nvar)); return float(sum(p for c, p in zip(cells, J) if c[i] == c[j]))

def analyse_run(c, W, out):
    """all metrics for one trained model (out = implied joints)"""
    nvar, nval, ctx, T, P = condition_data(c, W); Q = np.array(out["mean"]); r = dict(idx=c["idx"], kind=c["kind"])
    drift = [tv(a, b) for a, b in itertools.combinations(out["orders"], 2)]; r["drift"] = float(np.mean(drift))
    if P is not None:
        Pt = np.array([float(x) for x in P]); Ps = ipf(Q, T, ctx, nvar, nval)
        r.update(total=kl(Pt, Q), infibre=kl(Pt, Ps), offfibre=kl(Ps, Q), tv_true_Q=tv(Pt, Q), tv_true_Pstar=tv(Pt, Ps))
        r["pyth_rel_err"] = abs(r["total"] - r["infibre"] - r["offfibre"]) / max(r["total"], 1e-300)
        if nvar == 4:
            sH, sL, d = sections_4cycle(T); a_, b_ = tv(Ps, sH), tv(Ps, sL); r["iota"] = a_ / (a_ + b_) if a_ + b_ > 0 else 0.5
            lo, hi = G.interval(d, [1 if cc[0] == cc[1] else 0 for cc in G.CELLS])
            r.update(q_true=q_equal(Pt), q_Q=q_equal(Q), q_Pstar=q_equal(Ps), q_width=hi - lo)
    else:
        r.update(mismatch=marg_mismatch(Q, T, ctx, nvar, nval), mstar=mstar(T, ctx, nvar, nval))
    if c["kind"] == "W2":
        cells = list(itertools.product(range(nval), repeat=3)); comp = np.zeros((nval, nval))
        for cc, p in zip(cells, Q): comp[cc[0], cc[2]] += p
        r["composite"] = comp.tolist()
    return r

# ------------------------------------------------------------------ the analysis (once, on all runs)
def analyse(runs, conds, W):
    R = [analyse_run(conds[x["idx"]], W, x["joint"]) for x in runs]; byi = {r["idx"]: r for r in R}; res = dict(runs=R)
    g1 = [r for r in R if r["kind"] == "G1"]; g1_3000 = [r for r in g1 if conds[r["idx"]]["N"] == 3000]
    ckd = [r for r in R if r["kind"] == "C" and conds[r["idx"]]["path"] != "PR"]
    res["T1"] = all(r["pyth_rel_err"] < 1e-6 for r in g1 + ckd)
    res["T2"] = all(r["tv_true_Pstar"] < 1e-6 for r in ckd if conds[r["idx"]]["path"] == "CH" and conds[r["idx"]]["eps"] == "0")
    res["T3"] = all(r["mismatch"] >= r["mstar"] - 1e-6 for r in R if "mismatch" in r)
    res["T4"] = all(abs(r["q_Pstar"] - r["q_true"]) <= r["q_width"] + 1e-9 for r in g1 + ckd if "q_width" in r)
    io = [r["iota"] for r in g1]; f = lambda cond: sum(cond(x) for x in io) / len(io)
    res["H1"] = dict(share_below_half=f(lambda x: x < 0.5), verdict="maximum-entropy-like" if f(lambda x: x < 0.5) >= 0.75 else "leximin-like" if f(lambda x: x > 0.5) >= 0.75 else "mixed")
    res["H2"] = dict(share=float(np.mean([r["infibre"] > r["offfibre"] for r in g1_3000])), supported=bool(np.mean([r["infibre"] > r["offfibre"] for r in g1_3000]) >= 0.75))
    ch0 = [r["tv_true_Q"] for r in ckd if conds[r["idx"]]["eps"] == "0" and conds[r["idx"]]["path"] == "CH"]
    z0 = [r["tv_true_Q"] for r in ckd if conds[r["idx"]]["eps"] == "0" and conds[r["idx"]]["path"] == "zero"]
    res["H3"] = dict(ch=float(np.mean(ch0)), zero=float(np.mean(z0)), supported=bool(np.mean(ch0) < np.mean(z0)))
    res["H4"] = dict(share=float(np.mean([r["drift"] > 0.01 for r in g1])), supported=bool(np.mean([r["drift"] > 0.01 for r in g1]) >= 0.75))
    defects = []
    for w in range(6):
        for s in range(2):
            pick = {conds[r["idx"]]["part"]: r for r in R if r["kind"] == "W2" and conds[r["idx"]]["world"] == w and conds[r["idx"]]["rep"] == s}
            if len(pick) < 3: continue
            a, b, p = (np.array(pick[k]["composite"]) for k in ("c1", "c2", "prod")); prod = np.kron(a, b)
            defects.append(tv(prod.ravel(), p.ravel()))
    res["H5"] = dict(defects=defects, share=float(np.mean([d < 0.0188 for d in defects])) if defects else None,
                     supported=bool(defects and np.mean([d < 0.0188 for d in defects]) >= 0.75))
    g2 = [r for r in R if r["kind"] == "G2"]; rng = np.random.default_rng(SEED); diffs = []
    for _ in range(2000):
        ws = rng.integers(0, 8, 8)
        a = np.mean([r["drift"] for r in g2 if conds[r["idx"]]["world"] in ws]); b = np.mean([r["drift"] for r in g1 if conds[r["idx"]]["world"] in ws]); diffs.append(a - b)
    res["H6"] = dict(drift_G2=float(np.mean([r["drift"] for r in g2])), drift_G1=float(np.mean([r["drift"] for r in g1])),
                     ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], supported=bool(np.percentile(diffs, 2.5) > 0))
    return res

# ------------------------------------------------------------------ smoke (no study condition is trained)
def smoke():
    conds, W = conditions(); out = dict(n_conditions=len(conds))
    # non-neural responders on G1 world 0 and on the CH path at eps = 0
    for label, c in (("G1 world 0", next(x for x in conds if x["kind"] == "G1")), ("CH eps 0", next(x for x in conds if x["kind"] == "C" and x["path"] == "CH" and x["eps"] == "0"))):
        nvar, nval, ctx, T, P = condition_data(c, W); sH, sL, d = sections_4cycle(T); u = np.full(16, 1 / 16)
        for name, Q in (("maxent responder", sH * 0.999 + u * 0.001), ("noisy truth", 0.9 * np.array([float(x) for x in P]) + 0.1 * u)):
            r = analyse_run(c, W, dict(mean=Q.tolist(), orders=[Q.tolist(), Q.tolist()]))
            out[f"{label} | {name}"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k in ("infibre", "offfibre", "pyth_rel_err", "iota", "tv_true_Pstar", "q_width")}
    c = next(x for x in conds if x["kind"] == "C" and x["path"] == "PR" and x["eps"] == "1"); nvar, nval, ctx, T, P = condition_data(c, W)
    out["PR box | uniform responder"] = dict(mismatch=round(marg_mismatch(np.full(16, 1 / 16), T, ctx, 4, 2), 6), mstar=round(mstar(T, ctx, 4, 2, starts=5, iters=1000), 6))
    # a 60-step training run on a dummy world that is not part of the study (3 binary variables, chain contexts)
    rng = random.Random(1); dummy = dict(kind="W2", world=0, part="c1", N=None, rep=0, idx=-1, seed=1)
    Wd = dict(W2=[(chain_world(rng), chain_world(rng))]); t = time.time(); net, nvar, nval, traj = train_one(dummy, Wd, steps=60, log_every=60)
    J = implied(net, nvar, nval); out["dummy training"] = dict(seconds=round(time.time() - t, 1), orders=len(J["orders"]), sums=round(sum(J["mean"]), 9))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--out"); ap.add_argument("--runs"); ap.add_argument("--only", type=int); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.cmd == "plan":
        conds, _ = conditions(); from collections import Counter; print(len(conds), Counter(c["kind"] for c in conds))
    elif a.cmd == "smoke": smoke()
    elif a.cmd == "train":
        conds, W = conditions(); done = set()
        if a.resume and os.path.exists(a.out): done = {json.loads(l)["idx"] for l in open(a.out)}
        elif os.path.exists(a.out) and os.path.getsize(a.out) > 0: raise SystemExit(f"{a.out} exists; use --resume")
        todo = [c for c in conds if c["idx"] not in done and (a.only is None or c["idx"] % 2 == a.only)]
        with open(a.out, "a") as fh:
            for c in todo:
                t = time.time(); net, nvar, nval, traj = train_one(c, W); J = implied(net, nvar, nval)
                fh.write(json.dumps(dict(idx=c["idx"], joint=J, traj=[dict(step=x["step"], mean=x["joint"]["mean"]) for x in traj])) + "\n"); fh.flush()
                print(f"{c['idx']} {c['kind']} done in {time.time() - t:.0f}s", flush=True)
    elif a.cmd == "score":
        conds, W = conditions(); runs = [json.loads(l) for l in open(a.runs)]
        json.dump(analyse(runs, conds, W), open(a.out, "w"), indent=1)
