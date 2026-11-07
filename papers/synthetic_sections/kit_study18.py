"""kit_study18.py -- Study 18 of PREREG_study18.md: sparsity-matched rigidity, consistency training, the contextual fraction, noise
floors. It reuses the worlds and metrics of kit_learned_sections.py (Study 17, frozen and unchanged) and adds the new pieces.

  python kit_study18.py plan
  python kit_study18.py train --out runs18_X.jsonl --only {0,1} [--resume]     # weights are saved to weights18/<idx>.pt
  python kit_study18.py score --runs runs18.jsonl --out study18_final.json
  python kit_study18.py smoke                                                  # no study condition is trained
"""
import os, sys, json, math, time, argparse, itertools, random
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import kit_learned_sections as K17
from pce import margsec as G
from pce import selcomp as S
SEED = 20261108; STEPS = 8000; LAMBDA_C = 1.0; NREP = 3; BC = 64
CELLS4 = K17.CELLS4; CTX4 = K17.CTX4

# ------------------------------------------------------------------ worlds
def s_faces():
    """3 random faces in each of F0 (CH facets: dim 7, 8 vertices, fibre 0), F1 (dim 6, 8 vertices, fibre 1), F2 (dim 5, 8 vertices, fibre 2)"""
    F = G.facets_D(); faces = sorted(G.faces_D(F), key=lambda V: sorted(V)); rng = random.Random(SEED)
    fam = {}
    for name, (dim, fib) in (("F0", (7, 0)), ("F1", (6, 1)), ("F2", (5, 2))):
        cand = [V for V in faces if len(V) == 8 and G.face_dim(V) == dim and G.fibre_dim_face(V) == fib]; fam[name] = rng.sample(cand, 3)
    return fam

def face_law(V, rng):
    w = [rng.randint(5, 25) for _ in V]; s = sum(w); w = [x * 240 // s for x in w]; w[0] += 240 - sum(w)
    P = [Fr(0)] * 16
    for k, x in zip(sorted(V), w): P[k] = Fr(x, 240)
    return P

def conditions():
    conds = []; rng = random.Random(SEED); fam = s_faces(); W = dict(S={}, G1=[], G2=[], W2=[])
    for f in ("F0", "F1", "F2"):
        for j, V in enumerate(fam[f]):
            W["S"][(f, j)] = face_law(V, rng)
            for variant in ("plain", "consistent", "plain_wd0"):
                for s in range(NREP): conds.append(dict(kind="S", fam=f, face=j, variant=variant, rep=s, N=None))
    W["G1"] = [K17.latent_world(rng) for _ in range(8)]
    for w in range(8):
        for s in range(NREP): conds.append(dict(kind="G1", world=w, variant="plain", rep=s, N=3000))
    W["G2"] = [[K17.latent_world(rng) for _ in range(4)] for _ in range(3)]
    for w in range(3):
        for s in range(NREP): conds.append(dict(kind="G2", world=w, variant="plain", rep=s, N=3000))
    W["W2"] = [(K17.chain_world(rng), K17.chain_world(rng)) for _ in range(6)]
    for w in range(6):
        for part in ("c1", "c2", "prod"):
            for s in range(NREP): conds.append(dict(kind="W2", world=w, part=part, variant="consistent", rep=s, N=None))
    for i, c in enumerate(conds): c["idx"] = i; c["seed"] = SEED + 100 * i
    return conds, W

def condition_data(c, W):
    if c["kind"] == "S":
        P = W["S"][(c["fam"], c["face"])]; return 4, 2, CTX4, K17.tables_of(P), P
    if c["kind"] == "G1":
        P = W["G1"][c["world"]]; return 4, 2, CTX4, K17.tables_of(P), P
    if c["kind"] == "G2":
        Ws = W["G2"][c["world"]]; return 4, 2, CTX4, [K17.tables_of(Ws[k])[k] for k in range(4)], None
    return K17.condition_data(dict(kind="W2", world=c["world"], part=c["part"]), dict(W2=W["W2"]))

# ------------------------------------------------------------------ training with the consistency penalty
def train_one(c, W, steps=STEPS):
    import torch
    torch.manual_seed(c["seed"]); rng = np.random.default_rng(c["seed"]); torch.set_num_threads(1)
    nvar, nval, ctx, T, P = condition_data(c, W); net, V = K17.make_model(nvar, nval); BOS, EOS = 0, 1
    probs = [np.array([float(x) for row in t for x in row]) for t in T]
    if c.get("N"): data = [rng.choice(len(p), size=c["N"], p=p / p.sum()) for p in probs]
    wd = 0.0 if c["variant"] == "plain_wd0" else K17.WD
    opt = torch.optim.AdamW(net.parameters(), K17.LR, weight_decay=wd); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        ci = rng.integers(0, len(ctx), K17.BATCH)
        if c.get("N"): cell = np.array([data[k][j] for k, j in zip(ci, rng.integers(0, c["N"], K17.BATCH))])
        else: cell = np.array([rng.choice(len(probs[k]), p=probs[k] / probs[k].sum()) for k in ci])
        flip = rng.random(K17.BATCH) < 0.5; ci_ = np.array([ctx[k][0] for k in ci]); cj_ = np.array([ctx[k][1] for k in ci]); av, bv = cell // nval, cell % nval
        v1 = np.where(flip, cj_, ci_); x1 = np.where(flip, bv, av); v2 = np.where(flip, ci_, cj_); x2 = np.where(flip, av, bv)
        seq = np.stack([np.full(K17.BATCH, BOS), 2 + v1, 2 + nvar + x1, 2 + v2, 2 + nvar + x2, np.full(K17.BATCH, EOS)], axis=1).astype(np.int64)
        x = torch.tensor(seq); loss = torch.nn.functional.cross_entropy(net(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
        if c["variant"] == "consistent":
            # squared difference of the log joint probability of random full assignments under two random variable orders
            vals = rng.integers(0, nval, (BC, nvar)); lps = []
            for _ in range(2):
                orders = np.array([rng.permutation(nvar) for _ in range(BC)]); toks = [np.full(BC, BOS)]
                for t in range(nvar): toks += [2 + orders[:, t], 2 + nvar + vals[np.arange(BC), orders[:, t]]]
                sq = torch.tensor(np.stack(toks, 1).astype(np.int64)); logits = net(sq[:, :-1])
                lp = 0
                for t in range(nvar):
                    pos = 2 * t + 1; lv = torch.log_softmax(logits[:, pos, 2 + nvar: 2 + nvar + nval], -1)
                    lp = lp + lv.gather(1, torch.tensor(vals[np.arange(BC), orders[:, t]]).unsqueeze(1)).squeeze(1)
                lps.append(lp)
            loss = loss + LAMBDA_C * ((lps[0] - lps[1]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net, nvar, nval

def pair_tables(net, nvar, nval, ctx):
    """the model's direct 2-variable predictions for each context, averaged over the two orders (in distribution)"""
    import torch
    out = []
    with torch.no_grad():
        for (i, j) in ctx:
            t = np.zeros((nval, nval))
            for (u, w, flip) in ((i, j, False), (j, i, True)):
                l1 = torch.log_softmax(net(torch.tensor([[0, 2 + u]]))[0, -1, 2 + nvar: 2 + nvar + nval], -1).numpy()
                for a in range(nval):
                    l2 = torch.log_softmax(net(torch.tensor([[0, 2 + u, 2 + nvar + a, 2 + w]]))[0, -1, 2 + nvar: 2 + nvar + nval], -1).numpy()
                    for b in range(nval):
                        p = math.exp(l1[a] + l2[b])
                        if flip: t[b, a] += p / 2
                        else: t[a, b] += p / 2
            out.append(t.tolist())
    return out

# ------------------------------------------------------------------ new metrics
def contextual_fraction(T):
    """CF = 1 - max sum b subject to M b <= v, b >= 0 (Abramsky, Barbosa, Mansfield 2017), for 4-cycle tables T"""
    from scipy.optimize import linprog
    rows = []; v = []
    for k, (i, j) in enumerate(CTX4):
        for a in (0, 1):
            for b in (0, 1): rows.append([1.0 if (c[i] == a and c[j] == b) else 0.0 for c in CELLS4]); v.append(float(T[k][a][b]))
    r = linprog(-np.ones(16), A_ub=np.array(rows), b_ub=np.array(v), bounds=[(0, None)] * 16, method="highs"); return float(1 + r.fun)

def analyse_run(c, W, x):
    nvar, nval, ctx, T, P = condition_data(c, W); c17 = dict(c); c17["kind"] = "G1" if c["kind"] in ("S", "G1") else c["kind"]
    if c["kind"] == "W2": c17.update(kind="W2")
    # reuse Study 17's metric code through a shim that supplies this condition's data
    class Wshim: pass
    r = {}; Q = np.array(x["joint"]["mean"])
    r["drift"] = float(np.mean([K17.tv(a, b) for a, b in itertools.combinations(x["joint"]["orders"], 2)]))
    if P is not None:
        Pt = np.array([float(v) for v in P]); Ps = K17.ipf(Q, T, ctx, nvar, nval)
        r.update(total=K17.kl(Pt, Q), infibre=K17.kl(Pt, Ps), offfibre=K17.kl(Ps, Q), tv_true_Q=K17.tv(Pt, Q), tv_true_Pstar=K17.tv(Pt, Ps))
        r["pyth_rel_err"] = abs(r["total"] - r["infibre"] - r["offfibre"]) / max(r["total"], 1e-300)
        if nvar == 4:
            sH, sL, d = K17.sections_4cycle(T); a_, b_ = K17.tv(Ps, sH), K17.tv(Ps, sL); r["iota"] = a_ / (a_ + b_) if a_ + b_ > 0 else 0.5
            lo, hi = G.interval(d, [1 if cc[0] == cc[1] else 0 for cc in G.CELLS]); r.update(q_true=K17.q_equal(Pt), q_Pstar=K17.q_equal(Ps), q_width=hi - lo)
            r["fibre_dim"] = G.fibre_dim(d)
    else:
        r.update(mismatch=K17.marg_mismatch(Q, T, ctx, nvar, nval), mstar=K17.mstar(T, ctx, nvar, nval))
    pt = x["pairs"]; r["fit"] = sum(K17.kl(np.array([float(v) for row in T[k] for v in row]), np.array(pt[k]).ravel()) for k in range(len(ctx)))
    cells = np.array(list(itertools.product(range(nval), repeat=nvar))); comb = 0.0
    for k, (i, j) in enumerate(ctx):
        mq = np.array([[Q[(cells[:, i] == a) & (cells[:, j] == b)].sum() for b in range(nval)] for a in range(nval)]); comb += K17.tv(mq.ravel(), np.array(pt[k]).ravel())
    r["combination_gap"] = comb
    if nvar == 4: r.update(cf_data=contextual_fraction(T), cf_model=contextual_fraction(pt))
    if c["kind"] == "W2":
        comp = np.zeros((nval, nval))
        for cc, p in zip(itertools.product(range(nval), repeat=3), Q): comp[cc[0], cc[2]] += p
        r["composite"] = comp.tolist()
    return r

def boot_ci(f, groups, B=2000, seed=SEED):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(B): vals.append(f([g[rng.integers(0, len(g), len(g))] for g in groups]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def analyse(runs, conds, W):
    R = {x["idx"]: analyse_run(conds[x["idx"]], W, x) for x in runs}; res = dict(runs=R)
    sel = lambda f: [i for i, c in enumerate(conds) if i in R and f(c)]
    sp = sel(lambda c: c["kind"] in ("S", "G1"))
    res["T1"] = all(R[i]["pyth_rel_err"] < 1e-6 for i in sp)
    res["T2"] = all(R[i]["tv_true_Pstar"] < 1e-6 for i in sel(lambda c: c["kind"] == "S" and c["fam"] == "F0"))
    res["T3"] = all(R[i]["mismatch"] >= R[i]["mstar"] - 1e-6 for i in R if "mismatch" in R[i])
    res["T4"] = all(abs(R[i]["q_Pstar"] - R[i]["q_true"]) <= R[i]["q_width"] + 1e-9 for i in sp if "q_width" in R[i])
    res["fibre_dims"] = {f: sorted({R[i]["fibre_dim"] for i in sel(lambda c, f=f: c["kind"] == "S" and c["fam"] == f)}) for f in ("F0", "F1", "F2")}
    g = {f: np.array([R[i]["tv_true_Q"] for i in sel(lambda c, f=f: c["kind"] == "S" and c["fam"] == f and c["variant"] == "consistent")]) for f in ("F0", "F1", "F2")}
    c10 = boot_ci(lambda gs: gs[1].mean() - gs[0].mean(), [g["F0"], g["F1"]]); c21 = boot_ci(lambda gs: gs[1].mean() - gs[0].mean(), [g["F1"], g["F2"]])
    res["H7"] = dict(means={f: float(v.mean()) for f, v in g.items()}, ci_F1_minus_F0=c10, ci_F2_minus_F1=c21, supported=bool(c10[0] > 0 and c21[0] > 0))
    def paired(fam_ok, va, vb, key):
        d = []
        for i in sel(lambda c: c["kind"] == "S" and fam_ok(c["fam"]) and c["variant"] == va):
            j = next(k for k in sel(lambda c, ci=conds[i]: c["kind"] == "S" and c["fam"] == ci["fam"] and c["face"] == ci["face"] and c["rep"] == ci["rep"] and c["variant"] == vb))
            d.append(R[i][key] - R[j][key])
        d = np.array(d); return float(d.mean()), boot_ci(lambda gs: gs[0].mean(), [d])
    m8, ci8 = paired(lambda f: f == "F0", "plain", "consistent", "offfibre"); res["H8"] = dict(mean_plain_minus_consistent=m8, ci=ci8, supported=bool(ci8[0] > 0))
    m9, ci9 = paired(lambda f: True, "plain", "plain_wd0", "offfibre"); res["H9"] = dict(mean_wd001_minus_wd0=m9, ci=ci9, supported=bool(ci9[0] > 0))
    defects = []; floors = []
    for w in range(6):
        comps = {(p, s): np.array(R[i]["composite"]) for i in sel(lambda c, w=w: c["kind"] == "W2" and c["world"] == w) for p, s in [(conds[i]["part"], conds[i]["rep"])]}
        dw = [K17.tv(np.kron(comps[("c1", s)], comps[("c2", s)]).ravel(), comps[("prod", s)].ravel()) for s in range(NREP)]
        fw = [K17.tv(comps[("prod", s)].ravel(), comps[("prod", t)].ravel()) for s, t in itertools.combinations(range(NREP), 2)]
        defects.append(float(np.mean(dw))); floors.append(float(np.mean(fw)))
    d = np.array(defects) - np.array(floors); ci10 = boot_ci(lambda gs: gs[0].mean(), [d])
    res["H10"] = dict(defects=defects, floors=floors, ci_excess=ci10, supported=bool(ci10[0] <= 0 <= ci10[1]), alternative=bool(ci10[0] > 0))
    io = [R[i]["iota"] for i in sel(lambda c: c["kind"] == "G1")]; res["H11"] = dict(share_below_half=float(np.mean([x < 0.5 for x in io])), supported=bool(np.mean([x < 0.5 for x in io]) >= 0.75))
    return res

def smoke():
    conds, W = conditions(); fam = s_faces(); out = dict(n_conditions=len(conds), kinds={k: sum(c["kind"] == k for c in conds) for k in ("S", "G1", "G2", "W2")})
    out["fibre dims of the chosen faces"] = {f: [G.fibre_dim_face(V) for V in fam[f]] for f in fam}
    # non-neural responders on one S face of each family: the maximum-entropy section, with its own pair tables
    for f in ("F0", "F1", "F2"):
        c = next(x for x in conds if x["kind"] == "S" and x["fam"] == f); nvar, nval, ctx, T, P = condition_data(c, W)
        sH, sL, d = K17.sections_4cycle(T); Q = 0.999 * sH + 0.001 / 16
        pairs = [[[float(sum(Q[k] for k, cc in enumerate(CELLS4) if cc[i] == a and cc[j] == b)) for b in (0, 1)] for a in (0, 1)] for (i, j) in CTX4]
        r = analyse_run(c, W, dict(joint=dict(mean=Q.tolist(), orders=[Q.tolist()] * 2), pairs=pairs))
        out[f"{f} maxent responder"] = {k: round(v, 6) for k, v in r.items() if k in ("infibre", "offfibre", "tv_true_Pstar", "pyth_rel_err", "fibre_dim", "cf_data", "fit", "combination_gap")}
    pr = [[[Fr(1, 2), 0], [0, Fr(1, 2)]]] * 3 + [[[0, Fr(1, 2)], [Fr(1, 2), 0]]]; out["CF of the PR box (expect 1)"] = round(contextual_fraction(pr), 6)
    # a 40-step consistent training run on a dummy chain world outside the study, and its pair tables
    rng = random.Random(3); Wd = dict(W2=[(K17.chain_world(rng), K17.chain_world(rng))], S={}, G1=[], G2=[])
    t = time.time(); net, nvar, nval = train_one(dict(kind="W2", world=0, part="c1", variant="consistent", rep=0, N=None, seed=3, idx=-1), Wd, steps=40)
    J = K17.implied(net, nvar, nval); pt = pair_tables(net, nvar, nval, [(0, 1), (1, 2)])
    out["dummy consistent training"] = dict(seconds=round(time.time() - t, 1), joint_sum=round(sum(J["mean"]), 9), pair_sums=[round(float(np.sum(p)), 9) for p in pt])
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--out"); ap.add_argument("--runs"); ap.add_argument("--only", type=int); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.cmd == "plan":
        conds, _ = conditions(); print(len(conds), {k: sum(c["kind"] == k for c in conds) for k in ("S", "G1", "G2", "W2")})
    elif a.cmd == "smoke": smoke()
    elif a.cmd == "train":
        import torch
        conds, W = conditions(); done = set(); os.makedirs(os.path.join(HERE, "weights18"), exist_ok=True)
        if a.resume and os.path.exists(a.out): done = {json.loads(l)["idx"] for l in open(a.out)}
        elif os.path.exists(a.out) and os.path.getsize(a.out) > 0: raise SystemExit(f"{a.out} exists; use --resume")
        todo = [c for c in conds if c["idx"] not in done and (a.only is None or c["idx"] % 2 == a.only)]
        with open(a.out, "a") as fh:
            for c in todo:
                t = time.time(); net, nvar, nval = train_one(c, W); J = K17.implied(net, nvar, nval)
                ctx = CTX4 if nvar == 4 else [(0, 1), (1, 2)]; pt = pair_tables(net, nvar, nval, ctx)
                torch.save(net.state_dict(), os.path.join(HERE, "weights18", f"{c['idx']}.pt"))
                fh.write(json.dumps(dict(idx=c["idx"], joint=J, pairs=pt)) + "\n"); fh.flush()
                print(f"{c['idx']} {c['kind']} {c['variant']} done in {time.time() - t:.0f}s", flush=True)
    elif a.cmd == "score":
        conds, W = conditions(); runs = [json.loads(l) for l in open(a.runs)]
        json.dump(analyse(runs, conds, W), open(a.out, "w"), indent=1, default=float)
