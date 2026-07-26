"""
Intrinsic-dimension probe: is the batch->strand map low-dimensional?

Protocol follows scripts/strand_manifold.py, with three deliberate changes,
each of which is reported alongside the original-protocol number so the
comparison stays auditable:

 1. NB raised 200 -> 600. In the original, Z has D+VOCAB features (512 here)
    against h=150 training rows, so BOTH the linear fit and the 800-dim random
    feature fit are underdetermined min-norm interpolators. "Nonlinear does not
    beat linear / overfits" is then a statement about two interpolators, not
    about the map. 600 batches gives 360 train rows.
 2. Ridge, with the penalty tuned on a held-out split, for linear AND nonlinear
    alike. This gives each model its best shot; an untuned comparison is not a
    discriminator. Original untuned lstsq numbers are reported too.
 3. Strand coordinates restricted to block parameters (primary arm) with the
    embedding measured separately, per the epsilon-table confound: the embedding
    is non-perturbative because it memorises, and corpus structure changes that
    regime directly.

Discriminator: does nonlinear beat linear as structure rises? Read primarily off
the PAIRED shuffle contrasts (english vs english_shuf, code vs code_shuf), which
hold vocabulary and unigram statistics fixed. The cross-corpus ordering is
confounded: live-vocabulary sizes are iid 256, code 95, repeat 97, english 27.
"""
import sys, time, json, numpy as np, torch
from harness import build, param_groups

CORPUS = sys.argv[1]
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 600
NSUB = 20000
TRAIN_STEPS = 80
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 17

t0 = time.time()
g, model, get_batch = build(CORPUS)
torch.manual_seed(SEED)
P, block_mask, emb_mask, named = param_groups(model)
LR = g["LR"]; VOCAB = g["VOCAB"]

rng = np.random.default_rng(SEED)
block_idx = np.flatnonzero(block_mask)
emb_idx = np.flatnonzero(emb_mask)
SUB_BLOCK = torch.tensor(rng.choice(block_idx, NSUB, replace=False))
SUB_EMB = torch.tensor(rng.choice(emb_idx, min(NSUB, len(emb_idx)), replace=False))


def flat():
    return torch.cat([p.data.flatten() for _, p in named]).clone()


def gradv():
    return torch.cat([(p.grad.flatten() if p.grad is not None
                       else torch.zeros(p.numel())) for _, p in named]).clone()


# ---- train to checkpoint, then freeze ----
torch.manual_seed(SEED)
opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95), weight_decay=0.1)
for _ in range(TRAIN_STEPS):
    x, y = get_batch(); _, l = model(x, y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
loss_ckpt = float(l.item())
theta = flat()

# ---- sweep batches at frozen theta ----
Z, Sb, Se = [], [], []
emb_w = model.te.weight.detach()
for _ in range(NB):
    x, y = get_batch()
    model.zero_grad(); _, l = model(x, y); l.backward()
    gv = gradv()
    Sb.append(torch.sign(gv[SUB_BLOCK]).numpy().astype(np.int8))
    Se.append(torch.sign(gv[SUB_EMB]).numpy().astype(np.int8))
    hist = torch.bincount(x.flatten(), minlength=VOCAB).float(); hist /= hist.sum()
    ze = emb_w[x.flatten()].mean(0)
    Z.append(torch.cat([ze, hist]).numpy())
    with torch.no_grad():                      # restore theta (defensive)
        i = 0
        for _, p in named:
            k = p.numel(); p.data.copy_(theta[i:i + k].view_as(p)); i += k

Z = np.stack(Z).astype(np.float64)
Sb = np.stack(Sb).astype(np.float32)
Se = np.stack(Se).astype(np.float32)

# ---- splits: 60 / 15 / 25 ----
ntr, nva = int(NB * .60), int(NB * .15)
tr = slice(0, ntr); va = slice(ntr, ntr + nva); te = slice(ntr + nva, NB)

mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
Zs = (Z - mu) / sd


def agree(pred, true):
    m = true != 0
    return float((np.sign(pred) == np.sign(true))[m].mean())


def ridge_path(X, Y, alphas):
    """Fit ridge for a grid of alphas; pick alpha on val, score on test."""
    Xtr, Ytr = X[tr], Y[tr]
    ym = Ytr.mean(0); Yc = (Ytr - ym).astype(np.float64)
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    A = U.T @ Yc                                    # n x m
    XvV, XtV = X[va] @ Vt.T, X[te] @ Vt.T
    best = (-1, None)
    for a in alphas:
        d = s / (s ** 2 + a)
        pv = (XvV * d) @ A + ym
        sc = agree(pv, Y[va])
        if sc > best[0]:
            best = (sc, a, d)
    _, a_best, d = best
    pt = (XtV * d) @ A + ym
    return agree(pt, Y[te]), a_best


def lstsq_orig(X, Y):
    """Original protocol: unregularised lstsq, 75/25 split, min-norm."""
    h = NB * 3 // 4
    W = np.linalg.lstsq(np.hstack([X[:h], np.ones((h, 1))]), Y[:h], rcond=None)[0]
    p = np.hstack([X[h:], np.ones((NB - h, 1))]) @ W
    return agree(p, Y[h:])


def knn(X, Y, k):
    ag = []
    for i in range(ntr + nva, NB):
        d = ((X[:ntr] - X[i]) ** 2).sum(1)
        nn = np.argsort(d)[:k]
        ag.append(agree(np.sign(Y[nn].mean(0)), Y[i]))
    return float(np.mean(ag))


ALPHAS = np.logspace(-3, 6, 19)
rf = np.random.default_rng(SEED)
D_RF = 800
Wr = rf.normal(size=(Zs.shape[1], D_RF)) * 0.5
br = rf.uniform(0, 2 * np.pi, D_RF)
Phi = np.cos(Zs @ Wr + br)
Phi = (Phi - Phi[tr].mean(0)) / (Phi[tr].std(0) + 1e-9)

out = {"corpus": CORPUS, "NB": NB, "loss_at_ckpt": loss_ckpt,
       "live_vocab": int(len(np.unique(np.array(json.load(
           open(f"/home/claude/work/bundle/corpora/{CORPUS}.json"))))))}

for arm, S in [("block", Sb), ("emb", Se)]:
    r = {}
    r["base_rate_pos"] = float((S > 0).mean())
    r["lin_orig"] = lstsq_orig(Zs, S)
    r["nl_orig"] = lstsq_orig(Phi, S)
    r["lin_ridge"], r["lin_alpha"] = ridge_path(Zs, S, ALPHAS)
    r["nl_ridge"], r["nl_alpha"] = ridge_path(Phi, S, ALPHAS)
    r["gain_ridge"] = r["nl_ridge"] - r["lin_ridge"]
    for k in (1, 5, 15):
        r[f"knn{k}"] = knn(Zs, S, k)
    Sc = S - S.mean(0)
    sv = np.linalg.svd(Sc, compute_uv=False); sv2 = sv ** 2
    r["PR"] = float((sv2.sum() ** 2) / (sv2 ** 2).sum())
    ds, ss = [], []
    step = max(1, NB // 200)
    for i in range(0, NB, step):
        for j in range(i + step, NB, step):
            ds.append(np.sqrt(((Zs[i] - Zs[j]) ** 2).sum()))
            ss.append(float((S[i] == S[j]).mean()))
    r["corr_dist_dissim"] = float(np.corrcoef(ds, 1 - np.array(ss))[0, 1])
    r["mean_pair_agree"] = float(np.mean(ss))
    out[arm] = r

out["secs"] = time.time() - t0
json.dump(out, open(f"/home/claude/work/res_{CORPUS}_s{SEED}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
