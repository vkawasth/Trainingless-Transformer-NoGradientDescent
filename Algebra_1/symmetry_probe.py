"""
SYMMETRY INVARIANCE OF THE STRAND, STRATIFIED BY THE DISC.

For matched batch pairs (same underlying parse trees, different surface form)
measure sign agreement between strands -- overall, and split by r-decile, where
r = |m|/sqrt(v) is read from the frozen Adam state.

The stratification is the whole point. Under the disc/strand statement:
  top-r decile     -> sign is disc-determined, agreement ~1 for ANY batch pair
  bottom-r decile  -> noise-dominated, agreement ~chance for ANY pair
  middle band      -> the only batch-conditioned region
So a real symmetry invariance must show its excess in the MIDDLE. Excess in the
top band is disc-determined agreement misread as invariance -- which is exactly
how an unstratified version of this experiment would mislead.

Conditions, all against the same base batch:
  rename   exact symmetry, tree identical, token stats identical, 85% surface changed
  comm     exact symmetry, tree mirrored,  stats ~identical,      78% changed
  ws       exact symmetry, tree identical, stats CHANGED a lot
  lexctl   rename's tokens, shuffled -> same histogram, no structure  [KEY CONTROL]
  pert1..4 known structural distance ladder: terminal < operator < subtree < resample
  indep    unrelated batch (floor)
"""
import sys, json, time, numpy as np, torch
from harness import build, param_groups

NTRIAL = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 17
NSUB = 20000
TRAIN_STEPS = 80

t0 = time.time()
g, model, get_batch = build("cfg")
torch.manual_seed(SEED)
P, block_mask, emb_mask, named = param_groups(model)
LR = g["LR"]; BATCH = g["BATCH"]; SEQ = g["SEQ"]
NEED = BATCH * SEQ + 1

pairs = json.load(open("/home/claude/work/bundle/corpora/cfg_pairs.json"))
KEYS = ["base", "rename", "comm", "ws", "pert1", "pert2", "pert3", "pert4"]

# ---- train to checkpoint, freeze, read the disc off the optimizer ----
opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95), weight_decay=0.1)
for _ in range(TRAIN_STEPS):
    x, y = get_batch(); _, l = model(x, y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
loss_ckpt = float(l.item())

m_all, v_all = [], []
for p in model.parameters():
    st = opt.state[p]
    m_all.append(st["exp_avg"].flatten().clone())
    v_all.append(st["exp_avg_sq"].flatten().clone())
m_all = torch.cat(m_all); v_all = torch.cat(v_all)
r_all = (m_all.abs() / (v_all.sqrt() + 1e-12)).numpy()

rng = np.random.default_rng(SEED)
SUB = torch.tensor(rng.choice(np.flatnonzero(block_mask), NSUB, replace=False))
r_sub = r_all[SUB.numpy()]
dec = np.clip((np.searchsorted(np.sort(r_sub), r_sub) / NSUB * 10).astype(int), 0, 9)

theta = torch.cat([p.data.flatten() for _, p in named]).clone()


def restore():
    with torch.no_grad():
        i = 0
        for _, p in named:
            k = p.numel(); p.data.copy_(theta[i:i + k].view_as(p)); i += k


def gradv():
    return torch.cat([(p.grad.flatten() if p.grad is not None
                       else torch.zeros(p.numel())) for _, p in named])


def make_batch(toks):
    t = torch.tensor(toks[:NEED], dtype=torch.long)
    return t[:BATCH * SEQ].view(BATCH, SEQ), t[1:BATCH * SEQ + 1].view(BATCH, SEQ)


def stream_from(key, start):
    """Concatenate rendered expressions from `key` starting at index `start`."""
    L = pairs[key]; out = []; i = start
    while len(out) < NEED:
        out += list((L[i % len(L)] + ";").encode("ascii")); i += 1
    return out


def strand(x, y):
    model.zero_grad(); _, l = model(x, y); l.backward()
    s = torch.sign(gradv()[SUB]).numpy().astype(np.int8)
    restore()
    return s


CONDS = ["rename", "comm", "ws", "lexctl", "pert1", "pert2", "pert3", "pert4", "indep"]
acc = {c: [] for c in CONDS}
accd = {c: np.zeros((0, 10)) for c in CONDS}
rows = {c: [] for c in CONDS}
surf = {c: [] for c in CONDS}

for trial in range(NTRIAL):
    start = int(rng.integers(0, len(pairs["base"])))
    s_base = strand(*make_batch(stream_from("base", start)))
    variants = {}
    for k in KEYS[1:]:
        variants[k] = stream_from(k, start)
    # lexctl: rename's tokens with structure destroyed
    lc = np.array(stream_from("rename", start)); rng.shuffle(lc)
    variants["lexctl"] = lc.tolist()
    # indep: unrelated stretch of the same corpus
    variants["indep"] = stream_from("base", int(rng.integers(0, len(pairs["base"]))))

    xb, _ = make_batch(stream_from("base", start))
    hb = torch.bincount(xb.flatten(), minlength=256).float()
    for c in CONDS:
        xv, yv = make_batch(variants[c])
        s = strand(xv, yv)
        ag = (s_base == s)
        acc[c].append(float(ag.mean()))
        rows[c].append([float(ag[dec == d].mean()) for d in range(10)])
        hv = torch.bincount(xv.flatten(), minlength=256).float()
        surf[c].append((float((xb == xv).float().mean()),
                        float(torch.nn.functional.cosine_similarity(hb, hv, dim=0))))

print("=" * 78)
print(f"  SYMMETRY INVARIANCE, r-STRATIFIED   (cfg, {NTRIAL} trials, loss {loss_ckpt:.3f})")
print("=" * 78)
print(f"\n  r-decile agreement profile of the INDEPENDENT floor (disc-determined baseline):")
ind = np.array(rows["indep"]).mean(0)
print("   decile " + "".join(f"{d:>7}" for d in range(10)))
print("   agree  " + "".join(f"{v:>7.3f}" for v in ind))

print(f"\n  {'condition':>9}{'overall':>9}{'excess':>9}   {'low r (0-2)':>12}{'mid r (3-6)':>13}{'high r (7-9)':>13}")
base_ind = np.mean(acc["indep"])
res = {}
for c in CONDS:
    prof = np.array(rows[c]).mean(0)
    ov = np.mean(acc[c])
    lo, mid, hi = prof[:3].mean(), prof[3:7].mean(), prof[7:].mean()
    lo_e = lo - ind[:3].mean(); mid_e = mid - ind[3:7].mean(); hi_e = hi - ind[7:].mean()
    res[c] = dict(overall=ov, excess=ov - base_ind, lo=lo, mid=mid, hi=hi,
                  lo_e=lo_e, mid_e=mid_e, hi_e=hi_e,
                  sd=float(np.std(acc[c])), prof=prof.tolist())
    sp = np.array(surf[c]).mean(0)
    res[c]["pos_ident"] = float(sp[0]); res[c]["hist_cos"] = float(sp[1])
    print(f"  {c:>9}{ov:>9.3f}{ov-base_ind:>+9.3f}   "
          f"{lo:>6.3f}({lo_e:+.3f}){mid:>6.3f}({mid_e:+.3f}){hi:>6.3f}({hi_e:+.3f})")

print(f"\n  {'condition':>9}{'agree':>8}{'pos-ident':>11}{'hist-cos':>10}   (surface-distance confound)")
for c in CONDS:
    print(f"  {c:>9}{res[c]['overall']:>8.3f}{res[c]['pos_ident']:>11.3f}{res[c]['hist_cos']:>10.3f}")
_a=np.array([res[c]['overall'] for c in CONDS]); _p=np.array([res[c]['pos_ident'] for c in CONDS]); _h=np.array([res[c]['hist_cos'] for c in CONDS])
print(f"\n  corr(agreement, positional identity) = {np.corrcoef(_a,_p)[0,1]:+.3f}")
print(f"  corr(agreement, histogram cosine)    = {np.corrcoef(_a,_h)[0,1]:+.3f}")
print("\n  excess = vs independent-batch floor. Parenthesised = excess within band.")
print("  PREDICTION: real invariance -> excess concentrated in MID band.")
print("              excess in HIGH band -> disc-determined, not symmetry.")
json.dump({"loss": loss_ckpt, "ntrial": NTRIAL, "seed": SEED,
           "indep_profile": ind.tolist(), "res": res},
          open(f"/home/claude/work/res_sym_s{SEED}.json", "w"), indent=1)
print(f"\n  time {time.time()-t0:.0f}s")
