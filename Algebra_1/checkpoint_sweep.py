"""
DOES THE STABILIZER SUBGROUP OF THE STRAND GROW WITH TRAINING?

One training run on cfg. At a geometric ladder of checkpoints, pause and run the
full symmetry suite plus the r-diagnostic. Measurement never calls opt.step(),
and theta is restored after every backward, so the trajectory is untouched.

Two questions, both of which are currently load-bearing and untested:

 (1) Does the deterministic regime (r ~ 1) ever appear? The 80-step checkpoint
     had r_90 = 0.061, i.e. the entire system in the stochastic phase, which is
     why the r-stratification prediction sampled only one asymptotic regime.
 (2) Does agreement under the STRUCTURE-preserving transforms rise relative to
     the histogram control lexctl? If rename-minus-lexctl grows, the stabilizer
     is acquiring grammatical invariance. If it stays flat at ~0.02, strand
     generation never internalises syntax at this scale.

The discriminator is the GAP rename - lexctl, not rename alone. rename alone is
mostly histogram; lexctl subtracts it.
"""
import sys, json, time, numpy as np, torch
from harness import build, param_groups

CKPTS = [20, 40, 80, 160, 320, 640, 1280]
NTRIAL = int(sys.argv[1]) if len(sys.argv) > 1 else 12
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 17
NSUB = 20000

t0 = time.time()
g, model, get_batch = build("cfg")
torch.manual_seed(SEED)
P, block_mask, emb_mask, named = param_groups(model)
LR = g["LR"]; BATCH = g["BATCH"]; SEQ = g["SEQ"]; eval_val = g["eval_val"]
NEED = BATCH * SEQ + 1

pairs = json.load(open("/home/claude/work/bundle/corpora/cfg_pairs.json"))
rng = np.random.default_rng(SEED)
SUB_np = rng.choice(np.flatnonzero(block_mask), NSUB, replace=False)
SUB = torch.tensor(SUB_np)
CONDS = ["rename", "comm", "ws", "lexctl", "pert1", "pert4", "indep"]

opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95), weight_decay=0.1)


def gradv():
    return torch.cat([(p.grad.flatten() if p.grad is not None
                       else torch.zeros(p.numel())) for _, p in named])


def make_batch(toks):
    t = torch.tensor(toks[:NEED], dtype=torch.long)
    return t[:BATCH * SEQ].view(BATCH, SEQ), t[1:BATCH * SEQ + 1].view(BATCH, SEQ)


def stream_from(key, start):
    L = pairs[key]; out = []; i = start
    while len(out) < NEED:
        out += list((L[i % len(L)] + ";").encode("ascii")); i += 1
    return out


def disc_r():
    m, v = [], []
    for p in model.parameters():
        st = opt.state[p]
        if "exp_avg" not in st:
            return None
        m.append(st["exp_avg"].flatten()); v.append(st["exp_avg_sq"].flatten())
    m = torch.cat(m); v = torch.cat(v)
    return (m.abs() / (v.sqrt() + 1e-12)).numpy()[SUB_np]


def measure(step):
    theta = torch.cat([p.data.flatten() for _, p in named]).clone()

    def restore():
        with torch.no_grad():
            i = 0
            for _, p in named:
                k = p.numel(); p.data.copy_(theta[i:i + k].view_as(p)); i += k

    def strand(x, y):
        model.zero_grad(); _, l = model(x, y); l.backward()
        s = torch.sign(gradv()[SUB]).numpy().astype(np.int8)
        restore()
        return s

    r = disc_r()
    dec = np.clip((np.searchsorted(np.sort(r), r) / NSUB * 10).astype(int), 0, 9)
    agg = {c: [] for c in CONDS}
    prof = {c: [] for c in CONDS}
    for _ in range(NTRIAL):
        start = int(rng.integers(0, len(pairs["base"])))
        sb = strand(*make_batch(stream_from("base", start)))
        var = {k: stream_from(k, start) for k in ["rename", "comm", "ws", "pert1", "pert4"]}
        lc = np.array(stream_from("rename", start)); rng.shuffle(lc); var["lexctl"] = lc.tolist()
        var["indep"] = stream_from("base", int(rng.integers(0, len(pairs["base"]))))
        for c in CONDS:
            s = strand(*make_batch(var[c]))
            ag = (sb == s)
            agg[c].append(float(ag.mean()))
            prof[c].append([float(ag[dec == d].mean()) for d in range(10)])
    model.zero_grad()
    out = {c: float(np.mean(agg[c])) for c in CONDS}
    out["_sd"] = {c: float(np.std(agg[c])) for c in CONDS}
    out["_prof_indep"] = np.array(prof["indep"]).mean(0).tolist()
    out["_r"] = {f"p{q}": float(np.percentile(r, q)) for q in (10, 50, 70, 90, 99, 100)}
    out["_r_frac_gt_half"] = float((r > 0.5).mean())
    out["step"] = step
    out["val"] = float(eval_val(model, n=10)); model.train()
    return out


results = []
done = 0
for ck in CKPTS:
    while done < ck:
        x, y = get_batch(); _, l = model(x, y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        done += 1
    res = measure(ck)
    res["train_loss"] = float(l.item())
    results.append(res)
    print(f"  step {ck:>5}  val {res['val']:.3f}  r90 {res['_r']['p90']:.3f}  "
          f"rename {res['rename']:.3f}  lexctl {res['lexctl']:.3f}  "
          f"GAP {res['rename']-res['lexctl']:+.3f}  ws {res['ws']:.3f}  "
          f"indep {res['indep']:.3f}   [{time.time()-t0:.0f}s]", flush=True)

json.dump(results, open(f"/home/claude/work/res_ckpt_s{SEED}.json", "w"), indent=1)

print("\n" + "=" * 92)
print("  STABILIZER EVOLUTION")
print("=" * 92)
print(f"{'step':>6}{'val':>8}{'r_p90':>8}{'r>0.5':>8}"
      f"{'rename':>8}{'lexctl':>8}{'GAP':>8}{'comm':>8}{'ws':>8}{'pert1':>8}{'indep':>8}")
for r in results:
    print(f"{r['step']:>6}{r['val']:>8.3f}{r['_r']['p90']:>8.3f}{r['_r_frac_gt_half']:>8.4f}"
          f"{r['rename']:>8.3f}{r['lexctl']:>8.3f}{r['rename']-r['lexctl']:>+8.3f}"
          f"{r['comm']:>8.3f}{r['ws']:>8.3f}{r['pert1']:>8.3f}{r['indep']:>8.3f}")
print("\n  GAP = rename - lexctl = structural invariance net of histogram.")
print("  r-decile spread of the independent floor (deterministic-regime check):")
for r in results:
    p = r["_prof_indep"]
    print(f"    step {r['step']:>5}: d0 {p[0]:.3f} -> d9 {p[9]:.3f}   spread {p[9]-p[0]:+.3f}")
print(f"\n  time {time.time()-t0:.0f}s")
