#!/usr/bin/env python3
"""RHM2 -- 2-layer attention-only transformer with induction-circuit and
holonomy instrumentation.

    python3 build_corpus_rhm.py --out /tmp
    python3 rhm2.py --null --seeds 5        # Phase 1: untrained null floor
    python3 rhm2.py --steps 4000 --seed 1   # training run

PHASE 1 IS THE POINT OF THIS FILE RIGHT NOW. Before any training, we need to
know what D_ord, D_pos and D_perm read on an UNTRAINED model. Two reasons:
  1. If D_ord >> D_pos at initialisation, the sequence construction leaks and
     the metric is broken before training starts.
  2. It gives the scale D takes when nothing has been learned.
The untrained spread is NOT the yardstick for the trained comparison -- that
needs the SEM of the paired difference across trained seeds, because training
changes the variance. This file reports both separately and never pools them.

READOUT NOTES
-------------
h1 vs h2: h1 is the residual stream AFTER BLOCK 0, h2 after block 1. The
hypothesis is that block 0 resolves the grouping and block 1 uses it, so h1 is
the quantity of interest; h2 is logged alongside because assuming which block
does what is exactly the kind of thing that should be measured.

D is reported THREE ways because the raw L2 scales with ||h||, and if C_ord and
C_unord differ in token frequency an imbalance shows up as a fake effect:
  D_l2     raw ||h(a,b) - h(b,a)||
  D_rel    same, divided by mean ||h||
  D_cos    cosine distance
D_rel and D_cos are the ones to believe.

C_unord is the POSITIONAL control and C_perm is the VOCABULARY control. They
control different things. Never pool them into one denominator.
"""
import json, math, argparse, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp")
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--n-heads", type=int, default=4)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--seeds", type=int, default=5, help="how many seeds for --null")
ap.add_argument("--null", action="store_true", help="Phase 1: untrained only")
ap.add_argument("--eval-every", type=int, default=200)
ap.add_argument("--mlp", action="store_true",
                help="add a 4x GELU MLP sublayer to every block")
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--task", default="rhm", choices=["rhm", "copy"],
                help="copy = random S1S1 repeats; INSTRUMENT CALIBRATION only")
ap.add_argument("--copy-k", type=int, default=8)
ap.add_argument("--save", default="", help="write the trained weights here")
ap.add_argument("--load", default="", help="skip training; analyse this checkpoint")
ap.add_argument("--rev", action="store_true",
                help="reversible (RevNet-style) blocks: the block map is exactly "
                     "invertible, so activations can be reconstructed from the output")
a = ap.parse_args()

# ----------------------------------------------------------------- data
need = ["train_ids.json", "val_ids.json", "rhm_meta.json"]
for f in need:
    if not os.path.exists(f"{a.data}/{f}"):
        sys.exit(f"missing {a.data}/{f}; run: python3 build_corpus_rhm.py --out {a.data}")
META = json.load(open(f"{a.data}/rhm_meta.json"))
train = torch.tensor(json.load(open(f"{a.data}/train_ids.json")), dtype=torch.long)
val   = torch.tensor(json.load(open(f"{a.data}/val_ids.json")),   dtype=torch.long)

SEQ      = META["seq_len"]
NLEAF    = META["nleaf"]
PROBE    = NLEAF                      # appended probe token id
VOCAB    = NLEAF + 1
POS_KIND = META["pos_kind"]
CTX      = SEQ if a.task == "rhm" else 2 * a.copy_k

# Sequence-ALIGNED batches. The CE split by position kind is only meaningful if
# position-in-window equals position-in-grammar-sequence, so offsets must be
# multiples of SEQ. Unaligned chunking would silently scramble the labels.
#
# The target at the LAST window position is the first token of the NEXT grammar
# sequence -- a fresh uniform root draw, irreducibly ~ln(v) nats, unrelated to
# the current tree. It is masked with -100 (ignore_index) rather than counted,
# since it was inflating the `higher` component.
IGNORE = -100

def get_batch(split="train", bs=None):
    bs = bs or a.batch
    if a.task == "copy":
        # INSTRUMENT CALIBRATION ONLY. Random S1S1 repeats: the task induction
        # heads exist for. If ind does not exceed ~0.8 here, the scoring code
        # is broken and every induction number reported so far is meaningless.
        K = a.copy_k
        r = torch.randint(0, NLEAF, (bs, K))
        x = torch.cat([r, r], 1)
        y = torch.cat([x[:, 1:], torch.full((bs, 1), IGNORE)], 1)
        y[:, :K] = IGNORE          # only the 2nd copy is predictable in-context
        return x, y
    src = train if split == "train" else val
    nseq = src.numel() // SEQ
    j = torch.randint(0, nseq - 1, (bs,)) * SEQ
    x = torch.stack([src[k:k+CTX] for k in j])
    y = torch.stack([src[k+1:k+1+CTX] for k in j]).clone()
    y[:, -1] = IGNORE              # target belongs to the next sequence
    return x, y

# ---------------------------------------------------------------- model
class Attn(nn.Module):
    def __init__(s, d, nh):
        super().__init__()
        s.nh, s.dh = nh, d // nh
        s.q, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4))
    def forward(s, x, need_attn=False):
        B, T, D = x.shape
        def sp(t): return t.view(B, T, s.nh, s.dh).transpose(1, 2)
        q, k, v = sp(s.q(x)), sp(s.k(x)), sp(s.v(x))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(s.dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        out = s.o((att @ v).transpose(1, 2).contiguous().view(B, T, D))
        return (out, att) if need_attn else (out, None)

class Block(nn.Module):
    """Attention, optionally followed by an MLP sublayer.

    Attention MOVES information across positions; it can route the
    representations of a and b to one place. It has no mechanism for mapping
    that pair to a NEW symbol P_i whose representation is outside the span of
    what arrived. RHM tree reduction is exactly that lookup, which is what MLP
    key-value memories do. Hence --mlp for the hierarchy, attention-only to
    show the contrast.

    (Attention-only is NOT merely correlational: the QK path is quadratic and
    the softmax nonlinear, and a 2-layer attention-only model implements the
    induction algorithm. The limitation here is specific -- no content-
    conditional feature construction -- not an absence of learning.)"""
    def __init__(s, d, nh, mlp=False):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.at = Attn(d, nh)
        s.mlp = None
        if mlp:
            s.ln2 = nn.LayerNorm(d)
            s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x, need_attn=False):
        o, att = s.at(s.ln(x), need_attn)
        x = x + o
        if s.mlp is not None:
            x = x + s.mlp(s.ln2(x))
        return x, att

class RevBlock(nn.Module):
    """Reversible block, channel-split (Reformer/RevNet form).

        y1 = x1 + Attn(x2)
        y2 = x2 + MLP(y1)
    inverse:
        x2 = y2 - MLP(y1)
        x1 = y1 - Attn(x2)

    The split is over CHANNELS, not sequence positions: splitting positions
    would let the second half see the future and break causality.

    What this does and does not buy:
      DOES  the block map R^d -> R^d is a bijection, so no information is
            destroyed between blocks and any activation can be recomputed
            from the output. Inverses exist, which is what a groupoid needs.
      NOT   it does not make attention itself invertible. The softmax mixture
            is still many-to-one; reversibility comes from the additive
            coupling around it. Two different value sets can still produce the
            same Attn(x2), and the inverse recovers x2, not that ambiguity.
    """
    def __init__(s, d, nh, mlp=True):
        super().__init__()
        assert d % 2 == 0
        h = d // 2
        s.ln_a = nn.LayerNorm(h); s.at = Attn(h, nh)
        s.ln_b = nn.LayerNorm(h)
        s.mlp = nn.Sequential(nn.Linear(h, 4*h), nn.GELU(), nn.Linear(4*h, h))
    def _f(s, x2, need_attn=False):
        o, att = s.at(s.ln_a(x2), need_attn); return o, att
    def _g(s, y1):
        return s.mlp(s.ln_b(y1))
    def forward(s, x, need_attn=False):
        x1, x2 = x.chunk(2, dim=-1)
        o, att = s._f(x2, need_attn)
        y1 = x1 + o
        y2 = x2 + s._g(y1)
        return torch.cat([y1, y2], -1), att
    def inverse(s, y):
        y1, y2 = y.chunk(2, dim=-1)
        x2 = y2 - s._g(y1)
        o, _ = s._f(x2)
        x1 = y1 - o
        return torch.cat([x1, x2], -1)

class LM2(nn.Module):
    def __init__(s, d=64, nh=4, vocab=VOCAB, ctx=CTX, nl=2, mlp=False, rev=False):
        super().__init__()
        s.te = nn.Embedding(vocab, d)
        s.pe = nn.Embedding(ctx + 4, d)          # absolute positions
        s.rev = rev
        s.blocks = nn.ModuleList([
            (RevBlock(d, nh) if rev else Block(d, nh, mlp)) for _ in range(nl)])
        s.lnf = nn.LayerNorm(d)
        s.head = nn.Linear(d, vocab, bias=False)
    def forward(s, idx, targets=None, need_attn=False, need_hidden=False):
        B, T = idx.shape
        h = s.te(idx) + s.pe(torch.arange(T, device=idx.device))[None]
        hs, atts = [], []
        for blk in s.blocks:
            h, at = blk(h, need_attn)
            hs.append(h); atts.append(at)
        h1, h2 = hs[0], hs[-1]      # h1 = after FIRST block, h2 = after LAST
        logits = s.head(s.lnf(h2))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=IGNORE)
        out = {"logits": logits, "loss": loss}
        if need_attn:   out["attn"] = atts
        if need_hidden: out["h1"], out["h2"] = h1, h2
        return out

# ------------------------------------------------- CE split by position kind
@torch.no_grad()
def eval_split(model, nb=20):
    """CE on `local` (sibling-determined) vs `higher` (needs level>=2) targets.

    Logits at window position j predict the token at position j+1, so the kind
    of a prediction is POS_KIND[j+1], not POS_KIND[j]."""
    model.eval()
    tot = {"local": [0.0, 0], "higher": [0.0, 0]}
    for _ in range(nb):
        x, y = get_batch("val")
        lg = model(x)["logits"]
        ls = F.cross_entropy(lg.view(-1, VOCAB), y.reshape(-1),
                             reduction="none", ignore_index=IGNORE).view(x.shape)
        keep = (y != IGNORE)
        for j in range(x.shape[1] - 1):          # last target is masked
            kind = POS_KIND[(j + 1) % SEQ]
            m = keep[:, j]
            tot[kind][0] += float((ls[:, j] * m).sum()); tot[kind][1] += int(m.sum())
    model.train()
    r = {k: (v[0] / max(v[1], 1)) for k, v in tot.items()}
    r["all"] = (tot["local"][0] + tot["higher"][0]) / \
               max(tot["local"][1] + tot["higher"][1], 1)
    return r

# --------------------------------------------------- induction-circuit scores
@torch.no_grad()
def induction_scores(model, K=8, bs=64, seed=0):
    """Prefix-match and previous-token scores on RANDOM repeated sequences.

    Sequences are [r_1..r_K, r_1..r_K] with r drawn uniformly from the leaf
    vocabulary. These are NOT grammar sequences: the point is to measure a
    mechanism that must use the prompt, which memorising the grammar cannot
    supply. Scores are mean attention mass on the target position.

      prev_token(head)  attention from t to t-1
      induction(head)   attention from position K+i to position i+1, i.e. the
                        token that FOLLOWED the earlier occurrence of the
                        current token
    """
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(0, NLEAF, (bs, K), generator=g)
    x = torch.cat([r, r], 1)                       # length 2K
    att = model(x, need_attn=True)["attn"]         # [layer][B, nh, T, T]
    out = {}
    for li, A in enumerate(att):
        nh = A.shape[1]
        for hd in range(nh):
            M = A[:, hd]
            prev = torch.stack([M[:, t, t-1] for t in range(1, 2*K)], 1).mean()
            ind = torch.stack([M[:, K+i, i+1] for i in range(0, K-1)], 1).mean()
            out[f"L{li}H{hd}"] = (float(prev), float(ind))
    return out

# ----------------------------------------------------------- holonomy suite
def _probe_batch(pairs):
    """[a][b][PROBE] and [b][a][PROBE]. The probe holds the readout position
    fixed: without it, h(a,b) and h(b,a) are read after DIFFERENT current
    tokens and the difference is dominated by token identity, not by order."""
    fwd = torch.tensor([[p[0], p[1], PROBE] for p in pairs], dtype=torch.long)
    rev = torch.tensor([[p[1], p[0], PROBE] for p in pairs], dtype=torch.long)
    return fwd, rev

@torch.no_grad()
def holonomy(model, pairs):
    if len(pairs) == 0:
        return {k: float("nan") for k in
                ("D_l2", "D_rel", "D_cos", "n", "hnorm")}
    f, r = _probe_batch(pairs)
    of = model(f, need_hidden=True); orv = model(r, need_hidden=True)
    res = {}
    for tag, key in (("h1", "h1"), ("h2", "h2")):
        hf = of[key][:, -1, :]                     # PROBE position
        hr = orv[key][:, -1, :]
        d = (hf - hr).norm(dim=-1)                 # paired, same pair both ways
        hn = 0.5 * (hf.norm(dim=-1) + hr.norm(dim=-1))
        cos = 1 - F.cosine_similarity(hf, hr, dim=-1)
        res[tag] = {"D_l2": float(d.mean()),
                    "D_rel": float((d / hn.clamp_min(1e-8)).mean()),
                    "D_cos": float(cos.mean()),
                    "hnorm": float(hn.mean()), "n": len(pairs)}
    return res

def _pairstats(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))

@torch.no_grad()
def holonomy_vec(model, pairs, key="h2"):
    """Per-pair cosine distance between the two orderings. Returns the VECTOR,
    so classes can be compared with a real error bar instead of two bare means."""
    if not pairs:
        return np.array([])
    f, r = _probe_batch(pairs)
    hf = model(f, need_hidden=True)[key][:, -1, :]
    hr = model(r, need_hidden=True)[key][:, -1, :]
    return (1 - F.cosine_similarity(hf, hr, dim=-1)).numpy()

def separation_report(model):
    """D_ord vs D_pos with a two-sample error bar, per block.

    C_ord and C_unord are matched (under --reverse-valid) on unigram
    frequency, pair frequency in both directions, and grammaticality of the
    reverse. They differ only in whether the two orders share a parent. So a
    gap here is order-sensitivity rather than a validity detector."""
    print("\n  SEPARATION  D_ord - D_pos (cosine), with 2-sample SEM")
    out = {}
    for key in ("h1", "h2"):
        o = holonomy_vec(model, META["C_ord"],   key)
        p_ = holonomy_vec(model, META["C_unord"], key)
        q = holonomy_vec(model, META["C_perm"],  key)
        mo, so = _pairstats(o); mp, sp = _pairstats(p_); mq, sq = _pairstats(q)
        gap = mo - mp
        sem = math.sqrt(so**2 + sp**2)
        out[key] = (gap, sem)
        print(f"    {key}: D_ord={mo:.4f}+/-{so:.4f}  D_pos={mp:.4f}+/-{sp:.4f}"
              f"  D_perm={mq:.4f}+/-{sq:.4f}")
        print(f"        gap={gap:+.4f} +/- {sem:.4f}  "
              f"({gap/sem if sem > 0 else float('nan'):.1f} SEM)")
    return out

# ------------------------------------------------- is the composition a group?
@torch.no_grad()
def isometry_report(model, key="h2"):
    """A group action by rotations preserves norms. The composition here is
    built by a GELU MLP, which has no reason to.

      norm_ratio   ||h(a,b)|| / ||h(b,a)||  -- exactly 1 for an orthogonal map
      norm_cv      spread of ||h|| across pairs -- near 0 if norms are fixed
    """
    pairs = META["C_ord"]
    f, r = _probe_batch(pairs)
    hf = model(f, need_hidden=True)[key][:, -1, :].norm(dim=-1)
    hr = model(r, need_hidden=True)[key][:, -1, :].norm(dim=-1)
    ratio = (hf / hr.clamp_min(1e-8)).numpy()
    alln = torch.cat([hf, hr]).numpy()
    m, sem = _pairstats(ratio)
    print(f"\n  ISOMETRY ({key})")
    print(f"    ||h(a,b)||/||h(b,a)|| = {m:.4f} +/- {sem:.4f}  "
          f"(1.0000 iff orthogonal)   |dev| mean={np.abs(ratio-1).mean():.4f}")
    print(f"    CV of ||h|| across pairs = {alln.std()/alln.mean():.4f}")

@torch.no_grad()
def parent_invariance(model, key="h2"):
    """The homomorphism test, using the grammar's own answer key.

    If the model reduces a leaf pair to its PARENT symbol, then two DIFFERENT
    productions of the SAME parent must land in the same place, and productions
    of different parents must not. This is the RHM invariance claim stated as a
    measurement: within-parent distance << between-parent distance.

    Unlike D_ord > D_pos, this can distinguish 'order is encoded' from
    'composition maps to a parent representation'."""
    rules = {int(k): [tuple(t) for t in v]
             for k, v in META["rules_level1"].items()}
    sym = {int(k): v for k, v in META["is_sym_level1"].items()}
    reps, owner = [], []
    for i, prods in rules.items():
        uniq = list(dict.fromkeys(prods))
        if len(uniq) < 2:
            continue
        for (x, y) in uniq:
            reps.append([x, y, PROBE]); owner.append(i)
    if len(reps) < 4:
        print("\n  PARENT INVARIANCE: too few productions"); return
    H = model(torch.tensor(reps, dtype=torch.long), need_hidden=True)[key][:, -1, :]
    H = F.normalize(H, dim=-1)
    Dm = 1 - (H @ H.T)
    own = np.array(owner)
    # For a SYMMETRIC parent, two of its productions may be (x,y) and (y,x) --
    # the same unordered set flipped. That distance is D_pos, not a comparison
    # of two DISTINCT productions, so it is reported separately rather than
    # folded into the within-parent mean.
    within, between, flips = [], [], []
    n = len(own)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(Dm[i, j])
            if own[i] != own[j]:
                between.append(d)
            elif sorted(reps[i][:2]) == sorted(reps[j][:2]):
                flips.append(d)                 # same set, opposite order
            else:
                within.append(d)                # genuinely distinct productions
    if len(within) < 2 or len(between) < 2:
        print("\n  PARENT INVARIANCE: not enough distinct productions"); return
    mw, sw = _pairstats(within); mb, sb = _pairstats(between)
    if flips:
        mf, sf = _pairstats(flips)
    print(f"\n  PARENT INVARIANCE ({key})   n_within={len(within)} "
          f"n_between={len(between)}")
    print(f"    within-parent  cos-dist = {mw:.4f} +/- {sw:.4f}")
    print(f"    between-parent cos-dist = {mb:.4f} +/- {sb:.4f}")
    print(f"    ratio within/between    = {mw/max(mb,1e-9):.4f}   "
          f"(<<1 means productions collapse onto their parent)")
    print(f"    gap={mb-mw:+.4f} +/- {math.sqrt(sw**2+sb**2):.4f}")
    if flips:
        print(f"    (order-flips of one set, same parent: {mf:.4f} +/- {sf:.4f}"
              f"  n={len(flips)} -- this is D_pos, kept out of the mean above)")

@torch.no_grad()
def reversibility_check(model, n=8):
    """Numerical proof the block map is a bijection: push a batch forward
    through every block, invert back, report max |x - inv(fwd(x))|.
    Anything above ~1e-4 means the inverse is not the inverse."""
    if not getattr(model, "rev", False):
        print("\n  REVERSIBILITY: model is not --rev; nothing to check"); return
    x, _ = get_batch("val", n)
    h = model.te(x) + model.pe(torch.arange(x.shape[1]))[None]
    worst = 0.0
    for i, blk in enumerate(model.blocks):
        y = blk(h)[0]
        back = blk.inverse(y)
        err = float((back - h).abs().max()); worst = max(worst, err)
        print(f"    block {i}: max|x - inv(fwd(x))| = {err:.3e}")
        h = y
    print(f"  REVERSIBILITY: worst {worst:.3e}  "
          f"{'OK' if worst < 1e-4 else 'FAILED -- inverse is wrong'}")

@torch.no_grad()
def attribution_by_inversion(model, pairs, key="h2"):
    """What the inverse actually buys for misattribution.

    Take the two orderings of a pair, invert the LAST block from each output,
    and measure how much of the h2 difference is already present in the
    reconstructed input to that block. That localises where the order
    distinction was introduced -- block 0 or block 1 -- per pair, rather than
    inferring it from h1 vs h2 means.

    Caveat: this attributes across BLOCKS. It cannot attribute across source
    tokens, because the softmax mixture inside a block remains many-to-one and
    no amount of block-level invertibility changes that."""
    if not getattr(model, "rev", False):
        print("\n  ATTRIBUTION: needs --rev"); return
    f, r = _probe_batch(pairs)
    of = model(f, need_hidden=True); orv = model(r, need_hidden=True)
    d_out = (of["h2"][:, -1] - orv["h2"][:, -1]).norm(dim=-1)
    xf = model.blocks[-1].inverse(of["h2"])[:, -1]
    xr = model.blocks[-1].inverse(orv["h2"])[:, -1]
    d_in = (xf - xr).norm(dim=-1)
    frac = (d_in / d_out.clamp_min(1e-8)).numpy()
    print(f"\n  ATTRIBUTION BY INVERSION (n={len(pairs)})")
    print(f"    ||dh|| at block-{len(model.blocks)-1} output = {float(d_out.mean()):.4f}")
    print(f"    ||dh|| at its reconstructed input  = {float(d_in.mean()):.4f}")
    print(f"    fraction already present on entry  = {frac.mean():.4f} "
          f"+/- {frac.std(ddof=1)/math.sqrt(len(frac)):.4f}")
    print(f"    (near 1 = the last block only passes the distinction through;")
    print(f"     near 0 = the last block creates it)")

def transitivity_report(model):
    """THE POSET TEST.

    C_trans  a prec b prec c, with (a,b) and (b,c) both level-1 rules, but
             (a,c) appearing as NO rule. Transitivity of the order says
             a prec c regardless.
    C_incomp incomparable AND not a rule. Same never-seen-as-a-rule status,
             opposite order status, matched on unigram and pair frequency.

    If the model learned the RELATION, unseen comparable pairs should carry
    order (high D) and unseen incomparable pairs should not (low D). If it
    only memorised the rule table, both are equally unseen and D should be
    the same. That is the difference between a partial order and a bag of
    oriented pairs, and it is the one claim the earlier grammar could not
    make, because rules there were drawn independently and implied nothing
    about any third pair."""
    if not META.get("poset") or not META.get("C_trans"):
        print("\n  TRANSITIVITY: corpus is not --poset; skipping"); return
    print("\n  TRANSITIVITY (the poset test)")
    for key in ("h1", "h2"):
        t = holonomy_vec(model, META["C_trans"],  key)
        i = holonomy_vec(model, META["C_incomp"], key)
        o = holonomy_vec(model, META["C_ord"],    key)
        u = holonomy_vec(model, META["C_unord"],  key)
        mt, st = _pairstats(t); mi, si = _pairstats(i)
        mo, _ = _pairstats(o);  mu, _ = _pairstats(u)
        gap, sem = mt - mi, math.sqrt(st**2 + si**2)
        print(f"    {key}:  D_trans={mt:.4f}+/-{st:.4f} (n={len(t)})   "
              f"D_incomp={mi:.4f}+/-{si:.4f} (n={len(i)})")
        print(f"         seen-rule anchors: D_ord={mo:.4f}  D_unord={mu:.4f}")
        print(f"         gap={gap:+.4f} +/- {sem:.4f} "
              f"({gap/sem if sem>0 else float('nan'):.1f} SEM)  "
              f"-> {'ORDER GENERALISES' if gap > 2*sem else 'no generalisation'}")

@torch.no_grad()
def additivity_report(model, key="h2"):
    """IS THE COMPOSITION ADDITIVE?

    Transitivity is established: unseen comparable pairs read as ordered. That
    says the composite arrow EXISTS. It does not say how it is built.
    L_transitivity would assert  d(a,c) = d(a,b) + d(b,c)  -- additive triangle
    closure on the residual stream. That is a separate claim and this measures
    it before any loss presumes it.

      d(x,y) = h(x,y,PROBE) - mean over all probe pairs   (displacement from a
               neutral baseline, so the shared PROBE/position component cancels)

    Reported against two controls:
      random   d(a,c) vs d(a,b') + d(b',c) for a RANDOM b' not on the chain
      shuffled d(a,c) vs the sum from a DIFFERENT chain entirely
    Additivity only means something if the real chain beats both."""
    ch = META.get("C_chain") or []
    if len(ch) < 8:
        print("\n  ADDITIVITY: no chains exported; rebuild with build_corpus_rhm4.py")
        return
    def rep(pairs):
        t = torch.tensor([[x, y, PROBE] for x, y in pairs], dtype=torch.long)
        return model(t, need_hidden=True)[key][:, -1, :]
    ab = [(a, b) for a, b, c in ch]; bc = [(b, c) for a, b, c in ch]
    ac = [(a, c) for a, b, c in ch]
    allp = ab + bc + ac
    base = rep(allp).mean(0, keepdim=True)
    Dab, Dbc, Dac = rep(ab) - base, rep(bc) - base, rep(ac) - base
    g = torch.Generator().manual_seed(0)
    rb = torch.randint(0, NLEAF, (len(ch),), generator=g).tolist()
    Dab_r = rep([(a, b2) for (a, _, _), b2 in zip(ch, rb)]) - base
    Dbc_r = rep([(b2, c) for (_, _, c), b2 in zip(ch, rb)]) - base
    perm = torch.randperm(len(ch), generator=g)

    def score(S, name):
        res = (Dac - S).norm(dim=-1) / Dac.norm(dim=-1).clamp_min(1e-8)
        cos = F.cosine_similarity(Dac, S, dim=-1)
        m1, s1 = _pairstats(res.numpy()); m2, s2 = _pairstats(cos.numpy())
        print(f"    {name:<22} rel.residual={m1:.4f}+/-{s1:.4f}   "
              f"cos(d_ac, sum)={m2:+.4f}+/-{s2:.4f}")
        return m2, s2

    print(f"\n  ADDITIVITY ({key})   n={len(ch)} chains a<b<c, (a,c) never a rule")
    mr, sr = score(Dab + Dbc,       "real chain a-b-c")
    mv, sv = score(Dab_r + Dbc_r,   "random midpoint b'")
    mp, sp = score(Dab[perm] + Dbc[perm], "shuffled chains")
    print(f"    real vs random-midpoint gap = {mr-mv:+.4f} +/- "
          f"{math.sqrt(sr**2+sv**2):.4f}")
    print(f"    real vs shuffled gap        = {mr-mp:+.4f} +/- "
          f"{math.sqrt(sr**2+sp**2):.4f}")
    print(f"    cos near +1 and residual near 0 => additive triangle closure;")
    print(f"    real indistinguishable from controls => composition is NOT additive")

def _hop_classes(max_hops=4, cap=64):
    """k-hop chains x0 < x1 < ... < xk where EVERY consecutive (xi,xi+1) is a
    level-1 production, but the endpoint pair (x0,xk) is NOT a production.
    Transitive closure says x0 < xk at any k. Path length is the variable."""
    P = np.array(META["prec"], dtype=bool)
    prod = set()
    for v in META["rules_level1"].values():
        for x, y in v:
            prod.add((int(x), int(y)))
    edges = [(x, y) for (x, y) in prod if P[x, y]]        # order-respecting rules
    adj = {}
    for x, y in edges:
        adj.setdefault(x, []).append(y)
    out = {}
    frontier = [[x, y] for (x, y) in edges]
    for k in range(2, max_hops + 1):
        nxt, ends = [], []
        for ch in frontier:
            for z in adj.get(ch[-1], []):
                if z in ch:
                    continue
                nc = ch + [z]
                nxt.append(nc)
                a, e = nc[0], nc[-1]
                if (a, e) not in prod and (e, a) not in prod and P[a, e]:
                    ends.append([a, e])
        frontier = nxt
        uniq, seen = [], set()
        for a, e in ends:
            if (a, e) in seen:
                continue
            seen.add((a, e)); uniq.append([a, e])
        rng_ = np.random.RandomState(0); rng_.shuffle(uniq)
        out[k] = uniq[:cap]
        if not frontier:
            break
    return out

@torch.no_grad()
def hop_decay_report(model):
    """Does the inference survive path length?

    Bounded local propagation would decay with k. A represented transitive
    CLOSURE would not. D_incomp (incomparable, also never a rule) is the floor
    in both cases."""
    if not META.get("prec"):
        print("\n  HOP DECAY: corpus is not --poset; skipping"); return
    cls = _hop_classes()
    print("\n  HOP DECAY  (x0<...<xk, every step a rule, (x0,xk) never a rule)")
    for key in ("h1", "h2"):
        inc = holonomy_vec(model, META["C_incomp"], key)
        mi, si = _pairstats(inc)
        print(f"    {key}:  floor D_incomp={mi:.4f}+/-{si:.4f} (n={len(inc)})")
        for k in sorted(cls):
            pr = cls[k]
            if len(pr) < 8:
                print(f"       {k}-hop: n={len(pr)} too few"); continue
            v = holonomy_vec(model, pr, key)
            m, sd = _pairstats(v)
            gap = m - mi; sem = math.sqrt(sd**2 + si**2)
            print(f"       {k}-hop: D={m:.4f}+/-{sd:.4f} (n={len(pr)})  "
                  f"gap={gap:+.4f}+/-{sem:.4f} ({gap/sem:.1f} SEM)")

def frequency_table():
    """Are the hop classes matched on exposure?

    D tracks how much the model has seen of a pair. A pair reachable by a long
    chain of rule edges sits deep in the DAG with many edges attached, so it
    may simply be more frequent. If so, rising D with hop count is exposure,
    not path length."""
    import collections
    tr = json.load(open(f"{a.data}/train_ids.json"))
    uni = collections.Counter(tr)
    bg = collections.Counter(zip(tr, tr[1:]))
    cls = dict(_hop_classes())
    cls["C_incomp"] = META["C_incomp"]; cls["C_ord"] = META["C_ord"]
    print("\n  ENDPOINT FREQUENCY (exposure check)")
    print(f"    {'class':<10}{'n':>5}{'unigram':>12}{'pair_fwd':>10}{'pair_rev':>10}")
    for k in sorted(cls, key=lambda z: str(z)):
        pr = cls[k]
        if not pr: continue
        u = np.mean([(uni[x] + uni[y]) / 2 for x, y in pr])
        f = np.mean([bg[(x, y)] for x, y in pr])
        r = np.mean([bg[(y, x)] for x, y in pr])
        print(f"    {str(k)+('-hop' if isinstance(k,int) else ''):<10}"
              f"{len(pr):>5}{u:>12.1f}{f:>10.1f}{r:>10.1f}")

@torch.no_grad()
def _feat(E, pairs):
    idx = torch.tensor(pairs, dtype=torch.long)
    return torch.cat([E[idx[:, 0]], E[idx[:, 1]]], -1)

def _fit_probe(E, pos, neg, iters=3000):
    X = torch.cat([_feat(E, pos), _feat(E, neg)])
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    w = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(w.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(iters):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(w(X).squeeze(-1), y).backward(); opt.step()
    return w

def product_probe(model):
    """PRODUCT-ORDER BOUNDARY TEST.

    x < y iff u_x <_1 u_y AND v_x <_2 v_y. Transitive (inherited componentwise),
    but a conjunction over two independent factors -- not decidable by a linear
    function of [e_x,e_y] unless the embedding has linearised it.

    Three probes, all trained on seen rules vs the frequency-matched
    incomparable negatives, all evaluated on held-out C_trans:
      trained embeddings   the quantity of interest
      RANDOM embeddings    the floor: a linear probe picks up something from
                           any structured relation, so 0.5 is NOT the baseline
      factor-1 / factor-2  controls: neither factor alone defines the relation
    """
    if not META.get("product"):
        print("\n  PRODUCT PROBE: corpus is not --product; skipping"); return
    P = np.array(META["prec"], dtype=bool)
    prod = set()
    for v in META["rules_level1"].values():
        for x, y in v: prod.add((int(x), int(y)))
    pos = [[x, y] for (x, y) in prod if P[x, y]]
    neg = [list(q) for q in META["C_incomp"]]
    neg += [[y, x] for x, y in META["C_incomp"]]
    held = [list(q) for q in META["C_trans"]]
    if len(pos) < 20 or len(held) < 8:
        print("\n  PRODUCT PROBE: classes too small"); return
    Etr = model.te.weight.detach()
    Ern = torch.randn_like(Etr) / math.sqrt(Etr.shape[1])
    print(f"\n  PRODUCT PROBE  (trained on {len(pos)} seen rules vs {len(neg)} "
          f"matched incomparable; held out {len(held)} C_trans)")
    for tag, E in (("trained te.weight", Etr), ("RANDOM embeddings (floor)", Ern)):
        w = _fit_probe(E, pos, neg)
        with torch.no_grad():
            ins = float(((w(_feat(E, pos + neg)).squeeze(-1) > 0).float() ==
                         torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
                         ).float().mean())
            fwd = float((w(_feat(E, held)).squeeze(-1) > 0).float().mean())
            rev = float((w(_feat(E, [[j, i] for i, j in held])).squeeze(-1) <= 0
                         ).float().mean())
        print(f"    {tag:<28} in-sample={ins:.3f}  "
              f"C_trans predicted COMPARABLE={fwd:.3f}  reversed NOT={rev:.3f}")
    for fi, kk in ((1, "prec1"), (2, "prec2")):
        if kk not in META: continue
        Pf = np.array(META[kk], dtype=bool)
        fp = [[x, y] for (x, y) in prod if Pf[x, y]][:len(pos)]
        if len(fp) < 20: continue
        w = _fit_probe(Etr, fp, neg)
        with torch.no_grad():
            fwd = float((w(_feat(Etr, held)).squeeze(-1) > 0).float().mean())
        print(f"    factor-{fi} only (control)       "
              f"C_trans predicted COMPARABLE={fwd:.3f}")
    print("    trained >> floor  => the embedding linearised the conjunction")
    print("    trained ~ floor   => it did not; any C_trans generalisation the")
    print("                         full model shows is genuine composition")

def embedding_probe(model):
    """Is comparability decodable from te.weight ALONE -- no context, no
    attention, no MLP? If yes, the transitive closure has been absorbed into
    the embedding matrix, and 'the model infers a<c' is really 'the embeddings
    already encode the closure'. That localises the result and explains both
    the null midpoint control and the absence of hop decay."""
    if not META.get("prec"): print("\n  EMBEDDING PROBE: needs --poset"); return
    P = np.array(META["prec"], dtype=bool)
    prod = set()
    for v in META["rules_level1"].values():
        for x, y in v: prod.add((int(x), int(y)))
    pos = [[x, y] for (x, y) in prod if P[x, y]]              # seen ordered rules
    neg = [list(p) for p in META["C_incomp"]]
    neg += [[y, x] for x, y in META["C_incomp"]]              # incomparable, both ways
    E = model.te.weight.detach()
    X = torch.cat([_feat(E, pos), _feat(E, neg)])
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    w = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(w.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(2000):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(w(X).squeeze(-1), y).backward(); opt.step()
    def acc(pairs, want):
        if not pairs: return float("nan"), 0
        with torch.no_grad():
            p = (w(_feat(E, pairs)).squeeze(-1) > 0).float()
        return float((p == want).float().mean()), len(pairs)
    tr_acc, _ = acc(pos + neg, y)
    print("\n  EMBEDDING PROBE (linear, te.weight only, no context)")
    print(f"    trained on {len(pos)} seen ordered rules vs {len(neg)} "
          f"incomparable;  in-sample acc={tr_acc:.3f}")
    cls = _hop_classes()
    for k in sorted(cls):
        pr = cls[k]
        if len(pr) < 8: continue
        a_f, n = acc(pr, torch.ones(len(pr)))
        a_r, _ = acc([[j, i] for i, j in pr], torch.zeros(len(pr)))
        print(f"    {k}-hop held out: predicted COMPARABLE {a_f:.3f} (n={n})   "
              f"reversed predicted NOT {a_r:.3f}")
    print(f"    chance = 0.500. High held-out accuracy => the closure lives in")
    print(f"    the embedding matrix, not in the composition dynamics.")

@torch.no_grad()
def synonym_report(model):
    """THE QUOTIENT AXIOM:  a < c  and  c ~ d  =>  a < d.

    C_syn    (a,d) where (a,c) is a rule, d~c, and (a,d) was NEVER a rule.
    C_synsrc the (a,c) each fact came from -- the positive anchor.
    C_incomp incomparable and never a rule, matched on exposure -- the floor.

    D_syn near D_synsrc and far above D_incomp means the order transferred
    across the equivalence: the arrow is well-defined on the quotient, which
    is what makes the quotient CATEGORY well-defined rather than a partition
    that happens to exist.

    Two ways this could be trivially true, both reported:
      embedding identity   if e_c and e_d are near-identical the transfer is
                           free. Synonym cosine is compared against a
                           frequency-matched non-synonym baseline.
      shared context       synonyms placed under the same parents would be
                           identified by distribution alone; the builder keeps
                           their parent sets nearly disjoint (Jaccard ~0.16)."""
    if not META.get("C_syn"):
        print("\n  SYNONYM: corpus has no synonyms; skipping"); return
    syn = {int(k): int(v) for k, v in META.get("synonyms", {}).items()}
    E = model.te.weight.detach()
    En = F.normalize(E, dim=-1)
    sp = [(t, syn[t]) for t in sorted(syn) if t < syn[t]]
    s_cos = torch.tensor([float(En[i] @ En[j]) for i, j in sp])
    g = torch.Generator().manual_seed(0)
    rp = torch.randint(0, E.shape[0], (2, 400), generator=g)
    keep = rp[0] != rp[1]
    r_cos = (En[rp[0][keep]] * En[rp[1][keep]]).sum(-1)
    ms, ss = _pairstats(s_cos.numpy()); mr, sr = _pairstats(r_cos.numpy())
    print("\n  SYNONYM COMPATIBILITY (the quotient axiom)")
    print(f"    embedding check: cos(e_c,e_d) synonyms={ms:+.4f}+/-{ss:.4f} "
          f"(n={len(sp)})   random pairs={mr:+.4f}+/-{sr:.4f}")
    print(f"      (if synonyms were near +1 the transfer would be free; they")
    print(f"       have ~85% disjoint parent contexts by construction)")
    for key in ("h1", "h2"):
        a_ = holonomy_vec(model, META["C_syn"],    key)
        b_ = holonomy_vec(model, META["C_synsrc"], key)
        c_ = holonomy_vec(model, META["C_incomp"], key)
        ma, sa = _pairstats(a_); mb, sb = _pairstats(b_); mc, sc = _pairstats(c_)
        gap, sem = ma - mc, math.sqrt(sa**2 + sc**2)
        short = mb - ma
        print(f"    {key}:  D_syn={ma:.4f}+/-{sa:.4f} (n={len(a_)})   "
              f"D_synsrc={mb:.4f}+/-{sb:.4f}   D_incomp={mc:.4f}+/-{sc:.4f}")
        print(f"         vs floor  = {gap:+.4f} +/- {sem:.4f} "
              f"({gap/sem if sem>0 else float('nan'):.1f} SEM)  "
              f"-> {'ORDER TRANSFERS' if gap > 2*sem else 'no transfer'}")
        print(f"         shortfall vs anchor = {short:+.4f} +/- "
              f"{math.sqrt(sa**2+sb**2):.4f}  (0 = full transfer)")

def _probe_dirs(E, pos, neg, n_boot=100, frac=0.7, seed=0):
    """Bootstrap the order probe. ONE logistic fit gives a single weight
    vector, so a projector built from it is RANK 1 -- and any two vectors
    projected onto a 1-D subspace are collinear, cosine exactly +-1, for
    synonyms and random pairs alike. That test cannot fail, so it proves
    nothing. Refitting on resamples gives a genuine spread of directions
    whose rank we choose and report."""
    g = np.random.RandomState(seed)
    W = []
    for _ in range(n_boot):
        ip = g.choice(len(pos), int(frac*len(pos)), replace=False)
        ineg = g.choice(len(neg), int(frac*len(neg)), replace=False)
        w = _fit_probe(E, [pos[i] for i in ip], [neg[i] for i in ineg], iters=1200)
        W.append(w.weight.detach()[0].clone())
    return torch.stack(W)                      # n_boot x 2d

def subspace_report(model):
    """Is the equivalence localised to an ORDER SUBSPACE?

    Full-space synonym cosine is ~0. If the order lives in a low-rank subspace
    that synonyms share while differing elsewhere, then inside that subspace
    they should align and RANDOM pairs should not. Both are measured in the
    SAME subspace at each rank; a claim only counts where the two separate.

    The probe acts on [e_x, e_y], so a token has TWO order coordinates -- as
    the lesser element (left slot) and as the greater (right slot). They are
    reported separately: if they disagree, 'the order subspace' is two
    subspaces and the factorisation claim has to be restated."""
    syn = {int(k): int(v) for k, v in META.get("synonyms", {}).items()}
    if not syn: print("\n  SUBSPACE: no synonyms; skipping"); return
    P = np.array(META["prec"], dtype=bool)
    prod = set()
    for v in META["rules_level1"].values():
        for x, y in v: prod.add((int(x), int(y)))
    pos = [[x, y] for (x, y) in prod if P[x, y]]
    neg = [list(q) for q in META["C_incomp"]] + \
          [[y, x] for x, y in META["C_incomp"]]
    E = model.te.weight.detach().clone(); d = E.shape[1]
    W = _probe_dirs(E, pos, neg)
    sp = [(t, syn[t]) for t in sorted(syn) if t < syn[t]]
    # random control matched on unigram frequency to the synonym pairs
    import collections
    cnt = collections.Counter(json.load(open(f"{a.data}/train_ids.json")))
    gg = np.random.RandomState(1)
    rnd = []
    for i, j in sp:
        tgt = (cnt[i] + cnt[j]) / 2
        cand = [(abs((cnt[x]+cnt[y])/2 - tgt), (x, y))
                for x, y in zip(gg.randint(0, E.shape[0], 300),
                                gg.randint(0, E.shape[0], 300))
                if x != y and syn.get(x) != y]
        rnd.append(min(cand)[1])

    torch.set_grad_enabled(False)          # probes are fitted; measuring now
    print("\n  ORDER SUBSPACE  (bootstrapped rank-k, synonyms vs matched random)")
    for slot, sl in (("left (as lesser)", slice(0, d)),
                     ("right (as greater)", slice(d, 2*d))):
        Ws = W[:, sl]
        U, S, _ = torch.linalg.svd(Ws - Ws.mean(0, keepdim=True), full_matrices=False)
        V = torch.linalg.svd(Ws, full_matrices=False)[2]        # k x d basis
        print(f"    {slot}:  bootstrap spectrum "
              + " ".join(f"{float(x):.2f}" for x in S[:6]))
        for k in (1, 2, 4, 8, 16, 32, 64):
            if k > V.shape[0]: break
            B = V[:k]                                            # k x d
            def cos_in(pairs):
                A = torch.stack([B @ E[i] for i, j in pairs])
                C = torch.stack([B @ E[j] for i, j in pairs])
                return F.cosine_similarity(A, C, dim=-1).numpy()
            ms, ss = _pairstats(cos_in(sp)); mr, sr = _pairstats(cos_in(rnd))
            gap = ms - mr; sem = math.sqrt(ss**2 + sr**2)
            flag = "rank-1: collinear BY CONSTRUCTION, uninformative" if k == 1 else \
                   ("full space (k=d): equals the cosine already reported"
                    if k >= d else
                    ("SEPARATES" if gap > 2*sem else "no separation"))
            print(f"      k={k:<3} synonyms={ms:+.4f}+/-{ss:.4f}  "
                  f"random={mr:+.4f}+/-{sr:.4f}  gap={gap:+.4f}+/-{sem:.4f}  {flag}")
    # scalar order coordinate: sign/magnitude agreement along the mean direction
    print("    scalar order coordinate (mean bootstrap direction):")
    for slot, sl in (("left", slice(0, d)), ("right", slice(d, 2*d))):
        w = W[:, sl].mean(0); w = w / w.norm()
        z = (E @ w)
        agree_s = float(np.mean([np.sign(float(z[i])) == np.sign(float(z[j]))
                                 for i, j in sp]))
        agree_r = float(np.mean([np.sign(float(z[i])) == np.sign(float(z[j]))
                                 for i, j in rnd]))
        d_s = float(np.mean([abs(float(z[i]-z[j])) for i, j in sp]))
        d_r = float(np.mean([abs(float(z[i]-z[j])) for i, j in rnd]))
        def _corr(prs):
            u = np.array([float(z[i]) for i, j in prs])
            v = np.array([float(z[j]) for i, j in prs])
            if u.std() < 1e-9 or v.std() < 1e-9: return float("nan")
            return float(np.corrcoef(u, v)[0, 1])
        # orientation of a pair is arbitrary (which of c,d is listed first),
        # so report the correlation symmetrised over both orderings
        r_s = _corr(sp + [(j, i) for i, j in sp])
        r_r = _corr(rnd + [(j, i) for i, j in rnd])
        print(f"      {slot:<6} sign agreement syn={agree_s:.3f} rnd={agree_r:.3f}   "
              f"mean |z_c - z_d| syn={d_s:.4f} rnd={d_r:.4f}")
        print(f"      {'':<6} corr(z_c, z_d)  syn={r_s:+.3f} rnd={r_r:+.3f}  "
              f"(n={len(sp)} pairs, symmetrised over pair orientation)")
    _end_subspace()

def scalar_holdout_report(model):
    """Is the order coordinate a GENERAL property of the embedding, or did the
    probe partly read the synonym tokens themselves?

    The probe's positives are seen production rules, and synonym tokens appear
    in those rules -- so the direction is not independent of the tokens being
    tested. Here every rule containing ANY synonym token is removed from the
    fit, and the negatives are restricted the same way. The direction is then
    learned from tokens the test never touches.

      held-in   corr stays ~0.9  -> the order coordinate is general and
                synonyms inherit it
      collapses -> the probe was reading those specific tokens and the
                earlier scalar result does not stand on its own
    """
    syn = {int(k): int(v) for k, v in META.get("synonyms", {}).items()}
    if not syn: print("\n  SCALAR HOLD-OUT: no synonyms; skipping"); return
    P = np.array(META["prec"], dtype=bool)
    prod = set()
    for v in META["rules_level1"].values():
        for x, y in v: prod.add((int(x), int(y)))
    S = set(syn)
    pos_all = [[x, y] for (x, y) in prod if P[x, y]]
    neg_all = [list(q) for q in META["C_incomp"]] + \
              [[y, x] for x, y in META["C_incomp"]]
    pos_ho = [q for q in pos_all if q[0] not in S and q[1] not in S]
    neg_ho = [q for q in neg_all if q[0] not in S and q[1] not in S]
    E = model.te.weight.detach().clone(); d = E.shape[1]
    import collections
    cnt = collections.Counter(json.load(open(f"{a.data}/train_ids.json")))
    sp = [(t, syn[t]) for t in sorted(syn) if t < syn[t]]
    gg = np.random.RandomState(1)
    rnd = []
    for i, j in sp:
        tgt = (cnt[i] + cnt[j]) / 2
        cand = [(abs((cnt[x]+cnt[y])/2 - tgt), (x, y))
                for x, y in zip(gg.randint(0, E.shape[0], 300),
                                gg.randint(0, E.shape[0], 300))
                if x != y and syn.get(x) != y]
        rnd.append(min(cand)[1])

    print("\n  SCALAR ORDER COORDINATE -- SYNONYM-HOLD-OUT CONTROL")
    print(f"    all rules:            {len(pos_all)} pos / {len(neg_all)} neg")
    print(f"    synonym-free rules:   {len(pos_ho)} pos / {len(neg_ho)} neg"
          f"   ({len(S)} of {E.shape[0]} tokens excluded)")
    if len(pos_ho) < 20 or len(neg_ho) < 20:
        print("    too few synonym-free pairs to refit; inconclusive"); return
    # TIE-BREAKER: is the synonym-free probe underfit, or is the direction
    # real and the synonym correlation genuinely weaker? A direction that still
    # classifies held-out HOP pairs is a working direction; one that fails at
    # everything says the control was underpowered and proves nothing.
    cls = _hop_classes()
    for tag, (pp, nn) in (("all rules (as before)", (pos_all, neg_all)),
                          ("synonym-free fit", (pos_ho, neg_ho))):
        wfit = _fit_probe(E, pp, nn, iters=3000)
        torch.set_grad_enabled(False)
        line = []
        for k in sorted(cls):
            pr = [q for q in cls[k] if q[0] not in S and q[1] not in S]
            if len(pr) < 8: continue
            f_ = float((wfit(_feat(E, pr)).squeeze(-1) > 0).float().mean())
            r_ = float((wfit(_feat(E, [[j, i] for i, j in pr])).squeeze(-1) <= 0
                        ).float().mean())
            line.append(f"{k}-hop {f_:.2f}/{r_:.2f} (n={len(pr)})")
        ins = float(((wfit(_feat(E, pp + nn)).squeeze(-1) > 0).float() ==
                     torch.cat([torch.ones(len(pp)), torch.zeros(len(nn))])
                     ).float().mean())
        print(f"    [{tag}] probe health: in-sample={ins:.3f}  "
              f"synonym-free hop pairs fwd/rev: " + "  ".join(line))
        torch.set_grad_enabled(True)
        W = _probe_dirs(E, pp, nn, n_boot=100)
        torch.set_grad_enabled(False)
        print(f"    {tag}:")
        for slot, sl in (("left", slice(0, d)), ("right", slice(d, 2*d))):
            w = W[:, sl].mean(0); w = w / w.norm()
            z = (E @ w)
            def _c(prs):
                u = np.array([float(z[i]) for i, j in prs] +
                             [float(z[j]) for i, j in prs])
                v = np.array([float(z[j]) for i, j in prs] +
                             [float(z[i]) for i, j in prs])
                return float(np.corrcoef(u, v)[0, 1])
            ag = lambda prs: float(np.mean([np.sign(float(z[i])) ==
                                            np.sign(float(z[j])) for i, j in prs]))
            dz = lambda prs: float(np.mean([abs(float(z[i]-z[j])) for i, j in prs]))
            print(f"      {slot:<6} corr syn={_c(sp):+.3f} rnd={_c(rnd):+.3f}   "
                  f"sign syn={ag(sp):.3f} rnd={ag(rnd):.3f}   "
                  f"|dz| syn={dz(sp):.4f} rnd={dz(rnd):.4f}")
        torch.set_grad_enabled(True)
    torch.set_grad_enabled(True)

def _end_subspace():
    torch.set_grad_enabled(True)

def holonomy_suite(model):
    return {name: holonomy(model, META[name])
            for name in ("C_ord", "C_unord", "C_perm")}

def print_suite(s, prefix="  "):
    for tag in ("h1", "h2"):
        print(f"{prefix}{tag}:")
        for name, lbl in (("C_ord", "D_ord "), ("C_unord", "D_pos "),
                          ("C_perm", "D_perm")):
            d = s[name][tag]
            print(f"{prefix}  {lbl} n={d['n']:<3} "
                  f"l2={d['D_l2']:.4f}  rel={d['D_rel']:.4f}  "
                  f"cos={d['D_cos']:.4f}  |h|={d['hnorm']:.3f}")

# ============================================================ PHASE 1: NULL
if a.null:
    print("=" * 66)
    print("PHASE 1 -- UNTRAINED NULL FLOOR")
    print("=" * 66)
    print(f"  corpus {a.data}  leaf vocab {NLEAF}  seq {SEQ}  "
          f"exact floor {META['H_tok_exact']:.4f} nats/token")
    print(f"  pairs: C_ord {len(META['C_ord'])}  C_unord {len(META['C_unord'])}"
          f"  C_perm {len(META['C_perm'])}")
    print()
    acc = {n: {t: {m: [] for m in ("D_l2", "D_rel", "D_cos")}
               for t in ("h1", "h2")}
           for n in ("C_ord", "C_unord", "C_perm")}
    for s in range(a.seeds):
        torch.manual_seed(1000 + s)
        model = LM2(a.d_model, a.n_heads, nl=a.layers, mlp=a.mlp, rev=a.rev)
        suite = holonomy_suite(model)
        for n in acc:
            for t in acc[n]:
                for m in acc[n][t]:
                    acc[n][t][m].append(suite[n][t][m])
        print(f"  seed {s}:")
        print_suite(suite, prefix="    ")
    print()
    print("  SUMMARY  mean +/- SEM over "
          f"{a.seeds} untrained seeds")
    for t in ("h1", "h2"):
        print(f"    {t}:")
        for n, lbl in (("C_ord", "D_ord "), ("C_unord", "D_pos "),
                       ("C_perm", "D_perm")):
            line = []
            for m in ("D_l2", "D_rel", "D_cos"):
                v = np.array(acc[n][t][m])
                sem = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0
                line.append(f"{m}={v.mean():.4f}+/-{sem:.4f}")
            print(f"      {lbl} " + "  ".join(line))
    print()
    print("  READ THIS AS A SANITY CHECK, NOT A BASELINE:")
    print("   - D_ord ~ D_pos ~ D_perm at init is the EXPECTED result. All")
    print("     three then differ only by position embeddings and random")
    print("     weights, which know nothing about the grammar.")
    print("   - If D_ord is already separated from D_pos here, the sequence")
    print("     construction leaks grammar into the metric and the experiment")
    print("     is invalid before it starts. Stop and fix that first.")
    print("   - The trained comparison needs the SEM of the PAIRED difference")
    print("     across TRAINED seeds. Do not reuse these numbers as the gate.")
    raise SystemExit(0)

# ============================================================== TRAINING
torch.manual_seed(1000 + a.seed)
model = LM2(a.d_model, a.n_heads, nl=a.layers, mlp=a.mlp, rev=a.rev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                        weight_decay=0.01)
nparam = sum(p.numel() for p in model.parameters())
if a.load:
    # Analysis-only path: the probes are forward passes, so a finished run can
    # be re-analysed with new instrumentation without retraining it.
    model.load_state_dict(torch.load(a.load, map_location="cpu"))
    model.eval()
    print(f"  loaded {a.load}; skipping training")
    r = eval_split(model)
    print(f"  all={r['all']:.4f}  local={r['local']:.4f}  higher={r['higher']:.4f}")
    print("\n  holonomy suite:")
    print_suite(holonomy_suite(model))
    reversibility_check(model)
    separation_report(model)
    isometry_report(model, "h2")
    parent_invariance(model, "h2")
    parent_invariance(model, "h1")
    synonym_report(model)
    subspace_report(model)
    scalar_holdout_report(model)
    transitivity_report(model)
    hop_decay_report(model)
    frequency_table()
    embedding_probe(model)
    product_probe(model)
    additivity_report(model, "h2")
    additivity_report(model, "h1")
    attribution_by_inversion(model, META["C_ord"])
    raise SystemExit(0)
print(f"  {a.layers}-layer {'attn+MLP' if a.mlp else 'attention-only'}  "
      f"d={a.d_model} heads={a.n_heads} params={nparam:,}  seed={a.seed}  "
      f"task={a.task}")
if a.task == "rhm":
    print(f"  exact floor {META['H_tok_exact']:.4f}   "
          f"held-out bigram {META['H_bigram_heldout']:.4f}")
    print(f"  step      all    local   higher |  best prev-token head")
else:
    print(f"  CALIBRATION RUN: induction should exceed ~0.8. If it does not,")
    print(f"  the scoring code is wrong, not the hypothesis.")
    print(f"  step     loss |  best prev / best induction head")
for step in range(1, a.steps + 1):
    x, y = get_batch()
    loss = model(x, y)["loss"]
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % a.eval_every == 0 or step == 1:
        sc = induction_scores(model, K=a.copy_k)
        bp = max(sc.items(), key=lambda kv: kv[1][0])
        bi = max(sc.items(), key=lambda kv: kv[1][1])
        if a.task == "rhm":
            # induction is NOT reported here: RHM sequences contain no
            # in-context repeats, so the circuit has nothing to exploit and a
            # near-zero score is the correct reading, not a finding.
            r = eval_split(model)
            print(f"  {step:5d}  {r['all']:.4f}  {r['local']:.4f}  "
                  f"{r['higher']:.4f} | prev {bp[0]}={bp[1][0]:.3f}")
        else:
            print(f"  {step:5d}  {float(loss):.4f} | prev {bp[0]}={bp[1][0]:.3f}"
                  f"  ind {bi[0]}={bi[1][1]:.3f}")

print("\n  induction scores, all heads (prev_token, induction):")
for k, (p, i) in induction_scores(model).items():
    print(f"    {k}: prev={p:.3f}  ind={i:.3f}")
if a.save:
    torch.save(model.state_dict(), a.save)
    print(f"\n  saved {a.save}")

if a.task == "rhm":
    print("\n  holonomy suite (trained):")
    print_suite(holonomy_suite(model))
    separation_report(model)
    isometry_report(model, "h2")
    parent_invariance(model, "h2")
    parent_invariance(model, "h1")
    synonym_report(model)
    subspace_report(model)
    scalar_holdout_report(model)
    transitivity_report(model)
    hop_decay_report(model)
    frequency_table()
    embedding_probe(model)
    product_probe(model)
    additivity_report(model, "h2")
    additivity_report(model, "h1")
    reversibility_check(model)
    attribution_by_inversion(model, META["C_ord"])
    print("  (only interpretable once `higher` actually descends -- a model")
    print("   that has not learned the hierarchy cannot be asked how it")
    print("   represents it)")
