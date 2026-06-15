#!/usr/bin/env python3
"""
Monodromy Training Experiment
==============================
Tests whether training the 2-layer monodromy F directly with LM loss
is faster than training the 24-layer model that implements F through
600x cancellation.

THEORY:
  Standard 24-layer training:
    gradient ∂L/∂W_K^(0) passes through 23 layers of 600x cancellation
    → gradient diluted ~600x → needs ~300K steps to converge
  
  2-layer monodromy training:
    gradient ∂L/∂W_K passes through only 2 layers, no cancellation
    → gradient clean → should need ~1/12 the steps
  
  The 600x cancellation is NOT just a forward-pass phenomenon.
  It contaminates the backward pass too.
  Each intermediate layer's Jacobian ≈ (I + small), so gradients
  pass through cleanly but the SIGNAL is diluted by 22 near-identity
  Jacobians stacked multiplicatively.

EXPERIMENT:
  Fix d=256 (manageable on CPU)
  Train A: 24 layers, d=256  (full cancellation scaffold)
  Train B: 2 layers,  d=256  (direct monodromy)
  Train C: 4 layers,  d=256  (intermediate)
  Train D: 8 layers,  d=256  (intermediate)
  
  All: same vocab, same corpus, same LR, same batch
  Measure: val_loss at each checkpoint
  Measure: steps to reach target loss
  Measure: gradient norm at W_K per layer (gradient dilution signal)

KEY METRIC: steps_to_target(A) / steps_to_target(B)
  If ~12: gradient dilution confirmed (2 layers = 12x cleaner gradient)
  If ~2:  mostly just fewer params, not cleaner gradient  
  If ~1:  24 layers needed for quality, monodromy can't match it
"""

import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────────────────
D        = 256    # model dimension (same for all)
N_HEADS  = 4
BATCH    = 8
SEQ      = 64
LR       = 3e-4
STEPS    = 2000
LOG      = 100
TARGET_LOSS = 4.0   # what val_loss we race to reach

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Data ──────────────────────────────────────────────────────────────────────
with open('/tmp/train_ids.json') as f: train_ids = json.load(f)
with open('/tmp/val_ids.json')   as f: val_ids   = json.load(f)
with open('/tmp/vocab.json')     as f: vocab     = json.load(f)
VOCAB = len(vocab)
train_t = torch.tensor(train_ids, dtype=torch.long)
val_t   = torch.tensor(val_ids,   dtype=torch.long)

def get_batch(split='train'):
    data = train_t if split == 'train' else val_t
    ix   = torch.randint(0, len(data)-SEQ-1, (BATCH,))
    x    = torch.stack([data[i:i+SEQ]   for i in ix]).to(device)
    y    = torch.stack([data[i+1:i+SEQ+1] for i in ix]).to(device)
    return x, y

# ── Architecture (identical per block, only depth varies) ─────────────────────
class Attn(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh = nh; self.dh = d//nh; self.sc = math.sqrt(d//nh)
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.ln  = nn.LayerNorm(d)
        nn.init.normal_(self.W_Q.weight, std=0.02)
        nn.init.normal_(self.W_K.weight, std=0.02)
        nn.init.normal_(self.W_V.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, h):
        B, S, D = h.shape; H = self.nh; dh = self.dh
        Q = self.W_Q(h).view(B,S,H,dh).transpose(1,2)
        K = self.W_K(h).view(B,S,H,dh).transpose(1,2)
        V = self.W_V(h).view(B,S,H,dh).transpose(1,2)
        sc = Q @ K.transpose(-2,-1) / self.sc
        mask = torch.triu(torch.ones(S,S,device=h.device),diagonal=1).bool()
        sc = sc.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        out = (F.softmax(sc,dim=-1) @ V).transpose(1,2).reshape(B,S,D)
        return self.ln(h + self.out(out))

class FF(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Linear(d, d*2, bias=False)
        self.v = nn.Linear(d, d*2, bias=False)
        self.o = nn.Linear(d*2, d, bias=False)
        self.n = nn.LayerNorm(d)
        nn.init.normal_(self.g.weight, std=0.02)
        nn.init.normal_(self.v.weight, std=0.02)
        nn.init.normal_(self.o.weight, std=0.02)

    def forward(self, h):
        return self.n(h + self.o(F.silu(self.g(h)) * self.v(h)))

class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.attn = Attn(d, nh)
        self.ff   = FF(d)
    def forward(self, h):
        return self.ff(self.attn(h))

class LM(nn.Module):
    def __init__(self, d, nh, n_layers):
        super().__init__()
        self.te     = nn.Embedding(VOCAB, d)
        self.pe     = nn.Embedding(512, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(n_layers)])
        self.ln_f   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, VOCAB, bias=False)
        self.head.weight = self.te.weight
        nn.init.normal_(self.te.weight, std=0.02)
        nn.init.normal_(self.pe.weight, std=0.02)
        self._n_layers = n_layers

    def forward(self, x, y=None):
        B, S = x.shape
        h = self.te(x) + self.pe(torch.arange(S, device=x.device))
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(self.ln_f(h))
        loss = F.cross_entropy(logits.view(-1,VOCAB), y.view(-1)) if y is not None else None
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def wk_grad_norms(self):
        """Return gradient norm at W_K for each layer."""
        norms = []
        for blk in self.blocks:
            g = blk.attn.W_K.weight.grad
            norms.append(float(g.norm()) if g is not None else 0.0)
        return norms

    def attn_output_norms(self, x):
        """Measure per-layer attention output norms (the cancellation signal)."""
        self.eval()
        norms = []; h_prev = None
        with torch.no_grad():
            B, S = x.shape
            h = self.te(x) + self.pe(torch.arange(S, device=x.device))
            h_prev = h.clone()
            for blk in self.blocks:
                h_after_attn = blk.attn(h_prev)
                # attention output = h_after_attn - h_prev (residual removed)
                attn_out = h_after_attn - h_prev
                norms.append(float(attn_out.norm()))
                h_prev = blk.ff(h_after_attn)
        return norms

# ── Training ──────────────────────────────────────────────────────────────────
def cosine_lr(step, total, base, warmup=100):
    if step < warmup: return base * step / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * t))

def eval_val(model, n=20):
    model.eval(); ls = []
    with torch.no_grad():
        for _ in range(n):
            x, y = get_batch('val'); _, l = model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def train(n_layers, name, seed=42):
    torch.manual_seed(seed)
    model = LM(D, N_HEADS, n_layers).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR,
                               betas=(0.9,0.95), weight_decay=0.1)

    steps_to_target = None
    steps_log  = []; val_log = []
    grad_log   = {l: [] for l in range(n_layers)}  # W_K grad norms per layer
    t0 = time.time()

    print(f"\n  {name}  ({n_layers} layers, {model.n_params():,} params)")

    for step in range(1, STEPS+1):
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, STEPS, LR)

        model.train()
        x, y = get_batch('train')
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()

        # Record W_K gradient norms BEFORE clipping
        if step % LOG == 0:
            gnorms = model.wk_grad_norms()
            for l, gn in enumerate(gnorms):
                grad_log[l].append(gn)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % LOG == 0 or step == 1:
            vl = eval_val(model)
            steps_log.append(step); val_log.append(vl)
            elapsed = time.time() - t0
            print(f"    {step:>4}/{STEPS}  val={vl:.4f}  t={elapsed:.0f}s", flush=True)
            if vl < TARGET_LOSS and steps_to_target is None:
                steps_to_target = step
                print(f"    *** TARGET {TARGET_LOSS} REACHED at step {step} ***")

    # Measure cancellation in final model
    x_test, _ = get_batch('train')
    attn_norms = model.attn_output_norms(x_test)
    total_attn = sum(attn_norms)
    with torch.no_grad():
        model.eval()
        h0  = (model.te(x_test) + model.pe(torch.arange(SEQ,device=x_test.device)))
        h_final = h0.clone()
        for blk in model.blocks: h_final = blk(h_final)
        net_change = float((h_final - h0).norm())
    cancellation = total_attn / max(net_change, 1e-8)

    return {
        'name':        name,
        'n_layers':    n_layers,
        'n_params':    model.n_params(),
        'steps':       steps_log,
        'val':         val_log,
        'final_val':   val_log[-1],
        'steps_to_target': steps_to_target,
        'grad_norms':  grad_log,
        'attn_norms':  attn_norms,
        'total_attn':  total_attn,
        'net_change':  net_change,
        'cancellation': cancellation,
        'time_s':      time.time() - t0,
    }

# ── Run all four conditions ───────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  MONODROMY TRAINING EXPERIMENT")
print(f"  d={D}  target_loss={TARGET_LOSS}  steps={STEPS}")
print(f"  Corpus: {len(train_ids):,} train tokens  |  vocab={VOCAB}")
print(f"{'='*70}")

results = {}
for n_layers, name in [(2,'2-layer (monodromy)'),
                        (4,'4-layer'),
                        (8,'8-layer'),
                        (24,'24-layer (full)')]:
    results[n_layers] = train(n_layers, name)

# ── Analysis ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  RESULTS")
print(f"{'='*70}\n")

print(f"  {'Model':25}  {'Params':>10}  {'Final val':>10}  "
      f"{'Steps→target':>14}  {'Cancellation':>14}  {'Time':>6}")
print("  " + "-"*85)

for n, r in results.items():
    stt = str(r['steps_to_target']) if r['steps_to_target'] else f">{STEPS}"
    print(f"  {r['name']:25}  {r['n_params']:>10,}  {r['final_val']:>10.4f}  "
          f"{stt:>14}  {r['cancellation']:>14.1f}x  {r['time_s']:>5.0f}s")

# Key ratio: steps_to_target for 24-layer vs 2-layer
r2  = results[2];  r24 = results[24]
if r2['steps_to_target'] and r24['steps_to_target']:
    ratio = r24['steps_to_target'] / r2['steps_to_target']
    print(f"\n  Steps ratio (24-layer / 2-layer): {ratio:.1f}x")
    print(f"  Theory predicts: ~12x (proportional to depth)")
    if ratio > 8:
        verdict = "GRADIENT DILUTION CONFIRMED: 24-layer training is wasteful"
    elif ratio > 3:
        verdict = "PARTIAL: some gradient dilution, but also capacity difference"
    else:
        verdict = "NO DILUTION: 24-layer trains proportionally to depth, 2-layer lacks capacity"
    print(f"  Verdict: {verdict}")

print(f"\n  GRADIENT NORMS AT W_K (mean across training):")
print(f"  {'Model':25}  {'L0 grad':>10}  {'L_mid grad':>12}  {'L_last grad':>12}")
print("  " + "-"*65)
for n, r in results.items():
    gnorms = r['grad_norms']
    l0   = float(np.mean(gnorms[0]))          if 0 in gnorms and gnorms[0] else 0
    lmid = float(np.mean(gnorms[n//2]))       if n//2 in gnorms and gnorms[n//2] else 0
    llast= float(np.mean(gnorms[n-1]))        if n-1 in gnorms and gnorms[n-1] else 0
    print(f"  {r['name']:25}  {l0:>10.4f}  {lmid:>12.4f}  {llast:>12.4f}")

print(f"""
  INTERPRETATION:
  
  Cancellation ratio = (sum of per-layer attn output norms) / (net change)
  This is the 600x signal from BALBc ghost signal analysis.
  
  If 2-layer cancellation << 24-layer cancellation:
    The 2-layer model does NOT build up and cancel large intermediate outputs.
    It goes directly to the answer. Cleaner computation.
  
  If steps_to_target(24) >> steps_to_target(2) × (24/2):
    24-layer is disproportionately slow — gradient dilution confirmed.
    The cancellation scaffold is the bottleneck, not the loss landscape.
  
  If steps_to_target(24) ≈ steps_to_target(2) × (24/2):
    24-layer trains proportionally — just more parameters, same efficiency.
    The cancellation is not impeding gradient flow.
""")

# Save
import json as _json
out = {k: {kk: (vv if not isinstance(vv, dict) else
                {str(kkk): vvv for kkk, vvv in vv.items()})
           for kk, vv in v.items()} for k, v in results.items()}
with open('/mnt/user-data/outputs/monodromy_training.json', 'w') as f:
    _json.dump(out, f, indent=2, default=lambda x: float(x) if isinstance(x,(np.floating, np.integer)) else x)
print("  Saved → monodromy_training.json")
