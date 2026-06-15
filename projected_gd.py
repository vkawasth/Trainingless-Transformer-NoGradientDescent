#!/usr/bin/env python3
"""
Projected Gradient Descent via Pants Coproduct
================================================
Standard gradient descent wastes 73-99% of each step on dead directions
— gradient components that will be cancelled by later layers.

The pants coproduct Δ(∇L_l) = (live, dead) identifies these.

PROJECTED GD:
  For each parameter W in layer l:
    g = ∂L/∂W  (standard gradient)
    g_live = π_l(g)  (project onto active subspace of δJ_l)
    W ← W - η × g_live

Where π_l = U_l U_l^T, U_l = left singular vectors of δJ_l (rank-k).

The projection is in WEIGHT SPACE via the chain rule:
  ∂L/∂W_l is related to ∂L/∂h_l via ∂h_l/∂W_l.
  
  Projecting ∇_{h_l} L onto U_l gives the live hidden-state gradient.
  Pulling back: ∇_W^live L = (∂h_l/∂W_l)^T π_l ∇_{h_l} L

In practice (simpler approximation):
  Compute U_l from δJ_l at current step.
  For each weight matrix W in block l:
    g_W = ∂L/∂W
    Project g_W: g_W_live = U_l U_l^T g_W  (if shapes align)
    OR: scale g_W by live fraction at layer l

We test three variants:
  A: Standard AdamW (baseline)
  B: Projected GD — hard projection onto live subspace
  C: Weighted GD — scale lr by live/total ratio at each layer

Usage: python projected_gd.py
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=128; N_HEADS=4; N_LAYERS=8; BATCH=8; SEQ=64; LR=3e-4
PROJ=16   # smaller proj for speed — active subspace
TARGET=4.0
MAX_STEPS=400

print(f"\n{'='*65}")
print(f"  PROJECTED GRADIENT DESCENT")
print(f"  Purifying gradient via pants coproduct Δ(∇L) = (live, dead)")
print(f"  d={D}  layers={N_LAYERS}  proj={PROJ}")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=json.load(f)
with open('/tmp/val_ids.json')   as f: val_ids=json.load(f)
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

def get_batch(split='train'):
    data=train_t if split=='train' else val_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ]   for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

# ── Architecture ──────────────────────────────────────────────────────────────
class Attn(nn.Module):
    def __init__(self,d,nh):
        super().__init__()
        self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
        self.WQ=nn.Linear(d,d,bias=False); self.WK=nn.Linear(d,d,bias=False)
        self.WV=nn.Linear(d,d,bias=False); self.op=nn.Linear(d,d,bias=False)
        self.ln=nn.LayerNorm(d)
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h):
        B,S,D_=h.shape; H=self.nh; dh=self.dh
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2)
        K=self.WK(h).view(B,S,H,dh).transpose(1,2)
        V=self.WV(h).view(B,S,H,dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/self.sc
        mask=torch.triu(torch.ones(S,S),diagonal=1).bool()
        sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        out=(F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D_)
        return self.ln(h+self.op(out))

class FF(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.g=nn.Linear(d,d*2,bias=False); self.v=nn.Linear(d,d*2,bias=False)
        self.o=nn.Linear(d*2,d,bias=False); self.n=nn.LayerNorm(d)
        for w in [self.g,self.v,self.o]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h): return self.n(h+self.o(F.silu(self.g(h))*self.v(h)))

class Block(nn.Module):
    def __init__(self,d,nh): super().__init__(); self.attn=Attn(d,nh); self.ff=FF(d)
    def forward(self,h): return self.ff(self.attn(h))

class LM(nn.Module):
    def __init__(self,d,nh,nl):
        super().__init__()
        self.te=nn.Embedding(VOCAB,d); self.pe=nn.Embedding(512,d)
        self.blocks=nn.ModuleList([Block(d,nh) for _ in range(nl)])
        self.ln_f=nn.LayerNorm(d)
        self.head=nn.Linear(d,VOCAB,bias=False); self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
        self._nl=nl
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def forward_hs(self,x):
        """Forward returning per-layer hidden states."""
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h)
        for b in self.blocks: h=b(h); hs.append(h)
        return hs

def eval_val(model, n=40):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clr(s, total=MAX_STEPS, warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ── Active subspace at each layer ────────────────────────────────────────────
def get_active_subspaces(model, x, pos, m=PROJ):
    """
    For each layer l, compute U_l [d, rank_l] — active subspace of δJ_l.
    Returns list of U_l tensors and live fractions.
    """
    subspaces = []
    with torch.enable_grad():
        hs = []
        h = model.te(x) + model.pe(torch.arange(x.shape[1]))
        hs.append(h.detach())
        for b in model.blocks:
            h = b(h); hs.append(h.detach())

        for l in range(model._nl):
            h_in = hs[l][0]   # [SEQ, D]
            seq, d_ = h_in.shape; m_ = min(m, seq, d_)
            _, _, Vt = torch.linalg.svd(h_in, full_matrices=False)
            U_basis = Vt[:m_, :].T.detach()   # [d, m]

            # Jacobian via vjp
            J = torch.zeros(m_, m_)
            for i in range(m_):
                h_node = h_in.clone().unsqueeze(0).detach().requires_grad_(True)
                h_out = model.blocks[l](h_node)[0]
                v = h_out[0, pos, :] if h_out.dim()==3 else h_out[pos, :]
                (v * U_basis[:, i]).sum().backward()
                g = h_node.grad
                g = (g[0, pos, :] if g.dim()==3 else g[pos, :]).detach()
                J[:, i] = U_basis.T @ g

            J_np = J.T.numpy()
            dJ = J_np - np.eye(m_)
            sv = np.linalg.svd(dJ, compute_uv=False)
            rank = int(np.sum(sv > sv[0]*0.10)) if sv[0]>1e-8 else 1
            rank = max(rank, 1)

            # Active subspace: top singular vectors of δJ in m-space
            # lifted back to d-space via U_basis
            U_sv, _, _ = np.linalg.svd(dJ)
            U_active_m = U_sv[:, :rank]   # [m, rank]
            # Lift: U_active_d = U_basis @ U_active_m  [d, rank]
            U_active_d = U_basis.numpy() @ U_active_m   # [d, rank]

            live_frac = rank / m_

            subspaces.append({
                'U': torch.tensor(U_active_d, dtype=torch.float32),  # [d, rank]
                'rank': rank,
                'live_frac': live_frac,
                'dJ_norm': float(np.linalg.norm(dJ)),
            })

    return subspaces

def project_gradients(model, subspaces):
    """
    For each block l, project weight gradients onto active subspace.
    
    For W [d_out, d_in] in block l:
      U_l [d, rank] is the active subspace.
      
    Projection: g_proj = U_l U_l^T g  (acts on output dimension)
    This removes gradient components in dead output directions.
    """
    for l, sp in enumerate(subspaces):
        U = sp['U']   # [d, rank]
        P = U @ U.T   # [d, d] projection matrix

        block = model.blocks[l]
        # Project gradient of each weight matrix in the block
        for param_name, param in block.named_parameters():
            if param.grad is None: continue
            g = param.grad.data   # [d_out, d_in] or [d] for bias

            if g.dim() == 2:
                d_out, d_in = g.shape
                if d_out == D:
                    # Project along output dimension
                    param.grad.data = (P @ g)
                elif d_in == D:
                    # Project along input dimension
                    param.grad.data = (g @ P.T)
                # else: leave unchanged (different dimension, e.g. FFN intermediate)
            # LayerNorm biases etc: leave unchanged

def live_fraction_scale(model, subspaces):
    """
    Softer version: scale each layer's learning rate by its live fraction.
    Less aggressive, more stable.
    """
    for l, sp in enumerate(subspaces):
        scale = sp['live_frac']
        block = model.blocks[l]
        for param in block.parameters():
            if param.grad is not None:
                param.grad.data *= scale

# ── Run comparison ────────────────────────────────────────────────────────────
def run(name, project_fn=None, proj_every=5, seed=42):
    """
    project_fn: if not None, called after backward to modify gradients.
    proj_every: compute active subspaces every N steps (expensive).
    """
    torch.manual_seed(seed)
    model = LM(D, N_HEADS, N_LAYERS)
    model._nl = N_LAYERS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)

    stt = None; vals = []; t0 = time.time()
    subspaces = None
    x_ref, _ = get_batch('val'); x_ref = x_ref[0:1]
    pos = SEQ // 2

    print(f"\n  [{name}]")
    for step in range(1, MAX_STEPS+1):
        for pg in opt.param_groups: pg['lr'] = clr(step)

        model.train()
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()

        # Compute active subspaces periodically
        if project_fn is not None and step % proj_every == 1:
            subspaces = get_active_subspaces(model, x_ref, pos)

        # Apply gradient projection
        if project_fn is not None and subspaces is not None:
            project_fn(model, subspaces)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 25 == 0 or step == 1:
            vl = eval_val(model, n=20)
            vals.append((step, vl))
            if vl < TARGET and stt is None:
                stt = step
                print(f"    *** TARGET val<{TARGET} at step {step} ***")
            t_elapsed = time.time()-t0
            proj_info = f"  proj_every={proj_every}" if project_fn else ""
            print(f"    {step:>4}/{MAX_STEPS}  val={vl:.4f}  t={t_elapsed:.0f}s{proj_info}")

    return stt, vals, time.time()-t0

# ── A: Standard AdamW ─────────────────────────────────────────────────────────
print("Running A: Standard AdamW (baseline)...")
stt_A, vals_A, time_A = run("Standard AdamW", project_fn=None)

# ── B: Hard projection onto live subspace ─────────────────────────────────────
print("\nRunning B: Projected GD (hard projection, every 5 steps)...")
stt_B, vals_B, time_B = run("Projected GD (hard)", project_fn=project_gradients, proj_every=5)

# ── C: Live-fraction scaling ──────────────────────────────────────────────────
print("\nRunning C: Live-fraction scaled GD (every 5 steps)...")
stt_C, vals_C, time_C = run("Live-fraction GD", project_fn=live_fraction_scale, proj_every=5)

# ── D: Projection every step (maximum signal) ────────────────────────────────
print("\nRunning D: Projected GD (every step — maximum signal)...")
stt_D, vals_D, time_D = run("Projected GD (every step)", project_fn=project_gradients, proj_every=1)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RESULTS  (target = val < {TARGET})")
print("="*65)

def fmt(s): return str(s) if s else f">{MAX_STEPS}"
results = [
    ("Standard AdamW",            stt_A, vals_A[-1][1], time_A, None),
    ("Projected GD (hard, /5)",   stt_B, vals_B[-1][1], time_B, stt_A),
    ("Live-fraction GD (/5)",     stt_C, vals_C[-1][1], time_C, stt_A),
    ("Projected GD (every step)", stt_D, vals_D[-1][1], time_D, stt_A),
]

print(f"\n  {'Method':30}  {'Steps→<{}'.format(TARGET):>10}  {'Final val':>10}  {'Time':>8}  {'Speedup':>8}")
print("  "+"-"*72)
for name, stt, fval, t, base in results:
    stt_s = fmt(stt)
    if base and stt and stt < MAX_STEPS:
        speedup = f"{base/stt:.2f}x" if base else "—"
    elif base and stt and stt >= MAX_STEPS:
        speedup = "<1x"
    else:
        speedup = "—"
    print(f"  {name:30}  {stt_s:>10}  {fval:>10.4f}  {t:>7.1f}s  {speedup:>8}")

# Loss curve comparison
print(f"\n  Loss curve (val at each checkpoint):")
print(f"  {'step':>6}  {'Standard':>10}  {'Proj/5':>10}  {'LiveFrac':>10}  {'Proj/1':>10}")
print("  "+"-"*48)
steps_A = {s:v for s,v in vals_A}
steps_B = {s:v for s,v in vals_B}
steps_C = {s:v for s,v in vals_C}
steps_D = {s:v for s,v in vals_D}
all_steps = sorted(set(steps_A)|set(steps_B)|set(steps_C)|set(steps_D))
for s in all_steps:
    va = steps_A.get(s,'—')
    vb = steps_B.get(s,'—')
    vc = steps_C.get(s,'—')
    vd = steps_D.get(s,'—')
    print(f"  {s:>6}  "
          f"  {va:>8.4f}" if isinstance(va,float) else f"  {'—':>8}",
          end='')
    for v in [vb,vc,vd]:
        print(f"  {v:>8.4f}" if isinstance(v,float) else f"  {'—':>8}", end='')
    print()

print(f"""
INTERPRETATION:
  If Projected GD reaches val<{TARGET} in fewer steps than Standard:
    Dead gradient waste is real and the pants coproduct projection
    successfully removes it. The A∞ active subspace is the right
    coordinate system for gradient descent.

  If Projected GD is SLOWER or SAME speed:
    The projection is too crude (U_l from a single input at one pos)
    OR the dead directions are necessary — they carry gradient signal
    that is not dead in the Jacobian sense but live in the loss sense.
    This would mean: the pants coproduct decomposition at one position
    does not generalize to the full batch gradient.

  If Live-fraction scaling works better than hard projection:
    Soft signal (scale by live fraction) is more robust than
    hard cutoff (project to subspace). The active subspace is
    approximately correct but not precisely so.
""")
