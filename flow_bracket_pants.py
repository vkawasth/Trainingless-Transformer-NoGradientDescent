#!/usr/bin/env python3
"""
Flow, Manifold, Bracket, Pants Coproduct
==========================================
For each feature vector h_l(x) at each layer l and training step t:

1. FLOW:     dh_l/dt = -∂L/∂h_l   (gradient flow on hidden state manifold)
2. MANIFOLD: T_{h_l}M = span(U_l)  (tangent space = active subspace of δJ_l)
3. BRACKET:  [δh_l/dl, dh_l/dt] = δJ_l ∇L - ∇L δJ_l  (commutator)
4. PANTS:    Δ(h_l) = (π_A h_l, π_B h_l)  (live/dead decomposition)

The bracket ||[X,Y]||_l → 0 as t → T (convergence):
  When bracket vanishes: depth and training commute.
  The A∞ structure stabilizes. This IS convergence.

The pants coproduct tracks:
  ||π_A ∂L/∂h_l||  = gradient in live directions  (useful signal)
  ||π_B ∂L/∂h_l||  = gradient in dead directions  (cancelled signal)
  
  At W*: ||π_B ∂L/∂h_l|| → 0  (A∞ relations satisfied)

Usage: python flow_bracket_pants.py
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=128; N_HEADS=4; N_LAYERS=8; BATCH=8; SEQ=64; LR=3e-4
PROJ=24  # projection dim for Jacobians (< SEQ)
CHECKPOINTS = [1,5,10,20,40,80,160,300]  # training steps to measure at

print(f"\n{'='*65}")
print(f"  FLOW · MANIFOLD · BRACKET · PANTS COPRODUCT")
print(f"  d={D}  layers={N_LAYERS}  heads={N_HEADS}")
print(f"  Tracking ||[X,Y]||_l and Δ(h_l) along gradient descent")
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
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def forward_with_grads(self,x,y):
        """Forward pass retaining h_l and ∂L/∂h_l at each layer."""
        hs=[]; grad_hs=[]
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        h.requires_grad_(True); hs.append(h)
        for b in self.blocks:
            h=b(h); h.retain_grad(); hs.append(h)
        logits=self.head(self.ln_f(hs[-1]))
        loss=F.cross_entropy(logits.view(-1,VOCAB),y.view(-1))
        loss.backward()
        grad_hs=[h_.grad.detach() if h_.grad is not None else torch.zeros_like(h_)
                 for h_ in hs]
        return [h_.detach() for h_ in hs], grad_hs, loss.item()

# ── Jacobian via vjp ──────────────────────────────────────────────────────────
def layer_jacobian_proj(block, h_in, pos, m):
    """J_proj = U^T J U  [m,m], U = top-m left sing vecs of h_in."""
    seq,d_=h_in.shape; m=min(m,seq,d_)
    _,_,Vt=torch.linalg.svd(h_in,full_matrices=False)
    U=Vt[:m,:].T.detach()
    J=np.zeros((m,m))
    with torch.enable_grad():
        for i in range(m):
            h=h_in.clone().unsqueeze(0).detach().requires_grad_(True)
            h_out=block(h)[0]
            v=h_out[0,pos,:] if h_out.dim()==3 else h_out[pos,:]
            (v*U[:,i]).sum().backward()
            g=h.grad
            g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
            J[:,i]=(U.T@g).numpy()
    return J.T, U.detach().numpy(), m

# ── Measurements at one checkpoint ───────────────────────────────────────────
def measure(model, x_fixed, y_fixed, pos):
    """
    At current W:
    Returns per-layer:
      bracket_norm: ||[δJ_l, ∇L_l]||_F
      live_grad:    ||π_A ∇L_l||  (gradient in active subspace)
      dead_grad:    ||π_B ∇L_l||  (gradient in dead subspace)
      rank:         rank of δJ_l
      dJ_norm:      ||δJ_l||
    """
    # Forward with gradient accumulation at each h_l
    model.train()  # need grads
    hs, grad_hs, loss = model.forward_with_grads(x_fixed, y_fixed)
    model.zero_grad()

    results = []
    for l in range(N_LAYERS):
        h_in = hs[l][0]   # [SEQ, D] — single sample

        # Jacobian J_l in projected space
        with torch.no_grad():
            J, U, m = layer_jacobian_proj(model.blocks[l], h_in, pos, PROJ)
        dJ = J - np.eye(m)

        # Active subspace: top singular vectors of δJ
        sv_dJ = np.linalg.svd(dJ, compute_uv=False)
        rank = int(np.sum(sv_dJ > sv_dJ[0]*0.10)) if sv_dJ[0]>1e-8 else 0
        U_sv, _, _ = np.linalg.svd(dJ)
        U_active = U_sv[:, :rank] if rank > 0 else U_sv[:, :1]  # [m, rank]

        # Gradient ∇_h_l L at this layer, projected to m-space
        g_h = grad_hs[l][0, pos, :].numpy()   # [D]
        U_t = torch.tensor(U, dtype=torch.float32)   # [D, m]
        g_proj = U_t.numpy().T @ g_h   # [m]

        # Pants coproduct: live vs dead gradient
        g_live = U_active @ (U_active.T @ g_proj)   # [m] — in active subspace
        g_dead = g_proj - g_live                      # [m] — in dead subspace
        live_grad = float(np.linalg.norm(g_live))
        dead_grad = float(np.linalg.norm(g_dead))

        # Bracket [δJ_l, ∇L_l] = δJ_l @ diag(g_proj) - diag(g_proj) @ δJ_l
        # More precisely: [δJ, G] where G = outer product approximation
        # bracket = δJ_l @ g_proj_outer - g_proj_outer @ δJ_l
        # g_proj_outer [m,m] = g_proj @ g_proj^T / ||g_proj||
        if np.linalg.norm(g_proj) > 1e-10:
            g_outer = np.outer(g_proj, g_proj) / np.linalg.norm(g_proj)
        else:
            g_outer = np.zeros((m,m))
        bracket = dJ @ g_outer - g_outer @ dJ
        bracket_norm = float(np.linalg.norm(bracket, 'fro'))

        results.append({
            'bracket': bracket_norm,
            'live':    live_grad,
            'dead':    dead_grad,
            'rank':    rank,
            'dJ_norm': float(np.linalg.norm(dJ)),
            'loss':    loss,
        })

    return results, loss

# ── Training loop with checkpoint measurements ────────────────────────────────
torch.manual_seed(42)
model = LM(D, N_HEADS, N_LAYERS)
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
def clr(s,total=CHECKPOINTS[-1],warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# Fixed reference batch for measurements
torch.manual_seed(0)
x_fixed, y_fixed = get_batch('val')
x_fixed, y_fixed = x_fixed[0:1], y_fixed[0:1]  # single sample
pos_fixed = SEQ // 2

print(f"Reference: seq={SEQ} pos={pos_fixed} proj={PROJ}\n")
print(f"Training for {CHECKPOINTS[-1]} steps, measuring at: {CHECKPOINTS}\n")

history = {}   # step → list of per-layer dicts
MAX_STEP = CHECKPOINTS[-1]

t0 = time.time()
for step in range(1, MAX_STEP+1):
    for pg in opt.param_groups: pg['lr'] = clr(step)

    model.train()
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step in CHECKPOINTS:
        meas, val_loss = measure(model, x_fixed, y_fixed, pos_fixed)
        history[step] = meas
        mean_bracket = np.mean([m['bracket'] for m in meas])
        mean_live    = np.mean([m['live']    for m in meas])
        mean_dead    = np.mean([m['dead']    for m in meas])
        mean_rank    = np.mean([m['rank']    for m in meas])
        print(f"  step {step:>4}  loss={val_loss:.4f}  "
              f"||[X,Y]||={mean_bracket:.4f}  "
              f"live={mean_live:.4f}  dead={mean_dead:.4f}  "
              f"rank={mean_rank:.1f}  t={time.time()-t0:.0f}s")

# ── Analysis: bracket decay ───────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  BRACKET DECAY  ||[δJ_l, ∇L_l]||  along training")
print(f"  → 0 as training converges = depth and training commute at W*")
print("="*65)

steps = sorted(history.keys())
print(f"\n  {'step':>6}  {'loss':>7}", end='')
for l in range(0, N_LAYERS, 2): print(f"  L{l:>2}", end='')
print(f"  {'mean':>8}")
print("  "+"-"*(9+7+6*(N_LAYERS//2)+9))

for step in steps:
    meas = history[step]
    loss = meas[0]['loss']
    print(f"  {step:>6}  {loss:>7.4f}", end='')
    for l in range(0, N_LAYERS, 2):
        print(f"  {meas[l]['bracket']:>4.2f}", end='')
    print(f"  {np.mean([m['bracket'] for m in meas]):>8.4f}")

# ── Pants coproduct: live vs dead gradient ────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PANTS COPRODUCT  Δ(∇L_l) = (live, dead) gradient components")
print(f"  live  = gradient in active subspace (useful signal)")
print(f"  dead  = gradient in cancelled subspace (ghost signal)")
print(f"  At W*: dead → 0  (A∞ relations satisfied)")
print("="*65)

print(f"\n  {'step':>6}  {'loss':>7}  {'mean live':>10}  {'mean dead':>10}  "
      f"{'live/dead':>10}  {'mean rank':>10}")
print("  "+"-"*58)
for step in steps:
    meas = history[step]
    loss  = meas[0]['loss']
    live  = np.mean([m['live'] for m in meas])
    dead  = np.mean([m['dead'] for m in meas])
    rank  = np.mean([m['rank'] for m in meas])
    ratio = live / max(dead, 1e-8)
    print(f"  {step:>6}  {loss:>7.4f}  {live:>10.4f}  {dead:>10.4f}  "
          f"{ratio:>10.3f}  {rank:>10.1f}")

# ── Per-layer pants at final checkpoint ──────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PER-LAYER PANTS AT FINAL STEP (step={steps[-1]})")
print(f"  Which layers have most dead gradient? (those are the bottlenecks)")
print("="*65)
final = history[steps[-1]]
print(f"\n  {'L':>4}  {'bracket':>9}  {'live':>8}  {'dead':>8}  "
      f"{'live/dead':>10}  {'rank':>6}  {'||dJ||':>8}")
print("  "+"-"*58)
for l, m in enumerate(final):
    print(f"  L{l:>2}  {m['bracket']:>9.4f}  {m['live']:>8.4f}  "
          f"{m['dead']:>8.4f}  {m['live']/max(m['dead'],1e-8):>10.3f}  "
          f"{m['rank']:>6}  {m['dJ_norm']:>8.4f}")

# ── The key question: does bracket → 0? ─────────────────────────────────────
print(f"\n{'='*65}")
print(f"  CONVERGENCE OF BRACKET")
print("="*65)
b0   = np.mean([m['bracket'] for m in history[steps[0]]])
bfin = np.mean([m['bracket'] for m in history[steps[-1]]])
decay = b0 / max(bfin, 1e-8)
print(f"""
  Initial bracket (step {steps[0]}):  {b0:.4f}
  Final bracket   (step {steps[-1]}): {bfin:.4f}
  Decay ratio: {decay:.2f}x

  {"BRACKET DECAYS: depth and training converge to commuting flows." if decay > 2 else
   "BRACKET STABLE: depth-flow and training-flow remain non-commuting."}

  Interpretation:
  {"  The A∞ structure STABILIZES during training." if decay > 2 else
   "  The A∞ structure is STILL CHANGING at the final checkpoint."}
  {"  Gradient descent is finding a fixed point where [X,Y]=0." if decay > 2 else
   "  More training steps needed, or the fixed point has [X,Y]≠0."}

  Live/dead ratio at step {steps[0]}:  {np.mean([m['live'] for m in history[steps[0]]])/max(np.mean([m['dead'] for m in history[steps[0]]]),1e-8):.3f}
  Live/dead ratio at step {steps[-1]}: {np.mean([m['live'] for m in history[steps[-1]]])/max(np.mean([m['dead'] for m in history[steps[-1]]]),1e-8):.3f}

  {"Live/dead INCREASES: gradient becomes more concentrated in live directions." 
   if np.mean([m['live'] for m in history[steps[-1]]])/max(np.mean([m['dead'] for m in history[steps[-1]]]),1e-8) > np.mean([m['live'] for m in history[steps[0]]])/max(np.mean([m['dead'] for m in history[steps[0]]]),1e-8)
   else "Live/dead DECREASES: gradient spreads into dead directions (unusual)."}

  This IS the pants coproduct in action:
  Δ(∇L) = (live component, dead component)
  The dead component → 0 as the A∞ relations are satisfied.
  That satisfaction IS convergence.
""")
