#!/usr/bin/env python3
"""
Weight Projection Optimizer
=============================
After each gradient step, project W_K onto the geometry-consistent set.

This is PROJECTED GRADIENT DESCENT on the weight manifold.
The projection acts on weights AFTER the gradient update — not on gradients.

THREE PROJECTIONS:
  A: Hessenberg projection — W_K ← hessenberg(W_K)
     Moves W_K toward Toda fixed point (closed-form, O(d²))
     
  B: QR step — W_K ← RQ  (one discrete Toda step)
     Applies the time-1 Toda flow map directly to W_K
     
  C: Convex combination — W_K ← (1-α)*W_K + α*hessenberg(W_K)
     α sweeps from 0 (pure GD) to 1 (pure Hessenberg)
     Finds optimal blend of gradient signal + geometric correction

DIFFERENCE FROM ALL PRIOR EXPERIMENTS:
  Previous: filter/project the GRADIENT (failed — Adam cancels, SGD same)
  This: project the WEIGHTS after each update (untested, correct framing)
  
  The gradient is unmodified — full-rank, full-signal.
  The geometry corrects where the weights land, not how they move.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import hessenberg as scipy_hessenberg, qr as scipy_qr

D=128; N_HEADS=4; N_LAYERS=8; BATCH=8; SEQ=64
LR=0.05; MOMENTUM=0.9; TARGET=4.0; MAX_STEPS=400; LOG=25

print(f"\n{'='*65}")
print(f"  WEIGHT PROJECTION OPTIMIZER")
print(f"  Project W_K onto Hessenberg/Toda manifold AFTER each gradient step")
print(f"  d={D}  layers={N_LAYERS}")
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
        mask=torch.triu(torch.ones(S,S,device=h.device),diagonal=1).bool()
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

def eval_val(model,n=40):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clr(s,total=MAX_STEPS,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ── Weight projections ────────────────────────────────────────────────────────
def hessenberg_project(W_np):
    """H = P^T W P — upper Hessenberg form. Isospectral."""
    H, _ = scipy_hessenberg(W_np, calc_q=True)
    return H

def qr_toda_step(W_np):
    """One discrete Toda step: W = QR → RQ. Moves toward triangular."""
    Q, R = scipy_qr(W_np)
    return R @ Q

def apply_weight_projection(model, mode='hessenberg', alpha=1.0, proj_every=1, step=1):
    """
    After gradient step: project W_K at each layer.
    mode: 'hessenberg' | 'qr' | 'convex'
    alpha: blend (0=none, 1=full projection) for convex mode
    proj_every: apply every N steps
    """
    if step % proj_every != 0:
        return
    with torch.no_grad():
        for blk in model.blocks:
            W = blk.attn.WK.weight.data   # [d, d]
            W_np = W.numpy().copy()
            if mode == 'hessenberg':
                W_proj = hessenberg_project(W_np)
                W.copy_(torch.tensor(W_proj, dtype=torch.float32))
            elif mode == 'qr':
                W_proj = qr_toda_step(W_np)
                W.copy_(torch.tensor(W_proj, dtype=torch.float32))
            elif mode == 'convex':
                W_hess = hessenberg_project(W_np)
                W_proj = (1-alpha)*W_np + alpha*W_hess
                W.copy_(torch.tensor(W_proj, dtype=torch.float32))

# ── Measure Hessenberg distance ───────────────────────────────────────────────
def hessenberg_distance(model):
    """Mean ||W_K[i,j] for j<i-1||_F / ||W_K||_F across layers."""
    dists=[]
    with torch.no_grad():
        for blk in model.blocks:
            W=blk.attn.WK.weight.data.numpy()
            d=W.shape[0]
            below=np.array([W[i,j] for i in range(d) for j in range(i-1)]) if d>2 else np.array([0.])
            dists.append(float(np.linalg.norm(below)/max(np.linalg.norm(W),1e-8)))
    return float(np.mean(dists))

# ── Training runs ─────────────────────────────────────────────────────────────
def run(name, proj_mode=None, alpha=1.0, proj_every=1, seed=42):
    torch.manual_seed(seed)
    model=LM(D,N_HEADS,N_LAYERS)
    opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=MOMENTUM,nesterov=True)

    stt=None; vals=[]; hess_dists=[]; t0=time.time()
    print(f"\n  [{name}]")

    for step in range(1,MAX_STEPS+1):
        for pg in opt.param_groups: pg['lr']=clr(step)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()

        # WEIGHT PROJECTION — after gradient step
        if proj_mode is not None:
            apply_weight_projection(model, mode=proj_mode,
                                   alpha=alpha, proj_every=proj_every, step=step)

        if step%LOG==0 or step==1:
            vl=eval_val(model,n=20); vals.append((step,vl))
            hd=hessenberg_distance(model); hess_dists.append(hd)
            if vl<TARGET and stt is None:
                stt=step; print(f"    *** TARGET at step {step} ***")
            proj_info=f"  hess_dist={hd:.4f}" if proj_mode else ""
            print(f"    {step:>4}/{MAX_STEPS}  val={vl:.4f}"
                  f"  t={time.time()-t0:.0f}s{proj_info}")

    fval=eval_val(model,n=100)
    hd_final=hessenberg_distance(model)
    return stt, vals, time.time()-t0, fval, hd_final

# ── Baseline ──────────────────────────────────────────────────────────────────
print("A: SGD baseline (no projection)...")
stt_A,vals_A,t_A,fval_A,hd_A=run("SGD baseline", proj_mode=None)

print("\nB: SGD + Hessenberg projection (every step)...")
stt_B,vals_B,t_B,fval_B,hd_B=run("SGD + Hessenberg", proj_mode='hessenberg', proj_every=1)

print("\nC: SGD + QR Toda step (every step)...")
stt_C,vals_C,t_C,fval_C,hd_C=run("SGD + QR Toda", proj_mode='qr', proj_every=1)

print("\nD: SGD + Convex blend α=0.1 (light Hessenberg correction)...")
stt_D,vals_D,t_D,fval_D,hd_D=run("SGD + convex α=0.1", proj_mode='convex', alpha=0.1, proj_every=1)

print("\nE: SGD + Convex blend α=0.5 (equal gradient + geometry)...")
stt_E,vals_E,t_E,fval_E,hd_E=run("SGD + convex α=0.5", proj_mode='convex', alpha=0.5, proj_every=1)

print("\nF: SGD + Hessenberg every 10 steps (cheaper)...")
stt_F,vals_F,t_F,fval_F,hd_F=run("SGD + Hessenberg /10", proj_mode='hessenberg', proj_every=10)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RESULTS  (target val<{TARGET})")
print("="*65)

def fmt(s): return str(s) if s else f">{MAX_STEPS}"
rows=[
    ("SGD baseline",          stt_A,fval_A,t_A,hd_A),
    ("SGD+Hessenberg /1",     stt_B,fval_B,t_B,hd_B),
    ("SGD+QR Toda /1",        stt_C,fval_C,t_C,hd_C),
    ("SGD+convex α=0.1 /1",   stt_D,fval_D,t_D,hd_D),
    ("SGD+convex α=0.5 /1",   stt_E,fval_E,t_E,hd_E),
    ("SGD+Hessenberg /10",    stt_F,fval_F,t_F,hd_F),
]
print(f"\n  {'Method':24}  {'Steps→<4':>9}  {'Final':>8}  {'Time':>7}  {'HessDist':>10}  {'vs A':>8}")
print("  "+"-"*70)
base=stt_A or MAX_STEPS
for name,stt,fval,t,hd in rows:
    sp=f"{base/stt:.2f}x" if stt and stt<MAX_STEPS else "—"
    print(f"  {name:24}  {fmt(stt):>9}  {fval:>8.4f}  {t:>6.1f}s  {hd:>10.4f}  {sp:>8}")

print(f"\n  Loss curves:")
print(f"  {'step':>5}  {'base':>8}  {'Hess/1':>8}  {'QR/1':>8}  {'α0.1':>8}  {'α0.5':>8}  {'Hess/10':>9}")
print("  "+"-"*58)
vd={s:v for s,v in vals_A}; vB={s:v for s,v in vals_B}
vC={s:v for s,v in vals_C}; vD={s:v for s,v in vals_D}
vE={s:v for s,v in vals_E}; vF={s:v for s,v in vals_F}
for s in sorted(vd):
    print(f"  {s:>5}  {vd.get(s,0):>8.4f}  {vB.get(s,0):>8.4f}"
          f"  {vC.get(s,0):>8.4f}  {vD.get(s,0):>8.4f}"
          f"  {vE.get(s,0):>8.4f}  {vF.get(s,0):>9.4f}")

print(f"""
KEY QUESTIONS:

  Hessenberg distance (lower = more Hessenberg):
    Baseline final: {hd_A:.4f}
    After Hessenberg projection: {hd_B:.4f}  (should be ~0)
    After QR Toda: {hd_C:.4f}

  If ANY projection method reaches val<4.0 FASTER than baseline:
    Weight projection onto the Toda manifold genuinely accelerates training.
    The geometry is not just descriptive — it is prescriptive.
    The closed-form projection substitutes for gradient steps.

  If convex α=0.1 beats α=0.5:
    Light geometry correction helps, heavy correction hurts.
    The gradient needs room to explore; geometry corrects the endpoint.

  If all fail:
    The Hessenberg structure emerges naturally from the loss.
    Forcing it early interferes with the loss-guided exploration.
    The geometry is an EFFECT of convergence, not a CAUSE of it.
""")
