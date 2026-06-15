#!/usr/bin/env python3
"""
Early Exit at L14 Attractor
=============================
Protocol A: h_out = (1/sv_bwd) * h_14  → output head
Protocol B: h_out = M_fwd @ h_14       → output head

sv(M_bwd) ≈ 0.93  →  scale = 1/0.93 ≈ 1.075

Tests on the trained model:
  - Full 24-layer inference (baseline)
  - Exit at L14, scale by 1.075           (Protocol A)
  - Exit at L14, apply sqrtm(M_fwd)       (Protocol B)
  - Exit at L10, L12, L16, L18            (sensitivity)

Measures: val loss, logit cosine with full model, inference time.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import sqrtm as scipy_sqrtm

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; PROJ=48

print(f"\n{'='*65}")
print(f"  EARLY EXIT AT L{L_ATT} ATTRACTOR")
print(f"  Protocol A: h_out = 1.075 × h_14  (scalar scale)")
print(f"  Protocol B: h_out = M_fwd @ h_14  (monodromy correction)")
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
        self._nl=nl
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def forward_exit(self,x,exit_layer,scale=1.0,M_proj=None):
        """Forward pass exiting at exit_layer, applying scale or M_proj."""
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for l,b in enumerate(self.blocks):
            h=b(h)
            if l==exit_layer:
                h=h*scale
                if M_proj is not None:
                    # Apply M_proj in the top-m subspace
                    h_np=h.detach().numpy()
                    B_,S_,D_=h_np.shape
                    h_flat=h_np.reshape(-1,D_)
                    h_proj=(h_flat@M_proj.T)
                    h=torch.tensor(h_proj.reshape(B_,S_,D_),dtype=torch.float32)
                break
        logits=self.head(self.ln_f(h))
        return logits

def eval_exit(model,exit_layer,scale=1.0,M_proj=None,n=80):
    model.eval(); ls=[]; cos_list=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val')
            # Full model logits
            logits_full,_=model(x,y)
            # Exit logits
            logits_exit=model.forward_exit(x,exit_layer,scale,M_proj)
            loss=F.cross_entropy(logits_exit.reshape(-1,VOCAB),y.reshape(-1))
            ls.append(loss.item())
            # Cosine with full model
            lf=logits_full.reshape(-1,VOCAB)
            le=logits_exit.reshape(-1,VOCAB)
            cos=F.cosine_similarity(lf,le,dim=-1).mean().item()
            cos_list.append(cos)
    return float(np.mean(ls)), float(np.mean(cos_list))

def time_inference(model,exit_layer=None,scale=1.0,M_proj=None,n=200):
    model.eval(); t0=time.time()
    with torch.no_grad():
        for _ in range(n):
            x,_=get_batch('val')
            if exit_layer is None: model(x)
            else: model.forward_exit(x,exit_layer,scale,M_proj)
    return (time.time()-t0)/n*1000  # ms per batch

# ── Train source ──────────────────────────────────────────────────────────────
print("Step 1: Train 24-layer source (300 steps)...")
torch.manual_seed(42)
source=LM(D,N_HEADS,N_LAYERS)
opt=torch.optim.AdamW(source.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
def clr(s,total=300,warmup=100):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step)
    source.train(); x,y=get_batch(); _,loss=source(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(source.parameters(),1.0); opt.step()
    if step%100==0:
        source.eval()
        with torch.no_grad():
            vl=np.mean([source(*get_batch('val'))[1].item() for _ in range(10)])
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
        source.train()
source.eval()
val_full,_=eval_exit(source,N_LAYERS-1,scale=1.0)  # full model
# Re-evaluate properly
source.eval()
with torch.no_grad():
    ls=[source(*get_batch('val'))[1].item() for _ in range(80)]
val_src=float(np.mean(ls))
print(f"  Source val={val_src:.4f}\n")

# ── Compute M_fwd ─────────────────────────────────────────────────────────────
print(f"Step 2: Compute M_fwd (L0→L{L_ATT}) and sv(M_bwd)...")
x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
with torch.no_grad():
    hs=[]; h=source.te(x_ref)+source.pe(torch.arange(SEQ)); hs.append(h.detach())
    for b in source.blocks: h=b(h); hs.append(h.detach())
pos=SEQ//2; m=min(PROJ,SEQ,D)

def layer_jac(block,h_in,pos,m):
    seq,d_=h_in.shape; m=min(m,seq,d_)
    _,_,Vt=torch.linalg.svd(h_in,full_matrices=False)
    U=Vt[:m,:].T.detach(); J=np.zeros((m,m))
    with torch.enable_grad():
        for i in range(m):
            h=h_in.clone().unsqueeze(0).detach().requires_grad_(True)
            h_out=block(h)[0]; v=h_out[0,pos,:]
            (v*U[:,i]).sum().backward()
            g=h.grad[0,pos,:].detach(); J[:,i]=(U.T@g).numpy()
    return J.T, U.detach().numpy(), m

Js=[]; U0=None; ma=None
print("  Computing Jacobians...", flush=True)
for l in range(N_LAYERS):
    J,U,m_=layer_jac(source.blocks[l],hs[l][0],pos,m)
    Js.append(J)
    if U0 is None: U0=U; ma=m_
    if (l+1)%8==0: print(f"    L{l+1}...",flush=True)

M_fwd=np.eye(ma)
for l in range(L_ATT+1): M_fwd=Js[l]@M_fwd
M_bwd=np.eye(ma)
for l in range(N_LAYERS-1,L_ATT,-1): M_bwd=Js[l]@M_bwd

sv_fwd=np.linalg.svd(M_fwd,compute_uv=False)
sv_bwd=np.linalg.svd(M_bwd,compute_uv=False)
sv_bwd_mean=float(sv_bwd[:8].mean())
scale_A=1.0/sv_bwd_mean
print(f"  sv(M_fwd)[:4] = {sv_fwd[:4].round(3)}")
print(f"  sv(M_bwd)[:4] = {sv_bwd[:4].round(3)}")
print(f"  Mean sv(M_bwd) over top-8: {sv_bwd_mean:.4f}")
print(f"  Protocol A scale = 1/{sv_bwd_mean:.4f} = {scale_A:.4f}\n")

# Protocol B: lift M_fwd to d-space as a projection operator
# M_fwd lives in the m-dim subspace spanned by U0
# Lift: M_fwd_d = U0 @ M_fwd @ U0^T + (I - U0 U0^T)
U0_t=torch.tensor(U0,dtype=torch.float32)   # [d, m]
M_fwd_t=torch.tensor(M_fwd,dtype=torch.float32)   # [m, m]
M_fwd_d_np=(U0@M_fwd@U0.T + np.eye(D)-U0@U0.T)   # [d, d]
print(f"  M_fwd_d shape: {M_fwd_d_np.shape}  (lifted to full d-space)\n")

# ── Evaluate all exit strategies ──────────────────────────────────────────────
print(f"Step 3: Evaluate exit strategies...")
t_full=time_inference(source,exit_layer=None)

results=[]

# Full model
v0=val_src; c0=1.0; t0_=t_full
results.append(("Full 24-layer (baseline)", N_LAYERS-1, v0, c0, t0_))
print(f"  Full 24L:  val={v0:.4f}  cos=1.000  t={t_full:.1f}ms")

# Exit at various layers, no scaling
for l_exit in [10,12,14,16,18]:
    vl,cos=eval_exit(source,l_exit,scale=1.0)
    ti=time_inference(source,exit_layer=l_exit)
    tag=f"Exit L{l_exit} (no scale)"
    results.append((tag,l_exit,vl,cos,ti))
    print(f"  {tag}: val={vl:.4f}  cos={cos:.4f}  t={ti:.1f}ms")

# Protocol A: exit at L14, scale by 1/sv_bwd
vA,cosA=eval_exit(source,L_ATT,scale=scale_A)
tA=time_inference(source,exit_layer=L_ATT,scale=scale_A)
results.append((f"Protocol A: exit L{L_ATT}, scale={scale_A:.3f}",L_ATT,vA,cosA,tA))
print(f"  Protocol A (L{L_ATT}, ×{scale_A:.3f}): val={vA:.4f}  cos={cosA:.4f}  t={tA:.1f}ms")

# Protocol B: exit at L14, apply M_fwd_d
vB,cosB=eval_exit(source,L_ATT,scale=1.0,M_proj=M_fwd_d_np)
tB=time_inference(source,exit_layer=L_ATT,scale=1.0,M_proj=M_fwd_d_np)
results.append((f"Protocol B: exit L{L_ATT} + M_fwd",L_ATT,vB,cosB,tB))
print(f"  Protocol B (L{L_ATT} + M_fwd): val={vB:.4f}  cos={cosB:.4f}  t={tB:.1f}ms")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FINAL RESULTS")
print("="*65)
print(f"\n  {'Method':38}  {'val':>7}  {'cos':>7}  {'t(ms)':>7}  {'speedup':>8}")
print("  "+"-"*72)
for name,l,vl,cos,ti in results:
    sp=f"{t_full/ti:.2f}x" if ti<t_full else "—"
    dv=f"+{vl-val_src:.3f}" if vl>val_src else f"{vl-val_src:.3f}"
    print(f"  {name:38}  {vl:>7.4f}  {cos:>7.4f}  {ti:>7.1f}  {sp:>8}")

print(f"""
READING:

  sv(M_bwd) ≈ {sv_bwd_mean:.3f} across the tail (L{L_ATT+1}→L{N_LAYERS-1}).
  Protocol A scale = {scale_A:.3f} compensates the tail contraction.

  If Protocol A val ≈ full model val:
    The tail is pure contraction. Early exit with scalar rescaling
    is lossless. sv(M_bwd) completely characterises the tail.

  If Protocol A val >> full model val:
    The tail does directional work beyond scalar contraction.
    The Dehn gap (1.4) represents real structure, not just scale.
    Need Protocol B (M_fwd correction) or accept the quality gap.

  The speedup is {t_full/tA:.2f}x at L{L_ATT} exit.
  Layer sensitivity (exit L10-L18) shows where quality degrades.
""")
