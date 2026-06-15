#!/usr/bin/env python3
"""
Monodromy Distillation
=======================
Replace gradient descent on cross-entropy with:
  hidden state matching loss: ||h_out^(2L) - h_out^(24L)||^2

The 24-layer model's output IS the target.
The 2-layer model learns to reproduce it directly.

WHY THIS WORKS:
  Cross-entropy loss: gradient = dL/dh_24 × (dh_24/dW) — noisy
  Distillation loss:  gradient = (h_2 - h_24) × (dh_2/dW) — exact target
  
  Each step has a clean, exact target instead of a stochastic gradient.
  Expected: 2-4x fewer steps to equivalent quality.

Also tests Way C (spectral initialization):
  Initialize W_O, W_V aligned with top SVD directions of sqrtm(M_24)
  at standard norm (0.02) — right direction, safe norm.

Usage: python monodromy_distillation.py
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import sqrtm as scipy_sqrtm

D=256; N_HEADS=4; BATCH=8; SEQ=64; LR=3e-4
TARGET_LM = 4.0   # val loss target

print(f"\n{'='*65}")
print(f"  MONODROMY DISTILLATION  d={D}  n_heads={N_HEADS}")
print(f"  2-layer learns from 24-layer hidden states, not cross-entropy")
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
    def hidden_out(self,x):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        return self.ln_f(h)   # final hidden state before head [B,S,D]
    def lm_loss(self,x,y):
        _,loss=self.forward(x,y); return loss

def eval_lm(model, n=50):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clr(step, total, warmup=100):
    if step<=warmup: return LR*step/warmup
    return LR*0.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))

# ── Step 1: Train 24-layer reference ─────────────────────────────────────────
print("Step 1: Train 24-layer reference (200 steps, cross-entropy)...")
torch.manual_seed(42)
ref = LM(D,N_HEADS,24)
opt=torch.optim.AdamW(ref.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time(); stt_ref=None
for step in range(1,201):
    for pg in opt.param_groups: pg['lr']=clr(step,200)
    ref.train(); x,y=get_batch(); _,loss=ref(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(ref.parameters(),1.0); opt.step()
    if step%100==0:
        vl=eval_lm(ref,n=20)
        if vl<TARGET_LM and stt_ref is None: stt_ref=step
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
val_ref=eval_lm(ref)
ref.eval()
print(f"  24-layer val = {val_ref:.4f}  (steps to <{TARGET_LM}: {stt_ref or '>200'})\n")

# ── Step 2: Compute monodromy ─────────────────────────────────────────────────
print("Step 2: Compute sqrtm(M_24) for spectral initialization...")
m=min(64, SEQ, D)

def layer_jacobian(block, h_in, pos, m=64):
    seq,d_=h_in.shape; m=min(m,seq,d_)
    _,_,Vt=torch.linalg.svd(h_in,full_matrices=False)
    U=Vt[:m,:].T.detach()
    J=np.zeros((m,m))
    for i in range(m):
        h=h_in.clone().unsqueeze(0).detach().requires_grad_(True)
        h_out=block(h)[0]
        v=h_out[0,pos,:] if h_out.dim()==3 else h_out[pos,:]
        (v*U[:,i]).sum().backward()
        g=h.grad
        g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
        J[:,i]=(U.T@g).numpy()
    return J.T, U.detach().numpy(), m

x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
with torch.no_grad():
    hs_b=ref.hidden_out(x_ref)   # just need hidden states
    # get per-layer hidden states
    hs=[]; h_=ref.te(x_ref)+ref.pe(torch.arange(SEQ)); hs.append(h_.detach())
    for b in ref.blocks: h_=b(h_); hs.append(h_.detach())

pos=SEQ//2; Js=[]
print("  Computing Jacobians...", flush=True)
for l in range(24):
    J,U,ma=layer_jacobian(ref.blocks[l], hs[l][0], pos, m=m)
    Js.append(J)
    if (l+1)%8==0: print(f"    L{l+1}...", flush=True)

M24=np.eye(ma)
for J in reversed(Js): M24=J@M24

sqM=np.real(scipy_sqrtm(M24))   # 2-layer target monodromy
Usvd,svals,Vsvdh=np.linalg.svd(sqM)
print(f"  sqrtm(M_24): sv=[{svals[0]:.3f}, {svals[1]:.3f}, ..., {svals[-1]:.4f}]")

# ── Step 3: Spectral initialization (Way C) ───────────────────────────────────
print("\nStep 3: Spectral initialization — small norm, right directions...")

def spectral_init(model, Usvd, Vsvdh, svals, U_basis, scale=0.02):
    """
    Initialize W_op and W_V aligned with sqrtm(M_24) singular vectors
    at standard scale (0.02). Right subspace, safe norm.
    U_basis: [d, m] — the projection basis used for Jacobians
    """
    d_=D; m_=ma
    # Top n_heads singular directions (in m-space, lift to d-space)
    k=min(N_HEADS*2, m_)   # use 2*n_heads directions for richer init
    
    # Left singular vectors → output projection directions
    U_top = Usvd[:, :k]           # [m, k]
    U_d   = U_basis @ U_top        # [d, k] — lifted to d-space
    
    # Right singular vectors → value directions
    V_top = Vsvdh[:k, :]          # [k, m]
    V_d   = V_top @ U_basis.T     # [k, d] — lifted to d-space
    
    for blk in model.blocks:
        with torch.no_grad():
            # Set W_op (output projection) rows to U_d directions
            w = blk.attn.op.weight.data   # [d, d]
            w.zero_()
            for i in range(k):
                # add scale * outer product of i-th left/right singular vectors
                w += scale * torch.outer(
                    torch.tensor(U_d[:,i], dtype=torch.float32),
                    torch.tensor(V_d[i,:], dtype=torch.float32)
                ) * float(svals[i])
            
            # Set W_V (value) rows to V_d directions
            wv = blk.attn.WV.weight.data  # [d, d]
            wv.zero_()
            for i in range(k):
                wv[i % d_] = scale * torch.tensor(V_d[i % k,:], dtype=torch.float32)

torch.manual_seed(42)
model_spec = LM(D,N_HEADS,2)
# Need U_basis from the reference Jacobian computation
_,_,Vt_ref=torch.linalg.svd(hs[0][0], full_matrices=False)
U_basis_np = Vt_ref[:ma,:].T.numpy()
spectral_init(model_spec, Usvd, Vsvdh, svals, U_basis_np)
val_spec_0=eval_lm(model_spec)
print(f"  Spectral init (no training): val = {val_spec_0:.4f}")

# ── Step 4: Distillation training (Way B) ─────────────────────────────────────
print("\nStep 4: Distillation training — 2-layer learns from 24-layer hidden states...")

def distillation_loss(model2, ref_model, x):
    """
    Loss = ||hidden_out(model2, x) - hidden_out(ref, x)||^2  / (B*S*D)
    Exact target at each step. No cross-entropy noise.
    """
    h2  = model2.hidden_out(x)   # [B, S, D]
    with torch.no_grad():
        h24 = ref_model.hidden_out(x)  # [B, S, D] — fixed target
    return ((h2 - h24)**2).mean()

STEPS_DISTILL = 200
LOG = 25

def run_distillation(model, name, use_distill=True, seed=42):
    torch.manual_seed(seed)
    # Re-init to make fair comparison if needed
    opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    stt=None; t0=time.time()
    print(f"\n  [{name}]")
    for step in range(1, STEPS_DISTILL+1):
        for pg in opt.param_groups: pg['lr']=clr(step, STEPS_DISTILL)
        model.train()
        x,y=get_batch()
        if use_distill:
            loss=distillation_loss(model, ref, x)
        else:
            _,loss=model(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%LOG==0 or step==1:
            vl=eval_lm(model, n=20)
            if vl<TARGET_LM and stt is None:
                stt=step; print(f"    *** TARGET val<{TARGET_LM} at step {step} ***")
            loss_type="distill" if use_distill else "CE"
            print(f"    {step:>4}/{STEPS_DISTILL}  val={vl:.4f} [{loss_type}]  t={time.time()-t0:.0f}s")
    return stt, eval_lm(model)

# A: Random init + CE (baseline)
torch.manual_seed(99)
m_rand_ce = LM(D,N_HEADS,2)
stt_rand_ce, val_rand_ce = run_distillation(m_rand_ce, "Random init + CE", use_distill=False)

# B: Random init + distillation
torch.manual_seed(99)
m_rand_dist = LM(D,N_HEADS,2)
stt_rand_dist, val_rand_dist = run_distillation(m_rand_dist, "Random init + DISTILLATION", use_distill=True)

# C: Spectral init + CE
torch.manual_seed(42)
m_spec_ce = LM(D,N_HEADS,2)
spectral_init(m_spec_ce, Usvd, Vsvdh, svals, U_basis_np)
stt_spec_ce, val_spec_ce = run_distillation(m_spec_ce, "Spectral init + CE", use_distill=False)

# D: Spectral init + distillation
torch.manual_seed(42)
m_spec_dist = LM(D,N_HEADS,2)
spectral_init(m_spec_dist, Usvd, Vsvdh, svals, U_basis_np)
stt_spec_dist, val_spec_dist = run_distillation(m_spec_dist, "Spectral init + DISTILLATION", use_distill=True)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RESULTS  (target = val < {TARGET_LM})")
print("="*65)
def fmt_steps(s): return str(s) if s else f">{STEPS_DISTILL}"
rows = [
    ("24-layer reference (CE, 200 steps)", val_ref,       fmt_steps(stt_ref)),
    ("Random init + CE",                   val_rand_ce,   fmt_steps(stt_rand_ce)),
    ("Random init + DISTILLATION",         val_rand_dist, fmt_steps(stt_rand_dist)),
    ("Spectral init + CE",                 val_spec_ce,   fmt_steps(stt_spec_ce)),
    ("Spectral init + DISTILLATION",       val_spec_dist, fmt_steps(stt_spec_dist)),
]
print(f"\n  {'Method':40}  {'Final val':>10}  {'Steps→<{}'.format(TARGET_LM):>12}")
print("  "+"-"*66)
for name, val, stt in rows:
    print(f"  {name:40}  {val:>10.4f}  {stt:>12}")

# Compute speedups
print(f"\n  Speedups vs Random+CE (steps to reach val<{TARGET_LM}):")
base=int(stt_rand_ce) if stt_rand_ce else STEPS_DISTILL+1
for name, _, stt_s in rows[2:]:
    s = int(stt_s) if stt_s.isdigit() else STEPS_DISTILL+1
    speedup = base / s if s <= STEPS_DISTILL else 0
    print(f"    {name:40} {f'{speedup:.1f}x' if speedup > 0 else 'did not reach target'}")
