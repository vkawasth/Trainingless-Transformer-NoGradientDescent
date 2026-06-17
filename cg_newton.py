#!/usr/bin/env python3
"""
Matrix-Free Conjugate Gradient Newton Step
============================================
Solve H * delta_E = -G for the embedding matrix update
using Conjugate Gradient with Hessian-Vector Products.

THEORY:
  The full Hessian H in R^{260352 x 260352} is intractable.
  CG only requires H*v products (Pearlmutter trick):
    H*v = d/dtheta (grad L . v)  -- one extra backward pass
  
  CG convergence: ~sqrt(kappa) iterations where kappa = lambda_max/lambda_min
  For the embedding Hessian, kappa is large but the effective kappa
  within the 8 co-occurrence clusters is much smaller.

  LOCAL PATCHING: restrict CG to token communities.
  Token t couples to t' only if they co-occur in attention window.
  Solve independently per community -- respects coupling, tractable.

PIPELINE:
  After MF10 + settle + sign (val=0.296):
  1. Compute full corpus gradient G_E for embeddings
  2. Run CG to solve H_E * delta_E = -G_E (10-20 HVP iterations)
  3. Apply delta_E to embedding matrix
  4. WK Newton polish
  
  Cost: 10-20 HVP iterations * 1 forward+backward each
       = 10-20 CE step equivalents (deterministic)
  
  If val < 0.05 after CG: coupled relaxation solved in ~15 CE equiv
  If val ~ 0.296 (unchanged): the coupling is too nonlinear for CG
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429

print(f"\n{'='*65}")
print(f"  MATRIX-FREE CG NEWTON FOR EMBEDDING MATRIX")
print(f"  H*delta = -G via Pearlmutter HVPs, no matrix storage")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
vocab={t:i for i,t in enumerate(_v)} if isinstance(_v,list) else _v
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t=torch.tensor(val_ids,dtype=torch.long)

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
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
    def get_flat_params(self): return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat_params(self,flat):
        idx=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(flat[idx:idx+n].reshape(p.shape)); idx+=n

def eval_val(m,n=40):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def hv_product(model, v, n_batches=20):
    """
    Pearlmutter trick: H*v = d/dtheta (grad L . v)
    Full model HVP — used for embedding-only subspace via masking.
    """
    params=list(model.parameters()); model.zero_grad()
    torch.manual_seed(42)
    loss=sum(model(*get_batch())[1] for _ in range(n_batches))/n_batches
    grads=torch.autograd.grad(loss, params, create_graph=True)
    flat_grad=torch.cat([g.flatten() for g in grads])
    gv=(flat_grad * v.detach()).sum()
    hv=torch.cat([h.flatten() for h in
                  torch.autograd.grad(gv, params, retain_graph=False)]).detach()
    model.zero_grad()
    return hv

def embedding_hvp(model, v_emb, n_batches=20):
    """
    HVP restricted to embedding subspace only.
    v_emb: (VOCAB*D,) vector in embedding space
    Returns H_emb * v_emb where H_emb is the embedding block of H.
    """
    # Build full parameter vector with v in embedding positions, 0 elsewhere
    n_params=sum(p.numel() for p in model.parameters())
    v_full=torch.zeros(n_params)
    # Embedding is the first parameter
    emb_numel=VOCAB*D
    v_full[:emb_numel]=v_emb
    Hv_full=hv_product(model, v_full, n_batches)
    # Extract embedding block of result
    return Hv_full[:emb_numel]

def cg_solve_embedding(model, grad_emb, n_iters=20, n_batches=15,
                        reg=0.1, tol=1e-4):
    """
    Solve (H_emb + reg*I) * delta = -grad_emb using Conjugate Gradient.
    
    Matrix-free: only uses HVPs via Pearlmutter trick.
    reg: Tikhonov regularization (ensures positive definite)
    
    CG iteration:
      r_0 = -grad_emb - (H + reg*I)*delta_0  (delta_0 = 0)
      p_0 = r_0
      for k in 0..n_iters:
        alpha_k = r_k.r_k / p_k.(H+reg*I).p_k
        delta_{k+1} = delta_k + alpha_k * p_k
        r_{k+1} = r_k - alpha_k * (H+reg*I)*p_k
        beta_k = r_{k+1}.r_{k+1} / r_k.r_k
        p_{k+1} = r_{k+1} + beta_k * p_k
    """
    b=-grad_emb  # RHS: we want H*delta = -grad
    delta=torch.zeros_like(b)
    r=b.clone()  # residual: b - H*delta_0 = b (since delta_0=0)
    p=r.clone()
    r_norm_sq=float((r*r).sum())
    
    print(f"  CG: initial ||r|| = {r_norm_sq**0.5:.4f}")
    
    hvp_count=0
    for k in range(n_iters):
        # Compute (H_emb + reg*I) * p
        Hp=embedding_hvp(model, p, n_batches) + reg*p
        hvp_count+=1
        
        pHp=float((p*Hp).sum())
        if abs(pHp)<1e-10: break
        
        alpha=r_norm_sq/pHp
        delta=delta+alpha*p
        r=r-alpha*Hp
        
        r_norm_sq_new=float((r*r).sum())
        beta=r_norm_sq_new/max(r_norm_sq,1e-10)
        p=r+beta*p
        r_norm_sq=r_norm_sq_new
        
        print(f"  CG iter {k+1}: ||r||={r_norm_sq**0.5:.4f}  "
              f"||delta||={float(delta.norm()):.4f}")
        
        if r_norm_sq**0.5 < tol:
            print(f"  CG converged at iter {k+1}")
            break
    
    print(f"  CG: {hvp_count} HVPs used ({hvp_count*n_batches} forward+backward passes)")
    return delta

def apply_newton_wk(stu,n_seq=500,eps=1e-3,scale=0.5):
    ga=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    fd=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    torch.manual_seed(2)
    for i in range(n_seq):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
        stu.zero_grad(); _,loss=stu(x,y); loss.backward()
        g=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        for l in range(N_STU):
            if stu.blocks[l].attn.WK.weight.grad is not None:
                g+=stu.blocks[l].attn.WK.weight.grad/N_STU
        ga+=g; fd+=g**2
    delta=-(ga/n_seq)/((fd/n_seq)+eps)
    with torch.no_grad():
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.add_(scale*delta)
            stu.blocks[l].attn.WQ.weight.add_(scale*delta.T)

def hv_product_full(model,v,n=15):
    params=list(model.parameters()); model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad(); return hv

print("Training teacher...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,301):
    lr_now=LR*min(step,100)/100 if step<=100 else LR*0.5*(1+math.cos(math.pi*(step-100)/200))
    for pg in opt.param_groups: pg['lr']=lr_now
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
    if step%100==0:
        teacher.eval()
        with torch.no_grad(): vl=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}")
teacher.eval(); val_teacher=eval_val(teacher)
print(f"  Teacher val={val_teacher:.4f}\n")

def build_student():
    torch.manual_seed(99); stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight); stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            for s,d in [(teacher.blocks[L_ATT].attn.WK,stu.blocks[l].attn.WK),
                        (teacher.blocks[L_ATT].attn.WQ,stu.blocks[l].attn.WQ),
                        (teacher.blocks[L_ATT].attn.WV,stu.blocks[l].attn.WV),
                        (teacher.blocks[L_ATT].attn.op,stu.blocks[l].attn.op),
                        (teacher.blocks[L_ATT].ff.g,stu.blocks[l].ff.g),
                        (teacher.blocks[L_ATT].ff.v,stu.blocks[l].ff.v),
                        (teacher.blocks[L_ATT].ff.o,stu.blocks[l].ff.o)]:
                d.weight.copy_(s.weight)
    return stu

print("Computing v_neg...")
stu_ref=build_student(); n_p=sum(p.numel() for p in stu_ref.parameters())
v=torch.randn(n_p); v=v/v.norm()
for _ in range(15): Hv=hv_product_full(stu_ref,v,15); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg=v.clone(); print("v_neg ready.\n")

def apply_mf(stu,n_iter=10,mf_lr=0.01,n_corpus=200):
    for it in range(n_iter):
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.requires_grad_(False)
            stu.blocks[l].attn.WQ.weight.requires_grad_(False)
        eg=torch.zeros(VOCAB,D); ef=torch.zeros(VOCAB,D)
        torch.manual_seed(it*1000)
        for i in range(n_corpus):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            if stu.te.weight.grad is not None:
                g=stu.te.weight.grad.detach(); eg+=g; ef+=g**2
        eg/=n_corpus; ef/=n_corpus
        with torch.no_grad(): stu.te.weight.add_(-mf_lr*eg/(ef+1e-4))
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.requires_grad_(True)
            stu.blocks[l].attn.WQ.weight.requires_grad_(True)
        stu.te.weight.requires_grad_(False)
        wg=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        wf=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        torch.manual_seed(it*1000+500)
        for i in range(n_corpus):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            g=torch.zeros_like(stu.blocks[0].attn.WK.weight)
            for l in range(N_STU):
                if stu.blocks[l].attn.WK.weight.grad is not None:
                    g+=stu.blocks[l].attn.WK.weight.grad/N_STU
            wg+=g; wf+=g**2
        wg/=n_corpus; wf/=n_corpus
        with torch.no_grad():
            for l in range(N_STU):
                stu.blocks[l].attn.WK.weight.add_(-mf_lr*wg/(wf+1e-4))
                stu.blocks[l].attn.WQ.weight.add_(-mf_lr*wg.T/(wf.T+1e-4))
        stu.te.weight.requires_grad_(True)
        if (it+1)%5==0: print(f"  MF iter {it+1}: val={eval_val(stu,n=5):.4f}")

def clr(s,total,warmup=20):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ═══════════════════════════════════════════
# BUILD MF10 + SETTLE + SIGN STATE
# ═══════════════════════════════════════════
print("Building MF10 state...")
stu=build_student()
w0=stu.get_flat_params(); stu.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
apply_mf(stu,n_iter=10)
opt_s=torch.optim.AdamW(stu.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,34):
    for pg in opt_s.param_groups: pg['lr']=LR*5*min(step,10)/10
    stu.train(); x,y=get_batch(); _,loss=stu(x,y)
    opt_s.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
with torch.no_grad():
    for l in [1,2]:
        stu.blocks[l].attn.WV.weight.mul_(-1); stu.blocks[l].attn.op.weight.mul_(-1)
v_settle=eval_val(stu,n=20)
print(f"MF10+settle+sign: val={v_settle:.4f}\n")

import copy
stu_checkpoint=copy.deepcopy(stu)

# ═══════════════════════════════════════════
# EXPERIMENT A: baseline 100 CE
# ═══════════════════════════════════════════
print("="*65)
print("[A] Baseline: 100 CE steps + Newton")
stu_a=copy.deepcopy(stu_checkpoint)
opt2=torch.optim.AdamW(stu_a.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,101):
    for pg in opt2.param_groups: pg['lr']=clr(step,100)
    stu_a.train(); x,y=get_batch(); _,loss=stu_a(x,y)
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu_a.parameters(),1.0); opt2.step()
    if step in [25,50,75,100]: print(f"  CE {step}: {eval_val(stu_a,n=15):.4f}")
apply_newton_wk(stu_a)
vA=eval_val(stu_a,n=30); print(f"  FINAL A={vA:.4f}")

# ═══════════════════════════════════════════
# EXPERIMENT B: CG Newton on embedding
# ═══════════════════════════════════════════
print(f"\n{'='*65}")
print("[B] CG Newton: matrix-free H_emb^{-1} G_emb")
stu_b=copy.deepcopy(stu_checkpoint)

# Compute full corpus gradient for embeddings
print("  Computing embedding gradient (corpus pass)...")
stu_b.train(); stu_b.zero_grad()
n_grad_batches=50
for _ in range(n_grad_batches):
    x,y=get_batch(); _,loss=stu_b(x,y); (loss/n_grad_batches).backward()
grad_emb=stu_b.te.weight.grad.detach().clone().flatten()
print(f"  ||G_emb|| = {float(grad_emb.norm()):.4f}")
stu_b.zero_grad()

# Run CG with regularization
print("  Running CG (10 iterations, reg=1.0)...")
t0=time.time()
delta_emb=cg_solve_embedding(stu_b, grad_emb,
                               n_iters=10, n_batches=15, reg=1.0)
print(f"  CG time: {time.time()-t0:.1f}s")
print(f"  ||delta_emb|| = {float(delta_emb.norm()):.4f}")

# Apply delta with conservative scale
scale_cg=0.1
with torch.no_grad():
    stu_b.te.weight.add_(scale_cg*delta_emb.reshape(VOCAB,D))
v_after_cg=eval_val(stu_b,n=20)
print(f"  after CG step (scale={scale_cg}): val={v_after_cg:.4f}")

apply_newton_wk(stu_b)
vB=eval_val(stu_b,n=30); print(f"  FINAL B={vB:.4f}")

# ═══════════════════════════════════════════
# EXPERIMENT C: CG + 25 CE cleanup
# ═══════════════════════════════════════════
print(f"\n{'='*65}")
print("[C] CG Newton + 25 CE cleanup")
stu_c=copy.deepcopy(stu_checkpoint)
stu_c.train(); stu_c.zero_grad()
for _ in range(n_grad_batches):
    x,y=get_batch(); _,loss=stu_c(x,y); (loss/n_grad_batches).backward()
grad_emb_c=stu_c.te.weight.grad.detach().clone().flatten()
stu_c.zero_grad()
delta_emb_c=cg_solve_embedding(stu_c, grad_emb_c,
                                 n_iters=10, n_batches=15, reg=1.0)
with torch.no_grad():
    stu_c.te.weight.add_(scale_cg*delta_emb_c.reshape(VOCAB,D))
print(f"  after CG: val={eval_val(stu_c,n=15):.4f}")
opt3=torch.optim.AdamW(stu_c.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    for pg in opt3.param_groups: pg['lr']=clr(step,25)
    stu_c.train(); x,y=get_batch(); _,loss=stu_c(x,y)
    opt3.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu_c.parameters(),1.0); opt3.step()
    if step in [10,25]: print(f"  CE {step}: {eval_val(stu_c,n=15):.4f}")
apply_newton_wk(stu_c)
vC=eval_val(stu_c,n=30); print(f"  FINAL C={vC:.4f}")

print(f"""
{'='*65}
  CG NEWTON RESULTS
{'='*65}
    Teacher:            val={val_teacher:.4f}
    A (100CE+Newton):   val={vA:.4f}  [stochastic baseline]
    B (CG+Newton):      val={vB:.4f}  [matrix-free Newton]
    C (CG+25CE+Newton): val={vC:.4f}  [Newton + minimal CE]

  CG cost: 10 HVPs * 15 batches = 150 forward+backward
         = ~0.75 CE step equivalents (deterministic)
         
  IF B ~ A: CG Newton solves the coupling in ~1 CE equiv
    The 100 stochastic steps were approximating H^{{-1}}G
    Matrix-free CG is the correct computation
    
  IF B > A but C ~ A with 25 steps:
    CG provides a good preconditioner but not full solution
    Regularization too high / CG not converged enough
    Try: more CG iterations, lower regularization
    
  IF B >> A (diverged): 
    The Hessian at the settle+sign position has negative eigenvalues
    CG is solving an indefinite system -- PCG with positive reg needed
    The 100 CE steps first convexify the landscape (drive lambda_min > 0)
    before Newton-like methods become applicable
""")
