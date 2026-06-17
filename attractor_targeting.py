#!/usr/bin/env python3
"""
Attractor Targeting — Minimalist Pipeline Test
===============================================
Test the "Jump-to-Attractor" hypothesis:

PROPOSED:
  Step 0: Saddle exit v_neg (2 CE equiv)
  Step 1: 5x LR impulse (1-5 steps only)
  Step 2: Newton H^{-1} gradient (1 step)
  Total: ~5 CE equivalent steps → val=0.04

PREDICTION TO TEST:
  After saddle exit + sign correction + Newton:
  Is the Hessian positive definite? (Required for Newton to work)
  Does Newton at valley entrance reach val=0.04 in one step?

EXPERIMENTS:
  A: Full pipeline (confirmed best, val=0.035)
  B: Saddle exit + 5 CE steps (5x LR) + sign + Newton → zero-shot target
  C: Saddle exit + 10 CE steps (5x LR) + sign + Newton
  D: Saddle exit + 33 CE steps (5x LR) + sign + Newton (no 167 CE)
  E: Saddle exit + 33 CE (5x) + sign + 10 CE + Newton

Measure Hessian min eigenvalue at each stage to confirm
when the landscape becomes positive definite (Newton-valid).
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429  # from saddle_exit.py

print(f"\n{'='*65}")
print(f"  ATTRACTOR TARGETING")
print(f"  Test: can Newton replace 167 within-basin steps?")
print(f"  Measure Hessian sign at each stage")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
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
    def flat_grad(self): return torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()) for p in self.parameters()])

def clr(s,total=200,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))
def eval_val(model,n=40):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def hv_product(model,v,n=10):
    """H*v via double backprop."""
    params=list(model.parameters()); model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad(); return hv

def min_hessian_ev(model,n_iter=10,n_batches=10):
    """Min Hessian eigenvalue via power iteration on -H."""
    n=sum(p.numel() for p in model.parameters())
    v=torch.randn(n); v=v/v.norm(); lam=0.0
    for _ in range(n_iter):
        Hv=hv_product(model,v,n_batches); neg=-Hv
        lam=-float(neg.norm()); v=neg/max(-lam,1e-10)
    return lam

def apply_newton_wk(stu,n_seq=500,eps=1e-3,scale=0.5):
    ga=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    fd=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    torch.manual_seed(2)
    for i in range(n_seq):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
        stu.zero_grad(); _,loss=stu(x,y); loss.backward()
        g=sum(stu.blocks[l].attn.WK.weight.grad for l in range(N_STU) if stu.blocks[l].attn.WK.weight.grad is not None)/N_STU
        ga+=g; fd+=g**2
    delta=-(ga/n_seq)/((fd/n_seq)+eps)
    with torch.no_grad():
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.add_(scale*delta)
            stu.blocks[l].attn.WQ.weight.add_(scale*delta.T)

print("Training teacher (300 steps)...")
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
        stu.pe.weight.copy_(teacher.pe.weight); stu.ln_f.weight.copy_(teacher.ln_f.weight); stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            for src,dst in [(teacher.blocks[L_ATT].attn.WK,stu.blocks[l].attn.WK),
                            (teacher.blocks[L_ATT].attn.WQ,stu.blocks[l].attn.WQ),
                            (teacher.blocks[L_ATT].attn.WV,stu.blocks[l].attn.WV),
                            (teacher.blocks[L_ATT].attn.op,stu.blocks[l].attn.op),
                            (teacher.blocks[L_ATT].ff.g,stu.blocks[l].ff.g),
                            (teacher.blocks[L_ATT].ff.v,stu.blocks[l].ff.v),
                            (teacher.blocks[L_ATT].ff.o,stu.blocks[l].ff.o)]:
                dst.weight.copy_(src.weight)
    return stu

# Compute v_neg once
print("Computing saddle exit direction v_neg...")
stu_ref=build_student()
n_params=sum(p.numel() for p in stu_ref.parameters())
v=torch.randn(n_params); v=v/v.norm()
for it in range(15):
    Hv=hv_product(stu_ref,v,20); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg=v.clone()
print(f"  Done. v_neg computed.\n")

def build_with_saddle_exit():
    stu=build_student()
    w0=stu.get_flat_params()
    stu.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
    return stu

def run(label, settle_steps=33, settle_lr=5.0, basin_steps=167,
        do_sign=True, do_newton=True, measure_hessian_at=None):
    stu=build_with_saddle_exit()
    v0=eval_val(stu,n=20)
    print(f"\n  [{label}]  after saddle exit: val={v0:.4f}")

    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR*settle_lr,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,settle_steps+1):
        for pg in opt_s.param_groups: pg['lr']=LR*settle_lr*min(step,10)/10
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
    v_settle=eval_val(stu,n=20)
    print(f"    after {settle_steps} settle steps (lr={settle_lr}x): val={v_settle:.4f}")

    # Measure Hessian min eigenvalue HERE (valley entrance)
    if measure_hessian_at=='entrance':
        lam=min_hessian_ev(stu,n_iter=8,n_batches=8)
        print(f"    Hessian lambda_min at valley entrance: {lam:.4f}  "
              f"({'POSITIVE=convex' if lam>0 else 'NEGATIVE=saddle/ridge — Newton INVALID'})")

    if do_sign:
        with torch.no_grad():
            for l in [1,2]:
                stu.blocks[l].attn.WV.weight.mul_(-1)
                stu.blocks[l].attn.op.weight.mul_(-1)
        v_sign=eval_val(stu,n=20)
        print(f"    after sign correction: val={v_sign:.4f}")

    if basin_steps>0:
        opt2=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
        for step in range(1,basin_steps+1):
            for pg in opt2.param_groups: pg['lr']=clr(step,basin_steps)
            stu.train(); x,y=get_batch(); _,loss=stu(x,y)
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt2.step()
            if step in [10,25,50,100,167]:
                v=eval_val(stu,n=20)
                print(f"    basin step {step:>4}: val={v:.4f}{' ✓' if v<val_teacher else ''}")

    # Measure Hessian at basin floor
    if measure_hessian_at=='floor' or measure_hessian_at=='both':
        lam=min_hessian_ev(stu,n_iter=8,n_batches=8)
        print(f"    Hessian lambda_min at basin floor: {lam:.4f}  "
              f"({'POSITIVE=convex=Newton valid' if lam>0 else 'NEGATIVE=still in saddle'})")

    if do_newton:
        apply_newton_wk(stu)
        v_n=eval_val(stu); print(f"    after Newton: val={v_n:.4f}")

    vf=eval_val(stu,n=30); print(f"    FINAL={vf:.4f}")
    return vf

print("="*65)
print("EXPERIMENTS: Is the 167-step basin descent replaceable by Newton?")
print("="*65)

# A: Full confirmed pipeline
vA=run("A-Full-pipeline",settle_steps=33,settle_lr=5.0,
        basin_steps=167,do_sign=True,do_newton=True,
        measure_hessian_at='both')

# B: After sign correction only, Newton immediately (no basin steps)
vB=run("B-Saddle+sign+Newton-only",settle_steps=33,settle_lr=5.0,
        basin_steps=0,do_sign=True,do_newton=True,
        measure_hessian_at='entrance')

# C: 10 basin steps then Newton
vC=run("C-Saddle+sign+10CE+Newton",settle_steps=33,settle_lr=5.0,
        basin_steps=10,do_sign=True,do_newton=True)

# D: 50 basin steps then Newton
vD=run("D-Saddle+sign+50CE+Newton",settle_steps=33,settle_lr=5.0,
        basin_steps=50,do_sign=True,do_newton=True)

# E: 100 basin steps then Newton
vE=run("E-Saddle+sign+100CE+Newton",settle_steps=33,settle_lr=5.0,
        basin_steps=100,do_sign=True,do_newton=True)

print(f"""
{'='*65}
  ATTRACTOR TARGETING RESULTS
{'='*65}

  FINAL:
    Teacher:              val={val_teacher:.4f}
    A (full, 167 basin):  val={vA:.4f}  [confirmed best]
    B (0 basin steps):    val={vB:.4f}  diff A-B={vA-vB:+.4f}
    C (10 basin steps):   val={vC:.4f}  diff A-C={vA-vC:+.4f}
    D (50 basin steps):   val={vD:.4f}  diff A-D={vA-vD:+.4f}
    E (100 basin steps):  val={vE:.4f}  diff A-E={vA-vE:+.4f}

  HESSIAN SIGN TELLS THE STORY:
    If lambda_min < 0 at valley entrance (after sign correction):
      Newton is INVALID there — Hessian not positive definite
      The 167 basin steps are needed to reach convex region
      Basin descent is irreducible
      
    If lambda_min > 0 at valley entrance:
      Newton IS valid — one step to basin floor
      The 167 steps were unnecessary redundancy
      Attractor targeting works: 0 basin steps → val~0.04

  MINIMUM BASIN STEPS:
    Find smallest N where val(N CE + Newton) ~ val(167 CE + Newton)
    This is the irreducible within-basin computation cost.
""")
