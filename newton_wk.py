#!/usr/bin/env python3
"""
Newton Step on WK — One-Shot Corpus Integration
=================================================
GD takes 200 steps because the multiplier near W_K* is close to 1
(flat loss surface). Newton's method finds W_K* in one step.

NEWTON STEP:
  W_K* = W_K_teacher - H^{-1} @ grad E_D[L]
  
  where H = E_D[grad L @ grad L^T]  (Fisher = Hessian approx)
        grad E_D[L] = E_D[grad L]    (expected gradient over corpus)

Both computable in ONE forward+backward pass over the corpus.
No iteration. No oscillation. No 200 steps.

The corpus enters ONCE to compute:
  1. grad E_D[L]: mean gradient of WK over corpus
  2. H = Fisher: covariance of WK gradients over corpus

Then W_K* = W_K_teacher - H^{-1} @ grad E_D[L]

This is the one-shot computation that replaces 200 CE steps.

EXPERIMENTS:
  A: Teacher WK + 200CE (best known, val=0.156)
  B: Teacher WK + Newton step + 0CE  (one-shot, no training)
  C: Teacher WK + Newton step + 10CE (minimal refinement)
  D: Teacher WK + Newton step + 50CE
  E: Teacher WK + Newton step + 200CE

If B zero-shot ~ 0.156: Newton step replaces all 200 CE steps.
If C (10CE) ~ A (200CE): Newton step + 10 steps = same quality.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; N_CORPUS=200  # sequences for Newton step computation

print(f"\n{'='*65}")
print(f"  NEWTON STEP ON WK")
print(f"  One-shot corpus integration via Newton's method")
print(f"  W_K* = W_K_teacher - H^{{-1}} @ grad E_D[L]")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t=torch.tensor(val_ids,dtype=torch.long)

def get_batch(split='train'):
    data=train_t if split=='train' else val_t
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

def clr(s,total=200,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

# ════════════════════════════════════════════════════
# Train teacher
# ════════════════════════════════════════════════════
print("Training teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    lr_now=LR*min(step,100)/100 if step<=100 else \
           LR*0.5*(1+math.cos(math.pi*(step-100)/200))
    for pg in opt.param_groups: pg['lr']=lr_now
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
    if step%100==0:
        teacher.eval()
        with torch.no_grad():
            vl=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
        teacher.train()
teacher.eval(); val_teacher=eval_val(teacher)
print(f"  Teacher val={val_teacher:.4f}\n")

# ════════════════════════════════════════════════════
# BUILD STUDENT WITH TEACHER WK
# ════════════════════════════════════════════════════
def build_student_teacher_wk(newton_delta=None):
    """Student initialized with teacher WK at L14.
    If newton_delta provided, apply Newton correction to WK."""
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight)
        stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            WK=teacher.blocks[L_ATT].attn.WK.weight.data.clone()
            WQ=teacher.blocks[L_ATT].attn.WQ.weight.data.clone()
            if newton_delta is not None:
                WK=WK+newton_delta
                WQ=WQ+newton_delta.T
            stu.blocks[l].attn.WK.weight.copy_(WK)
            stu.blocks[l].attn.WQ.weight.copy_(WQ)
            stu.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            stu.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            stu.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            stu.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            stu.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
    return stu

# ════════════════════════════════════════════════════
# NEWTON STEP COMPUTATION
# ════════════════════════════════════════════════════
print("="*65)
print(f"NEWTON STEP COMPUTATION ({N_CORPUS} corpus sequences)")
print("  grad_mean = E_D[grad_{WK} L]")
print("  Fisher    = E_D[grad_{WK} L @ grad_{WK} L^T]  (diagonal)")
print("  delta_WK  = -Fisher^{-1} @ grad_mean")
print("="*65)

# Build student at teacher WK starting point
stu_newton=build_student_teacher_wk()

# Accumulate gradient and diagonal Fisher over corpus
grad_acc=torch.zeros_like(stu_newton.blocks[0].attn.WK.weight)
fisher_diag=torch.zeros_like(stu_newton.blocks[0].attn.WK.weight)
n_samples=0

print(f"\n  Computing gradient and Fisher over {N_CORPUS} sequences...")
torch.manual_seed(0)
for i in range(N_CORPUS):
    ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
    x=train_t[ix:ix+SEQ].unsqueeze(0)
    y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)

    stu_newton.zero_grad()
    _,loss=stu_newton(x,y)
    loss.backward()

    # Accumulate gradient of WK (average over all blocks — they share same init)
    g=stu_newton.blocks[0].attn.WK.weight.grad.detach()
    grad_acc+=g
    fisher_diag+=g**2  # diagonal Fisher = E[g^2]
    n_samples+=1

    if (i+1)%50==0: print(f"  {i+1}/{N_CORPUS}...",flush=True)

grad_mean=grad_acc/n_samples        # E[grad L]
fisher_d =fisher_diag/n_samples    # E[grad L ^2]  (diagonal Fisher)

# Newton step: delta = -Fisher^{-1} @ grad_mean
# With diagonal Fisher: delta_ij = -grad_mean_ij / fisher_d_ij
eps_newton=1e-4  # regularization
fisher_inv_diag=1.0/(fisher_d+eps_newton)
newton_delta=-fisher_inv_diag*grad_mean

print(f"\n  ||grad_mean||:    {grad_mean.norm():.6f}")
print(f"  ||fisher_diag||:  {fisher_d.norm():.6f}")
print(f"  ||newton_delta||: {newton_delta.norm():.6f}")
print(f"  max |newton_delta|: {newton_delta.abs().max():.6f}")

# Scale newton delta — the raw Newton step may be too large
# Test multiple step sizes
print(f"\n  Newton step size search (zero-shot val):")
best_scale=1.0; best_val=float('inf')
for scale in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
    stu_test=build_student_teacher_wk(newton_delta=scale*newton_delta)
    v=eval_val(stu_test,n=20)
    print(f"  scale={scale:.3f}: val={v:.4f}")
    if v<best_val: best_val=v; best_scale=scale

print(f"\n  Best Newton scale: {best_scale}, val={best_val:.4f}")

# ════════════════════════════════════════════════════
# EXPERIMENTS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("EXPERIMENTS")
print("  A: Teacher WK + 200CE (best known baseline)")
print("  B: Teacher WK + Newton step + 0CE  (one-shot target)")
print(f"  C: Teacher WK + Newton (scale={best_scale}) + 10CE")
print(f"  D: Teacher WK + Newton (scale={best_scale}) + 50CE")
print(f"  E: Teacher WK + Newton (scale={best_scale}) + 200CE")
print("="*65)

def run(build_fn,label,steps=200):
    stu=build_fn()
    v0=eval_val(stu,n=30)
    print(f"\n  [{label}] zero-shot={v0:.4f}")
    if steps==0: return v0,{0:v0}
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={0:v0}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [10,25,50,100,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

vA,ckA=run(build_student_teacher_wk,"A-TeachWK+200CE")
vB,ckB=run(lambda: build_student_teacher_wk(best_scale*newton_delta),
           "B-Newton+0CE",steps=0)
vC,ckC=run(lambda: build_student_teacher_wk(best_scale*newton_delta),
           f"C-Newton+10CE",steps=10)
vD,ckD=run(lambda: build_student_teacher_wk(best_scale*newton_delta),
           f"D-Newton+50CE",steps=50)
vE,ckE=run(lambda: build_student_teacher_wk(best_scale*newton_delta),
           f"E-Newton+200CE",steps=200)

print(f"\n{'='*65}")
print("  NEWTON STEP RESULTS")
print("="*65)
print(f"""
  NEWTON COMPUTATION:
    Corpus sequences: {N_CORPUS}
    ||grad_mean||:    {grad_mean.norm():.6f}
    ||newton_delta||: {(best_scale*newton_delta).norm():.6f}
    Best scale: {best_scale}

  ZERO-SHOT:
    A (teacher WK, no Newton): {ckA[0]:.4f}
    B (teacher WK + Newton):   {vB:.4f}
    Teacher:                   {val_teacher:.4f}

  CONVERGENCE:
  {'step':>6}  {'A-base':>7}  {'C-Nwt10':>8}  {'D-Nwt50':>8}  {'E-Nwt200':>9}""")
for s in [0,10,25,50,100,150,200]:
    a=ckA.get(s); c=ckC.get(s); d=ckD.get(s); e=ckE.get(s)
    row=f"  {s:>6}"
    for v in [a,c,d,e]:
        row+=f"  {v:>8.4f}" if v else f"  {'---':>8}"
    print(row)

print(f"""
  FINAL:
    Teacher:                val={val_teacher:.4f}
    A (TeachWK+200CE):      val={vA:.4f}  (baseline)
    B (Newton, 0CE):        val={vB:.4f}
    C (Newton+10CE):        val={vC:.4f}  diff A-C={vA-vC:+.4f}
    D (Newton+50CE):        val={vD:.4f}  diff A-D={vA-vD:+.4f}
    E (Newton+200CE):       val={vE:.4f}  diff A-E={vA-vE:+.4f}

  THE MORPHISM ANSWER:
    IF B ~ 0.156: Newton finds W_K* in one shot.
      The 200 oscillating steps are replaced by:
        1. One corpus pass (compute grad + Fisher)
        2. One matrix inversion (diagonal, trivial)
        3. One weight update (delta = -Fisher^{{-1}} grad)
      The morphism IS the Newton step on the corpus Fisher.

    IF C (10CE) ~ A (200CE): Newton + 10 steps = 200 steps.
      The Newton step captures 190 steps worth of information.
      Only 10 refinement steps needed after Newton correction.

    IF E ~ A but B >> A: Newton improves final quality but
      still needs CE steps to converge.
      The oscillation IS necessary but Newton reduces its count.
""")
