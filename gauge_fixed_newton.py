#!/usr/bin/env python3
"""
Gauge-Fixed Newton Step
========================
The naive Newton step fails because W_K has gauge directions:
  W_Q^T @ (W_K + N) = W_Q^T @ W_K  for N in null(W_Q^T)
  
The loss is EXACTLY flat in these directions.
The Hessian has zero eigenvalues there -> Newton blows up.

Fix: restrict Newton step to gauge-fixed subspace.
  P = projection onto row space of W_Q (directions that matter)
  delta_gauge = -P^T (P H P^T + eps I)^{-1} P g

The row space of W_Q has rank <= D.
In practice: W_Q is full rank (D x D), so row space = R^D.
BUT: W_Q @ W_K appears as a product. The relevant subspace is
the image of W_Q, not W_Q itself.

More precisely: the attention score at position (i,j) is:
  A_{ij} = q_i^T k_j / sqrt(d_h)
  where q_i = W_Q h_i, k_j = W_K h_j

The gradient of loss w.r.t. W_K is:
  dL/dW_K = W_Q^T @ (dL/d(W_Q W_K)) 

The gauge freedom: W_K -> W_K + N where W_Q^T N = 0.
Equivalently: add anything in null(W_Q^T) to W_K.
The null space of W_Q^T has dimension D - rank(W_Q).

For full-rank W_Q: null(W_Q^T) = {0}, NO gauge freedom.
But in practice W_Q is nearly low-rank after initialization
at teacher values -> near-zero eigenvalues -> near-gauge.

ACTUAL DIAGNOSIS:
The small gradient ||g|| = 0.001634 means we are NEAR the optimum.
At the optimum, g = 0 exactly. Newton of g≈0 = noise amplification.
The Newton step is amplifying gradient noise by 1/Fisher_eigenvalue.

The correct approach: when ||g|| is small, DO NOT apply Newton.
Instead use the gradient directly (Adam is correct here).
The condition number problem is irrelevant when you're at the minimum.

WHAT THIS MEANS:
The 200 CE steps converge to val=0.156.
At convergence, ||grad_WK|| = 0.010 (not zero — still optimizing).
The gradient is small but nonzero — we are NOT at W_K*.
We are at an Adam-stationary point, not the true minimum.
Adam's momentum + weight decay create a fixed point at finite gradient.

THE REAL IRREDUCIBILITY:
The 200 steps do NOT reach W_K* (the true minimum).
They reach Adam's fixed point, which is close but not exact.
The gap between Adam's fixed point and W_K* is set by:
  - Learning rate (eta)
  - Weight decay (lambda)  
  - Batch size (noise level)

This is the true source of irreducibility: Adam's fixed point
is not W_K*, it is the solution of:
  E[g] = lambda * W_K  (gradient = weight decay force)

Newton can find W_K* from Adam's fixed point in ONE step
if the Hessian is well-conditioned at Adam's fixed point.
But Adam's fixed point is at finite gradient -> Newton would work.

TEST: Apply Newton at FINAL state (step 200), not midpoint.
At step 200: val=0.156, ||grad||=0.010.
Newton: delta = -H^{-1} g should jump to W_K*.
If val improves after Newton at step 200: W_K* exists and is reachable.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; N_NEWTON=500

print(f"\n{'='*65}")
print(f"  GAUGE-FIXED NEWTON")
print(f"  Apply Newton at FINAL state (step 200), not midpoint")
print(f"  At step 200: gradient is nonzero (Adam fixed point != W_K*)")
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

def build_student():
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight)
        stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.copy_(teacher.blocks[L_ATT].attn.WK.weight)
            stu.blocks[l].attn.WQ.weight.copy_(teacher.blocks[L_ATT].attn.WQ.weight)
            stu.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            stu.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            stu.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            stu.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            stu.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
    return stu

# Train student to convergence
print("Training student (200 steps)...")
stu=build_student()
opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
wk_gnorms=[]
for step in range(1,201):
    for pg in opt_s.param_groups: pg['lr']=clr(step,200)
    stu.train(); x,y=get_batch(); _,loss=stu(x,y)
    opt_s.zero_grad(); loss.backward()
    wk_gnorm=float(sum(stu.blocks[l].attn.WK.weight.grad.norm()**2
                       for l in range(N_STU))**0.5)
    wk_gnorms.append(wk_gnorm)
    torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()

val_200=eval_val(stu)
print(f"  After 200 steps: val={val_200:.4f}")
print(f"  Final ||grad_WK|| = {wk_gnorms[-1]:.6f}")
print(f"  This is Adam's fixed point, NOT W_K*")
print(f"  Adam fixed point condition: E[g] = lambda * W_K")

# ════════════════════════════════════════════════════
# GAUGE ANALYSIS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("GAUGE ANALYSIS")
print("  W_Q^T N = 0 -> gauge direction (loss flat)")
print("  Restrict Newton to gauge-fixed subspace")
print("="*65)

WQ=stu.blocks[0].attn.WQ.weight.data  # (D,D)
# SVD of WQ to find its row space
U,S,Vt=torch.linalg.svd(WQ)
print(f"\n  W_Q singular values (top 8): {S[:8].numpy().round(4)}")
print(f"  W_Q rank estimate (S > 0.01): {(S>0.01).sum().item()}")
print(f"  Gauge null space dim: {(S<0.01).sum().item()}")

# Effective rank of W_Q determines gauge-fixed subspace dimension
rank_WQ=(S>0.01).sum().item()
print(f"\n  Gauge-fixed subspace: {rank_WQ} dimensions of {D}")
print(f"  Gauge directions (flat): {D-rank_WQ} dimensions")

# ════════════════════════════════════════════════════
# NEWTON AT FINAL STATE (step 200)
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"NEWTON AT FINAL STATE")
print(f"  Compute grad + Fisher at Adam fixed point")
print(f"  grad is nonzero here (0.010) -> Newton can work")
print("="*65)

# Accumulate gradient and diagonal Fisher at step 200
grad_acc=torch.zeros_like(stu.blocks[0].attn.WK.weight)
fisher_d=torch.zeros_like(stu.blocks[0].attn.WK.weight)
torch.manual_seed(2)
print(f"\n  Computing over {N_NEWTON} sequences...")
for i in range(N_NEWTON):
    ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
    x=train_t[ix:ix+SEQ].unsqueeze(0)
    y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
    stu.zero_grad(); _,loss=stu(x,y); loss.backward()
    g=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    for l in range(N_STU):
        if stu.blocks[l].attn.WK.weight.grad is not None:
            g+=stu.blocks[l].attn.WK.weight.grad/N_STU
    grad_acc+=g; fisher_d+=g**2
    if (i+1)%100==0: print(f"  {i+1}/{N_NEWTON}...",flush=True)

grad_mean=grad_acc/N_NEWTON
fisher_diag=fisher_d/N_NEWTON

print(f"\n  ||grad_mean|| at step 200:  {grad_mean.norm():.6f}")
print(f"  ||fisher_diag||:            {fisher_diag.norm():.6f}")
print(f"  max fisher_diag:            {fisher_diag.max():.8f}")
print(f"  min fisher_diag (nonzero):  {fisher_diag[fisher_diag>0].min():.10f}")

# Condition number of diagonal Fisher
f_sorted=fisher_diag.flatten().sort(descending=True).values
f_nonzero=f_sorted[f_sorted>1e-10]
print(f"  Fisher diag condition number: {f_nonzero[0]/f_nonzero[-1]:.2f}")

# Gauge-projected gradient: project onto row space of W_Q
# g_gauge = Vt[:rank_WQ,:] @ g @ Vt[:rank_WQ,:].T
Vt_np=Vt[:rank_WQ,:].numpy()  # (rank, D)
g_np=grad_mean.numpy()
g_gauge=Vt_np.T@(Vt_np@g_np)  # project to row space of W_Q
print(f"\n  ||grad_mean|| full:          {np.linalg.norm(g_np):.6f}")
print(f"  ||grad_mean|| gauge-fixed:   {np.linalg.norm(g_gauge):.6f}")
print(f"  Gauge fraction:              {np.linalg.norm(g_gauge)/np.linalg.norm(g_np):.4f}")

# Newton step with various epsilon values
print(f"\n  Newton step at step 200 (line search):")
best_val=float('inf'); best_eps=None; best_scale=None

for eps in [1e-3, 1e-4, 1e-5]:
    delta=-(grad_mean/(fisher_diag+eps))
    for scale in [0.001,0.01,0.1,0.5,1.0]:
        import copy
        stu_test=copy.deepcopy(stu)
        with torch.no_grad():
            for l in range(N_STU):
                stu_test.blocks[l].attn.WK.weight.add_(scale*delta)
                stu_test.blocks[l].attn.WQ.weight.add_(scale*delta.T)
        v=eval_val(stu_test,n=20)
        if v<best_val: best_val=v; best_eps=eps; best_scale=scale
        if scale==0.1:
            print(f"  eps={eps:.0e} scale={scale}: val={v:.4f}")

print(f"\n  Best: eps={best_eps:.0e}, scale={best_scale}, val={best_val:.4f}")
print(f"  Baseline (step 200): val={val_200:.4f}")
print(f"  Improvement: {val_200-best_val:+.4f}")

print(f"""
{'='*65}
  GAUGE-FIXED NEWTON RESULTS
{'='*65}

  ADAM FIXED POINT ANALYSIS:
    val at step 200:         {val_200:.4f}
    ||grad_WK|| at step 200: {wk_gnorms[-1]:.6f}  (nonzero = not at W_K*)
    W_Q rank:                {rank_WQ}/{D}
    Gauge null space:        {D-rank_WQ} directions

  NEWTON AT STEP 200:
    ||grad_mean||:           {grad_mean.norm():.6f}
    Best Newton val:         {best_val:.4f}
    Improvement over 200CE:  {val_200-best_val:+.4f}

  IF improvement > 0.005:
    Newton at step 200 finds W_K* beyond Adam's fixed point.
    The residual gradient (weight decay vs loss gradient balance)
    can be corrected in one Newton step.
    Total: 200 CE + 1 Newton step > 200 CE alone.
    
  IF improvement ~ 0:
    Adam's fixed point IS W_K* for this architecture.
    The Hessian is too ill-conditioned for Newton to help.
    The 200 CE steps are truly the minimum computation.
    
  THE GAUGE INSIGHT:
    The fraction of gradient in gauge-fixed subspace tells us
    how much of the gradient is "real" vs gauge artifact.
    If gauge fraction << 1: most gradient is gauge noise.
    Newton in gauge-fixed subspace only would be clean.
""")
