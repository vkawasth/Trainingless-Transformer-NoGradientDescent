#!/usr/bin/env python3
"""
Levenberg-Marquardt Adaptive Damping
======================================
Previous CG used mu=1.0 >> lambda_min(H)=0.921.
Result: CG solved (H + I)delta = -G  ≈  delta = -G  (gradient descent).
This was an UNFAIR comparison. We were not testing Newton.

TRUE COMPARISON REQUIRES:
  mu chosen so that mu << ||H_signal|| but mu > |lambda_min|

lambda_min(H) = -0.921 at settle+sign position.
||H_signal|| ≈ Fisher lambda_1 ≈ 1.38-195 (varies with MF state).

For MF10 position: Fisher lambda_1 ~ 23.4 (from mf10_analysis).
So H_signal ~ 23.4, lambda_min ~ -0.921.
Safe mu range: [0.921, 23.4]
Optimal mu: just above 0.921 (say 1.0-2.0)... BUT
This still gives effective condition number kappa = (23.4+1.0)/(1.0-0.921) ~ 309.

Actually: mu must be > |lambda_min| to ensure PSD.
But the signal eigenvalues are ~ 1-23.
So mu in [0.921+eps, 1.0] makes the system barely PSD.
The issue is that H has a WIDE spectrum: lambda in [-0.921, 23.4].
Any fixed mu that ensures PSD will be comparable to signal eigenvalues.

LEVENBERG-MARQUARDT SOLUTION:
  Start with mu = |lambda_min| + 0.1 = 1.02 (minimum safe)
  After each CG step:
    If L decreases: mu *= 0.5 (more Newton-like)
    If L increases: mu *= 2.0 (more gradient-like)
  This adapts mu to the actual nonlinearity.

DIAGONAL PRECONDITIONER:
  Instead of mu*I, use M = diag(Fisher) + mu_diag*I
  Per-parameter scaling respects the embedding geometry.
  Fisher diagonal ~ P(t) * E[(g_t)^2] -- known from corpus.

HONEST EXPERIMENTS:
  A: 100CE + Newton (gold standard)
  B: LM-CG mu=1.02 (minimum safe, adaptive) 
  C: LM-CG mu=0.1 (aggressive, likely diverges -- honest test)
  D: LM-CG mu=0.01 (very aggressive -- measures true Newton)
  E: Diagonal preconditioned CG (Fisher-based)
  F: LM-CG adaptive (start mu=1.02, decay on success)
"""
import json, math, time, copy, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429

print(f"\n{'='*65}")
print(f"  LEVENBERG-MARQUARDT ADAPTIVE DAMPING")
print(f"  Honest comparison: mu chosen relative to spectrum")
print(f"  lambda_min(H) = -0.921  Fisher_lambda1 ~ 23.4")
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
        super().__init__(); self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
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
class LM_net(nn.Module):
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
    def flat_params(self): return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat(self,flat):
        idx=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(flat[idx:idx+n].reshape(p.shape)); idx+=n
    def flat_grad(self):
        return torch.cat([p.grad.flatten() if p.grad is not None
                         else torch.zeros(p.numel()) for p in self.parameters()])

def eval_val(m,n=40):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def get_loss(m,n=20):
    m.eval()
    with torch.no_grad():
        ls=[m(*get_batch())[1].item() for _ in range(n)]
    return float(np.mean(ls))

def clr(s,total,warmup=20):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def full_grad(model, n_batches=40):
    model.train(); model.zero_grad()
    for _ in range(n_batches):
        x,y=get_batch(); _,loss=model(x,y); (loss/n_batches).backward()
    g=model.flat_grad().detach().clone()
    model.zero_grad()
    return g

def hvp(model, v, n_batches=20):
    """Raw Hessian vector product via double backprop."""
    params=list(model.parameters()); model.zero_grad()
    torch.manual_seed(77)
    loss=sum(model(*get_batch())[1] for _ in range(n_batches))/n_batches
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    Hv=torch.cat([h.flatten() for h in
                  torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad()
    return Hv

def cg_solve(model, g, mu, n_iters=15, n_batches=15, tol=1e-5):
    """CG solve for (H + mu*I) delta = -g."""
    b=-g; delta=torch.zeros_like(b); r=b.clone(); p=r.clone()
    r_sq=float((r*r).sum())
    iters_used=0
    for k in range(n_iters):
        Hp=hvp(model,p,n_batches)+mu*p
        pHp=float((p*Hp).sum())
        if abs(pHp)<1e-12: break
        alpha=r_sq/pHp
        delta=delta+alpha*p; r=r-alpha*Hp
        r_sq_new=float((r*r).sum())
        beta=r_sq_new/max(r_sq,1e-10)
        p=r+beta*p; r_sq=r_sq_new; iters_used=k+1
        if r_sq**0.5<tol: break
    return delta, iters_used, r_sq**0.5

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

def hv_p(model,v,n=15):
    params=list(model.parameters()); model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad(); return hv

print("Training teacher...")
torch.manual_seed(42)
teacher=LM_net(D,N_HEADS,N_LAYERS_T)
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
    torch.manual_seed(99); stu=LM_net(D,N_HEADS,N_STU)
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

# Build base state
print("Building MF10+settle+sign...")
stu_ref=build_student(); n_p=sum(p.numel() for p in stu_ref.parameters())
v=torch.randn(n_p); v=v/v.norm()
for _ in range(15): Hv=hv_p(stu_ref,v,15); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg=v.clone()
stu_base=build_student()
w0=stu_base.flat_params(); stu_base.set_flat(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
apply_mf(stu_base,n_iter=10)
opt_s=torch.optim.AdamW(stu_base.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,34):
    for pg in opt_s.param_groups: pg['lr']=LR*5*min(step,10)/10
    stu_base.train(); x,y=get_batch(); _,loss=stu_base(x,y)
    opt_s.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu_base.parameters(),1.0); opt_s.step()
with torch.no_grad():
    for l in [1,2]:
        stu_base.blocks[l].attn.WV.weight.mul_(-1); stu_base.blocks[l].attn.op.weight.mul_(-1)
v_base=eval_val(stu_base,n=20)
print(f"Base val: {v_base:.4f}  L_train: {get_loss(stu_base):.4f}\n")

print("="*65)
print("HONEST COMPARISON")
print(f"  lambda_min(H) = -0.921  Fisher_lambda1 ~ 23.4")
print(f"  Safe mu range: (0.921, inf)")
print(f"  Signal range:  ~[1.0, 23.4]")
print(f"  mu=1.0: PSD but barely, delta ~ -G (gradient)")
print(f"  mu=0.1: INDEFINITE (0.1 < 0.921), CG will diverge")
print(f"  mu=0.01: INDEFINITE, CG will diverge badly")
print("="*65)

results={}

# A: 100 CE steps (gold standard)
s=copy.deepcopy(stu_base)
opt2=torch.optim.AdamW(s.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,101):
    for pg in opt2.param_groups: pg['lr']=clr(step,100)
    s.train(); x,y=get_batch(); _,loss=s(x,y)
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt2.step()
    if step in [25,50,75,100]: print(f"  A CE {step}: {eval_val(s,n=10):.4f}")
apply_newton_wk(s); results['A']=eval_val(s,n=30)
print(f"  A FINAL={results['A']:.4f}\n")

# B: mu=1.0 (previous, gradient descent in disguise)
print("B: mu=1.0 (barely PSD — was gradient descent)")
s=copy.deepcopy(stu_base); g=full_grad(s,n_batches=40)
print(f"  ||G||={float(g.norm()):.4f}")
delta,ni,res=cg_solve(s,g,mu=1.0,n_iters=8,n_batches=15)
print(f"  CG: {ni} iters, final ||r||={res:.5f}, ||delta||={float(delta.norm()):.4f}")
print(f"  Effective step = delta/||G||*||G|| = {float(delta.norm()):.4f}")
print(f"  Compare: gradient step size = 1/mu * ||G|| = {float(g.norm())/1.0:.4f}")
s.set_flat(s.flat_params()+delta); print(f"  After mu=1.0: {eval_val(s,n=15):.4f}")
apply_newton_wk(s); results['B']=eval_val(s,n=30); print(f"  B FINAL={results['B']:.4f}\n")

# C: mu=0.1 (indefinite, expected divergence — HONEST TEST)
print("C: mu=0.1 (INDEFINITE — honest test of divergence)")
s=copy.deepcopy(stu_base); g=full_grad(s,n_batches=40)
try:
    delta,ni,res=cg_solve(s,g,mu=0.1,n_iters=5,n_batches=15)
    print(f"  CG: {ni} iters, final ||r||={res:.5f}, ||delta||={float(delta.norm()):.4f}")
    w0=s.flat_params().clone()
    s.set_flat(w0+0.01*delta)  # tiny scale for safety
    v_c=eval_val(s,n=10); print(f"  After mu=0.1 (scale=0.01): {v_c:.4f}")
    if v_c>50: print("  DIVERGED as expected")
    results['C']=v_c
except Exception as e:
    print(f"  DIVERGED: {e}"); results['C']=float('inf')
print()

# D: LM adaptive (start mu=1.02, decay on success)
print("D: LM Adaptive (mu=1.02, decay 0.5 on success, grow 2.0 on failure)")
s=copy.deepcopy(stu_base)
mu=1.02  # just above |lambda_min|
L_prev=get_loss(s)
print(f"  Initial L={L_prev:.4f}")
for lm_iter in range(8):
    g=full_grad(s,n_batches=30)
    delta,ni,res=cg_solve(s,g,mu=mu,n_iters=6,n_batches=12)
    # Try the step
    w0=s.flat_params().clone()
    s.set_flat(w0+delta)
    L_new=get_loss(s)
    if L_new < L_prev:
        # Accept: decrease mu (more Newton-like)
        L_prev=L_new; mu=max(mu*0.5, 0.95)
        v_now=eval_val(s,n=8)
        print(f"  LM iter {lm_iter+1}: ACCEPT mu={mu:.3f} L={L_new:.4f} val={v_now:.4f} ||d||={float(delta.norm()):.4f}")
    else:
        # Reject: increase mu (more gradient-like)
        s.set_flat(w0); mu=min(mu*2.0, 10.0)
        print(f"  LM iter {lm_iter+1}: REJECT mu={mu:.3f} L={L_new:.4f} (L_prev={L_prev:.4f})")
apply_newton_wk(s); results['D']=eval_val(s,n=30); print(f"  D FINAL={results['D']:.4f}\n")

# E: LM adaptive + 25CE
print("E: LM Adaptive + 25CE cleanup")
s=copy.deepcopy(stu_base)
mu=1.02; L_prev=get_loss(s)
for lm_iter in range(5):
    g=full_grad(s,n_batches=30)
    delta,ni,res=cg_solve(s,g,mu=mu,n_iters=6,n_batches=12)
    w0=s.flat_params().clone(); s.set_flat(w0+delta)
    L_new=get_loss(s)
    if L_new<L_prev: L_prev=L_new; mu=max(mu*0.5,0.95)
    else: s.set_flat(w0); mu=min(mu*2.0,10.0)
print(f"  After LM: {eval_val(s,n=10):.4f}")
opt3=torch.optim.AdamW(s.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    for pg in opt3.param_groups: pg['lr']=clr(step,25)
    s.train(); x,y=get_batch(); _,loss=s(x,y)
    opt3.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt3.step()
    if step in [10,25]: print(f"  E CE {step}: {eval_val(s,n=10):.4f}")
apply_newton_wk(s); results['E']=eval_val(s,n=30); print(f"  E FINAL={results['E']:.4f}")

print(f"""
{'='*65}
  HONEST LM RESULTS
{'='*65}
    Teacher:                 val={val_teacher:.4f}
    A (100CE+Newton):        val={results['A']:.4f}  [gold standard]
    B (CG mu=1.0):           val={results['B']:.4f}  [was gradient descent]
    C (CG mu=0.1, indef):    val={results.get('C','n/a')}  [honest diverge test]
    D (LM adaptive):         val={results['D']:.4f}  [Newton with trust region]
    E (LM + 25CE):           val={results['E']:.4f}  [LM + minimal CE]

  SPECTRUM ANALYSIS:
    lambda_min(H):  -0.921
    Fisher_lambda1: ~23.4
    Safe mu range:  [0.921+eps, inf)
    
    mu=1.0: (H+I) has eigenvalues in [0.079, 24.4]
            kappa = 24.4/0.079 = 309  (poorly conditioned)
            delta ~ -G/1.0  (gradient step)
            
    mu=1.02 (LM minimum): marginally better
            The Hessian spectrum is what limits Newton here.
            
  CONCLUSION:
    IF D ~ A: LM adaptive reaches basin in ~8 steps
    IF D >> A: The indefinite Hessian IS the fundamental barrier.
               No Newton method works until the landscape convexifies.
               The 100 CE steps are truly irreducible.
""")
