#!/usr/bin/env python3
"""
Extended LM — More Iterations to Close the Gap
================================================
LM 8 iters:  val=0.040  (3.6 CE equiv)
Target:      val=0.027  (100 CE gold standard)
Gap:         0.013 nats

Hypothesis: the gap is embedding relaxation.
Each LM iteration recomputes the gradient -> tracks embedding changes.
More iterations should close the embedding gap.

Also test: does increasing n_batches help?
  n_batches=15: current (fast)
  n_batches=30: 2x more accurate HVP
  n_batches=50: 3x more accurate (as suggested)
"""
import json, math, time, copy, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429

print(f"\n{'='*65}")
print(f"  EXTENDED LM — Close the gap to val=0.027")
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
    def set_flat(self,f):
        idx=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(f[idx:idx+n].reshape(p.shape)); idx+=n
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
        return float(np.mean([m(*get_batch())[1].item() for _ in range(n)]))

def clr(s,total,warmup=20):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def hvp(model,v,n_batches=15):
    params=list(model.parameters()); model.zero_grad()
    torch.manual_seed(42)
    loss=sum(model(*get_batch())[1] for _ in range(n_batches))/n_batches
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    Hv=torch.cat([h.flatten() for h in
                  torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad()
    return Hv

def cg_solve(model,g,mu,n_iters=6,n_batches=15):
    b=-g; delta=torch.zeros_like(b); r=b.clone(); p=r.clone()
    r_sq=float((r*r).sum())
    for k in range(n_iters):
        Hp=hvp(model,p,n_batches)+mu*p
        pHp=float((p*Hp).sum())
        if pHp<=0: break
        alpha=r_sq/pHp; delta=delta+alpha*p; r=r-alpha*Hp
        r_sq_new=float((r*r).sum()); beta=r_sq_new/max(r_sq,1e-10)
        p=r+beta*p; r_sq=r_sq_new
        if r_sq**0.5<1e-5: break
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

print("Building MF10+settle+sign state...")
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
print(f"Base state: val={eval_val(stu_base,n=20):.4f}\n")

results={}

# A: gold standard
print("[A] 100CE + Newton")
s=copy.deepcopy(stu_base)
opt2=torch.optim.AdamW(s.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,101):
    for pg in opt2.param_groups: pg['lr']=clr(step,100)
    s.train(); x,y=get_batch(); _,loss=s(x,y)
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt2.step()
    if step in [25,50,75,100]: print(f"  CE {step}: {eval_val(s,n=10):.4f}")
apply_newton_wk(s); results['A']=eval_val(s,n=30)
print(f"  A FINAL={results['A']:.4f}\n")

def run_lm(n_iters, n_batches_grad=30, n_batches_hvp=15, label="LM"):
    s=copy.deepcopy(stu_base)
    mu=1.02; L_prev=get_loss(s)
    print(f"  [{label}] initial L={L_prev:.4f}")
    ce_equiv=0.0
    for it in range(n_iters):
        # Gradient
        s.train(); s.zero_grad()
        for _ in range(n_batches_grad):
            x,y=get_batch(); _,loss=s(x,y); (loss/n_batches_grad).backward()
        g=s.flat_grad().detach().clone(); s.zero_grad()
        # CG solve
        delta=cg_solve(s,g,mu=mu,n_iters=6,n_batches=n_batches_hvp)
        ce_equiv+=n_batches_hvp*6*2/512  # 2 passes per HVP, 512 tokens per CE step
        # Try step
        w0=s.flat_params().clone(); s.set_flat(w0+delta)
        L_new=get_loss(s)
        if L_new<L_prev:
            L_prev=L_new; mu=max(mu*0.5,0.95)
            v_now=eval_val(s,n=8)
            print(f"    iter {it+1}: ACCEPT mu={mu:.3f} L={L_new:.4f} "
                  f"val={v_now:.4f} ||d||={float(delta.norm()):.4f}")
        else:
            s.set_flat(w0); mu=min(mu*2.0,5.0)
            print(f"    iter {it+1}: REJECT mu={mu:.3f}")
    apply_newton_wk(s)
    vf=eval_val(s,n=30)
    print(f"    FINAL={vf:.4f}  CE equiv≈{ce_equiv:.1f}")
    return vf, ce_equiv

# B: LM 8 iters (confirmed)
print("[B] LM 8 iters, n_hvp=15 (confirmed baseline)")
results['B'],ce_B=run_lm(8,n_batches_hvp=15,label="LM8-fast")

# C: LM 16 iters (more iterations)
print("\n[C] LM 16 iters, n_hvp=15 (more iterations)")
results['C'],ce_C=run_lm(16,n_batches_hvp=15,label="LM16-fast")

# D: LM 8 iters, n_hvp=30 (more accurate HVP)
print("\n[D] LM 8 iters, n_hvp=30 (2x accurate HVP)")
results['D'],ce_D=run_lm(8,n_batches_hvp=30,label="LM8-accurate")

# E: LM 8 iters, n_hvp=50 (as suggested)
print("\n[E] LM 8 iters, n_hvp=50 (3x accurate HVP)")
results['E'],ce_E=run_lm(8,n_batches_hvp=50,label="LM8-full")

print(f"""
{'='*65}
  EXTENDED LM RESULTS
{'='*65}
    Teacher:                  val={val_teacher:.4f}
    A (100CE+Newton):         val={results['A']:.4f}  [gold standard, 100 CE]
    B (LM 8 iters, hvp=15):   val={results['B']:.4f}  [{ce_B:.1f} CE equiv]
    C (LM 16 iters, hvp=15):  val={results['C']:.4f}  [{ce_C:.1f} CE equiv]
    D (LM 8 iters, hvp=30):   val={results['D']:.4f}  [{ce_D:.1f} CE equiv]
    E (LM 8 iters, hvp=50):   val={results['E']:.4f}  [{ce_E:.1f} CE equiv]

  DIAGNOSIS:
    IF C < B (more iters help): gap IS embedding relaxation
      More LM iterations accumulate more corpus observations
      for the embeddings (near-zero Hessian subspace)
      
    IF D~B and E~B (more batches don't help): HVP noise not limiting
      The gap is NOT from HVP inaccuracy
      Increasing n_batches is not the right fix
      
    IF D < B or E < B (more batches help): HVP noise was limiting
      The gap WAS from HVP inaccuracy
      Increasing n_batches to 50 closes the gap
      
  The answer tells us whether the residual gap (0.040->0.027)
  is from: (1) insufficient iterations, (2) HVP noise, or
  (3) fundamental embedding relaxation requiring gradient descent.
""")
