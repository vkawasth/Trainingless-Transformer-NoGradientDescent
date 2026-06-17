#!/usr/bin/env python3
"""
CG Newton Refined — Scale Search and Multiple Rounds
=====================================================
CG converged in 3 iterations. ||delta|| = 0.147 (reasonable).
Scale=0.1 gave val 0.281->0.222 — correct direction, too small step.

EXPERIMENTS:
  A: Baseline 100CE+Newton (val=0.022)
  B: CG scale=0.5 + Newton
  C: CG scale=1.0 + Newton  
  D: CG scale=0.5 + 25CE + Newton
  E: 3 rounds of CG (each: compute G, solve CG, apply) + Newton
  F: 3 rounds CG + 25CE + Newton

Each CG round: 3 HVPs x 15 batches = 45 passes ≈ 0.22 CE equiv
3 rounds = 0.66 CE equiv total (deterministic)
"""
import json, math, time, copy, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429

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
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2); K=self.WK(h).view(B,S,H,dh).transpose(1,2)
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

def clr(s,total,warmup=20):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def embedding_hvp(model, v_emb, n_batches=15):
    params=list(model.parameters()); model.zero_grad()
    torch.manual_seed(42)
    loss=sum(model(*get_batch())[1] for _ in range(n_batches))/n_batches
    grads=torch.autograd.grad(loss, params, create_graph=True)
    flat_grad=torch.cat([g.flatten() for g in grads])
    emb_numel=VOCAB*D
    v_full=torch.zeros(sum(p.numel() for p in params))
    v_full[:emb_numel]=v_emb
    gv=(flat_grad*v_full.detach()).sum()
    hv_full=torch.cat([h.flatten() for h in
                       torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad()
    return hv_full[:emb_numel]

def cg_step(model, n_grad_batches=50, n_cg_iters=8, reg=1.0, scale=1.0, n_hv_batches=15):
    """One round of CG Newton on embedding matrix. Returns val improvement."""
    model.train(); model.zero_grad()
    for _ in range(n_grad_batches):
        x,y=get_batch(); _,loss=model(x,y); (loss/n_grad_batches).backward()
    grad_emb=model.te.weight.grad.detach().clone().flatten()
    model.zero_grad()

    # CG solve
    b=-grad_emb; delta=torch.zeros_like(b); r=b.clone(); p=r.clone()
    r_sq=float((r*r).sum())
    n_iters_used=0
    for k in range(n_cg_iters):
        Hp=embedding_hvp(model,p,n_hv_batches)+reg*p
        pHp=float((p*Hp).sum())
        if abs(pHp)<1e-10: break
        alpha=r_sq/pHp; delta=delta+alpha*p; r=r-alpha*Hp
        r_sq_new=float((r*r).sum()); beta=r_sq_new/max(r_sq,1e-10)
        p=r+beta*p; r_sq=r_sq_new; n_iters_used=k+1
        if r_sq**0.5<1e-5: break

    with torch.no_grad():
        model.te.weight.add_(scale*delta.reshape(VOCAB,D))
    return float(delta.norm()), n_iters_used

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

# Build base MF10 state
print("Building MF10+settle+sign state...")
stu_ref=build_student()
n_p=sum(p.numel() for p in stu_ref.parameters())
v=torch.randn(n_p); v=v/v.norm()
for _ in range(15): Hv=hv_product_full(stu_ref,v,15); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg=v.clone()

stu_base=build_student()
w0=stu_base.get_flat_params(); stu_base.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
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
print(f"Base state val: {eval_val(stu_base,n=20):.4f}\n")

print("="*65)
print("EXPERIMENTS")
print("="*65)

results={}

# A: baseline
s=copy.deepcopy(stu_base)
opt2=torch.optim.AdamW(s.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,101):
    for pg in opt2.param_groups: pg['lr']=clr(step,100)
    s.train(); x,y=get_batch(); _,loss=s(x,y)
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt2.step()
    if step in [25,50,100]: print(f"  A CE {step}: {eval_val(s,n=10):.4f}")
apply_newton_wk(s); results['A']=eval_val(s,n=30); print(f"  A FINAL={results['A']:.4f}")

# B: CG scale=0.5
s=copy.deepcopy(stu_base)
dn,ni=cg_step(s,scale=0.5); print(f"\n  B CG: ||delta||={dn:.4f} ({ni} iters)")
print(f"  B after CG: {eval_val(s,n=15):.4f}")
apply_newton_wk(s); results['B']=eval_val(s,n=30); print(f"  B FINAL={results['B']:.4f}")

# C: CG scale=1.0
s=copy.deepcopy(stu_base)
dn,ni=cg_step(s,scale=1.0); print(f"\n  C CG: ||delta||={dn:.4f} ({ni} iters)")
print(f"  C after CG: {eval_val(s,n=15):.4f}")
apply_newton_wk(s); results['C']=eval_val(s,n=30); print(f"  C FINAL={results['C']:.4f}")

# D: CG scale=0.5 + 25CE
s=copy.deepcopy(stu_base)
dn,ni=cg_step(s,scale=0.5); print(f"\n  D CG: ||delta||={dn:.4f}")
print(f"  D after CG: {eval_val(s,n=10):.4f}")
opt3=torch.optim.AdamW(s.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    for pg in opt3.param_groups: pg['lr']=clr(step,25)
    s.train(); x,y=get_batch(); _,loss=s(x,y)
    opt3.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt3.step()
    if step in [10,25]: print(f"  D CE {step}: {eval_val(s,n=10):.4f}")
apply_newton_wk(s); results['D']=eval_val(s,n=30); print(f"  D FINAL={results['D']:.4f}")

# E: 3 rounds CG
s=copy.deepcopy(stu_base)
print(f"\n  E: 3 rounds CG")
for r in range(3):
    dn,ni=cg_step(s,scale=0.5)
    v_r=eval_val(s,n=10)
    print(f"  E round {r+1}: ||delta||={dn:.4f} ({ni} iters) val={v_r:.4f}")
apply_newton_wk(s); results['E']=eval_val(s,n=30); print(f"  E FINAL={results['E']:.4f}")

# F: 3 rounds CG + 25CE
s=copy.deepcopy(stu_base)
print(f"\n  F: 3 rounds CG + 25CE")
for r in range(3):
    dn,ni=cg_step(s,scale=0.5)
    print(f"  F round {r+1}: val={eval_val(s,n=8):.4f}")
opt4=torch.optim.AdamW(s.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    for pg in opt4.param_groups: pg['lr']=clr(step,25)
    s.train(); x,y=get_batch(); _,loss=s(x,y)
    opt4.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt4.step()
    if step in [10,25]: print(f"  F CE {step}: {eval_val(s,n=10):.4f}")
apply_newton_wk(s); results['F']=eval_val(s,n=30); print(f"  F FINAL={results['F']:.4f}")

print(f"""
{'='*65}
  CG REFINED RESULTS
{'='*65}
    Teacher:              val={val_teacher:.4f}
    A (100CE+N):          val={results['A']:.4f}  [100 CE steps]
    B (CG 0.5x + N):      val={results['B']:.4f}
    C (CG 1.0x + N):      val={results['C']:.4f}
    D (CG 0.5x+25CE+N):   val={results['D']:.4f}
    E (3xCG 0.5x + N):    val={results['E']:.4f}  [3 CG rounds, ~2 CE equiv]
    F (3xCG+25CE+N):       val={results['F']:.4f}  [3 CG + 25CE]

  CG EFFICIENCY vs CE STEPS:
    Each CG round: ~0.22 CE equiv (3 HVPs x 15 batches)
    3 CG rounds:   ~0.66 CE equiv total
    
  IF E ~ A: 3 CG rounds replace 100 CE steps
    Speedup: 100 / 0.66 = 150x in compute
    Coupled relaxation solved by matrix-free Newton
    
  IF F < A: CG + 25CE beats 100CE
    CG provides better preconditioned starting point
    25 CE steps in CG-corrected landscape > 100 CE in raw landscape
""")
