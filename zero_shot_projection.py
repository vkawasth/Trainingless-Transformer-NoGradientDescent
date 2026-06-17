#!/usr/bin/env python3
"""
Zero-Shot Projection: MF10 + Newton only, no CE steps.
"""
import json, math, warnings
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
        super().__init__()
        self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
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

def hv_product(model,v,n=15):
    params=list(model.parameters()); model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad(); return hv

def apply_newton_wk(stu, n_seq=500, eps=1e-3, scale=0.5):
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

def apply_newton_full(stu, n_seq=500, eps=1e-4, scale=0.1):
    """Full Newton on ALL parameters via diagonal Fisher."""
    ga={}; fd={}
    for name,p in stu.named_parameters():
        ga[name]=torch.zeros_like(p.data)
        fd[name]=torch.zeros_like(p.data)
    torch.manual_seed(3)
    for i in range(n_seq):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
        stu.zero_grad(); _,loss=stu(x,y); loss.backward()
        for name,p in stu.named_parameters():
            if p.grad is not None:
                ga[name]+=p.grad.detach()
                fd[name]+=p.grad.detach()**2
    with torch.no_grad():
        for name,p in stu.named_parameters():
            delta=-(ga[name]/n_seq)/((fd[name]/n_seq)+eps)
            p.data.add_(scale*delta)

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
for _ in range(15): Hv=hv_product(stu_ref,v,15); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg=v.clone(); print("v_neg ready.\n")

def apply_mf(stu, n_iter=10, mf_lr=0.01, n_corpus=200):
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

def run_settle_sign(stu):
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,34):
        for pg in opt_s.param_groups: pg['lr']=LR*5*min(step,10)/10
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
    with torch.no_grad():
        for l in [1,2]:
            stu.blocks[l].attn.WV.weight.mul_(-1)
            stu.blocks[l].attn.op.weight.mul_(-1)
    return eval_val(stu,n=20)

print("="*65)
print("ZERO-SHOT PROJECTION EXPERIMENTS")
print("="*65)

results={}

# A: Full confirmed pipeline (reference)
print("\n[A] Full pipeline (MF10+settle+sign+167CE+Newton)")
stu=build_student()
w0=stu.get_flat_params(); stu.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
apply_mf(stu,n_iter=10)
v_settle=run_settle_sign(stu)
print(f"  after settle+sign: {v_settle:.4f}")
opt2=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,168):
    for pg in opt2.param_groups: pg['lr']=clr(step,167)
    stu.train(); x,y=get_batch(); _,loss=stu(x,y)
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt2.step()
    if step in [50,100,167]: print(f"  basin {step}: {eval_val(stu,n=15):.4f}")
apply_newton_wk(stu)
results['A']=eval_val(stu,n=30); print(f"  FINAL={results['A']:.4f}")

# B: MF10 + settle + sign + Newton only (NO CE steps)
print("\n[B] MF10 + settle + sign + Newton (ZERO CE basin steps)")
stu=build_student()
w0=stu.get_flat_params(); stu.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
apply_mf(stu,n_iter=10)
v_settle=run_settle_sign(stu)
print(f"  after settle+sign: {v_settle:.4f}")
apply_newton_wk(stu)
v_wkn=eval_val(stu,n=20); print(f"  after WK Newton: {v_wkn:.4f}")
# Also try full Newton
apply_newton_full(stu,n_seq=500,eps=1e-4,scale=0.1)
v_fn=eval_val(stu,n=20); print(f"  after full Newton: {v_fn:.4f}")
results['B']=eval_val(stu,n=30); print(f"  FINAL={results['B']:.4f}")

# C: MF10 + settle + sign + 10CE + Newton
print("\n[C] MF10 + settle + sign + 10CE + Newton")
stu=build_student()
w0=stu.get_flat_params(); stu.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
apply_mf(stu,n_iter=10)
v_settle=run_settle_sign(stu)
print(f"  after settle+sign: {v_settle:.4f}")
opt2=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,11):
    for pg in opt2.param_groups: pg['lr']=clr(step,10)
    stu.train(); x,y=get_batch(); _,loss=stu(x,y)
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt2.step()
print(f"  after 10CE: {eval_val(stu,n=15):.4f}")
apply_newton_wk(stu)
results['C']=eval_val(stu,n=30); print(f"  FINAL={results['C']:.4f}")

# D: MF10 + settle + sign + 25CE + Newton
print("\n[D] MF10 + settle + sign + 25CE + Newton")
stu=build_student()
w0=stu.get_flat_params(); stu.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
apply_mf(stu,n_iter=10)
v_settle=run_settle_sign(stu)
opt2=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    for pg in opt2.param_groups: pg['lr']=clr(step,25)
    stu.train(); x,y=get_batch(); _,loss=stu(x,y)
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt2.step()
print(f"  after 25CE: {eval_val(stu,n=15):.4f}")
apply_newton_wk(stu)
results['D']=eval_val(stu,n=30); print(f"  FINAL={results['D']:.4f}")

print(f"""
{'='*65}
  ZERO-SHOT PROJECTION RESULTS
{'='*65}
    Teacher:              val={val_teacher:.4f}
    A (MF10+167CE+N):     val={results['A']:.4f}  [full pipeline]
    B (MF10+0CE+Newton):  val={results['B']:.4f}  [zero-shot target]
    C (MF10+10CE+Newton): val={results['C']:.4f}
    D (MF10+25CE+Newton): val={results['D']:.4f}

  IF B < 0.05: Zero-shot projection works.
    MF10 parametric pumping + Newton eliminates CE steps.
    
  IF B ~ settle_val (~0.15):
    Newton cannot correct from the settle position.
    The 167 CE steps integrate information Newton cannot reach.
    Statistical floor is real and irreducible.
""")
