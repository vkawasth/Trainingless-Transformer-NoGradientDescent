#!/usr/bin/env python3
"""
MINRES Step — Minimum Residual Direction
==========================================
Standard Newton minimizes L(theta + delta).
This minimizes ||g(theta + delta)||^2 — the gradient norm.

The gradient of ||g||^2 w.r.t. delta is:
  d/d(delta) ||g(theta+delta)||^2 = 2 H^T g(theta+delta)
  At delta=0: = 2 H g(theta)

So the MINRES direction is -H g = -(H)(gradient).
This is a single HVP: one forward+backward pass.

WHY THIS FIXES ROTATION:
  Newton step: delta_N = -H^{-1} g  (moves to where g=0 by quadratic approx)
  MINRES step: delta_MR = -H g      (moves in direction that reduces ||g||)
  
  Newton assumes quadratic landscape (no rotation).
  MINRES accounts for the fact that g rotates — it steps in the direction
  that most reduces the gradient norm at the next point.
  
  For the gradient rotation problem:
    g(theta) points at 74° from basin axis (alignment=0.28)
    H g(theta) points in the direction that rotates g toward the basin axis
    -H g(theta) is the anti-rotation direction
    
  After one MINRES step: g(theta + delta_MR) is more aligned with basin axis.
  This is the pre-compensation for gradient rotation.

RELATIONSHIP TO THIRD DERIVATIVE:
  g(theta + delta) ≈ g + H delta + (1/2) T[delta,delta]
  MINRES minimizes this by choosing delta = -H^{-1} g + correction
  The MINRES direction at first order IS -H g (not -H^{-1} g).
  
COST: one HVP (same as one CG iteration in the Tikhonov CG)
  = ~20 forward+backward passes = ~0.1 CE equiv

EXPERIMENTS:
  A: 100CE + Newton (gold standard)
  B: MINRES step + Newton (0 CE, ~0.1 CE equiv)
  C: MINRES step + 25CE + Newton
  D: 10CE + MINRES step + 90CE + Newton
  E: 25CE + MINRES step + 75CE + Newton (optimal t*)
"""
import json, math, time, copy, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429

print(f"\n{'='*65}")
print(f"  MINRES STEP — MINIMUM RESIDUAL DIRECTION")
print(f"  delta = -H * g  (not -H^{{-1}} g)")
print(f"  Pre-compensates for gradient rotation")
print(f"  Cost: 1 HVP = ~0.1 CE equiv")
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

def full_grad(m,n=30):
    m.train(); m.zero_grad()
    for _ in range(n): x,y=get_batch(); _,l=m(x,y); (l/n).backward()
    g=m.flat_grad().detach().clone(); m.zero_grad()
    return g

def hvp(model,v,n=20):
    """Exact Pearlmutter HVP: H*v = grad(grad_L . v)"""
    params=list(model.parameters()); model.zero_grad()
    torch.manual_seed(42)
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    Hv=torch.cat([h.flatten() for h in
                  torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad()
    return Hv

def clr(s,total,warmup=20):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

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

def minres_step(model, scale=1.0, n_grad=30, n_hvp=20):
    """
    MINRES step: delta = -scale * H * g / ||H*g||
    
    1. Compute g = gradient
    2. Compute Hg = H * g  (one HVP)
    3. Step: theta -= scale * Hg / ||Hg||
    
    This steps in the direction that most reduces ||g||.
    Pre-compensates for gradient rotation.
    
    Note: H has negative eigenvalues (lambda_min=-0.921).
    H*g may point in a direction where L increases.
    Use line search to ensure L decreases.
    """
    g = full_grad(model, n=n_grad)
    gnorm = float(g.norm())
    
    # Compute H*g
    Hg = hvp(model, g, n=n_hvp)
    Hgnorm = float(Hg.norm())
    
    # The MINRES direction: -Hg (normalized)
    # Check: does -Hg point downhill? (g . Hg < 0 means Hg points uphill)
    gHg = float((g * Hg).sum())
    
    print(f"  ||g||={gnorm:.4f}  ||Hg||={Hgnorm:.4f}")
    print(f"  g.Hg={gHg:.4f}  "
          f"({'Hg uphill: -Hg is downhill' if gHg>0 else 'Hg downhill: -Hg is uphill'})")
    
    # Apply step with line search
    w0 = model.flat_params().clone()
    
    # Try scale values
    best_val = eval_val(model, n=15)
    best_scale = 0.0
    
    direction = -Hg / max(Hgnorm, 1e-10)  # normalized MINRES direction
    
    for s in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]:
        model.set_flat(w0 + s * direction)
        v = eval_val(model, n=10)
        print(f"  scale={s:.3f}: val={v:.4f}")
        if v < best_val:
            best_val = v
            best_scale = s
    
    model.set_flat(w0 + best_scale * direction)
    print(f"  Best scale={best_scale}: val={best_val:.4f}")
    return best_val, direction

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
print("="*65)

# A: baseline
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

# B: MINRES + Newton (0 CE)
print("[B] MINRES step + Newton (0 CE)")
s=copy.deepcopy(stu_base)
v_before=eval_val(s,n=15)
print(f"  Before: {v_before:.4f}")
best_v,direction=minres_step(s,n_grad=30,n_hvp=20)
apply_newton_wk(s); results['B']=eval_val(s,n=30)
print(f"  B FINAL={results['B']:.4f}\n")

# C: MINRES + 25CE + Newton
print("[C] MINRES step + 25CE + Newton")
s=copy.deepcopy(stu_base)
minres_step(s,n_grad=30,n_hvp=20)
opt3=torch.optim.AdamW(s.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    for pg in opt3.param_groups: pg['lr']=clr(step,25)
    s.train(); x,y=get_batch(); _,loss=s(x,y)
    opt3.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt3.step()
    if step in [10,25]: print(f"  CE {step}: {eval_val(s,n=10):.4f}")
apply_newton_wk(s); results['C']=eval_val(s,n=30)
print(f"  C FINAL={results['C']:.4f}\n")

# D: 25CE + MINRES + 75CE + Newton
print("[D] 25CE + MINRES step + 75CE + Newton")
s=copy.deepcopy(stu_base)
opt4=torch.optim.AdamW(s.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    for pg in opt4.param_groups: pg['lr']=clr(step,100)
    s.train(); x,y=get_batch(); _,loss=s(x,y)
    opt4.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt4.step()
print(f"  After 25CE: {eval_val(s,n=10):.4f}")
minres_step(s,n_grad=30,n_hvp=20)
opt5=torch.optim.AdamW(s.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,76):
    for pg in opt5.param_groups: pg['lr']=clr(step,75)
    s.train(); x,y=get_batch(); _,loss=s(x,y)
    opt5.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt5.step()
    if step in [25,50,75]: print(f"  post-CE {step}: {eval_val(s,n=10):.4f}")
apply_newton_wk(s); results['D']=eval_val(s,n=30)
print(f"  D FINAL={results['D']:.4f}")

print(f"""
{'='*65}
  MINRES RESULTS
{'='*65}
    Teacher:              val={val_teacher:.4f}
    A (100CE+Newton):     val={results['A']:.4f}  [gold standard]
    B (MINRES+Newton):    val={results['B']:.4f}  [0 CE, ~0.1 CE equiv]
    C (MINRES+25CE+N):    val={results['C']:.4f}  [25 CE steps]
    D (25CE+MINRES+75CE): val={results['D']:.4f}  [MINRES at t*=25]

  KEY: sign of g.Hg
    g.Hg > 0: -Hg is downhill (good MINRES direction)
    g.Hg < 0: -Hg is uphill (MINRES would worsen loss)
    
  IF g.Hg > 0 AND B < 0.16:
    MINRES pre-compensates for rotation better than raw gradient
    The third-order correction works
    
  IF D < A with 100 total steps: optimal t*=25 exists
    Rotation correction at t*=25 reduces total CE steps
""")
