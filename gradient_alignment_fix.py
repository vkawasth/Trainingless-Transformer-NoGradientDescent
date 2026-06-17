#!/usr/bin/env python3
"""
Gradient Rotation Fix — Basin Principal Axis Alignment
=======================================================
The 0.16 barrier is caused by gradient rotation:
  - Entry point (val=0.281) has gradient pointing off-axis
  - Each CE step partially corrects the direction
  - 100 steps needed because gradient rotates ~72 degrees
    (alignment 0.28 at entry vs 1.0 at floor)

THE FIX:
  The basin principal axis = direction from entry to floor.
  Proxy: gradient at the floor (grad@T_final after full training).
  
  If we run a REFERENCE trajectory to get grad@final,
  then align the entry gradient with grad@final before
  taking the single large step, the step lands on-axis.
  
  But: we don't have grad@final without running 100 CE steps.
  
  ALTERNATIVE: use the gradient AFTER a few CE steps as proxy.
  After 10 CE steps from the entry, the gradient has rotated
  significantly toward the basin axis (alignment rises from 0.28
  to ~0.40 in 10 steps). Use this rotated gradient for the
  large Newton step instead of the entry gradient.
  
  This is: 10 CE steps (to rotate gradient) + 1 large step + Newton.
  Total: ~10-12 CE equiv but should reach val~0.05-0.08.

GEOMETRIC INTERPRETATION:
  The entry gradient G_entry points at angle theta to the basin axis.
  The large step -G_entry/mu moves distance d in direction theta.
  The component along the basin axis: d*cos(theta).
  The wasted component (off-axis): d*sin(theta).
  
  If theta=72 degrees (alignment 0.28 = cos(72)):
    cos(72) = 0.31 -- only 31% of the step reaches the floor
    sin(72) = 0.95 -- 95% is wasted as off-axis movement
    
  After 10 CE steps, alignment rises to ~0.40 (theta=66 degrees):
    cos(66) = 0.41 -- 41% efficiency
    
  After 25 CE steps, alignment rises to ~0.50 (theta=60 degrees):
    cos(60) = 0.50 -- 50% efficiency
    
  After 100 CE steps: alignment~0.53, theta~58, efficiency~53%
  (plateau: gradient cannot fully align because the basin is curved)
  
  The OPTIMAL strategy: run just enough CE steps to rotate the gradient
  toward the basin axis, then take the large step.
  
MEASUREMENT PLAN:
  At each CE step t, measure:
  1. val(t)
  2. gradient alignment with val@100 gradient (proxy for floor direction)
  3. ||G(t)|| (gradient magnitude)
  4. Predicted val after large step: val(t) - ||G(t)|| * alignment(t) * step_size
  
  Find optimal t where predicted improvement is maximized.
  Then run: t CE steps + large step + (100-t) CE steps + Newton.
  Compare with: 100 CE steps + Newton.
"""
import json, math, time, copy, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429

print(f"\n{'='*65}")
print(f"  GRADIENT ROTATION FIX")
print(f"  Find optimal CE steps before large step")
print(f"  Measure alignment with basin principal axis")
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

def full_grad(m, n=30):
    m.train(); m.zero_grad()
    for _ in range(n): x,y=get_batch(); _,l=m(x,y); (l/n).backward()
    g=m.flat_grad().detach().clone(); m.zero_grad()
    return g

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

# Build base state
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

# ═══════════════════════════════════════
# MEASUREMENT: gradient rotation profile
# ═══════════════════════════════════════
print("="*65)
print("MEASUREMENT: Gradient rotation profile along basin descent")
print("  Track alignment with floor gradient at each CE step")
print("  Identify optimal t* for large step injection")
print("="*65)

# Step 1: Get the floor gradient (run to convergence)
print("\nStep 1: Get floor gradient (100 CE steps)...")
stu_ref2=copy.deepcopy(stu_base)
opt2=torch.optim.AdamW(stu_ref2.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,101):
    for pg in opt2.param_groups: pg['lr']=clr(step,100)
    stu_ref2.train(); x,y=get_batch(); _,loss=stu_ref2(x,y)
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu_ref2.parameters(),1.0); opt2.step()
g_floor=full_grad(stu_ref2,n=30)
v_floor=eval_val(stu_ref2,n=20)
print(f"  Floor val: {v_floor:.4f}  ||g_floor||={float(g_floor.norm()):.4f}")

# Step 2: Measure gradient rotation profile
print("\nStep 2: Gradient alignment profile (entry -> floor)...")
stu_probe=copy.deepcopy(stu_base)
opt3=torch.optim.AdamW(stu_probe.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

print(f"\n  {'step':>5}  {'val':>7}  {'align':>8}  {'||g||':>7}  "
      f"{'eff_axis':>9}  {'pred_improvement'}")
print("  "+"-"*65)

grads_at_steps={}
for step in range(0,76):
    if step>0:
        for pg in opt3.param_groups: pg['lr']=clr(step,100)
        stu_probe.train(); x,y=get_batch(); _,loss=stu_probe(x,y)
        opt3.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu_probe.parameters(),1.0); opt3.step()

    if step in [0,5,10,15,20,25,33,50,75]:
        v=eval_val(stu_probe,n=15)
        g=full_grad(stu_probe,n=20)
        gnorm=float(g.norm())
        flnorm=float(g_floor.norm())
        align=float((g*g_floor).sum())/(gnorm*flnorm+1e-10)
        # Effective component along basin axis
        eff=gnorm*max(align,0)
        # Predicted improvement from large step
        step_size=0.15  # typical ||delta||
        pred_imp=eff*step_size
        grads_at_steps[step]=(g.clone(),v,align,gnorm)
        print(f"  {step:>5}  {v:>7.4f}  {align:>8.4f}  {gnorm:>7.4f}  "
              f"{eff:>9.4f}  {pred_imp:.4f}")

# ═══════════════════════════════════════
# EXPERIMENTS: inject large step at optimal t*
# ═══════════════════════════════════════
print(f"\n{'='*65}")
print("EXPERIMENTS: Large step injection at different t*")
print("  A: standard 100CE + Newton")
print("  B: 0 CE + large step + 100CE + Newton")
print("  C: 10CE + large step + 90CE + Newton")
print("  D: 25CE + large step + 75CE + Newton")
print("  E: 25CE + large step (aligned) + 75CE + Newton")
print("="*65)

def large_step_inject(stu_start, pre_steps, post_steps,
                      use_aligned_grad=False, step_size=0.15):
    """
    Run pre_steps CE, inject large step in gradient direction,
    run post_steps CE, Newton.
    """
    s=copy.deepcopy(stu_start)
    opt=torch.optim.AdamW(s.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    total=pre_steps+post_steps

    # Pre-steps
    for step in range(1,pre_steps+1):
        for pg in opt.param_groups: pg['lr']=clr(step,total)
        s.train(); x,y=get_batch(); _,loss=s(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt.step()

    v_pre=eval_val(s,n=10)

    # Large step in gradient direction
    if use_aligned_grad:
        # Use component of gradient aligned with floor direction
        g=full_grad(s,n=20)
        flnorm=float(g_floor.norm()); gnorm=float(g.norm())
        align=float((g*g_floor).sum())/(gnorm*flnorm+1e-10)
        # Project g onto floor direction
        g_aligned=g_floor*(float((g*g_floor).sum())/(flnorm**2))
        g_step=g_aligned
        print(f"    alignment={align:.3f}  using aligned gradient")
    else:
        g=full_grad(s,n=20)
        g_step=g

    g_norm=float(g_step.norm())
    delta=-(g_step/g_norm)*step_size  # step in -gradient direction, fixed size
    w0=s.flat_params().clone()
    s.set_flat(w0+delta)
    v_post_step=eval_val(s,n=10)
    print(f"    pre={v_pre:.4f}  after_step={v_post_step:.4f}  ||delta||={step_size:.3f}")

    # Post-steps
    opt2=torch.optim.AdamW(s.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,post_steps+1):
        for pg in opt2.param_groups: pg['lr']=clr(step,post_steps)
        s.train(); x,y=get_batch(); _,loss=s(x,y)
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(s.parameters(),1.0); opt2.step()
        if step in [25,50,75,post_steps]:
            v=eval_val(s,n=10)
            print(f"    post CE {step}: {v:.4f}")

    apply_newton_wk(s)
    vf=eval_val(s,n=30)
    print(f"    FINAL={vf:.4f}")
    return vf

results={}

# A: baseline
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

print("\n[B] 0CE + large step + 100CE:")
results['B']=large_step_inject(stu_base,0,100,step_size=0.15)

print("\n[C] 10CE + large step + 90CE:")
results['C']=large_step_inject(stu_base,10,90,step_size=0.15)

print("\n[D] 25CE + large step + 75CE:")
results['D']=large_step_inject(stu_base,25,75,step_size=0.15)

print("\n[E] 25CE + ALIGNED large step + 75CE:")
results['E']=large_step_inject(stu_base,25,75,use_aligned_grad=True,step_size=0.15)

print(f"""
{'='*65}
  GRADIENT ROTATION FIX RESULTS
{'='*65}
    Teacher:              val={val_teacher:.4f}
    A (100CE standard):   val={results['A']:.4f}  [baseline]
    B (step at t=0):      val={results['B']:.4f}  [no rotation correction]
    C (step at t=10):     val={results['C']:.4f}  [10 steps of rotation]
    D (step at t=25):     val={results['D']:.4f}  [25 steps of rotation]
    E (aligned step t=25):val={results['E']:.4f}  [projected onto floor axis]

  IF C or D < A: gradient rotation was the cause of meandering
    Waiting for gradient to rotate toward basin axis reduces total CE steps
    Optimal t* is where alignment * ||g|| is maximized
    
  IF E < D: alignment projection helps
    Removing off-axis gradient component reduces wasted movement
    Pure axis step is more efficient than raw gradient step
    
  KEY INSIGHT: if optimal t* exists where C or D requires fewer
  total steps (pre + post) than A's 100 steps, then:
  gradient rotation correction reduces the irreducible minimum.
""")
