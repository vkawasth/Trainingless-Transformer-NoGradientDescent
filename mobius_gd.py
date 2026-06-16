#!/usr/bin/env python3
"""
Möbius-Corrected Gradient Descent
===================================
The hyperbolic preconditioner failed because:
1. Im(z_0) = 0 always (coordinate artifact) -> block 0 explodes
2. Preconditioner was per-block spatial, not per-step temporal
3. Sign was wrong: 1/Im^2 amplifies near real axis where libration lives

THE CORRECT CONSTRUCTION:
  The librating trajectory z(t) ≈ A*exp(i*omega*t) winds around the unit circle.
  The fixed point z* is where the trajectory settles (step 200 value).
  
  Möbius transformation: w(z) = (z - z*) / (1 - conj(z*) * z)
  This maps the unit disk to itself with z* -> 0.
  In w-coordinates, the trajectory converges monotonically to 0.
  
  The Möbius-corrected step size at position z(t):
    |dw/dz| = (1 - |z*|^2) / |1 - conj(z*)*z|^2
  
  This is the CORRECT preconditioner: large when z is far from z*,
  small when z is near z*. It straightens the libration.

PRACTICAL IMPLEMENTATION:
  We know z*(t=200) from the complex_grassmannian data:
    z*_mid ≈ -0.514 + 0.896i  (step 200 value)
  
  Per-step preconditioner (scalar, applied to ALL WK gradients):
    P(t) = |dw/dz|(z_mid(t)) = (1 - |z*|^2) / |1 - conj(z*)*z_mid(t)|^2
  
  This is computed from the current complex Grassmannian coordinate,
  uses the known fixed point, and gives the correct amplification.

  When z_mid(t) is far from z* (early training, large libration):
    |1 - conj(z*)*z| is large -> P(t) is small -> conservative step
  When z_mid(t) is near z* (late training, converging):
    |1 - conj(z*)*z| is small -> P(t) is large -> aggressive final step
    
  This is the OPPOSITE of the failed hyperbolic preconditioner.
  The Möbius metric accelerates convergence, not libration.

ALSO TEST: Three-stage pipeline
  A: Standard Adam 200 steps (baseline)
  B: Möbius Adam 200 steps
  C: Standard Adam 200 steps + Newton (confirmed +0.0035 nats)
  D: Möbius Adam 100 steps + Newton (goal: match C in fewer steps)
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

# Fixed point from complex_grassmannian.py step-200 data
Z_STAR = complex(-0.514, 0.896)  # z* for mid block at convergence

print(f"\n{'='*65}")
print(f"  MÖBIUS-CORRECTED GRADIENT DESCENT")
print(f"  w(z) = (z-z*)/(1-conj(z*)*z)  maps libration -> monotone")
print(f"  z* = {Z_STAR:.3f}  (from complex_grassmannian step 200)")
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
    def hidden_states_all(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs

def clr(s,total=200,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def layer_jac(block,h_in,pos,m):
    seq,d_=h_in.shape; m=min(m,seq,d_)
    _,_,Vt=torch.linalg.svd(h_in,full_matrices=False)
    U=Vt[:m,:].T.detach(); J=np.zeros((m,m))
    with torch.enable_grad():
        for i in range(m):
            hh=h_in.clone().unsqueeze(0).detach().requires_grad_(True)
            ho=block(hh)
            v=(ho[0,pos,:] if ho.dim()==3 else ho[pos,:])
            (v*U[:,i]).sum().backward()
            g=hh.grad; g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
            J[:,i]=(U.T@g).numpy()
    return J.T,U.detach().numpy()

def get_z_mid(stu, x_ref, pos, m):
    """Get complex Grassmannian coordinate of middle block."""
    stu.eval()
    with torch.no_grad(): hs=stu.hidden_states_all(x_ref); hs=[h[0] for h in hs]
    mid=N_STU//2
    J,_=layer_jac(stu.blocks[mid],hs[mid],pos,m)
    U,sv,_=np.linalg.svd(J,full_matrices=False)
    u1=U[:,0]; sv1=sv[0]
    # Accumulated angle up to mid block
    theta=0.0; prev_u1=None
    for l in range(mid+1):
        Jl,_=layer_jac(stu.blocks[l],hs[l],pos,m)
        Ul,_,_=np.linalg.svd(Jl,full_matrices=False)
        u1l=Ul[:,0]
        if prev_u1 is not None:
            cos_t=float(np.clip(prev_u1@u1l,-1,1))
            dt=math.acos(abs(cos_t))
            if prev_u1@u1l<0: dt=-dt
            theta+=dt
        prev_u1=u1l
    sv_mid,_=get_complex_z_fast(J)
    return complex(sv_mid*math.cos(theta), sv_mid*math.sin(theta))

def get_complex_z_fast(J):
    U,sv,_=np.linalg.svd(J,full_matrices=False)
    return sv[0],U[:,0]

def mobius_scale(z_current, z_star=Z_STAR):
    """
    Möbius derivative |dw/dz| at z_current.
    w(z) = (z - z*) / (1 - conj(z*) * z)
    |dw/dz| = (1 - |z*|^2) / |1 - conj(z*) * z|^2
    
    This is the correct preconditioner: large near z*, small far from z*.
    Clip to [0.1, 10.0] for stability.
    """
    z_star_conj=z_star.conjugate()
    numerator=1.0 - abs(z_star)**2
    denominator=abs(1.0 - z_star_conj*z_current)**2
    scale=numerator/max(denominator,1e-6)
    return float(np.clip(scale, 0.1, 10.0))

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

pos=SEQ//2; m=min(PROJ,SEQ,D)
torch.manual_seed(0); x_ref,_=get_batch('val'); x_ref=x_ref[0:1]

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

def apply_newton(stu, n_seq=500, eps=1e-3, scale=0.5):
    """Apply one Newton step to WK at current state."""
    grad_acc=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    fisher_d=torch.zeros_like(stu.blocks[0].attn.WK.weight)
    torch.manual_seed(2)
    for i in range(n_seq):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0)
        y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
        stu.zero_grad(); _,loss=stu(x,y); loss.backward()
        g=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        for l in range(N_STU):
            if stu.blocks[l].attn.WK.weight.grad is not None:
                g+=stu.blocks[l].attn.WK.weight.grad/N_STU
        grad_acc+=g; fisher_d+=g**2
    grad_mean=grad_acc/n_seq
    fisher_diag=fisher_d/n_seq
    delta=-(grad_mean/(fisher_diag+eps))
    with torch.no_grad():
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.add_(scale*delta)
            stu.blocks[l].attn.WQ.weight.add_(scale*delta.T)
    return float(grad_mean.norm())

# ════════════════════════════════════════════════════
# EXPERIMENTS
# ════════════════════════════════════════════════════
print("="*65)
print("EXPERIMENTS")
print("  A: Standard Adam 200 steps")
print("  B: Möbius Adam 200 steps (scalar preconditioner from z*)")
print("  C: Standard Adam 200 + Newton (confirmed +0.0035 improvement)")
print("  D: Möbius Adam 100 + Newton (goal: match C in fewer steps)")
print("="*65)

def run_standard(label, steps=200, apply_newton_after=False):
    stu=build_student()
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={0:v0}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    if apply_newton_after:
        print(f"  [{label}] Applying Newton step...")
        gnorm=apply_newton(stu)
        v_newton=eval_val(stu)
        ck['newton']=v_newton
        print(f"  [{label}] After Newton: val={v_newton:.4f}  ||g||={gnorm:.6f}")
    return eval_val(stu),ck

def run_mobius(label, steps=200, apply_newton_after=False):
    stu=build_student()
    v0=eval_val(stu,n=20)
    z0=get_z_mid(stu,x_ref,pos,m)
    p0=mobius_scale(z0)
    print(f"\n  [{label}] zero-shot={v0:.4f}  z_mid={z0:.3f}  P={p0:.3f}")

    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={0:v0}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0)

        # Möbius preconditioner: scale WK/WQ gradients only
        if step%5==0:  # update z_mid every 5 steps (cheap)
            z_mid=get_z_mid(stu,x_ref,pos,m)
            P=mobius_scale(z_mid)
        else:
            P=1.0  # use last computed P
        with torch.no_grad():
            for l in range(N_STU):
                if stu.blocks[l].attn.WK.weight.grad is not None:
                    stu.blocks[l].attn.WK.weight.grad.mul_(P)
                    stu.blocks[l].attn.WQ.weight.grad.mul_(P)

        opt_s.step()

        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            z_now=get_z_mid(stu,x_ref,pos,m) if step%25==0 else z_mid
            P_now=mobius_scale(z_now)
            print(f"  [{label}] step {step:>4}  val={v:.4f}  "
                  f"P={P_now:.3f}{' ✓' if v<val_teacher else ''}")

    if apply_newton_after:
        print(f"  [{label}] Applying Newton step...")
        gnorm=apply_newton(stu)
        v_newton=eval_val(stu)
        ck['newton']=v_newton
        print(f"  [{label}] After Newton: val={v_newton:.4f}  ||g||={gnorm:.6f}")
    return eval_val(stu),ck

vA,ckA=run_standard("A-Adam-200")
vB,ckB=run_mobius("B-Mobius-200")
vC,ckC=run_standard("C-Adam-200+Newton",apply_newton_after=True)
vD,ckD=run_mobius("D-Mobius-100+Newton",steps=100,apply_newton_after=True)

print(f"\n{'='*65}")
print("  MÖBIUS GD RESULTS")
print("="*65)
print(f"\n  CONVERGENCE:")
print(f"  {'step':>6}  {'A-Adam':>7}  {'B-Mob':>7}  {'C-A+N':>7}  {'D-M+N':>7}")
for s in [25,50,75,100,125,150,200,'newton']:
    a=ckA.get(s); b=ckB.get(s); c=ckC.get(s); d=ckD.get(s)
    row=f"  {str(s):>6}"
    for v in [a,b,c,d]:
        row+=f"  {v:>7.4f}" if v else f"  {'---':>7}"
    best=min((v for v in [b,c,d] if v),default=99)
    if a and best<a-0.003: row+=" ←"
    print(row)

print(f"""
  FINAL:
    Teacher:                  val={val_teacher:.4f}
    A (Adam 200):             val={vA:.4f}
    B (Möbius 200):           val={vB:.4f}  diff={vA-vB:+.4f}
    C (Adam 200 + Newton):    val={vC:.4f}  diff={vA-vC:+.4f}
    D (Möbius 100 + Newton):  val={vD:.4f}  diff={vA-vD:+.4f}

  THE MÖBIUS QUESTION:
    IF B < A: Möbius preconditioner straightens the trajectory.
      The complex geometry is ACTING — the libration is reduced.
      Fewer oscillations = faster convergence.
      
    IF D ~ C with half the CE steps:
      Möbius + Newton achieves the same quality in 100+1 steps
      vs Adam's 200+1 steps. 2x reduction confirmed.
      
    IF B ~ A and D ~ C:
      The Möbius scale P(t) is near 1.0 throughout
      (z_mid stays far from z*, P = (1-|z*|^2)/|1-z*_conj*z|^2 ~ 1).
      The geometry does not provide additional information beyond Adam.
      The three-stage pipeline (cascade + 200CE + Newton) is optimal.
""")
