#!/usr/bin/env python3
"""
Hyperbolic Gradient Descent
============================
The complex Grassmannian showed the training trajectory is a
libration on the unit sphere |z| ~ 1.03 in Gr(2,4)_C.

In the upper half-plane H^2, this libration maps to oscillation
along the real axis. The hyperbolic metric straightens geodesics
into the correct path to z* (the MC fixed point).

THE METRIC:
  In complex coordinates z = Re + i*Im:
  G_hyp(z) = 1/Im(z)^2 * I

  This amplifies gradients when Im(z) is small (near the real axis,
  where the libration spends most time) and suppresses them when
  Im(z) is large (away from the oscillation region).

TRANSLATION TO WEIGHT SPACE:
  The complex coordinate z_l = sv_1(J_l) * exp(i*theta_l)
  Im(z_l) = sv_1(J_l) * sin(theta_l)

  The hyperbolic preconditioner for W_K at block l:
    P_l = 1 / Im(z_l)^2

  This is a SCALAR preconditioner per block — not diagonal Fisher.
  It uses the GEOMETRY of the trajectory, not the gradient statistics.

  Update: W_K^l <- W_K^l - eta * P_l * grad_{W_K^l} L

  When the trajectory is near the real axis (Im(z) small, librating):
    P_l is LARGE -> large step -> pushes through the flat region fast
  When the trajectory is away from real axis (Im(z) large, on path):
    P_l is small -> small step -> careful refinement

THIS IS THE COMPLEX ANALYSIS ACTING, NOT JUST DETECTING.

The hyperbolic metric knows the trajectory shape and uses it
to accelerate through the libration region and slow down
near the fixed point. No 200 steps of equal-sized oscillation —
instead, large steps near the flat region and small steps near z*.

PREDICTION:
  Hyperbolic GD should converge in ~5-10 steps (spectral gap = 0.20)
  because the hyperbolic metric makes the convergence rate
  independent of the libration amplitude.
  
  The hyperbolic geodesic from z_0 to z* is a SEMICIRCLE
  in H^2 — a direct path. The flat-metric trajectory is
  the librating oscillation. These are the same path in
  different coordinate systems, but the geodesic parameterization
  reaches z* exponentially faster.
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  HYPERBOLIC GRADIENT DESCENT")
print(f"  Metric from complex Grassmannian trajectory geometry")
print(f"  G_hyp = 1/Im(z)^2 — amplifies steps near flat region")
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

def get_complex_z(J):
    """Get complex Grassmannian coordinate of Jacobian J."""
    U,sv,_=np.linalg.svd(J,full_matrices=False)
    return sv[0], U[:,0]  # (|z|, u1)

def get_student_Im_z(stu, x_ref, pos, m):
    """
    Get Im(z_l) for each student block.
    Im(z_l) = sv_1(J_l) * sin(theta_l)
    theta_l = accumulated angle of dominant SV direction.
    """
    stu.eval()
    with torch.no_grad(): hs=stu.hidden_states_all(x_ref); hs=[h[0] for h in hs]
    Im_z=[]
    theta_acc=0.0; prev_u1=None
    for l in range(N_STU):
        J,_=layer_jac(stu.blocks[l],hs[l],pos,m)
        sv1,u1=get_complex_z(J)
        if prev_u1 is not None:
            cos_t=float(np.clip(prev_u1@u1,-1,1))
            dtheta=math.acos(abs(cos_t))
            if prev_u1@u1<0: dtheta=-dtheta
            theta_acc+=dtheta
        Im_z.append(sv1*math.sin(theta_acc))
        prev_u1=u1
    return Im_z

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

pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
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

def clr(s,total=200,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ════════════════════════════════════════════════════
# HYPERBOLIC ADAM — metric from complex trajectory
# ════════════════════════════════════════════════════
class HyperbolicAdam(torch.optim.Optimizer):
    """
    Adam with hyperbolic preconditioner per block.
    
    For W_K at block l:
      P_l = 1 / (Im(z_l)^2 + eps)
      update = P_l * adam_direction
    
    When Im(z_l) is small (trajectory near real axis, librating):
      P_l large -> large step -> accelerates through flat region
    When Im(z_l) is large (trajectory moving toward z*):
      P_l small -> careful step -> precise convergence
    
    This uses the complex Grassmannian geometry to act,
    not just detect. The metric IS the trajectory.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999),
                 eps=1e-8, weight_decay=0.1,
                 block_Im_z=None, hyp_eps=0.1):
        defaults=dict(lr=lr,betas=betas,eps=eps,
                     weight_decay=weight_decay,
                     hyp_eps=hyp_eps)
        super().__init__(params,defaults)
        # Im(z_l) for each block — updated each step
        self.block_Im_z=block_Im_z or [1.0]*N_STU

    def update_Im_z(self, Im_z_list):
        """Update hyperbolic preconditioner from current trajectory."""
        self.block_Im_z=Im_z_list

    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()

        for group in self.param_groups:
            lr=group['lr']; b1,b2=group['betas']
            eps=group['eps']; wd=group['weight_decay']
            hyp_eps=group['hyp_eps']

            for p in group['params']:
                if p.grad is None: continue
                g=p.grad.data
                if wd!=0: g=g+wd*p.data

                state=self.state[p]
                if 'step' not in state:
                    state['step']=0
                    state['m']=torch.zeros_like(p.data)
                    state['v']=torch.zeros_like(p.data)
                    state['block_idx']=None  # set below

                state['step']+=1; t=state['step']
                m,v=state['m'],state['v']
                m.mul_(b1).add_(g,alpha=1-b1)
                v.mul_(b2).addcmul_(g,g,value=1-b2)
                bc1=1-b1**t; bc2=1-b2**t
                m_hat=m/bc1; v_hat=v/bc2
                adam_dir=m_hat/(v_hat.sqrt()+eps)

                # Hyperbolic preconditioner
                block_idx=state.get('block_idx', None)
                if block_idx is not None and block_idx<len(self.block_Im_z):
                    Im_z=self.block_Im_z[block_idx]
                    hyp_scale=1.0/(Im_z**2+hyp_eps)
                    # Clip to avoid explosion when Im_z is very small
                    hyp_scale=min(hyp_scale, 10.0)
                    adam_dir=adam_dir*hyp_scale

                p.data.add_(adam_dir,alpha=-lr)
        return loss

def run_hyperbolic(label,steps=200):
    stu=build_student()
    v0=eval_val(stu,n=20)
    print(f"\n  [{label}] zero-shot={v0:.4f}")

    # Initialize Im(z) for each block
    Im_z=get_student_Im_z(stu,x_ref,pos,m)
    print(f"  Initial Im(z) per block: {[f'{z:.3f}' for z in Im_z]}")

    opt_h=HyperbolicAdam(stu.parameters(),lr=LR,betas=(0.9,0.95),
                          weight_decay=0.1,block_Im_z=Im_z,hyp_eps=0.1)

    # Assign block indices to parameters
    for l in range(N_STU):
        for p in stu.blocks[l].parameters():
            opt_h.state[p]['block_idx']=l

    ck={0:v0}
    for step in range(1,steps+1):
        for pg in opt_h.param_groups: pg['lr']=clr(step,steps)

        # Update Im(z) every 10 steps
        if step%10==0:
            Im_z=get_student_Im_z(stu,x_ref,pos,m)
            opt_h.update_Im_z(Im_z)

        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_h.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0)
        opt_h.step()

        if step in [10,25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            Im_z_now=Im_z[N_STU//2]
            print(f"  [{label}] step {step:>4}  val={v:.4f}  "
                  f"Im(z_mid)={Im_z_now:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

def run_standard(label,steps=200):
    stu=build_student()
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={0:v0}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [10,25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

print("="*65)
print("EXPERIMENTS")
print("  A: Standard Adam (baseline)")
print("  B: Hyperbolic Adam (Im(z)^{-2} preconditioner per block)")
print("="*65)

vA,ckA=run_standard("A-Standard-Adam")
vB,ckB=run_hyperbolic("B-Hyperbolic-Adam")

print(f"\n{'='*65}")
print("  HYPERBOLIC GD RESULTS")
print("="*65)
print(f"\n  CONVERGENCE:")
print(f"  {'step':>6}  {'A-Adam':>8}  {'B-Hyp':>8}  {'speedup'}")
for s in [10,25,50,75,100,125,150,200]:
    a=ckA.get(s); b=ckB.get(s)
    row=f"  {s:>6}  {a:>8.4f}  {b:>8.4f}" if a and b else f"  {s:>6}  ---"
    if a and b and b<a-0.003: row+="  ← FASTER"
    print(row)

print(f"""
  FINAL:
    Teacher:           val={val_teacher:.4f}
    A (Standard Adam): val={vA:.4f}
    B (Hyperbolic):    val={vB:.4f}  diff={vA-vB:+.4f}

  IF B converges faster than A (same final val, fewer steps):
    The hyperbolic metric correctly straightens the librating
    trajectory into a direct geodesic path.
    The complex Grassmannian geometry is ACTING, not just detecting.
    
  IF B reaches lower final val than A:
    The hyperbolic metric finds a better fixed point than Adam.
    The Im(z)^{{-2}} preconditioner removes the oscillation cost
    and reaches W_K* that Adam cannot find in 200 steps.

  THE INSIGHT:
    The libration in arg(z) is the model oscillating between
    Stokes chambers. Each oscillation wastes steps.
    The hyperbolic metric makes each step proportional to
    the chamber distance — large steps when far from z*,
    small steps when close. No wasted oscillation.
""")
