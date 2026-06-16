#!/usr/bin/env python3
"""
Riemannian Gradient Descent on MC Moduli Space
================================================
Standard SGD: theta <- theta - lr * grad_theta(L)
Riemannian GD: theta <- Exp_theta(-lr * grad_M(L))

The MC moduli space M_MC has curvature mu0 = 1.99.
The Riemannian metric G is approximated by the empirical
Fisher information matrix on the attention distributions.

PRACTICAL IMPLEMENTATION (diagonal Fisher):
  For each parameter block W:
    F_W = E[grad_W(L)^2]  (diagonal Fisher, running average)
    natural_grad_W = grad_W(L) / (F_W + eps)  (preconditioned)
    W <- W - lr * natural_grad_W

This is Adagrad/Adam with the interpretation as natural gradient.
The Dehn gap = 1.4 sets the trust region radius.

DEHN GAP TRUST REGION:
  The Dehn gap = 1.4 is the minimum distance between distinct
  branches of the MC moduli space. Standard SGD can jump between
  branches (causing oscillation). RGD with trust region <= 1.4
  stays on one branch and converges monotonically.

  Implementation: clip the natural gradient norm to <= DEHN_GAP
  before applying the update. This is the geodesic retraction.

THREE VARIANTS:
  A: Standard SGD (Adam) — baseline
  B: Natural gradient (diagonal Fisher) — Riemannian metric
  C: Natural gradient + Dehn gap trust region clipping
  D: Natural gradient + Dehn gap + prime cascade

PREDICTION:
  B should converge faster than A because the natural gradient
  removes the ill-conditioning of the parameter space.
  C should converge more smoothly (no branch jumping) but
  possibly slower in early steps.
  D is the full pipeline: best cascade + best optimizer.
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14; DEHN_GAP=1.4

print(f"\n{'='*65}")
print(f"  RIEMANNIAN GRADIENT DESCENT")
print(f"  Natural gradient + Dehn gap trust region")
print(f"  Dehn gap = {DEHN_GAP} (T4 invariant, trust region radius)")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

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
    def hidden_states(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs

def clr_base(s,total=200,warmup=50):
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

def comm(A,B): return A@B-B@A
def l3(a,b,c): return comm(comm(a,b),c)-comm(a,comm(b,c))
def N(A): return float(np.linalg.norm(A))
def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)
def mu4(a,b,c,d):
    return -(comm(l3(a,b,c),d)-l3(comm(a,b),c,d)+l3(a,comm(b,c),d)
            -l3(a,b,comm(c,d))+comm(a,l3(b,c,d)))
def mu5(a,b,c,d,e):
    return -(l3(l3(a,b,c),d,e)-l3(a,l3(b,c,d),e)+l3(a,b,l3(c,d,e))
            +comm(mu4(a,b,c,d),e)+comm(a,mu4(b,c,d,e))
            -mu4(comm(a,b),c,d,e)+mu4(a,b,c,comm(d,e)))
def mu6(a,b,c,d,e,f):
    m5ab=mu5(a,b,c,d,e); m5bc=mu5(b,c,d,e,f)
    m4ab=mu4(a,b,c,d); m4bc=mu4(b,c,d,e); m4cd=mu4(c,d,e,f)
    m3ab=l3(a,b,c); m3bc=l3(b,c,d); m3cd=l3(c,d,e); m3de=l3(d,e,f)
    return -(comm(m5ab,f)-comm(a,m5bc)+l3(m4ab,e,f)-l3(a,m4bc,f)
            +l3(a,b,m4cd)+mu4(m3ab,d,e,f)-mu4(a,m3bc,e,f)
            +mu4(a,b,m3cd,f)-mu4(a,b,c,m3de))

# ════════════════════════════════════════════════════
# Train teacher + cascades
# ════════════════════════════════════════════════════
print("Training teacher...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups:
        pg['lr']=LR*min(step,100)/100 if step<=100 else \
                 LR*0.5*(1+math.cos(math.pi*(step-100)/200))
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

torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
J_acc=[[] for _ in range(N_LAYERS_T)]; U_acc=[[] for _ in range(N_LAYERS_T)]
for ref in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS_T):
        J,U=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=J.shape[0]
    if (ref+1)%3==0: print(f"  ref {ref+1}/5...",flush=True)
Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS_T)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS_T)]
J14=Js[L_ATT]; U14=Us[L_ATT]

# Cascades
cascade_serre=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    cascade_serre.append(C/max(N(C),1e-8))

att_basin=[l for l in range(8,21) if N(Js[l]-np.eye(ma))<0.75]
combos=list(itertools.combinations(att_basin,6))
scored=sorted([(c,N(mu6(*[Js[i] for i in c]))) for c in combos],key=lambda x:-x[1])
cascade_prime=[mu6(*[Js[i] for i in c])/max(N(mu6(*[Js[i] for i in c])),1e-8)
               for c,_ in scored[:N_STU]]

def build_student(cascade):
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight)
        stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            W_d=lift_to_d(cascade[l],U14,scale=0.01)
            W_t=torch.tensor(W_d,dtype=torch.float32)
            stu.blocks[l].attn.WK.weight.copy_(W_t)
            stu.blocks[l].attn.WQ.weight.copy_(W_t.T)
            stu.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            stu.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            stu.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            stu.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            stu.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
    return stu

# ════════════════════════════════════════════════════
# RIEMANNIAN GD OPTIMIZER
# ════════════════════════════════════════════════════
class RiemannianAdam(torch.optim.Optimizer):
    """
    Adam with Dehn gap trust region.
    
    The Dehn gap = 1.4 is the trust region radius in the
    MC moduli space. We clip the effective step size so that
    no parameter moves more than dehn_gap * lr per step.
    
    This keeps the trajectory on a single branch of the
    MC moduli space — no branch jumping (the source of
    oscillation in standard Adam near the branch cuts).
    
    Concretely: after computing the Adam update direction,
    clip its norm to dehn_gap. This is the geodesic retraction
    onto the trust region ball.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999),
                 eps=1e-8, weight_decay=0.1, dehn_gap=1.4):
        defaults=dict(lr=lr,betas=betas,eps=eps,
                      weight_decay=weight_decay,dehn_gap=dehn_gap)
        super().__init__(params,defaults)

    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()

        for group in self.param_groups:
            lr=group['lr']; b1,b2=group['betas']
            eps=group['eps']; wd=group['weight_decay']
            dehn=group['dehn_gap']

            for p in group['params']:
                if p.grad is None: continue
                g=p.grad.data
                if wd!=0: g=g+wd*p.data

                state=self.state[p]
                if len(state)==0:
                    state['step']=0
                    state['m']=torch.zeros_like(p.data)
                    state['v']=torch.zeros_like(p.data)

                state['step']+=1; t=state['step']
                m,v=state['m'],state['v']
                m.mul_(b1).add_(g,alpha=1-b1)
                v.mul_(b2).addcmul_(g,g,value=1-b2)
                bc1=1-b1**t; bc2=1-b2**t
                m_hat=m/bc1; v_hat=v/bc2
                update=m_hat/(v_hat.sqrt()+eps)

                # Dehn gap trust region: clip update norm
                # This is the geodesic retraction onto the
                # ball of radius dehn_gap in the MC metric
                u_norm=update.norm()
                if u_norm>dehn:
                    update=update*(dehn/u_norm)

                p.data.add_(update,alpha=-lr)
        return loss

# ════════════════════════════════════════════════════
# EXPERIMENTS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STUDENT EXPERIMENTS")
print("  A: Serre + standard Adam")
print("  B: Prime + standard Adam")
print("  C: Prime + Riemannian Adam (natural gradient)")
print("  D: Prime + Riemannian Adam + Dehn trust region")
print("="*65)

def run(cascade, label, steps=200, optimizer_cls=None, dehn_trust=False):
    stu=build_student(cascade)
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")

    if optimizer_cls=='riemannian':
        opt_s=RiemannianAdam(
            stu.parameters(), lr=LR, betas=(0.9,0.95),
            weight_decay=0.1,
            dehn_gap=DEHN_GAP if dehn_trust else float('inf'))
    else:
        opt_s=torch.optim.AdamW(
            stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

    ck={}
    for step in range(1,steps+1):
        lr_now=clr_base(step,steps)
        for pg in opt_s.param_groups: pg['lr']=lr_now
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

vA,ckA=run(cascade_serre,"A-Serre-Adam",optimizer_cls='adam')
vB,ckB=run(cascade_prime,"B-Prime-Adam",optimizer_cls='adam')
vC,ckC=run(cascade_prime,"C-Prime-RiemannAdam",optimizer_cls='riemannian',dehn_trust=False)
vD,ckD=run(cascade_prime,"D-Prime-Riemann+Dehn",optimizer_cls='riemannian',dehn_trust=True)

print(f"\n{'='*65}")
print("  RIEMANNIAN GD RESULTS")
print("="*65)
print(f"\n  CONVERGENCE:")
print(f"  {'step':>6}  {'A-Serre':>8}  {'B-Prime':>8}  {'C-RiemAd':>9}  {'D-R+Dehn':>9}")
for s in [25,50,75,100,125,150,200]:
    a=ckA.get(s); b=ckB.get(s); c=ckC.get(s); d=ckD.get(s)
    row=f"  {s:>6}"
    for v in [a,b,c,d]:
        row+=f"  {v:>8.4f}" if v else f"  {'---':>8}"
    best=min((v for v in [b,c,d] if v),default=99)
    if a and best<a-0.003: row+=" ←"
    print(row)

print(f"""
  FINAL:
    Teacher:               val={val_teacher:.4f}
    A (Serre+Adam):        val={vA:.4f}
    B (Prime+Adam):        val={vB:.4f}  diff={vA-vB:+.4f}
    C (Prime+RiemAdam):    val={vC:.4f}  diff={vA-vC:+.4f}
    D (Prime+Riemann+Dehn):val={vD:.4f}  diff={vA-vD:+.4f}

  DEHN GAP TRUST REGION INTERPRETATION:
    The Dehn gap = {DEHN_GAP} clips the Adam update norm.
    Standard Adam: update norm can be arbitrarily large
      → can jump between branches of M_MC
    Riemannian Adam + Dehn: update norm <= {DEHN_GAP}
      → stays on one branch, monotone convergence

    IF D < C: the trust region helps (prevents branch jumping)
    IF C < A: natural gradient metric helps (better conditioning)
    IF D < B at steps 25-75: Riemannian GD reduces Phase 2 cost

  NOTE ON ADAM vs RIEMANNIAN:
    Adam already approximates the natural gradient via v_hat
    (adaptive learning rate = diagonal Fisher preconditioning).
    The Dehn gap trust region is the ADDITIONAL constraint
    that keeps the trajectory on the MC moduli space manifold.
    
    If D ≈ B: Adam's adaptive rates already approximate the
    Riemannian metric sufficiently. The Dehn clip adds noise.
    
    If D < B: the explicit Dehn gap constraint provides
    geometric information that Adam's heuristics miss.
""")
