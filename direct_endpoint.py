#!/usr/bin/env python3
"""
Direct Endpoint Assignment
===========================
We know start (cascade), end (teacher), path (Lefschetz thimble).
We do not need to walk the path. Assign the endpoint directly.

THE CONSTRUCTION:
  The student's 6 blocks should reproduce the teacher's behavior
  at the 6 prime path representative layers.
  
  The endpoint in the MC moduli space is the teacher's Jacobian
  at the attractor, averaged over the corpus.
  We have this: it is J_14 = the teacher's attractor Jacobian.
  
  The student block l should be initialized to reproduce
  the teacher's transformation at the l-th prime path layer.
  
  DIRECT ASSIGNMENT:
    For each prime path P_l = (i1, i2, i3, i4, i5, i6):
      W_K^l <- lift_to_D(mean(J_{i1}, ..., J_{i6}), U14)
  
  This assigns the ENDPOINT directly — the average teacher Jacobian
  over the prime path layers — which is where gradient descent
  is trying to reach.
  
  NO INTERPOLATION. NO 200 STEPS. Direct endpoint assignment.

THREE VARIANTS:
  A: Mean J over prime path layers (simple average)
  B: Weighted mean J (weighted by mu6 prime path weight)
  C: Geodesic midpoint between cascade and teacher endpoint
     alpha* = (1-t)*cascade + t*teacher_J  for t in [0,1]
     Find optimal t by line search (5 evaluations, not 200)
  D: Full teacher J_14 (attractor layer only, no averaging)

The key question: does direct endpoint assignment give
val ~ teacher val without ANY gradient steps?
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  DIRECT ENDPOINT ASSIGNMENT")
print(f"  Start=cascade, End=teacher, Path=known")
print(f"  Assign endpoint directly — no gradient steps")
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

# ════════════════════════════════════
# Train teacher
# ════════════════════════════════════
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

torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
with torch.no_grad():
    hs=teacher.hidden_states_all(x_ref); hs=[h[0] for h in hs]
Js=[]; Us=[]
for l in range(N_LAYERS_T):
    J,U=layer_jac(teacher.blocks[l],hs[l],pos,m)
    Js.append(J); Us.append(U)
    if ma is None: ma=J.shape[0]
    if (l+1)%8==0: print(f"  jac {l+1}/{N_LAYERS_T}...",flush=True)
J14=Js[L_ATT]; U14=Us[L_ATT]

att_basin=[l for l in range(8,21) if N(Js[l]-np.eye(ma))<0.75]
combos=list(itertools.combinations(att_basin,6))
scored=sorted([(c,N(mu6(*[Js[i] for i in c]))) for c in combos],key=lambda x:-x[1])
prime_paths=[c for c,_ in scored[:N_STU]]
prime_weights=[N(mu6(*[Js[i] for i in c])) for c in prime_paths]

# ════════════════════════════════════
# ENDPOINT CASCADES
# ════════════════════════════════════
print(f"\n{'='*65}")
print("ENDPOINT CONSTRUCTION")
print("  The teacher's J_l at prime path layers IS the endpoint")
print("="*65)

# Cascade (start point)
cascade_serre=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    cascade_serre.append(C/max(N(C),1e-8))

_cascade_prime_raw=[]
for c in prime_paths:
    op=mu6(*[Js[i] for i in c])
    n=N(op)
    if n>1e-10:
        _cascade_prime_raw.append(op/n)
# Pad with Serre cascade if fewer than N_STU prime paths found
while len(_cascade_prime_raw)<N_STU:
    _cascade_prime_raw.append(cascade_serre[len(_cascade_prime_raw)])
cascade_prime=_cascade_prime_raw[:N_STU]
print(f"  Prime cascade: {len(cascade_prime)} operators")

# ENDPOINT A: Mean teacher J over prime path layers
# Endpoint cascade: mean teacher J at prime path layers
# Always build N_STU entries; fall back to J14 if prime_paths is short
cascade_mean_J=[]
J14_norm=J14/max(N(J14),1e-8)
for pi in range(N_STU):
    if pi<len(prime_paths):
        pp=prime_paths[pi]
        J_avg=np.mean([Js[l] for l in pp],axis=0)
        n=N(J_avg)
        if n>1e-10:
            cascade_mean_J.append(J_avg/n)
            norms=[N(Js[l]) for l in pp]
            print(f"  PP{pi+1} {pp}: mean ||J|| = {np.mean(norms):.4f}")
        else:
            cascade_mean_J.append(J14_norm)
            print(f"  PP{pi+1}: zero mu6, using J14")
    else:
        cascade_mean_J.append(J14_norm)
        print(f"  PP{pi+1}: fallback J14")

# ENDPOINT B: J_14 directly (attractor layer — the single best point)
# The attractor Jacobian IS the endpoint of the MC integration
cascade_J14=[J14/max(N(J14),1e-8)]*N_STU
print(f"\n  J_14 norm: {N(J14):.4f}")

# ENDPOINT C: Geodesic interpolation
# alpha(t) = (1-t)*S_l + t*J_l_endpoint
# Find optimal t by evaluating val at t=0.0, 0.25, 0.5, 0.75, 1.0
# This is the "line search on the thimble"

def make_interp_cascade(cascade_start, cascade_end, t):
    return [(1-t)*s + t*e for s,e in zip(cascade_start,cascade_end)]

# ENDPOINT D: Teacher's actual WK weights projected to student
# This is the true endpoint — copy teacher WK at L14 to ALL student blocks
# (the student is a 6-block version of the teacher's attractor)

print(f"\n  Teacher WK at L14:")
WK_teacher=teacher.blocks[L_ATT].attn.WK.weight.data.numpy()  # (D,D)
print(f"  Shape: {WK_teacher.shape}, norm: {np.linalg.norm(WK_teacher):.4f}")

# ════════════════════════════════════
# BUILD STUDENTS
# ════════════════════════════════════
def build_student(cascade, use_teacher_WK=False, t_interp=None,
                  cascade_end=None):
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight)
        stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            if use_teacher_WK:
                # Direct endpoint: copy teacher's WK at L14
                stu.blocks[l].attn.WK.weight.copy_(
                    teacher.blocks[L_ATT].attn.WK.weight)
                stu.blocks[l].attn.WQ.weight.copy_(
                    teacher.blocks[L_ATT].attn.WQ.weight)
            else:
                op=cascade[l]
                if t_interp is not None and cascade_end is not None:
                    op=(1-t_interp)*op + t_interp*cascade_end[l]
                W_d=lift_to_d(op,U14,scale=0.01)
                W_t=torch.tensor(W_d,dtype=torch.float32)
                stu.blocks[l].attn.WK.weight.copy_(W_t)
                stu.blocks[l].attn.WQ.weight.copy_(W_t.T)
            stu.blocks[l].attn.WV.weight.copy_(
                teacher.blocks[L_ATT].attn.WV.weight)
            stu.blocks[l].attn.op.weight.copy_(
                teacher.blocks[L_ATT].attn.op.weight)
            stu.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            stu.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            stu.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
    return stu

def run(build_fn, label, steps=200):
    stu=build_fn()
    v0=eval_val(stu,n=30)
    print(f"\n  [{label}] zero-shot={v0:.4f}")
    if steps==0: return v0,{0:v0}
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={0:v0}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [10,25,50,100,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

print(f"\n{'='*65}")
print("EXPERIMENTS")
print("  A: Prime cascade + 200CE (baseline)")
print("  B: Mean teacher J at prime path layers (endpoint) + 200CE")
print("  C: J_14 directly (attractor endpoint) + 200CE")
print("  D: Teacher WK weights directly (true endpoint) + 200CE")
print("  E: Geodesic line search (t=0,0.25,0.5,0.75,1.0) — 5 evals only")
print("  F: Best geodesic t + 200CE")
print("="*65)

vA,ckA=run(lambda: build_student(cascade_prime),"A-Prime-std")
vB,ckB=run(lambda: build_student(cascade_mean_J),"B-MeanJ-endpoint")
vC,ckC=run(lambda: build_student(cascade_J14),"C-J14-endpoint")
vD,ckD=run(lambda: build_student(None,use_teacher_WK=True),"D-TeacherWK-direct")

# Geodesic line search: evaluate zero-shot at 5 t values
print(f"\n  E: Geodesic line search (5 evaluations, no training):")
best_t=0.0; best_v=float('inf')
for t in [0.0,0.25,0.5,0.75,1.0]:
    stu_t=build_student(cascade_prime,t_interp=t,cascade_end=cascade_mean_J)
    v_t=eval_val(stu_t,n=30)
    print(f"    t={t:.2f}: val={v_t:.4f}")
    if v_t<best_v: best_v=v_t; best_t=t
print(f"  Best t={best_t:.2f}, val={best_v:.4f}")

vF,ckF=run(lambda: build_student(cascade_prime,t_interp=best_t,
                                   cascade_end=cascade_mean_J),
           f"F-Geodesic-t{best_t:.2f}+200CE")

print(f"\n{'='*65}")
print("  DIRECT ENDPOINT RESULTS")
print("="*65)
print(f"\n  ZERO-SHOT COMPARISON (no CE steps):")
print(f"    Baseline cascade:           val=3.54  (known)")
print(f"    B (mean teacher J):         val={ckB.get(0,'?'):.4f}")
print(f"    C (J14 endpoint):           val={ckC.get(0,'?'):.4f}")
print(f"    D (teacher WK direct):      val={ckD.get(0,'?'):.4f}")
print(f"    E (geodesic t={best_t:.2f}):        val={best_v:.4f}")
print(f"    Teacher:                    val={val_teacher:.4f}")

print(f"\n  CONVERGENCE WITH 200CE:")
print(f"  {'step':>6}  {'A-Prime':>8}  {'B-MeanJ':>8}  {'C-J14':>7}  "
      f"{'D-WK':>7}  {'F-Geo':>7}")
for s in [0,10,25,50,100,150,200]:
    row=f"  {s:>6}"
    for ck,nm in [(ckA,'A'),(ckB,'B'),(ckC,'C'),(ckD,'D'),(ckF,'F')]:
        v=ck.get(s)
        row+=f"  {v:>7.4f}" if v else f"  {'---':>7}"
    print(row)

print(f"""
  FINAL:
    Teacher:                val={val_teacher:.4f}
    A (Prime+CE200):        val={vA:.4f}
    B (MeanJ+CE200):        val={vB:.4f}  diff={vA-vB:+.4f}
    C (J14+CE200):          val={vC:.4f}  diff={vA-vC:+.4f}
    D (TeachWK+CE200):      val={vD:.4f}  diff={vA-vD:+.4f}
    F (Geodesic+CE200):     val={vF:.4f}  diff={vA-vF:+.4f}

  THE DECISIVE QUESTION:
    IF D zero-shot ~ teacher val (0.25):
      Copying teacher WK directly gives the endpoint.
      The 200 CE steps were finding something already known.
      The student IS the teacher at the attractor layer.
      
    IF D zero-shot ~ 3.5 but D+CE200 < A+CE200:
      The teacher WK is a better starting point but still
      needs corpus adaptation. The corpus-specific Stokes
      coefficients cannot be bypassed even with the true endpoint.

    IF all zero-shots ~ 3.5:
      The endpoint alone is not sufficient.
      The path (the 200 CE steps) is computing something
      the endpoint does not contain: the corpus-specific
      embedding co-adaptation. The MC element is known
      but the embedding must be adapted to this specific corpus.
      The 200 steps are embedding adaptation, not MC search.
""")
