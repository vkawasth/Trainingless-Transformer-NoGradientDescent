#!/usr/bin/env python3
"""
Spectral Initialization — Ihara Radius Targeting
==================================================
The cascade initializes operators with ||S_l|| = 1 (unit norm).
The teacher's attractor Jacobians have ||J_l|| ~ 1.0 but the
prime path PRODUCT norms give spectral radius ~ 33.

The student cascade needs spectral rescaling to match the teacher.

Target: rho_teacher = 33.29 on teacher prime paths.
Method: scale each cascade operator by factor alpha such that
  the teacher transfer matrix, evaluated with scaled student ops,
  gives spectral radius ~ 33.

Since T_ij = mean ||J_l|| for l in overlap(P_i, P_j),
and teacher gives rho=33 with ||J_l||~1.0,
we need student ||J_l|| scaled to match teacher norms at
the corresponding prime path layers.

SPECTRAL SCALE FACTOR:
  Teacher: ||J_l||_attractor ~ 1.0 (from Jacobian data)
  Cascade: ||S_l|| = 1.0 (normalized)
  But the teacher's J_l at prime path layers has specific
  off-diagonal structure that gives rho=33.

  The simplest approach: scale cascade operators by
  alpha = (rho_teacher / rho_cascade_on_teacher_paths)^(1/N)
  where N = number of prime path layers.

THREE EXPERIMENTS:
  A: Standard cascade (baseline)
  B: Cascade scaled by alpha to match teacher Ihara radius
  C: Cascade with teacher J_l norms injected directly
     (replace cascade with normalized teacher Jacobians)
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  SPECTRAL INITIALIZATION")
print(f"  Ihara radius targeting via cascade rescaling")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab); train_t=torch.tensor(train_ids,dtype=torch.long)
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
        self.ln_f=nn.LayerNorm(d); self.head=nn.Linear(d,VOCAB,bias=False)
        self.head.weight=self.te.weight
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

def ihara_on_teacher_paths(Js_stu, prime_paths, Js_teacher):
    """
    Compute Ihara radius using teacher prime path structure
    but with STUDENT Jacobian norms.
    T_ij = mean ||J_l^student|| for l in overlap(P_i, P_j)
    where P_i, P_j are TEACHER prime paths (layer indices).
    But student only has N_STU blocks -> map teacher layer l
    to student block: block_l = min(max(0, l - L_ATT + N_STU//2), N_STU-1)
    """
    n=len(prime_paths); T=np.zeros((n,n))
    for i,p_i in enumerate(prime_paths):
        for j,p_j in enumerate(prime_paths):
            if i!=j:
                overlap=sorted(set(p_i)&set(p_j))
                if overlap:
                    # Map teacher layers to student blocks
                    stu_blocks=[min(max(0,l-L_ATT+N_STU//2),N_STU-1)
                                for l in overlap]
                    # Use student Jacobian norms at those blocks
                    norms=[N(Js_stu[b]) for b in stu_blocks]
                    T[i,j]=np.mean(norms)
    ev=np.linalg.eigvals(T)
    return float(np.max(np.abs(ev))), T

def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)

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

# Extract Jacobians
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
with torch.no_grad():
    hs=teacher.hidden_states_all(x_ref); hs=[h[0] for h in hs]
Js=[]; Us=[]
for l in range(N_LAYERS_T):
    J,U=layer_jac(teacher.blocks[l],hs[l],pos,m)
    Js.append(J); Us.append(U)
    if ma is None: ma=J.shape[0]
J14=Js[L_ATT]; U14=Us[L_ATT]
print(f"  ma={ma}")

# Prime paths
att_basin=[l for l in range(8,21) if N(Js[l]-np.eye(ma))<0.75]
combos=list(itertools.combinations(att_basin,6))
scored=sorted([(c,N(mu6(*[Js[i] for i in c]))) for c in combos],key=lambda x:-x[1])
prime_paths=[c for c,_ in scored[:N_STU]]

# Teacher prime path Jacobian norms
print(f"\n  Teacher prime path Jacobian norms:")
for pi,pp in enumerate(prime_paths):
    norms=[N(Js[l]) for l in pp]
    print(f"  P{pi+1} {pp}: {[f'{n:.3f}' for n in norms]}")

# Teacher Ihara radius (self)
n_pp=len(prime_paths); T_teach=np.zeros((n_pp,n_pp))
for i,p_i in enumerate(prime_paths):
    for j,p_j in enumerate(prime_paths):
        if i!=j:
            overlap=sorted(set(p_i)&set(p_j))
            if overlap:
                T_teach[i,j]=np.mean([N(Js[l]) for l in overlap])
ev_teach=np.linalg.eigvals(T_teach)
rho_teacher=float(np.max(np.abs(ev_teach)))
print(f"\n  Teacher Ihara radius: {rho_teacher:.4f}")
print(f"  Teacher T matrix:\n{T_teach.round(3)}")

# Cascades
cascade_serre=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    cascade_serre.append(C/max(N(C),1e-8))

cascade_prime=[mu6(*[Js[i] for i in c])/max(N(mu6(*[Js[i] for i in c])),1e-8)
               for c in prime_paths]

# ════════════════════════════════════════════════════
# SPECTRAL SCALE FACTOR
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("SPECTRAL SCALE FACTOR COMPUTATION")
print("  Find alpha to make student Ihara radius = teacher's")
print("="*65)

# Cascade operators have ||S_l|| = 1.0 by normalization
# Teacher J_l at attractor layers also have ||J_l|| ~ 1.0
# But teacher T matrix gives rho=33 while unit-norm cascade would give rho~5
# The difference: teacher J_l norms at PRIME PATH layers (not just L14)
mean_teacher_norm=np.mean([N(Js[l]) for pp in prime_paths for l in pp])
print(f"\n  Mean teacher ||J_l|| at prime path layers: {mean_teacher_norm:.4f}")
print(f"  Cascade operator norm: 1.0000 (by construction)")

# Approximate scale factor: if T_ij ~ alpha * mean_norm * overlap_size
# and rho ~ (N-1) * alpha * mean_norm (for near-uniform matrix)
# target: (N-1) * alpha * mean_norm = rho_teacher
alpha = rho_teacher / ((N_STU-1) * mean_teacher_norm)
print(f"\n  Approximate scale factor alpha = rho_teacher / ((N-1)*mean_norm)")
print(f"  alpha = {rho_teacher:.4f} / ({N_STU-1} * {mean_teacher_norm:.4f}) = {alpha:.4f}")

# Verify: what rho do we get with scaled cascade?
cascade_scaled=[alpha*S for S in cascade_prime]
Js_scaled=[alpha*np.eye(ma) for _ in range(N_STU)]  # approx: scaled identity
rho_scaled,T_scaled=ihara_on_teacher_paths(Js_scaled,prime_paths,Js)
print(f"  Predicted rho with alpha={alpha:.3f}: {rho_scaled:.4f}")
print(f"  Target rho: {rho_teacher:.4f}")

# Grid search for exact alpha
print(f"\n  Grid search for exact alpha:")
for a in [0.5,1.0,2.0,5.0,alpha,10.0,20.0]:
    Js_a=[a*np.eye(ma) for _ in range(N_STU)]
    r,_=ihara_on_teacher_paths(Js_a,prime_paths,Js)
    print(f"  alpha={a:.2f}: rho={r:.4f}  {'<--target' if abs(r-rho_teacher)<2 else ''}")

# ════════════════════════════════════════════════════
# STUDENT EXPERIMENTS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STUDENT EXPERIMENTS")
print(f"  A: Prime cascade (||S_l||=1, baseline)")
print(f"  B: Prime cascade * alpha={alpha:.3f} (Ihara-targeted scale)")
print(f"  C: Direct teacher J_l injection (at prime path layers)")
print(f"  D: Prime cascade * alpha + 200CE")
print(f"  E: Direct J_l injection + 200CE")
print("="*65)

def build_student(cascade, scale=1.0):
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight)
        stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            W_d=lift_to_d(cascade[l]*scale,U14,scale=0.01)
            W_t=torch.tensor(W_d,dtype=torch.float32)
            stu.blocks[l].attn.WK.weight.copy_(W_t)
            stu.blocks[l].attn.WQ.weight.copy_(W_t.T)
            stu.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            stu.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            stu.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            stu.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            stu.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
    return stu

def run(cascade,label,steps=200,scale=1.0):
    stu=build_student(cascade,scale)
    v0=eval_val(stu,n=20)
    # Measure student Ihara using teacher path structure
    torch.manual_seed(0)
    x_s,_=get_batch('val'); x_s=x_s[0:1]
    with torch.no_grad(): hs_s=stu.hidden_states_all(x_s); hs_s=[h[0] for h in hs_s]
    Js_s=[]
    for bl in range(N_STU):
        J,_=layer_jac(stu.blocks[bl],hs_s[bl],pos,m)
        Js_s.append(J)
    rho_s,_=ihara_on_teacher_paths(Js_s,prime_paths,Js)
    print(f"\n  [{label}] zero-shot={v0:.4f}  rho_student={rho_s:.4f}  "
          f"(teacher={rho_teacher:.4f})")
    if steps==0: return v0,{0:v0}
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={0:v0}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,100,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

# Direct teacher J_l injection: use teacher Jacobians at prime path layers
# as cascade operators (the target we want to match)
cascade_teacher_direct=[]
for pp in prime_paths:
    # Average J_l at the 6 prime path layers for this path
    J_avg=np.mean([Js[l] for l in pp],axis=0)
    n=N(J_avg); cascade_teacher_direct.append(J_avg/max(n,1e-8))

vA,ckA=run(cascade_prime,"A-Prime-std",scale=1.0)
vB,ckB=run(cascade_prime,f"B-Prime-alpha{alpha:.2f}",scale=alpha)
vC,ckC=run(cascade_teacher_direct,"C-TeacherJ-direct",scale=1.0)
vD,ckD=run(cascade_prime,f"D-Prime-alpha{alpha:.2f}+200CE",scale=alpha,steps=200)
vE,ckE=run(cascade_teacher_direct,"E-TeacherJ+200CE",scale=1.0,steps=200)

print(f"\n{'='*65}")
print("  SPECTRAL INITIALIZATION RESULTS")
print("="*65)
print(f"\n  Teacher Ihara: {rho_teacher:.4f}")
print(f"  Scale factor alpha: {alpha:.4f}")
print(f"\n  CONVERGENCE:")
print(f"  {'step':>6}  {'A-Prime':>8}  {'B-Scaled':>9}  {'C-TeachJ':>9}  {'D-Scl+CE':>9}  {'E-TJ+CE':>8}")
for s in [0,25,50,100,150,200]:
    a=ckA.get(s); b=ckB.get(s); c=ckC.get(s); d=ckD.get(s); e=ckE.get(s)
    row=f"  {s:>6}"
    for v in [a,b,c,d,e]:
        row+=f"  {v:>8.4f}" if v else f"  {'---':>8}"
    print(row)

print(f"""
  FINAL:
    Teacher:              val={val_teacher:.4f}
    A (Prime std):        val={vA:.4f}
    B (Prime*alpha):      val={vB:.4f}  diff={vA-vB:+.4f}
    C (Teacher J direct): val={vC:.4f}  diff={vA-vC:+.4f}
    D (Scaled+CE200):     val={vD:.4f}  diff={vA-vD:+.4f}
    E (TeachJ+CE200):     val={vE:.4f}  diff={vA-vE:+.4f}

  IF B or C significantly better:
    Spectral scale matters — the cascade norm is wrong by factor alpha.
    Ihara-targeted initialization is the correct spectral init.

  IF B < A but C ~ A:
    Scale helps but direction (teacher J) doesn't add info beyond cascade.
    The cascade direction is correct, magnitude was wrong.

  IF all ~ A:
    The cascade norm is already correct (||S_l||=1 is right).
    The Ihara radius of 269 is a mapping artifact as diagnosed.
    The spectral initialization has no measurable effect.
    Final and complete: 200 CE steps are irreducible.
""")
