#!/usr/bin/env python3
"""
Stokes Chamber Projection
==========================
The complex_grassmannian.py showed:
  - Trajectory lives on unit sphere |z| ~ 1.03 (not spiraling)
  - arg(z_mid) settles near theta* ~ 0.66*pi at convergence
  - 200 steps = libration damping from theta(0) to theta*

STOKES PROJECTION HYPOTHESIS:
  If we initialize the student WITH the correct phase theta*,
  the libration starts at equilibrium.
  Damping time -> 0 steps.
  Zero-shot val should be near teacher val (~0.25).

CONSTRUCTION:
  1. Compute teacher's complex Grassmannian coordinate at L_ATT:
       z_teacher = sv1(J_14) * exp(i * theta_14)
     where theta_14 = accumulated angle from L0 to L14.

  2. The target phase: theta* = arg(z_teacher_mid)
     (the phase where the student should start, not finish)

  3. Phase-correct the cascade operators:
     For each cascade level l:
       S_l_phased = S_l * exp(i * (theta* - theta_l(cascade)))
     Since S_l is real, this means rotating the operator by
     the phase correction angle: R(theta* - theta_l) @ S_l @ R(theta* - theta_l)^T
     where R(phi) is a rotation matrix in the dominant SV subspace.

  4. Initialize student with phase-corrected cascade.
     Measure zero-shot val and convergence.

IHARA RADIUS CONNECTION:
  If student Ihara radius -> teacher value (33.29) AS the student
  converges, then rho is the correct convergence criterion.
  The Stokes projection would make rho_student(0) ~ rho_teacher(0)
  by starting in the correct chamber.

FALSIFICATION:
  If zero-shot val of Stokes-projected student ~ 0.25:
    The libration explains the 200 steps. Phase injection bypasses them.
    This is the one-shot computation.
  If zero-shot val ~ 3.5 (same as before):
    The phase of z_mid is not the relevant degree of freedom.
    The 200 steps compute something the phase cannot capture.
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  STOKES CHAMBER PROJECTION")
print(f"  Phase injection to eliminate libration damping")
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

def mu6_op(js):
    a,b,c,d,e,f=js
    def mu4(x,y,z,w):
        return -(comm(l3(x,y,z),w)-l3(comm(x,y),z,w)+l3(x,comm(y,z),w)
                -l3(x,y,comm(z,w))+comm(x,l3(y,z,w)))
    def mu5(x,y,z,w,v):
        return -(l3(l3(x,y,z),w,v)-l3(x,l3(y,z,w),v)+l3(x,y,l3(z,w,v))
                +comm(mu4(x,y,z,w),v)+comm(x,mu4(y,z,w,v))
                -mu4(comm(x,y),z,w,v)+mu4(x,y,z,comm(w,v)))
    m5ab=mu5(a,b,c,d,e); m5bc=mu5(b,c,d,e,f)
    m4ab=mu4(a,b,c,d); m4bc=mu4(b,c,d,e); m4cd=mu4(c,d,e,f)
    m3ab=l3(a,b,c); m3bc=l3(b,c,d); m3cd=l3(c,d,e); m3de=l3(d,e,f)
    return -(comm(m5ab,f)-comm(a,m5bc)+l3(m4ab,e,f)-l3(a,m4bc,f)
            +l3(a,b,m4cd)+mu4(m3ab,d,e,f)-mu4(a,m3bc,e,f)
            +mu4(a,b,m3cd,f)-mu4(a,b,c,m3de))

def complex_z(J):
    """Map Jacobian to complex Grassmannian coordinate."""
    U,sv,_=np.linalg.svd(J)
    return sv[0], U[:,0]  # dominant SV and direction

def rotation_matrix_2d(phi, u1, u2, size):
    """Rotation by phi in the plane spanned by u1, u2."""
    R=np.eye(size)
    # Gram-Schmidt
    u1=u1/max(np.linalg.norm(u1),1e-8)
    u2=u2-np.dot(u2,u1)*u1; u2=u2/max(np.linalg.norm(u2),1e-8)
    R += (math.cos(phi)-1)*(np.outer(u1,u1)+np.outer(u2,u2))
    R += math.sin(phi)*(np.outer(u2,u1)-np.outer(u1,u2))
    return R

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

# ════════════════════════════════════════════════════
# STEP 1: TEACHER STOKES PHASE
# ════════════════════════════════════════════════════
print("="*65)
print("STEP 1: TEACHER STOKES PHASE COMPUTATION")
print("  theta* = accumulated arg(z_l) at attractor L14")
print("="*65)

# Compute accumulated angle up to L_ATT
theta_acc=0.0
prev_u1=None
layer_phases=[]
for l in range(N_LAYERS_T):
    sv1,u1=complex_z(Js[l])
    if prev_u1 is not None:
        cos_t=float(np.clip(prev_u1@u1,-1,1))
        dtheta=math.acos(abs(cos_t))
        if prev_u1@u1<0: dtheta=-dtheta
        theta_acc+=dtheta
    layer_phases.append((l,sv1,theta_acc,u1))
    prev_u1=u1

# Target phase = teacher's accumulated angle at L_ATT
theta_star=layer_phases[L_ATT][2]
sv1_att=layer_phases[L_ATT][1]
u1_att=layer_phases[L_ATT][3]

# Second dominant direction at L_ATT (for rotation plane)
U_att,sv_att,_=np.linalg.svd(Js[L_ATT])
u2_att=U_att[:,1]  # sub-dominant direction

print(f"\n  Teacher phase at L{L_ATT}: theta* = {theta_star:.4f} rad = {theta_star/math.pi:.4f} pi")
print(f"  Teacher sv1 at L{L_ATT}: {sv1_att:.4f}")
print(f"\n  Layer-by-layer accumulated phase:")
print(f"  {'L':>3}  {'sv1':>7}  {'theta/pi':>10}  {'note'}")
print("  "+"-"*35)
for l,sv1,theta,u1 in layer_phases[::3]:
    note=" <-- ATT" if l==L_ATT else ""
    print(f"  L{l:>2}  {sv1:>7.3f}  {theta/math.pi:>10.4f}{note}")

# Prime paths and cascades
att_basin=[l for l in range(8,21) if N(Js[l]-np.eye(ma))<0.75]
combos=list(itertools.combinations(att_basin,6))
scored=sorted([(c,N(mu6_op([Js[i] for i in c]))) for c in combos],key=lambda x:-x[1])
prime_paths=[c for c,_ in scored[:N_STU]]

cascade_serre=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    cascade_serre.append(C/max(N(C),1e-8))

cascade_prime=[mu6_op([Js[i] for i in c])/max(N(mu6_op([Js[i] for i in c])),1e-8)
               for c in prime_paths]

# ════════════════════════════════════════════════════
# STEP 2: PHASE-CORRECTED CASCADE
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 2: STOKES PHASE INJECTION")
print(f"  Rotate each cascade operator by (theta* - theta_l)")
print(f"  to start the student at the equilibrium phase")
print("="*65)

def phase_correct_cascade(cascade, theta_star, u1_att, u2_att, Js):
    """
    Rotate each cascade operator S_l by the Stokes phase correction.
    
    The cascade operator S_l has an implicit phase theta_l coming from
    the Jacobian chain used to construct it. The student starts with
    this phase but needs to end at theta*.
    
    Phase correction: rotate S_l by (theta* - theta_l) in the
    dominant SV subspace of J_14.
    """
    corrected=[]
    for l,S in enumerate(cascade):
        # Phase of this cascade level (from the Jacobian it's built from)
        teacher_l=min(L_ATT+l+1, N_LAYERS_T-1)
        _,theta_l,_=layer_phases[teacher_l][1],layer_phases[teacher_l][2],layer_phases[teacher_l][3]

        # Phase correction angle
        dphi=theta_star-theta_l

        # Rotation matrix in (u1_att, u2_att) plane
        R=rotation_matrix_2d(dphi, u1_att, u2_att, ma)

        # Rotate operator: S_corrected = R @ S @ R^T
        S_corr=R@S@R.T
        corrected.append(S_corr/max(N(S_corr),1e-8))

        print(f"  Level {l+1}: teacher_l=L{teacher_l}  "
              f"theta_l={theta_l/math.pi:.3f}pi  "
              f"dphi={dphi/math.pi:.3f}pi")
    return corrected

print(f"\n  Phase corrections for Serre cascade:")
cascade_serre_phased=phase_correct_cascade(
    cascade_serre, theta_star, u1_att, u2_att, Js)

print(f"\n  Phase corrections for Prime cascade:")
cascade_prime_phased=phase_correct_cascade(
    cascade_prime, theta_star, u1_att, u2_att, Js)

# ════════════════════════════════════════════════════
# STEP 3: STUDENT EXPERIMENTS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 3: STUDENT EXPERIMENTS")
print("  A: Serre + 200CE (baseline)")
print("  B: Prime + 200CE (best confirmed)")
print("  C: Serre + Stokes phase + 200CE")
print("  D: Prime + Stokes phase + 200CE")
print("  E: Prime + Stokes phase + 0CE (zero-shot target)")
print("="*65)

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

def run(cascade,label,steps=200):
    stu=build_student(cascade)
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
        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

vA,ckA=run(cascade_serre,"A-Serre-std")
vB,ckB=run(cascade_prime,"B-Prime-std")
vC,ckC=run(cascade_serre_phased,"C-Serre-Stokes")
vD,ckD=run(cascade_prime_phased,"D-Prime-Stokes")
vE,ckE=run(cascade_prime_phased,"E-Prime-Stokes-0CE",steps=0)

print(f"\n{'='*65}")
print("  STOKES PROJECTION RESULTS")
print("="*65)
print(f"\n  PHASE ANALYSIS:")
print(f"  theta* (teacher at L{L_ATT}) = {theta_star:.4f} rad = {theta_star/math.pi:.4f} pi")
print(f"  Observed theta* at convergence ~ 0.66 pi (from complex_grassmannian.py)")
print(f"  Match: {'YES' if abs(theta_star/math.pi - 0.66) < 0.1 else 'NO — different phase'}")

print(f"\n  CONVERGENCE:")
print(f"  {'step':>6}  {'A-Serre':>8}  {'B-Prime':>8}  {'C-S+Stok':>9}  {'D-P+Stok':>9}")
for s in [0,25,50,75,100,125,150,200]:
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
    A (Serre std):         val={vA:.4f}
    B (Prime std):         val={vB:.4f}
    C (Serre+Stokes):      val={vC:.4f}  diff={vA-vC:+.4f}
    D (Prime+Stokes):      val={vD:.4f}  diff={vA-vD:+.4f}
    E (Prime+Stokes 0CE):  val={vE:.4f}  (ZERO SHOT TARGET)

  THE TEST:
    If E ~ 0.25 (teacher val): Stokes projection works.
      The 200 steps are librational damping. Phase injection
      starts the student at the equilibrium chamber.
      One geometric projection replaces 200 CE steps.

    If E ~ 3.5 (random val): Phase is not the degree of freedom.
      The libration settles a different quantity than the phase.
      The 200 steps compute Stokes coefficients that require
      the actual corpus data — the phase is architecture-dependent
      but the coefficients are corpus-dependent.
      Final confirmation: 200 steps are truly irreducible.

    If C < A or D < B (faster convergence with Stokes init):
      Even if E fails, the Stokes phase reduces damping time.
      Partial speedup: start closer to equilibrium, fewer steps.
""")
