#!/usr/bin/env python3
"""
Serre-Constrained Maurer-Cartan Newton Solver
===============================================
The standard MC solver finds alpha* anti-correlated with dJ14 (corr=-0.18).
This is "correct algebra, wrong direction" — it satisfies the L3-MC equation
but is gauge-equivalent to a representative orthogonal to the Serre cascade.

Fix: constrain the Newton search to the tangent space of the Serre operators.

LOSS = ||MC(alpha)||^2 + lambda * ||alpha - P_Serre(alpha)||^2

where P_Serre projects alpha onto the subspace spanned by the Serre cascade
{ad(J14)^l(J14+l)}_{l=1}^6.

PIPELINE:
  A: Prime paths -> build quiver backbone (deformation filter)
  B: Initialize Newton from weighted mean of prime Jacobians (not log M_fwd)
  C: Serre-constrained Newton (50 steps)
  D: Short CE fine-tune with frozen Jacobian backbone (50 steps)

Compare against:
  Baseline: Serre cascade + 200 CE
  Prime:    Prime cascade + 125 CE
  Prime-MC: Prime init + Serre-constrained Newton + 100 CE
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import logm as scipy_logm
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14; DEFORM_THRESHOLD=1.0

print(f"\n{'='*65}")
print(f"  SERRE-CONSTRAINED MC SOLVER")
print(f"  Prime init + topological constraint + short fine-tune")
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

def clr(s,total,warmup=30):
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
def mu4(a,b,c,d):
    return -(comm(l3(a,b,c),d)-l3(comm(a,b),c,d)
            +l3(a,comm(b,c),d)-l3(a,b,comm(c,d))+comm(a,l3(b,c,d)))
def mu5(a,b,c,d,e):
    return -(l3(l3(a,b,c),d,e)-l3(a,l3(b,c,d),e)+l3(a,b,l3(c,d,e))
            +comm(mu4(a,b,c,d),e)+comm(a,mu4(b,c,d,e))
            -mu4(comm(a,b),c,d,e)+mu4(a,b,c,comm(d,e)))
def mu6(a,b,c,d,e,f):
    m5ab=mu5(a,b,c,d,e); m5bc=mu5(b,c,d,e,f)
    m4ab=mu4(a,b,c,d); m4bc=mu4(b,c,d,e); m4cd=mu4(c,d,e,f)
    m3ab=l3(a,b,c); m3bc=l3(b,c,d); m3cd=l3(c,d,e); m3de=l3(d,e,f)
    return -(comm(m5ab,f)-comm(a,m5bc)
            +l3(m4ab,e,f)-l3(a,m4bc,f)+l3(a,b,m4cd)
            +mu4(m3ab,d,e,f)-mu4(a,m3bc,e,f)
            +mu4(a,b,m3cd,f)-mu4(a,b,c,m3de))
def N(A): return float(np.linalg.norm(A))
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
    for pg in opt.param_groups: pg['lr']=clr(step,300,100)
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
    if step%100==0:
        teacher.eval()
        with torch.no_grad():
            vl=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
        teacher.train()
teacher.eval()
val_teacher=eval_val(teacher)
print(f"  Teacher val={val_teacher:.4f}\n")

# Extract Jacobians
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
J14=Js[L_ATT]; U14=Us[L_ATT]; dJ14=J14-np.eye(ma)
print(f"  Done. ma={ma}\n")

# Serre cascade
cascade_serre=[]
for l in range(1,N_STU+1):
    C=J14.copy()
    for _ in range(l): C=J14@C-C@J14
    n=N(C); cascade_serre.append(C/max(n,1e-8))

# ════════════════════════════════════════════════════
# PART A: PRIME LIBRARY
# ════════════════════════════════════════════════════
print("="*65)
print("PART A: PRIME LIBRARY")
print("  Filter: ||delta J_l|| < 1.0  (deformation criterion)")
print("="*65)

deform_norms=[N(Js[l]-np.eye(ma)) for l in range(N_LAYERS_T)]
prime_layers=[l for l in range(N_LAYERS_T) if deform_norms[l]<DEFORM_THRESHOLD]
print(f"\n  Prime layers: {prime_layers}")

# Restrict to attractor basin: L8-L20, deform < 0.75
# (confirmed prime paths from verify_mu4.py are in L11-L18)
att_basin=[l for l in prime_layers if 8<=l<=20 and deform_norms[l]<0.75]
print(f"  Attractor basin (L8-L20, ||dJ||<0.75): {att_basin}")

combos=list(itertools.combinations(att_basin,6))
print(f"  Scoring {len(combos)} 6-tuples in attractor basin...")
scored=sorted([(c,N(mu6(*[Js[i] for i in c]))) for c in combos],key=lambda x:-x[1])
top_combos=[c for c,_ in scored[:N_STU]]

print(f"\n  Top prime paths (attractor basin):")
for i,(combo,w) in enumerate(scored[:6]):
    print(f"  [{i+1}] {combo}  mu6={w:.5f}  deform_mean={np.mean([deform_norms[l] for l in combo]):.3f}")

# Prime cascade from confirmed attractor sequences
cascade_prime=[mu6(*[Js[i] for i in c])/max(N(mu6(*[Js[i] for i in c])),1e-8)
               for c,_ in scored[:N_STU]]

# Prime init: weighted mean of ATTRACTOR BASIN Jacobians only
att_Js=[Js[l] for l in att_basin]
weights=[1.0/max(deform_norms[l],1e-8) for l in att_basin]
w_sum=sum(weights)
alpha_prime_init=sum(w*J for w,J in zip(weights,att_Js))/w_sum - np.eye(ma)
print(f"\n  Prime init alpha: ||alpha|| = {N(alpha_prime_init):.4f}")
print(f"  corr(alpha_prime, dJ14) = "
      f"{float(np.corrcoef(alpha_prime_init.flatten(),dJ14.flatten())[0,1]):.4f}")

# ════════════════════════════════════════════════════
# PART B: SERRE-CONSTRAINED MC SOLVER
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART B: SERRE-CONSTRAINED MC NEWTON SOLVER")
print("  L(alpha) = ||MC(alpha)||^2 + lambda * ||alpha - P_Serre(alpha)||^2")
print("="*65)

# Build Serre projection matrix P_Serre
# Serre cascade spans a subspace of the space of mxm matrices
# P_Serre(alpha) = projection of alpha onto span{S_1,...,S_6}
serre_basis=np.stack([s.flatten() for s in cascade_serre],axis=1)  # (m^2, 6)
# Orthonormalize
Q_serre,_=np.linalg.qr(serre_basis)  # (m^2, 6) orthonormal
Q_serre=Q_serre[:,:min(6,Q_serre.shape[1])]

def project_serre(alpha):
    """Project alpha onto the Serre cascade subspace."""
    v=alpha.flatten()
    coeffs=Q_serre.T@v
    return (Q_serre@coeffs).reshape(ma,ma)

def mc_residual(alpha):
    r=comm(dJ14,alpha)+(1/6)*l3(alpha,alpha,alpha)
    return r,N(r)

def constrained_loss(alpha,lam):
    mc_vec,mc_norm=mc_residual(alpha)
    proj=project_serre(alpha)
    constraint_vec=alpha-proj
    constraint_norm=N(constraint_vec)
    return mc_vec,mc_norm,constraint_vec,constraint_norm

print(f"\n  Serre subspace rank: {Q_serre.shape[1]}")

# Test multiple lambda values
print(f"\n  Lambda sweep (constraint strength):")
print(f"  {'lambda':>8}  {'MC_res':>8}  {'serre_align':>12}  {'corr_dJ14':>11}")
print("  "+"-"*44)

best_alpha=None; best_score=float('inf')
for lam in [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]:
    # Initialize from prime Jacobians (not monodromy log)
    alpha=alpha_prime_init.copy()
    lr_mc=0.005

    for it in range(300):
        mc_vec,mc_norm,constr_vec,constr_norm=constrained_loss(alpha,lam)
        # Gradient of total loss
        grad_mc=2*comm(dJ14+alpha,mc_vec)
        grad_constr=2*constr_vec if lam>0 else 0
        grad=grad_mc+lam*grad_constr
        alpha=alpha-lr_mc*grad

    mc_vec,mc_norm,_,_=constrained_loss(alpha,lam)
    proj=project_serre(alpha)
    serre_align=float(np.sum((alpha/max(N(alpha),1e-8))*(proj/max(N(proj),1e-8))))
    corr_dJ14=float(np.corrcoef(alpha.flatten(),dJ14.flatten())[0,1])

    # Score: want low MC residual AND good serre alignment
    score=mc_norm-0.5*serre_align  # lower is better
    marker=" ← best" if score<best_score else ""
    if score<best_score:
        best_score=score; best_alpha=alpha.copy(); best_lam=lam

    print(f"  {lam:>8.2f}  {mc_norm:>8.4f}  {serre_align:>12.4f}  {corr_dJ14:>11.4f}{marker}")

print(f"\n  Best lambda={best_lam}")
_,best_mc_res,_,_=constrained_loss(best_alpha,best_lam)
best_corr=float(np.corrcoef(best_alpha.flatten(),dJ14.flatten())[0,1])
print(f"  MC residual: {best_mc_res:.4f}")
print(f"  corr(alpha*, dJ14): {best_corr:.4f}")

# Build MC cascade from best_alpha
# Use Serre-modulated alpha as cascade levels
cascade_mc_serre=[]
for l in range(N_STU):
    # Level l: best_alpha modulated by Serre at level l
    serre_l=cascade_serre[l]
    # Project alpha onto Serre direction at this level, keep residual
    proj_l=float(np.sum(best_alpha.flatten()*serre_l.flatten()))
    C=best_alpha+proj_l*serre_l  # alpha + alignment boost
    n=N(C); cascade_mc_serre.append(C/max(n,1e-8))

# ════════════════════════════════════════════════════
# PART C: SINGULAR VALUE ALIGNMENT TEST
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART C: SINGULAR VALUE ALIGNMENT")
print("  Compare assembled Jacobians vs teacher at convergence")
print("="*65)

def sv_alignment(C1,C2,k=4):
    """Subspace alignment between top-k singular vectors."""
    U1,_,_=np.linalg.svd(C1); U1=U1[:,:k]
    U2,_,_=np.linalg.svd(C2); U2=U2[:,:k]
    sv=np.linalg.svd(U1.T@U2,compute_uv=False)
    return float(np.mean(np.clip(sv,0,1)))

print(f"\n  Cascade level vs teacher J14 subspace alignment (top-4 SV):")
print(f"  {'level':>6}  {'serre_align':>12}  {'prime_align':>12}  {'mc_serre_align':>16}")
print("  "+"-"*50)
for l in range(N_STU):
    sa=sv_alignment(cascade_serre[l],dJ14)
    pa=sv_alignment(cascade_prime[l],dJ14)
    ma_=sv_alignment(cascade_mc_serre[l],dJ14)
    print(f"  {l+1:>6}  {sa:>12.4f}  {pa:>12.4f}  {ma_:>16.4f}")

# ════════════════════════════════════════════════════
# PART D: STUDENT EXPERIMENTS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART D: STUDENT COMPARISON")
print("  A: Serre + 200CE  (baseline)")
print("  B: Prime + 125CE  (phase 2 only)")
print("  C: Prime-MC + 100CE  (Serre-constrained alpha + short CE)")
print("="*65)

def run_student(cascade,label,steps=200,freeze_after=None):
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

    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    checkpoints={}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps,30)
        if freeze_after and step>freeze_after:
            # Freeze Jacobian backbone blocks, train only embeddings
            for b in stu.blocks:
                b.attn.WK.requires_grad_(False)
                b.attn.WQ.requires_grad_(False)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); checkpoints[step]=v
            b_t="✓" if v<val_teacher else " "
            print(f"  [{label}] step {step:>4}  val={v:.4f} {b_t}")
    return eval_val(stu),checkpoints

vA,ckA=run_student(cascade_serre,"A-Serre+200CE",200)
vB,ckB=run_student(cascade_prime,"B-Prime+125CE",125)
vC,ckC=run_student(cascade_mc_serre,"C-PrimeMC+100CE",100)

print(f"\n{'='*65}")
print(f"  SERRE-CONSTRAINED MC RESULTS")
print("="*65)

print(f"\n  CONVERGENCE TABLE:")
print(f"  {'step':>6}  {'A-Serre':>9}  {'B-Prime':>9}  {'C-Prime-MC':>11}")
for s in [25,50,75,100,125,150,200]:
    vAs=ckA.get(s); vBs=ckB.get(s); vCs=ckC.get(s)
    row=f"  {s:>6}  "
    row+=f"{vAs:>9.4f}  " if vAs else f"{'---':>9}  "
    row+=f"{vBs:>9.4f}  " if vBs else f"{'---':>9}  "
    row+=f"{vCs:>11.4f}" if vCs else f"{'---':>11}"
    print(row)

teacher_ls=300*N_LAYERS_T
print(f"""
  FINAL RESULTS:
    Teacher:             val={val_teacher:.4f}  ({teacher_ls} layer-steps)
    A: Serre+200CE:      val={vA:.4f}  (1200 layer-steps)
    B: Prime+125CE:      val={vB:.4f}  (750 layer-steps)
    C: Prime-MC+100CE:   val={vC:.4f}  (600 layer-steps)

  SERRE CONSTRAINT (best lambda={best_lam}):
    MC residual: {best_mc_res:.4f}
    corr(alpha*, dJ14): {best_corr:.4f}
    (vs unconstrained: corr=-0.18)

  KEY QUESTION: Does C beat A at fewer layer-steps?
    If C (600 steps) beats A (1200 steps): the Serre-constrained
    MC solver reduces Phase 2 by >50% — gradient-free MC integration
    replaces ~100 CE steps.
    
    If C ≈ A: the constraint helps alignment but Phase 2
    is still irreducibly data-dependent regardless of MC solver quality.
""")
