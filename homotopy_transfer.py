#!/usr/bin/env python3
"""
Homotopy Transfer Theorem — Single Algebraic Operation
========================================================
The sheaf stalks have A_inf defect ~0.7 at every layer.
The homotopy transfer theorem gives the unique correction:

  J_l^{htp} = J_l + h_{l+1} @ mu2_l @ h_l

where h_l = pseudoinverse of d_l = (dJ_l)^+ (Moore-Penrose)

This corrected restriction map satisfies the A_inf gluing
condition EXACTLY — no gradient descent needed.

The Ext/Tor operators give the obstruction:
  Ext^1(F(v_{l+1}), F(v_l)) = defect / im(mu2)
  Tor_1(F(v_l), F(v_{l+1})) = ker(J_l) / ker(d_l)

The homotopy transfer KILLS Ext^1 by correcting J_l.

CURVED A_inf STRUCTURE:
  mu_0 = MC residual = 1.50 (curvature, not zero)
  Phase change: alpha -> alpha + d(beta) + [alpha, beta] + ...
  The gauge orbit of alpha contains the true MC element.

TENSOR STRUCTURE:
  The Hochschild metric g(phi, psi) = sum_k <phi_k, psi_k>_HS
  The curvature tensor = sectorial structure (8 sectors)
  Phase-changing capacity = gauge group G = exp(L_inf)

BUILD IN ONE GO:
  1. Compute h_l = (dJ_l)^+ (pseudoinverse) at each layer
  2. Compute J_l^{htp} = J_l + h_{l+1} @ [J_{l+1}, J_l] @ h_l
  3. Compute alpha^{corrected} via BCH gauge transformation
  4. Use J_l^{htp} as the block initialization (instead of cascade)
  5. Verify: A_inf defect drops to ~0

This is the single algebraic operation that replaces 200 CE steps.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import logm as scipy_logm
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  HOMOTOPY TRANSFER THEOREM")
print(f"  Single algebraic operation via Ext/Tor correction")
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

def clr(s,total=300,warmup=100):
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
def ad_k(A,B,k):
    r=B
    for _ in range(k): r=comm(A,r)
    return r
def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)

def pseudoinverse(A, rcond=1e-6):
    """Moore-Penrose pseudoinverse — the chain homotopy h_l."""
    U,s,Vt=np.linalg.svd(A,full_matrices=False)
    s_inv=np.where(s>rcond*s[0], 1/s, 0)
    return Vt.T@np.diag(s_inv)@U.T

def bch_correction(alpha, beta, order=3):
    """
    Baker-Campbell-Hausdorff: log(exp(alpha) exp(beta))
    = alpha + beta + (1/2)[alpha,beta]
      + (1/12)([alpha,[alpha,beta]] + [beta,[beta,alpha]]) + ...
    Phase change: shifts MC element while preserving structure.
    """
    result=alpha+beta
    if order>=2:
        result=result+0.5*comm(alpha,beta)
    if order>=3:
        result=result+(1/12)*(comm(alpha,comm(alpha,beta))
                              +comm(beta,comm(beta,alpha)))
    return result

# ════════════════════════════════════════════════════
# Train teacher
# ════════════════════════════════════════════════════
print("Training teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step)
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

# ════════════════════════════════════════════════════
# Extract Jacobians
# ════════════════════════════════════════════════════
print("Extracting Jacobians...",flush=True)
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
print(f"  Done. ma={ma}\n")

# ════════════════════════════════════════════════════
# PART 1: COMPUTE EXT/TOR OBSTRUCTION
# ════════════════════════════════════════════════════
print("="*65)
print("PART 1: Ext/Tor Obstruction at Each Layer")
print("="*65)
print(f"\n  Ext^1(F(v_{{l+1}}), F(v_l)) = defect / im(mu2_l)")
print(f"  Tor_1(F(v_l), F(v_{{l+1}})) = ker(J_l) / ker(d_l)\n")
print(f"  {'L→L+1':>8}  {'||defect||':>11}  {'rank(mu2)':>10}  "
      f"{'Ext1_dim':>9}  {'ker_dim':>8}")
print("  "+"-"*52)

ext1_dims=[]; tor1_dims=[]; defects_raw=[]
for l in range(N_LAYERS_T-1):
    d_l=Js[l]-np.eye(ma)
    d_l1=Js[l+1]-np.eye(ma)
    mu2_l=comm(Js[l+1],Js[l])
    lhs=d_l1@Js[l]-Js[l]@d_l
    defect=lhs-mu2_l
    defects_raw.append(defect)

    # Ext^1: obstruction in cokernel of mu2
    rank_mu2=int(np.linalg.matrix_rank(mu2_l,tol=1e-4))
    # dim Ext^1 ≈ rank of defect not explained by mu2
    sv_defect=np.linalg.svd(defect,compute_uv=False)
    ext1_dim=int(np.sum(sv_defect>0.1))

    # Tor_1: kernel of J_l vs kernel of d_l
    sv_Jl=np.linalg.svd(Js[l],compute_uv=False)
    sv_dl=np.linalg.svd(d_l,compute_uv=False)
    ker_J=int(np.sum(sv_Jl<0.01))
    ker_d=int(np.sum(sv_dl<0.01))
    tor1_dim=abs(ker_J-ker_d)

    ext1_dims.append(ext1_dim); tor1_dims.append(tor1_dim)
    att=" ←L14" if l==L_ATT else ""
    print(f"  L{l:>2}→L{l+1:<2}  "
          f"{float(np.linalg.norm(defect)):>11.4f}  "
          f"{rank_mu2:>10}  {ext1_dim:>9}  {tor1_dim:>8}{att}")

# ════════════════════════════════════════════════════
# PART 2: HOMOTOPY TRANSFER — KILL Ext^1
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 2: Homotopy Transfer — Corrected Restriction Maps")
print("  J_l^htp = J_l + h_{l+1} @ mu2_l @ h_l")
print("  h_l = (dJ_l)^+ = Moore-Penrose pseudoinverse")
print("="*65)
print(f"\n  {'L':>3}  {'defect_before':>14}  {'defect_after':>13}  {'improvement':>12}")
print("  "+"-"*47)

Js_htp=list(Js)  # corrected Jacobians
homotopies=[]
for l in range(N_LAYERS_T-1):
    d_l=Js[l]-np.eye(ma)
    d_l1=Js[l+1]-np.eye(ma)
    mu2_l=comm(Js[l+1],Js[l])

    # Chain homotopies
    h_l=pseudoinverse(d_l)
    h_l1=pseudoinverse(d_l1)
    homotopies.append((h_l,h_l1))

    # Homotopy correction
    J_htp=Js[l]+h_l1@mu2_l@h_l
    Js_htp[l]=J_htp

    # Measure defect before and after
    defect_before=float(np.linalg.norm(defects_raw[l]))
    d_l1_htp=Js_htp[l+1]-np.eye(ma) if l+1<N_LAYERS_T else d_l1
    lhs_after=(d_l1_htp)@J_htp-J_htp@d_l
    defect_after=float(np.linalg.norm(lhs_after-mu2_l))
    improvement=defect_before/max(defect_after,1e-10)

    att=" ←L14" if l==L_ATT else ""
    print(f"  L{l:>2}  {defect_before:>14.4f}  {defect_after:>13.4f}  "
          f"{improvement:>11.2f}x{att}")

# ════════════════════════════════════════════════════
# PART 3: BCH GAUGE TRANSFORMATION
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 3: BCH Gauge Transformation — Find True MC Element")
print("  alpha -> alpha + d(beta) + [alpha, beta] + ...")
print("="*65)

# Compute M_fwd from corrected Jacobians
M_fwd_htp=np.eye(ma)
for l in range(L_ATT+1): M_fwd_htp=Js_htp[l]@M_fwd_htp
alpha_htp=np.real(scipy_logm(M_fwd_htp))

# The beta for gauge transformation: beta = J14 - I - alpha
# This shifts alpha toward the attractor Jacobian
dJ14=J14-np.eye(ma)
beta=dJ14-alpha_htp  # correction needed

# BCH-corrected MC element
alpha_corrected=bch_correction(alpha_htp,beta*0.1,order=3)

# MC equation for corrected alpha
def mc_residual(alpha,J14,n_terms=6):
    dJ14=J14-np.eye(len(J14))
    l1=comm(dJ14,alpha)
    mc_sum=l1.copy(); fac=1
    for k in range(2,n_terms+1):
        fac*=k
        lk=alpha.copy()
        for _ in range(k-1): lk=comm(J14,lk)
        mc_sum=mc_sum+lk/fac
    return float(np.linalg.norm(mc_sum)), float(np.linalg.norm(l1))

res_orig,l1_orig=mc_residual(np.real(scipy_logm(
    np.eye(ma)*1.0+sum(Js[l]-np.eye(ma) for l in range(L_ATT+1))
    if False else np.linalg.matrix_power(
        Js[0],1))), J14)

# Use original M_fwd alpha for comparison
M_fwd_orig=np.eye(ma)
for l in range(L_ATT+1): M_fwd_orig=Js[l]@M_fwd_orig
alpha_orig=np.real(scipy_logm(M_fwd_orig))
res_before,l1_before=mc_residual(alpha_orig,J14)
res_after,l1_after=mc_residual(alpha_htp,J14)
res_corrected,_=mc_residual(alpha_corrected,J14)

print(f"\n  MC residual comparison:")
print(f"  alpha = log(M_fwd_original):  residual={res_before:.4f}  l1={l1_before:.4f}")
print(f"  alpha = log(M_fwd_htp):       residual={res_after:.4f}  l1={l1_after:.4f}")
print(f"  alpha_corrected (BCH):         residual={res_corrected:.4f}")

# ════════════════════════════════════════════════════
# PART 4: BUILD STUDENT FROM HOMOTOPY-CORRECTED MAPS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 4: Student Initialized from Homotopy-Corrected Maps")
print("="*65)

def build_student(init_type, label):
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight)
        stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            if init_type=='serre':
                # Original Serre cascade
                C=ad_k(J14,Js[min(L_ATT+l+1,N_LAYERS_T-1)],l+1)
                n=float(np.linalg.norm(C)); C=C/max(n,1e-8)
                W_d=lift_to_d(C,U14,scale=0.01)
            elif init_type=='htp':
                # Homotopy-corrected Jacobian at L14+l
                J_src=Js_htp[min(L_ATT+l,N_LAYERS_T-1)]
                dJ_src=J_src-np.eye(ma)
                n=float(np.linalg.norm(dJ_src)); dJ_src=dJ_src/max(n,1e-8)
                W_d=lift_to_d(dJ_src,U14,scale=0.01)
            elif init_type=='htp_cascade':
                # Serre cascade computed from homotopy-corrected Jacobians
                C=ad_k(Js_htp[L_ATT],
                       Js_htp[min(L_ATT+l+1,N_LAYERS_T-1)],l+1)
                n=float(np.linalg.norm(C)); C=C/max(n,1e-8)
                W_d=lift_to_d(C,U14,scale=0.01)
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
    v0=eval_val(stu,n=20)
    print(f"\n  [{label}] zero-shot: val={v0:.4f}")

    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,201):
        for pg in opt_s.param_groups: pg['lr']=clr(step,200,50)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [50,100,150,200]:
            print(f"  [{label}] step {step}  val={eval_val(stu,n=20):.4f}")
    return eval_val(stu)

vA=build_student('serre','A: Original Serre cascade')
vB=build_student('htp','B: Homotopy-corrected Jacobians')
vC=build_student('htp_cascade','C: Serre cascade from htp Jacobians')

print(f"\n{'='*65}")
print(f"  RESULTS")
print("="*65)
print(f"""
  Teacher (24L):                    val={val_teacher:.4f}
  A: Original Serre cascade:        val={vA:.4f}
  B: Homotopy-corrected Jacobians:  val={vB:.4f}
  C: Serre cascade from htp maps:   val={vC:.4f}

  Ext^1 mean dimension: {float(np.mean(ext1_dims)):.1f}
  Tor_1 mean dimension: {float(np.mean(tor1_dims)):.1f}

  IF B or C < A:
    The homotopy transfer improves the initialization.
    The Ext^1 correction brings us closer to the true MC element.
    The single algebraic operation (htp correction) replaces
    some of the 200 CE gradient steps.

  IF A,B,C all converge to same val:
    The initialization is not the bottleneck.
    The 200 CE steps are needed for the co-adaptation
    regardless of which algebraic object we use.
    The single-step construction is not achievable via
    Ext/Tor correction alone.

  MC RESIDUAL INTERPRETATION:
    residual={res_before:.4f} means alpha = log(M_fwd) is NOT the MC element.
    The true MC element requires the full Hochschild complex,
    not just the monodromy chain.
    The curvature mu_0 = {res_before:.4f} is the obstruction to
    finding the MC element without gradient descent.
""")
