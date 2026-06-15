#!/usr/bin/env python3
"""
Massey Products and Gauge Flattening
======================================
Two experiments targeting the irreducible gradient content:

PART A: DIRECT MASSEY PRODUCT COMPUTATION
  mu_3, mu_4, mu_5, mu_6 are the higher A_inf structure maps.
  They encode the deep topological interaction between the
  cascade and embeddings that gradient descent learns in 200 steps.
  
  Massey product formula (from A_inf relations):
    mu_3(a,b,c) = mu_2(mu_2(a,b),c) - mu_2(a,mu_2(b,c))
                  + (homotopy correction)
  
  For the transformer:
    a,b,c = elements of the active subspace at L14
    mu_2(a,b) = [a,b] (Lie bracket = layer commutator)
    mu_3 = triple Massey product = obstruction to mu_2 being associative
  
  Compute mu_3...mu_6 from the teacher Jacobians in closed form,
  then use them to initialize the student embedding.
  
  The hypothesis: mu_3...mu_6 encode the 45% missing from the
  Laplacian. If we can compute them directly, we eliminate the
  co-adaptation phase.

PART B: GAUGE FLATTENING
  The A_inf algebra has curvature mu_0 = 1.99 (MC residual).
  A gauge transformation g: A -> A'
    mu_k' = sum_{i1+...+il=k} g^{i1} o mu_l o (g^{-i1} x ... x g^{-il})
  
  We seek g such that mu_0' = 0 (flat) while preserving:
    - Property T gap (topological invariant, set by cascade)
    - Kac-Moody orbit (Serre decay rate, set by algebra)
  
  If such g exists: the obstruction to partial^2 = 0 is removed
  algebraically. The student can be initialized at the flat point
  without running CE steps to flatten the curvature.
  
  Strategy: find g = exp(beta) where beta solves
    d(beta) + mu_0 = 0
  i.e., beta is a primitive of the curvature in the Hochschild complex.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import logm as scipy_logm, expm as scipy_expm
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  MASSEY PRODUCTS + GAUGE FLATTENING")
print(f"  Direct construction of mu_3..mu_6 and mu_0 -> 0")
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

# ════════════════════════════════════════════════════
# Train teacher + extract Jacobians
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
J14=Js[L_ATT]; U14=Us[L_ATT]; dJ14=J14-np.eye(ma)
print(f"  Done. ma={ma}\n")

# Serre cascade (standard)
cascade_std=[]
for l in range(1,N_STU+1):
    C=ad_k(J14,Js[min(L_ATT+l,N_LAYERS_T-1)],l)
    n=float(np.linalg.norm(C)); C=C/max(n,1e-8)
    cascade_std.append(C)

def inject_blocks(model, cascade):
    with torch.no_grad():
        model.pe.weight.copy_(teacher.pe.weight)
        model.ln_f.weight.copy_(teacher.ln_f.weight)
        model.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            W_d=lift_to_d(cascade[l],U14,scale=0.01)
            W_t=torch.tensor(W_d,dtype=torch.float32)
            model.blocks[l].attn.WK.weight.copy_(W_t)
            model.blocks[l].attn.WQ.weight.copy_(W_t.T)
            model.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            model.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            model.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            model.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            model.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)

def run_student(cascade, emb, label, steps=200):
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    with torch.no_grad():
        stu.te.weight.copy_(torch.tensor(emb[:VOCAB,:D].copy(),dtype=torch.float32))
    inject_blocks(stu, cascade)
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    v0=eval_val(stu,n=20); print(f"  [{label}] zero-shot={v0:.4f}")
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps,50)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [50,100,150,200]:
            print(f"  [{label}] step {step}  val={eval_val(stu,n=20):.4f}")
    return eval_val(stu)

# ════════════════════════════════════════════════════
# PART A: MASSEY PRODUCTS mu_3 ... mu_6
# ════════════════════════════════════════════════════
print("="*65)
print("PART A: DIRECT MASSEY PRODUCT COMPUTATION")
print("="*65)
print("""
  The A_inf relations give recursive formulas for mu_k:
  
  mu_2(a,b) = [a,b]   (Lie bracket)
  mu_3(a,b,c) = mu_2(mu_2(a,b),c) - mu_2(a,mu_2(b,c))
              = [[a,b],c] - [a,[b,c]]  (Jacobi defect)
  mu_4(a,b,c,d) = accumulated Jacobi defect at order 4
  ...
  
  For transformer Jacobians: a,b,c,d = J_l, J_{l+1}, J_{l+2}, J_{l+3}
  at the attractor neighborhood (L13-L17).
""")

# Compute Massey products from attractor neighborhood Jacobians
# Use J at L12, L13, L14, L15, L16, L17 as the 6 inputs
att_Js=[Js[L_ATT-2], Js[L_ATT-1], Js[L_ATT],
        Js[L_ATT+1], Js[L_ATT+2], Js[L_ATT+3]]

def massey2(a,b): return comm(a,b)
def massey3(a,b,c):
    """Triple Massey product = Jacobi defect."""
    return comm(comm(a,b),c) - comm(a,comm(b,c))
def massey4(a,b,c,d):
    """Quadruple Massey product = order-4 associativity defect."""
    m3_abc=massey3(a,b,c)
    return comm(m3_abc,d) - massey3(a,comm(b,c),d) + massey3(a,b,comm(c,d))
def massey5(a,b,c,d,e):
    """Order-5 Massey product."""
    m4=massey4(a,b,c,d)
    return comm(m4,e) - massey4(a,b,c,comm(d,e))
def massey6(a,b,c,d,e,f):
    """Order-6 Massey product."""
    m5=massey5(a,b,c,d,e)
    return comm(m5,f) - massey5(a,b,c,d,comm(e,f))

a,b,c,d,e,f = att_Js

mu2=massey2(a,b)
mu3=massey3(a,b,c)
mu4=massey4(a,b,c,d)
mu5=massey5(a,b,c,d,e)
mu6=massey6(a,b,c,d,e,f)

print(f"  Massey product norms at attractor neighborhood:")
print(f"  ||mu_2|| = {float(np.linalg.norm(mu2)):.6f}")
print(f"  ||mu_3|| = {float(np.linalg.norm(mu3)):.6f}  "
      f"(ratio mu3/mu2 = {float(np.linalg.norm(mu3)/max(np.linalg.norm(mu2),1e-8)):.4f})")
print(f"  ||mu_4|| = {float(np.linalg.norm(mu4)):.6f}  "
      f"(ratio mu4/mu2 = {float(np.linalg.norm(mu4)/max(np.linalg.norm(mu2),1e-8)):.4f})")
print(f"  ||mu_5|| = {float(np.linalg.norm(mu5)):.6f}  "
      f"(ratio mu5/mu2 = {float(np.linalg.norm(mu5)/max(np.linalg.norm(mu2),1e-8)):.4f})")
print(f"  ||mu_6|| = {float(np.linalg.norm(mu6)):.6f}  "
      f"(ratio mu6/mu2 = {float(np.linalg.norm(mu6)/max(np.linalg.norm(mu2),1e-8)):.4f})")

# Check Serre relation: does mu_k match ad(J14)^k?
print(f"\n  Alignment of Massey products with Serre cascade:")
print(f"  {'k':>3}  {'||Massey_k||':>13}  {'||Serre_k||':>12}  {'alignment':>10}")
print("  "+"-"*42)
massey_ops=[None, None, mu2, mu3, mu4, mu5, mu6]
for k in range(2,7):
    mk=massey_ops[k]
    sk=cascade_std[k-1] if k-1<len(cascade_std) else None
    if sk is not None and mk is not None:
        mk_n=mk/max(float(np.linalg.norm(mk)),1e-8)
        sk_n=sk/max(float(np.linalg.norm(sk)),1e-8)
        align=float(np.sum(mk_n*sk_n))
        print(f"  {k:>3}  {float(np.linalg.norm(mk)):>13.6f}  "
              f"{float(np.linalg.norm(sk)):>12.6f}  {align:>10.4f}")

# Use Massey products directly as cascade levels
print(f"\n  Building cascade from Massey products...")
cascade_massey=[]
for k,mk in enumerate([mu2,mu3,mu4,mu5,mu6,comm(mu6,att_Js[0])]):
    n=float(np.linalg.norm(mk))
    cascade_massey.append(mk/max(n,1e-8))
    print(f"  Level {k+1}: ||mu_{k+2}|| = {n:.6f}")

# ════════════════════════════════════════════════════
# PART B: GAUGE FLATTENING (mu_0 -> 0)
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART B: GAUGE FLATTENING")
print("  Find beta s.t. d(beta) + mu_0 = 0")
print("  g = exp(beta), mu_k' = gauge-transformed structure maps")
print("="*65)

# Current curvature: mu_0 = MC residual
M_fwd=np.eye(ma)
for l in range(L_ATT+1): M_fwd=Js[l]@M_fwd
alpha=np.real(scipy_logm(M_fwd))

def mc_sum(alpha_in, J14_in, n_terms=6):
    dJ=J14_in-np.eye(ma)
    l1=comm(dJ,alpha_in)
    s=l1.copy(); fac=1
    for k in range(2,n_terms+1):
        fac*=k
        lk=alpha_in.copy()
        for _ in range(k-1): lk=comm(J14_in,lk)
        s=s+lk/fac
    return s

mu0=mc_sum(alpha,J14)
print(f"\n  Initial curvature ||mu_0|| = {float(np.linalg.norm(mu0)):.4f}")

# Strategy 1: beta = -h(mu_0) where h = pseudoinverse of d
# d(beta) = [dJ14, beta] (Hochschild differential)
# Solve: [dJ14, beta] = -mu_0
# This is a Sylvester equation: dJ14 @ beta - beta @ dJ14 = -mu_0

def solve_sylvester_approx(A, C, n_iter=50, lr=0.01):
    """Iterative solution of AX - XA = C (Lyapunov/Sylvester)."""
    X=np.zeros_like(C)
    for _ in range(n_iter):
        grad=comm(A,X)-C
        X=X-lr*grad
    return X

print(f"  Solving Sylvester equation [dJ14, beta] = -mu_0...")
beta=solve_sylvester_approx(dJ14,-mu0,n_iter=500,lr=0.001)
residual_sylvester=float(np.linalg.norm(comm(dJ14,beta)+mu0))
print(f"  Sylvester residual ||[dJ14,beta]+mu_0|| = {residual_sylvester:.4f}")

# Gauge transformation: g = exp(epsilon * beta)
print(f"\n  Gauge transformation g = exp(eps*beta) for eps in [0.01, 0.1, 1.0]:")
print(f"  {'eps':>8}  {'||mu_0_gauged||':>16}  {'PropertyT_gap':>14}  "
      f"{'Serre_R2':>10}  {'improvement'}")
print("  "+"-"*60)

def gauge_transform_alpha(alpha_in, beta_in, eps):
    """Apply gauge: alpha' = BCH(alpha, eps*beta)."""
    b=eps*beta_in
    # BCH to order 3: log(exp(alpha)exp(b))
    alpha_prime=(alpha_in+b
                 +0.5*comm(alpha_in,b)
                 +(1/12)*(comm(alpha_in,comm(alpha_in,b))
                          +comm(b,comm(b,alpha_in))))
    return alpha_prime

def check_propertyT(Js_list):
    """Fast approximation of Property T gap."""
    n=len(Js_list)
    A=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j: A[i,j]=float(np.linalg.norm(comm(Js_list[i],Js_list[j])))
    A=(A+A.T)/2
    eigs=np.linalg.eigvalsh(A)[::-1]
    return float(eigs[0]-eigs[1]) if len(eigs)>1 else 0.0

def check_serre_decay(J14_in, Js_in):
    """Check Serre decay rate after gauge."""
    residuals=[]
    for k in range(2,7):
        l_idx=min(L_ATT+k-1,N_LAYERS_T-1)
        C=ad_k(J14_in,Js_in[l_idx],k)
        n_mu2=float(np.linalg.norm(comm(J14_in,Js_in[l_idx])))
        residuals.append(float(np.linalg.norm(C))/max(n_mu2,1e-8))
    if len(residuals)>=2:
        # Fit log-linear decay
        ks=np.arange(2,2+len(residuals),dtype=float)
        log_r=np.log(np.maximum(residuals,1e-10))
        slope,_=np.polyfit(ks,log_r,1)
        return slope
    return 0.0

gap_orig=check_propertyT(Js[L_ATT-2:L_ATT+3])
slope_orig=check_serre_decay(J14,Js)
mu0_norm_orig=float(np.linalg.norm(mu0))
print(f"  {'orig':>8}  {mu0_norm_orig:>16.4f}  {gap_orig:>14.3f}  "
      f"{slope_orig:>10.4f}  {'baseline'}")

best_eps=0; best_mu0=mu0_norm_orig; best_alpha=alpha
for eps in [0.001,0.005,0.01,0.05,0.1,0.5,1.0]:
    alpha_g=gauge_transform_alpha(alpha,beta,eps)
    mu0_g=mc_sum(alpha_g,J14)
    mu0_norm=float(np.linalg.norm(mu0_g))
    # Reconstruct gauged Jacobians: J_l' = exp(eps*beta) @ J_l @ exp(-eps*beta)
    g_mat=scipy_expm(eps*beta)
    g_inv=scipy_expm(-eps*beta)
    Js_gauged=[g_mat@Js[l]@g_inv for l in range(N_LAYERS_T)]
    J14_g=Js_gauged[L_ATT]
    gap_g=check_propertyT(Js_gauged[L_ATT-2:L_ATT+3])
    slope_g=check_serre_decay(J14_g,Js_gauged)
    improvement=(mu0_norm_orig/max(mu0_norm,1e-8))
    marker=""
    if mu0_norm<best_mu0:
        best_mu0=mu0_norm; best_eps=eps; best_alpha=alpha_g; marker=" ← best"
    print(f"  {eps:>8.3f}  {mu0_norm:>16.4f}  {gap_g:>14.3f}  "
          f"{slope_g:>10.4f}  {improvement:>8.2f}x{marker}")

print(f"\n  Best gauge: eps={best_eps}  ||mu_0|| reduced to {best_mu0:.4f}")
print(f"  Original:                            {mu0_norm_orig:.4f}")
print(f"  Reduction: {mu0_norm_orig/max(best_mu0,1e-8):.2f}x")

# ════════════════════════════════════════════════════
# PART C: TEST BOTH IN STUDENT
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART C: STUDENT EXPERIMENTS")
print("="*65)

E_teacher=teacher.te.weight.data.numpy().copy()

# Baseline
print(f"\n  Baseline A: Standard Serre cascade...")
vA=run_student(cascade_std, E_teacher, "A-Serre-std")

# Massey cascade
print(f"\n  B: Massey product cascade...")
vB=run_student(cascade_massey, E_teacher, "B-Massey")

# Gauge-flattened cascade
print(f"\n  C: Gauge-flattened Serre cascade (best eps={best_eps})...")
if best_eps>0:
    g_mat=scipy_expm(best_eps*beta)
    g_inv=scipy_expm(-best_eps*beta)
    Js_g=[g_mat@Js[l]@g_inv for l in range(N_LAYERS_T)]
    J14_g=Js_g[L_ATT]; U14_g=U14  # subspace unchanged by similarity transform
    cascade_gauge=[]
    for l in range(1,N_STU+1):
        C=ad_k(J14_g,Js_g[min(L_ATT+l,N_LAYERS_T-1)],l)
        n=float(np.linalg.norm(C)); C=C/max(n,1e-8)
        cascade_gauge.append(C)
    vC=run_student(cascade_gauge, E_teacher, "C-gauge-flat")
else:
    vC=vA; print(f"  Gauge flat: no improvement found, same as baseline")

# Combined: Massey + gauge
print(f"\n  D: Massey cascade + gauge-flattened Jacobians...")
if best_eps>0:
    cascade_massey_g=[]
    for k,mk in enumerate([mu2,mu3,mu4,mu5,mu6,comm(mu6,att_Js[0])]):
        mk_g=g_mat@mk@g_inv  # apply gauge to Massey products
        n=float(np.linalg.norm(mk_g)); cascade_massey_g.append(mk_g/max(n,1e-8))
    vD=run_student(cascade_massey_g, E_teacher, "D-Massey+gauge")
else:
    vD=vB

# ════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  MASSEY + GAUGE RESULTS")
print("="*65)
print(f"""
  MASSEY PRODUCTS (closed form from teacher Jacobians):
    ||mu_2|| = {float(np.linalg.norm(mu2)):.6f}
    ||mu_3|| = {float(np.linalg.norm(mu3)):.6f}
    ||mu_4|| = {float(np.linalg.norm(mu4)):.6f}
    ||mu_5|| = {float(np.linalg.norm(mu5)):.6f}
    ||mu_6|| = {float(np.linalg.norm(mu6)):.6f}

  GAUGE FLATTENING:
    Initial curvature ||mu_0||:    {mu0_norm_orig:.4f}
    Best gauge (eps={best_eps}):      {best_mu0:.4f}
    Reduction:                     {mu0_norm_orig/max(best_mu0,1e-8):.2f}x
    Sylvester residual:            {residual_sylvester:.4f}

  STUDENT RESULTS (200 CE steps, teacher embeddings):
    Teacher oracle:                val={val_teacher:.4f}
    A: Standard Serre:             val={vA:.4f}  (baseline)
    B: Massey cascade:             val={vB:.4f}
    C: Gauge-flat cascade:         val={vC:.4f}
    D: Massey + gauge:             val={vD:.4f}

  INTERPRETATION:
  If B < A: Massey products give better cascade than ad(J14)^k.
    The mu_3..mu_6 computed directly from attractor Jacobians
    are better block initializations than the Serre cascade.
    The 45% missing from the Laplacian is partially captured.

  If C < A: Gauge flattening improves the topological class.
    Reducing mu_0 brings the initialization closer to
    M_{{O=0}}, reducing the CE steps needed for co-adaptation.

  If D < A: Both effects compound.
    The combined Massey + gauge transformation may bring
    the student close enough to the flat section that
    fewer than 200 CE steps are needed.

  If A ≈ B ≈ C ≈ D:
    Neither Massey products nor gauge flattening improve
    initialization quality. The 200 CE steps are truly
    irreducible — they compute something not accessible
    from the Jacobian chain alone.
""")
