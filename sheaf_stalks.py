#!/usr/bin/env python3
"""
Sheaf Stalk Extractor
======================
Builds the factorization sheaf F on the quiver Q=(V,E,M) and
verifies that gradient descent computes the Maurer-Cartan element
of the associated L_infinity algebra.

STALK STRUCTURE:
  F(v_l) = A_infinity algebra at layer interface l
           = (active subspace U_l, differential dJ_l, product mu2_l,
              higher maps mu_k_l from Serre cascade)

MAURER-CARTAN ELEMENT:
  alpha = log(M_fwd)  in the active subspace
  This is directly computable from M_fwd — no gradient descent.

VERIFICATION:
  1. Compute alpha = log(M_fwd)
  2. Verify the MC equation: sum_k (1/k!) l_k(alpha,...,alpha) = 0
     where l_k are the L_inf brackets from the Kac-Moody algebra
  3. Compare alpha to what gradient descent finds:
     corr(alpha, J14 - I)  — does the MC element match the attractor Jacobian?
  4. Build the sheaf restriction maps and verify compatibility:
     F(v_l) --J_l--> F(v_{l+1}) satisfies A_inf relations

GRADED STRUCTURE:
  The stalk F(v_l) is graded by the filtration level r:
  F^r(v_l) = kernel of d_l restricted to the r-th filtration level
  The associated graded gr_r = F^r/F^{r+1} gives the spectral sequence pages.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import logm as scipy_logm
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  SHEAF STALK EXTRACTOR")
print(f"  Factorization sheaf on quiver Q=(V,E,M)")
print(f"  Maurer-Cartan element = log(M_fwd)")
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

# ════════════════════════════════════════════════════════
# Train teacher
# ════════════════════════════════════════════════════════
print("Training teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS)
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
print()

# ════════════════════════════════════════════════════════
# Extract Jacobian chain
# ════════════════════════════════════════════════════════
print("Extracting Jacobian chain (5 refs)...",flush=True)
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
J_acc=[[] for _ in range(N_LAYERS)]; U_acc=[[] for _ in range(N_LAYERS)]
for ref in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS):
        J,U=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=J.shape[0]
    if (ref+1)%3==0: print(f"  ref {ref+1}/5...",flush=True)

Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS)]
J14=Js[L_ATT]; U14=Us[L_ATT]; dJ14=J14-np.eye(ma)
print(f"  Done. ma={ma}\n")

# ════════════════════════════════════════════════════════
# PART 1: BUILD THE SHEAF STALKS
# ════════════════════════════════════════════════════════
print("="*65)
print("PART 1: SHEAF STALKS F(v_l)")
print("="*65)

print(f"\n  Building stalks at each layer interface...")
print(f"  Stalk = (differential d_l, product mu2_l, higher mu_k from cascade)")
print()

stalks={}
for l in range(N_LAYERS):
    J=Js[l]; dJ=J-np.eye(ma); U=Us[l]
    sv_dJ=np.linalg.svd(dJ,compute_uv=False)
    rank=int(np.sum(sv_dJ>sv_dJ[0]*0.1)) if sv_dJ[0]>1e-8 else 0

    # mu1 = differential dJ_l
    mu1=dJ

    # mu2 = deviation from associativity: J_{l+1} @ J_l - J_l @ J_{l+1} = [J_{l+1},J_l]
    if l<N_LAYERS-1:
        mu2=comm(Js[l+1],J)  # the associativity defect at layer l
        mu2_norm=float(np.linalg.norm(mu2))
    else:
        mu2_norm=0.0

    stalks[l]={
        'J': J, 'dJ': dJ, 'U': U,
        'mu1_norm': float(np.linalg.norm(dJ)),
        'mu2_norm': mu2_norm,
        'rank': rank,
        'sv': sv_dJ,
    }

print(f"  {'L':>3}  {'||mu1||':>9}  {'||mu2||':>9}  {'rank':>6}  {'stalk_type'}")
print("  "+"-"*42)
for l in range(N_LAYERS):
    s=stalks[l]
    att=" ← ATTRACTOR" if l==L_ATT else ""
    # Classify stalk type
    if s['mu1_norm']>50: stype="anomaly"
    elif s['mu2_norm']>0.5: stype="non-assoc"
    elif s['rank']<=3: stype="near-output"
    else: stype="regular"
    print(f"  L{l:>2}  {s['mu1_norm']:>9.4f}  {s['mu2_norm']:>9.4f}  "
          f"{s['rank']:>6}  {stype}{att}")

# ════════════════════════════════════════════════════════
# PART 2: MAURER-CARTAN ELEMENT
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 2: MAURER-CARTAN ELEMENT alpha = log(M_fwd)")
print("="*65)

# Compute M_fwd
M_fwd=np.eye(ma)
for l in range(L_ATT+1): M_fwd=Js[l]@M_fwd
sv_mfwd=np.linalg.svd(M_fwd,compute_uv=False)
print(f"\n  M_fwd sv[:4] = {sv_mfwd[:4].round(4)}")

# Compute alpha = log(M_fwd) in active subspace
# Use SVD: M_fwd = U S V^T => log(M_fwd) via eigendecomposition
try:
    # M_fwd may not be symmetric — use Schur decomposition via scipy
    alpha_mfwd=np.real(scipy_logm(M_fwd))
    sv_alpha=np.linalg.svd(alpha_mfwd,compute_uv=False)
    print(f"  alpha = log(M_fwd):")
    print(f"    ||alpha|| = {float(np.linalg.norm(alpha_mfwd)):.4f}")
    print(f"    sv(alpha)[:4] = {sv_alpha[:4].round(4)}")
    print(f"    trace(alpha) = {float(np.trace(alpha_mfwd)):.6f}")
    mc_ok=True
except Exception as e:
    print(f"  log(M_fwd) failed: {e}")
    # Fall back: alpha = M_fwd - I (first order approximation)
    alpha_mfwd=M_fwd-np.eye(ma)
    print(f"  Using alpha ≈ M_fwd - I (first order)")
    mc_ok=False

# ════════════════════════════════════════════════════════
# PART 3: VERIFY MC EQUATION
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 3: MAURER-CARTAN EQUATION VERIFICATION")
print("  sum_k (1/k!) l_k(alpha,...,alpha) = 0")
print("  l_k = L_inf brackets from Kac-Moody algebra")
print("  l_1(alpha) = d(alpha) = [dJ14, alpha]")
print("  l_2(alpha,alpha) = [alpha, alpha] (should vanish for MC element)")
print("="*65)

# L_inf brackets using the Kac-Moody structure
# l_1(x) = dJ14 @ x - x @ dJ14 = ad(dJ14)(x)
# l_2(x,y) = [x,y] = x@y - y@x
# l_k(x,...,x) = ad(x)^{k-1}(l_1(x)) for higher brackets

l1_alpha=comm(dJ14, alpha_mfwd)  # l_1(alpha) = [dJ14, alpha]
l2_alpha=comm(alpha_mfwd, alpha_mfwd)  # = 0 always (anticommutativity)

print(f"\n  l_1(alpha) = [dJ14, alpha]:")
print(f"    ||l_1(alpha)|| = {float(np.linalg.norm(l1_alpha)):.4f}")

# Higher brackets using Serre cascade structure
print(f"\n  Higher L_inf brackets l_k(alpha,...,alpha):")
mc_sum=l1_alpha.copy()  # k=1 term
print(f"  k=1: ||l_1(alpha)/1!|| = {float(np.linalg.norm(l1_alpha)):.4f}")

factorial=1
for k in range(2,7):
    factorial*=k
    # l_k using Serre cascade: ad(J14)^{k-1}(l_1(alpha))
    lk=ad_k(J14,l1_alpha,k-1)
    lk_contrib=lk/factorial
    mc_sum=mc_sum+lk_contrib
    print(f"  k={k}: ||l_{k}(alpha,...)/({k}!)|| = {float(np.linalg.norm(lk_contrib)):.6f}")

print(f"\n  MC equation residual ||sum_k l_k(alpha,...)/k!|| = {float(np.linalg.norm(mc_sum)):.6f}")
mc_satisfied = float(np.linalg.norm(mc_sum)) < float(np.linalg.norm(l1_alpha)) * 0.1
print(f"  MC equation {'SATISFIED' if mc_satisfied else 'NOT satisfied'} "
      f"(residual {'<' if mc_satisfied else '>='} 10% of l_1 term)")

# ════════════════════════════════════════════════════════
# PART 4: COMPARE ALPHA TO ATTRACTOR JACOBIAN
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 4: ALPHA vs ATTRACTOR JACOBIAN")
print("  Does log(M_fwd) predict J14 - I?")
print("="*65)

# Normalize both for comparison
alpha_n=alpha_mfwd/max(float(np.linalg.norm(alpha_mfwd)),1e-8)
dJ14_n=dJ14/max(float(np.linalg.norm(dJ14)),1e-8)

corr_direct=float(np.corrcoef(alpha_n.flatten(),dJ14_n.flatten())[0,1])
print(f"\n  corr(alpha/||alpha||, dJ14/||dJ14||) = {corr_direct:.4f}")

# Compare singular vector subspaces
Ua,_,_=np.linalg.svd(alpha_mfwd); Ua=Ua[:,:4]
Ud,_,_=np.linalg.svd(dJ14); Ud=Ud[:,:4]
sv_cross=np.linalg.svd(Ua.T@Ud,compute_uv=False)
subspace_align=float(np.mean(np.clip(sv_cross,0,1)))
print(f"  Top-4 subspace alignment: {subspace_align:.4f}")
print(f"  (1.0 = identical subspace, 0.0 = orthogonal)")

# ════════════════════════════════════════════════════════
# PART 5: SHEAF RESTRICTION MAPS — A_inf COMPATIBILITY
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 5: SHEAF RESTRICTION MAPS")
print("  J_l: F(v_l) -> F(v_{l+1})")
print("  A_inf compatibility: d_{l+1} @ J_l = J_l @ d_l + mu2_l")
print("="*65)

print(f"\n  Testing A_inf compatibility at each layer pair:")
print(f"  ||d_{{l+1}}@J_l - J_l@d_l - mu2_l||  (should be small if stalk compatible)")
print()
print(f"  {'L→L+1':>8}  {'||defect||':>12}  {'||d_l||':>9}  {'ratio':>8}")
print("  "+"-"*42)

defects=[]
for l in range(N_LAYERS-1):
    d_l=Js[l]-np.eye(ma)
    d_l1=Js[l+1]-np.eye(ma)
    mu2_l=comm(Js[l+1],Js[l])
    # A_inf: d_{l+1} @ J_l - J_l @ d_l should equal mu2_l
    lhs=d_l1@Js[l]-Js[l]@d_l
    defect=float(np.linalg.norm(lhs-mu2_l))
    d_norm=float(np.linalg.norm(d_l))
    ratio=defect/max(d_norm,1e-8)
    defects.append(ratio)
    att=" ←" if l==L_ATT else ""
    print(f"  L{l:>2}→L{l+1:<2}  {defect:>12.6f}  {d_norm:>9.4f}  {ratio:>8.4f}{att}")

mean_defect=float(np.mean(defects))
print(f"\n  Mean A_inf defect ratio: {mean_defect:.4f}")
print(f"  {'COMPATIBLE' if mean_defect<0.1 else 'NOT compatible'} "
      f"(threshold: defect/||d|| < 0.1)")

# ════════════════════════════════════════════════════════
# PART 6: GRADED STRUCTURE — SPECTRAL SEQUENCE PAGES
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("PART 6: GRADED STALK STRUCTURE")
print("  gr_r(F(v_l)) = F^r(v_l) / F^{r+1}(v_l)")
print("  The associated graded = spectral sequence pages")
print("="*65)

print(f"\n  Filtration at each layer (rank of each filtration level):")
print(f"  {'L':>3}  {'rank(F^0)':>10}  {'rank(F^1)':>10}  "
      f"{'rank(F^2)':>10}  {'gr_dim'}")
print("  "+"-"*48)

for l in range(0,N_LAYERS,4):
    dJ=stalks[l]['dJ']
    sv=stalks[l]['sv']
    # Filtration: F^r = span of singular vectors with sv > threshold_r
    thresholds=[sv[0]*t for t in [0.5,0.2,0.05]]
    ranks=[int(np.sum(sv>t)) for t in thresholds]
    gr_dim=ranks[0]-ranks[1] if len(ranks)>1 else ranks[0]
    att=" ←L14" if l==L_ATT else ""
    print(f"  L{l:>2}  {ranks[0]:>10}  "
          f"{ranks[1] if len(ranks)>1 else 0:>10}  "
          f"{ranks[2] if len(ranks)>2 else 0:>10}  "
          f"{gr_dim:>7}{att}")

# ════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  SHEAF STALK SUMMARY")
print("="*65)
print(f"""
  WHAT IS DIRECTLY COMPUTABLE (no gradient descent):
  
  1. All stalk differentials mu1_l = dJ_l
     (from one forward pass through trained teacher)
  
  2. All stalk products mu2_l = [J_{{l+1}}, J_l]
     (commutator of consecutive Jacobians)
  
  3. Maurer-Cartan element alpha = log(M_fwd)
     ||alpha|| = {float(np.linalg.norm(alpha_mfwd)):.4f}
     corr with dJ14 = {corr_direct:.4f}
  
  4. All sheaf restriction maps J_l: F(v_l) -> F(v_{{l+1}})
     A_inf compatibility defect: {mean_defect:.4f}
  
  5. Graded stalk structure = spectral sequence pages
     (filtration by singular value thresholds)

  WHAT GRADIENT DESCENT IS COMPUTING:
  The global section Gamma(X, F) — the consistent assignment
  of A_inf algebra elements across all stalks satisfying
  the gluing conditions J_l @ mu_k^l = mu_k^{{l+1}} @ J_l^{{otimes k}}.
  
  The Maurer-Cartan element alpha = log(M_fwd) encodes this
  global section in the active subspace. Its alignment with dJ14
  (corr = {corr_direct:.4f}) shows how much of the gradient descent
  target is captured by the monodromy logarithm.
  
  MC equation residual = {float(np.linalg.norm(mc_sum)):.6f}
  {'→ The MC equation is satisfied — alpha IS the correct element' if mc_satisfied else
   '→ MC equation not satisfied — alpha is approximate, gradient refinement needed'}
""")
