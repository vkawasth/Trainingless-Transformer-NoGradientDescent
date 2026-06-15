#!/usr/bin/env python3
"""
Token Orientation from Jacobian Gram Matrix
=============================================
The correct embedding orientation is the eigenvectors of the
token affinity matrix A modulated by the accumulated Jacobian Gram:

  Gram = sum_l J_l^T @ J_l   (accumulated Gram in m-space)
  Gram_D = U14 @ Gram @ U14^T  (lifted to D-space)
  
  A[i,j] = C[i,j] * <e_i, Gram_D e_j>

where C[i,j] is the token co-occurrence count and e_i is the
canonical basis vector for token i (one-hot → position in D-space
via the U14 subspace).

The eigenvectors of A are the correct token orientations:
  E_orient[token] = eigenvector of A corresponding to token

This does NOT require teacher embeddings as input.
It requires: corpus co-occurrence + Jacobian Gram matrix.

The Jacobians can come from:
  a) The trained teacher (what we test here)
  b) A tiny reference model (5x cheaper)
  c) The cyclic nerve M^n (fully training-free)

VERIFY: row_cos between E_orient and teacher embeddings.
If row_cos > 0.3: the orientation is learnable algebraically.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; N_ALG=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  TOKEN ORIENTATION FROM JACOBIAN GRAM MATRIX")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(json.load(f))
with open('/tmp/val_ids.json')   as f: val_ids=list(json.load(f))
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(list(map(int,train_ids)),dtype=torch.long)
val_t  =torch.tensor(list(map(int,val_ids)),  dtype=torch.long)

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
    def hidden_out(self,x):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        return self.ln_f(h)
    def hidden_states(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs

def clr(s,total,warmup=50):
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
    return J.T,U.detach().numpy(),m

# ── Train teacher ─────────────────────────────────────────────────────────────
print("Training teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS)
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
val_t=eval_val(teacher)
print(f"  Teacher val={val_t:.4f}\n")
E_teacher=teacher.te.weight.data.numpy().copy()

# ── Extract Jacobians ─────────────────────────────────────────────────────────
print("Extracting Jacobian chain...",flush=True)
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D)
J_acc=[[] for _ in range(N_LAYERS)]; U_acc=[[] for _ in range(N_LAYERS)]; ma=None
for _ in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS):
        J,U,m_=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=m_
Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS)]
print(f"  Done. m={ma}\n")

J14=Js[L_ATT]; U14=Us[L_ATT]

# ── Step 1: Accumulated Jacobian Gram matrix ──────────────────────────────────
print("Step 1: Accumulated Jacobian Gram matrix...")
Gram=np.zeros((ma,ma))
for l in range(N_LAYERS):
    Gram+=Js[l].T@Js[l]
Gram=Gram/N_LAYERS
sv_gram=np.linalg.svd(Gram,compute_uv=False)
print(f"  Gram sv[:4] = {sv_gram[:4].round(3)}")

# Lift Gram to D-space
Gram_D=U14@Gram@U14.T  # [D,D]
print(f"  Gram_D shape: {Gram_D.shape}")

# Forward monodromy M_fwd (T3 invariant)
M_fwd=np.eye(ma)
for l in range(L_ATT+1): M_fwd=Js[l]@M_fwd
sv_fwd=np.linalg.svd(M_fwd,compute_uv=False)
print(f"  M_fwd sv[:4] = {sv_fwd[:4].round(3)}\n")

# ── Step 2: Corpus co-occurrence matrix C ────────────────────────────────────
print("Step 2: Corpus token co-occurrence matrix...")
# C[i,j] = count of token j following token i (bigram)
C=np.zeros((VOCAB,VOCAB),dtype=np.float32)
for k in range(len(train_ids)-1):
    a,b=train_ids[k],train_ids[k+1]
    if a<VOCAB and b<VOCAB: C[a,b]+=1

# Also include skip-bigrams (within window of 3)
for k in range(len(train_ids)-2):
    a,b=train_ids[k],train_ids[k+2]
    if a<VOCAB and b<VOCAB: C[a,b]+=0.5

row_s=C.sum(axis=1,keepdims=True); row_s[row_s==0]=1
C_norm=C/row_s  # row-stochastic
print(f"  C shape: {C.shape}  nnz: {int((C>0).sum())}\n")

# ── Step 3: Token affinity matrix A ──────────────────────────────────────────
print("Step 3: Token affinity A = C * (Gram modulation)...")
# A[i,j] = C[i,j] * token_interaction(i,j via Gram_D)
# We approximate: token i maps to direction Gram_D[:,i%D] in D-space
# Better: use the Gram_D as a kernel on the co-occurrence graph

# Method: spectral embedding of C weighted by Gram_D
# 1. Get top-D singular vectors of C (token → D-dim)
print("  SVD of co-occurrence matrix C...")
U_c,s_c,Vt_c=np.linalg.svd(C_norm,full_matrices=False)
# Token embeddings from C-SVD
E_cooc=U_c[:,:D]*np.sqrt(np.maximum(s_c[:D],0))  # [VOCAB, D]

# 2. Modulate by Gram_D: rotate E_cooc into the Gram eigenbasis
# The correct orientation: E_orient = E_cooc @ Gram_D^{1/2}
print("  Computing Gram_D^{1/2} for rotation...")
sv_g,Uv_g=np.linalg.eigh(Gram_D)
sv_g=np.maximum(sv_g,0)
Gram_sqrt=Uv_g@np.diag(np.sqrt(sv_g))@Uv_g.T

E_gram=E_cooc@Gram_sqrt  # [VOCAB, D] — co-occurrence modulated by Gram

# Normalize to match teacher scale
scale=np.linalg.norm(E_teacher)/max(np.linalg.norm(E_gram),1e-8)
E_gram=E_gram*scale

print(f"  E_gram shape: {E_gram.shape}\n")

# ── Step 4: Measure orientation alignment ────────────────────────────────────
print("Step 4: Measuring alignment with teacher embeddings...")

def row_cos_sim(A,B):
    An=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-8)
    Bn=B/(np.linalg.norm(B,axis=1,keepdims=True)+1e-8)
    return float(np.mean(np.sum(An*Bn,axis=1)))

def subspace_align(A,B,k=32):
    """Grassmannian alignment of top-k singular subspaces."""
    Ua,_,_=np.linalg.svd(A,full_matrices=False); Ua=Ua[:,:k]
    Ub,_,_=np.linalg.svd(B,full_matrices=False); Ub=Ub[:,:k]
    sv=np.linalg.svd(Ua.T@Ub,compute_uv=False)
    return float(np.mean(np.clip(sv,0,1)))

# Baselines
E_rand=np.random.randn(VOCAB,D).astype(np.float32)*0.02
E_pmi_1step=np.zeros((VOCAB,D),dtype=np.float32)
Uc,sc,_=np.linalg.svd(C_norm,full_matrices=False)
E_pmi_1step=Uc[:,:D]*np.sqrt(np.maximum(sc[:D],0))
scale=np.linalg.norm(E_teacher)/max(np.linalg.norm(E_pmi_1step),1e-8)
E_pmi_1step=E_pmi_1step*scale

print(f"\n  Orientation alignment results:")
print(f"  {'Method':>40}  {'row_cos':>9}  {'subspace':>10}")
print("  "+"-"*60)
# Full-basis U14 projection of teacher embeddings
# U14: [D, ma] — the full ma-dim attractor basis
# This uses ALL 48 dimensions, not just top-4
E_u14_proj = E_teacher @ U14 @ U14.T  # [VOCAB, D] — project & lift
scale_u14 = np.linalg.norm(E_teacher)/max(np.linalg.norm(E_u14_proj),1e-8)
E_u14_proj = E_u14_proj * scale_u14

# M_fwd full-basis: apply M_fwd in projected space, lift to D
# This is the correct closed form: M_fwd rotates embedding→attractor
M_fwd_proj = U14 @ M_fwd @ U14.T  # [D,D] — M_fwd in full D-space
E_mfwd = E_teacher @ M_fwd_proj.T  # apply monodromy rotation
scale_mf = np.linalg.norm(E_teacher)/max(np.linalg.norm(E_mfwd),1e-8)
E_mfwd = E_mfwd * scale_mf

# M_fwd^{-1} pullback: if U* = M_fwd @ U_emb, then U_emb = M_fwd^{-1} @ U*
# Apply to teacher embeddings to see what the "pre-monodromy" orientation is
try:
    M_fwd_inv = np.linalg.inv(M_fwd_proj + 1e-4*np.eye(D))
    E_mfwd_inv = E_teacher @ M_fwd_inv.T
    scale_mi = np.linalg.norm(E_teacher)/max(np.linalg.norm(E_mfwd_inv),1e-8)
    E_mfwd_inv = E_mfwd_inv * scale_mi
except:
    E_mfwd_inv = E_rand

methods=[
    ("Random baseline",E_rand),
    ("1-step PMI (bigram SVD)",E_pmi_1step),
    ("Gram-modulated co-occurrence (Ours)",E_gram),
    ("Full U14 projection (all 48 dims)",E_u14_proj),
    ("M_fwd @ teacher (full basis)",E_mfwd),
    ("M_fwd^{-1} @ teacher (pullback)",E_mfwd_inv),
    ("Teacher (oracle)",E_teacher),
]
for name,E in methods:
    rc=row_cos_sim(E,E_teacher)
    sa=subspace_align(E,E_teacher)
    print(f"  {name:>40}  {rc:>9.4f}  {sa:>10.4f}")

# ── Step 5: Use E_gram in algebraic transformer ───────────────────────────────
print(f"\n{'='*65}")
print(f"  ALGEBRAIC TRANSFORMER WITH GRAM-ORIENTED EMBEDDINGS")
print("="*65)

# Serre cascade
def comm(A,B): return A@B-B@A
def ad_k(A,B,k):
    r=B
    for _ in range(k): r=comm(A,r)
    return r
def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)

cascade=[]
for l in range(1,N_ALG+1):
    C_l=ad_k(J14,Js[min(L_ATT+l,N_LAYERS-1)],l)
    n=float(np.linalg.norm(C_l))
    if n>1e-8: C_l=C_l/n
    cascade.append(C_l)

def build_alg(E_init):
    torch.manual_seed(99)
    m=LM(D,N_HEADS,N_ALG)
    with torch.no_grad():
        E_t=torch.tensor(E_init[:VOCAB,:D].astype(np.float32))
        # Rescale
        teacher_norms=torch.tensor(np.linalg.norm(E_teacher,axis=1,keepdims=True),
                                    dtype=torch.float32)
        E_t=E_t/(E_t.norm(dim=1,keepdim=True).clamp(min=1e-8))*teacher_norms
        m.te.weight.copy_(E_t)
        m.pe.weight.copy_(teacher.pe.weight)
        m.ln_f.weight.copy_(teacher.ln_f.weight)
        m.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_ALG):
            W_d=lift_to_d(cascade[l],U14)
            W_t=torch.tensor(W_d)
            m.blocks[l].attn.WK.weight.copy_(W_t)
            m.blocks[l].attn.WQ.weight.copy_(W_t.T)
            m.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            m.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            m.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            m.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            m.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
    return m

def finetune_head(model,steps=100,label=""):
    for p in model.parameters(): p.requires_grad_(False)
    model.head.weight.requires_grad_(True)
    model.te.weight.requires_grad_(True)
    opt=torch.optim.AdamW([model.head.weight,model.te.weight],lr=LR,weight_decay=0.01)
    for step in range(1,steps+1):
        for pg in opt.param_groups: pg['lr']=clr(step,steps,20)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([model.head.weight,model.te.weight],1.0)
        opt.step()
        if step%(steps//4)==0:
            vl=eval_val(model,n=20)
            print(f"  [{label}] step {step}  val={vl:.4f}")
    for p in model.parameters(): p.requires_grad_(True)
    return eval_val(model)

# Build and test each embedding type
print(f"\n  Testing each embedding in algebraic transformer:")
print(f"  {'Embedding':>35}  {'val_0':>7}  {'val_100':>8}")
print("  "+"-"*54)

for name,E_init in [("Random",E_rand),
                     ("1-step PMI",E_pmi_1step),
                     ("Gram-modulated (Ours)",E_gram),
                     ("Teacher emb (oracle)",E_teacher)]:
    m=build_alg(E_init)
    v0=eval_val(m)
    v100=finetune_head(m,100,name)
    print(f"  {name:>35}  {v0:>7.4f}  {v100:>8.4f}")

print(f"\n  Teacher oracle val={val_t:.4f}")
print(f"""
  READING:
  If Gram-modulated < 1-step PMI:
    The Jacobian Gram matrix carries genuine orientation signal.
    The network's transformation structure improves on raw corpus statistics.
    
  If Gram-modulated ≈ Teacher emb:
    The Gram modulation recovers most of the embedding orientation.
    Training-free orientation is achievable.
    
  If all ≈ Random:
    Orientation requires gradient descent through data.
    The algebraic structure cannot encode it.
""")
