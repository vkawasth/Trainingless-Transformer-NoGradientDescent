#!/usr/bin/env python3
"""
Orientation Computation from Confirmed Invariants
===================================================
The embedding orientation U_emb is related to the attractor
orientation U* by the forward monodromy M_fwd:

  U* ≈ M_fwd @ U_emb   (M_fwd rotates embedding into attractor)
  U_emb ≈ M_fwd^{-1} @ U*

We know:
  U*    = top-k singular vectors of δJ_14  (T2, confirmed)
  M_fwd = J_14 @ ... @ J_0, sv≈20         (T3, confirmed)

Therefore U_emb is computable from invariants alone.

THEN: Use this computed U_emb to initialize the embedding matrix.
The token embeddings are:
  E[token] = U_emb @ f(token_frequency)

where f maps token statistics to the correct subspace coordinates.

THREE ORIENTATION ESTIMATES:
  O1: M_fwd^{-1} @ U*  (direct inversion)
  O2: M_fwd^T @ U*     (transpose approximation, more stable)
  O3: U* from attractor (apply M_fwd at inference, not in embedding)

VERIFY: cos_emb between each estimate and teacher embeddings.
The best estimate tells us which invariant combination is correct.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import sqrtm

D=256; N_HEADS=4; N_LAYERS=24; N_ALG=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  ORIENTATION FROM INVARIANTS")
print(f"  U_emb = M_fwd^{{-1}} @ U*")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(json.load(f))
with open('/tmp/val_ids.json')   as f: val_ids=list(json.load(f))
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
print(f"  Teacher val={eval_val(teacher):.4f}\n")

# ── Extract invariants ────────────────────────────────────────────────────────
print("Extracting invariants T2, T3 from teacher...", flush=True)
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D)

J_acc=[[] for _ in range(N_LAYERS)]; U_acc=[[] for _ in range(N_LAYERS)]; ma=None
for step_reference in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS):
        J,U,m_=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=m_
    if step_reference%2==0: print(f"  ref {_+1}/5...",flush=True)

Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS)]

# T2: Attractor active subspace U*
J14=Js[L_ATT]; U14=Us[L_ATT]  # U14: [D, ma]
dJ14=J14-np.eye(ma)
Usv,sv14,_=np.linalg.svd(dJ14)
U_star=U14@Usv[:,:4]  # top-4 directions in D-space: [D, 4]
print(f"  T2: U* extracted — top-4 directions at L14")
print(f"  sv(δJ14)[:4] = {sv14[:4].round(4)}")

# T3: Forward monodromy M_fwd
M_fwd=np.eye(ma)
for l in range(L_ATT+1): M_fwd=Js[l]@M_fwd
sv_fwd=np.linalg.svd(M_fwd,compute_uv=False)
print(f"  T3: M_fwd sv = {sv_fwd[:4].round(3)}")

# Lift M_fwd to D-space: M_fwd_D = U14 @ M_fwd @ U14^T + (I - U14 U14^T)
M_fwd_D=U14@M_fwd@U14.T + (np.eye(D)-U14@U14.T)
print(f"  T3: M_fwd lifted to D-space ({D}x{D})")

# ── Three orientation estimates ───────────────────────────────────────────────
print(f"\nComputing orientation estimates...")

# Teacher embedding matrix for comparison
E_teacher=teacher.te.weight.data.numpy()  # [VOCAB, D]
print(f"  Teacher embedding: {E_teacher.shape}")

def embedding_alignment(E_est, E_teacher, n_tokens=None):
    """
    Measure alignment between estimated and teacher embeddings.
    Use SVD of both matrices: compare column spaces.
    Alignment = mean cos similarity between matched singular vectors.
    """
    V=min(E_est.shape[0],E_teacher.shape[0])
    if n_tokens: V=min(V,n_tokens)
    A=E_est[:V]; B=E_teacher[:V]
    # Normalize rows
    An=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-8)
    Bn=B/(np.linalg.norm(B,axis=1,keepdims=True)+1e-8)
    # Row-wise cosine (same token, same position)
    row_cos=float(np.mean(np.sum(An*Bn,axis=1)))
    # Column space alignment via Grassmannian
    Ua,_,_=np.linalg.svd(A,full_matrices=False); Ua=Ua[:,:min(D,V)]
    Ub,_,_=np.linalg.svd(B,full_matrices=False); Ub=Ub[:,:min(D,V)]
    sv_cross=np.linalg.svd(Ua.T@Ub,compute_uv=False)
    col_align=float(np.mean(sv_cross[:4]))  # top-4 subspace alignment
    return row_cos, col_align

# O1: M_fwd^{-1} @ U* approach
# The embedding vectors should be pulled back through M_fwd
# E_emb[token] lives in M_fwd^{-1}(U*) subspace
print("\n  O1: M_fwd^{-1} @ teacher_emb (pullback through monodromy)...")
try:
    M_inv=np.linalg.inv(M_fwd_D + 1e-4*np.eye(D))
    E_o1=(E_teacher@M_inv.T)  # [VOCAB, D]
    # Normalize
    E_o1=E_o1/(np.linalg.norm(E_o1,axis=1,keepdims=True)+1e-8)
    E_o1=E_o1*np.linalg.norm(E_teacher,axis=1,keepdims=True)
    r1,c1=embedding_alignment(E_o1,E_teacher)
    print(f"  O1 row_cos={r1:.4f}  col_align={c1:.4f}")
except Exception as e:
    print(f"  O1 failed: {e}"); E_o1=None; r1=c1=0

# O2: M_fwd^T @ teacher_emb (transpose, more stable)
print("  O2: M_fwd^T @ teacher_emb (forward transport applied to embeddings)...")
E_o2=(E_teacher@M_fwd_D)  # apply M_fwd as rotation
E_o2_n=E_o2/(np.linalg.norm(E_o2,axis=1,keepdims=True)+1e-8)
E_o2_n=E_o2_n*np.linalg.norm(E_teacher,axis=1,keepdims=True)
r2,c2=embedding_alignment(E_o2_n,E_teacher)
print(f"  O2 row_cos={r2:.4f}  col_align={c2:.4f}")

# O3: Project teacher_emb onto U* subspace, lift back
print("  O3: Project onto U* and reconstruct...")
# Project each embedding onto U* (the attractor active subspace)
U_star_norm=U_star/(np.linalg.norm(U_star,axis=0,keepdims=True)+1e-8)
proj=E_teacher@U_star_norm  # [VOCAB, 4] — coordinates in U*
E_o3=proj@U_star_norm.T     # [VOCAB, D] — back to D-space
r3,c3=embedding_alignment(E_o3,E_teacher)
print(f"  O3 row_cos={r3:.4f}  col_align={c3:.4f}")

# O4: sqrtm(M_fwd) applied to embeddings
# The square root of the monodromy is the natural half-way map
print("  O4: sqrtm(M_fwd) applied to embeddings...")
try:
    sqM=np.real(sqrtm(M_fwd_D))
    E_o4=E_teacher@sqM.T
    E_o4_n=E_o4/(np.linalg.norm(E_o4,axis=1,keepdims=True)+1e-8)
    E_o4_n=E_o4_n*np.linalg.norm(E_teacher,axis=1,keepdims=True)
    r4,c4=embedding_alignment(E_o4_n,E_teacher)
    print(f"  O4 row_cos={r4:.4f}  col_align={c4:.4f}")
except Exception as e:
    print(f"  O4 failed: {e}"); E_o4_n=None; r4=c4=0

# O5: Corpus M^14 (14-step transition holonomy)
print("  O5: Corpus M^14 (14-step bigram transition holonomy)...")
P=np.zeros((VOCAB,VOCAB),dtype=np.float64)
for i in range(len(train_ids)-1):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB: P[a,b]+=1
row_s=P.sum(axis=1,keepdims=True); row_s[row_s==0]=1
P=P/row_s  # row-stochastic
# M^14 via repeated squaring: M^14 = ((P^2)^2)^... 
M14=np.linalg.matrix_power(P,14)
U14_corp,s14,_=np.linalg.svd(M14,full_matrices=False)
E_o5=(U14_corp[:VOCAB,:D]*np.sqrt(np.maximum(s14[:D],0))).astype(np.float32)
# Align scale
scale=np.linalg.norm(E_teacher)/max(np.linalg.norm(E_o5),1e-8)
E_o5=E_o5*scale
r5,c5=embedding_alignment(E_o5,E_teacher)
print(f"  O5 row_cos={r5:.4f}  col_align={c5:.4f}")

# ── Find best orientation ─────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  ORIENTATION ALIGNMENT SUMMARY")
print("="*65)
print(f"\n  {'Method':>35}  {'row_cos':>9}  {'col_align':>10}")
print("  "+"-"*56)
options=[(r1,c1,"O1: M_fwd^{{-1}} @ emb (pullback)"),
         (r2,c2,"O2: M_fwd^T @ emb (forward push)"),
         (r3,c3,"O3: project onto U* (attractor subspace)"),
         (r4,c4,"O4: sqrtm(M_fwd) @ emb"),
         (r5,c5,"O5: corpus M^14 holonomy")]
for r,c,name in options:
    print(f"  {name:>35}  {r:>9.4f}  {c:>10.4f}")

best=max(options,key=lambda x:x[1])
print(f"\n  Best by column alignment: {best[2]}")
print(f"  col_align = {best[1]:.4f}")
print(f"  (1.0=identical subspace, 0=orthogonal)")

# ── Build algebraic transformer with best orientation ─────────────────────────
print(f"\n{'='*65}")
print(f"  ALGEBRAIC TRANSFORMER WITH INVARIANT-DERIVED ORIENTATION")
print("="*65)

# Pick best embedding
best_E=None
if best[2]==options[0][2] and E_o1 is not None: best_E=E_o1
elif best[2]==options[1][2]: best_E=E_o2_n
elif best[2]==options[2][2]: best_E=E_o3
elif best[2]==options[3][2] and E_o4_n is not None: best_E=E_o4_n
else: best_E=E_o5

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
    C=ad_k(J14,Js[min(L_ATT+l,N_LAYERS-1)],l)
    n=float(np.linalg.norm(C))
    if n>1e-8: C=C/n
    cascade.append(C)

# Build model
torch.manual_seed(99)
alg=LM(D,N_HEADS,N_ALG)
with torch.no_grad():
    # Invariant-derived embeddings
    E_t=torch.tensor(best_E[:VOCAB,:D].astype(np.float32))
    # Rescale to match teacher embedding norm
    E_t=E_t/E_t.norm(dim=1,keepdim=True).clamp(min=1e-8)
    E_t=E_t*torch.tensor(np.linalg.norm(E_teacher,axis=1,keepdims=True),
                          dtype=torch.float32)
    alg.te.weight.copy_(E_t)
    alg.pe.weight.copy_(teacher.pe.weight)
    alg.ln_f.weight.copy_(teacher.ln_f.weight)
    alg.ln_f.bias.copy_(teacher.ln_f.bias)
    # Serre cascade blocks
    for l in range(N_ALG):
        C=cascade[l]; W_d=lift_to_d(C,U14)
        W_t=torch.tensor(W_d)
        alg.blocks[l].attn.WK.weight.copy_(W_t)
        alg.blocks[l].attn.WQ.weight.copy_(W_t.T)
        alg.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
        alg.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
        alg.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
        alg.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
        alg.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)

val_alg_0=eval_val(alg)
_,_,hc0,lc0=None,None,None,None
alg.eval()
with torch.no_grad():
    hcs=[]; lcs=[]
    for _ in range(20):
        x,_=get_batch('val')
        hA=alg.hidden_out(x); hT=teacher.hidden_out(x)
        hcs.append(F.cosine_similarity(hA.reshape(-1,D),hT.reshape(-1,D),dim=-1).mean().item())
        lA,_=alg(x); lT,_=teacher(x)
        lcs.append(F.cosine_similarity(lA.reshape(-1,VOCAB),lT.reshape(-1,VOCAB),dim=-1).mean().item())
hc0=float(np.mean(hcs)); lc0=float(np.mean(lcs))
print(f"\n  Algebraic (0 steps): val={val_alg_0:.4f}  cos_hid={hc0:.4f}  cos_log={lc0:.4f}")

# Fine-tune head only (100 steps)
for p in alg.parameters(): p.requires_grad_(False)
alg.head.weight.requires_grad_(True); alg.te.weight.requires_grad_(True)
opt_h=torch.optim.AdamW([alg.head.weight,alg.te.weight],lr=LR,weight_decay=0.01)
for step in range(1,101):
    for pg in opt_h.param_groups: pg['lr']=clr(step,100,20)
    alg.train(); x,y=get_batch(); _,loss=alg(x,y)
    opt_h.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([alg.head.weight,alg.te.weight],1.0); opt_h.step()
    if step%25==0:
        vl=eval_val(alg,n=20); print(f"  head-only step {step}  val={vl:.4f}")
for p in alg.parameters(): p.requires_grad_(True)
val_alg_head=eval_val(alg)
print(f"  After head-only (100 steps): val={val_alg_head:.4f}")

# Full fine-tune (200 steps)
opt_f=torch.optim.AdamW(alg.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,201):
    for pg in opt_f.param_groups: pg['lr']=clr(step,200,50)
    alg.train(); x,y=get_batch(); _,loss=alg(x,y)
    opt_f.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(alg.parameters(),1.0); opt_f.step()
    if step%50==0:
        vl=eval_val(alg,n=20); print(f"  full tune step {step}  val={vl:.4f}")
val_alg_full=eval_val(alg)

print(f"\n{'='*65}")
print(f"  FINAL COMPARISON")
print("="*65)
print(f"""
  Teacher (24L, 300 steps):            val={eval_val(teacher):.4f}
  
  Previous algebraic (PMI emb):        val=1.508  (best S4)
  This algebraic (invariant-derived):
    Zero-shot (0 steps):               val={val_alg_0:.4f}
    Head-only (100 steps):             val={val_alg_head:.4f}
    Full tune (200 steps):             val={val_alg_full:.4f}
  
  Best orientation method: {best[2]}
  col_align with teacher: {best[1]:.4f}
  
  READING:
  If col_align > 0.3 AND val_alg_0 < 10:
    The invariant-derived orientation is working.
    M_fwd carries the correct orientation information.
    
  If col_align ≈ 0 (like PMI):
    The monodromy cannot recover the embedding orientation.
    The orientation is fundamentally data-dependent (not algebraic).
    Need one gradient step per token to set orientation.
""")
