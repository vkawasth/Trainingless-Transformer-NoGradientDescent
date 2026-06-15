#!/usr/bin/env python3
"""
Laplacian Eigenmap Embedding
==============================
Tokens as nodes, co-occurrence as edge weights.
Embedding = smallest d eigenvectors of the normalized graph Laplacian.

This is closed-form, one-shot, no gradient descent.
The orientation is determined by the graph's spectral geometry.

THREE GRAPH CONSTRUCTIONS:
  L1: Bigram co-occurrence (A[i,j] = count(i followed by j))
  L2: Symmetric skip-gram (A[i,j] = count(i,j within window 5))
  L3: PMI-weighted (A[i,j] = max(PMI(i,j), 0) — positive associations only)

LAPLACIAN EIGENMAP:
  D = diag(A @ ones)  (degree matrix)
  L = D - A           (unnormalized Laplacian)
  L_norm = D^{-1/2} L D^{-1/2}  (normalized)
  E = eigenvectors of L_norm, smallest d eigenvalues (skip λ=0)

WHY THIS MIGHT WORK BETTER:
  The Laplacian eigenvectors encode global graph structure.
  Token i and token j have similar embeddings if they appear
  in similar contexts — which is exactly what a trained
  embedding learns via gradient descent.
  
  The spectral geometry of the co-occurrence graph IS the
  semantic geometry of the vocabulary. The Laplacian gives
  it in closed form.

COMPARE:
  Laplacian embedding vs PPMI vs teacher oracle
  in the Serre cascade pipeline.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  LAPLACIAN EIGENMAP EMBEDDING")
print(f"  Closed-form orientation from co-occurrence graph")
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

def comm(A,B): return A@B-B@A
def ad_k(A,B,k):
    r=B
    for _ in range(k): r=comm(A,r)
    return r
def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)

def laplacian_embedding(A, d=D, norm=True):
    """
    Compute d-dimensional Laplacian eigenmap embedding.
    A: [V,V] adjacency matrix (non-negative, symmetric)
    Returns E: [V,d] embedding
    """
    # Symmetrize
    A = (A + A.T) / 2
    A = np.maximum(A, 0)

    # Degree matrix
    deg = A.sum(axis=1)
    deg = np.maximum(deg, 1e-10)

    if norm:
        # Normalized Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        # L_norm v = λ v  ↔  find smallest eigenvalues
        # Use sparse for efficiency
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        A_sp = sp.csr_matrix(A)
        L_norm = sp.eye(VOCAB) - D_inv_sqrt @ A_sp @ D_inv_sqrt
    else:
        # Unnormalized: L = D - A
        D_mat = sp.diags(deg)
        A_sp = sp.csr_matrix(A)
        L_norm = D_mat - A_sp

    # Find d+1 smallest eigenvectors (skip the trivial λ=0)
    k = min(d+2, VOCAB-1)
    try:
        vals, vecs = spla.eigsh(L_norm, k=k, which='SM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        vals = vals[idx]; vecs = vecs[:, idx]
        # Skip first (trivial constant vector, λ≈0)
        # Take next d eigenvectors
        skip = int(np.sum(vals < 1e-8))
        E = vecs[:, skip:skip+d].astype(np.float32)
        if E.shape[1] < d:
            # Pad if needed
            E = np.hstack([E, np.zeros((VOCAB, d-E.shape[1]))])
    except Exception as ex:
        print(f"  eigsh failed: {ex}, falling back to dense")
        vals, vecs = np.linalg.eigh(L_norm.toarray())
        skip = int(np.sum(vals < 1e-8))
        E = vecs[:, skip:skip+d].astype(np.float32)

    return E, vals

def scale_to(E, target_norm):
    n = float(np.linalg.norm(E, 'fro'))
    if n > 1e-8: E = E * (target_norm / n)
    return E

def row_cos(E, E_teacher):
    En = E / (np.linalg.norm(E,axis=1,keepdims=True)+1e-8)
    Tn = E_teacher / (np.linalg.norm(E_teacher,axis=1,keepdims=True)+1e-8)
    return float(np.mean(np.sum(En*Tn,axis=1)))

def gram_align(E, E_teacher):
    G1=(E@E.T).flatten(); G2=(E_teacher@E_teacher.T).flatten()
    return float(np.corrcoef(G1,G2)[0,1])

# ════════════════════════════════════════════════════════
# STAGE 0: Train teacher
# ════════════════════════════════════════════════════════
print("Stage 0: Train teacher (300 steps)...")
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
val_teacher=eval_val(teacher)
E_teacher=teacher.te.weight.data.numpy().copy()
teacher_norm=float(np.linalg.norm(E_teacher,'fro'))
print(f"  Teacher val={val_teacher:.4f}\n")

# ════════════════════════════════════════════════════════
# STAGE 1: Extract invariants + cascade
# ════════════════════════════════════════════════════════
print("Stage 1: Extract invariants...")
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
J_acc=[[] for _ in range(N_LAYERS)]; U_acc=[[] for _ in range(N_LAYERS)]
for ref in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS):
        J,U,m_=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=m_
    if (ref+1)%3==0: print(f"  ref {ref+1}/5...",flush=True)
Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS)]
J14=Js[L_ATT]; U14=Us[L_ATT]
Gram=sum(Js[l].T@Js[l] for l in range(N_LAYERS))/N_LAYERS
Gram_D=U14@Gram@U14.T+(np.eye(D)-U14@U14.T)

cascade=[]
for l in range(1,N_STU+1):
    C_l=ad_k(J14,Js[min(L_ATT+l,N_LAYERS-1)],l)
    n=float(np.linalg.norm(C_l))
    if n>1e-8: C_l=C_l/n
    cascade.append(C_l)
print(f"  Done.\n")

def inject_cascade(model):
    with torch.no_grad():
        model.pe.weight.copy_(teacher.pe.weight)
        model.ln_f.weight.copy_(teacher.ln_f.weight)
        model.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            W_d=lift_to_d(cascade[l],U14,scale=0.01)
            W_t=torch.tensor(W_d,dtype=torch.float32)
            model.blocks[l].attn.WK.weight.copy_(W_t)
            model.blocks[l].attn.WQ.weight.copy_(W_t.T)
            model.blocks[l].attn.WV.weight.copy_(
                teacher.blocks[L_ATT].attn.WV.weight)
            model.blocks[l].attn.op.weight.copy_(
                teacher.blocks[L_ATT].attn.op.weight)
            model.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            model.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            model.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)

# ════════════════════════════════════════════════════════
# STAGE 2: Build adjacency matrices
# ════════════════════════════════════════════════════════
print("Stage 2: Build co-occurrence adjacency matrices...")

# Raw bigram counts
bigram=np.zeros((VOCAB,VOCAB),dtype=np.float32)
unigram=np.zeros(VOCAB,dtype=np.float32)
for k in range(len(train_ids)-1):
    a,b=train_ids[k],train_ids[k+1]
    if 0<=a<VOCAB and 0<=b<VOCAB:
        bigram[a,b]+=1; unigram[a]+=1

# A1: Symmetric bigram
A1=(bigram+bigram.T)/2
print(f"  A1 (symmetric bigram): nnz={int((A1>0).sum())}")

# A2: Skip-gram window=5
A2=np.zeros((VOCAB,VOCAB),dtype=np.float32)
W=5
for k in range(len(train_ids)):
    for j in range(k+1,min(k+W+1,len(train_ids))):
        a,b=train_ids[k],train_ids[j]
        if 0<=a<VOCAB and 0<=b<VOCAB and a!=b:
            w=1.0/(j-k)  # distance-weighted
            A2[a,b]+=w; A2[b,a]+=w
print(f"  A2 (skip-gram w=5):    nnz={int((A2>0).sum())}")

# A3: PMI-weighted (positive only)
total=bigram.sum()
up=unigram/max(total,1)
pmi=np.log((bigram/max(total,1)+1e-10)/(up[:,None]*up[None,:]+1e-10))
A3=np.maximum(pmi,0).astype(np.float32)
A3=(A3+A3.T)/2  # symmetrize
print(f"  A3 (PPMI):             nnz={int((A3>0).sum())}\n")

# ════════════════════════════════════════════════════════
# STAGE 3: Compute Laplacian embeddings
# ════════════════════════════════════════════════════════
print("Stage 3: Compute Laplacian eigenmap embeddings (closed form)...")

embeddings={}
for name,A in [("L1_bigram",A1),("L2_skipgram",A2),("L3_ppmi",A3)]:
    t0=time.time()
    print(f"  Computing {name}...",flush=True)
    E,vals=laplacian_embedding(A,d=D,norm=True)
    E=scale_to(E,teacher_norm)
    rc=row_cos(E,E_teacher)
    ga=gram_align(E,E_teacher)
    print(f"  {name}: row_cos={rc:.4f}  gram_align={ga:.4f}  "
          f"λ[1]={vals[1]:.4f}  λ[D]={vals[min(D,len(vals)-1)]:.4f}  "
          f"t={time.time()-t0:.1f}s")
    embeddings[name]=E

# Also: Jacobian-modulated Laplacian
# A_jac[i,j] = A3[i,j] * (e_i @ Gram_D @ e_j) — but we don't have e_i yet
# Use the L3 embedding to modulate
print(f"\n  Computing Jacobian-modulated Laplacian...")
E_l3=embeddings["L3_ppmi"]
# Modulate adjacency by Gram structure
sv_g,Uv_g=np.linalg.eigh(Gram_D)
sv_g=np.maximum(sv_g,0)
Gram_sqrt=(Uv_g*np.sqrt(sv_g)[None,:])@Uv_g.T
E_l3_mod=(E_l3@Gram_sqrt).astype(np.float32)
E_l3_mod=scale_to(E_l3_mod,teacher_norm)
rc_mod=row_cos(E_l3_mod,E_teacher)
ga_mod=gram_align(E_l3_mod,E_teacher)
print(f"  L3+Gram_mod: row_cos={rc_mod:.4f}  gram_align={ga_mod:.4f}")
embeddings["L3_jac_mod"]=E_l3_mod

# Teacher oracle
embeddings["Teacher_oracle"]=E_teacher

# Summary
print(f"\n  {'Method':>20}  {'row_cos':>9}  {'gram_align':>11}")
print("  "+"-"*44)
for name,E in embeddings.items():
    rc=row_cos(E,E_teacher)
    ga=gram_align(E,E_teacher)
    print(f"  {name:>20}  {rc:>9.4f}  {ga:>11.4f}")

# ════════════════════════════════════════════════════════
# STAGE 4: Test in Serre pipeline
# ════════════════════════════════════════════════════════
print(f"\nStage 4: Serre pipeline with each Laplacian embedding...")

def run(E_init,label,head_steps=100,full_steps=200):
    torch.manual_seed(99)
    model=LM(D,N_HEADS,N_STU)
    with torch.no_grad():
        E_t=torch.tensor(E_init[:VOCAB,:D].copy(),dtype=torch.float32)
        model.te.weight.copy_(E_t)
    inject_cascade(model)
    v0=eval_val(model,n=20)
    print(f"\n  [{label}] zero-shot={v0:.4f}")

    # Head only
    for p in model.parameters(): p.requires_grad_(False)
    model.head.weight.requires_grad_(True)
    opt_h=torch.optim.AdamW([model.head.weight],lr=LR,weight_decay=0.01)
    for step in range(1,head_steps+1):
        for pg in opt_h.param_groups: pg['lr']=clr(step,head_steps,20)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt_h.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([model.head.weight],1.0); opt_h.step()
    for p in model.parameters(): p.requires_grad_(True)
    vh=eval_val(model)
    print(f"  [{label}] head-only {head_steps}st: val={vh:.4f}")

    # Full fine-tune
    opt_f=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,full_steps+1):
        for pg in opt_f.param_groups: pg['lr']=clr(step,full_steps,50)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt_f.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt_f.step()
        if step%(full_steps//4)==0:
            print(f"    step {step}  val={eval_val(model,n=20):.4f}")
    vf=eval_val(model)
    return v0,vh,vf

results={}
for name,E in embeddings.items():
    v0,vh,vf=run(E,name)
    results[name]=(v0,vh,vf)

# ════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  LAPLACIAN EIGENMAP RESULTS")
print("="*65)
print(f"\n  {'Method':>20}  {'gram_align':>11}  {'zero-shot':>10}  "
      f"{'head-100':>9}  {'full-200':>9}")
print("  "+"-"*65)
for name,E in embeddings.items():
    ga=gram_align(E,E_teacher)
    v0,vh,vf=results[name]
    print(f"  {name:>20}  {ga:>11.4f}  {v0:>10.4f}  {vh:>9.4f}  {vf:>9.4f}")

print(f"""
  Teacher full model:          val={val_teacher:.4f}
  Prior best (Serre+teacher):  val=0.1865

  KEY READING:
  The Laplacian eigenmap is a ONE-SHOT closed-form computation.
  No gradient descent. No iteration. Pure linear algebra.

  If gram_align(Laplacian) > gram_align(PPMI=0.317):
    The graph Laplacian recovers more token similarity structure
    than raw co-occurrence statistics.
    The spectral geometry of the vocabulary graph encodes
    the same information that gradient descent finds.

  If full-tune val(Laplacian) ≈ full-tune val(Teacher oracle):
    The Laplacian embedding is sufficient.
    Training reduces to head alignment only (~100 steps).
    90% training reduction is achievable.

  If Laplacian val >> Teacher oracle:
    Spectral graph theory cannot recover the orientation.
    The missing structure requires gradient descent.
    Minimum training remains ~120 joint CE steps.
""")
