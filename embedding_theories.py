#!/usr/bin/env python3
"""
Three Embedding Dimension Theories
====================================

THEORY 1: Attractor subspace basis (48-dim)
  G_input = M_fwd^{-1} @ U14^T  [48 x D]
  C: [V x 48]  (32,400 params, 81% fewer than full)
  Prediction: variance ~80-90%, val approaches 0.187

THEORY 2: Kac-Moody rank basis (8-dim)
  G_km = top-8 singular vectors of accumulated Gram_D  [8 x D]
  C: [V x 8]  (5,400 params, 97% fewer than full)
  Prediction: 6-12 active roots per context

THEORY 3: Laplacian + joint fine-tune (full D, ~90 steps)
  Initialize embedding from Laplacian eigenmap (gram_align=0.547)
  Joint fine-tune embedding+head for N steps (blocks frozen Serre)
  Prediction: ~90 steps from Laplacian vs 200 from random

All compared against:
  - Teacher oracle embedding (val=0.187 with Serre+200CE)
  - Full embedding random init (val=0.185)
  - Teacher (val=0.250)
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
print(f"  THREE EMBEDDING THEORIES")
print(f"  T1: attractor pullback (48-dim)")
print(f"  T2: Kac-Moody rank basis (8-dim)")
print(f"  T3: Laplacian + joint fine-tune")
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

class LowDimLM(nn.Module):
    """LM with E[t] = C[t] @ G, C learned, G fixed."""
    def __init__(self,d,nh,nl,G_fixed):
        super().__init__()
        ng=G_fixed.shape[0]
        self.C=nn.Parameter(torch.randn(VOCAB,ng)*0.02)
        self.register_buffer('G',torch.tensor(G_fixed,dtype=torch.float32))
        self.pe=nn.Embedding(512,d)
        self.blocks=nn.ModuleList([Block(d,nh) for _ in range(nl)])
        self.ln_f=nn.LayerNorm(d)
        nn.init.normal_(self.pe.weight,std=0.02)
    def get_embeddings(self): return self.C@self.G
    def forward(self,x,y=None):
        E=self.get_embeddings()
        h=E[x]+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        h=self.ln_f(h)
        logits=h@E.T
        loss=F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1)) if y is not None else None
        return logits,loss
    def hidden_states(self,x):
        hs=[]; E=self.get_embeddings()
        h=E[x]+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
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

def variance_explained(G,E_teacher):
    """What fraction of teacher embedding variance is in G's subspace?"""
    Gn=G/(np.linalg.norm(G,axis=1,keepdims=True)+1e-8)
    proj=E_teacher@Gn.T@Gn
    return float(np.var(proj)/np.var(E_teacher)*100)

def gram_align(E,E_teacher):
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
# STAGE 1: Extract invariants
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

# M_fwd
M_fwd=np.eye(ma)
for l in range(L_ATT+1): M_fwd=Js[l]@M_fwd
sv_fwd=np.linalg.svd(M_fwd,compute_uv=False)
print(f"  M_fwd sv[:4]={sv_fwd[:4].round(3)}")

# Gram_D
Gram=sum(Js[l].T@Js[l] for l in range(N_LAYERS))/N_LAYERS
Gram_D=U14@Gram@U14.T+(np.eye(D)-U14@U14.T)

# Serre cascade
cascade=[]
for l in range(1,N_STU+1):
    C_l=ad_k(J14,Js[min(L_ATT+l,N_LAYERS-1)],l)
    n=float(np.linalg.norm(C_l))
    if n>1e-8: C_l=C_l/n
    cascade.append(C_l)
print(f"  Cascade: {N_STU} levels\n")

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
            model.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            model.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            model.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            model.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            model.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)

def inject_cascade_lowdim(model):
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

def run_lowdim(G_fixed, label, steps=200, init_C=None):
    """Train low-dim model: only C is updated."""
    ng=G_fixed.shape[0]
    torch.manual_seed(99)
    model=LowDimLM(D,N_HEADS,N_STU,G_fixed)
    inject_cascade_lowdim(model)
    if init_C is not None:
        with torch.no_grad(): model.C.copy_(torch.tensor(init_C,dtype=torch.float32))
    v0=eval_val(model,n=20)

    # Train C only
    for p in model.parameters(): p.requires_grad_(False)
    model.C.requires_grad_(True)
    opt=torch.optim.AdamW([model.C],lr=LR,betas=(0.9,0.95),weight_decay=0.01)
    checkpoints={}
    for step in range(1,steps+1):
        for pg in opt.param_groups: pg['lr']=clr(step,steps,max(10,steps//10))
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([model.C],1.0); opt.step()
        if step in [50,100,150,200]:
            vl=eval_val(model,n=20)
            checkpoints[step]=vl
            print(f"    [{label}] step {step}  val={vl:.4f}")
    for p in model.parameters(): p.requires_grad_(True)
    vf=eval_val(model)

    # Measure alignment
    E_curr=model.get_embeddings().detach().numpy()
    ga=gram_align(E_curr,E_teacher)
    ve=variance_explained(G_fixed,E_teacher)
    n_params=ng*VOCAB
    print(f"  [{label}] zero-shot={v0:.4f}  final={vf:.4f}  "
          f"gram_align={ga:.4f}  var_exp={ve:.1f}%  params={n_params:,}")
    return v0,vf,ga,ve,n_params,checkpoints

# ════════════════════════════════════════════════════════
# THEORY 1: Attractor pullback basis (48-dim)
# ════════════════════════════════════════════════════════
print("="*65)
print("THEORY 1: Attractor pullback basis")
print("  G_input = M_fwd^{-1} @ U14^T  [48 x D]")
print("="*65)

# M_fwd^{-1} maps attractor space back to embedding space
M_fwd_D=U14@M_fwd@U14.T+(np.eye(D)-U14@U14.T)
try:
    M_inv=np.linalg.inv(M_fwd_D+1e-3*np.eye(D))
    # G1: rows are M_fwd^{-1} applied to each attractor basis vector
    # U14: [D, ma] — attractor basis in D-space
    # G1[k,:] = M_fwd^{-1} @ U14[:,k]  (pullback of k-th attractor direction)
    G1=(M_inv@U14).T  # [ma, D]
    # Orthonormalize
    G1_q,_=np.linalg.qr(G1.T); G1=G1_q.T[:ma,:]
    G1=G1.astype(np.float32)
    ve1=variance_explained(G1,E_teacher)
    print(f"  G1 shape: {G1.shape}  variance_explained={ve1:.2f}%")
    # Init C from teacher projection
    C1_teacher=(E_teacher@G1.T).astype(np.float32)
    print(f"  Running with random C init...")
    r1_rand=run_lowdim(G1,"T1-random",200)
    print(f"  Running with teacher C init (oracle)...")
    r1_oracle=run_lowdim(G1,"T1-oracle",100,init_C=C1_teacher)
except Exception as e:
    print(f"  T1 failed: {e}")
    r1_rand=r1_oracle=(0,99,0,0,0,{})

# ════════════════════════════════════════════════════════
# THEORY 2: Kac-Moody rank basis (8-dim)
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("THEORY 2: Kac-Moody rank basis (8-dim)")
print("  G_km = top-8 eigenvectors of Gram_D")
print("="*65)

# Top-8 eigenvectors of Gram_D — the 8 most amplified directions
sv_g,Uv_g=np.linalg.eigh(Gram_D)
# Sort descending
idx=np.argsort(sv_g)[::-1]
G2=Uv_g[:,idx[:8]].T.astype(np.float32)  # [8, D]
ve2=variance_explained(G2,E_teacher)
print(f"  G2 shape: {G2.shape}  variance_explained={ve2:.2f}%")
C2_teacher=(E_teacher@G2.T).astype(np.float32)
print(f"  Running with random C init...")
r2_rand=run_lowdim(G2,"T2-random",200)
print(f"  Running with teacher C init (oracle)...")
r2_oracle=run_lowdim(G2,"T2-oracle",100,init_C=C2_teacher)

# Also test 16-dim and 32-dim KM basis
for km_dim in [16,32]:
    G_km=Uv_g[:,idx[:km_dim]].T.astype(np.float32)
    ve=variance_explained(G_km,E_teacher)
    C_t=(E_teacher@G_km.T).astype(np.float32)
    print(f"\n  KM-{km_dim} variance_explained={ve:.2f}%")
    run_lowdim(G_km,f"T2-{km_dim}dim",200)

# ════════════════════════════════════════════════════════
# THEORY 3: Laplacian + joint fine-tune (full D)
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("THEORY 3: Laplacian init + joint fine-tune (blocks frozen)")
print("  Initialize from Laplacian eigenmap, train emb+head only")
print("="*65)

# Build Laplacian embedding
print("  Computing Laplacian eigenmap...")
bigram=np.zeros((VOCAB,VOCAB),dtype=np.float32)
unigram=np.zeros(VOCAB,dtype=np.float32)
for k in range(len(train_ids)-1):
    a,b=train_ids[k],train_ids[k+1]
    if 0<=a<VOCAB and 0<=b<VOCAB:
        bigram[a,b]+=1; unigram[a]+=1
A=(bigram+bigram.T)/2
deg=A.sum(axis=1); deg=np.maximum(deg,1e-10)
d_inv_sqrt=1.0/np.sqrt(deg)
D_inv_sqrt=sp.diags(d_inv_sqrt)
A_sp=sp.csr_matrix(A)
L_norm=sp.eye(VOCAB)-D_inv_sqrt@A_sp@D_inv_sqrt
k_eig=min(D+2,VOCAB-1)
vals,vecs=spla.eigsh(L_norm,k=k_eig,which='SM')
idx_e=np.argsort(vals); vals=vals[idx_e]; vecs=vecs[:,idx_e]
skip=int(np.sum(vals<1e-8))
E_lap=vecs[:,skip:skip+D].astype(np.float32)
scale=teacher_norm/max(float(np.linalg.norm(E_lap,'fro')),1e-8)
E_lap=E_lap*scale
ga_lap=gram_align(E_lap,E_teacher)
print(f"  Laplacian gram_align={ga_lap:.4f}")

# Build standard LM with Laplacian init + Serre cascade
def run_joint(E_init, label, steps_sweep=[50,75,100,125,150,200]):
    torch.manual_seed(99)
    model=LM(D,N_HEADS,N_STU)
    with torch.no_grad():
        E_t=torch.tensor(E_init[:VOCAB,:D].copy(),dtype=torch.float32)
        model.te.weight.copy_(E_t)
    inject_cascade(model)
    v0=eval_val(model,n=20)
    print(f"  [{label}] zero-shot={v0:.4f}")

    # Joint: train emb+head only (blocks frozen)
    for p in model.parameters(): p.requires_grad_(False)
    model.te.weight.requires_grad_(True)
    model.head.weight.requires_grad_(True)  # tied but set explicitly
    trainable=[model.te.weight]
    opt=torch.optim.AdamW(trainable,lr=LR,betas=(0.9,0.95),weight_decay=0.01)

    results={}; step=0
    for target in steps_sweep:
        while step<target:
            for pg in opt.param_groups: pg['lr']=clr(step+1,max(steps_sweep),30)
            model.train(); x,y=get_batch(); _,loss=model(x,y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
            step+=1
        vl=eval_val(model,n=20)
        ga=gram_align(model.te.weight.data.numpy(),E_teacher)
        results[target]=vl
        print(f"    [{label}] step {target}  val={vl:.4f}  gram_align={ga:.4f}")
    for p in model.parameters(): p.requires_grad_(True)
    return v0,results

print("\n  T3a: Laplacian init + emb-only training (blocks frozen)...")
r3_lap=run_joint(E_lap,"T3-Laplacian")

print("\n  T3b: Teacher emb + emb-only training (oracle reference)...")
r3_teacher=run_joint(E_teacher,"T3-Teacher")

print("\n  T3c: Random init + emb-only training (baseline)...")
E_rand=np.random.randn(VOCAB,D).astype(np.float32)*0.02
r3_rand=run_joint(E_rand,"T3-Random")

# ════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  THREE THEORIES — FINAL RESULTS")
print("="*65)

print(f"""
  THEORY 1: Attractor pullback basis G = M_fwd^{{-1}} @ U14^T
    Dimension: {ma}  Params: {ma*VOCAB:,}
    Variance explained: {variance_explained(G1,E_teacher):.2f}%
    Random C init final:  val={r1_rand[1]:.4f}
    Oracle C init final:  val={r1_oracle[1]:.4f}

  THEORY 2: Kac-Moody rank basis (Gram eigenvectors)
    Dimension: 8  Params: {8*VOCAB:,}
    Variance explained: {ve2:.2f}%
    Random C init final:  val={r2_rand[1]:.4f}
    Oracle C init final:  val={r2_oracle[1]:.4f}

  THEORY 3: Laplacian init + emb-only training
    gram_align(Laplacian): {ga_lap:.4f}
    Steps to val<0.25:""")

for step,vl in sorted(r3_lap[1].items()):
    marker=" ← beats teacher" if vl<0.25 else ""
    print(f"      step {step:>3}: val={vl:.4f}{marker}")

print(f"""
  Reference:
    Teacher oracle (val=0.250)
    Serre+teacher emb+200CE (val=0.187)
    Full embedding random+200CE (val=0.185)

  THEORY VERDICT:
  T1 (attractor pullback):
    If oracle val << random val → basis is correct, need better C init
    If both vals ≈ teacher → attractor pullback is the right subspace

  T2 (KM rank basis, 8-dim):
    If oracle val << random val → KM basis is correct
    If both fail → 8 dims insufficient, try 16/32

  T3 (Laplacian + fine-tune):
    Steps to val<0.25 from Laplacian vs random gives the
    exact training reduction from Laplacian initialization.
    If Laplacian needs 90 steps vs random 150: 1.67x speedup.
""")
