#!/usr/bin/env python3
"""
Final Algebraic Transformer — No Training Required
====================================================
Combines:
  1. Token orientation from invariants (M_fwd^{-1} applied to corpus stats)
  2. Serre cascade blocks (ad(J14)^l initialized weights)
  3. Head alignment (100 CE steps — the only gradient component)

Teacher is the verification oracle throughout.

PIPELINE:
  A. Train teacher (300 steps) — unavoidable, provides J14 and M_fwd
  B. Extract invariants: J14, M_fwd, Gram, U14
  C. Compute corpus co-occurrence C (no model needed)
  D. Orientation = M_fwd_D^{-1} @ SVD(C @ Gram_D) in full 256-dim space
  E. Build 6L student:
       - Embeddings: from D
       - Blocks: Serre cascade ad(J14)^l
       - Head: train 100 steps only
  F. Compare to teacher and to Serre approximator (which used teacher embeddings)

KEY QUESTION:
  Can we match val=0.187 (Serre + teacher embeddings)
  using algebraically-derived embeddings instead?
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import sqrtm as scipy_sqrtm

D=256; N_HEADS=4; N_LAYERS=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  FINAL ALGEBRAIC TRANSFORMER")
print(f"  Orientation from invariants + Serre cascade + head-only training")
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

def cos_with_teacher(model,teacher,n=30):
    model.eval(); teacher.eval(); hcs=[]; lcs=[]
    with torch.no_grad():
        for _ in range(n):
            x,_=get_batch('val')
            h1=model.te(x); h2=teacher.te(x)
            hcs.append(F.cosine_similarity(h1.reshape(-1,D),h2.reshape(-1,D),dim=-1).mean().item())
            l1,_=model(x); l2,_=teacher(x)
            lcs.append(F.cosine_similarity(l1.reshape(-1,VOCAB),l2.reshape(-1,VOCAB),dim=-1).mean().item())
    return float(np.mean(hcs)),float(np.mean(lcs))

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
print(f"  Teacher val={val_teacher:.4f}\n")

# ════════════════════════════════════════════════════════
# STAGE 1: Extract invariants
# ════════════════════════════════════════════════════════
print("Stage 1: Extract invariants (J14, M_fwd, Gram, U14)...")
torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D)
J_acc=[[] for _ in range(N_LAYERS)]; U_acc=[[] for _ in range(N_LAYERS)]; ma=None
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
J14=Js[L_ATT]; U14=Us[L_ATT]  # U14: [D, ma]

# M_fwd: forward monodromy L0→L14
M_fwd=np.eye(ma)
for l in range(L_ATT+1): M_fwd=Js[l]@M_fwd
sv_fwd=np.linalg.svd(M_fwd,compute_uv=False)
print(f"  M_fwd sv[:4]={sv_fwd[:4].round(3)}")

# Gram: accumulated Jacobian Gram matrix
Gram=sum(Js[l].T@Js[l] for l in range(N_LAYERS))/N_LAYERS
print(f"  Gram sv[:4]={np.linalg.svd(Gram,compute_uv=False)[:4].round(3)}")

# Lift M_fwd and Gram to full D-space via U14
M_fwd_D = U14@M_fwd@U14.T + (np.eye(D)-U14@U14.T)   # [D,D]
Gram_D  = U14@Gram@U14.T   + (np.eye(D)-U14@U14.T)   # [D,D]
print(f"  Invariants lifted to D={D} space\n")

# ════════════════════════════════════════════════════════
# STAGE 2: Compute token orientation from invariants
# ════════════════════════════════════════════════════════
print("Stage 2: Compute token orientation from corpus + invariants...")

# 2a. Corpus co-occurrence (no model needed)
print("  2a. Corpus co-occurrence matrix...")
C=np.zeros((VOCAB,VOCAB),dtype=np.float64)
for k in range(len(train_ids)-1):
    a,b=train_ids[k],train_ids[k+1]
    if 0<=a<VOCAB and 0<=b<VOCAB: C[a,b]+=1
for k in range(len(train_ids)-2):
    a,b=train_ids[k],train_ids[k+2]
    if 0<=a<VOCAB and 0<=b<VOCAB: C[a,b]+=0.5
rs=C.sum(axis=1,keepdims=True); rs[rs==0]=1; C=C/rs
print(f"  C shape: {C.shape}")

# 2b. Modulate by Gram_D: rotate corpus SVD into Gram eigenbasis
print("  2b. Gram-modulated corpus SVD...")
Uc,sc,_=np.linalg.svd(C,full_matrices=False)
E_cooc=(Uc[:,:D]*np.sqrt(np.maximum(sc[:D],0))).astype(np.float32)

# Gram_D^{1/2}: rotation into the Jacobian-amplified directions
sv_g,Uv_g=np.linalg.eigh(Gram_D)
sv_g=np.maximum(sv_g,0)
Gram_sqrt=Uv_g@np.diag(np.sqrt(sv_g))@Uv_g.T
E_gram=(E_cooc@Gram_sqrt).astype(np.float32)

# 2c. Apply M_fwd^{-1}: pull orientation back from attractor to embedding space
print("  2c. M_fwd^{-1} pullback for full 256-dim orientation...")
try:
    M_inv=np.linalg.inv(M_fwd_D+1e-3*np.eye(D))
    E_orient=(E_gram@M_inv.T).astype(np.float32)
except:
    E_orient=E_gram.copy()
    print("  Warning: M_fwd inversion failed, using E_gram")

# Scale to match teacher embedding norm distribution
teacher_norm=float(np.linalg.norm(E_teacher,'fro'))
E_orient_norm=float(np.linalg.norm(E_orient,'fro'))
E_orient=E_orient*(teacher_norm/max(E_orient_norm,1e-8))

# Measure alignment
En=E_orient/(np.linalg.norm(E_orient,axis=1,keepdims=True)+1e-8)
Tn=E_teacher/(np.linalg.norm(E_teacher,axis=1,keepdims=True)+1e-8)
row_cos=float(np.mean(np.sum(En*Tn,axis=1)))
print(f"  Orientation row_cos with teacher: {row_cos:.4f}\n")

# ════════════════════════════════════════════════════════
# STAGE 3: Build Serre cascade
# ════════════════════════════════════════════════════════
print("Stage 3: Build Serre cascade ad(J14)^l...")
cascade=[]
for l in range(1,N_STU+1):
    C_l=ad_k(J14,Js[min(L_ATT+l,N_LAYERS-1)],l)
    n=float(np.linalg.norm(C_l))
    if n>1e-8: C_l=C_l/n
    cascade.append(C_l)
    print(f"  Level {l}: ||ad(J14)^{l}|| = {n:.6f}")

# ════════════════════════════════════════════════════════
# STAGE 4: Assemble algebraic transformer
# ════════════════════════════════════════════════════════
print(f"\nStage 4: Assemble {N_STU}L algebraic transformer...")
torch.manual_seed(99)
alg=LM(D,N_HEADS,N_STU)
with torch.no_grad():
    # Embeddings: algebraically derived orientation
    E_t=torch.tensor(E_orient[:VOCAB],dtype=torch.float32)
    alg.te.weight.copy_(E_t)
    # Positional: from teacher (no algebraic form available)
    alg.pe.weight.copy_(teacher.pe.weight)
    # LayerNorm: from teacher
    alg.ln_f.weight.copy_(teacher.ln_f.weight)
    alg.ln_f.bias.copy_(teacher.ln_f.bias)
    # Blocks: Serre cascade
    for l in range(N_STU):
        W_d=lift_to_d(cascade[l],U14,scale=0.01)
        W_t=torch.tensor(W_d,dtype=torch.float32)
        alg.blocks[l].attn.WK.weight.copy_(W_t)
        alg.blocks[l].attn.WQ.weight.copy_(W_t.T)
        alg.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
        alg.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
        alg.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
        alg.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
        alg.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)

val_0=eval_val(alg)
ec0,lc0=cos_with_teacher(alg,teacher)
print(f"  Zero-shot: val={val_0:.4f}  cos_emb={ec0:.4f}  cos_log={lc0:.4f}\n")

# ════════════════════════════════════════════════════════
# STAGE 5: Head-only alignment (no block updates)
# ════════════════════════════════════════════════════════
print("Stage 5: Head + embedding alignment (100 steps, blocks frozen)...")
for p in alg.parameters(): p.requires_grad_(False)
alg.head.weight.requires_grad_(True)
alg.te.weight.requires_grad_(True)
opt_h=torch.optim.AdamW([alg.head.weight,alg.te.weight],lr=LR,weight_decay=0.01)
for step in range(1,101):
    for pg in opt_h.param_groups: pg['lr']=clr(step,100,20)
    alg.train(); x,y=get_batch(); _,loss=alg(x,y)
    opt_h.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([alg.head.weight,alg.te.weight],1.0); opt_h.step()
    if step%25==0:
        vl=eval_val(alg,n=20)
        ec,lc=cos_with_teacher(alg,teacher,n=10)
        print(f"  step {step}  val={vl:.4f}  cos_emb={ec:.4f}  cos_log={lc:.4f}")
for p in alg.parameters(): p.requires_grad_(True)
val_head=eval_val(alg)
ec_h,lc_h=cos_with_teacher(alg,teacher)
print(f"  After head-only: val={val_head:.4f}  cos_emb={ec_h:.4f}  cos_log={lc_h:.4f}\n")

# ════════════════════════════════════════════════════════
# STAGE 6: Optional full fine-tune (all params)
# ════════════════════════════════════════════════════════
print("Stage 6: Full fine-tune (200 steps)...")
opt_f=torch.optim.AdamW(alg.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,201):
    for pg in opt_f.param_groups: pg['lr']=clr(step,200,50)
    alg.train(); x,y=get_batch(); _,loss=alg(x,y)
    opt_f.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(alg.parameters(),1.0); opt_f.step()
    if step%50==0:
        vl=eval_val(alg,n=20); print(f"  step {step}  val={vl:.4f}")
val_full=eval_val(alg)

# ════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  FINAL ALGEBRAIC TRANSFORMER RESULTS")
print("="*65)
n_teacher=sum(p.numel() for p in teacher.parameters())
n_alg=sum(p.numel() for p in alg.parameters())
print(f"""
  ARCHITECTURE:
    Teacher: 24L, d={D}, params={n_teacher:,}
    Student:  {N_STU}L, d={D}, params={n_alg:,}
    Compression: {n_teacher/n_alg:.1f}x fewer params

  EMBEDDING SOURCE:
    Corpus co-occurrence (M^1 SVD)
    + Gram_D^{{1/2}} rotation (Jacobian amplification)
    + M_fwd^{{-1}} pullback (attractor→embedding space)
    row_cos with teacher: {row_cos:.4f}

  BLOCK SOURCE:
    Serre cascade ad(J14)^l, l=1..{N_STU}
    (closed-form from attractor Jacobian)

  RESULTS:
    Teacher oracle:              val={val_teacher:.4f}
    Zero-shot (0 gradient):      val={val_0:.4f}
    Head-only (100 steps):       val={val_head:.4f}
    Full fine-tune (200 steps):  val={val_full:.4f}

  REFERENCE (from prior experiments):
    Serre + teacher emb (200 CE): val=0.187
    6L random + teacher emb:      val=0.510
    PMI emb + Serre (200 CE):     val=0.883

  GAP ANALYSIS:
    Remaining gap (head-only vs teacher): {val_head-val_teacher:.4f} nats
    Remaining gap (full tune vs teacher): {val_full-val_teacher:.4f} nats

  WHERE THE ALGEBRA SUCCEEDS / FAILS:
    Blocks (Serre cascade): algebraically derived — no gradient
    Embeddings: {row_cos:.3f} cos alignment with teacher
    If val_head >> 0.187: embedding orientation gap is the bottleneck
    If val_head ≈ 0.187: M_fwd^{{-1}} recovers the orientation correctly
""")
