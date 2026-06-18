#!/usr/bin/env python3
"""
Pass 12: Algebraic J_14 from Corpus Logit SVD
==============================================
With the TEACHER's J_14, the Serre cascade synced everything in one shot.
This pass computes J_14 algebraically from A_corpus — no teacher, no CE steps.

THE INSIGHT:
  At the attractor, the attention scores match the corpus bigram distribution:
    softmax(E W_Q^T W_K E^T / sqrt(d))[s,:] ≈ A_corpus[x_s, :]
  
  Taking the inverse softmax (logit):
    (W_Q E[s]) · (W_K E[t]) / sqrt(d) = logit(A_corpus)[x_s, x_t]
  
  This means:
    (W_Q E) @ (W_K E)^T / sqrt(d) = logit(A_corpus)   [rank-D factorization]
  
  SVD gives the solution directly:
    logit(A_corpus) ≈ U S V^T   (rank D)
    W_Q E_0 = sqrt(sqrt(d)) * U[:,:D] @ diag(sqrt(S[:D]))
    W_K E_0 = sqrt(sqrt(d)) * V[:,:D]^T @ diag(sqrt(S[:D]))
  
  Then:
    W_Q = WQ_target @ pinv(E_0)
    W_K = WK_target @ pinv(E_0)

This is J_14 in one shot: O(V^2 D) matrix operations, no gradient.

PREDICTION:
  With algebraic W_K* (= J_14 from corpus), the slow-manifold projection
  should work at high efficiency — because now ALL W_K is at θ*_attn.
  Expected: val drops from 0.92 to ~0.3-0.5 in one shot.
"""
import json, math, warnings, collections
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F
import copy, os, sys

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

if VOCAB != 1017:
    print(f"ERROR: VOCAB={VOCAB}. Run: python build_corpus.py --out /tmp/ --loops 300")
    sys.exit(1)

if not os.path.exists('/tmp/model_post_pass6.pt'):
    print("ERROR: /tmp/model_post_pass6.pt not found.")
    print("Run: python build_pass6_checkpoint.py")
    sys.exit(1)

print(f"VOCAB={VOCAB}, train={len(train_ids)} ({len(train_ids)//1364} loops)")

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

class Attn(nn.Module):
    def __init__(self,d,nh):
        super().__init__(); self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
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

def eval_val(m,n=30):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

# ── Step 1: Build A_corpus and logit(A_corpus) ────────────────────────────────
print("\nStep 1: Building A_corpus and logit...")
bigram = collections.Counter()
for i in range(len(train_ids)-1):
    a,b = train_ids[i], train_ids[i+1]
    if a<VOCAB and b<VOCAB: bigram[(a,b)] += 1

A = np.zeros((VOCAB, VOCAB), dtype=np.float64)
for (a,b), cnt in bigram.items(): A[a,b] += cnt
row_sums = A.sum(1, keepdims=True)

# Only rows with observations contribute
active = (row_sums > 0).flatten()
A[active] /= row_sums[active]
# Fill unobserved rows with uniform
A[~active] = 1.0 / VOCAB

# Logit: log A - logsumexp(log A) per row
log_A = np.where(A > 1e-12, np.log(A), -30.0)
logit_A = log_A - log_A.mean(1, keepdims=True)  # center rows
logit_A = logit_A.astype(np.float32)

print(f"  A_corpus nnz: {(A>1e-10).sum()}")
print(f"  logit(A) norm: {np.linalg.norm(logit_A):.4f}")

# ── Step 2: SVD of logit(A_corpus) → algebraic W_Q, W_K ─────────────────────
print("\nStep 2: SVD of logit(A_corpus) → algebraic J_14...")
U, S, Vt = np.linalg.svd(logit_A, full_matrices=False)
print(f"  Singular values (top 10): {S[:10].round(2)}")
print(f"  Top-D energy: {(S[:D]**2).sum()/(S**2).sum()*100:.1f}%")

# Rank-D factorization: logit_A ≈ (U[:,:D]*sqrt(S[:D])) @ (Vt[:D,:]*sqrt(S[:D]))^T
# W_Q E_0 = scale * U[:,:D] * sqrt(S[:D])   → "query targets" [VOCAB, D]
# W_K E_0 = scale * Vt[:D,:].T * sqrt(S[:D]) → "key targets" [VOCAB, D]
scale = float(D ** 0.25)  # sqrt(sqrt(d)) normalization for score = Q·K/sqrt(d)
WQ_tgt = (scale * U[:, :D] * np.sqrt(S[:D])).astype(np.float32)  # [VOCAB, D]
WK_tgt = (scale * Vt[:D, :].T * np.sqrt(S[:D])).astype(np.float32)  # [VOCAB, D]

# Reconstruction check
recon = WQ_tgt @ WK_tgt.T / math.sqrt(D)
resid = np.linalg.norm(recon - logit_A) / np.linalg.norm(logit_A)
print(f"  Reconstruction error: {resid:.4f}")
print(f"  WQ_tgt norm: {np.linalg.norm(WQ_tgt):.4f}")
print(f"  WK_tgt norm: {np.linalg.norm(WK_tgt):.4f}")

# ── Step 3: Load model, extract E_0, compute W_Q and W_K ─────────────────────
print("\nStep 3: Loading post-Pass-6 model and computing algebraic W_Q, W_K...")
model = LM(D, N_HEADS, N_STU)
model.load_state_dict(torch.load('/tmp/model_post_pass6.pt', weights_only=True))
v_start = eval_val(model, n=20)
print(f"  Post-Pass-6 val: {v_start:.4f}")

E0 = model.te.weight.data.numpy().copy()  # [VOCAB, D]
print(f"  E_0 norm: {np.linalg.norm(E0):.4f}")

# Solve: W_Q @ E_0^T = WQ_tgt^T  →  W_Q = WQ_tgt^T @ pinv(E_0^T)
#   i.e., W_Q [D,D] such that E_0 @ W_Q^T = WQ_tgt
# lstsq: min ||E_0 @ W_Q^T - WQ_tgt||
WQ_new, _, _, _ = np.linalg.lstsq(E0, WQ_tgt, rcond=None)  # [D, D]
WK_new, _, _, _ = np.linalg.lstsq(E0, WK_tgt, rcond=None)  # [D, D]

# Normalize to same scale as original weights
WQ_orig_norm = model.blocks[0].attn.WQ.weight.data.norm().item()
WK_orig_norm = model.blocks[0].attn.WK.weight.data.norm().item()
WQ_new = WQ_new.T.astype(np.float32)  # [D, D]
WK_new = WK_new.T.astype(np.float32)  # [D, D]
WQ_new *= WQ_orig_norm / max(np.linalg.norm(WQ_new), 1e-8)
WK_new *= WK_orig_norm / max(np.linalg.norm(WK_new), 1e-8)
print(f"  WQ_new norm: {np.linalg.norm(WQ_new):.4f} (orig: {WQ_orig_norm:.4f})")
print(f"  WK_new norm: {np.linalg.norm(WK_new):.4f} (orig: {WK_orig_norm:.4f})")

# ── Step 4: Apply algebraic J_14 to all layers ────────────────────────────────
print("\nStep 4: Applying algebraic J_14 to all layers...")
model_j14 = copy.deepcopy(model)
WQ_t = torch.tensor(WQ_new, dtype=torch.float32)
WK_t = torch.tensor(WK_new, dtype=torch.float32)
with torch.no_grad():
    for l in range(N_STU):
        model_j14.blocks[l].attn.WQ.weight.data.copy_(WQ_t)
        model_j14.blocks[l].attn.WK.weight.data.copy_(WK_t)

v_j14 = eval_val(model_j14, n=30)
print(f"  After algebraic J_14: val={v_j14:.4f}")

# ── Step 5: Fine-tune comparison ──────────────────────────────────────────────
print("\n" + "="*65)
print("COMPARISON")
print("="*65)

# A: Pass 7 from original post-Pass-6
model_A = copy.deepcopy(model)
opt_a = torch.optim.AdamW(model_A.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
print("\n[A] Pass 7: 167 CE from post-Pass-6 (reference)")
for step in range(1,168):
    model_A.train(); x,y=get_batch(); _,loss=model_A(x,y)
    opt_a.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model_A.parameters(),1.0); opt_a.step()
    if step in [25,50,100,167]:
        print(f"  CE {step}: val={eval_val(model_A,n=10):.4f}")
val_A = eval_val(model_A, n=40)
print(f"  PATH A FINAL: val={val_A:.4f}")

# B: J_14 + 167 CE
model_B = copy.deepcopy(model_j14)
opt_b = torch.optim.AdamW(model_B.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
print("\n[B] Algebraic J_14 + 167 CE")
for step in range(1,168):
    model_B.train(); x,y=get_batch(); _,loss=model_B(x,y)
    opt_b.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model_B.parameters(),1.0); opt_b.step()
    if step in [25,50,100,167]:
        print(f"  CE {step}: val={eval_val(model_B,n=10):.4f}")
val_B = eval_val(model_B, n=40)
print(f"  PATH B FINAL: val={val_B:.4f}")

# C: J_14 + 25 CE only
model_C = copy.deepcopy(model_j14)
opt_c = torch.optim.AdamW(model_C.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
print("\n[C] Algebraic J_14 + 25 CE only")
for step in range(1,26):
    model_C.train(); x,y=get_batch(); _,loss=model_C(x,y)
    opt_c.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model_C.parameters(),1.0); opt_c.step()
val_C = eval_val(model_C, n=30)
print(f"  PATH C FINAL: val={val_C:.4f}")

print(f"""
{'='*65}
ALGEBRAIC J_14 RESULTS
{'='*65}

  Post-Pass-6 start:          val={v_start:.4f}
  After algebraic J_14 only:  val={v_j14:.4f}
  
  [A] Pass 7 (167 CE, no J_14):   val={val_A:.4f}
  [B] J_14 + 167 CE:              val={val_B:.4f}
  [C] J_14 + 25 CE:               val={val_C:.4f}

  If B < A: algebraic J_14 gives better basin than CE convergence
  If C ≈ A: J_14 replaces ~142 CE steps (167-25)
  
  The algebraic J_14 is:
    logit(A_corpus) ≈ U S V^T   (rank-D SVD)
    W_K = pinv(E_0)^T @ (sqrt(sqrt(d)) * Vt[:D].T * sqrt(S[:D]))
    No CE steps required — pure linear algebra from corpus statistics.
""")
