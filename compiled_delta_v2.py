#!/usr/bin/env python3
"""
Compiled Delta v2 — Joint Emb+FF solve with structured h_s
============================================================
Fixes v1 failure (gap=2.5 nats) by:

  1. Applying compiled WQ*, WK* FIRST
  2. Running ONE forward pass to get structured h_s
  3. Solving Emb and FF JOINTLY from those h_s:
       FF* = ridge regression: H_in -> R_corpus
       E*  = r_corpus (corpus-weighted h_out)

The Emb-FF conjugate relationship:
  E[t] must align with h_out_s = LN(h_in_s + FF*(h_in_s))
  FF* maps h_in → h_out aligned with next-token embedding
  They form one coherent system — solved jointly in one pass.

Pipeline:
  [Offline]   E_0      ← Laplacian SVD
  [Offline]   WQ*, WK* ← logit(A^k) SVD @ pinv(E_0)
  [1 fwd]     H_in     ← pre-FF hidden states with WQ*, WK*
  [Offline]   FF*      ← ridge regression H_in → R_corpus
  [Offline]   E*       ← corpus-weighted h_out after FF*
  [Apply]     Δθ = (E*, WQ*, WK*, FF*) - θ_0
  [0-5 CE]    residual correction
"""
import json, math, warnings, collections, os, copy, sys
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f):
        print(f"ERROR: {f} missing. Run: python build_corpus.py --out /tmp/ --loops 300")
        sys.exit(1)

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)
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
    def forward_with_internals(self,x):
        """Returns (final_h, list of pre-FF hidden states per layer)."""
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        pre_ff = []
        for block in self.blocks:
            h_attn = block.attn(h)   # after attention
            pre_ff.append(h_attn.detach().clone())
            h = block.ff(h_attn)     # after FF
        return self.ln_f(h), pre_ff

def eval_val(m,n=30):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

# ── Step 1: Corpus statistics ─────────────────────────────────────────────────
print("\n[OFFLINE] Step 1: Corpus statistics...")
bigram = collections.Counter()
for i in range(len(train_ids)-1):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB: bigram[(a,b)]+=1
freq=np.zeros(VOCAB)
for t in train_ids:
    if t<VOCAB: freq[t]+=1
P_token=(freq/freq.sum()).astype(np.float32)

A=np.zeros((VOCAB,VOCAB),dtype=np.float32)
for (a,b),cnt in bigram.items(): A[a,b]+=cnt
A/=(A.sum(1,keepdims=True)+1e-10)
# Most likely next token per current token
next_token_map = np.argmax(A, axis=1)  # [VOCAB]
print(f"  A_corpus nnz={(A>1e-10).sum()}, next_token_map computed")

# ── Step 2: Spectral embedding ────────────────────────────────────────────────
print("\n[OFFLINE] Step 2: Spectral embedding...")
rows,cols,vals=[],[],[]
for (a,b),cnt in bigram.items():
    rows.append(a); cols.append(b); vals.append(float(cnt))
W_sp=sp.csr_matrix((vals,(rows,cols)),shape=(VOCAB,VOCAB),dtype=np.float32)
W_sp=W_sp+W_sp.T
d_inv=np.array(1.0/(W_sp.sum(1)+1e-8)).flatten()
Dsi=sp.diags(np.sqrt(d_inv))
L_sym=sp.eye(VOCAB)-Dsi@W_sp@Dsi
evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)
idx=np.argsort(evals)
evecs=evecs[:,idx][:,1:D+1]
scales=1.0/(np.sqrt(evals[idx[1:D+1]])+1e-8)
E_0=(evecs*scales[np.newaxis,:]).astype(np.float32)
E_0=(E_0/(E_0.std()+1e-8)*0.02)
print(f"  E_0: {E_0.shape}, std={E_0.std():.4f}")

# ── Step 3: WQ*, WK* from logit(A^k) SVD ─────────────────────────────────────
print("\n[OFFLINE] Step 3: WQ*, WK* from logit(A^k) SVD (k=1..6)...")
A_power=A.copy().astype(np.float64)
best_k,best_err=1,1e9
best_WQ,best_WK=None,None
for k in range(1,7):
    if k>1: A_power=A_power@A
    logA=np.where(A_power>1e-12,np.log(A_power),-30.0)
    logitA=(logA-logA.mean(1,keepdims=True)).astype(np.float32)
    U,S,Vt=np.linalg.svd(logitA,full_matrices=False)
    sc=float(D**0.25)
    WQ_tgt=(sc*U[:,:D]*np.sqrt(S[:D])).astype(np.float32)
    WK_tgt=(sc*Vt[:D,:].T*np.sqrt(S[:D])).astype(np.float32)
    WQ_w,_,_,_=np.linalg.lstsq(E_0,WQ_tgt,rcond=None)
    WK_w,_,_,_=np.linalg.lstsq(E_0,WK_tgt,rcond=None)
    recon=(E_0@WQ_w)@(E_0@WK_w).T/math.sqrt(D)
    err=float(np.linalg.norm(recon-logitA)/np.linalg.norm(logitA))
    print(f"  k={k}: recon error={err:.4f}")
    if err<best_err:
        best_err=err; best_k=k
        best_WQ=WQ_w.T.astype(np.float32)
        best_WK=WK_w.T.astype(np.float32)
print(f"  Using k={best_k}")

# ── Step 4: Build model with WQ*, WK* and run ONE forward pass ────────────────
print("\n[1 FWD PASS] Step 4: Build WQ*/WK* model, collect structured h_s...")
torch.manual_seed(99)
model_wqwk=LM(D,N_HEADS,N_STU)
model_wqwk.te.weight.data.copy_(torch.tensor(E_0))

WQ_t=torch.tensor(best_WQ)
WK_t=torch.tensor(best_WK)
init_wq_norm=model_wqwk.blocks[0].attn.WQ.weight.data.norm()
init_wk_norm=model_wqwk.blocks[0].attn.WK.weight.data.norm()
WQ_t=WQ_t*(init_wq_norm/max(WQ_t.norm(),1e-8))
WK_t=WK_t*(init_wk_norm/max(WK_t.norm(),1e-8))
with torch.no_grad():
    for l in range(N_STU):
        model_wqwk.blocks[l].attn.WQ.weight.data.copy_(WQ_t)
        model_wqwk.blocks[l].attn.WK.weight.data.copy_(WK_t)

v_wqwk=eval_val(model_wqwk)
print(f"  Model with WQ*+WK*: val={v_wqwk:.4f}")

# Collect H_in (pre-FF) and H_out (post-FF) and next tokens
# for the joint Emb+FF solve
H_in_all  = []   # pre-FF hidden states at last layer
H_out_all = []   # post-FF (= final) hidden states at last layer
T_next_all= []   # next token at each position
T_curr_all= []   # current token at each position

N_SEQS = 1000
model_wqwk.eval()
A_t = torch.tensor(A, dtype=torch.float32)

with torch.no_grad():
    torch.manual_seed(42)
    for i in range(N_SEQS):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ]; y=train_t[ix+1:ix+SEQ+1]
        h_final, pre_ff_list = model_wqwk.forward_with_internals(x.unsqueeze(0))
        # Use last layer's pre-FF as H_in
        h_in  = pre_ff_list[-1][0]   # [SEQ, D]
        h_out = h_final[0]           # [SEQ, D] (post all layers + LN)
        H_in_all.append(h_in.numpy())
        H_out_all.append(h_out.numpy())
        T_next_all.append(y.numpy())
        T_curr_all.append(x.numpy())

H_in  = np.vstack(H_in_all).astype(np.float32)   # [N_SEQS*SEQ, D]
H_out = np.vstack(H_out_all).astype(np.float32)   # [N_SEQS*SEQ, D]
T_next= np.concatenate(T_next_all)                 # [N_SEQS*SEQ]
T_curr= np.concatenate(T_curr_all)                 # [N_SEQS*SEQ]
print(f"  Collected H_in/H_out: {H_in.shape}")
print(f"  H_in norm: {np.linalg.norm(H_in[0]):.4f} per token (should be ~1 from LN)")
print(f"  H_out norm: {np.linalg.norm(H_out[0]):.4f} per token")

# ── Step 5: Compute r_corpus from REAL h_s ────────────────────────────────────
print("\n[OFFLINE] Step 5: r_corpus from structured h_s...")
# r_corpus[t] = sum_s P(s→t) * h_out_s
# = (P_token[T_curr] * A[T_curr, t]) weighted sum of H_out
R_corpus = np.zeros((VOCAB, D), dtype=np.float32)
counts   = np.zeros(VOCAB, dtype=np.float32)
for i in range(len(T_curr)):
    s = int(T_curr[i])
    if s >= VOCAB: continue
    w = P_token[s]
    # r_corpus[t] += w * A[s,t] * h_out_s  for all t
    R_corpus += w * np.outer(A[s], H_out[i])
    counts[s] += 1.0
R_corpus /= max(float(counts.sum()), 1)
R_norm = float(np.linalg.norm(R_corpus))
print(f"  r_corpus norm: {R_norm:.6f}")
print(f"  r_corpus / E_0 norm: {R_norm/float(np.linalg.norm(E_0)):.6f}")
# This should now be NON-TRIVIAL (h_s is structured after WQ*, WK*)

# ── Step 6: Joint Emb+FF solve ────────────────────────────────────────────────
print("\n[OFFLINE] Step 6: Joint Emb+FF solve...")
# 
# FF* maps h_in → h_out such that h_out aligns with E*[next_token]
# E*[t] = R_corpus[t] (scaled)
#
# The FF network: FF(h) = W_o(silu(W_g h) * W_v h)
# For a LINEARIZED version: FF*(h) ≈ W_ff * h
# W_ff = argmin ||H_in @ W_ff^T - R_next||^2
# where R_next[i] = R_corpus[T_next[i]]  (target output for each position)
#
# Ridge regression: W_ff = (H_in^T H_in + λI)^{-1} H_in^T R_next

R_next = R_corpus[T_next.clip(0, VOCAB-1)]  # [N, D]: target h_out per position
# Only use positions where T_next is valid
valid = (T_next >= 0) & (T_next < VOCAB)
H_in_v  = H_in[valid]
R_next_v = R_next[valid]

# Ridge regression
lam = 1e-3
HtH = H_in_v.T @ H_in_v + lam * np.eye(D)  # [D,D]
HtR = H_in_v.T @ R_next_v                    # [D,D]
W_ff_linear, _,_,_ = np.linalg.lstsq(HtH, HtR, rcond=None)  # [D,D]

# Reconstruction quality
R_pred = H_in_v @ W_ff_linear
resid = float(np.linalg.norm(R_pred - R_next_v) / (np.linalg.norm(R_next_v) + 1e-8))
print(f"  FF* linear solve: reconstruction error = {resid:.4f}")
print(f"  W_ff_linear norm: {np.linalg.norm(W_ff_linear):.4f}")

# E* = R_corpus scaled to E_0 norm
E_scale = float(np.linalg.norm(E_0))
R_scale = float(np.linalg.norm(R_corpus))
E_star = R_corpus * (E_scale / max(R_scale, 1e-8))
print(f"  E* norm: {np.linalg.norm(E_star):.4f} (matches E_0 norm={E_scale:.4f})")

# ── Step 7: Assemble compiled model ──────────────────────────────────────────
print("\n[COMPILE] Step 7: Assembling compiled model...")
torch.manual_seed(99)
model_compiled = LM(D, N_HEADS, N_STU)

# Apply E*
model_compiled.te.weight.data.copy_(torch.tensor(E_star))

# Apply WQ*, WK*
with torch.no_grad():
    for l in range(N_STU):
        model_compiled.blocks[l].attn.WQ.weight.data.copy_(WQ_t)
        model_compiled.blocks[l].attn.WK.weight.data.copy_(WK_t)

v_no_ff = eval_val(model_compiled, n=20)
print(f"  E* + WQ* + WK* (no FF): val={v_no_ff:.4f}")

# Apply FF* as a linear correction to the FF output weights W_o
# The FF network output: W_o(silu(W_g h) * W_v h)
# Linearized correction: multiply W_o by W_ff_linear
# W_o_new = W_ff_linear^T @ W_o_old  (maps the output toward R_corpus)
W_ff_t = torch.tensor(W_ff_linear.T.astype(np.float32))  # [D,D]
with torch.no_grad():
    for l in range(N_STU):
        W_o_old = model_compiled.blocks[l].ff.o.weight.data.clone()  # [D, 2D]
        # W_o maps from 2D → D, we want to rotate the D output space
        # W_o_new[:, :] = W_ff_t @ W_o_old
        W_o_new = W_ff_t @ W_o_old  # [D, 2D]
        # Normalize to preserve scale
        scale = W_o_old.norm() / max(W_o_new.norm(), 1e-8)
        model_compiled.blocks[l].ff.o.weight.data.copy_(W_o_new * scale)

v_compiled = eval_val(model_compiled, n=30)
print(f"  Full compiled (E*+WQ*+WK*+FF*): val={v_compiled:.4f}")

# ── Reference ─────────────────────────────────────────────────────────────────
print("\n[REFERENCE] 167 CE steps from spectral init...")
torch.manual_seed(99)
model_ref=LM(D,N_HEADS,N_STU)
model_ref.te.weight.data.copy_(torch.tensor(E_0))
opt=torch.optim.AdamW(model_ref.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,168):
    model_ref.train(); x,y=get_batch(); _,l=model_ref(x,y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model_ref.parameters(),1.0); opt.step()
    if step in [25,50,100,167]:
        print(f"  CE {step}: val={eval_val(model_ref,n=8):.4f}")
v_ref=eval_val(model_ref,n=40)
print(f"  Reference: val={v_ref:.4f}")

# ── Residual correction ────────────────────────────────────────────────────────
print("\n[RESIDUAL] CE steps from compiled init...")
model_res=copy.deepcopy(model_compiled)
opt_r=torch.optim.AdamW(model_res.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,26):
    model_res.train(); x,y=get_batch(); _,l=model_res(x,y)
    opt_r.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model_res.parameters(),1.0); opt_r.step()
    if step in [1,3,5,10,25]:
        print(f"  CE step {step:>3}: val={eval_val(model_res,n=15):.4f}")
v_res=eval_val(model_res,n=30)

print(f"""
{'='*65}
COMPILED DELTA v2 RESULTS
{'='*65}

  Spectral init (E_0 only):       val={eval_val(model_wqwk,n=10):.4f}
  + WQ* + WK* (no fwd pass yet):  val={v_wqwk:.4f}
  Compiled (0 CE, with FF*):       val={v_compiled:.4f}
  Compiled + 25 CE steps:          val={v_res:.4f}
  Reference (167 CE steps):        val={v_ref:.4f}

  Gap compiled vs reference:       {v_compiled-v_ref:+.4f} nats
  Gap compiled+25CE vs reference:  {v_res-v_ref:+.4f} nats

  r_corpus norm: {R_norm:.6f}  (was 0.000058 in v1 — should be larger now)

KEY: if r_corpus norm >> 0.001 AND compiled val << 4.4 → WQ*/WK* structured h_s
""")

torch.save(model_compiled.state_dict(), '/tmp/model_compiled_v2.pt')
print("Saved → /tmp/model_compiled_v2.pt")
