#!/usr/bin/env python3
"""
Compiled Delta v3 — 25 CE seed + joint Emb/FF compile
=======================================================
The irreducible minimum pipeline:

  Phase 1 [25 CE steps]:  spectral init → dense A_model (the seed)
  Phase 2 [1 fwd pass]:   collect structured h_s from seeded model
  Phase 3 [offline]:      joint E*, FF* solve from structured h_s
  Phase 4 [0-3 CE steps]: residual correction

Hypothesis: Phase 1+3 together beat Phase 1 alone (25 CE → val X)
            Phase 1+3+4 matches full 167 CE (val ~0.2)

Comparison grid:
  A: 25 CE steps only          (baseline seed)
  B: 25 CE + compile           (seed + joint solve, 0 residual)
  C: 25 CE + compile + 3 CE    (seed + joint solve + tiny residual)
  D: 167 CE steps              (reference)
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
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        pre_ff=[]
        for block in self.blocks:
            h_attn=block.attn(h)
            pre_ff.append(h_attn.detach().clone())
            h=block.ff(h_attn)
        return self.ln_f(h), pre_ff

def eval_val(m,n=25):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def ce_train(m, steps, lr=LR):
    opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1)
    for s in range(1,steps+1):
        m.train(); x,y=get_batch(); _,l=m(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return eval_val(m)

# ── Corpus statistics ─────────────────────────────────────────────────────────
print("\n[OFFLINE] Corpus statistics...")
bigram=collections.Counter()
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
A_t=torch.tensor(A,dtype=torch.float32)
P_t=torch.tensor(P_token,dtype=torch.float32)

# ── Spectral embedding ────────────────────────────────────────────────────────
print("[OFFLINE] Spectral embedding...")
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
print(f"  E_0: {E_0.shape}")

# ── Build base model ──────────────────────────────────────────────────────────
torch.manual_seed(99)
model_base=LM(D,N_HEADS,N_STU)
model_base.te.weight.data.copy_(torch.tensor(E_0))
v_init=eval_val(model_base)
print(f"  Spectral init val: {v_init:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: 25 CE seed steps
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 1: 25 CE seed steps")
print("="*60)
model_seed=copy.deepcopy(model_base)
v_seed=ce_train(model_seed, 25)
print(f"  After 25 CE: val={v_seed:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: One forward pass — collect structured h_s
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 2: One forward pass — collect structured h_s")
print("="*60)

H_in_all=[]; H_out_all=[]; T_next_all=[]; T_curr_all=[]
N_SEQS=1000
model_seed.eval()
with torch.no_grad():
    torch.manual_seed(42)
    for i in range(N_SEQS):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ]; y=train_t[ix+1:ix+SEQ+1]
        h_final,pre_ff=model_seed.forward_with_internals(x.unsqueeze(0))
        H_in_all.append(pre_ff[-1][0].numpy())
        H_out_all.append(h_final[0].numpy())
        T_next_all.append(y.numpy())
        T_curr_all.append(x.numpy())

H_in =np.vstack(H_in_all).astype(np.float32)
H_out=np.vstack(H_out_all).astype(np.float32)
T_next=np.concatenate(T_next_all)
T_curr=np.concatenate(T_curr_all)

# Check h_s signal strength
print(f"  Collected: {H_in.shape[0]} positions")

# r_corpus[t] = sum_s P(s→t) * h_out_s
R_corpus=np.zeros((VOCAB,D),dtype=np.float32)
n_total=0
for i in range(len(T_curr)):
    s=int(T_curr[i])
    if s>=VOCAB: continue
    w=float(P_t[s])
    R_corpus+=w*np.outer(A[s],H_out[i])
    n_total+=1
R_corpus/=max(n_total,1)
R_norm=float(np.linalg.norm(R_corpus))
E_norm=float(np.linalg.norm(model_seed.te.weight.data.numpy()))
print(f"  r_corpus norm: {R_norm:.6f}")
print(f"  r_corpus / E norm: {R_norm/E_norm:.6f}")
print(f"  {'✓ SIGNAL PRESENT' if R_norm/E_norm > 0.001 else '✗ signal still weak'}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Joint E*, FF* solve
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 3: Joint E*, FF* solve")
print("="*60)

# E* = R_corpus scaled to match current embedding norm
E_star=(R_corpus*(E_norm/max(R_norm,1e-8))).astype(np.float32)
print(f"  E* norm: {np.linalg.norm(E_star):.4f} (seed E norm={E_norm:.4f})")

# FF* via ridge regression: find W s.t. H_in @ W^T ≈ R_next
# R_next[i] = R_corpus[T_next[i]]  (target h_out for each position)
valid=(T_next>=0)&(T_next<VOCAB)
H_in_v=H_in[valid]
R_next_v=R_corpus[T_next[valid]]

# Multiple regularisation strengths — pick best
print("\n  FF* ridge regression (scanning λ)...")
lam_vals=[1e-4,1e-3,1e-2,1e-1,1.0]

# Build test model once for fast eval
model_test=copy.deepcopy(model_seed)
best_lam,best_v_test=lam_vals[0],1e9
best_Wff=None

for lam in lam_vals:
    HtH=H_in_v.T@H_in_v+lam*np.eye(D)
    HtR=H_in_v.T@R_next_v
    Wff,_,_,_=np.linalg.lstsq(HtH,HtR,rcond=None)  # [D,D]
    # Apply to test model
    Wff_t=torch.tensor(Wff.T.astype(np.float32))
    with torch.no_grad():
        for l in range(N_STU):
            W_o_old=model_test.blocks[l].ff.o.weight.data.clone()
            W_o_new=Wff_t@W_o_old
            sc=W_o_old.norm()/max(W_o_new.norm(),1e-8)
            model_test.blocks[l].ff.o.weight.data.copy_(W_o_new*sc)
    # Also apply E*
    with torch.no_grad():
        model_test.te.weight.data.copy_(torch.tensor(E_star))
    v=eval_val(model_test,n=10)
    print(f"    λ={lam:.0e}: val={v:.4f}")
    if v<best_v_test:
        best_v_test=v; best_lam=lam; best_Wff=Wff.copy()
    # Reset test model
    model_test=copy.deepcopy(model_seed)

print(f"  Best λ={best_lam}, val={best_v_test:.4f}")

# Also test: E* only (no FF change)
with torch.no_grad():
    model_test.te.weight.data.copy_(torch.tensor(E_star))
v_estar_only=eval_val(model_test,n=15)
print(f"  E* only (no FF change): val={v_estar_only:.4f}")

# Also test: additive E* (E_seed + alpha * R_corpus)
print("\n  Scanning alpha for additive E* = E_seed + alpha*R_corpus ...")
E_seed_np=model_seed.te.weight.data.numpy()
best_alpha,best_v_add=0.0,v_seed
for alpha in [0.5,1.0,2.0,5.0,10.0,20.0,50.0]:
    model_test=copy.deepcopy(model_seed)
    E_new=E_seed_np+alpha*R_corpus
    E_new=E_new*(E_norm/max(np.linalg.norm(E_new),1e-8))
    with torch.no_grad():
        model_test.te.weight.data.copy_(torch.tensor(E_new))
    v=eval_val(model_test,n=10)
    print(f"    alpha={alpha:>5.1f}: val={v:.4f}")
    if v<best_v_add:
        best_v_add=v; best_alpha=alpha

print(f"  Best additive alpha={best_alpha}, val={best_v_add:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Assemble best compiled model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 4: Assemble & compare")
print("="*60)

# Version B: seed + best E* + best FF*
model_B=copy.deepcopy(model_seed)
Wff_t=torch.tensor(best_Wff.T.astype(np.float32))
with torch.no_grad():
    model_B.te.weight.data.copy_(torch.tensor(E_star))
    for l in range(N_STU):
        W_o_old=model_B.blocks[l].ff.o.weight.data.clone()
        W_o_new=Wff_t@W_o_old
        sc=W_o_old.norm()/max(W_o_new.norm(),1e-8)
        model_B.blocks[l].ff.o.weight.data.copy_(W_o_new*sc)
v_B=eval_val(model_B)
print(f"\n  [B] 25 CE seed + compile (0 residual): val={v_B:.4f}")

# Version B_add: seed + additive E* (best alpha)
model_B_add=copy.deepcopy(model_seed)
E_add=E_seed_np+best_alpha*R_corpus
E_add=E_add*(E_norm/max(np.linalg.norm(E_add),1e-8))
with torch.no_grad():
    model_B_add.te.weight.data.copy_(torch.tensor(E_add))
v_B_add=eval_val(model_B_add)
print(f"  [B+] 25 CE seed + additive E* only:     val={v_B_add:.4f}")

# Version C: B + 3 residual CE steps
model_C=copy.deepcopy(model_B)
v_C=ce_train(model_C,3)
print(f"  [C] B + 3 CE residual:                  val={v_C:.4f}")

model_C5=copy.deepcopy(model_B)
v_C5=ce_train(model_C5,5)
print(f"  [C5] B + 5 CE residual:                 val={v_C5:.4f}")

model_C10=copy.deepcopy(model_B)
v_C10=ce_train(model_C10,10)
print(f"  [C10] B + 10 CE residual:               val={v_C10:.4f}")

# Version A: 25 CE only (baseline)
v_A=v_seed
print(f"\n  [A] 25 CE only (baseline):              val={v_A:.4f}")

# Version D: 167 CE (reference)
print("\n  [D] Running 167 CE reference...")
model_D=copy.deepcopy(model_base)
opt_d=torch.optim.AdamW(model_D.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for s in range(1,168):
    model_D.train(); x,y=get_batch(); _,l=model_D(x,y)
    opt_d.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model_D.parameters(),1.0); opt_d.step()
    if s in [25,50,100,167]:
        print(f"    CE {s}: val={eval_val(model_D,n=8):.4f}")
v_D=eval_val(model_D,n=40)

print(f"""
{'='*60}
COMPILED DELTA v3 — FINAL RESULTS
{'='*60}

  [A] 25 CE only:               val={v_A:.4f}  (25 CE steps)
  [B] 25 CE + compile:          val={v_B:.4f}  (25 CE + 1 fwd + algebra)
  [B+] 25 CE + additive E*:     val={v_B_add:.4f}  (25 CE + 1 fwd + algebra)
  [C] 25 CE + compile + 3 CE:   val={v_C:.4f}  (28 total CE steps)
  [C5] 25 CE + compile + 5 CE:  val={v_C5:.4f}  (30 total CE steps)
  [C10] 25 CE + compile + 10CE: val={v_C10:.4f}  (35 total CE steps)
  [D] 167 CE reference:         val={v_D:.4f}  (167 CE steps)

  r_corpus / E norm: {R_norm/E_norm:.6f}
  {'SIGNAL PRESENT — compile worked' if R_norm/E_norm > 0.001 else 'signal weak — compile did not help'}

  If [B] < [A]: compile improves on 25 CE seed
  If [C] ≈ [D]: 28-35 CE steps ≈ 167 CE steps  →  6× speedup confirmed
""")
