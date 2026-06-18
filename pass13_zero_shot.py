#!/usr/bin/env python3
"""
Pass 13: Zero-Shot Compiler — Symbolic Support Pruning → 1 LM Step
====================================================================
Eliminates Phase 1 (25 CE steps) entirely via symbolic support pruning.

THE INSIGHT (from Joyal AU analysis):
  Phase 1's 25 CE steps do two things:
  1. Learn attention focus: W_V learns to map E[t] → E[perm(t)]
  2. Prune species support: r_corpus drops from 0.000918 → 0.000086
     but becomes SHARPER (concentrated on 1014 prime paths)

  Both can be computed algebraically:
  1. W_V* = permutation operator: E_next = E_0[perm(t)]
     → r_corpus with W_V* gives ratio ~0.001 (same order as after 25 CE)
  2. Sparse prime paths: R_sparse[t] = E_0[t] * P(t is successor)
     → directly encodes the 1014 support skeleton

  Combining: E* = Serre cascade from R_sparse → 1 LM step

PIPELINE (Phase 0 only → 1 LM):
  [Offline] E_0         ← Laplacian eigenvectors
  [Offline] E_next      ← E_0[perm(t)] (permutation of embeddings)
  [Offline] R_sparse    ← sparse prime paths projection
  [Offline] E*          ← Serre cascade from R_sparse
  [1 LM step]           ← LayerNorm lock-in handshake
  Total: 0 CE + 1 LM = 1 evaluation

If this works: the 25 CE minimum is eliminated.
If it fails: confirms 25 CE as irreducible.
"""
import json, math, warnings, collections, os, copy, sys
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f): print(f"ERROR: {f} missing."); sys.exit(1)

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids=list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t=torch.tensor(val_ids,dtype=torch.long)
print(f"VOCAB={VOCAB}")

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return torch.stack([data[i:i+SEQ] for i in ix]),torch.stack([data[i+1:i+SEQ+1] for i in ix])

class Attn(nn.Module):
    def __init__(self,d,nh):
        super().__init__(); self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
        self.WQ=nn.Linear(d,d,bias=False); self.WK=nn.Linear(d,d,bias=False)
        self.WV=nn.Linear(d,d,bias=False); self.op=nn.Linear(d,d,bias=False)
        self.ln=nn.LayerNorm(d)
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h):
        B,S,D_=h.shape; H=self.nh; dh=self.dh
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2); K=self.WK(h).view(B,S,H,dh).transpose(1,2)
        V=self.WV(h).view(B,S,H,dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/self.sc
        mask=torch.triu(torch.ones(S,S,device=h.device),diagonal=1).bool()
        sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        return self.ln(h+self.op((F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D_)))
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
        self.ln_f=nn.LayerNorm(d); self.head=nn.Linear(d,VOCAB,bias=False)
        self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def flat_params(self): return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat(self,f):
        idx=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(f[idx:idx+n].reshape(p.shape)); idx+=n

def eval_val(m,n=25):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def lm_step(model, mu=0.1, n_grad=20, n_hvp=8, n_cg=8):
    model.zero_grad()
    ls=[]
    for _ in range(n_grad): x,y=get_batch(); _,l=model(x,y); ls.append(l)
    loss=torch.stack(ls).mean(); loss.backward()
    g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                 for p in model.parameters()]).detach()
    model.zero_grad()
    def hvp(v):
        model.zero_grad()
        ls2=[]
        for _ in range(n_hvp): x,y=get_batch(); _,l=model(x,y); ls2.append(l)
        loss2=torch.stack(ls2).mean()
        grads=torch.autograd.grad(loss2,list(model.parameters()),create_graph=True)
        gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
        hv=torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)
        model.zero_grad()
        return torch.cat([h.flatten() for h in hv]).detach()
    d=torch.zeros_like(g); r=-g.clone(); p_cg=r.clone(); rr=float((r*r).sum())
    for _ in range(n_cg):
        Hp=hvp(p_cg)+mu*p_cg; alpha=rr/max(float((p_cg*Hp).sum()),1e-10)
        d+=alpha*p_cg; r-=alpha*Hp; rr_new=float((r*r).sum())
        p_cg=r+(rr_new/max(rr,1e-10))*p_cg; rr=rr_new
    w0=model.flat_params()
    for scale in [1.0,0.5,0.25,0.1]:
        model.set_flat(w0+scale*d)
        v_new=eval_val(model,n=8)
        if scale==0.1 or v_new<eval_val(copy.deepcopy(model),n=3)+0.2:
            return eval_val(model,n=12), True
    model.set_flat(w0); return float(loss), False

# ── Corpus ─────────────────────────────────────────────────────────────────────
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
perm={}
for i in range(len(train_ids)-1):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB and a not in perm: perm[a]=b

# ── Spectral embedding ─────────────────────────────────────────────────────────
rows,cols,vals_sp=[],[],[]
for (a,b),cnt in bigram.items(): rows.append(a); cols.append(b); vals_sp.append(float(cnt))
W_sp=sp.csr_matrix((vals_sp,(rows,cols)),shape=(VOCAB,VOCAB),dtype=np.float32)
W_sp=W_sp+W_sp.T
d_inv=np.array(1.0/(W_sp.sum(1)+1e-8)).flatten()
Dsi=sp.diags(np.sqrt(d_inv)); L_sym=sp.eye(VOCAB)-Dsi@W_sp@Dsi
evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)
idx_s=np.argsort(evals); evecs=evecs[:,idx_s][:,1:D+1]
sc=1.0/(np.sqrt(evals[idx_s[1:D+1]])+1e-8)
E_0=(evecs*sc[np.newaxis,:]).astype(np.float32)
E_0=(E_0/(E_0.std()+1e-8)*0.02)
E_norm=float(np.linalg.norm(E_0))

print(f"\n[OFFLINE] Corpus: {len(perm)} tokens with successor")
print(f"[OFFLINE] Species support: {len(bigram)} prime paths (0.13% of V²)")

# ── Symbolic support pruning (mimic Phase 1 algebraically) ────────────────────
print("\n[SYMBOLIC PRUNING] Building sparse prime paths support...")

# Method 1: W_V = permutation operator → r_corpus via E_next
E_next=np.array([E_0[perm.get(t,t)] for t in range(VOCAB)],dtype=np.float32)
R_perm=(A.T@np.diag(P_token)@E_next).astype(np.float32)

# Method 2: Sparse prime paths — only observed successors
observed_next=set(perm.values())
count_as_next={}
for s,nt in perm.items():
    count_as_next[nt]=count_as_next.get(nt,0)+1
R_sparse=np.zeros((VOCAB,D),dtype=np.float32)
for t in range(VOCAB):
    if t in observed_next:
        R_sparse[t]=E_0[t]*count_as_next.get(t,1)/len(perm)

# Method 3: Combined (permutation × sparse mask)
R_combined=(A.T@np.diag(P_token)@E_next).astype(np.float32)
# Mask: zero out tokens not in observed support
mask=np.array([1.0 if t in observed_next else 0.0 for t in range(VOCAB)])
R_combined_masked=(R_combined*mask[:,np.newaxis]).astype(np.float32)

for name,R in [("W_V=perm", R_perm), ("sparse_paths", R_sparse),
               ("perm+masked", R_combined_masked)]:
    ratio=float(np.linalg.norm(R))/E_norm
    print(f"  {name}: r_corpus/E = {ratio:.6f}")

# Choose best: combined gives sparsest directional signal
R_use = R_combined_masked
ratio_use = float(np.linalg.norm(R_use))/E_norm
print(f"\nUsing: perm+masked (ratio={ratio_use:.6f})")
print(f"Target from Phase 1 data: 0.000086")

# ── Serre cascade → E* ────────────────────────────────────────────────────────
print("\n[SERRE CASCADE] E* from sparse support...")
E_star=(R_use*(E_norm/max(float(np.linalg.norm(R_use)),1e-8))).astype(np.float32)

# Also test Phase 0 pre-baking + symbolic pruning
E_init_prebaked = (0.9*E_0 + 0.1*E_next).astype(np.float32)
E_init_prebaked = (E_init_prebaked*(E_norm/max(np.linalg.norm(E_init_prebaked),1e-8))).astype(np.float32)

# ── Build and test model ───────────────────────────────────────────────────────
print("\n[TEST] Scanning E* combinations → 1 LM step...")
print(f"  {'Method':<35} {'init val':>9}  {'after LM':>9}  {'accepted':>8}")
print("  " + "-"*65)

best_v = 999; best_label = ""
for label, E_test in [
    ("E_star (Serre from sparse)", E_star),
    ("E_prebaked (Phase0 alpha=0.1)", E_init_prebaked),
    ("E_0 (spectral baseline)", E_0),
    ("0.5*E_star + 0.5*E_0", (0.5*E_star+0.5*E_0)),
    ("0.3*E_star + 0.7*E_0", (0.3*E_star+0.7*E_0)),
    ("0.1*E_star + 0.9*E_0", (0.1*E_star+0.9*E_0)),
]:
    E_t = E_test.astype(np.float32)
    E_t = E_t*(E_norm/max(float(np.linalg.norm(E_t)),1e-8))
    torch.manual_seed(99); m=LM(D,N_HEADS,N_STU)
    m.te.weight.data.copy_(torch.tensor(E_t))
    v0=eval_val(m,n=10)
    v_lm,acc=lm_step(m)
    marker='←BEST' if v_lm<best_v else ''
    print(f"  {label:<35} {v0:>9.4f}  {v_lm:>9.4f}  {'✓' if acc else '~'} {marker}")
    if v_lm<best_v: best_v=v_lm; best_label=label

print(f"\n  BEST: {best_label}")
print(f"  Best val after 1 LM: {best_v:.4f}")
print(f"  Pass 12 baseline (25CE+1LM): val≈2.54")
print(f"  Reference (167CE): run separately")
print(f"\n  If best_v < 2.54: symbolic pruning eliminates 25 CE steps ✓")
print(f"  If best_v > 2.54: 25 CE steps are irreducible ✗")
