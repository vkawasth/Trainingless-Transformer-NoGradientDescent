#!/usr/bin/env python3
"""
Spectral Embedding Initialisation
====================================
FINDING: PMI embeddings destroy basin structure (val=0.78 vs random val=0.29)
CAUSE: PMI normalisation collapses all embeddings to low-magnitude cluster
SOLUTION: Spectral embedding of corpus transition graph

The corpus token bigram graph has a random-walk Laplacian L_rw.
Its eigenvectors phi_1,...,phi_D give the diffusion coordinates
of token t in the semantic space. These are:
  - Isotropic (spread through R^D by construction)
  - Semantically structured (similar tokens cluster in eigenvector space)
  - Computable from corpus alone (no teacher)

COMPARISON:
  A: PMI embeddings (broken)          -> val=0.78 at 200CE
  B: Random embeddings (baseline)     -> val=0.29 at 200CE
  C: Spectral/Laplacian embeddings    -> val=?  at 200CE
  D: Frequency-scaled random          -> val=?  at 200CE

Also test: does the MF pumping pipeline work with spectral embeddings?
  E: Spectral + MF10 + settle + LM + 25CE -> val=?
"""
import json, math, collections, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; ALPHA_STAR=1.429
N_LAYERS_T=24

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

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
    def flat_params(self): return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat(self,f):
        idx=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(f[idx:idx+n].reshape(p.shape)); idx+=n

def eval_val(m,n=30):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clr(s,total,warmup=20):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ─── Build Spectral Embedding ─────────────────────────────────────────────────
print("="*65)
print("CORPUS SPECTRAL EMBEDDING")
print("  Laplacian eigenvectors of token bigram graph")
print("="*65)

# Build sparse bigram transition matrix
print("  Building sparse bigram matrix...")
rows, cols, vals = [], [], []
freq = np.zeros(VOCAB)
for t in train_ids:
    if t < VOCAB: freq[t] += 1

bigram = collections.Counter()
for i in range(len(train_ids)-1):
    a,b = train_ids[i], train_ids[i+1]
    if a<VOCAB and b<VOCAB: bigram[(a,b)] += 1

for (a,b),cnt in bigram.items():
    rows.append(a); cols.append(b); vals.append(float(cnt))

W = sp.csr_matrix((vals,(rows,cols)), shape=(VOCAB,VOCAB), dtype=np.float32)
W = W + W.T  # symmetrise

# Row-normalised to get random-walk matrix P
d_inv = np.array(1.0/(W.sum(1)+1e-8)).flatten()
D_inv = sp.diags(d_inv)
P = D_inv @ W  # row-stochastic random walk

# Normalised Laplacian L = I - D^{-1/2} W D^{-1/2}
d_sqrt_inv = np.sqrt(d_inv)
D_sqrt_inv = sp.diags(d_sqrt_inv)
L_sym = sp.eye(VOCAB) - D_sqrt_inv @ W @ D_sqrt_inv

# Compute top D eigenvectors (smallest eigenvalues = smoothest)
print(f"  Computing {D} eigenvectors of L_sym ({VOCAB}×{VOCAB})...")
t0 = time.time()
try:
    # Use ARPACK for sparse eigenvectors
    eigenvalues, eigenvectors = spla.eigsh(L_sym, k=D+1, which='SM', tol=1e-4, maxiter=2000)
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Skip the constant eigenvector (eigenvalue ≈ 0)
    eigenvectors = eigenvectors[:, 1:D+1]  # [VOCAB, D]
    print(f"  Eigenvectors computed in {time.time()-t0:.1f}s")
    print(f"  Eigenvalue range: [{eigenvalues[1]:.4f}, {eigenvalues[D]:.4f}]")

    # Scale: each eigenvector scaled by 1/sqrt(eigenvalue) for diffusion coords
    scales = 1.0 / (np.sqrt(eigenvalues[1:D+1]) + 1e-8)
    spectral_emb_np = eigenvectors * scales[np.newaxis, :]  # diffusion map
    # Normalise to std ≈ 0.02 (same as random init)
    spectral_emb_np = spectral_emb_np / (spectral_emb_np.std() + 1e-8) * 0.02
    spectral_emb = torch.tensor(spectral_emb_np, dtype=torch.float32)
    print(f"  Spectral emb: std={spectral_emb.std():.4f}, range=[{spectral_emb.min():.4f},{spectral_emb.max():.4f}]")
    spectral_ok = True
except Exception as e:
    print(f"  Eigenvector computation failed: {e}")
    print("  Falling back to frequency-scaled random")
    spectral_ok = False

# Frequency-scaled random (Option D): std proportional to sqrt(P(t))
freq_t = torch.tensor(freq/freq.sum(), dtype=torch.float32)
freq_scale = torch.sqrt(freq_t).unsqueeze(1)  # [VOCAB,1]
freq_scaled_emb = torch.randn(VOCAB, D) * 0.02 * freq_scale / (freq_scale.mean()+1e-8)

print(f"  Freq-scaled emb: std={freq_scaled_emb.std():.4f}")

# ─── Experiment ───────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("EXPERIMENT: Embedding init comparison (200 CE steps)")
print("="*65)

def run_student(emb_init, label, n_steps=200):
    torch.manual_seed(99)
    stu = LM(D, N_HEADS, N_STU)
    if emb_init is not None:
        stu.te.weight.data.copy_(emb_init)
    opt = torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,n_steps+1):
        for pg in opt.param_groups: pg['lr']=clr(step,n_steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt.step()
        if step in [33,50,100,150,200]:
            v=eval_val(stu,n=8)
            print(f"  {label} CE {step}: {v:.4f}")
    return eval_val(stu,n=20), stu

results={}

print("\n[A] PMI embeddings (broken baseline)")
_, _ = run_student(None, "A(rnd)", n_steps=33)  # just confirm random is fine

print("\n[B] Random (confirmed good)")
v,_ = run_student(None, "B")
results['B_random'] = v

if spectral_ok:
    print("\n[C] Spectral/Laplacian embeddings")
    v,stu_c = run_student(spectral_emb, "C")
    results['C_spectral'] = v

print("\n[D] Frequency-scaled random")
v,stu_d = run_student(freq_scaled_emb, "D")
results['D_freqscale'] = v

print(f"""
{'='*65}
  SPECTRAL EMBEDDING RESULTS
{'='*65}
    B (random):          val={results.get('B_random','?'):.4f}
    C (spectral):        val={results.get('C_spectral','?'):.4f}
    D (freq-scaled):     val={results.get('D_freqscale','?'):.4f}
    
  Reference: teacher val≈0.247, Serre cascade val=0.187

  IF C < B: spectral embeddings provide better semantic structure
            than random — basin selection is faster
  IF C ≈ B: geometry doesn't help, any isotropic init works
  IF C < 0.187: spectral embeddings beat Serre cascade (teacher-free!)
""")
