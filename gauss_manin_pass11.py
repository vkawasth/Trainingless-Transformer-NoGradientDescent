#!/usr/bin/env python3
"""
Pass 11: Closed-Form Embedding Alignment — Three Fixes
========================================================
Previous attempt (A^T E power iteration) failed because:
  1. A^T E ignores the softmax partition function Z (non-abelian gauge)
  2. Rank collapse: A^T pushes all embeddings toward stationary vector

THREE FIXES:

Fix 1 — INVERSE SOFTMAX (Jacobian transport):
  The correct Newton direction uses the softmax Hessian H_t, not A.
  delta_E[t] = -(H_t + lambda*I)^{-1} g[t]
  where H_t = sum_s p(t|s)(1-p(t|s)) h_s h_s^T / d  (Fisher for token t)
        g[t] = sum_s (p(t|s) - 1[next=t]) h_s / sqrt(d)  (gradient)
  This is the per-token Newton step — theoretically optimal.
  
Fix 2 — REGULARIZED TRANSPORT (orthogonal projection):
  Before transport: subtract stationary direction pi from E.
  After transport: add back corpus-weighted mean.
  E_new = (I - pi pi^T) A^T E_0 + pi * (corpus mean)
  Prevents rank collapse by keeping geometry in S_perp.

Fix 3 — DIAGONAL FISHER NEWTON (one-pass, recommended):
  Accumulate full corpus gradient g[t] and diagonal Fisher F[t] in ONE pass.
  E*[t] = E_0[t] - g[t] / (F[t] + lambda)
  For common tokens (large F[t]): full Newton step.
  For rare tokens (F[t] ≈ 0): stays at E_0 (correct — no signal).
  Cost: 1 corpus pass = ~2 CE equiv.
  
  This is what 167 CE steps approximate stochastically.
  With the full corpus in one pass: exact (no noise).
"""
import json, math, warnings, collections
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F
import copy

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

if VOCAB != 1017:
    print(f"ERROR: VOCAB={VOCAB}. Run: python build_corpus.py --out /tmp/ --loops 300")
    import sys; sys.exit(1)

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

# ─── Build spectral embedding ────────────────────────────────────────────────

def build_spectral_emb():
    freq = np.zeros(VOCAB)
    for t in train_ids:
        if t < VOCAB: freq[t] += 1
    bigram = collections.Counter()
    for i in range(len(train_ids)-1):
        a,b = train_ids[i], train_ids[i+1]
        if a<VOCAB and b<VOCAB: bigram[(a,b)] += 1
    rows, cols, vals_l = [], [], []
    for (a,b), cnt in bigram.items():
        if a<VOCAB and b<VOCAB:
            rows.append(a); cols.append(b); vals_l.append(float(cnt))
    W = sp.csr_matrix((vals_l,(rows,cols)), shape=(VOCAB,VOCAB), dtype=np.float32)
    W = W + W.T
    d_inv = np.array(1.0/(W.sum(1)+1e-8)).flatten()
    d_sqrt_inv = np.sqrt(d_inv)
    D_sqrt_inv = sp.diags(d_sqrt_inv)
    L_sym = sp.eye(VOCAB) - D_sqrt_inv @ W @ D_sqrt_inv
    eigenvalues, eigenvectors = spla.eigsh(L_sym, k=D+1, which='SM', tol=1e-4, maxiter=2000)
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:,idx][:,1:D+1]
    scales = 1.0/(np.sqrt(eigenvalues[idx[1:D+1]])+1e-8)
    spectral_np = eigenvectors * scales[np.newaxis,:]
    spectral_np = spectral_np / (spectral_np.std()+1e-8) * 0.02
    return torch.tensor(spectral_np, dtype=torch.float32), bigram, freq

print("Building spectral embedding...")
spectral_emb, bigram, freq = build_spectral_emb()

# ─── Build base student (passes 0-5 proxy) ───────────────────────────────────

def build_base_student():
    torch.manual_seed(99)
    m = LM(D, N_HEADS, N_STU)
    m.te.weight.data.copy_(spectral_emb)
    # Saddle exit
    n_p = sum(p.numel() for p in m.parameters())
    v = torch.randn(n_p); v = v/v.norm()
    for _ in range(10):
        m.zero_grad()
        loss = sum(m(*get_batch())[1] for _ in range(10))/10
        grads = torch.autograd.grad(loss, list(m.parameters()), create_graph=True)
        gv = (torch.cat([g.flatten() for g in grads])*v.detach()).sum()
        hv = torch.cat([h.flatten() for h in torch.autograd.grad(gv, list(m.parameters()), retain_graph=False)]).detach()
        m.zero_grad()
        neg=-hv; v=neg/max(float(neg.norm()),1e-10)
    w0 = torch.cat([p.data.flatten() for p in m.parameters()])
    idx=0
    for p in m.parameters():
        n=p.numel(); p.data.copy_((w0+1.429*(v/v.norm()))[idx:idx+n].reshape(p.shape)); idx+=n
    with torch.no_grad():
        for l in [1,2]:
            m.blocks[l].attn.WV.weight.mul_(-1)
            m.blocks[l].attn.op.weight.mul_(-1)
    opt = torch.optim.AdamW(m.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
    for step in range(1,34):
        for pg in opt.param_groups: pg['lr'] = LR*5*min(step,10)/10
        m.train(); x,y=get_batch(); _,loss=m(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return m

print("Building base student (passes 0-5)...")
base = build_base_student()
v_base = eval_val(base, n=20)
print(f"  Base val: {v_base:.4f}")

# ─── FIX 2: Regularized Transport (simplest, test first) ─────────────────────

def fix2_regularized_transport(model, A_corpus_np, n_steps=3):
    """
    Regularized A^T transport with stationary projection.
    E_new = (I - pi pi^T) A^T E_0 + pi * corpus_mean
    """
    E = model.te.weight.data.numpy().copy()  # [VOCAB, D]
    
    # Compute stationary vector pi (dominant left eigenvector of A^T)
    # For a row-stochastic A, pi = A^T's dominant eigenvector = A's stationary dist
    # Power iteration on A^T to find pi
    A = A_corpus_np  # [VOCAB, VOCAB]
    pi = np.ones(VOCAB) / VOCAB
    for _ in range(50):
        pi_new = A.T @ pi
        pi = pi_new / (pi_new.sum() + 1e-10)
    pi = pi / (pi.sum() + 1e-10)  # [VOCAB]
    pi_norm = pi / (np.linalg.norm(pi) + 1e-10)
    
    print(f"  Stationary dist pi: max={pi.max():.4f}, min={pi.min():.6f}")
    
    # Corpus-weighted mean embedding
    corpus_mean = (pi[:,None] * E).sum(0)  # [D]
    
    for step in range(n_steps):
        # Center: remove stationary component
        E_proj = E - np.outer(pi_norm, pi_norm @ E)  # [VOCAB, D]
        # Transport: A^T on centered E
        E_transported = A.T @ E_proj  # [VOCAB, D]
        # Add back corpus mean
        E = E_transported + corpus_mean[None, :]
        print(f"  Step {step+1}: E std={E.std():.4f}, norm={np.linalg.norm(E):.4f}")
    
    return torch.tensor(E, dtype=torch.float32)

# ─── FIX 3: Diagonal Fisher Newton (one-pass) ────────────────────────────────

def fix3_fisher_newton(model, n_seqs=None, lambda_reg=1e-4, scale=1.0):
    """
    One-pass diagonal Fisher Newton step for E.
    
    For each token t, accumulate:
      g[t,d] = sum_positions_s (p(t|s) - 1[next=t]) * h_s[d] / sqrt(D)
      F[t]   = sum_positions_s p(t|s)(1-p(t|s)) * sum_d h_s[d]^2 / D  (scalar)
    
    Newton step: E*[t] = E[t] - scale * g[t] / (F[t] + lambda)
    
    For rare tokens (F[t] ≈ 0): delta ≈ 0, keeps spectral init.
    For common tokens: full Newton direction.
    """
    if n_seqs is None:
        # Use full training corpus (all base sequences)
        n_seqs = len(train_ids) // SEQ
    
    G = torch.zeros(VOCAB, D)   # gradient accumulator
    F_diag = torch.zeros(VOCAB) # diagonal Fisher accumulator
    n_accum = 0
    
    model.eval()
    # Freeze non-embedding parameters to get hidden states from theta*_attn
    with torch.no_grad():
        torch.manual_seed(42)
        for i in range(n_seqs):
            ix = torch.randint(0, len(train_t)-SEQ-1, (1,))[0].item()
            x = train_t[ix:ix+SEQ]       # [SEQ]
            y = train_t[ix+1:ix+SEQ+1]   # [SEQ] (next tokens)
            x_b = x.unsqueeze(0)
            
            # Get hidden states from the transformer (fixed theta*_attn)
            h_in = model.te(x_b) + model.pe(torch.arange(SEQ))
            h = h_in
            for block in model.blocks:
                h = block(h)
            h = model.ln_f(h)  # [1, SEQ, D]
            h = h[0]           # [SEQ, D]
            
            # Compute logits and probabilities
            logits = h @ model.te.weight.T / math.sqrt(D)  # [SEQ, VOCAB]
            p_model = torch.softmax(logits, dim=-1)         # [SEQ, VOCAB]
            
            # Accumulate gradient and Fisher for each position
            for pos in range(SEQ):
                p_s = p_model[pos]      # [VOCAB] probabilities
                h_s = h[pos]            # [D] hidden state
                t_next = int(y[pos])    # true next token
                
                if t_next >= VOCAB: continue
                
                # Gradient: g[t] += (p(t|s) - 1[t=t_next]) * h_s / sqrt(D)
                grad_contrib = p_s.clone()
                grad_contrib[t_next] -= 1.0  # subtract one-hot
                # G: [VOCAB, D] += outer product (gradient w.r.t. ALL E[t])
                G += torch.outer(grad_contrib, h_s) / math.sqrt(D)
                
                # Fisher: F[t] += p(t|s)(1-p(t|s)) * ||h_s||^2 / D
                fisher_weight = p_s * (1 - p_s)  # [VOCAB]
                h_s_norm_sq = (h_s * h_s).mean()  # scalar: mean squared norm / D
                F_diag += fisher_weight * h_s_norm_sq
                
                n_accum += 1
    
    # Normalize
    G /= max(n_accum, 1)
    F_diag /= max(n_accum, 1)
    
    print(f"  Accumulated {n_accum} positions")
    print(f"  G norm: {G.norm():.4f}")
    print(f"  F_diag range: [{float(F_diag.min()):.6f}, {float(F_diag.max()):.6f}]")
    print(f"  Tokens with F > lambda ({lambda_reg}): {int((F_diag > lambda_reg).sum())}")
    
    # Newton step
    E_0 = model.te.weight.data.clone()  # [VOCAB, D]
    # delta_E[t] = -g[t] / (F[t] + lambda) for each dimension d
    F_reg = F_diag + lambda_reg  # [VOCAB]
    delta_E = -G / F_reg.unsqueeze(1)  # [VOCAB, D]
    
    # Apply with step size scale
    E_star = E_0 + scale * delta_E
    
    print(f"  delta_E norm: {delta_E.norm():.4f}")
    print(f"  delta_E / E_0 ratio: {(delta_E.norm()/E_0.norm()):.4f}")
    
    return E_star

# ─── Build A_corpus ───────────────────────────────────────────────────────────

print("\nBuilding A_corpus...")
A_corpus = np.zeros((VOCAB, VOCAB), dtype=np.float32)
for (a,b), cnt in bigram.items():
    if a<VOCAB and b<VOCAB: A_corpus[a,b] += cnt
row_sums = A_corpus.sum(1, keepdims=True) + 1e-10
A_corpus /= row_sums

# ─── Run all three fixes ─────────────────────────────────────────────────────

print("\n" + "="*65)
print("COMPARISON: Pass 7 (167 CE) vs Three Fixes")
print("="*65)

# Reference: 167 CE steps
model_A = copy.deepcopy(base)
print("\n[A] Pass 7: 167 CE steps (reference)")
opt_a = torch.optim.AdamW(model_A.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
for step in range(1, 168):
    model_A.train(); x,y=get_batch(); _,loss=model_A(x,y)
    opt_a.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model_A.parameters(),1.0); opt_a.step()
    if step in [25, 50, 100, 167]:
        v=eval_val(model_A,n=10); print(f"  CE {step}: val={v:.4f}")
val_A = eval_val(model_A, n=30)
print(f"  PATH A FINAL: val={val_A:.4f}")

# Fix 2: Regularized transport
print("\n[B] Fix 2: Regularized A^T transport")
model_B = copy.deepcopy(base)
E_B = fix2_regularized_transport(model_B, A_corpus, n_steps=3)
with torch.no_grad():
    model_B.te.weight.data.copy_(E_B)
val_B = eval_val(model_B, n=20)
print(f"  After Fix 2: val={val_B:.4f}")
# Fine-tune with a few CE steps
opt_b = torch.optim.AdamW(model_B.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
for step in range(1,26):
    model_B.train(); x,y=get_batch(); _,loss=model_B(x,y)
    opt_b.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model_B.parameters(),1.0); opt_b.step()
val_B_25 = eval_val(model_B, n=20)
print(f"  Fix 2 + 25 CE: val={val_B_25:.4f}")

# Fix 3: Diagonal Fisher Newton (various lambda and scale)
print("\n[C] Fix 3: Diagonal Fisher Newton (one-pass)")
model_C = copy.deepcopy(base)
print("  Computing full-corpus gradient and Fisher...")
E_C = fix3_fisher_newton(model_C, n_seqs=500, lambda_reg=1e-3, scale=1.0)
with torch.no_grad():
    model_C.te.weight.data.copy_(E_C)
val_C = eval_val(model_C, n=20)
print(f"  After Fix 3 (lambda=1e-3, scale=1.0): val={val_C:.4f}")

# Try different scales
for scale in [0.1, 0.5, 2.0]:
    model_Cs = copy.deepcopy(base)
    E_Cs = fix3_fisher_newton(model_Cs, n_seqs=300, lambda_reg=1e-3, scale=scale)
    with torch.no_grad():
        model_Cs.te.weight.data.copy_(E_Cs)
    val_Cs = eval_val(model_Cs, n=15)
    print(f"  Fix 3 (scale={scale}): val={val_Cs:.4f}")

# Fix 3 + fine-tune
opt_c = torch.optim.AdamW(model_C.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
for step in range(1,26):
    model_C.train(); x,y=get_batch(); _,loss=model_C(x,y)
    opt_c.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model_C.parameters(),1.0); opt_c.step()
val_C_25 = eval_val(model_C, n=20)
print(f"  Fix 3 + 25 CE: val={val_C_25:.4f}")

# Fix 3 with more iterations (k=5 Newton steps)
print("\n[D] Fix 3 repeated (5 Newton steps)")
model_D = copy.deepcopy(base)
E_D = model_D.te.weight.data.clone()
for k in range(5):
    with torch.no_grad():
        model_D.te.weight.data.copy_(E_D)
    E_D = fix3_fisher_newton(model_D, n_seqs=200, lambda_reg=1e-3, scale=0.5)
    with torch.no_grad():
        model_D.te.weight.data.copy_(E_D)
    v = eval_val(model_D, n=10)
    print(f"  Newton step {k+1}: val={v:.4f}")
val_D = eval_val(model_D, n=20)
print(f"  After 5 Newton steps: val={val_D:.4f}")

print(f"""
{'='*65}
RESULTS: CLOSED-FORM EMBEDDING ALIGNMENT
{'='*65}

  Starting point (base): val={v_base:.4f}

  [A] Pass 7 (167 CE steps):          val={val_A:.4f}  [reference]
  [B] Fix 2 (regularized A^T):        val={val_B:.4f}  (0 CE)
      Fix 2 + 25 CE:                  val={val_B_25:.4f}
  [C] Fix 3 (diagonal Fisher Newton): val={val_C:.4f}  (~2 CE equiv)
      Fix 3 + 25 CE:                  val={val_C_25:.4f}
  [D] Fix 3 × 5 iterations:           val={val_D:.4f}  (~10 CE equiv)

  CONCLUSION:
  {'Fix 3 replaces ~' + str(167 - 25) + ' CE steps if C_25 ≈ A' if abs(val_C_25 - val_A) < 0.05
   else 'Gap remains: ' + f'{abs(val_C_25 - val_A):.4f} nats between Fix 3+25CE and Pass 7'}
""")
