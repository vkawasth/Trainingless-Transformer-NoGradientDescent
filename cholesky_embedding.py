#!/usr/bin/env python3
"""
Cholesky Embedding — Gauge-Invariant Closed Form
==================================================
The embedding orientation is gauge-free: E and E@O give identical
predictions if the head is also rotated by O.

The gram matrix G = E @ E^T IS computable from corpus statistics
(PMI, M^14 holonomy). Any factorization G = L @ L^T gives a valid
embedding. The Cholesky factor L is the canonical closed-form choice.

PROTOCOL:
  G_corpus = chol_factor of PPMI / M^14 gram matrix
  E_chol   = G_corpus[:, :D] scaled to teacher norm
  
  This gives an embedding where E[i] @ E[j] = G_corpus[i,j]
  — the correct token similarity structure from corpus statistics.
  The orientation (which direction E[i] points) is set by
  the Cholesky factorization, not by gradient descent.
  
  Crucially: if head weight = E^T (tied), then predictions are
  gauge-invariant — the specific orientation cancels out.

THREE GRAM SOURCES:
  G1: PPMI matrix (1-step co-occurrence)
  G2: M^14 holonomy (14-step transition)  
  G3: Gram-modulated (PPMI @ Gram_D) — Jacobian-weighted

COMPARE:
  Each gram source → Cholesky embedding → Serre cascade → head training
  vs teacher embeddings (oracle)
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14

print(f"\n{'='*65}")
print(f"  CHOLESKY EMBEDDING — GAUGE-INVARIANT CLOSED FORM")
print(f"  E = chol(G_corpus)  →  Serre cascade  →  head training")
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

def gram_to_embedding(G, target_norm, d=D):
    """
    Convert gram matrix G [V,V] to embedding E [V,d] via SVD.
    E @ E^T ≈ G  (best rank-d approximation).
    SVD is more stable than Cholesky for non-PSD matrices.
    Scale to match target_norm.
    """
    # Symmetrize and make PSD
    G = (G + G.T) / 2
    # SVD for best rank-d approximation
    U, s, _ = np.linalg.svd(G, full_matrices=False)
    s = np.maximum(s, 0)  # clip negative eigenvalues
    # E = U[:, :d] * sqrt(s[:d])
    E = (U[:, :d] * np.sqrt(s[:d])).astype(np.float32)
    # Scale
    current_norm = float(np.linalg.norm(E, 'fro'))
    if current_norm > 1e-8:
        E = E * (target_norm / current_norm)
    return E

def row_cos_with_teacher(E, E_teacher):
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    Tn = E_teacher / (np.linalg.norm(E_teacher, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(En * Tn, axis=1)))

def gram_alignment(E, E_teacher):
    """How well does E@E^T match E_teacher@E_teacher^T?"""
    G1 = E @ E.T
    G2 = E_teacher @ E_teacher.T
    # Correlation of gram matrices
    g1f = G1.flatten(); g2f = G2.flatten()
    return float(np.corrcoef(g1f, g2f)[0, 1])

# ════════════════════════════════════════════════════════
# STAGE 0: Train teacher
# ════════════════════════════════════════════════════════
print("Stage 0: Train teacher (300 steps)...")
torch.manual_seed(42)
teacher = LM(D, N_HEADS, N_LAYERS)
opt = torch.optim.AdamW(teacher.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
t0 = time.time()
for step in range(1, 301):
    for pg in opt.param_groups: pg['lr'] = clr(step, 300, 100)
    teacher.train(); x, y = get_batch(); _, loss = teacher(x, y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0); opt.step()
    if step % 100 == 0:
        teacher.eval()
        with torch.no_grad():
            vl = float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
        teacher.train()
teacher.eval()
val_teacher = eval_val(teacher)
E_teacher = teacher.te.weight.data.numpy().copy()
teacher_norm = float(np.linalg.norm(E_teacher, 'fro'))
print(f"  Teacher val={val_teacher:.4f}  ||E_teacher||={teacher_norm:.2f}\n")

# ════════════════════════════════════════════════════════
# STAGE 1: Extract invariants + Serre cascade
# ════════════════════════════════════════════════════════
print("Stage 1: Extract invariants...")
torch.manual_seed(0); pos = SEQ//2; m = min(PROJ, SEQ, D); ma = None
J_acc = [[] for _ in range(N_LAYERS)]; U_acc = [[] for _ in range(N_LAYERS)]
for ref in range(5):
    x_ref, _ = get_batch('val'); x_ref = x_ref[0:1]
    with torch.no_grad(): hs = teacher.hidden_states(x_ref); hs = [h[0] for h in hs]
    for l in range(N_LAYERS):
        J, U, m_ = layer_jac(teacher.blocks[l], hs[l], pos, m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma = m_
    if (ref+1) % 3 == 0: print(f"  ref {ref+1}/5...", flush=True)

Js = [np.mean(J_acc[l], axis=0) for l in range(N_LAYERS)]
Us = [np.mean(U_acc[l], axis=0) for l in range(N_LAYERS)]
J14 = Js[L_ATT]; U14 = Us[L_ATT]

# Accumulated Gram in D-space
Gram = sum(Js[l].T @ Js[l] for l in range(N_LAYERS)) / N_LAYERS
Gram_D = U14 @ Gram @ U14.T + (np.eye(D) - U14 @ U14.T)

# Serre cascade
cascade = []
for l in range(1, N_STU+1):
    C_l = ad_k(J14, Js[min(L_ATT+l, N_LAYERS-1)], l)
    n = float(np.linalg.norm(C_l))
    if n > 1e-8: C_l = C_l / n
    cascade.append(C_l)
print(f"  Cascade built: {N_STU} levels\n")

# Teacher gram matrix (oracle)
G_teacher = E_teacher @ E_teacher.T
print(f"  Teacher gram matrix: {G_teacher.shape}")
print(f"  Teacher gram rank: {np.linalg.matrix_rank(G_teacher, tol=0.01)}\n")

# ════════════════════════════════════════════════════════
# STAGE 2: Build gram matrices from corpus
# ════════════════════════════════════════════════════════
print("Stage 2: Build corpus gram matrices...")

# Bigram transition matrix
P = np.zeros((VOCAB, VOCAB), dtype=np.float64)
unigram = np.zeros(VOCAB, dtype=np.float64)
for k in range(len(train_ids)-1):
    a, b = train_ids[k], train_ids[k+1]
    if 0 <= a < VOCAB and 0 <= b < VOCAB:
        P[a, b] += 1; unigram[a] += 1
row_s = P.sum(axis=1, keepdims=True); row_s[row_s == 0] = 1
P_norm = P / row_s  # row-stochastic

# G1: PPMI gram matrix
print("  G1: PPMI gram matrix...")
total = P.sum()
unigram_p = unigram / max(total, 1)
pmi = np.log((P / max(total,1) + 1e-10) /
             (unigram_p[:, None] * unigram_p[None, :] + 1e-10))
ppmi = np.clip(pmi, 0, None).astype(np.float32)
G1 = ppmi @ ppmi.T  # [V,V] — PPMI gram
g1_align = gram_alignment(
    gram_to_embedding(G1, teacher_norm), E_teacher)
print(f"  G1 gram alignment with teacher: {g1_align:.4f}")

# G2: M^14 holonomy gram matrix
print("  G2: M^14 holonomy gram matrix...")
M14 = np.linalg.matrix_power(P_norm, 14).astype(np.float32)
G2 = M14 @ M14.T  # [V,V] — M^14 gram
g2_align = gram_alignment(
    gram_to_embedding(G2, teacher_norm), E_teacher)
print(f"  G2 gram alignment with teacher: {g2_align:.4f}")

# G3: Jacobian-modulated PPMI gram
print("  G3: Jacobian-modulated PPMI gram...")
# E_ppmi_d = SVD of ppmi in D-space
U_p, s_p, _ = np.linalg.svd(ppmi, full_matrices=False)
E_ppmi = (U_p[:, :D] * np.sqrt(np.maximum(s_p[:D], 0))).astype(np.float32)
# Modulate by Gram_D (Jacobian amplification)
sv_g, Uv_g = np.linalg.eigh(Gram_D)
sv_g = np.maximum(sv_g, 0)
Gram_sqrt = (Uv_g * np.sqrt(sv_g)[None, :]) @ Uv_g.T
E_gram_mod = (E_ppmi @ Gram_sqrt).astype(np.float32)
G3 = E_gram_mod @ E_gram_mod.T
g3_align = gram_alignment(
    gram_to_embedding(G3, teacher_norm), E_teacher)
print(f"  G3 gram alignment with teacher: {g3_align:.4f}\n")

# ════════════════════════════════════════════════════════
# STAGE 3: Build embeddings from gram matrices
# ════════════════════════════════════════════════════════
print("Stage 3: Build embeddings via SVD factorization of gram matrices...")

E_g1 = gram_to_embedding(G1, teacher_norm)
E_g2 = gram_to_embedding(G2, teacher_norm)
E_g3 = gram_to_embedding(G3, teacher_norm)

embeddings = {
    'G1_PPMI':     E_g1,
    'G2_M14':      E_g2,
    'G3_Jac_mod':  E_g3,
    'Teacher_oracle': E_teacher,
}

print(f"  {'Method':>20}  {'row_cos':>9}  {'gram_align':>11}")
print("  " + "-"*44)
for name, E in embeddings.items():
    rc = row_cos_with_teacher(E, E_teacher)
    ga = gram_alignment(E, E_teacher)
    print(f"  {name:>20}  {rc:>9.4f}  {ga:>11.4f}")

# ════════════════════════════════════════════════════════
# STAGE 4: Inject cascade + test each embedding
# ════════════════════════════════════════════════════════
print(f"\nStage 4: Build 6L student with each embedding + Serre cascade...")

def inject_cascade(model):
    with torch.no_grad():
        model.pe.weight.copy_(teacher.pe.weight)
        model.ln_f.weight.copy_(teacher.ln_f.weight)
        model.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            W_d = lift_to_d(cascade[l], U14, scale=0.01)
            W_t = torch.tensor(W_d, dtype=torch.float32)
            model.blocks[l].attn.WK.weight.copy_(W_t)
            model.blocks[l].attn.WQ.weight.copy_(W_t.T)
            model.blocks[l].attn.WV.weight.copy_(
                teacher.blocks[L_ATT].attn.WV.weight)
            model.blocks[l].attn.op.weight.copy_(
                teacher.blocks[L_ATT].attn.op.weight)
            model.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            model.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            model.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)

def run_experiment(E_init, label, head_steps=100, full_steps=200):
    torch.manual_seed(99)
    model = LM(D, N_HEADS, N_STU)
    with torch.no_grad():
        E_t = torch.tensor(E_init[:VOCAB, :D].copy(), dtype=torch.float32)
        model.te.weight.copy_(E_t)
    inject_cascade(model)
    v0 = eval_val(model, n=20)

    # Head-only training
    for p in model.parameters(): p.requires_grad_(False)
    model.head.weight.requires_grad_(True)
    opt_h = torch.optim.AdamW([model.head.weight], lr=LR, weight_decay=0.01)
    for step in range(1, head_steps+1):
        for pg in opt_h.param_groups: pg['lr'] = clr(step, head_steps, 20)
        model.train(); x, y = get_batch(); _, loss = model(x, y)
        opt_h.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([model.head.weight], 1.0); opt_h.step()
    for p in model.parameters(): p.requires_grad_(True)
    v_head = eval_val(model)

    # Full fine-tune
    opt_f = torch.optim.AdamW(model.parameters(), lr=LR,
                               betas=(0.9,0.95), weight_decay=0.1)
    for step in range(1, full_steps+1):
        for pg in opt_f.param_groups: pg['lr'] = clr(step, full_steps, 50)
        model.train(); x, y = get_batch(); _, loss = model(x, y)
        opt_f.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt_f.step()
        if step % 50 == 0:
            vl = eval_val(model, n=20)
            print(f"    [{label}] step {step}  val={vl:.4f}")
    v_full = eval_val(model)
    return v0, v_head, v_full

results = {}
for name, E in embeddings.items():
    print(f"\n  [{name}]")
    v0, vh, vf = run_experiment(E, name)
    results[name] = (v0, vh, vf)
    print(f"  zero-shot={v0:.4f}  head-only={vh:.4f}  full-tune={vf:.4f}")

# ════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  CHOLESKY EMBEDDING RESULTS")
print("="*65)
print(f"""
  GRAM MATRIX ALIGNMENT WITH TEACHER:
    G1 (PPMI):         {g1_align:.4f}
    G2 (M^14):         {g2_align:.4f}
    G3 (Jac-modulated): {g3_align:.4f}
    
  {'Method':>20}  {'zero-shot':>10}  {'head-only':>10}  {'full-tune':>10}
  {'-'*55}""")

for name, (v0, vh, vf) in results.items():
    print(f"  {name:>20}  {v0:>10.4f}  {vh:>10.4f}  {vf:>10.4f}")

print(f"""
  Teacher oracle:              val={val_teacher:.4f}
  Prior best (Serre+teacher emb+200CE): val=0.1865

  KEY READING:
  gram_align measures whether E@E^T ≈ E_teacher@E_teacher^T
  If gram_align is high BUT row_cos is low:
    The gram structure is correct but orientation differs.
    This is the gauge freedom — all factorizations are equivalent.
    The full-tune val should approach teacher regardless of orientation.
    
  If full-tune val of corpus gram ≈ full-tune val of teacher oracle:
    The gram matrix captures everything needed.
    Training = head alignment only (orientation-invariant).
    Closed-form embedding IS achievable via gram factorization.
    
  If full-tune val of corpus gram >> teacher oracle:
    The corpus gram matrix does not capture the teacher's structure.
    Something beyond co-occurrence statistics is needed.
""")
