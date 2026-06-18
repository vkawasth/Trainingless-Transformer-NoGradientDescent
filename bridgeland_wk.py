#!/usr/bin/env python3
"""
Bridgeland W_K Initialisation Experiment
==========================================
FINDING: Im(z_k) = arg(lambda_1(phi_k)) takes only {0, pi}
         across trained transformer layers.
         The five layers where it transitions are Bridgeland walls,
         not Lefschetz critical values.

HYPOTHESIS: The correct corpus-derived W_K must respect the
            Bridgeland phase structure {0, pi} of the trained W_K.

THREE EXPERIMENTS:
  A: Spectral (bigram Laplacian) W_K — does NOT encode central charge
  B: Conditional entropy W_K — encodes Z(t) = P(t) + i*H(t)
  C: Random W_K — baseline

MEASUREMENT: For each init, compute Im(z_k) profile and compare
             to teacher. Also compare val at 200CE.

THEORETICAL PREDICTION:
  B (conditional entropy) should match the teacher's {0,pi} profile
  better than A or C, and should reach a lower val after the same
  number of CE steps, because the Bridgeland phase structure is
  the binding constraint on W_K initialisation.
"""
import json, math, collections, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; N_LAYERS_T=24; BATCH=8; SEQ=64; LR=3e-4

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

if VOCAB != 1017:
    print(f"""
  ERROR: VOCAB={VOCAB}, expected 1017.
  The /tmp/ corpus files are stale or missing.
  Fix: python build_corpus.py --out /tmp/
""")
    import sys; sys.exit(1)
print(f"VOCAB={VOCAB}, corpus={len(train_ids)}")

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

def clr(s,total,warmup=20):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ─── Bridgeland phase measurement ─────────────────────────────────────────────

def bridgeland_profile(model):
    """
    Compute Im(z_k) = arg(lambda_1(phi_k)) for each layer.
    phi_k = WK_{k+1} @ WK_k^{-1}
    Returns list of (k, Im_z_k, wall_type) for k=0..n-2
    """
    n = len(model.blocks)
    profile = []
    with torch.no_grad():
        for k in range(n-1):
            WK_k  = model.blocks[k  ].attn.WK.weight.numpy().astype(np.float64)
            WK_k1 = model.blocks[k+1].attn.WK.weight.numpy().astype(np.float64)
            try:
                phi_k = WK_k1 @ np.linalg.pinv(WK_k)  # [D,D]
                # Leading eigenvalue (by magnitude)
                evals = np.linalg.eigvals(phi_k[:32,:32])  # use top 32 for speed
                lead = evals[np.argmax(np.abs(evals))]
                Im_z = float(np.angle(lead))  # in [-pi, pi]
                # Classify
                if abs(Im_z) < 0.3:
                    wall = 'positive_real (Im≈0)'
                elif abs(Im_z - math.pi) < 0.3 or abs(Im_z + math.pi) < 0.3:
                    wall = 'BRIDGELAND_WALL (Im≈π)'
                else:
                    wall = f'intermediate (Im={Im_z:.2f})'
            except Exception as e:
                Im_z = 0.0; wall = f'error: {e}'
            profile.append((k, Im_z, wall))
    return profile

# ─── Corpus statistics ────────────────────────────────────────────────────────

print("\n" + "="*65)
print("CORPUS STATISTICS: Bridgeland central charge structure")
print("="*65)

# Bigram conditional: P(next=t' | current=t)
print("  Building bigram conditional...")
bigram = collections.Counter()
freq   = np.zeros(VOCAB)
for t in train_ids:
    if t < VOCAB: freq[t] += 1
for i in range(len(train_ids)-1):
    a,b = train_ids[i], train_ids[i+1]
    if a < VOCAB and b < VOCAB: bigram[(a,b)] += 1
P = freq / freq.sum()

# Conditional entropy H(next | t) = -sum_t' P(t'|t) log P(t'|t)
print("  Computing conditional entropy H(next|t)...")
cond_H = np.zeros(VOCAB)
for t in range(VOCAB):
    if P[t] < 1e-8: continue
    successors = {b: cnt for (a,b),cnt in bigram.items() if a==t}
    total = sum(successors.values())
    if total == 0: continue
    for b,cnt in successors.items():
        p = cnt/total
        cond_H[t] -= p * math.log(p + 1e-15)
print(f"  H range: [{cond_H.min():.3f}, {cond_H.max():.3f}] nats")
print(f"  H mean: {cond_H.mean():.3f}")

# Central charge Z(t) = P(t) + i*H(t)
# Bridgeland wall: tokens with H(t) near log(V) (maximum uncertainty)
H_max = math.log(VOCAB)
wall_tokens = (cond_H > 0.8*H_max).sum()
print(f"  Wall tokens (H > 0.8*log(V)={0.8*H_max:.2f}): {wall_tokens}")
print(f"  Non-wall tokens: {VOCAB - wall_tokens}")

# Conditional entropy W_K: 
# W_K[i,j] encodes how token i's conditional entropy
# relates to token j's co-occurrence
# Build H-weighted co-occurrence matrix
print("  Building conditional entropy W_K basis...")
H_norm = cond_H / (cond_H.max() + 1e-8)  # [0,1]
top_D = min(D, VOCAB)

# Co-occurrence matrix weighted by conditional entropy product
H_coo = np.zeros((top_D, top_D), dtype=np.float32)
for (a,b), cnt in bigram.items():
    if a < top_D and b < top_D:
        weight = (H_norm[a] * H_norm[b]) ** 0.5  # geometric mean
        H_coo[a,b] += cnt * weight

H_sym = (H_coo + H_coo.T) / 2
H_sym /= (H_sym.max() + 1e-8)

try:
    UH, SH, VtH = np.linalg.svd(H_sym, full_matrices=False)
    wk_H_basis = torch.tensor(UH[:D,:D], dtype=torch.float32) * 0.1
    print(f"  Conditional entropy SVD top-5: {np.round(SH[:5],3).tolist()}")
    print(f"  W_K_H range: [{wk_H_basis.min():.4f}, {wk_H_basis.max():.4f}]")
except Exception as e:
    print(f"  SVD failed: {e}")
    wk_H_basis = torch.randn(D, D) * 0.02

# Also build spectral (bigram Laplacian) W_K for comparison
print("  Building spectral W_K basis (bigram Laplacian)...")
rows, cols, vals_l = [], [], []
for (a,b), cnt in bigram.items():
    if a < VOCAB and b < VOCAB:
        rows.append(a); cols.append(b); vals_l.append(float(cnt))
W = sp.csr_matrix((vals_l,(rows,cols)), shape=(VOCAB,VOCAB), dtype=np.float32)
W = W + W.T
d_inv = np.array(1.0/(W.sum(1)+1e-8)).flatten()
d_sqrt_inv = np.sqrt(d_inv)
D_sqrt_inv = sp.diags(d_sqrt_inv)
L_sym = sp.eye(VOCAB) - D_sqrt_inv @ W @ D_sqrt_inv
evals_L, evecs_L = spla.eigsh(L_sym, k=D+1, which='SM', tol=1e-4, maxiter=2000)
idx = np.argsort(evals_L)
evecs_L = evecs_L[:,idx][:,1:D+1]
scales_L = 1.0/(np.sqrt(evals_L[idx[1:D+1]])+1e-8)
spec_emb_np = evecs_L * scales_L[np.newaxis,:]
spec_emb_np = spec_emb_np / (spec_emb_np.std()+1e-8) * 0.02
spectral_emb = torch.tensor(spec_emb_np, dtype=torch.float32)

# W_K from spectral embedding columns
wk_spec_basis = torch.tensor(
    evecs_L[:min(D,VOCAB),:] * 0.1, dtype=torch.float32
)
if wk_spec_basis.shape[0] < D:
    pad = torch.randn(D-wk_spec_basis.shape[0], D)*0.01
    wk_spec_basis = torch.cat([wk_spec_basis, pad], dim=0)

print(f"  Spectral W_K range: [{wk_spec_basis.min():.4f}, {wk_spec_basis.max():.4f}]")

# ─── Build models ─────────────────────────────────────────────────────────────

def build_model(wk_init=None, emb_init=None):
    torch.manual_seed(99)
    m = LM(D, N_HEADS, N_STU)
    if emb_init is not None:
        m.te.weight.data.copy_(emb_init)
    if wk_init is not None:
        with torch.no_grad():
            n = min(wk_init.shape[0], D)
            for l in range(N_STU):
                m.blocks[l].attn.WK.weight[:n,:n] = wk_init[:n,:n]
                m.blocks[l].attn.WQ.weight[:n,:n] = wk_init[:n,:n].T
    return m

# ─── EXPERIMENT 1: Bridgeland profile comparison ──────────────────────────────

print("\n" + "="*65)
print("EXPERIMENT 1: Bridgeland Im(z_k) profile at initialisation")
print("="*65)

print("\n  Teacher profile (300-step trained, 24L):")
print("  [Training teacher...]")
torch.manual_seed(42)
teacher = LM(D, N_HEADS, N_LAYERS_T)
opt_t = torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,301):
    lr_now = LR*min(step,100)/100 if step<=100 else LR*0.5*(1+math.cos(math.pi*(step-100)/200))
    for pg in opt_t.param_groups: pg['lr']=lr_now
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt_t.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt_t.step()
    if step in [100,200,300]:
        teacher.eval()
        with torch.no_grad():
            vl=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  Teacher step {step}: val={vl:.4f}")
v_teacher = eval_val(teacher, n=40)
print(f"  Teacher final: val={v_teacher:.4f}")

# Teacher Bridgeland profile
teacher_profile = bridgeland_profile(teacher)
print("\n  Teacher Im(z_k) profile:")
print(f"  {'k':>3}  {'Im(z_k)':>9}  {'type'}")
print("  " + "-"*45)
wall_layers_teacher = []
for k, Im_z, wtype in teacher_profile:
    print(f"  {k:>3}  {Im_z:>9.4f}  {wtype}")
    if 'WALL' in wtype: wall_layers_teacher.append(k)
print(f"\n  Teacher wall layers: {wall_layers_teacher}")

# Student profiles
print("\n  [A] Spectral W_K init:")
m_A = build_model(wk_init=wk_spec_basis, emb_init=spectral_emb)
prof_A = bridgeland_profile(m_A)
walls_A = [k for k,Im_z,wtype in prof_A if 'WALL' in wtype]
print(f"  Wall layers: {walls_A}")

print("\n  [B] Conditional entropy W_K init:")
m_B = build_model(wk_init=wk_H_basis, emb_init=spectral_emb)
prof_B = bridgeland_profile(m_B)
walls_B = [k for k,Im_z,wtype in prof_B if 'WALL' in wtype]
print(f"  Wall layers: {walls_B}")

print("\n  [C] Random W_K init:")
m_C = build_model()
prof_C = bridgeland_profile(m_C)
walls_C = [k for k,Im_z,wtype in prof_C if 'WALL' in wtype]
print(f"  Wall layers: {walls_C}")

# Profile comparison table
print("\n  Im(z_k) comparison: Teacher vs A (spectral) vs B (cond.H) vs C (random)")
print(f"  {'k':>3}  {'Teacher':>8}  {'A(spec)':>8}  {'B(H)':>8}  {'C(rnd)':>8}  {'B=T?':>5}")
print("  " + "-"*55)
matches_B = 0; matches_A = 0
for i in range(min(len(teacher_profile), len(prof_A), len(prof_B), len(prof_C))):
    k, ImT, _ = teacher_profile[i]
    _, ImA, _ = prof_A[i]
    _, ImB, _ = prof_B[i]
    _, ImC, _ = prof_C[i]
    # Match = same half-plane (both near 0 or both near pi)
    def sign(x): return 'pi' if abs(abs(x)-math.pi)<0.5 else '0'
    mB = sign(ImB)==sign(ImT); mA = sign(ImA)==sign(ImT)
    if mB: matches_B+=1
    if mA: matches_A+=1
    print(f"  {k:>3}  {ImT:>8.4f}  {ImA:>8.4f}  {ImB:>8.4f}  {ImC:>8.4f}  {'✓' if mB else '✗':>5}")

n_layers_compared = min(len(teacher_profile), len(prof_A), len(prof_B))
print(f"\n  B (cond.H) matches teacher: {matches_B}/{n_layers_compared}")
print(f"  A (spectral) matches teacher: {matches_A}/{n_layers_compared}")

# ─── EXPERIMENT 2: Training comparison ────────────────────────────────────────

print("\n" + "="*65)
print("EXPERIMENT 2: val at 200CE for each W_K init")
print("="*65)

def run_200CE(model, label):
    opt = torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,201):
        for pg in opt.param_groups: pg['lr']=clr(step,200)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step in [50,100,150,200]:
            v=eval_val(model,n=10)
            print(f"  {label} CE {step}: {v:.4f}")
    return eval_val(model,n=30)

print("\n[A] Spectral W_K + spectral emb:")
vA = run_200CE(m_A, "A")
print(f"  A FINAL: {vA:.4f}")

print("\n[B] Conditional entropy W_K + spectral emb:")
vB = run_200CE(m_B, "B")
print(f"  B FINAL: {vB:.4f}")

print("\n[C] Random W_K (baseline):")
vC = run_200CE(m_C, "C")
print(f"  C FINAL: {vC:.4f}")

print(f"""
{'='*65}
  BRIDGELAND W_K RESULTS
{'='*65}

  Bridgeland phase match with teacher:
    A (spectral W_K):    {matches_A}/{n_layers_compared} layers match
    B (cond.H W_K):      {matches_B}/{n_layers_compared} layers match
    C (random):          [not measured — no corpus structure]
  
  Teacher wall layers: {wall_layers_teacher}
  A wall layers:       {walls_A}
  B wall layers:       {walls_B}
  
  200CE val comparison:
    A (spectral W_K):    val={vA:.4f}
    B (cond.H W_K):      val={vB:.4f}
    C (random):          val={vC:.4f}
    Teacher (24L,300):   val={v_teacher:.4f}
    
  BRIDGELAND PHASE HYPOTHESIS:
    If B better than A: conditional entropy W_K encodes correct phase
    If B ≈ A: Bridgeland phase does not determine W_K quality
    If B > C: corpus structure (any kind) hurts vs random
  
  Next step: if B beats A and C -> conditional entropy W_K
             is the correct teacher-free initialisation.
             Then run full compiler pipeline with B init.
""")
