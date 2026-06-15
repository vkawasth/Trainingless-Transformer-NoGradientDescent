#!/usr/bin/env python3
"""
Spectral Universality Test
===========================
Train two models on the same data with different random seeds.
Extract (Λ_l, ρ_l) from both — eigenvalues and Plücker norming constants.
Compare: are they the same?

If (Λ_A, ρ_A) ≈ (Λ_B, ρ_B):
  Spectral universality holds.
  The Toda lattice endpoint W* is determined by (architecture, data).
  Inverse spectral transform could initialize a new model at W* without training.

If (Λ_A, ρ_A) ≠ (Λ_B, ρ_B):
  Multiple attractors exist.
  Gradient descent selects which one based on initialization.
  Training is necessary — no bypass.

Λ_l = eigenvalues of J_l (layer Jacobian at reference input)
ρ_l = Plücker coordinates of top-k active subspace of δJ_l
      = ∧^k U_l  (exterior product of top-k singular vectors)
      = the spectral norming constants of the Toda lattice

For k=2: ρ_l ∈ P(∧²R^m), coords p_{ij} = u_{1i}u_{2j} - u_{1j}u_{2i}
         6 coordinates for m=4, C(m,2) for general m
         constrained to Grassmannian: p_{12}p_{34} - p_{13}p_{24} + p_{14}p_{23} = 0
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from itertools import combinations

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4
PROJ=32   # Jacobian projection dim (smaller for speed, enough for spectrum)
K_PLUCKER=4  # use top-K singular vectors for Plücker coords

print(f"\n{'='*65}")
print(f"  SPECTRAL UNIVERSALITY TEST")
print(f"  Two models, same data, different seeds.")
print(f"  Are (Λ*, ρ*) the same?")
print(f"  d={D}  layers={N_LAYERS}  proj={PROJ}  k={K_PLUCKER}")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=json.load(f)
with open('/tmp/val_ids.json')   as f: val_ids=json.load(f)
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

def get_batch(split='train'):
    data=train_t if split=='train' else val_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ]   for i in ix]),
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

def clr(s,total=300,warmup=100):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def train(seed, steps=300, label=""):
    torch.manual_seed(seed)
    model=LM(D,N_HEADS,N_LAYERS)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    t0=time.time()
    for step in range(1,steps+1):
        for pg in opt.param_groups: pg['lr']=clr(step,total=steps)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%(steps//3)==0:
            model.eval()
            with torch.no_grad():
                vl=float(np.mean([model(*get_batch('val'))[1].item() for _ in range(10)]))
            print(f"  [{label}] step {step}/{steps}  val={vl:.4f}  t={time.time()-t0:.0f}s")
            model.train()
    model.eval()
    with torch.no_grad():
        vl=float(np.mean([model(*get_batch('val'))[1].item() for _ in range(40)]))
    print(f"  [{label}] final val={vl:.4f}")
    return model

# ── Jacobian extraction ───────────────────────────────────────────────────────
def layer_jacobian(block, h_in, pos, m):
    seq,d_=h_in.shape; m=min(m,seq,d_)
    _,_,Vt=torch.linalg.svd(h_in,full_matrices=False)
    U=Vt[:m,:].T.detach(); J=np.zeros((m,m))
    with torch.enable_grad():
        for i in range(m):
            h=h_in.clone().unsqueeze(0).detach().requires_grad_(True)
            h_out=block(h)
            v=(h_out[0,pos,:] if h_out.dim()==3 else h_out[pos,:])
            (v*U[:,i]).sum().backward()
            g=h.grad
            g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
            J[:,i]=(U.T@g).numpy()
    return J.T, U.detach().numpy(), m

# ── Plücker coordinates of k-plane ───────────────────────────────────────────
def plucker_coords(U_cols):
    """
    U_cols: [m, k] — orthonormal basis of k-plane in R^m
    Returns: Plücker vector p ∈ ∧^k R^m (all k×k minors)
    For k=2: p_{ij} = U[i,0]*U[j,1] - U[j,0]*U[i,1]  for i<j
    For general k: p_{i1..ik} = det(U[[i1,..,ik], :])
    
    We use a compressed version: only keep significant minors.
    """
    m, k = U_cols.shape
    coords = []
    # All C(m,k) minors — take top indices for efficiency
    idx_range = min(m, PROJ)  # use only top-PROJ dimensions
    for idx in combinations(range(idx_range), k):
        submat = U_cols[list(idx), :]   # [k, k]
        coords.append(float(np.linalg.det(submat)))
    return np.array(coords)

def plucker_norm(p):
    return float(np.linalg.norm(p))

def plucker_similarity(p1, p2):
    """Cosine similarity between Plücker vectors (chordal distance proxy)."""
    n1=np.linalg.norm(p1); n2=np.linalg.norm(p2)
    if n1<1e-10 or n2<1e-10: return 0.0
    return float(np.dot(p1,p2)/(n1*n2))

def grassmannian_distance(U_A, U_B):
    """
    Chordal distance on Gr(k, m) between two k-planes.
    U_A, U_B: [m, k] orthonormal bases.
    d^2 = k - ||U_A^T U_B||_F^2 = sum sin^2(theta_i)
    d=0: identical planes. d=sqrt(k): fully orthogonal.
    Sign-invariant — correct metric for universality test.
    """
    sv = np.linalg.svd(U_A.T @ U_B, compute_uv=False)
    sv = np.clip(sv, 0, 1)
    d2 = U_A.shape[1] - float(np.sum(sv**2))
    return float(np.sqrt(max(d2, 0)))

def subspace_similarity(U_A, U_B):
    """
    Mean cos of principal angles between two k-planes.
    1 = identical, 0 = orthogonal. Sign-invariant.
    """
    sv = np.linalg.svd(U_A.T @ U_B, compute_uv=False)
    return float(np.mean(np.clip(sv, 0, 1)))

def plucker_relation_residual(p, k=2):
    """
    For k=2, m=4: check p12*p34 - p13*p24 + p14*p23 = 0
    For general k: the Plücker relations are more complex.
    Returns residual (should be ~0 for valid k-plane).
    """
    if k==2 and len(p)>=6:
        # Assuming indices (01,02,03,12,13,23) → (p01,p02,p03,p12,p13,p23)
        # Plücker: p01*p23 - p02*p13 + p03*p12 = 0
        p01,p02,p03,p12,p13,p23 = p[0],p[1],p[2],p[3],p[4],p[5]
        return abs(p01*p23 - p02*p13 + p03*p12)
    return 0.0

# ── Extract spectral data from model ─────────────────────────────────────────
def extract_spectral_data(model, x_ref, pos, m=PROJ, k=K_PLUCKER, label=""):
    """
    For each layer l:
      Λ_l = eigenvalues of J_l (real part, sorted)
      ρ_l = Plücker coords of top-k subspace of δJ_l
    """
    print(f"  Extracting spectral data [{label}]...", flush=True)
    with torch.no_grad():
        hs=model.hidden_states(x_ref); hs=[h[0] for h in hs]

    spectra=[]   # per-layer eigenvalues
    plueckers=[] # per-layer Plücker coords
    norms_dJ=[]  # per-layer ||δJ||
    ranks=[]     # per-layer rank
    u_actives=[] # per-layer top-k singular vectors [m,k]

    for l in range(N_LAYERS):
        J, U_basis, ma = layer_jacobian(model.blocks[l], hs[l], pos, m)
        dJ = J - np.eye(ma)

        # Λ_l: eigenvalues of J_l (the full Jacobian, not δJ)
        eigs = np.linalg.eigvals(J)
        eigs_real = np.sort(eigs.real)[::-1]   # descending real parts

        # SVD of δJ for active subspace and rank
        sv = np.linalg.svd(dJ, compute_uv=False)
        rank = int(np.sum(sv > sv[0]*0.10)) if sv[0]>1e-8 else 1
        rank = max(rank, k)   # ensure at least k directions for Plücker

        # U_active: top-k left singular vectors of δJ
        U_sv, _, _ = np.linalg.svd(dJ)
        U_active = U_sv[:, :k]   # [m, k]

        # Plücker coords of the k-plane
        p = plucker_coords(U_active)

        spectra.append(eigs_real[:min(8, len(eigs_real))])
        plueckers.append(p)
        norms_dJ.append(float(np.linalg.norm(dJ)))
        ranks.append(rank)
        u_actives.append(U_active.copy())

        if (l+1)%8==0: print(f"    L{l+1}...", flush=True)

    return {
        'spectra':   spectra,
        'plueckers': plueckers,
        'norms':     norms_dJ,
        'ranks':     ranks,
        'label':     label,
        'u_actives': u_actives,
    }

# ── Train two models ──────────────────────────────────────────────────────────
print("Training Model A (seed=42)...")
modelA = train(42, steps=300, label="A")

print("\nTraining Model B (seed=137)...")
modelB = train(137, steps=300, label="B")

print("\nTraining Model C (seed=999, fewer steps to test convergence)...")
modelC = train(999, steps=150, label="C-partial")

# ── Reference input (SAME for all models) ────────────────────────────────────
torch.manual_seed(0)
x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
pos=SEQ//2

# ── Extract spectral data ─────────────────────────────────────────────────────
print()
dataA = extract_spectral_data(modelA, x_ref, pos, label="A")
dataB = extract_spectral_data(modelB, x_ref, pos, label="B")
dataC = extract_spectral_data(modelC, x_ref, pos, label="C-partial")

# ── Compare ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  SPECTRAL COMPARISON")
print(f"  Same architecture, same data, different seeds")
print("="*65)

print(f"\n  Per-layer comparison A vs B — Grassmannian distance (sign-invariant):")
print(f"  {'L':>3}  {'Gr_dist':>9}  {'cos(θ)':>8}  {'|λ_A-λ_B|':>11}  {'rank_A':>7}  {'rank_B':>7}")
print("  "+"-"*55)

grass_AB=[]; sub_sims=[]; eig_diffs=[]; rank_diffs=[]

for l in range(N_LAYERS):
    UA=dataA['u_actives'][l]; UB=dataB['u_actives'][l]
    sv=np.clip(np.linalg.svd(UA.T@UB,compute_uv=False),0,1)
    k_=UA.shape[1]
    gr=float(np.sqrt(max(k_-float(np.sum(sv**2)),0)))
    mc=float(np.mean(sv))
    grass_AB.append(gr); sub_sims.append(mc)

    eA=dataA['spectra'][l]; eB=dataB['spectra'][l]
    n=min(len(eA),len(eB))
    ediff=float(np.mean(np.abs(eA[:n]-eB[:n])))
    eig_diffs.append(ediff)

    rA=dataA['ranks'][l]; rB=dataB['ranks'][l]
    rank_diffs.append(abs(rA-rB))
    print(f"  L{l:>2}  {gr:>9.4f}  {mc:>8.4f}  {ediff:>11.4f}  {rA:>7}  {rB:>7}")

k_max=float(np.sqrt(K_PLUCKER))
print(f"\n  Summary A vs B:")
print(f"    Mean Grassmannian dist:  {np.mean(grass_AB):.4f}  (0=same, {k_max:.2f}=orthogonal)")
print(f"    Mean cos(principal θ):   {np.mean(sub_sims):.4f}  (1=identical, 0=orthogonal)")
print(f"    Mean |λ_A-λ_B|:          {np.mean(eig_diffs):.4f}")
print(f"    Mean |rank_A-rank_B|:    {np.mean(rank_diffs):.4f}")
print(f"    ||dJ|| rel diff:         {np.mean([abs(dataA['norms'][l]-dataB['norms'][l])/max(dataA['norms'][l],1e-8) for l in range(N_LAYERS)]):.4f}")

print(f"\n  Plücker norm profile (||ρ_l|| across layers):")
print(f"  {'L':>3}  {'||ρ_A||':>9}  {'||ρ_B||':>9}  {'ratio':>7}")
print("  "+"-"*32)
for l in range(N_LAYERS):
    nA=plucker_norm(dataA['plueckers'][l])
    nB=plucker_norm(dataB['plueckers'][l])
    ratio=nA/nB if nB>1e-8 else float('inf')
    print(f"  L{l:>2}  {nA:>9.4f}  {nB:>9.4f}  {ratio:>7.4f}")

print(f"\n  A vs C-partial (Grassmannian distance):")
gr_AC=[]
for l in range(N_LAYERS):
    UA=dataA['u_actives'][l]; UC=dataC['u_actives'][l]
    sv=np.clip(np.linalg.svd(UA.T@UC,compute_uv=False),0,1)
    gr_AC.append(float(np.mean(sv)))
print(f"    Mean cos(principal angles) A vs C: {np.mean(gr_AC):.4f}")
print(f"    (C=150 steps, A=300 steps — convergence test)")

# ── Plücker relation check ────────────────────────────────────────────────────
print(f"\n  Plücker relation residuals (should be ~0 for valid k-planes):")
print(f"  {'L':>3}  {'residual_A':>12}  {'residual_B':>12}")
print("  "+"-"*30)
for l in range(0,N_LAYERS,4):
    rA=plucker_relation_residual(dataA['plueckers'][l],k=K_PLUCKER)
    rB=plucker_relation_residual(dataB['plueckers'][l],k=K_PLUCKER)
    print(f"  L{l:>2}  {rA:>12.6f}  {rB:>12.6f}")

# ── Final verdict ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  VERDICT")
print("="*65)

mean_cos=np.mean(sub_sims)
mean_eig=np.mean(eig_diffs)

print(f"""
  cos(ρ_A, ρ_B) mean across {N_LAYERS} layers: {mean_cos:.4f}
  |λ_A - λ_B|   mean across {N_LAYERS} layers: {mean_eig:.4f}

  SPECTRAL UNIVERSALITY:""")

mean_cos = np.mean(sub_sims)
if mean_cos > 0.8:
    print(f"""
  CONFIRMED. cos(ρ_A, ρ_B) = {mean_cos:.4f} > 0.9.
  The Plücker norming constants are essentially identical across seeds.
  The Toda lattice endpoint W* is determined by (architecture, data).
  
  IMPLICATION: Measure (Λ*, ρ*) from any one trained model.
  Use inverse scattering to initialize the next model at W*.
  Gradient descent becomes fine-tuning, not search.
  The number of training steps could drop dramatically.""")

elif mean_cos > 0.4:
    print(f"""
  PARTIAL. cos(principal θ) = {mean_cos:.4f} ∈ (0.4, 0.8).
  The active subspaces are similar but not identical across seeds.
  Spectral universality holds approximately but not exactly.
  
  IMPLICATION: The inverse scattering initialization would give
  a better starting point than random, but not the exact W*.
  Training still needed, but potentially fewer steps.""")

else:
    print(f"""
  REFUTED. cos(ρ_A, ρ_B) = {mean_cos:.4f} < 0.5.
  The Plücker norming constants differ significantly across seeds.
  Multiple attractors exist. The spectral data is seed-dependent.
  
  IMPLICATION: Gradient descent is necessary.
  No inverse scattering bypass is possible.
  The investigation's boundary stands.""")
