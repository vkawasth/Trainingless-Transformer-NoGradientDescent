#!/usr/bin/env python3
"""
Householder W_K Initialization
================================
Directly instantiates the topological fixed point of the Kac-Moody orbit.

PROTOCOL (from spec):
  1. MEASURE: Use measured σ_l profile as primary constraint
  2. REFLECT:  Identify wall layers {L1, L3, L4} from path_characterizer
               Householder H = I - 2vv^T on U_l basis
  3. INJECT:   W_K(l) = H_l·U_base × diag(σ_l) × V_base^T
               Wall layers get reflected U → Im(z_l) = π
               Non-wall layers keep U  → Im(z_l) = 0
  4. TEST:     Single LM Newton step (Pass 6) immediately after init
               + 100 CE → target val=0.045 (gradient_alignment_fix B)

MEASURED PROFILES:
  σ_l (GD-300 student):  [0.95, 2.47, 2.54, 2.15, 2.03, 1.75]
  σ_l (teacher J14):     [0.64, 1.16, 1.23, 1.16, 1.01, 0.96]
  Wall layers (stable):  {1, 3, 4}  from path_characterizer Φ=(0,π,0,π,π)

WALL LAYER IDENTIFICATION:
  The monodromy φ_l = W_K(l+1)·W_K(l)^{-1}
  Im(z_l) = arg(λ_1(φ_l))
  Wall: Im(z_l) = π → λ_1 is negative real → Householder flips sign
  Non-wall: Im(z_l) = 0 → λ_1 is positive real → no flip

THEORY:
  The Householder reflection H = I - 2vv^T applied to U_l rotates
  the leading left singular vector of W_K(l) by π (sign flip).
  This makes the monodromy φ_l = W_K(l+1)·W_K(l)^{-1} have a
  negative leading eigenvalue → Im(z_l) = π (Bridgeland wall).
  The algebraic construction directly encodes the wall structure
  without iterative optimization.
"""
import json, math, warnings, collections, os, sys, time
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
MU_LM = 0.950

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f): print(f"ERROR: {f}"); sys.exit(1)

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

class Attn(nn.Module):
    def __init__(self):
        super().__init__(); dh=D//N_HEADS
        self.WQ=nn.Linear(D,D,bias=False); self.WK=nn.Linear(D,D,bias=False)
        self.WV=nn.Linear(D,D,bias=False); self.op=nn.Linear(D,D,bias=False)
        self.ln=nn.LayerNorm(D); self.sc=math.sqrt(dh); self.nh=N_HEADS; self.dh=dh
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h):
        B,S,_=h.shape
        Q=self.WQ(h).view(B,S,self.nh,self.dh).transpose(1,2)
        K=self.WK(h).view(B,S,self.nh,self.dh).transpose(1,2)
        V=self.WV(h).view(B,S,self.nh,self.dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/self.sc
        mask=torch.triu(torch.ones(S,S),diagonal=1).bool()
        sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        return self.ln(h+self.op((F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D)))
class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.g=nn.Linear(D,D*2,bias=False); self.v=nn.Linear(D,D*2,bias=False)
        self.o=nn.Linear(D*2,D,bias=False); self.n=nn.LayerNorm(D)
        for w in [self.g,self.v,self.o]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h): return self.n(h+self.o(F.silu(self.g(h))*self.v(h)))
class Block(nn.Module):
    def __init__(self): super().__init__(); self.attn=Attn(); self.ff=FF()
    def forward(self,h): return self.ff(self.attn(h))
class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.te=nn.Embedding(VOCAB,D); self.pe=nn.Embedding(512,D)
        self.blocks=nn.ModuleList([Block() for _ in range(N_STU)])
        self.ln_f=nn.LayerNorm(D); self.head=nn.Linear(D,VOCAB,bias=False)
        self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def flat_params(self): return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat(self,v):
        i=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(v[i:i+n].reshape(p.shape)); i+=n

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

def eval_val(m, n=15):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def sheet_angles(model):
    """Measure Im(z_l) = arg(λ_1(W_K(l+1)·W_K(l)^{-1})) for each transition."""
    out=[]; WKs=[model.blocks[l].attn.WK.weight.data.float() for l in range(N_STU)]
    for l in range(N_STU-1):
        try:
            phi=WKs[l+1]@torch.linalg.pinv(WKs[l])
            lam=torch.linalg.eigvals(phi)
            # Use most negative real eigenvalue — this detects Householder walls
            # (wall: has λ=-1; smooth: all λ=+1; argmax(abs) can't distinguish)
            lam_real=lam.real; most_neg=lam[lam_real.argmin()]
            # True Householder wall: eigenvalue exactly -1
            # Numerical smooth: min eigenvalue > -0.9 (positive scaling)
            if float(most_neg.real) < -0.9:
                a=float(torch.angle(most_neg))  # wall: π
            else:
                # smooth: use largest positive eigenvalue
                pos_lam=lam[lam_real > 0]; 
                a=float(torch.angle(pos_lam[pos_lam.real.argmax()])) if len(pos_lam)>0 else 0.0
            out.append('π' if abs(abs(a)-math.pi)<0.3 else '0' if abs(a)<0.3 else f'{a:.2f}')
        except: out.append('?')
    return out

def lm_step(model, mu=MU_LM, n_grad=25, n_hvp=15, n_cg=8):
    """Newton-LM pass 6 — stronger version matching pass13_drinfeld."""
    model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n_grad))/n_grad
    loss.backward()
    g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                 for p in model.parameters()]).detach(); model.zero_grad()
    def _hvp(v):
        model.zero_grad()
        ls=[model(*get_batch())[1] for _ in range(n_hvp)]
        loss2=torch.stack(ls).mean()
        grads=torch.autograd.grad(loss2,list(model.parameters()),create_graph=True)
        gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
        hv=torch.cat([h.flatten() for h in
                      torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)])
        model.zero_grad(); return hv.detach()
    d=torch.zeros_like(g); r=-g.clone(); p_cg=r.clone(); rr=float((r*r).sum())
    for _ in range(n_cg):
        Hp=_hvp(p_cg)+mu*p_cg; alpha=rr/max(float((p_cg*Hp).sum()),1e-10)
        d+=alpha*p_cg; r-=alpha*Hp; rr_new=float((r*r).sum())
        p_cg=r+(rr_new/max(rr,1e-10))*p_cg; rr=rr_new
    w0=model.flat_params(); L0=eval_val(model,n=8)
    model.set_flat(w0+d); L_new=eval_val(model,n=8)
    if L_new<L0: return L_new, True
    model.set_flat(w0); return L0, False


def householder_reflection(v):
    """H = I - 2vv^T  (v must be unit vector, D×1)"""
    v = v / v.norm()
    return torch.eye(D) - 2 * torch.outer(v, v)

def build_wk_householder(U_base, sigma_profile, wall_layers, scale=1.0):
    """
    Construct W_K(l) = H_l · U_base · diag(σ_l) · V_base^T
    Wall layers get Householder reflection H_l = I - 2v_l v_l^T
    Non-wall layers: H_l = I (identity)

    The reflection vector v_l is chosen as the leading left singular
    vector of U_base — this maximally flips the dominant direction,
    making λ_1(φ_l) negative → Im(z_l) = π.

    Args:
        U_base: D×D orthogonal matrix (base left singular vectors)
        sigma_profile: list of D singular values per layer [σ_0,...,σ_{N_STU-1}]
        wall_layers: set of layer indices where Im(z_l) = π is required
        scale: overall scale factor

    Returns:
        list of D×D W_K matrices, one per layer
    """
    WKs = []
    # Base right singular vectors (fixed across layers for consistent monodromy)
    # V_base = I (identity) so W_K = U_l · diag(σ) is non-symmetric
    # This ensures monodromy φ_l = W_K(l+1)·W_K(l)^{-1} has clear wall structure
    V_base = torch.eye(D)

    # Reflection vector: leading column of U_base
    v_reflect = U_base[:, 0].clone()

    for l in range(N_STU):
        sigma_l = sigma_profile[l]  # D-dimensional vector of singular values

        if l in wall_layers:
            # Apply Householder: H_l = I - 2v v^T
            H_l = householder_reflection(v_reflect)
            U_l = H_l @ U_base
        else:
            U_l = U_base.clone()

        # W_K(l) = U_l · diag(σ_l) · V_base^T
        WK_l = (U_l * sigma_l.unsqueeze(0)) @ V_base.T
        WKs.append(WK_l * scale)

    return WKs

def verify_wall_structure(WKs, wall_layers):
    """Verify that wall layers have Im(z_l)=π and non-wall have Im(z_l)=0."""
    print(f"  Verifying Householder wall structure:")
    # wall_layers = set of layer indices using H·U_base
    # transition (l→l+1) is a wall when layer (l+1) ∈ wall_layers
    for l in range(len(WKs)-1):
        phi = WKs[l+1] @ torch.linalg.pinv(WKs[l])
        lam = torch.linalg.eigvals(phi)
        most_neg = lam[lam.real.argmin()]
        lam1 = most_neg if float(most_neg.real) < -0.9 else lam[lam.abs().argmax()]
        angle = float(torch.angle(lam1))
        is_wall = (l+1) in wall_layers  # transition wall = next layer uses H·U
        expected = 'π (wall)' if is_wall else '0 (smooth)'
        actual = 'π' if abs(abs(angle)-math.pi)<0.3 else '0' if abs(angle)<0.3 else f'{angle:.2f}'
        match = '✓' if (is_wall and actual=='π') or (not is_wall and actual=='0') else '✗'
        print(f"  L{l}→L{l+1}: expected {expected:12s}  got {actual:6s}  {match}")


# ══════════════════════════════════════════════════════════════
print("="*65)
print("HOUSEHOLDER W_K INITIALIZATION EXPERIMENT")
print("Directly constructing Kac-Moody orbit topological fixed point")
print("="*65); print()

# ── CORPUS + SPECTRAL E₀ ─────────────────────────────────────
bigram=collections.Counter(); perm={}
for i in range(len(train_ids)-1):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB:
        bigram[(a,b)]+=1
        if a not in perm: perm[a]=b
rows,cols,vv=[],[],[]
for (a,b),cnt in bigram.items(): rows.append(a); cols.append(b); vv.append(float(cnt))
W_sp=sp.csr_matrix((vv,(rows,cols)),shape=(VOCAB,VOCAB),dtype=np.float32)
W_sp=W_sp+W_sp.T; d_inv=np.array(1.0/(W_sp.sum(1)+1e-8)).flatten()
Dsi=sp.diags(np.sqrt(d_inv)); L_sym=sp.eye(VOCAB)-Dsi@W_sp@Dsi
evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)
idx_s=np.argsort(evals); evecs=evecs[:,idx_s][:,1:D+1]
E_0=(evecs/(np.sqrt(evals[idx_s[1:D+1]])+1e-8)[np.newaxis,:]).astype(np.float32)
E_0=(E_0/(E_0.std()+1e-8)*0.02)
E_next=np.array([E_0[perm.get(t,t)] for t in range(VOCAB)],dtype=np.float32)
E_init=(0.9*E_0+0.1*E_next)
E_norm=float(np.linalg.norm(E_0))
E_init=(E_init*(E_norm/max(float(np.linalg.norm(E_init)),1e-8))).astype(np.float32)
print(f"Corpus: VOCAB={VOCAB}, nnz={len(bigram)}")

# Get U_base from spectral E₀ SVD (the natural basis)
E0_t = torch.tensor(E_0[:D, :D]).float()
U_base, S_base, Vh_base = torch.linalg.svd(E0_t, full_matrices=True)
print(f"U_base from spectral E₀ SVD: shape {U_base.shape}")
print()

# ── MEASURED PROFILES ──────────────────────────────────────
# From serre_decay_experiment.py (GD-300 trained student)
sigma_student = [0.95, 2.47, 2.54, 2.15, 2.03, 1.75]
# From teacher J14 profile (compiler_j14.py run)
sigma_teacher = [0.6378, 1.1627, 1.2337, 1.1591, 1.0070, 0.9560]

# Wall layers from stable orbit in path_characterizer
# Φ=(0,π,0,π,π) → walls at transitions 1,3,4
# Transition l means between layer l and l+1
# Wall at transition 1: layer index 1 in wall_layers
WALL_LAYERS = {1, 3, 4}  # from confirmed stable orbit (0,π,0,π,π)
print(f"Wall layers: {WALL_LAYERS}  (from stable orbit Φ=(0,π,0,π,π))")
print()

# ── CONSTRUCT SIGMA PROFILES (D-dim per layer) ───────────────
def make_sigma_profile(sv1_per_layer, scale=1.0):
    """Build D-dim sigma vector for each layer.
    Leading SV = sv1_per_layer[l], rest decay geometrically."""
    profiles = []
    for sv1 in sv1_per_layer:
        # Geometric decay for remaining singular values
        sigma = torch.zeros(D)
        for i in range(D):
            sigma[i] = sv1 * (0.5 ** i) * scale
        sigma[0] = sv1 * scale
        profiles.append(sigma)
    return profiles

# ── FOUR EXPERIMENTS ──────────────────────────────────────────
results = {}

# CORRECT WALL GEOMETRY: walls at {1,3} give alternating (0,π,0,π,0) pattern
# Consecutive walls cancel (H²=I), so use non-consecutive indices
for exp_name, sigma_list, wall_set, scale in [
    ("A: student σ, walls {1,3}",         sigma_student, {1,3},   1.0),
    ("B: teacher σ, walls {1,3}",         sigma_teacher, {1,3},   1.0),
    ("C: teacher σ, walls {0,2,4}",       sigma_teacher, {0,2,4}, 1.0),
    ("D: teacher σ, no walls (baseline)", sigma_teacher, set(),    1.0),
]:
    print(f"━━━ EXPERIMENT {exp_name} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    sigma_profile = make_sigma_profile(sigma_list, scale=scale)

    torch.manual_seed(99)
    model = LM()
    model.te.weight.data.copy_(torch.tensor(E_init))

    # Build and inject Householder W_K
    WKs = build_wk_householder(U_base, sigma_profile, wall_set)
    with torch.no_grad():
        for l in range(N_STU):
            model.blocks[l].attn.WK.weight.data.copy_(WKs[l])
            # W_Q = W_K for simplicity (symmetric attention at init)
            model.blocks[l].attn.WQ.weight.data.copy_(WKs[l])

    v_init = eval_val(model)
    phi_init = sheet_angles(model)
    verify_wall_structure(WKs, wall_set)
    print(f"  After Householder init: val={v_init:.4f}  Φ={phi_init}")

    # Single LM Newton step (Pass 6) immediately
    v_lm, acc = lm_step(model)
    phi_lm = sheet_angles(model)
    print(f"  After LM (Pass 6):      val={v_lm:.4f}  {'✓' if acc else '~'}  Φ={phi_lm}")

    # 100 CE continuation
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
    for step in range(1, 101):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in {25, 50, 75, 100}:
            v = eval_val(model)
            print(f"    CE {step:3d}: val={v:.4f}  Φ={sheet_angles(model)}")

    v_final = eval_val(model)
    phi_final = sheet_angles(model)
    results[exp_name] = v_final
    print(f"  FINAL val={v_final:.4f}  Φ={phi_final}")
    print()

# ── COMPARISON ────────────────────────────────────────────────
print("="*65)
print("HOUSEHOLDER RESULTS")
print("="*65)
print()
print(f"  {'Experiment':<40} {'val':>7}")
print("  "+"-"*48)
for name, val in results.items():
    print(f"  {name:<40} {val:>7.4f}")
print()
print("  BASELINES:")
print("  GD-300 (cosine):            val=0.244")
print("  MF10+basin+167CE (confirmed):val=0.062")
print("  gradient_alignment_fix B:    val=0.045  (LM+100CE from val=0.284)")
print()
print("  IF any Householder experiment < 0.244:")
print("  → algebraic W_K init beats GD without MF pump")
print("  IF < 0.062: → beats confirmed compiler")
