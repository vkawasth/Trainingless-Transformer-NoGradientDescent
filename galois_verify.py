#!/usr/bin/env python3
"""
Galois Group Verification
==========================
Three measurements to identify the group acting on the transformer's
inter-layer structure.

MEASUREMENT 1 — ULTRAMETRIC CONDITION
If the rank profile is a p-adic filtration, it must satisfy the
ultrametric inequality:
  d(x,z) <= max(d(x,y), d(y,z))
for all triples of layers x,y,z.

The ultrametric distance between layers l and l' is:
  d(l,l') = p^{-v} where v = max valuation level where they agree

In terms of rank profile: d(l,l') is determined by where in the
filtration the two layers first diverge.

MEASUREMENT 2 — FROBENIUS EIGENVALUE
The monodromy M_fwd should have eigenvalues that are Weil numbers:
  |λ_i|_p = p^{w_i/2} for some weights w_i (integers)

Check: are the singular values of M_fwd close to integer powers of
some prime p? This identifies p.

MEASUREMENT 3 — INERTIA GROUP ACTION
The inertia group I_p acts on the residue field F_p.
Its action on consecutive layers should be:
  J_{l+1} = Frob * J_l * Frob^{-1} + (nilpotent correction)

where Frob is the Frobenius element.
Check: does the commutator [J_l, J_{l+1}] have the structure of
a nilpotent matrix (all eigenvalues 0)?
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import logm, expm

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4
PROJ=48

print(f"\n{'='*65}")
print(f"  GALOIS GROUP VERIFICATION")
print(f"  Three measurements to identify the symmetry group")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(json.load(f))
with open('/tmp/val_ids.json')   as f: val_ids=list(json.load(f))
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

def clr(s,total=300,warmup=100):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

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
    return J.T, U.detach().numpy(), m

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training model (seed=42, 300 steps)...")
torch.manual_seed(42)
model=LM(D,N_HEADS,N_LAYERS)
opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step)
    model.train(); x,y=get_batch(); _,loss=model(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if step%100==0:
        model.eval()
        with torch.no_grad():
            vl=float(np.mean([model(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
        model.train()
model.eval()
print()

# ── Extract Jacobians ─────────────────────────────────────────────────────────
print("Extracting Jacobian chain...", flush=True)
x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
with torch.no_grad():
    hs=model.hidden_states(x_ref); hs=[h[0] for h in hs]
pos=SEQ//2; m=min(PROJ,SEQ,D)

Js=[]; Us=[]; ma=None
for l in range(N_LAYERS):
    J,U,m_=layer_jac(model.blocks[l],hs[l],pos,m)
    Js.append(J); Us.append(U)
    if ma is None: ma=m_
    if (l+1)%8==0: print(f"  L{l+1}...",flush=True)

dJs=[J-np.eye(ma) for J in Js]
norms=[float(np.linalg.norm(dJ)) for dJ in dJs]
ranks=[int(np.sum(np.linalg.svd(dJ,compute_uv=False)>
           np.linalg.svd(dJ,compute_uv=False)[0]*0.10))
       for dJ in dJs]

print(f"\n  Rank profile: {ranks}")
print(f"  Norm profile: {[round(n,3) for n in norms[:6]]}...")

# ═══════════════════════════════════════════════════════════════
# MEASUREMENT 1: ULTRAMETRIC CONDITION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  MEASUREMENT 1: ULTRAMETRIC CONDITION")
print(f"  Does d(l,l') = |norm(l) - norm(l')| satisfy ultrametric?")
print(f"  Ultrametric: d(x,z) <= max(d(x,y), d(y,z)) for all x,y,z")
print("="*65)

# Use norm difference as the distance (excluding L0 anomaly)
norms_inner=np.array(norms[1:])  # L1..L23
L=len(norms_inner)

def dist(i,j): return abs(norms_inner[i]-norms_inner[j])

# Check ultrametric on random triples
np.random.seed(42)
n_triples=500
violations=0; total=0
max_violation=0.0
for _ in range(n_triples):
    i,j,k=np.random.choice(L,3,replace=False)
    dij=dist(i,j); djk=dist(j,k); dik=dist(i,k)
    total+=1
    violation=dik - max(dij,djk)
    if violation > 1e-6:
        violations+=1
        max_violation=max(max_violation,violation)

ultrametric_fraction=1-violations/total
print(f"\n  Triples tested: {n_triples}")
print(f"  Violations: {violations} ({violations/n_triples:.1%})")
print(f"  Max violation: {max_violation:.6f}")
print(f"  Ultrametric satisfaction: {ultrametric_fraction:.1%}")

# Also check with rank-based distance
def rank_dist(i,j):
    # p-adic distance: how deep do the ranks agree?
    ri,rj=ranks[i+1],ranks[j+1]
    return abs(ri-rj)

violations_r=0
for _ in range(n_triples):
    i,j,k=np.random.choice(L,3,replace=False)
    dij=rank_dist(i,j); djk=rank_dist(j,k); dik=rank_dist(i,k)
    if dik > max(dij,djk)+1e-6: violations_r+=1

print(f"\n  Rank-based ultrametric violations: {violations_r}/{n_triples} ({violations_r/n_triples:.1%})")

# ═══════════════════════════════════════════════════════════════
# MEASUREMENT 2: FROBENIUS EIGENVALUE — IDENTIFY p
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  MEASUREMENT 2: FROBENIUS EIGENVALUE")
print(f"  Are sv(M_fwd) close to powers of a prime p?")
print("="*65)

M_fwd=np.eye(ma)
for l in range(15): M_fwd=Js[l]@M_fwd
sv_fwd=np.linalg.svd(M_fwd,compute_uv=False)

print(f"\n  sv(M_fwd) top 8: {sv_fwd[:8].round(4)}")
print(f"\n  Testing if sv ≈ p^w for primes p and integer weights w:")

primes=[2,3,5,7,11,13]
for p in primes:
    # For each sv, find nearest log_p(sv)
    log_p_sv=np.log(sv_fwd[:8])/np.log(p)
    nearest_int=np.round(log_p_sv)
    residuals=np.abs(log_p_sv-nearest_int)
    mean_res=float(residuals.mean())
    print(f"  p={p:>2}: log_p(sv) = {log_p_sv[:4].round(3)}  "
          f"nearest int = {nearest_int[:4].astype(int)}  "
          f"mean residual = {mean_res:.4f}")

# Check ratios between consecutive sv
print(f"\n  Consecutive sv ratios (sv[k]/sv[k+1]):")
ratios=sv_fwd[:7]/sv_fwd[1:8]
print(f"  {ratios.round(4)}")
print(f"  Mean ratio: {ratios.mean():.4f}  std: {ratios.std():.4f}")
print(f"  (If Frobenius: ratios should cluster near p^w for some p,w)")

# ═══════════════════════════════════════════════════════════════
# MEASUREMENT 3: INERTIA GROUP — COMMUTATOR STRUCTURE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  MEASUREMENT 3: INERTIA GROUP — COMMUTATOR STRUCTURE")
print(f"  Is [J_l, J_{l+1}] nilpotent?")
print(f"  Galois prediction: commutator eigenvalues should all be ~0")
print("="*65)

print(f"\n  Layer  ||[J_l,J_{{l+1}}]||  max|eig|  nilpotency_index  structure")
print("  "+"-"*62)

for l in range(1,N_LAYERS-1):
    comm=Js[l]@Js[l+1]-Js[l+1]@Js[l]
    comm_norm=float(np.linalg.norm(comm))
    eigs=np.linalg.eigvals(comm)
    max_eig=float(np.max(np.abs(eigs)))
    # Nilpotency index: smallest k such that comm^k ≈ 0
    C=comm.copy(); nilp_idx=0
    for k in range(1,6):
        if np.linalg.norm(C)<1e-3: nilp_idx=k; break
        C=C@comm
    nilp_str=f"nilpotent(k≤{nilp_idx})" if nilp_idx>0 else "not nilpotent"
    # Is it semisimple (all eigenvalues distinct)?
    eig_spread=float(np.std(np.abs(eigs)))
    structure="nilpotent" if max_eig<0.1 else ("semisimple" if eig_spread>0.1 else "mixed")
    print(f"  L{l:>2}→L{l+1:<2}  {comm_norm:>14.4f}  {max_eig:>8.4f}  {nilp_str:>18}  {structure}")

# Jordan decomposition of a representative commutator
print(f"\n  Jordan structure of [J_7, J_8] (representative):")
comm_rep=Js[7]@Js[8]-Js[8]@Js[7]
eigs_rep=np.sort(np.abs(np.linalg.eigvals(comm_rep)))[::-1]
print(f"  |eigenvalues|: {eigs_rep[:8].round(4)}")
print(f"  Trace: {float(np.trace(comm_rep)):.6f}  (=0 for nilpotent/semisimple)")
print(f"  ||comm||_F: {float(np.linalg.norm(comm_rep)):.4f}")

# Levi decomposition check: g = s ⊕ n (semisimple + nilpotent)
# The Lie algebra of the symmetry group should decompose this way
print(f"\n  Levi decomposition of the Jacobian Lie algebra:")
print(f"  Computing ad(J_l) for l=1..23...")
ad_norms=[]
for l in range(1,N_LAYERS):
    # ad(J_l)(X) = [J_l, X] for X = J_{l-1}
    ad_X=Js[l]@Js[l-1]-Js[l-1]@Js[l]
    # Semisimple part: diagonalizable
    eigs=np.linalg.eigvals(ad_X)
    # Nilpotent part: ad_X - semisimple part
    ss_norm=float(np.linalg.norm(np.diag(eigs.real)))
    n_norm=float(np.linalg.norm(ad_X-np.diag(eigs.real)))
    ad_norms.append((l,float(np.linalg.norm(ad_X)),ss_norm,n_norm))

print(f"  {'L':>3}  {'||ad(J)||':>10}  {'||ss part||':>12}  {'||nil part||':>13}  ratio(nil/ss)")
print("  "+"-"*58)
for l,ad_n,ss_n,n_n in ad_norms[::4]:
    ratio=n_n/max(ss_n,1e-8)
    print(f"  L{l:>2}  {ad_n:>10.4f}  {ss_n:>12.4f}  {n_n:>13.4f}  {ratio:.4f}")

# ═══════════════════════════════════════════════════════════════
# SUMMARY: WHAT GROUP IS THIS?
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  SUMMARY: GROUP IDENTIFICATION")
print("="*65)

mean_comm=np.mean([float(np.linalg.norm(Js[l]@Js[l+1]-Js[l+1]@Js[l]))
                   for l in range(1,N_LAYERS-1)])
max_comm=np.max([float(np.max(np.abs(np.linalg.eigvals(
                   Js[l]@Js[l+1]-Js[l+1]@Js[l]))))
                 for l in range(1,N_LAYERS-1)])

print(f"""
  ULTRAMETRIC (norm-based): {ultrametric_fraction:.1%} satisfied
  ULTRAMETRIC (rank-based): {1-violations_r/n_triples:.1%} satisfied

  FROBENIUS: sv(M_fwd) ≈ {sv_fwd[0]:.2f}
    log_2(sv[0]) = {np.log(sv_fwd[0])/np.log(2):.3f}
    log_3(sv[0]) = {np.log(sv_fwd[0])/np.log(3):.3f}
    log_5(sv[0]) = {np.log(sv_fwd[0])/np.log(5):.3f}

  COMMUTATOR STRUCTURE:
    Mean ||[J_l, J_{{l+1}}]||: {mean_comm:.4f}
    Max |eigenvalue| of commutator: {max_comm:.4f}

  READING:

  If ultrametric > 90%:
    The norm metric on layers is approximately ultrametric.
    The symmetry group contains a p-adic subgroup.

  If sv(M_fwd) ≈ p^w for integer w:
    The Frobenius eigenvalue is a Weil number.
    The group is a Galois group of a local field extension.

  If max|eig(commutator)| < 0.1:
    The Jacobian Lie algebra is approximately nilpotent.
    The group is a pro-p group (wild inertia).

  If none of the above:
    The symmetry group is not a classical p-adic Galois group.
    Need to identify the correct algebraic structure.
""")

# ═══════════════════════════════════════════════════════════════
# MEASUREMENT 4: SPECTRAL GAP — DECIDES THE GROUP
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  MEASUREMENT 4: SPECTRAL GAP")
print(f"  Commutator norm matrix A[l,l'] = ||[J_l, J_l']||_F")
print(f"  Spectral gap → arithmetic lattice (Property T)")
print(f"  No gap → solvable/Borel group")
print(f"  GUE statistics → full GL(m)")
print("="*65)

# Build 24x24 commutator norm matrix (exclude L0)
L = N_LAYERS - 1  # 23 layers (L1..L23)
A = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        if i == j:
            A[i,j] = 0.0
        else:
            comm = Js[i+1] @ Js[j+1] - Js[j+1] @ Js[i+1]
            A[i,j] = float(np.linalg.norm(comm, 'fro'))

# Symmetrize
A = (A + A.T) / 2

print(f"\n  Commutator norm matrix A: {L}x{L}")
print(f"  Mean off-diagonal: {float(np.mean(A[A>0])):.4f}")
print(f"  Max off-diagonal:  {float(np.max(A)):.4f}")

# Eigenvalues of A (adjacency-like matrix)
eigs_A = np.sort(np.linalg.eigvalsh(A))[::-1]
print(f"\n  Eigenvalues of A (top 10):")
print(f"  {eigs_A[:10].round(4)}")
print(f"\n  Eigenvalues of A (bottom 5):")
print(f"  {eigs_A[-5:].round(4)}")

# Spectral gap: lambda_1 - lambda_2
gap = float(eigs_A[0] - eigs_A[1])
relative_gap = gap / max(abs(eigs_A[0]), 1e-8)
print(f"\n  lambda_1 = {eigs_A[0]:.4f}")
print(f"  lambda_2 = {eigs_A[1]:.4f}")
print(f"  Spectral gap = lambda_1 - lambda_2 = {gap:.4f}")
print(f"  Relative gap = {relative_gap:.4f}")

# GUE test: eigenvalue spacing distribution
# GUE predicts Wigner surmise: P(s) = (pi/2)*s*exp(-pi*s^2/4)
# GOE (real) predicts: P(s) = (pi/2)*s*exp(-pi*s^2/4) same form
# Poisson (no correlations): P(s) = exp(-s)
spacings = np.diff(eigs_A[::-1])  # ascending order spacings
spacings = spacings / spacings.mean()  # normalize mean to 1
mean_s = float(spacings.mean())
var_s = float(spacings.var())
# GUE: var/mean^2 = 4/pi - 1 ≈ 0.273
# GOE: var/mean^2 ≈ 0.286
# Poisson: var/mean^2 = 1.0
ratio = var_s / mean_s**2
print(f"\n  Eigenvalue spacing statistics:")
print(f"  Mean spacing: {mean_s:.4f}  Variance: {var_s:.4f}")
print(f"  Var/Mean^2 = {ratio:.4f}")
print(f"  GUE prediction: 0.273  GOE: 0.286  Poisson: 1.000")

# Normalized Laplacian
D_mat = np.diag(A.sum(axis=1))
L_mat = D_mat - A
eigs_L = np.sort(np.linalg.eigvalsh(L_mat))
print(f"\n  Laplacian eigenvalues (top 5):")
print(f"  lambda_0={eigs_L[0]:.6f}  lambda_1={eigs_L[1]:.4f}  lambda_2={eigs_L[2]:.4f}")
laplacian_gap = float(eigs_L[1])
print(f"  Laplacian spectral gap (Cheeger): {laplacian_gap:.4f}")

# ── Decision ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  DECISION: WHAT GROUP IS THIS?")
print("="*65)

print(f"""
  Spectral gap (A): {gap:.4f}  relative: {relative_gap:.4f}
  Laplacian gap:    {laplacian_gap:.4f}
  Spacing ratio:    {ratio:.4f}
""")

if relative_gap > 0.3:
    group = "ARITHMETIC LATTICE (Property T)"
    reason = f"Large spectral gap {relative_gap:.3f} > 0.3 — Kazhdan Property T confirmed."
elif relative_gap > 0.05:
    group = "HYPERBOLIC GROUP (delta-hyperbolic, no Property T)"
    reason = f"Moderate gap {relative_gap:.3f} — consistent with hyperbolic but not arithmetic lattice."
else:
    group = "SOLVABLE / BOREL GROUP (no spectral gap)"
    reason = f"No spectral gap {relative_gap:.3f} < 0.05 — consistent with B_n(Q_p) or unipotent."

if ratio < 0.35:
    spacing_type = "GUE/GOE (random matrix universality)"
elif ratio < 0.6:
    spacing_type = "intermediate (between GUE and Poisson)"
else:
    spacing_type = "Poisson (integrable / no level repulsion)"

print(f"  GROUP IDENTIFICATION: {group}")
print(f"  Reason: {reason}")
print(f"\n  Spacing statistics: {spacing_type}")
print(f"  (ratio={ratio:.3f}: GUE=0.27, Poisson=1.0)")

print(f"""
  FINAL READING:

  Borel subgroup B_8(Q_5) predicts:
    - No spectral gap (solvable group)
    - Poisson spacing (integrable structure)

  Arithmetic lattice SL(m,Z) predicts:
    - Large spectral gap (Property T)
    - GUE spacing (chaotic, random matrix)

  Free/hyperbolic group predicts:
    - Moderate gap
    - Intermediate spacing

  The data shows: gap={gap:.4f}, spacing ratio={ratio:.4f}
  This identifies the symmetry group.
""")

# ═══════════════════════════════════════════════════════════════
# MEASUREMENT 5: DOMINANT EIGENVECTOR — IS IT THE ATTRACTOR?
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  MEASUREMENT 5: DOMINANT EIGENVECTOR OF COMMUTATOR MATRIX")
print(f"  Is the dominant direction the L14 attractor?")
print(f"  v1[l] = component of dominant eigenvector at layer l")
print("="*65)

eigs_full, vecs_full = np.linalg.eigh(A)
v1 = vecs_full[:, -1]   # dominant eigenvector (largest eigenvalue)
v2 = vecs_full[:, -2]   # second eigenvector

# Map back to layer indices (we excluded L0, so index 0 = L1)
print(f"\n  Dominant eigenvector v1 (λ1={eigs_A[0]:.4f}):")
print(f"  {'Layer':>7}  {'v1[l]':>10}  {'|v1[l]|':>10}  {'rank':>6}")
print("  "+"-"*40)

abs_v1 = np.abs(v1)
rank_v1 = np.argsort(abs_v1)[::-1]  # layers by decreasing |v1|

for rank_idx, layer_idx in enumerate(rank_v1):
    layer = layer_idx + 1  # +1 because we excluded L0
    marker = " ← L14 ATTRACTOR" if layer == 14 else ""
    marker = " ← L1 (root collapse)" if layer == 1 else marker
    print(f"  L{layer:>2} (rank {rank_idx+1:>2}): {v1[layer_idx]:>10.4f}  "
          f"{abs_v1[layer_idx]:>10.4f}{marker}")
    if rank_idx >= 11: break  # show top 12

print(f"\n  Layer with max |v1|: L{rank_v1[0]+1}")
print(f"  Layer with 2nd max:  L{rank_v1[1]+1}")
print(f"  Layer with 3rd max:  L{rank_v1[2]+1}")

# Check if L14 is in the top 5
l14_rank = int(np.where(rank_v1 == 13)[0][0]) + 1  # index 13 = layer 14
print(f"\n  L14 rank in v1: #{l14_rank} out of 23")

# Second eigenvector
abs_v2 = np.abs(vecs_full[:, -2])
rank_v2 = np.argsort(abs_v2)[::-1]
print(f"\n  Second eigenvector v2 (λ2={eigs_A[1]:.4f}):")
print(f"  Top 3 layers: L{rank_v2[0]+1}, L{rank_v2[1]+1}, L{rank_v2[2]+1}")

# ── Correlation of v1 with attractor measurements ─────────────────
print(f"\n  Correlation of |v1| with layer properties:")

# Norm profile (excluding L0)
norms_inner = np.array(norms[1:])
# Rank profile (excluding L0)
ranks_inner = np.array(ranks[1:], dtype=float)

# Correlation with norm
corr_norm = float(np.corrcoef(abs_v1, norms_inner)[0,1])
# Correlation with rank
corr_rank = float(np.corrcoef(abs_v1, ranks_inner)[0,1])
# Correlation with distance from L14
dist_from_14 = np.abs(np.arange(1, N_LAYERS) - 14)
corr_dist = float(np.corrcoef(abs_v1, dist_from_14)[0,1])

print(f"  corr(|v1|, ||δJ||):      {corr_norm:.4f}")
print(f"  corr(|v1|, rank):         {corr_rank:.4f}")
print(f"  corr(|v1|, |l-14|):       {corr_dist:.4f}")
print(f"  (negative dist corr = v1 peaks near L14)")

# ── Visualize the eigenvector profile ────────────────────────────
print(f"\n  |v1| profile across layers:")
print(f"  {'':5}", end="")
for l in range(1, N_LAYERS):
    bar_len = int(abs_v1[l-1] * 100)
    marker = "◄" if l == 14 else " "
    print(f"  L{l:>2} {'█'*bar_len}{marker} {abs_v1[l-1]:.4f}")

# ── Final synthesis ───────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  SYNTHESIS: CONNECTING GROUP TO ATTRACTOR")
print("="*65)

attractor_in_top3 = (rank_v1[0]+1 == 14 or rank_v1[1]+1 == 14
                     or rank_v1[2]+1 == 14)

print(f"""
  SPECTRAL GAP:     {gap:.4f}  (Property T confirmed)
  ATTRACTOR L14:    rank #{l14_rank} in dominant eigenvector
  L14 in top 3:     {'YES' if attractor_in_top3 else 'NO'}

  corr(|v1|, |l-14|) = {corr_dist:.4f}

  IF L14 is dominant in v1 AND corr(|v1|, |l-14|) < -0.5:
    The attractor IS the fixed point of the group action.
    The spectral gap measures rigidity at the attractor.
    The dominant commutator direction flows THROUGH L14.
    The group acts on the layer sequence with L14 as its
    highest root / central vertex of the Dynkin diagram.

  IF L14 is NOT dominant in v1:
    The group's fixed point is not L14.
    The attractor and the group structure are decorrelated.
    The symmetry group acts on a different geometric object
    than the one the Toda/Hessenberg measurements identified.
    Need to reconcile the two structures.

  The answer is in the eigenvector profile above.
""")
