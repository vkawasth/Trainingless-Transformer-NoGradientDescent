#!/usr/bin/env python3
"""
Group Universality Tests
=========================
Three tests to determine whether the group structure
(Property T, L14 centrality, two-component inertia)
is a universal architectural invariant or seed/training-dependent.

TEST 1: Spectral gap across seeds
  Compare gap(A) for Model A (seed=42) vs Model B (seed=137)
  If gap_A ≈ gap_B: Property T is architectural

TEST 2: Attractor center vs training depth
  Compare dominant eigenvector for Model C (150 steps, partially trained)
  vs Model A (300 steps, converged)
  If center shifts: group is training-dependent
  If center stays at L14: group is architectural

TEST 3: Full spectral universality
  Compare all 23 eigenvalues of A for Model A vs B
  If spectra match: group determined by (architecture, data)
  If spectra differ: group is seed-dependent
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4; PROJ=48

print(f"\n{'='*65}")
print(f"  GROUP UNIVERSALITY TESTS")
print(f"  Is the group an architectural invariant?")
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

def extract_commutator_matrix(model, x_ref, pos, m=PROJ):
    """Extract full 23x23 commutator norm matrix (excluding L0)."""
    with torch.no_grad():
        hs=model.hidden_states(x_ref); hs=[h[0] for h in hs]
    Js=[]
    for l in range(N_LAYERS):
        J,U,m_=layer_jac(model.blocks[l],hs[l],pos,m)
        Js.append(J)
    # Build 23x23 commutator norm matrix (L1..L23)
    L=N_LAYERS-1
    A=np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            if i!=j:
                comm=Js[i+1]@Js[j+1]-Js[j+1]@Js[i+1]
                A[i,j]=float(np.linalg.norm(comm,'fro'))
    A=(A+A.T)/2
    return A, Js

def analyze_commutator_matrix(A, label=""):
    """Extract spectral gap, dominant eigenvector, and spacing statistics."""
    eigs,vecs=np.linalg.eigh(A)
    eigs_desc=eigs[::-1]; vecs_desc=vecs[:,::-1]
    gap=float(eigs_desc[0]-eigs_desc[1])
    rel_gap=gap/max(abs(eigs_desc[0]),1e-8)
    v1=np.abs(vecs_desc[:,0])
    # Attractor correlation
    dist_from_14=np.abs(np.arange(1,N_LAYERS)-14)
    corr=float(np.corrcoef(v1,dist_from_14)[0,1])
    # Peak layer
    peak_layer=int(np.argmax(v1))+1
    # Top 3 layers
    top3=[int(np.argsort(v1)[::-1][k])+1 for k in range(3)]
    # Spacing stats
    spacings=np.diff(eigs[::-1])
    spacings=spacings/max(spacings.mean(),1e-8)
    var_ratio=float(spacings.var()/max(spacings.mean()**2,1e-8))
    return {
        'label':label,
        'lambda1':float(eigs_desc[0]),
        'lambda2':float(eigs_desc[1]),
        'gap':gap, 'rel_gap':rel_gap,
        'v1':v1, 'eigs':eigs_desc,
        'corr_l14':corr, 'peak_layer':peak_layer,
        'top3':top3, 'var_ratio':var_ratio,
        'vecs':vecs_desc
    }

# ── Train models ──────────────────────────────────────────────────────────────
def train(seed, steps, label):
    torch.manual_seed(seed)
    m=LM(D,N_HEADS,N_LAYERS)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    t0=time.time()
    for step in range(1,steps+1):
        for pg in opt.param_groups: pg['lr']=clr(step,steps)
        m.train(); x,y=get_batch(); _,loss=m(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if step%(steps//3)==0:
            m.eval()
            with torch.no_grad():
                vl=float(np.mean([m(*get_batch('val'))[1].item() for _ in range(10)]))
            print(f"  [{label}] step {step}/{steps}  val={vl:.4f}  t={time.time()-t0:.0f}s")
            m.train()
    m.eval()
    with torch.no_grad():
        vl=float(np.mean([m(*get_batch('val'))[1].item() for _ in range(30)]))
    print(f"  [{label}] final val={vl:.4f}")
    return m

print("Training models...")
mA=train(42,  300, "A  seed=42  300 steps")
mB=train(137, 300, "B  seed=137 300 steps")
mC=train(999, 150, "C  seed=999 150 steps (partial)")
print()

# ── Reference input (same for all) ───────────────────────────────────────────
torch.manual_seed(0)
x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
pos=SEQ//2

# ── Extract commutator matrices ───────────────────────────────────────────────
print("Extracting commutator matrices...", flush=True)
print("  Model A...", flush=True); AA,JsA=extract_commutator_matrix(mA,x_ref,pos)
print("  Model B...", flush=True); AB,JsB=extract_commutator_matrix(mB,x_ref,pos)
print("  Model C...", flush=True); AC,JsC=extract_commutator_matrix(mC,x_ref,pos)

rA=analyze_commutator_matrix(AA,"A")
rB=analyze_commutator_matrix(AB,"B")
rC=analyze_commutator_matrix(AC,"C-partial")

# ═══════════════════════════════════════════════════════════════
# TEST 1: SPECTRAL GAP ACROSS SEEDS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  TEST 1: SPECTRAL GAP ACROSS SEEDS")
print(f"  Does Property T persist for different seeds?")
print("="*65)

print(f"\n  {'Model':>12}  {'λ1':>8}  {'λ2':>8}  {'gap':>8}  {'rel_gap':>9}  {'verdict'}")
print("  "+"-"*60)
for r in [rA,rB,rC]:
    ok="✓ Property T" if r['rel_gap']>0.3 else "✗ No gap"
    print(f"  {r['label']:>12}  {r['lambda1']:>8.4f}  {r['lambda2']:>8.4f}"
          f"  {r['gap']:>8.4f}  {r['rel_gap']:>9.4f}  {ok}")

gap_diff_AB=abs(rA['gap']-rB['gap'])
gap_diff_AC=abs(rA['gap']-rC['gap'])
print(f"\n  Gap difference A vs B: {gap_diff_AB:.4f}")
print(f"  Gap difference A vs C: {gap_diff_AC:.4f}")
t1_ok=(rA['rel_gap']>0.3 and rB['rel_gap']>0.3 and gap_diff_AB/rA['gap']<0.3)
print(f"\n  TEST 1: {'✓ CONFIRMED — Property T is seed-invariant' if t1_ok else '✗ FAILED — gap is seed-dependent'}")

# ═══════════════════════════════════════════════════════════════
# TEST 2: ATTRACTOR CENTER VS TRAINING DEPTH
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  TEST 2: ATTRACTOR CENTER VS TRAINING DEPTH")
print(f"  Does L14 remain central for partially trained model?")
print("="*65)

print(f"\n  {'Model':>12}  {'peak_l':>7}  {'top3':>12}  {'corr(v1,|l-14|)':>18}  {'verdict'}")
print("  "+"-"*64)
for r in [rA,rB,rC]:
    ok="✓ L14 central" if r['corr_l14']<-0.5 else "✗ shifted"
    print(f"  {r['label']:>12}  L{r['peak_layer']:>2}     "
          f"  {str([f'L{l}' for l in r['top3']]):>12}  {r['corr_l14']:>18.4f}  {ok}")

t2_ok=(rC['corr_l14']<-0.4)
print(f"\n  TEST 2: {'✓ CONFIRMED — L14 centrality is training-invariant' if t2_ok else '✗ FAILED — center shifts with training depth'}")

# ═══════════════════════════════════════════════════════════════
# TEST 3: FULL SPECTRAL UNIVERSALITY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  TEST 3: FULL SPECTRAL UNIVERSALITY")
print(f"  Do all 23 eigenvalues match across seeds?")
print("="*65)

# Align spectra (both sorted descending)
eA=rA['eigs']; eB=rB['eigs']; eC=rC['eigs']
corr_AB=float(np.corrcoef(eA,eB)[0,1])
corr_AC=float(np.corrcoef(eA,eC)[0,1])
mean_abs_diff_AB=float(np.mean(np.abs(eA-eB)))
mean_abs_diff_AC=float(np.mean(np.abs(eA-eC)))

print(f"\n  Eigenvalue correlation A vs B: {corr_AB:.4f}")
print(f"  Eigenvalue correlation A vs C: {corr_AC:.4f}")
print(f"  Mean |λ_A - λ_B|: {mean_abs_diff_AB:.4f}")
print(f"  Mean |λ_A - λ_C|: {mean_abs_diff_AC:.4f}")

print(f"\n  Per-eigenvalue comparison A vs B (top 10):")
print(f"  {'k':>4}  {'λ_A':>8}  {'λ_B':>8}  {'diff':>8}  {'rel_diff':>10}")
print("  "+"-"*44)
for k in range(10):
    diff=eA[k]-eB[k]
    rel=diff/max(abs(eA[k]),1e-8)
    print(f"  {k+1:>4}  {eA[k]:>8.4f}  {eB[k]:>8.4f}  {diff:>8.4f}  {rel:>10.4f}")

t3_ok=(corr_AB>0.9 and mean_abs_diff_AB/max(abs(eA[0]),1e-8)<0.2)
print(f"\n  TEST 3: {'✓ CONFIRMED — spectrum is seed-universal' if t3_ok else '✗ FAILED — spectrum is seed-dependent'}")

# ── Eigenvector comparison ────────────────────────────────────────────────────
print(f"\n  Dominant eigenvector |v1| comparison:")
v1A=rA['v1']; v1B=rB['v1']; v1C=rC['v1']
corr_v1_AB=float(np.corrcoef(v1A,v1B)[0,1])
corr_v1_AC=float(np.corrcoef(v1A,v1C)[0,1])
print(f"  corr(|v1_A|, |v1_B|) = {corr_v1_AB:.4f}")
print(f"  corr(|v1_A|, |v1_C|) = {corr_v1_AC:.4f}")

print(f"\n  |v1| profiles (A, B, C) at each layer:")
print(f"  {'L':>3}  {'|v1_A|':>8}  {'|v1_B|':>8}  {'|v1_C|':>8}  {'agree?'}")
print("  "+"-"*42)
for l in range(N_LAYERS-1):
    layer=l+1
    agree="✓" if abs(v1A[l]-v1B[l])<0.05 else " "
    att="←L14" if layer==14 else ""
    print(f"  L{layer:>2}  {v1A[l]:>8.4f}  {v1B[l]:>8.4f}  {v1C[l]:>8.4f}  {agree}{att}")

# ═══════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  FINAL VERDICT: IS THE GROUP AN ARCHITECTURAL INVARIANT?")
print("="*65)

all_confirmed=t1_ok and t2_ok and t3_ok
print(f"""
  Test 1 (Property T across seeds):      {'✓' if t1_ok else '✗'}
  Test 2 (L14 centrality vs depth):      {'✓' if t2_ok else '✗'}
  Test 3 (Spectral universality):        {'✓' if t3_ok else '✗'}

  Eigenvector corr A vs B: {corr_v1_AB:.4f}
  Eigenvector corr A vs C: {corr_v1_AC:.4f}
""")

if all_confirmed:
    print("""  VERDICT: THE GROUP IS AN ARCHITECTURAL INVARIANT.

  The Property T spectral gap, the L14 attractor centrality,
  and the commutator spectrum are all determined by
  (architecture, data) — not by initialization seed or
  training depth.

  The quiver structure Q=(V,E,M) with its group action
  is universal for this transformer architecture class.
  This is the foundation for the Context Algebra paper:
  the group is not a property of specific weights —
  it is a property of the architecture itself.""")
elif t1_ok and t2_ok:
    print("""  VERDICT: PARTIALLY INVARIANT.

  Property T and L14 centrality are universal.
  The full spectrum varies between seeds.
  The group TYPE is architectural but the specific
  representation is seed-dependent.""")
else:
    print("""  VERDICT: SEED-DEPENDENT.

  The group structure varies with initialization.
  Multiple attractors exist, each with its own group.
  The quiver is not a universal invariant.""")
