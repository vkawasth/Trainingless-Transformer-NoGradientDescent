#!/usr/bin/env python3
"""
A2 Serre Relation Check
========================
For the Lie algebra of type A2 (sl_3), the Serre relations are:

  [e_i, f_j] = delta_ij * h_i           (Chevalley basis)
  [h_i, e_j] = a_ij * e_j               (Cartan action)
  [h_i, f_j] = -a_ij * f_j
  ad(e_i)^{1-a_ij}(e_j) = 0            (Serre: nilpotency of adE)
  ad(f_i)^{1-a_ij}(f_j) = 0            (Serre: nilpotency of adF)

For A2, the Cartan matrix is:
  a_ij = [[2,-1],[-1,2]]

So the Serre relations become:
  ad(e_1)^2(e_2) = [e_1,[e_1,e_2]] = 0
  ad(e_2)^2(e_1) = [e_2,[e_2,e_1]] = 0
  ad(f_1)^2(f_2) = [f_1,[f_1,f_2]] = 0
  ad(f_2)^2(f_1) = [f_2,[f_2,f_1]] = 0

In our setting:
  The simple roots correspond to the two dominant directions
  in the commutator Lie algebra.
  
  e_1 ↔ J_l (Jacobian at attractor layer)
  e_2 ↔ J_{l+1} (adjacent layer)
  f_i = J_i^T (adjoint)

We test: ||[J_l, [J_l, J_{l+1}]]|| / ||[J_l, J_{l+1}]||
         (should be ~0 for A2, substantial for larger algebra)

Also test B2, G2, A3 for comparison.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4; PROJ=48

print(f"\n{'='*65}")
print(f"  A2 SERRE RELATION CHECK")
print(f"  Testing [J_l,[J_l,J_{{l+1}}]] ≈ 0 (A2 prediction)")
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
print("Training (seed=42, 300 steps)...")
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

# ── Extract Jacobians (multiple reference inputs for robustness) ──────────────
print("Extracting Jacobians (5 reference inputs for robustness)...", flush=True)
torch.manual_seed(0)
pos=SEQ//2; m=min(PROJ,SEQ,D)

all_Js=[]  # [n_refs, N_LAYERS, m, m]
for ref_idx in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad():
        hs=model.hidden_states(x_ref); hs=[h[0] for h in hs]
    Js=[]
    for l in range(N_LAYERS):
        J,U,m_=layer_jac(model.blocks[l],hs[l],pos,m)
        Js.append(J)
    all_Js.append(Js)
    print(f"  ref {ref_idx+1}/5...", flush=True)

# Average Jacobians across references for stability
Js_mean=[]
for l in range(N_LAYERS):
    J_avg=np.mean([all_Js[r][l] for r in range(5)], axis=0)
    Js_mean.append(J_avg)
print(f"  Done. m={m_}\n")

# ── Serre relation tests ──────────────────────────────────────────────────────
def comm(A,B): return A@B - B@A
def ad(A,B,k=1):
    """ad(A)^k(B) = [A,[A,...[A,B]...]] k times"""
    result=B
    for _ in range(k): result=comm(A,result)
    return result

def serre_residual(e1, e2, k):
    """
    Serre relation: ad(e1)^k(e2) = 0
    For A2: k = 1-a_12 = 2 (off-diagonal Cartan entry = -1)
    For B2: one pair has k=3, other k=2
    For G2: one pair has k=4, other k=2
    Returns ||ad(e1)^k(e2)|| / ||[e1,e2]|| (normalized residual)
    """
    comm_12=comm(e1,e2)
    norm_comm=float(np.linalg.norm(comm_12))
    if norm_comm < 1e-10: return float('nan')
    serre=ad(e1,e2,k)
    return float(np.linalg.norm(serre)/norm_comm)

print(f"{'='*65}")
print(f"  SERRE RELATION RESIDUALS")
print(f"  Normalized: ||ad(e1)^k(e2)|| / ||[e1,e2]||")
print(f"  A2 predicts k=2 residual ≈ 0")
print(f"  B2 predicts k=3 residual ≈ 0 (for long root)")
print(f"  G2 predicts k=4 residual ≈ 0 (for long root)")
print("="*65)

# Test all consecutive layer pairs
print(f"\n  Per-layer A2 Serre residuals ||[J_l,[J_l,J_{{l+1}}]]||/||[J_l,J_{{l+1}}]||:")
print(f"  {'L→L+1':>8}  {'A2(k=2)':>10}  {'A2(k=2)rev':>12}  "
      f"{'B2(k=3)':>10}  {'G2(k=4)':>10}  {'baseline||comm||':>18}")
print("  "+"-"*75)

a2_residuals=[]; a2_rev_residuals=[]
b2_residuals=[]; g2_residuals=[]
comm_norms=[]

for l in range(1, N_LAYERS-1):
    e1=Js_mean[l]; e2=Js_mean[l+1]
    f1=e1.T; f2=e2.T  # adjoints

    # A2: ad(e1)^2(e2)=0 and ad(e2)^2(e1)=0
    a2_fwd=serre_residual(e1,e2,2)
    a2_rev=serre_residual(e2,e1,2)

    # B2: ad(long)^3(short)=0 and ad(short)^2(long)=0
    b2_fwd=serre_residual(e1,e2,3)

    # G2: ad(long)^4(short)=0
    g2_fwd=serre_residual(e1,e2,4)

    comm_norm=float(np.linalg.norm(comm(e1,e2)))

    a2_residuals.append(a2_fwd)
    a2_rev_residuals.append(a2_rev)
    b2_residuals.append(b2_fwd)
    g2_residuals.append(g2_fwd)
    comm_norms.append(comm_norm)

    att_marker=" ←L14" if l==14 else ""
    print(f"  L{l:>2}→L{l+1:<2}  {a2_fwd:>10.4f}  {a2_rev:>12.4f}  "
          f"{b2_fwd:>10.4f}  {g2_fwd:>10.4f}  {comm_norm:>18.4f}{att_marker}")

# Summary statistics
a2_mean=float(np.nanmean(a2_residuals))
a2_rev_mean=float(np.nanmean(a2_rev_residuals))
b2_mean=float(np.nanmean(b2_residuals))
g2_mean=float(np.nanmean(g2_residuals))

print(f"\n  Summary (mean across L1..L22):")
print(f"  A2 k=2 fwd:  {a2_mean:.4f}")
print(f"  A2 k=2 rev:  {a2_rev_mean:.4f}")
print(f"  B2 k=3:      {b2_mean:.4f}")
print(f"  G2 k=4:      {g2_mean:.4f}")

# ── Attractor neighborhood specifically ───────────────────────────────────────
print(f"\n  Serre residuals in attractor neighborhood (L11-L17):")
att_a2=[a2_residuals[l-2] for l in range(11,18) if l-2 < len(a2_residuals)]
att_b2=[b2_residuals[l-2] for l in range(11,18) if l-2 < len(b2_residuals)]
print(f"  A2 mean in L11-L17: {float(np.nanmean(att_a2)):.4f}")
print(f"  B2 mean in L11-L17: {float(np.nanmean(att_b2)):.4f}")

# ── Random baseline ───────────────────────────────────────────────────────────
print(f"\n  Random matrix baseline (what to expect from random Jacobians):")
np.random.seed(42)
rand_a2=[]
for _ in range(20):
    R1=np.random.randn(m_,m_)*0.3
    R2=np.random.randn(m_,m_)*0.3
    rand_a2.append(serre_residual(R1,R2,2))
print(f"  Random A2 k=2 residual: {float(np.nanmean(rand_a2)):.4f} ± {float(np.nanstd(rand_a2)):.4f}")
print(f"  (This is the baseline — transformer should be lower if A2 structure is real)")

# ── Decision ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  DECISION")
print("="*65)

random_baseline=float(np.nanmean(rand_a2))

print(f"""
  A2 Serre residual (mean): {a2_mean:.4f}
  Random baseline:          {random_baseline:.4f}
  Relative to random:       {a2_mean/random_baseline:.3f}x
""")

if a2_mean < 0.05:
    verdict="A2 CONFIRMED — SL3 identification strongly supported"
    detail="Serre relations hold to near-machine precision. The Lie algebra is type A2."
elif a2_mean < 0.20:
    verdict="A2 PLAUSIBLE — partial Serre satisfaction"
    detail="Serre residuals are below random but not near zero. Consistent with A2 but not confirmed."
elif a2_mean < random_baseline * 0.5:
    verdict="BELOW RANDOM — some structure, not A2"
    detail=f"Residuals ({a2_mean:.3f}) below random ({random_baseline:.3f}) but not A2. Larger algebra type."
else:
    verdict="A2 RULED OUT — residuals at random level"
    detail=f"Residuals ({a2_mean:.3f}) ≈ random ({random_baseline:.3f}). No A2 structure. General algebra."

print(f"  VERDICT: {verdict}")
print(f"  {detail}")

print(f"""
  ALGEBRA TYPE SUMMARY:
  A2 (SL3): k=2 residual ≈ 0  → {a2_mean:.4f}  {'✓' if a2_mean<0.05 else ('~' if a2_mean<0.2 else '✗')}
  B2 (Sp4): k=3 residual ≈ 0  → {b2_mean:.4f}  {'✓' if b2_mean<0.05 else ('~' if b2_mean<0.2 else '✗')}
  G2:       k=4 residual ≈ 0  → {g2_mean:.4f}  {'✓' if g2_mean<0.05 else ('~' if g2_mean<0.2 else '✗')}
  Random baseline:             → {random_baseline:.4f}

  The algebra with smallest residual relative to its k value
  is the most likely candidate.
""")
