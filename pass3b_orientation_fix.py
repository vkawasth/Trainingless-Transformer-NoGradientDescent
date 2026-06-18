#!/usr/bin/env python3
"""
Pass 3b: Embedding Orientation Correction
==========================================
The gradient trace showed cos(gEmb, ΔEmb) = -0.576 at step 1.
This means E_0 (spectral init) is anti-aligned with E*:
the gradient immediately pushes E AWAY from E_0.

Pass 3 already corrects orientation for W_V and W_O (sign flip).
This pass extends that logic to the embedding matrix.

THE FIX:
  1. Compute g_0 = ∇_E L(E_0)  at the spectral init
  2. Measure cos(g_0, E_0)
  3. If anti-aligned: apply Householder reflection to remove
     the anti-aligned component of E_0

     E_0_fixed = E_0 - 2 * (E_0 · ĝ_0) * ĝ_0

     This is a reflection through the hyperplane perpendicular
     to g_0 — the SAME operation as a sign flip but for a
     general direction, not just ±1.

WHY THIS WORKS:
  After the fix, cos(g, E_0_fixed) ≈ 0.
  The gradient is now orthogonal to the embedding.
  Pass 7 starts in the correct orientation basin.
  Expected: 25 CE steps instead of 167.

Fits between Pass 3 (sign correction) and Pass 4 (MF pumping).
Cost: one forward-backward pass + O(VOCAB×D) reflection.
No teacher. No external weights. Pure corpus + architecture.
"""
import json, math, warnings, collections, os, copy, sys
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f):
        print(f"ERROR: {f} missing."); sys.exit(1)

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)
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

def eval_val(m,n=20):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def measure_orientation(model, n_batches=20):
    """
    Measure cos(∇_E L, E) at current parameters.
    Returns (cos_value, gradient_norm, embedding_norm)
    """
    model.zero_grad()
    losses = []
    for _ in range(n_batches):
        x,y = get_batch()
        _,l = model(x,y)
        losses.append(l)
    torch.stack(losses).mean().backward()

    g_E = model.te.weight.grad.detach()   # [VOCAB, D]
    E   = model.te.weight.data.detach()   # [VOCAB, D]

    g_flat = g_E.flatten()
    E_flat = E.flatten()
    cos = float((g_flat * E_flat).sum() /
                (g_flat.norm() * E_flat.norm() + 1e-10))
    model.zero_grad()
    return cos, float(g_E.norm()), float(E.norm()), g_E

# ── Build spectral embedding ──────────────────────────────────────────────────
print("\n[OFFLINE] Spectral embedding...")
bigram=collections.Counter()
for i in range(len(train_ids)-1):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB: bigram[(a,b)]+=1
rows,cols,vals=[],[],[]
for (a,b),cnt in bigram.items():
    rows.append(a); cols.append(b); vals.append(float(cnt))
W_sp=sp.csr_matrix((vals,(rows,cols)),shape=(VOCAB,VOCAB),dtype=np.float32)
W_sp=W_sp+W_sp.T
d_inv=np.array(1.0/(W_sp.sum(1)+1e-8)).flatten()
Dsi=sp.diags(np.sqrt(d_inv))
L_sym=sp.eye(VOCAB)-Dsi@W_sp@Dsi
evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)
idx=np.argsort(evals)
evecs=evecs[:,idx][:,1:D+1]
scales=1.0/(np.sqrt(evals[idx[1:D+1]])+1e-8)
E_0=(evecs*scales[np.newaxis,:]).astype(np.float32)
E_0=(E_0/(E_0.std()+1e-8)*0.02)

torch.manual_seed(99)
model=LM(D,N_HEADS,N_STU)
model.te.weight.data.copy_(torch.tensor(E_0))

# Pass 3: sign correction for WV, WO (existing)
with torch.no_grad():
    for l in [1,2]:
        model.blocks[l].attn.WV.weight.mul_(-1)
        model.blocks[l].attn.op.weight.mul_(-1)

v_before = eval_val(model)
print(f"After Pass 3 (sign correction): val={v_before:.4f}")

# ── Pass 3b: Measure embedding orientation ────────────────────────────────────
print("\n" + "="*60)
print("PASS 3b: EMBEDDING ORIENTATION MEASUREMENT")
print("="*60)

cos_E, g_norm, E_norm, g_E = measure_orientation(model, n_batches=30)
print(f"\n  cos(∇_E L, E) = {cos_E:.4f}")
print(f"  |∇_E L| = {g_norm:.4f}")
print(f"  |E|     = {E_norm:.4f}")
print(f"  Angle   = {math.degrees(math.acos(max(-1,min(1,cos_E)))):.1f}°")

if cos_E < -0.3:
    print(f"\n  ✓ ANTI-ALIGNED (cos={cos_E:.3f} < -0.3)")
    print(f"  E_0 has large anti-aligned component → apply Householder reflection")
elif cos_E > 0.3:
    print(f"\n  ⚠ ALIGNED (cos={cos_E:.3f} > 0.3)")
    print(f"  Gradient pushes E toward current direction — no flip needed")
else:
    print(f"\n  ~ ORTHOGONAL (cos={cos_E:.3f})")
    print(f"  Gradient orthogonal to E — standard case, minor correction")

# ── Pass 3b: Householder reflection ──────────────────────────────────────────
print("\n" + "="*60)
print("PASS 3b: HOUSEHOLDER REFLECTION")
print("="*60)

# Reflect E_0 through the hyperplane perpendicular to g_E
# E_fixed = E_0 - 2*(E_0·ĝ)*(ĝ)
# where ĝ = g_E / ||g_E||

with torch.no_grad():
    g_hat = g_E / (g_E.norm() + 1e-10)  # [VOCAB, D] normalised gradient
    E_curr = model.te.weight.data.clone()

    # Component of E along g_hat
    proj = (E_curr * g_hat).sum()  # scalar dot product of flattened vectors
    # Householder: E_fixed = E - 2*proj*g_hat
    E_fixed = E_curr - 2.0 * proj * g_hat

    # Preserve original norm
    scale = E_curr.norm() / max(E_fixed.norm(), 1e-8)
    E_fixed = E_fixed * scale

model_fixed = copy.deepcopy(model)
with torch.no_grad():
    model_fixed.te.weight.data.copy_(E_fixed)

# Measure orientation after fix
cos_fixed, g_norm_fixed, E_norm_fixed, _ = measure_orientation(model_fixed, n_batches=30)
v_fixed = eval_val(model_fixed)

print(f"\n  Before reflection: cos(g,E) = {cos_E:.4f}  val={v_before:.4f}")
print(f"  After  reflection: cos(g,E) = {cos_fixed:.4f}  val={v_fixed:.4f}")
print(f"  Orientation corrected: {abs(cos_fixed) < abs(cos_E)}")

# ── Test: does orientation fix accelerate Pass 7? ────────────────────────────
print("\n" + "="*60)
print("TEST: 25 CE steps with and without orientation fix")
print("="*60)

def run_ce(m, steps):
    mc = copy.deepcopy(m)
    opt = torch.optim.AdamW(mc.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
    checkpoints = {}
    for s in range(1, steps+1):
        mc.train(); x,y=get_batch(); _,l=mc(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(),1.0); opt.step()
        if s in [5,10,25,50,100,167]:
            v=eval_val(mc,n=8)
            checkpoints[s]=v
            print(f"    CE {s}: val={v:.4f}")
    return eval_val(mc,n=20), checkpoints

print("\n  [A] Without orientation fix:")
v_A, ckpts_A = run_ce(model, 167)

print(f"\n  [B] With orientation fix (Pass 3b):")
v_B, ckpts_B = run_ce(model_fixed, 167)

print(f"""
{'='*60}
ORIENTATION FIX RESULTS
{'='*60}

  cos(g,E) before fix:  {cos_E:.4f}  ({math.degrees(math.acos(max(-1,min(1,cos_E)))):.0f}°)
  cos(g,E) after fix:   {cos_fixed:.4f}  ({math.degrees(math.acos(max(-1,min(1,cos_fixed)))):.0f}°)

  Without fix (167 CE):  val={v_A:.4f}
  With fix    (167 CE):  val={v_B:.4f}

  At 25 CE steps:
    Without fix: val={ckpts_A.get(25, 'N/A')}
    With fix:    val={ckpts_B.get(25, 'N/A')}

  At 50 CE steps:
    Without fix: val={ckpts_A.get(50, 'N/A')}
    With fix:    val={ckpts_B.get(50, 'N/A')}

  If [B @ 25 CE] ≈ [A @ 167 CE]: orientation fix saves ~142 CE steps
  The fix costs: 1 forward-backward pass + O(VOCAB×D) reflection
""")
