#!/usr/bin/env python3
"""
Adjoint Sensitivity / Shooting Method for 2-Point BVP
======================================================
The 167 CE steps solve a 2-point boundary value problem:
  dθ/dt = -∇L(θ)      forward gradient flow
  θ(0)  = θ_0          known (spectral init)
  ∇L(θ(T)) = 0         known (corpus attractor condition)

Standard approach: integrate forward with 167 small steps.

SHOOTING METHOD (quadratic convergence):
  1. Run N=25 forward steps → approximate θ̃*
  2. Residual: r = ∇L(θ̃*)  (nonzero — 25 steps not fully converged)
  3. Adjoint backward: dλ/dt = -H(θ)ᵀ λ  from t=T to t=0
     Terminal condition: λ(T) = r = ∇L(θ̃*)
     → gives λ(0) = dθ̃*/dθ_0 · r  (sensitivity of residual to init)
  4. Newton update: θ_0 ← θ_0 - α * λ(0)
     Corrects the STARTING POINT to reduce residual at endpoint
  5. Repeat: 3 outer iterations converges quadratically

ADJOINT IMPLEMENTATION:
  The adjoint ODE dλ/dt = -H(θ)ᵀ λ backward in time
  is equivalent to: λ(0) = Π_{t=T}^{0} (I - H(θ_t)ᵀ Δt) λ(T)
  ≈ backward product of HVPs along the forward trajectory.
  
  In practice: store the forward trajectory, then HVP backward.
  Cost: N forward + N backward HVPs = 2N total vs N^2 for finite diff.

EXPECTED RESULT:
  3 outer iterations × 25 steps = 75 total forward steps
  vs 167 Adam steps
  Each outer iteration quadratically reduces the residual ||∇L(θ̃*)||.
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
print(f"VOCAB={VOCAB}")

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

def eval_val(m,n=20):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def get_grad(model, n_batches=8):
    """Compute gradient, return flat vector and loss."""
    model.zero_grad()
    ls=[]
    for _ in range(n_batches):
        x,y=get_batch(); _,l=model(x,y); ls.append(l)
    loss=torch.stack(ls).mean(); loss.backward()
    g=torch.cat([p.grad.flatten() if p.grad is not None
                 else torch.zeros(p.numel())
                 for p in model.parameters()]).detach()
    model.zero_grad()
    return g, float(loss)

def hvp(model, v, n_batches=4):
    """H(θ)ᵀ v via reverse-mode AD."""
    model.zero_grad()
    ls=[]
    for _ in range(n_batches):
        x,y=get_batch(); _,l=model(x,y); ls.append(l)
    loss=torch.stack(ls).mean()
    grads=torch.autograd.grad(loss,list(model.parameters()),create_graph=True)
    gflat=torch.cat([g.flatten() for g in grads])
    gv=(gflat*v.detach()).sum()
    hv=torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)
    model.zero_grad()
    return torch.cat([h.flatten() for h in hv]).detach()

# ── Build spectral init ───────────────────────────────────────────────────────
print("[OFFLINE] Spectral embedding...")
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
idx_s=np.argsort(evals)
evecs=evecs[:,idx_s][:,1:D+1]
scales=1.0/(np.sqrt(evals[idx_s[1:D+1]])+1e-8)
E_0=(evecs*scales[np.newaxis,:]).astype(np.float32)
E_0=(E_0/(E_0.std()+1e-8)*0.02)

# ── Adjoint shooting ──────────────────────────────────────────────────────────
N_FORWARD  = 25   # forward steps per outer iteration
N_OUTER    = 3    # outer shooting iterations
N_ADJOINT  = 10   # backward HVP steps for adjoint
ALPHA      = 0.3  # Newton step size for init correction

print(f"\n{'='*60}")
print(f"ADJOINT SHOOTING: {N_OUTER} outer × {N_FORWARD} forward = "
      f"{N_OUTER*N_FORWARD} total steps")
print(f"{'='*60}")

torch.manual_seed(99)
model=LM(D,N_HEADS,N_STU)
model.te.weight.data.copy_(torch.tensor(E_0))
theta_0_init = model.flat_params().clone()  # keep original init

for outer in range(N_OUTER):
    print(f"\n--- Outer iteration {outer+1}/{N_OUTER} ---")
    v_start=eval_val(model)
    print(f"  Start val: {v_start:.4f}")

    # ── Step 1: Forward integration (N_FORWARD Adam steps) ───────────────────
    trajectory = []  # store (theta_t, grad_t) for adjoint pass
    opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

    for step in range(N_FORWARD):
        # Store state before step
        theta_t = model.flat_params().clone()

        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

        g_t = torch.cat([p.grad.flatten() if p.grad is not None
                         else torch.zeros(p.numel())
                         for p in model.parameters()]).detach()
        trajectory.append((theta_t, g_t))
        opt.step()

    v_end=eval_val(model)
    print(f"  After {N_FORWARD} forward steps: val={v_end:.4f}")

    # ── Step 2: Residual at approximate θ* ────────────────────────────────────
    r, loss_T = get_grad(model, n_batches=12)
    r_norm = float(r.norm())
    print(f"  Residual ||∇L(θ̃*)|| = {r_norm:.4f}  (target: 0)")

    # ── Step 3: Adjoint backward pass ─────────────────────────────────────────
    # dλ/dt = -H(θ)ᵀ λ  backward from t=T to t=0
    # λ(T) = r = ∇L(θ̃*)
    # Discrete: λ_{t-1} = λ_t + H(θ_t)ᵀ λ_t * Δt
    # (backward Euler, Δt absorbed into HVP scaling)
    
    lam = r.clone()  # λ(T) = residual

    # Use a subset of trajectory for efficiency (every k-th step)
    k_skip = max(1, N_FORWARD // N_ADJOINT)
    adj_steps = trajectory[::k_skip][::-1]  # reverse order

    print(f"  Adjoint backward: {len(adj_steps)} HVP steps...")
    for theta_t, g_t in adj_steps:
        model.set_flat(theta_t)
        # H(θ_t)ᵀ λ
        Ht_lam = hvp(model, lam, n_batches=3)
        # Backward Euler: λ_{t-1} = λ_t + H(θ_t)ᵀ λ_t * η
        lam = lam + LR * Ht_lam
        lam_norm = float(lam.norm())
        if lam_norm > 10.0:  # prevent explosion
            lam = lam * (10.0 / lam_norm)

    lam_0_norm = float(lam.norm())
    print(f"  λ(0) norm: {lam_0_norm:.4f}")

    # ── Step 4: Newton update to initial condition ────────────────────────────
    # θ_0 ← θ_0 - α * λ(0)
    # λ(0) approximates dθ̃*/dθ_0 · r = sensitivity of residual
    theta_0_current = model.flat_params()  # currently at θ̃*
    
    # Restore to start of this outer iteration, apply correction
    model.set_flat(theta_0_init if outer == 0 else theta_0_corrected)
    theta_0_current_start = model.flat_params()
    
    # Correction: move init in direction that reduces endpoint residual
    theta_0_new = theta_0_current_start - ALPHA * lam / max(lam_0_norm, 1e-8) * r_norm
    model.set_flat(theta_0_new)
    theta_0_corrected = theta_0_new.clone()

    v_corrected = eval_val(model)
    print(f"  After init correction: val={v_corrected:.4f}")
    print(f"  (correction size: {float((theta_0_new - theta_0_current_start).norm()):.4f})")

# ── Final: run forward from corrected init ────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL: Forward from corrected init")
print(f"{'='*60}")
# Already at corrected init — run remaining steps
opt_final=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for s in range(1, N_FORWARD+1):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt_final.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt_final.step()
    if s in [10, 25]:
        print(f"  Final CE {s}: val={eval_val(model,n=10):.4f}")
v_final = eval_val(model,n=30)

# ── Reference ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("REFERENCE: 167 standard CE steps")
print(f"{'='*60}")
torch.manual_seed(99)
model_ref=LM(D,N_HEADS,N_STU)
model_ref.te.weight.data.copy_(torch.tensor(E_0))
opt_ref=torch.optim.AdamW(model_ref.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for s in range(1,168):
    model_ref.train(); x,y=get_batch(); _,l=model_ref(x,y)
    opt_ref.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model_ref.parameters(),1.0); opt_ref.step()
    if s in [25,75,167]:
        print(f"  CE {s}: val={eval_val(model_ref,n=8):.4f}")
v_ref=eval_val(model_ref,n=40)

total_steps = N_OUTER * N_FORWARD + N_ADJOINT * N_OUTER + N_FORWARD
print(f"""
{'='*60}
ADJOINT SHOOTING RESULTS
{'='*60}
  Outer iterations:    {N_OUTER}
  Forward steps each:  {N_FORWARD}
  Adjoint steps each:  {N_ADJOINT}
  Total forward steps: {N_OUTER*N_FORWARD} (shooting) + {N_FORWARD} (final) = {N_OUTER*N_FORWARD+N_FORWARD}
  Total HVP calls:     {N_ADJOINT*N_OUTER}

  Final val (shooting):  {v_final:.4f}
  Reference (167 CE):    {v_ref:.4f}
  Gap:                   {v_final-v_ref:+.4f}

  Residual reduction per outer iteration demonstrates
  whether quadratic convergence is achieved.
  If ||r|| decreases by ~10× each iteration: success.
""")
