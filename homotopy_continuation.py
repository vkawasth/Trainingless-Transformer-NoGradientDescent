#!/usr/bin/env python3
"""
Homotopy Continuation for Transformer Training
===============================================
Replaces 167 CE gradient steps with ~10-20 predictor-corrector steps.

THE PATH:
  The gradient trace showed cos(gEmb, ΔEmb) = -0.5 throughout.
  This means the embedding follows a ~120-degree arc from E_0 to E*.
  Gradient descent takes 167 tiny steps along this arc.
  Homotopy continuation takes 10-20 large corrected steps.

THE HOMOTOPY:
  H(θ, t) = ∇_θ L(θ; A(t)) = 0
  where A(t) = (1-t)*A_model(θ) + t*A_uniform  (or A_corpus)

  At t=1: A = A_uniform = 1/VOCAB  (easy, known solution: spectral init)
  At t=0: A = A_model*(θ)           (hard, the target)

  As t decreases 1→0, the system deforms from easy to hard.
  We track the solution θ*(t) along this path.

PREDICTOR-CORRECTOR:
  Predictor (Euler tangent step):
    dθ/dt = -(∂²L/∂θ²)^{-1} * ∂²L/∂θ∂t
    θ̃ = θ - Δt * dθ/dt  [tangent to path]
    Implemented via: one gradient step in the t-deformed landscape

  Corrector (Newton step):
    θ ← θ̃ - (H_θ)^{-1} * ∇L(θ̃; A(t-Δt))
    Implemented via: 3-5 CG steps (HVP, same as LM projection Pass 6)

NO TEACHER REQUIRED:
  The start system at t=1 has solution E_0 = spectral init (from corpus).
  The path is entirely determined by the loss landscape.
  No external weights needed.
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
    def forward(self,h,t_homotopy=0.0):
        """Standard forward pass. t_homotopy used externally via loss_at_t."""
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
    def flat_params(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat(self,f):
        idx=0
        for p in self.parameters():
            n=p.numel(); p.data.copy_(f[idx:idx+n].reshape(p.shape)); idx+=n

def eval_val(m,n=20):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def loss_at_t(model, t, n_batches=8):
    """
    Homotopy loss: L_t(θ) = (1-t)*L_standard(θ) + t*L_uniform(θ)
    
    L_standard = standard CE loss (A = A_model(θ), dense)
    L_uniform  = CE loss as if attention were uniform (A = 1/SEQ)
    
    At t=1: uniform attention — no learning signal from attention structure
            The model predicts purely from embedding + FF
            Solution: E* = spectral embedding (already at init)
    
    At t=0: standard CE — full attention-coupled loss
            Solution: E* = trained embedding
    
    The path from t=1 to t=0 is smooth and follows the 120-degree arc
    seen in the gradient trace.
    
    IMPORTANT: uniform attention is the natural easy system because:
    - At t=1, gradient of attention weights = 0 (uniform attention has no gradient)
    - Only Emb and FF receive gradient signal
    - The start solution is exactly E_0 = spectral init (which already minimizes
      the uniform-attention loss approximately)
    """
    model.train()
    loss_standard = torch.tensor(0.0, requires_grad=False)
    loss_uniform  = torch.tensor(0.0, requires_grad=False)
    
    # Standard loss
    ls_std = []
    for _ in range(n_batches):
        x,y = get_batch()
        logits,l = model(x,y)
        ls_std.append(l)
    loss_standard = torch.stack(ls_std).mean()
    
    if t < 1e-6:
        return loss_standard
    
    # Uniform attention loss: replace attention with uniform
    # h_uniform[s] = (1/S) * sum_s' V[s'] = mean of value vectors
    ls_uni = []
    for _ in range(n_batches):
        x,y = get_batch()
        # Forward with uniform attention
        h = model.te(x) + model.pe(torch.arange(x.shape[1]))
        for block in model.blocks:
            B,S,D_ = h.shape
            # Uniform attention: each position attends equally to all prior positions
            V = block.attn.WV(h)
            # Causal uniform: mean of past + current values
            V_mean = V.cumsum(1) / torch.arange(1, S+1, device=h.device).float().unsqueeze(-1)
            # Apply output projection
            out = block.attn.op(V_mean)
            h = block.attn.ln(h + out)
            h = block.ff(h)
        logits = model.head(model.ln_f(h))
        l = F.cross_entropy(logits.view(-1,VOCAB), y.view(-1))
        ls_uni.append(l)
    loss_uniform = torch.stack(ls_uni).mean()
    
    return (1.0 - t) * loss_standard + t * loss_uniform

def compute_gradient(model, t, n_batches=8):
    """Compute gradient of L_t w.r.t. all parameters."""
    model.zero_grad()
    l = loss_at_t(model, t, n_batches)
    l.backward()
    g = torch.cat([p.grad.flatten() if p.grad is not None 
                   else torch.zeros(p.numel()) 
                   for p in model.parameters()])
    model.zero_grad()
    return g, float(l)

def hvp(model, v, t=0.0, n_batches=4):
    """Hessian-vector product H(θ) @ v via Pearlmutter R-operator."""
    model.zero_grad()
    l = loss_at_t(model, t, n_batches)
    grads = torch.autograd.grad(l, list(model.parameters()), create_graph=True)
    g_flat = torch.cat([g.flatten() for g in grads])
    gv = (g_flat * v.detach()).sum()
    hv = torch.autograd.grad(gv, list(model.parameters()), retain_graph=False)
    model.zero_grad()
    return torch.cat([h.flatten() for h in hv]).detach()

def newton_corrector(model, t, n_cg=5, mu=0.1):
    """
    Newton correction step: solve (H + μI)d = -g, update θ ← θ + d
    Uses CG to avoid storing the full Hessian.
    """
    g, loss_val = compute_gradient(model, t, n_batches=6)
    
    # CG solve: (H + μI)d = -g
    d = torch.zeros_like(g)
    r = -g.clone()
    p_cg = r.clone()
    rr = (r*r).sum()
    
    for _ in range(n_cg):
        Hp = hvp(model, p_cg, t, n_batches=4) + mu * p_cg
        alpha = float(rr) / max(float((p_cg*Hp).sum()), 1e-10)
        d = d + alpha * p_cg
        r = r - alpha * Hp
        rr_new = (r*r).sum()
        beta = float(rr_new) / max(float(rr), 1e-10)
        p_cg = r + beta * p_cg
        rr = rr_new
    
    # Apply Newton step with line search
    theta0 = model.flat_params()
    l0 = loss_val
    
    for step_scale in [1.0, 0.5, 0.25, 0.1]:
        model.set_flat(theta0 + step_scale * d)
        _, l_new = compute_gradient(model, t, n_batches=4)
        if l_new < l0:
            return float(l_new), True
    
    # Revert if no improvement
    model.set_flat(theta0)
    return l0, False

# ── Build spectral embedding ──────────────────────────────────────────────────
print("\n[OFFLINE] Building spectral embedding...")
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

# ── Initialize model ──────────────────────────────────────────────────────────
torch.manual_seed(99)
model_hc = LM(D, N_HEADS, N_STU)
model_hc.te.weight.data.copy_(torch.tensor(E_0))
v_init = eval_val(model_hc)
print(f"  Spectral init val: {v_init:.4f}")

# Verify: at t=1 (uniform attention), gradient is small for attn weights
print("\n[CHECK] Gradient at t=1 (uniform attention):")
g1, l1 = compute_gradient(model_hc, t=1.0, n_batches=8)
# Split by parameter type
idx_start = 0
type_gnorms = {}
for name, param in model_hc.named_parameters():
    n = param.numel()
    g_slice = g1[idx_start:idx_start+n]
    pt = ('WQ' if '.attn.WQ.' in name else 'WK' if '.attn.WK.' in name else
          'WV' if '.attn.WV.' in name else 'WO' if '.attn.op.' in name else
          'Emb' if 'te.weight' in name else 'FF' if '.ff.' in name else 'other')
    type_gnorms[pt] = type_gnorms.get(pt, 0) + float(g_slice.norm()**2)
    idx_start += n
type_gnorms = {k: math.sqrt(v) for k,v in type_gnorms.items()}
for k,v in sorted(type_gnorms.items(), key=lambda x: -x[1]):
    print(f"  |g_{k}| at t=1: {v:.4f}")

print(f"\n  At t=1: loss={l1:.4f}")
print(f"  |gEmb|/|gWK| at t=1: {type_gnorms.get('Emb',0)/max(type_gnorms.get('WK',1e-8),1e-8):.1f}×")
print(f"  If |gWK| ≈ 0 at t=1: uniform attention decouples attn weights ✓")

# ── Homotopy path: t from 1 → 0 ──────────────────────────────────────────────
print("\n" + "="*60)
print("HOMOTOPY CONTINUATION: t = 1 → 0")
print("="*60)
print(f"\n  {'t':>6}  {'val':>7}  {'loss_t':>8}  {'corrector':>10}  {'accepted':>8}")
print("  " + "-"*50)

N_PATH_STEPS = 20        # number of steps from t=1 to t=0
N_CG = 5                 # CG iterations per Newton step
N_CORRECTOR = 3          # Newton corrections per path step
MU = 0.05                # Levenberg-Marquardt damping

t_schedule = np.linspace(1.0, 0.0, N_PATH_STEPS + 1)
val_history = [(1.0, v_init)]

for step_idx, t in enumerate(t_schedule[1:]):
    t_prev = t_schedule[step_idx]
    
    # PREDICTOR: gradient step at current t
    # The predictor follows the tangent: θ̃ = θ - η * ∇L_t(θ)
    # Use a slightly larger step than standard (homotopy allows this)
    eta_pred = LR * 5.0  # 5× larger than standard CE step
    
    model_hc.zero_grad()
    l_t = loss_at_t(model_hc, t, n_batches=6)
    l_t.backward()
    torch.nn.utils.clip_grad_norm_(model_hc.parameters(), 1.0)
    with torch.no_grad():
        for p in model_hc.parameters():
            if p.grad is not None:
                p.data -= eta_pred * p.grad
    model_hc.zero_grad()
    
    # CORRECTOR: Newton steps to snap back to solution manifold
    l_after_corrector = float(l_t)
    n_accepted = 0
    for _ in range(N_CORRECTOR):
        l_new, accepted = newton_corrector(model_hc, t, n_cg=N_CG, mu=MU)
        if accepted:
            n_accepted += 1
            l_after_corrector = l_new
    
    v = eval_val(model_hc, n=10)
    val_history.append((t, v))
    print(f"  t={t:.3f}  val={v:.4f}  loss_t={l_after_corrector:.4f}  "
          f"corrector={n_accepted}/{N_CORRECTOR}  {'✓' if n_accepted > 0 else '~'}")

v_hc = eval_val(model_hc, n=30)
print(f"\n  Homotopy final val: {v_hc:.4f}")

# ── Reference: 167 CE steps ───────────────────────────────────────────────────
print("\n" + "="*60)
print("REFERENCE: 167 CE steps from same init")
print("="*60)
torch.manual_seed(99)
model_ref = LM(D, N_HEADS, N_STU)
model_ref.te.weight.data.copy_(torch.tensor(E_0))
opt = torch.optim.AdamW(model_ref.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
for s in range(1, 168):
    model_ref.train(); x,y=get_batch(); _,l=model_ref(x,y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model_ref.parameters(), 1.0)
    opt.step()
    if s in [20, 50, 100, 167]:
        print(f"  CE {s}: val={eval_val(model_ref,n=8):.4f}")
v_ref = eval_val(model_ref, n=40)

# ── Homotopy + residual CE ────────────────────────────────────────────────────
print("\n[RESIDUAL] HC result + standard CE steps:")
model_hc_ft = copy.deepcopy(model_hc)
opt_ft = torch.optim.AdamW(model_hc_ft.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
for s in range(1, 26):
    model_hc_ft.train(); x,y=get_batch(); _,l=model_hc_ft(x,y)
    opt_ft.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model_hc_ft.parameters(), 1.0)
    opt_ft.step()
    if s in [5, 10, 25]:
        print(f"  HC + CE {s}: val={eval_val(model_hc_ft,n=10):.4f}")
v_hc_ft = eval_val(model_hc_ft, n=30)

print(f"""
{'='*60}
HOMOTOPY CONTINUATION RESULTS
{'='*60}

  Spectral init:               val={v_init:.4f}
  Homotopy ({N_PATH_STEPS} path steps):     val={v_hc:.4f}
  Homotopy + 25 CE:            val={v_hc_ft:.4f}
  Reference (167 CE):          val={v_ref:.4f}

  Path steps used: {N_PATH_STEPS} (each = 1 predictor + {N_CORRECTOR} Newton correctors)
  Total loss evaluations: {N_PATH_STEPS*(1+N_CORRECTOR*N_CG)} (homotopy) vs 167 (CE)

  If homotopy val < spectral init: path tracking works
  If homotopy + 25 CE ≈ reference: homotopy is a better initialization
""")

torch.save(model_hc.state_dict(), '/tmp/model_homotopy.pt')
print("Saved → /tmp/model_homotopy.pt")
