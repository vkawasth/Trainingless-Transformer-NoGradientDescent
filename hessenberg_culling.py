#!/usr/bin/env python3
"""
Hessenberg Band Culling Optimizer
===================================
The Toda flow drives matrices toward Hessenberg form.
QR finds this in one shot. Gradient descent finds it in many steps.

The insight: at the Toda fixed point W*, off-diagonal entries decay as
  |W*[i,j]| ~ exp(-|i-j| * t)
So entries with |i-j| > k_threshold are already dead.

CULLING: zero the gradient for dead entries, focus on active band.

k_threshold(W) = band beyond which ||W_band|| < epsilon
  At random init: k_threshold ≈ d  (everything active)
  At convergence: k_threshold ≈ 1  (only diagonal + subdiag active)

The culling mask changes each step as W flows toward Hessenberg.
This IS the adaptive face-size from the lattice picture:
  Big face (k > threshold): skip — already at minimum
  Small face (k <= threshold): full gradient — still converging

THREE CONDITIONS:
  A: Standard SGD
  B: Hessenberg band culling + SGD (the new algorithm)
  C: QR one-shot initialization + SGD fine-tune
     (Schur decompose W_K, set to Hessenberg form, then train)
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=128; N_HEADS=4; N_LAYERS=8; BATCH=8; SEQ=64
LR=0.05; MOMENTUM=0.9; TARGET=4.0; MAX_STEPS=400; LOG=25

print(f"\n{'='*65}")
print(f"  HESSENBERG BAND CULLING OPTIMIZER")
print(f"  Toda flow — cull dead off-diagonal gradient bands")
print(f"  d={D}  layers={N_LAYERS}")
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
        self._nl=nl
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)

def eval_val(model,n=40):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clr(s,total=MAX_STEPS,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ── Hessenberg band analysis ──────────────────────────────────────────────────
def band_energy_profile(W):
    """
    For weight matrix W [d, d]:
    Compute energy in each diagonal band k = 0,1,...,d-1
    E(k) = ||{W[i,j] : |i-j|=k}||^2  (Frobenius norm of band k)
    Returns normalised profile and k_threshold.
    """
    d = W.shape[0]
    W_np = W.detach().numpy() if isinstance(W, torch.Tensor) else W
    energies = np.zeros(d)
    for k in range(d):
        # Band k: entries (i,j) with |i-j|=k
        if k == 0:
            band = np.diag(W_np)
        else:
            band = np.concatenate([np.diag(W_np, k), np.diag(W_np, -k)])
        energies[k] = float(np.sum(band**2))

    total = energies.sum()
    if total < 1e-10: return energies, d
    energies_norm = energies / total

    # k_threshold: smallest k beyond which cumulative energy > 95%
    cumsum = np.cumsum(energies_norm)
    k_thresh = int(np.searchsorted(cumsum, 0.95)) + 1
    return energies_norm, k_thresh

def compute_hessenberg_masks(model, frac=0.95):
    """
    For each weight matrix in each block:
    Compute the culling mask — 1 for active band, 0 for dead band.
    Returns per-block per-param masks and k_thresholds.
    """
    masks = []
    k_thresholds = []
    for blk in model.blocks:
        blk_masks = {}
        blk_k = []
        for name, param in blk.named_parameters():
            if param.data.dim() != 2: continue
            W = param.data
            d_o, d_i = W.shape
            if d_o != d_i:
                # Non-square: use the square sub-block
                d_min = min(d_o, d_i)
                W_sq = W[:d_min, :d_min]
            else:
                W_sq = W

            _, k_thresh = band_energy_profile(W_sq)
            blk_k.append(k_thresh)

            # Build mask same shape as param
            mask = torch.zeros_like(param.data)
            d_min = min(d_o, d_i)
            for i in range(d_o):
                for j in range(d_i):
                    if i < d_min and j < d_min:
                        if abs(i-j) <= k_thresh:
                            mask[i, j] = 1.0
                    else:
                        mask[i, j] = 1.0  # outside square: always active
            blk_masks[name] = mask

        masks.append(blk_masks)
        k_thresholds.append(float(np.mean(blk_k)) if blk_k else float(d_o))
    return masks, k_thresholds

def apply_hessenberg_culling(model, masks):
    """Zero out gradient entries in dead bands."""
    for l, (blk, blk_masks) in enumerate(zip(model.blocks, masks)):
        for name, param in blk.named_parameters():
            if param.grad is None or name not in blk_masks: continue
            param.grad.data *= blk_masks[name]

# ── QR one-shot initialization ────────────────────────────────────────────────
def hessenberg_init(model):
    """
    For each W_K in each layer:
    Compute Hessenberg reduction H = P^T W_K P (orthogonal similarity)
    and set W_K = H.
    This places W_K at the Toda fixed point in one shot.
    """
    with torch.no_grad():
        for blk in model.blocks:
            W = blk.attn.WK.weight.data.numpy()   # [d, d]
            # Hessenberg reduction: H = Q^T W Q  (orthogonal similarity)
            # scipy gives the reduction directly
            from scipy.linalg import hessenberg
            H, Q = hessenberg(W, calc_q=True)
            blk.attn.WK.weight.data = torch.tensor(H, dtype=torch.float32)

# ── Training ──────────────────────────────────────────────────────────────────
def run(name, use_culling=False, use_hess_init=False,
        cull_every=5, seed=42):
    torch.manual_seed(seed)
    model=LM(D,N_HEADS,N_LAYERS); model._nl=N_LAYERS

    if use_hess_init:
        hessenberg_init(model)
        print(f"    Hessenberg init applied to all W_K")

    opt=torch.optim.SGD(model.parameters(),lr=LR,
                         momentum=MOMENTUM,nesterov=True)

    masks=None; stt=None; vals=[]; t0=time.time()
    k_thresh_history=[]

    print(f"\n  [{name}]")
    for step in range(1,MAX_STEPS+1):
        for pg in opt.param_groups: pg['lr']=clr(step)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()

        if use_culling and (step==1 or step%cull_every==0):
            masks, k_thresholds = compute_hessenberg_masks(model)
            k_thresh_history.append(np.mean(k_thresholds))

        if use_culling and masks is not None:
            apply_hessenberg_culling(model, masks)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step%LOG==0 or step==1:
            vl=eval_val(model,n=20); vals.append((step,vl))
            if vl<TARGET and stt is None:
                stt=step; print(f"    *** TARGET at step {step} ***")
            k_info=""
            if k_thresh_history:
                k_info=f"  k_thresh={k_thresh_history[-1]:.1f}"
                # Fraction of gradient retained
                frac_active = sum(k_thresh_history[-1]/D for _ in range(N_LAYERS))/N_LAYERS
                k_info += f"  active={frac_active:.0%}"
            print(f"    {step:>4}/{MAX_STEPS}  val={vl:.4f}"
                  f"  t={time.time()-t0:.0f}s{k_info}")

    fval=eval_val(model,n=100)
    # Final k_threshold profile
    if use_culling:
        _,final_k=compute_hessenberg_masks(model)
        print(f"    Final k_threshold per layer: "
              f"{[f'{k:.0f}' for k in final_k]}")
    return stt,vals,time.time()-t0,fval,k_thresh_history

# ── Band energy profile at init ───────────────────────────────────────────────
print("Band energy profile of W_K at random init:")
torch.manual_seed(42)
probe=LM(D,N_HEADS,N_LAYERS); probe._nl=N_LAYERS
W_probe=probe.blocks[0].attn.WK.weight.data
energies,k0=band_energy_profile(W_probe)
print(f"  k_threshold at init = {k0}  (95% energy in bands 0..{k0})")
print(f"  Band energies (normalised): ", end='')
for k in range(min(10,D)):
    bar='█' if energies[k]>0.05 else ('▌' if energies[k]>0.01 else '·')
    print(bar,end='')
print(f"  (first 10 bands)")
del probe

print(f"\nBand energy profile at Hessenberg form (expected):")
print(f"  Band 0 (diagonal):    concentrated (eigenvalues)")
print(f"  Band 1 (subdiagonal): second most")
print(f"  Band k:               ~ exp(-k) decay")
print(f"  k_threshold at W*:    1-2  (only diagonal + subdiag active)\n")

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"A: Standard SGD (baseline)...")
stt_A,vals_A,t_A,fval_A,_=run("SGD baseline", use_culling=False)

print(f"\nB: SGD + Hessenberg band culling...")
stt_B,vals_B,t_B,fval_B,k_hist_B=run("SGD + Hessenberg culling",
                                       use_culling=True, cull_every=5)

print(f"\nC: Hessenberg init (W_K → Schur form) + SGD culling...")
stt_C,vals_C,t_C,fval_C,k_hist_C=run("Hessenberg init + culling",
                                       use_culling=True, use_hess_init=True,
                                       cull_every=5)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RESULTS")
print("="*65)

def fmt(s): return str(s) if s else f">{MAX_STEPS}"
rows=[
    ("SGD baseline",             stt_A,fval_A,t_A),
    ("SGD + Hessenberg culling", stt_B,fval_B,t_B),
    ("Hessenberg init + culling",stt_C,fval_C,t_C),
]
print(f"\n  {'Method':28}  {'Steps→<4':>9}  {'Final val':>10}  {'Time':>7}  {'vs SGD':>8}")
print("  "+"-"*64)
for name,stt,fval,t in rows:
    base=stt_A or MAX_STEPS
    sp=f"{base/stt:.2f}x" if stt and stt<MAX_STEPS else "—"
    print(f"  {name:28}  {fmt(stt):>9}  {fval:>10.4f}  {t:>6.1f}s  {sp:>8}")

# k_threshold trajectory for B
if k_hist_B:
    print(f"\n  k_threshold trajectory (B — culling):")
    checkpoints=list(range(0,len(k_hist_B),max(1,len(k_hist_B)//8)))
    for i in checkpoints:
        step_approx=(i+1)*5
        print(f"    step~{step_approx:>4}: k={k_hist_B[i]:.1f}  "
              f"({'converging' if i>0 and k_hist_B[i]<k_hist_B[0] else 'init'})")

print(f"\n  Loss curve:")
print(f"  {'step':>5}  {'SGD':>10}  {'SGD+cull':>10}  {'Hess+cull':>10}")
print("  "+"-"*40)
sA={s:v for s,v in vals_A}; sB={s:v for s,v in vals_B}; sC={s:v for s,v in vals_C}
for s in sorted(sA):
    print(f"  {s:>5}  {sA.get(s,0):>10.4f}  "
          f"{sB.get(s,0):>10.4f}  {sC.get(s,0):>10.4f}")

print(f"""
WHAT TO READ:

  k_threshold DECREASING = Hessenberg convergence happening.
  W_K is flowing toward triangular form under gradient descent.
  The culling accelerates this by removing gradient noise in dead bands.

  If SGD+culling faster than SGD:
    The Hessenberg band structure IS the right lattice.
    Toda flow is the underlying dynamics.
    Culling dead bands removes waste at each gradient step.

  If Hessenberg init + culling fastest:
    QR one-shot initialization works.
    The Schur form of W_K at init places weights near the fixed point.
    Gradient descent only needs to fine-tune within the active band.
    This IS "one shot" — one Schur decomposition replaces many gradient steps.

  The k_threshold trajectory is the key diagnostic:
    If it decreases during training: confirmed Toda flow.
    If it stays constant: the weights are not flowing to Hessenberg.
""")
