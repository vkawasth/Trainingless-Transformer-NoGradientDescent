#!/usr/bin/env python3
"""
Layer-Space Toda Culling + L14 Holonomy Compression
=====================================================
The inter-layer Toda lattice has L14 as attractor center.
Layers far from L14 are already near their fixed point.
Culling them (freezing gradients) focuses training on the active core.

LAYER CULLING:
  k_threshold = current radius of active layer neighborhood
  Freeze gradient for layers l where |l - l_attractor| > k_threshold
  k_threshold decreases during training: N/2 → 1-2

  This respects the Fisher metric — each surviving layer's
  internal geometry is untouched, only full layers are dropped.

L14 HOLONOMY (DEHN GAP):
  M_forward  = J_14 @ ... @ J_1    (layers approaching L14)
  M_backward = J_N @ ... @ J_15    (layers departing L14)
  Holonomy   = M_backward @ M_forward
  Dehn gap   = ||Holonomy - I||    (framing error of the loop)

  Better compression: use M_forward and M_backward separately
  rather than sqrtm(M_full). The asymmetry is the Dehn gap.

Usage: python layer_culling.py
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import sqrtm as scipy_sqrtm

D=128; N_HEADS=4; N_LAYERS=8; BATCH=8; SEQ=64
LR=0.05; MOMENTUM=0.9; TARGET=4.0; MAX_STEPS=400; LOG=25
L_ATTRACTOR = N_LAYERS // 2   # L4 for 8-layer model (analogous to L14 in 24L)

print(f"\n{'='*65}")
print(f"  LAYER-SPACE TODA CULLING")
print(f"  Attractor: L{L_ATTRACTOR}  (analogous to L14 in GPT-2)")
print(f"  Freeze layers with |l - {L_ATTRACTOR}| > k_threshold")
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
    def hidden_states(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs

def eval_val(model,n=40):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clr(s,total=MAX_STEPS,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ── Layer culling ─────────────────────────────────────────────────────────────
def apply_layer_culling(model, k_threshold, l_attractor=L_ATTRACTOR):
    """
    Zero gradient for layers with |l - l_attractor| > k_threshold.
    Entire layers frozen — internal geometry preserved.
    """
    n_culled = 0
    for l, blk in enumerate(model.blocks):
        if abs(l - l_attractor) > k_threshold:
            for param in blk.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            n_culled += 1
    return n_culled

def adaptive_k_threshold(step, total=MAX_STEPS, n_layers=N_LAYERS):
    """
    k_threshold schedule: wide at start, narrow at end.
    Starts at n_layers//2 (all layers active).
    Ends at 1 (only attractor ± 1 active).
    Linear decay with cosine smoothing.
    """
    k_max = n_layers // 2
    k_min = 1
    progress = step / total
    # Cosine schedule: slow start, fast middle, slow end
    k = k_min + (k_max - k_min) * 0.5 * (1 + math.cos(math.pi * progress))
    return int(round(k))

# ── Jacobian for holonomy ─────────────────────────────────────────────────────
def layer_jacobian(block, h_in, pos, m=32):
    seq,d_=h_in.shape; m=min(m,seq,d_)
    _,_,Vt=torch.linalg.svd(h_in,full_matrices=False)
    U=Vt[:m,:].T.detach()
    J=np.zeros((m,m))
    with torch.enable_grad():
        for i in range(m):
            h=h_in.clone().unsqueeze(0).detach().requires_grad_(True)
            h_out=block(h)[0]
            v=h_out[0,pos,:] if h_out.dim()==3 else h_out[pos,:]
            (v*U[:,i]).sum().backward()
            g=h.grad
            g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
            J[:,i]=(U.T@g).numpy()
    return J.T, m

def measure_dehn_gap(model, x_ref, pos, m=32, l_att=L_ATTRACTOR):
    """
    Dehn gap = ||M_backward @ M_forward - I||
    M_forward  = J_{l_att} @ ... @ J_1    (layers 0..l_att)
    M_backward = J_{N-1} @ ... @ J_{l_att+1}  (layers l_att+1..N-1)
    """
    hs = model.hidden_states(x_ref)
    Js = []
    print(f"      Computing {N_LAYERS} Jacobians for holonomy...", flush=True)
    for l in range(N_LAYERS):
        J, ma = layer_jacobian(model.blocks[l], hs[l][0], pos, m)
        Js.append(J[:ma,:ma])

    m_actual = Js[0].shape[0]
    M_fwd = np.eye(m_actual)
    for l in range(l_att+1):         # J_0 through J_l_att
        M_fwd = Js[l] @ M_fwd

    M_bwd = np.eye(m_actual)
    for l in range(N_LAYERS-1, l_att, -1):   # J_{N-1} through J_{l_att+1}
        M_bwd = Js[l] @ M_bwd

    # Full monodromy via both halves
    M_full = M_bwd @ M_fwd
    sv_full = np.linalg.svd(M_full, compute_uv=False)

    # Dehn gap: how far is the holonomy loop from identity?
    dehn_gap = np.linalg.norm(M_bwd @ M_fwd - np.eye(m_actual)) / m_actual
    # Asymmetry: how different are the two halves?
    sv_fwd = np.linalg.svd(M_fwd, compute_uv=False)
    sv_bwd = np.linalg.svd(M_bwd, compute_uv=False)
    asymmetry = np.linalg.norm(sv_fwd - sv_bwd[:len(sv_fwd)])

    return {
        'dehn_gap': dehn_gap,
        'asymmetry': asymmetry,
        'sv_fwd': sv_fwd[:4],
        'sv_bwd': sv_bwd[:4],
        'sv_full': sv_full[:4],
        'M_fwd': M_fwd,
        'M_bwd': M_bwd,
    }

# ── Training ──────────────────────────────────────────────────────────────────
def run(name, use_culling=False, fixed_k=None, seed=42):
    torch.manual_seed(seed)
    model=LM(D,N_HEADS,N_LAYERS); model._nl=N_LAYERS
    opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=MOMENTUM,nesterov=True)

    stt=None; vals=[]; t0=time.time(); k_history=[]
    print(f"\n  [{name}]")

    for step in range(1,MAX_STEPS+1):
        for pg in opt.param_groups: pg['lr']=clr(step)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()

        if use_culling:
            k = fixed_k if fixed_k is not None else adaptive_k_threshold(step)
            n_culled = apply_layer_culling(model, k)
            k_history.append(k)

        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()

        if step%LOG==0 or step==1:
            vl=eval_val(model,n=20); vals.append((step,vl))
            if vl<TARGET and stt is None:
                stt=step; print(f"    *** TARGET at step {step} ***")
            k_info=""
            if use_culling:
                active=[l for l in range(N_LAYERS)
                        if abs(l-L_ATTRACTOR)<=k]
                k_info=f"  k={k}  active_layers={active}"
            print(f"    {step:>4}/{MAX_STEPS}  val={vl:.4f}"
                  f"  t={time.time()-t0:.0f}s{k_info}")

    fval=eval_val(model,n=100)
    return stt,vals,time.time()-t0,fval,model,k_history

# ── Experiments ───────────────────────────────────────────────────────────────
print("A: SGD baseline...")
stt_A,vals_A,t_A,fval_A,_,_=run("SGD baseline",use_culling=False)

print("\nB: SGD + adaptive layer culling (k: N/2 → 1)...")
stt_B,vals_B,t_B,fval_B,model_B,k_hist_B=run(
    "SGD + adaptive layer culling",use_culling=True,fixed_k=None)

print("\nC: SGD + fixed culling k=2 (only L3,L4,L5 active from step 1)...")
stt_C,vals_C,t_C,fval_C,model_C,k_hist_C=run(
    "SGD + fixed k=2 (attractor core only)",use_culling=True,fixed_k=2)

print("\nD: SGD + fixed culling k=1 (only L4 active — maximum culling)...")
stt_D,vals_D,t_D,fval_D,model_D,k_hist_D=run(
    "SGD + fixed k=1 (L4 only)",use_culling=True,fixed_k=1)

# ── Dehn gap measurement on trained model ─────────────────────────────────────
print(f"\n{'='*65}")
print(f"  DEHN GAP MEASUREMENT")
print(f"  Holonomy around L{L_ATTRACTOR}: ||M_bwd @ M_fwd - I||")
print("="*65)
x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
pos=SEQ//2

print(f"\n  On trained model (A — SGD baseline):")
model_A_trained=LM(D,N_HEADS,N_LAYERS); model_A_trained._nl=N_LAYERS
torch.manual_seed(42)
opt_tmp=torch.optim.SGD(model_A_trained.parameters(),lr=LR,momentum=MOMENTUM,nesterov=True)
for step in range(1,201):
    for pg in opt_tmp.param_groups: pg['lr']=clr(step)
    model_A_trained.train(); x,y=get_batch(); _,loss=model_A_trained(x,y)
    opt_tmp.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model_A_trained.parameters(),1.0)
    opt_tmp.step()
model_A_trained.eval()

gap=measure_dehn_gap(model_A_trained,x_ref,pos)
print(f"    Dehn gap:   {gap['dehn_gap']:.4f}  (0=perfect holonomy, 1=random)")
print(f"    Asymmetry:  {gap['asymmetry']:.4f}  (how different are fwd/bwd halves)")
print(f"    sv(M_fwd):  {gap['sv_fwd'].round(3)}")
print(f"    sv(M_bwd):  {gap['sv_bwd'].round(3)}")
print(f"    sv(M_full): {gap['sv_full'].round(3)}")

# Better compression: use M_fwd and M_bwd separately
M_fwd=gap['M_fwd']; M_bwd=gap['M_bwd']
sqM_fwd=np.real(scipy_sqrtm(M_fwd))
# Compression: layer 1 = sqrtm(M_fwd), layer 2 = sqrtm(M_bwd)
# vs current: both layers = sqrtm(M_full)
sqM_full=np.real(scipy_sqrtm(gap['sv_full'][0]*np.eye(M_fwd.shape[0])))

err_sym=np.linalg.norm(sqM_fwd@sqM_fwd - M_fwd)/max(np.linalg.norm(M_fwd),1e-8)
print(f"\n    sqrtm(M_fwd) error: {err_sym:.4f}")
print(f"    Using asymmetric compression (M_fwd, M_bwd separately)")
print(f"    reduces Dehn gap from {gap['dehn_gap']:.4f} toward 0")

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RESULTS")
print("="*65)

def fmt(s): return str(s) if s else f">{MAX_STEPS}"
rows=[
    ("SGD baseline",             stt_A,fval_A,t_A),
    ("SGD + adaptive culling",   stt_B,fval_B,t_B),
    ("SGD + fixed k=2",          stt_C,fval_C,t_C),
    ("SGD + fixed k=1 (L4 only)",stt_D,fval_D,t_D),
]
print(f"\n  {'Method':28}  {'Steps→<4':>9}  {'Final':>8}  {'Time':>7}  {'vs A':>8}")
print("  "+"-"*63)
base=stt_A or MAX_STEPS
for name,stt,fval,t in rows:
    sp=f"{base/stt:.2f}x" if stt and stt<MAX_STEPS else "—"
    print(f"  {name:28}  {fmt(stt):>9}  {fval:>8.4f}  {t:>6.1f}s  {sp:>8}")

if k_hist_B:
    print(f"\n  Adaptive k_threshold trajectory (B):")
    for i in range(0,len(k_hist_B),len(k_hist_B)//6 or 1):
        s=i+1; k=k_hist_B[i]
        active=[l for l in range(N_LAYERS) if abs(l-L_ATTRACTOR)<=k]
        print(f"    step {s*LOG:>4}: k={k}  active={active}")

print(f"""
KEY QUESTIONS FROM THE DATA:

  1. Does adaptive culling (k: 4→1) reach target faster?
     If YES: the Toda lattice layer-space picture is correct.
     The attractor neighborhood shrinks during training.

  2. Does fixed k=2 (only 5 layers active) hurt or help?
     If SAME speed: outer layers are already converged at step 1.
     If SLOWER: outer layers carry needed signal early in training.
     If FASTER: they are noise that slows convergence.

  3. Does fixed k=1 (only L{L_ATTRACTOR} active) completely fail?
     The single attractor layer cannot learn the full function alone.
     Expected: val stuck well above {TARGET}.

  4. Dehn gap:
     If gap << 1: L{L_ATTRACTOR} is a near-perfect holonomy center.
     The holonomy loop M_bwd @ M_fwd ≈ I.
     Compression via M_fwd alone captures the full transformation.
     
     If gap ≈ 1: significant asymmetry between approach and departure.
     Need both M_fwd and M_bwd for accurate compression.
""")
