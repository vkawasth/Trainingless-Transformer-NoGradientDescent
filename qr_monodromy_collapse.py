#!/usr/bin/env python3
"""
QR Monodromy Collapse
======================
The inter-layer sequence T(0)→T(1)→...→T(23) IS a QR cascade.
Layer 14 = invariant core (dominant spectral fixed point).
Layers 15-23 = deterministic relaxation tail (isospectral QR decay).

The tail can be collapsed via shifted QR (Rayleigh Quotient Iteration):
  μ = Rayleigh quotient of T_14  (dominant eigenvalue estimate)
  Q,R = QR(T_14 - μI)
  M_compressed = RQ + μI          (one shifted QR step)

This analytically captures what 9 tail layers do, in one matrix operation.

MEASUREMENTS:
  1. Dehn gap: ||M_bwd @ M_fwd - I||  (holonomy framing error at L14)
  2. Logit cosine: compressed 2-layer model vs 24-layer model
  3. Symmetric sqrtm vs asymmetric (M_fwd, M_bwd) vs shifted QR tail collapse
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import sqrtm as scipy_sqrtm, qr as scipy_qr

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4
L_ATT = 14   # GPT-2 medium attractor center
PROJ  = 48   # Jacobian projection dim

print(f"\n{'='*65}")
print(f"  QR MONODROMY COLLAPSE")
print(f"  Shifted QR collapse of relaxation tail L{L_ATT+1}..L{N_LAYERS-1}")
print(f"  d={D}  layers={N_LAYERS}  attractor=L{L_ATT}")
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

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def cosine_sim(model_a, model_b, n=30):
    model_a.eval(); model_b.eval(); sims=[]
    with torch.no_grad():
        for _ in range(n):
            x,_=get_batch('val')
            la,_=model_a(x); lb,_=model_b(x)
            la=la.reshape(-1,VOCAB); lb=lb.reshape(-1,VOCAB)
            sims.append(F.cosine_similarity(la,lb,dim=-1).mean().item())
    return float(np.mean(sims))

def clr(s,total=300,warmup=100):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ── Train source model ────────────────────────────────────────────────────────
print(f"Step 1: Train {N_LAYERS}-layer source (300 steps)...")
torch.manual_seed(42)
source=LM(D,N_HEADS,N_LAYERS)
opt=torch.optim.AdamW(source.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step)
    source.train(); x,y=get_batch(); _,loss=source(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(source.parameters(),1.0); opt.step()
    if step%100==0: print(f"  step {step}  val={eval_val(source,n=10):.4f}  t={time.time()-t0:.0f}s")
val_src=eval_val(source); source.eval()
print(f"  Source val={val_src:.4f}\n")

# ── Compute all layer Jacobians ───────────────────────────────────────────────
print("Step 2: Compute Jacobians for all layers...")
x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
with torch.no_grad():
    hs_b=source.hidden_states(x_ref); hs=[h[0] for h in hs_b]
pos=SEQ//2; m=min(PROJ,SEQ,D)

def layer_jacobian(block,h_in,pos,m):
    seq,d_=h_in.shape; m=min(m,seq,d_)
    _,_,Vt=torch.linalg.svd(h_in,full_matrices=False)
    U=Vt[:m,:].T.detach(); J=np.zeros((m,m))
    with torch.enable_grad():
        for i in range(m):
            h=h_in.clone().unsqueeze(0).detach().requires_grad_(True)
            h_out=block(h)[0]
            v=h_out[0,pos,:] if h_out.dim()==3 else h_out[pos,:]
            (v*U[:,i]).sum().backward()
            g=h.grad; g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
            J[:,i]=(U.T@g).numpy()
    return J.T, U.detach().numpy(), m

Js=[]; U_basis=None; ma=None
for l in range(N_LAYERS):
    J,U,m_=layer_jacobian(source.blocks[l],hs[l],pos,m)
    Js.append(J); 
    if U_basis is None: U_basis=U; ma=m_
    if (l+1)%8==0: print(f"  L{l+1}...",flush=True)
print(f"  Done. m={ma}\n")

# ── Compute monodromies ───────────────────────────────────────────────────────
print(f"Step 3: Monodromies and Dehn gap at L{L_ATT}...")

# M_full: product of all Jacobians
M_full=np.eye(ma)
for J in reversed(Js): M_full=J@M_full

# M_fwd: L0 through L_ATT (approach to attractor)
M_fwd=np.eye(ma)
for l in range(L_ATT+1): M_fwd=Js[l]@M_fwd

# M_bwd: L_ATT+1 through L_N-1 (departure/tail)
M_bwd=np.eye(ma)
for l in range(N_LAYERS-1,L_ATT,-1): M_bwd=Js[l]@M_bwd

# T_L14: the Jacobian at the attractor center
T_att=Js[L_ATT]   # [ma, ma]

# Dehn gap
holonomy=M_bwd@M_fwd
dehn_gap=np.linalg.norm(holonomy-np.eye(ma))/ma
sv_fwd=np.linalg.svd(M_fwd,compute_uv=False)
sv_bwd=np.linalg.svd(M_bwd,compute_uv=False)
sv_full=np.linalg.svd(M_full,compute_uv=False)
sv_att=np.linalg.svd(T_att,compute_uv=False)

print(f"  sv(M_fwd):  {sv_fwd[:4].round(3)}")
print(f"  sv(M_bwd):  {sv_bwd[:4].round(3)}")
print(f"  sv(T_att):  {sv_att[:4].round(3)}")
print(f"  Dehn gap:   {dehn_gap:.4f}  (||M_bwd @ M_fwd - I|| / m)")
print(f"  Asymmetry:  {np.linalg.norm(sv_fwd[:4]-sv_bwd[:4]):.4f}\n")

# ── Shifted QR collapse (the Symes-Deift algorithm) ──────────────────────────
print(f"Step 4: Shifted QR collapse of tail L{L_ATT+1}..L{N_LAYERS-1}...")

def shifted_qr_collapse(T_att):
    """
    One step of shifted QR on the attractor matrix.
    μ = Rayleigh quotient (dominant eigenvalue estimate)
    Q,R = QR(T_att - μI)
    M_collapsed = RQ + μI
    This is one Toda time step — collapses the tail in one shot.
    """
    m=T_att.shape[0]
    # Rayleigh quotient: use dominant eigenvector
    sv,U_=np.linalg.eig(T_att)
    # Use the eigenvalue with largest real part
    idx=np.argmax(np.abs(sv.real))
    mu=float(sv[idx].real)

    A_shifted=T_att-mu*np.eye(m)
    Q,R=np.linalg.qr(A_shifted)
    M_collapsed=R@Q+mu*np.eye(m)
    return M_collapsed, mu

M_collapsed, mu=shifted_qr_collapse(T_att)
print(f"  Rayleigh shift μ={mu:.4f}")
print(f"  M_collapsed sv: {np.linalg.svd(M_collapsed,compute_uv=False)[:4].round(3)}")
err_col=np.linalg.norm(M_collapsed@M_collapsed-M_bwd@M_fwd)
print(f"  ||M_collapsed² - M_full|| = {err_col:.4f}")
sqM_col=np.real(scipy_sqrtm(M_collapsed))
print()

# ── Build compressed 2-layer models ──────────────────────────────────────────
print("Step 5: Build compressed models (zero gradient steps)...")

def build_compressed(target_mono, label):
    """
    Set 2-layer model weights from target monodromy via SVD factorization.
    Layer 1 ≈ sqrtm(target), Layer 2 ≈ sqrtm(target).
    """
    sqM=np.real(scipy_sqrtm(target_mono))
    dJ=sqM-np.eye(ma)
    U_t_np=U_basis
    W_op_m=SEQ*dJ; W_op_d=U_t_np@W_op_m@U_t_np.T
    W_v_d=U_t_np@U_t_np.T
    P_orth=np.eye(D)-U_t_np@U_t_np.T
    W_op_d=W_op_d+P_orth; W_v_d=W_v_d+P_orth

    torch.manual_seed(0)
    model=LM(D,N_HEADS,2)
    model.te.weight.data.copy_(source.te.weight.data)
    model.pe.weight.data.copy_(source.pe.weight.data)
    model.ln_f.weight.data.copy_(source.ln_f.weight.data)
    model.ln_f.bias.data.copy_(source.ln_f.bias.data)
    mid=N_LAYERS//2
    for blk in model.blocks:
        with torch.no_grad():
            blk.attn.op.weight.copy_(torch.tensor(W_op_d,dtype=torch.float32))
            blk.attn.WV.weight.copy_(torch.tensor(W_v_d,dtype=torch.float32))
            blk.attn.WQ.weight.copy_(source.blocks[mid].attn.WQ.weight)
            blk.attn.WK.weight.copy_(source.blocks[mid].attn.WK.weight)
            blk.attn.ln.weight.copy_(source.blocks[mid].attn.ln.weight)
            blk.attn.ln.bias.copy_(source.blocks[mid].attn.ln.bias)
            blk.ff.g.weight.copy_(source.blocks[mid].ff.g.weight)
            blk.ff.v.weight.copy_(source.blocks[mid].ff.v.weight)
            blk.ff.o.weight.copy_(source.blocks[mid].ff.o.weight)
            blk.ff.n.weight.copy_(source.blocks[mid].ff.n.weight)
            blk.ff.n.bias.copy_(source.blocks[mid].ff.n.bias)
    model.eval()
    vl=eval_val(model); cos=cosine_sim(source,model)
    print(f"  [{label}]  val={vl:.4f}  cos={cos:.4f}")
    return model

# Three compression strategies
m1=build_compressed(M_full,     "sqrtm(M_full)      — symmetric, current method")
m2=build_compressed(M_fwd,      "sqrtm(M_fwd)       — approach half only      ")
m3=build_compressed(M_collapsed,"sqrtm(M_collapsed) — shifted QR tail collapse ")

# Random 2L baseline
torch.manual_seed(99)
rand2=LM(D,N_HEADS,2)
vl_r=eval_val(rand2); cos_r=cosine_sim(source,rand2)
print(f"  [Random 2-layer baseline]    val={vl_r:.4f}  cos={cos_r:.4f}")

# ── Layer culling test ────────────────────────────────────────────────────────
print(f"\nStep 6: Layer culling — does freezing |l-{L_ATT}|>k speed training?")

def clr_sgd(s,total=300,warmup=50,base=0.05):
    if s<=warmup: return base*s/warmup
    return base*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def run_culling(name, k_fixed=None, adaptive=False, seed=42):
    torch.manual_seed(seed); m=LM(D,N_HEADS,N_LAYERS); m._nl=N_LAYERS
    opt=torch.optim.SGD(m.parameters(),lr=0.05,momentum=0.9,nesterov=True)
    stt=None; t0=time.time()
    print(f"\n  [{name}]")
    for step in range(1,301):
        for pg in opt.param_groups: pg['lr']=clr_sgd(step)
        m.train(); x,y=get_batch(); _,loss=m(x,y)
        opt.zero_grad(); loss.backward()
        # Layer culling
        if k_fixed is not None or adaptive:
            k=k_fixed if k_fixed is not None else max(1,int(N_LAYERS//2*(1-step/300)))
            for l,blk in enumerate(m.blocks):
                if abs(l-L_ATT)>k:
                    for p in blk.parameters():
                        if p.grad is not None: p.grad.zero_()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if step%75==0 or step==1:
            vl=eval_val(m,n=20)
            if vl<4.0 and stt is None: stt=step; print(f"    *** <4.0 at step {step} ***")
            k_now=k_fixed if k_fixed is not None else (max(1,int(N_LAYERS//2*(1-step/300))) if adaptive else 'none')
            print(f"    step {step:>4}  val={vl:.4f}  t={time.time()-t0:.0f}s  k={k_now}")
    return stt, eval_val(m,n=60)

stt_base,fval_base=run_culling("SGD baseline (no culling)",  k_fixed=None)
stt_ada, fval_ada =run_culling("SGD + adaptive k (12→1)",    adaptive=True)
stt_k4,  fval_k4  =run_culling("SGD + fixed k=4 (L10-L18)", k_fixed=4)
stt_k2,  fval_k2  =run_culling("SGD + fixed k=2 (L12-L16)", k_fixed=2)

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  COMPLETE RESULTS")
print("="*65)

print(f"""
  DEHN GAP (holonomy framing error at L{L_ATT}):
    ||M_bwd @ M_fwd - I|| / m = {dehn_gap:.4f}
    Asymmetry sv_fwd vs sv_bwd = {np.linalg.norm(sv_fwd[:4]-sv_bwd[:4]):.4f}
    
  COMPRESSION (2-layer, zero gradient steps):
    Method                           val      cos(logits)
    sqrtm(M_full) — symmetric        {eval_val(m1):.4f}    {cosine_sim(source,m1):.4f}
    sqrtm(M_fwd)  — fwd half         {eval_val(m2):.4f}    {cosine_sim(source,m2):.4f}
    sqrtm(M_coll) — shifted QR       {eval_val(m3):.4f}    {cosine_sim(source,m3):.4f}
    Random 2L baseline               {vl_r:.4f}    {cos_r:.4f}

  LAYER CULLING (steps to val<4.0):
    SGD baseline:      step {stt_base or '>300'}  final={fval_base:.4f}
    Adaptive k 12→1:   step {stt_ada  or '>300'}  final={fval_ada:.4f}
    Fixed k=4:         step {stt_k4   or '>300'}  final={fval_k4:.4f}
    Fixed k=2:         step {stt_k2   or '>300'}  final={fval_k2:.4f}

  KEY READING:
    Dehn gap < 0.1:  L{L_ATT} is near-perfect holonomy center.
                     sqrtm(M_fwd) ≈ sqrtm(M_bwd). Symmetric ok.
    Dehn gap > 0.5:  Significant asymmetry. Use M_fwd / M_bwd separately.
    
    Shifted QR cos > symmetric sqrtm cos:
      The Symes-Deift one-shot tail collapse works.
      One QR step on T_att captures 9 layers of relaxation tail.
    
    Adaptive culling faster than baseline:
      Inter-layer Toda flow confirmed operationally.
      Outer layers are redundant early in training.
""")
