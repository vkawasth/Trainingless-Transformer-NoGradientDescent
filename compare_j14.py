#!/usr/bin/env python3
"""
3-Way J14 Comparison
====================
ARM 1: Standard GD-300
ARM 2: Compiler WITH J14 (GD300's W_K at layer 14 → all student layers)
ARM 3: Compiler WITHOUT J14 (algebraic only — current compiler_demo)

QUESTIONS:
  Q1: Does ARM 2 beat ARM 3? (does J14 geometry matter?)
  Q2: What sheet path does each arm take? (s_1, s_2) ∈ {0,π}²
  Q3: At what point do ARM 2 and ARM 3 diverge?
  Q4: What does ARM 2's W_K encode that ARM 3's doesn't?

SHEET NUMBERING:
  Not +1/-1 but the actual Bridgeland path:
  (s_1, s_2) where s_l = Im(z_l) ∈ {0, π}
  z_l = arg(λ₁(φ_l)) where φ_l = W_K(l+1)·W_K(l)⁻¹
  Teacher path (after TopoGate): (0, 0) = both layers positive real half-plane
  Wrong path: any other combination
"""
import json, math, warnings, collections, os, copy, sys, time
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
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

def sheet_path(model):
    """
    Bridgeland sheet path: for each layer pair (l, l+1),
    compute φ_l = W_K(l+1) · W_K(l)^{-1}
    s_l = Im(z_l) = arg(λ₁(φ_l)) ∈ {0, π, other}
    Returns list of sheet labels per layer transition.
    
    s_l ≈ 0:  positive real λ₁ → correct half-plane
    s_l ≈ π:  negative real λ₁ → wall crossing (wrong sheet)
    other:    complex λ₁ → intermediate (transition)
    """
    path = []
    WKs = [model.blocks[l].attn.WK.weight.data for l in range(N_STU)]
    for l in range(N_STU-1):
        WK_l   = WKs[l].float()
        WK_lp1 = WKs[l+1].float()
        try:
            # φ_l = W_K(l+1) · W_K(l)^{-1}
            phi = WK_lp1 @ torch.linalg.pinv(WK_l)
            # Leading eigenvalue of φ_l
            eigs = torch.linalg.eigvals(phi)
            # Sort by magnitude descending
            idx = torch.argsort(eigs.abs(), descending=True)
            lam1 = eigs[idx[0]]
            # Im(z_l) = arg(λ₁)
            angle = float(torch.angle(lam1))
            # Classify
            if abs(angle) < 0.3:          s = '0'      # positive real
            elif abs(abs(angle)-math.pi) < 0.3: s = 'π'  # negative real (wall)
            else:                          s = f'{angle:.2f}'  # intermediate
        except Exception:
            s = '?'
        path.append(s)
    return path

def sheet_str(path):
    return '(' + ','.join(path) + ')'

def monodromy_sv(model):
    """Forward monodromy singular value: sv(M_fwd) where M_fwd = J_5 ∘ ... ∘ J_0.
    Approximated via product of W_K norms (proxy for Jacobian SV)."""
    svs = []
    for l in range(N_STU):
        WK = model.blocks[l].attn.WK.weight.data.float()
        s = float(torch.linalg.svdvals(WK)[0])
        svs.append(s)
    # Product = forward transport amplification
    sv_fwd = float(np.prod(svs))**(1/N_STU)  # geometric mean
    return sv_fwd, svs

def run_adam(model, n, lr=LR, checkpoints=None, label=''):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1)
    results={}
    for step in range(1,n+1):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if checkpoints and step in checkpoints:
            v=eval_val(model)
            results[step]=v
            sp=sheet_str(sheet_path(model))
            sv,_=monodromy_sv(model)
            print(f"    [{label}] step {step:3d}: val={v:.4f}  sheet={sp}  sv={sv:.2f}")
    return results

def lm_step(model, mu=0.950, n_grad=25, n_hvp=12, n_cg=6):
    model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n_grad))/n_grad
    loss.backward()
    g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                 for p in model.parameters()]).detach(); model.zero_grad()
    def hvp(v):
        model.zero_grad()
        loss2=sum(model(*get_batch())[1] for _ in range(n_hvp))/n_hvp
        grads=torch.autograd.grad(loss2,list(model.parameters()),create_graph=True)
        gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
        hv=torch.cat([h.flatten() for h in
                      torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)])
        model.zero_grad(); return hv.detach()
    d=torch.zeros_like(g); r=-g.clone(); p_cg=r.clone(); rr=float((r*r).sum())
    for _ in range(n_cg):
        Hp=hvp(p_cg)+mu*p_cg; alpha=rr/max(float((p_cg*Hp).sum()),1e-10)
        d+=alpha*p_cg; r-=alpha*Hp; rr_new=float((r*r).sum())
        p_cg=r+(rr_new/max(rr,1e-10))*p_cg; rr=rr_new
    w0=model.flat_params(); L0=eval_val(model,n=8)
    model.set_flat(w0+d); L_new=eval_val(model,n=8)
    if L_new<L0: return eval_val(model), True
    model.set_flat(w0); return L0, False

# ── CORPUS ────────────────────────────────────────────────────
print("="*65)
print("3-WAY J14 COMPARISON")
print("="*65); print()
print("Building corpus + spectral E₀...")
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
print(f"  VOCAB={VOCAB}, nnz={len(bigram)}, E₀ ready")
print()

CKPTS_GD={25,50,100,167,200,274,300}
CKPTS_COMP={12,25,26}  # key: step 26 = after 1 LM

# ══════════════════════════════════════════════════════════════
# ARM 1: GD-300 — extract W_K at L14 mapping (L5 in 6-layer)
# ══════════════════════════════════════════════════════════════
print("━━━ ARM 1: STANDARD GD-300 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Running 300 CE steps (Adam, standard LR)")
print("  Will extract W_K at 'L14-equivalent' = layer 5 (deepest)")
print()
torch.manual_seed(99)
gd=LM(); gd.te.weight.data.copy_(torch.tensor(E_init))
opt=torch.optim.AdamW(gd.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
sp_init=sheet_str(sheet_path(gd)); sv_init,svs_init=monodromy_sv(gd)
print(f"  [GD] init: val={eval_val(gd):.4f}  sheet={sp_init}  sv={sv_init:.2f}")
gd_ckpts={}
for step in range(1,301):
    gd.train(); x,y=get_batch(); _,l=gd(x,y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(gd.parameters(),1.0); opt.step()
    if step in CKPTS_GD:
        v=eval_val(gd)
        sp=sheet_str(sheet_path(gd)); sv,_=monodromy_sv(gd)
        print(f"  [GD] step {step:3d}: val={v:.4f}  sheet={sp}  sv={sv:.2f}")
        gd_ckpts[step]=v
# Extract J14: W_K from the deepest layer (layer 5 in 6-layer student)
# = the Jacobian fixed-point layer (equivalent to teacher L14)
j14_layer = N_STU - 1  # deepest layer = fixed-point attractor
j14_WK = gd.blocks[j14_layer].attn.WK.weight.data.clone()
j14_WQ = gd.blocks[j14_layer].attn.WQ.weight.data.clone()
sp_final=sheet_str(sheet_path(gd)); sv_final,svs_final=monodromy_sv(gd)
print(f"  [GD] final: val={gd_ckpts[300]:.4f}  sheet={sp_final}  sv={sv_final:.2f}")
print(f"  J14 extracted: W_K from layer {j14_layer} (deepest = fixed-point)")
print(f"  J14 W_K norm={float(j14_WK.norm()):.3f}  sv₁={float(torch.linalg.svdvals(j14_WK)[0]):.3f}")
print()

# ══════════════════════════════════════════════════════════════
# ARM 2: Compiler WITH J14 (GD300's W_K at L5 → all layers)
# ══════════════════════════════════════════════════════════════
print("━━━ ARM 2: COMPILER WITH J14 (GD300 W_K → all layers) ━━━━")
print(f"  J14 = GD300's W_K at layer {j14_layer}, broadcast to all layers")
print("  Hypothesis: pre-positions in correct Kac-Moody orbit")
print("  → 25 CE does co-adaptation, not Moran fixation")
print()
torch.manual_seed(99)
arm2=LM(); arm2.te.weight.data.copy_(torch.tensor(E_init))
# Apply J14: copy W_K (and W_Q) to ALL layers
with torch.no_grad():
    for l in range(N_STU):
        arm2.blocks[l].attn.WK.weight.data.copy_(j14_WK)
        arm2.blocks[l].attn.WQ.weight.data.copy_(j14_WQ)
v_a2_init=eval_val(arm2)
sp_a2=sheet_str(sheet_path(arm2)); sv_a2,_=monodromy_sv(arm2)
print(f"  After J14 init: val={v_a2_init:.4f}  sheet={sp_a2}  sv={sv_a2:.2f}")
# 25 CE
print("  Running 25 CE...")
run_adam(arm2, 25, checkpoints={12,25}, label='A2_CE')
v_a2_ce=eval_val(arm2)
sp_a2_ce=sheet_str(sheet_path(arm2))
print(f"  After 25 CE: val={v_a2_ce:.4f}  sheet={sp_a2_ce}")
# 1 LM
v_a2_lm,acc2=lm_step(arm2)
sp_a2_lm=sheet_str(sheet_path(arm2))
print(f"  After 1 LM: val={v_a2_lm:.4f}  {'✓' if acc2 else '~'}  sheet={sp_a2_lm}")
# 167 CE
print("  Running 167 CE continuation...")
run_adam(arm2, 167, checkpoints={25,50,100,141,167}, label='A2_cont')
v_a2_final=eval_val(arm2)
sp_a2_f=sheet_str(sheet_path(arm2))
print(f"  ARM 2 FINAL (193 steps): val={v_a2_final:.4f}  sheet={sp_a2_f}")
print()

# ══════════════════════════════════════════════════════════════
# ARM 3: Compiler WITHOUT J14 (algebraic only)
# ══════════════════════════════════════════════════════════════
print("━━━ ARM 3: COMPILER WITHOUT J14 (algebraic only) ━━━━━━━━━")
print("  Standard spectral E₀ init, random W_K")
print("  This is the current compiler_demo pipeline")
print()
torch.manual_seed(99)
arm3=LM(); arm3.te.weight.data.copy_(torch.tensor(E_init))
v_a3_init=eval_val(arm3)
sp_a3=sheet_str(sheet_path(arm3)); sv_a3,_=monodromy_sv(arm3)
print(f"  After spectral init: val={v_a3_init:.4f}  sheet={sp_a3}  sv={sv_a3:.2f}")
# 25 CE
print("  Running 25 CE...")
run_adam(arm3, 25, checkpoints={12,25}, label='A3_CE')
v_a3_ce=eval_val(arm3)
sp_a3_ce=sheet_str(sheet_path(arm3))
print(f"  After 25 CE: val={v_a3_ce:.4f}  sheet={sp_a3_ce}")
# 1 LM
v_a3_lm,acc3=lm_step(arm3)
sp_a3_lm=sheet_str(sheet_path(arm3))
print(f"  After 1 LM: val={v_a3_lm:.4f}  {'✓' if acc3 else '~'}  sheet={sp_a3_lm}")
# 167 CE
print("  Running 167 CE continuation...")
run_adam(arm3, 167, checkpoints={25,50,100,141,167}, label='A3_cont')
v_a3_final=eval_val(arm3)
sp_a3_f=sheet_str(sheet_path(arm3))
print(f"  ARM 3 FINAL (193 steps): val={v_a3_final:.4f}  sheet={sp_a3_f}")
print()

# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════
print("="*65)
print("3-WAY COMPARISON RESULTS")
print("="*65); print()
gd300=gd_ckpts[300]
print(f"  {'Arm':<44} {'Steps':>6}  {'val':>7}  sheet")
print("  "+"-"*62)
print(f"  {'ARM 1: GD-300 (pure gradient descent)':44} {'300':>6}  {gd300:>7.4f}  {sp_final}")
print(f"  {'ARM 2: Compiler + J14 (GD W_K at L5→all)':44} {'193':>6}  {v_a2_final:>7.4f}  {sp_a2_f}")
print(f"  {'ARM 3: Compiler (algebraic only)':44} {'193':>6}  {v_a3_final:>7.4f}  {sp_a3_f}")
print()
print("  DIVERGENCE POINTS:")
print(f"  After J14 init:  A2={v_a2_init:.4f}  A3={v_a3_init:.4f}  Δ={v_a2_init-v_a3_init:+.4f}")
print(f"  After 25 CE:     A2={v_a2_ce:.4f}  A3={v_a3_ce:.4f}  Δ={v_a2_ce-v_a3_ce:+.4f}")
print(f"  After 1 LM:      A2={v_a2_lm:.4f}  A3={v_a3_lm:.4f}  Δ={v_a2_lm-v_a3_lm:+.4f}")
print(f"  Final (193):     A2={v_a2_final:.4f}  A3={v_a3_final:.4f}  Δ={v_a2_final-v_a3_final:+.4f}")
print()
print("  SHEET PATH EVOLUTION:")
print(f"  ARM 2: init={sp_a2} → 25CE={sp_a2_ce} → LM={sp_a2_lm} → final={sp_a2_f}")
print(f"  ARM 3: init={sp_a3} → 25CE={sp_a3_ce} → LM={sp_a3_lm} → final={sp_a3_f}")
print()
print("  MONODROMY SV (W_K geometric mean singular value):")
print(f"  GD-300 init: {sv_init:.3f} → final: {sv_final:.3f} (rescaling from 27.4→13.8)")
print(f"  ARM 2 init: {sv_a2:.3f}  (inherits GD300 geometry)")
print(f"  ARM 3 init: {sv_a3:.3f}  (random — no Bridgeland structure)")
print()
print("  KEY QUESTION ANSWERED:")
if v_a2_final < v_a3_final - 0.01:
    print(f"  ✓ ARM 2 beats ARM 3 by {v_a3_final-v_a2_final:.4f} nats")
    print(f"    → J14 geometry IS providing critical orbit information")
    print(f"    → Sheet path: ARM 2 takes teacher's path, ARM 3 searches randomly")
elif abs(v_a2_final - v_a3_final) < 0.02:
    print(f"  ~ ARM 2 ≈ ARM 3: J14 geometry marginal at this level")
    print(f"    → Sheet path difference: {sp_a2_f} vs {sp_a3_f}")
else:
    print(f"  ✗ ARM 3 beats ARM 2 by {v_a2_final-v_a3_final:.4f}")
    print(f"    → Unexpected: algebraic path better than J14")
    print(f"    → Check sheet paths: {sp_a2_f} vs {sp_a3_f}")
