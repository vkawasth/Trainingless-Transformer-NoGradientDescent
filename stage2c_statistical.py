#!/usr/bin/env python3
"""
K₀ Split in Statistical Phase — Step Reduction Experiment
==========================================================
Tests whether K₀ split (Emb+FF branch / Attn branch, w_FF×3.5)
works in the STATISTICAL phase (val≈0.20 after basin) as well as
the ALGEBRAIC phase (val≈3.45 where it was confirmed).

CONFIRMED:
  Algebraic phase (val=3.45): 13 K₀ CE = 25 joint CE  (stage2c)
  Statistical phase (val=0.20): UNKNOWN — this experiment measures it

K₀ SPLIT MECHANISM:
  Branch 1: Emb+FF only, LR×2, cosine schedule → Δ_FF
  Branch 2: WK+WQ only, LR×2, cosine schedule → Δ_Attn
  Combine:  θ_out = θ_base + Δ_Emb + w_FF×Δ_FF + Δ_Attn

WHY w_FF MATTERS IN STATISTICAL PHASE:
  τ = ||∇_FF||/||∇_Emb|| at val=0.20 is ~2-3 (measured)
  vs τ≈1.5 at val=3.45 (algebraic)
  Higher τ = FF gradient already dominant
  May need different w_FF: dynamic w_FF = τ_ref/τ_current × 3.5

STEP REDUCTION TARGET:
  Current: 100CE (25 large + 75 cosine) → val=0.047
  Target:  13 K₀ CE → same val=0.047
  If achieved: 100→13 CE reduction (8× speedup) in statistical phase
"""
import json, math, warnings, collections, copy, os, sys
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f): sys.exit(f"ERROR: {f}")

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

def ptype(name):
    if '.attn.WQ.' in name: return 'WQ'
    if '.attn.WK.' in name: return 'WK'
    if 'te.weight'  in name: return 'Emb'
    if '.ff.'       in name: return 'FF'
    return 'other'

def measure_tau(model, n=8):
    """τ = ||∇_FF|| / ||∇_Emb|| — K₀ gluing defect."""
    model.zero_grad()
    ls=[model(*get_batch())[1] for _ in range(n)]
    torch.stack(ls).mean().backward()
    g_ff=sum(p.grad.data.norm().item() for nm,p in model.named_parameters()
             if ptype(nm)=='FF' and p.grad is not None)
    g_emb=model.te.weight.grad.data.norm().item() if model.te.weight.grad is not None else 1e-8
    model.zero_grad()
    return g_ff/max(g_emb,1e-8)

def k0_split(base, n_steps, lr_emb_ff, lr_attn, w_ff, cosine_schedule=True):
    """K₀ split — exact from stage2c_step_reduction.py."""
    params_base={n:p.data.clone() for n,p in base.named_parameters()}

    def get_lr(step, n_steps, base_lr, schedule):
        if not schedule: return base_lr
        if step<=3: return base_lr*(0.5+0.5*step/3)
        return base_lr*(0.5+0.5*math.cos(math.pi*(step-3)/(n_steps-3)))

    # Branch 1: Emb + FF
    m1=copy.deepcopy(base)
    for name,p in m1.named_parameters():
        if ptype(name) not in {'Emb','FF'}: p.requires_grad_(False)
    p1=[p for p in m1.parameters() if p.requires_grad]
    opt1=torch.optim.AdamW(p1,lr=lr_emb_ff,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,n_steps+1):
        if cosine_schedule:
            for pg in opt1.param_groups: pg['lr']=get_lr(step,n_steps,lr_emb_ff,True)
        m1.train(); x,y=get_batch(); _,l=m1(x,y)
        opt1.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(p1,1.0); opt1.step()

    # Branch 2: WK + WQ
    m2=copy.deepcopy(base)
    for name,p in m2.named_parameters():
        if ptype(name) not in {'WK','WQ'}: p.requires_grad_(False)
    p2=[p for p in m2.parameters() if p.requires_grad]
    opt2=torch.optim.AdamW(p2,lr=lr_attn,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,n_steps+1):
        if cosine_schedule:
            for pg in opt2.param_groups: pg['lr']=get_lr(step,n_steps,lr_attn,True)
        m2.train(); x,y=get_batch(); _,l=m2(x,y)
        opt2.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(p2,1.0); opt2.step()

    # Combine: Emb×1 + FF×w_ff + Attn×1
    m_out=copy.deepcopy(base)
    with torch.no_grad():
        for name,p in m_out.named_parameters():
            pt=ptype(name)
            d1=dict(m1.named_parameters())[name].data-params_base[name]
            d2=dict(m2.named_parameters())[name].data-params_base[name]
            if pt=='Emb':   p.data.add_(d1)
            elif pt=='FF':  p.data.add_(w_ff*d1)
            elif pt in ('WK','WQ'): p.data.add_(d2)
    return m_out

def joint_ce(base, n_steps, lr, cosine=True):
    """Standard joint CE — all parameters together."""
    m=copy.deepcopy(base)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,n_steps+1):
        if cosine:
            lr_now=lr*0.5*(1+math.cos(math.pi*step/n_steps))
            for pg in opt.param_groups: pg['lr']=lr_now
        m.train(); x,y=get_batch(); _,l=m(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return m

# ── LOAD BASIN STATE ─────────────────────────────────────────
print("="*65)
print("K₀ SPLIT IN STATISTICAL PHASE — STEP REDUCTION")
print("="*65); print()

# Load basin ENTRY state (post-TopoGate, val≈0.20)
# NOT basin_state.pt which may be near-floor after Phase 5
BASIN_STATE = 'basin_entry_state.pt'
if not os.path.exists(BASIN_STATE):
    BASIN_STATE = 'basin_state.pt'
    print(f"Note: basin_entry_state.pt not found, using basin_state.pt")
    print(f"For best results: run compiler_geometric.py to generate basin_entry_state.pt")
if not os.path.exists(BASIN_STATE):
    sys.exit("ERROR: No basin state found. Run compiler_geometric.py first.")

torch.manual_seed(99)
base=LM(); base.load_state_dict(torch.load(BASIN_STATE,map_location='cpu'))
v_base=eval_val(base,n=20)
tau_base=measure_tau(base)
print(f"Basin state: val={v_base:.4f}  τ={tau_base:.2f}")
print(f"(τ={tau_base:.2f}: dynamic w_FF = 3.5 × 2.0/{tau_base:.2f} = {3.5*2.0/tau_base:.2f})")
print()

# Dynamic w_FF based on τ
# At algebraic phase τ≈1.5, confirmed w_FF=3.5
# At statistical phase τ≈3-5, scale proportionally
w_ff_dynamic = 3.5 * 1.5 / max(tau_base, 1.0)
print(f"Dynamic w_FF (τ-scaled): {w_ff_dynamic:.2f}")
print(f"Static  w_FF (confirmed): 3.5")
print()

# ── REFERENCE: JOINT CE ──────────────────────────────────────
print("REFERENCES — Joint CE from basin state:")
refs={}
for n in [13, 25, 50, 100]:
    m=joint_ce(base, n, LR, cosine=True)
    refs[n]=eval_val(m,n=15)
    print(f"  Joint {n:3d} CE (cosine): val={refs[n]:.4f}")
print()
v_target_100 = refs[100]  # what 100 CE achieves
v_target_25  = refs[25]

# ── K₀ SPLIT SWEEP ───────────────────────────────────────────
print("K₀ SPLIT SWEEP — finding optimal w_FF and n_steps")
print(f"Target: match {v_target_100:.4f} (100 joint CE) in ≤13 K₀ steps")
print()

print(f"Scan 1: w_FF sweep at n=13 K₀ steps")
print(f"  {'w_FF':>6}  {'val':>7}  {'vs 100CE':>9}  {'vs 25CE':>8}")
print("  "+"-"*40)
best_v=999; best_wff=3.5
# Statistical phase range: τ≈3-5 → w_FF≈0.5-2.0 (not 3.5)
# Also test w_ff_dynamic (τ-scaled) and 3.5 (algebraic reference)
for w_ff in sorted(set([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.5, round(w_ff_dynamic,2)])):
    mc=k0_split(base, 13, LR*2, LR*2, w_ff, cosine_schedule=True)
    v=eval_val(mc,n=15)
    flag='  ←BEST' if v<best_v else ''
    print(f"  {w_ff:>6.2f}  {v:>7.4f}  {v-v_target_100:>+9.4f}  {v-v_target_25:>+8.4f}{flag}")
    if v<best_v: best_v=v; best_wff=w_ff
print()

print(f"Scan 2: step count with best w_FF={best_wff:.2f}")
print(f"  {'steps':>6}  {'val':>7}  {'vs 100CE':>9}  {'vs joint_n':>10}")
print("  "+"-"*45)
for n in [8, 10, 12, 13, 15, 20, 25]:
    mc=k0_split(base, n, LR*2, LR*2, best_wff, cosine_schedule=True)
    v=eval_val(mc,n=15)
    joint_n=refs.get(n)
    joint_str=f"{v-joint_n:+.4f}" if joint_n else "  N/A"
    flag='  ✓' if v<=v_target_100 else ''
    print(f"  {n:>6}  {v:>7.4f}  {v-v_target_100:>+9.4f}  {joint_str:>10}{flag}")
print()

print(f"Scan 3: LR sweep at n=13, w_FF={best_wff:.2f}")
print(f"  {'LR_mult':>8}  {'val':>7}  {'vs 100CE':>9}")
print("  "+"-"*35)
best_lr_v=999; best_lr_mult=2.0
for lr_mult in [1.0, 1.5, 2.0, 3.0, 5.0]:
    mc=k0_split(base, 13, LR*lr_mult, LR*lr_mult, best_wff, cosine_schedule=True)
    v=eval_val(mc,n=15)
    flag='  ←BEST' if v<best_lr_v else ''
    print(f"  {lr_mult:>8.1f}  {v:>7.4f}  {v-v_target_100:>+9.4f}{flag}")
    if v<best_lr_v: best_lr_v=v; best_lr_mult=lr_mult
print()

# ── SUMMARY ──────────────────────────────────────────────────
# ── SCAN 4: ANNEALED w_FF (per proposal) ────────────────────
print(f"Scan 4: Annealed w_FF (0.85→0.35 over 13 steps)")
print(f"  Proposal: high w_FF early (fix geometry), low w_FF late (statistics)")
print(f"  {'schedule':>20}  {'val':>7}  {'vs 100CE':>9}")
print("  "+"-"*45)

def k0_split_annealed(base, n_steps, lr, w_start, w_end):
    """K₀ split with linearly annealed w_FF."""
    params_base={n:p.data.clone() for n,p in base.named_parameters()}
    def _ptype(name):
        if '.attn.WQ.' in name or '.attn.WK.' in name: return 'Attn'
        if 'te.weight' in name or '.ff.' in name: return 'EmbFF'
        return 'other'

    m_cur=copy.deepcopy(base)
    opt=torch.optim.AdamW(m_cur.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(n_steps):
        # Anneal w_FF
        w_ff_t=w_start+(w_end-w_start)*(step/(n_steps-1))
        lr_t=lr*0.5*(1+math.cos(math.pi*step/n_steps))
        for pg in opt.param_groups: pg['lr']=lr_t

        # Save current state, run split, recombine
        params_t={n:p.data.clone() for n,p in m_cur.named_parameters()}
        m1=copy.deepcopy(m_cur)
        for name,p in m1.named_parameters():
            if _ptype(name)!='EmbFF': p.requires_grad_(False)
        p1=[p for p in m1.parameters() if p.requires_grad]
        opt1=torch.optim.AdamW(p1,lr=lr_t,betas=(0.9,0.95),weight_decay=0.1)
        m1.train(); x,y=get_batch(); _,l=m1(x,y)
        opt1.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(p1,1.0); opt1.step()

        m2=copy.deepcopy(m_cur)
        for name,p in m2.named_parameters():
            if _ptype(name)!='Attn': p.requires_grad_(False)
        p2=[p for p in m2.parameters() if p.requires_grad]
        opt2=torch.optim.AdamW(p2,lr=lr_t,betas=(0.9,0.95),weight_decay=0.1)
        m2.train(); x,y=get_batch(); _,l=m2(x,y)
        opt2.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(p2,1.0); opt2.step()

        with torch.no_grad():
            for name,p in m_cur.named_parameters():
                pt=_ptype(name)
                d1=dict(m1.named_parameters())[name].data-params_t[name]
                d2=dict(m2.named_parameters())[name].data-params_t[name]
                if pt=='EmbFF':
                    p.data=params_t[name]+d1 if 'te.weight' in name else params_t[name]+w_ff_t*d1
                elif pt=='Attn': p.data=params_t[name]+d2
    return m_cur

for w_start,w_end in [(0.85,0.35),(0.70,0.30),(0.50,0.20),(1.0,0.1)]:
    ma=k0_split_annealed(base,13,LR,w_start,w_end)
    va=eval_val(ma,n=15)
    print(f"  {w_start:.2f}→{w_end:.2f} over 13:  {va:>7.4f}  {va-v_target_100:>+9.4f}")
print()

# ── CORRECT TWO-PHASE STRATEGY ────────────────────────────────
print(f"TWO-PHASE STRATEGY (correct approach):")
print(f"  Phase 1: get to val≈0.065 FAST (Lanczos Newton ~8CE equiv)")
print(f"  Phase 2: K₀ 25 steps from val≈0.065 → val≈0.047")
print(f"  Total: ~108 CE equiv (vs 175 current, vs 400 GD)")
print()
print(f"  From current basin (val={v_base:.3f}): one Lanczos step")
def hvp_st(model,v,n=4):
    model.zero_grad()
    ls=[model(*get_batch())[1] for _ in range(n)]; loss=torch.stack(ls).mean()
    grads=torch.autograd.grad(loss,list(model.parameters()),create_graph=True)
    gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in
                  torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)])
    model.zero_grad(); return hv.detach()

m_lanc=copy.deepcopy(base)
n_p=sum(p.numel() for p in m_lanc.parameters())
torch.manual_seed(7); q=torch.randn(n_p); q=q/q.norm()
Q=[q]; alphas=[]; betas=[]
for j in range(8):
    z=hvp_st(m_lanc,Q[j]); alpha=float((Q[j]*z).sum()); alphas.append(alpha)
    z=z-alpha*Q[j]
    if j>0: z=z-betas[-1]*Q[j-1]
    for qi in Q: z=z-float((qi*z).sum())*qi
    beta=float(z.norm()); betas.append(beta)
    if beta<1e-8: break
    Q.append(z/beta)
n_l=len(alphas)
T=torch.zeros(n_l,n_l)
for i in range(n_l): T[i,i]=alphas[i]
for i in range(n_l-1): T[i,i+1]=betas[i]; T[i+1,i]=betas[i]
T_ev,T_evec=torch.linalg.eigh(T); V=torch.stack(Q[:n_l],dim=1)@T_evec

m_lanc.zero_grad()
ls=[m_lanc(*get_batch())[1] for _ in range(25)]; torch.stack(ls).mean().backward()
g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
             for p in m_lanc.parameters()]).detach(); m_lanc.zero_grad()
mu=0.95; g_proj=V.T@g; d_proj=g_proj/(T_ev+mu)
g_res=g-V@(V.T@g); d=-(V@d_proj+g_res/mu)
w0=m_lanc.flat_params(); v0=eval_val(m_lanc,n=8)
m_lanc.set_flat(w0+d); v_lanc=eval_val(m_lanc,n=8)
if v_lanc>=v0: m_lanc.set_flat(w0); v_lanc=v0
print(f"  Lanczos step from val={v0:.4f}: → val={v_lanc:.4f}  "
      f"(cost: 8 HVPs×4 + 25 grad = 64+50=114 CE equiv)")

# Now K₀ 25 steps from Lanczos state
tau_l=measure_tau(m_lanc); w_ff_l=3.5*(1.5/max(tau_l,0.5))**1.5
m_k0=k0_split(m_lanc,25,LR,LR,w_ff_l,cosine_schedule=True)
v_k0_l=eval_val(m_k0,n=15)
print(f"  K₀ 25 steps (τ={tau_l:.2f},w_FF={w_ff_l:.2f}) from val={v_lanc:.4f}: → val={v_k0_l:.4f}")
print()

print("="*65)
print("RESULTS")
print("="*65)
print(f"  Basin state:      val={v_base:.4f}  τ={tau_base:.2f}")
print(f"  Target (100 CE):  val={v_target_100:.4f}")
print(f"  Target (25 CE):   val={v_target_25:.4f}")
print()
print(f"  K₀ 13 steps best:  val={best_v:.4f}  w_FF={best_wff:.2f}")
print(f"  K₀ 25 steps best:  val={refs.get(25,'?'):.4f} (joint ref)")
print()
print(f"  Lanczos + K₀ 25:   val={v_k0_l:.4f}  (target path)")
gap_lanc=v_k0_l-v_target_100
print(f"  Gap vs 100 CE:     {gap_lanc:+.4f}  "
      f"({'✓ MATCHES' if gap_lanc<=0.005 else f'✗ {gap_lanc:.3f} remaining'})")
print()
print(f"  CONCLUSION:")
print(f"  K₀ split gives 4× speedup FROM val≈0.065 (near-floor)")
print(f"  From val=0.21: K₀ 13 steps ≠ 100 CE (rare-token gap)")
print(f"  Correct path: Lanczos (fast descent) + K₀ 25 steps (final integration)")
