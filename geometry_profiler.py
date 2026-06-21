#!/usr/bin/env python3
"""
AU-Fukaya Geometry Profiler
============================
Tracks ALL geometric quantities at each step.
Compares compiler algebraic passes vs standard GD side-by-side.
Diagnoses where angular/curvature errors accumulate.

QUANTITIES TRACKED:
  val          — validation loss (primary metric)
  gnorm        — gradient norm (phase indicator)
  cos(g,Δθ)   — alignment of gradient with update direction
                 >0: update aligns with gradient (good)
                 <0: update fights gradient (TWIST ERROR)
  cos(g,E)     — gradient alignment with embedding
                 -0.576 at init = the anti-alignment we found
  λ_min(H)     — minimum Hessian eigenvalue
                 <0: saddle (topological phase)
                 >0: valley (algebraic/statistical phase)
  λ_max(H)     — max Hessian eigenvalue (curvature scale)
  ridge_cross  — whether we've crossed from saddle to valley
                 (λ_min changes sign)
  basin_depth  — how far into valley-2 (||θ - θ_saddle||)
  sheet        — Z/2Z sheet indicator (sign of WV·WO det)
  kv_error     — KV/accumulation error: ||∇L - H·Δθ|| / ||∇L||
                 measures 2nd-order defect from Adam's 1st-order steps

CHECKPOINTS:
  For GD:       every 5 steps up to 50
  For compiler: after each algebraic pass
"""
import json, math, warnings, collections, os, copy, sys, time
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; ETA_MF=0.01

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f):
        print(f"ERROR: {f} missing."); sys.exit(1)

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

# ── Model ─────────────────────────────────────────────────────
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

# ── GEOMETRY PROBES ───────────────────────────────────────────
def get_gradient(model, n=20):
    model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    loss.backward()
    g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                 for p in model.parameters()]).detach()
    model.zero_grad(); return g

def hvp(model, v, n=8):
    model.zero_grad()
    loss2=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss2,list(model.parameters()),create_graph=True)
    gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in
                  torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)])
    model.zero_grad(); return hv.detach()

def cos_angle(a, b):
    return float((a@b)/(a.norm()*b.norm()+1e-10))

def probe_hessian(model, n_iter=6):
    """Power iteration for λ_max, deflated for λ_min."""
    v=torch.randn(sum(p.numel() for p in model.parameters()))
    v=v/v.norm()
    for _ in range(n_iter):
        Hv=hvp(model,v,n=4); v=Hv/max(Hv.norm().item(),1e-10)
    lam_max=float(v@hvp(model,v,n=4))
    # deflated iteration for λ_min
    sigma=lam_max*1.05
    v2=torch.randn_like(v); v2=v2/v2.norm()
    for _ in range(n_iter):
        Hv2=hvp(model,v2,n=4)-sigma*v2; v2=Hv2/max(Hv2.norm().item(),1e-10)
    lam_min=float(v2@hvp(model,v2,n=4))
    return lam_min, lam_max

def sheet_indicator(model):
    """Z/2Z sheet: sign of trace(WV[1]).
    After TopoGate (WV.mul_(-1)): trace flips sign.
    trace(WV[1]) > 0 → correct sheet (+1)
    trace(WV[1]) < 0 → wrong sheet (-1), needs TopoGate."""
    wv=model.blocks[1].attn.WV.weight.data
    return 1 if float(wv.trace())>0 else -1

def kv_accumulation_error(model, delta_theta, n=8):
    """KV/2nd-order accumulation error: ||∇L - H·Δθ|| / ||∇L||
    Measures how much the 1st-order update differs from Newton.
    Large = Adam is accumulating curvature error."""
    if delta_theta.norm().item() < 1e-10: return float('nan')
    g=get_gradient(model,n=n)
    Hd=hvp(model,delta_theta,n=n)
    error=float((g-Hd).norm())/max(float(g.norm()),1e-10)
    return error

def twist_check(model, delta_theta):
    """cos(g, Δθ): should be negative (update opposes gradient = descent).
    If positive: update is ALIGNED with gradient = TWIST ERROR (not descending)."""
    g=get_gradient(model,n=6)
    return cos_angle(g, delta_theta)

# ── PROFILER RECORD ───────────────────────────────────────────
records = []
theta_init = None

def record(label, model, step, delta_theta=None, extra=''):
    global theta_init
    v   = eval_val(model)
    g   = get_gradient(model, n=10)
    gn  = float(g.norm())
    E   = model.te.weight.data.flatten()
    cos_gE = cos_angle(g[:E.numel()], E)   # cos(∇_E L, E)
    sheet  = sheet_indicator(model)
    theta  = model.flat_params()
    if theta_init is None: theta_init=theta.clone()
    basin_d= float((theta-theta_init).norm())

    # Hessian (expensive — do every ~5 steps only)
    do_hess = (step % 5 == 0) or step <= 2 or 'Pass' in label
    if do_hess:
        lmin,lmax=probe_hessian(model,n_iter=5)
        ridge='CROSSED' if lmin>-0.05 else 'SADDLE'
    else:
        lmin=lmax=float('nan'); ridge='...'

    twist=float('nan'); kv_err=float('nan')
    if delta_theta is not None and delta_theta.norm().item()>1e-8:
        twist=twist_check(model,delta_theta)
        kv_err=kv_accumulation_error(model,delta_theta,n=6)

    rec=dict(label=label,step=step,val=v,gnorm=gn,cos_gE=cos_gE,
             lmin=lmin,lmax=lmax,ridge=ridge,basin=basin_d,
             sheet=sheet,twist=twist,kv_err=kv_err,extra=extra)
    records.append(rec)

    # Print compact line
    hess_str=(f"λ[{lmin:+.3f},{lmax:+.3f}] {ridge}"
              if do_hess else f"λ[...,...]")
    twist_str=f" twist={twist:+.3f}" if not math.isnan(twist) else ""
    kv_str   =f" kv={kv_err:.3f}" if not math.isnan(kv_err) else ""
    print(f"  {label:<22} step={step:3d}  val={v:.4f}  "
          f"gnorm={gn:.3f}  cos(g,E)={cos_gE:+.3f}  "
          f"sheet={sheet:+d}  {hess_str}{twist_str}{kv_err and kv_str or ''}")
    return v

# ═══════════════════════════════════════════════════════════════
# CORPUS ALGEBRA
# ═══════════════════════════════════════════════════════════════
print("="*75)
print("AU-FUKAYA GEOMETRY PROFILER")
print("="*75); print()
print("Building corpus statistics...")
bigram=collections.Counter()
perm={}
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

# J14 scale
np.random.seed(42)
dot_next=[float(E_0[t]@E_0[nt]) for t,nt in list(perm.items())[:300]]
dot_rnd =[float(E_0[t]@E_0[np.random.randint(VOCAB)])
          for t in list(perm.keys())[:300] for _ in range(10)]
e_gap=max(np.mean(dot_next)-np.mean(dot_rnd),1e-6)
dh=D//N_HEADS
scale_wk=5.0*math.sqrt(dh)/e_gap
print(f"  E-gap={e_gap:.5f}, W_K* scale={scale_wk:.1f}")

# ═══════════════════════════════════════════════════════════════
# ARM A: STANDARD GRADIENT DESCENT (50 steps)
# ═══════════════════════════════════════════════════════════════
print()
print("─"*75)
print("ARM A: STANDARD GRADIENT DESCENT (50 steps)")
print("─"*75)
print(f"  {'Label':<22} {'step':>4}  {'val':>6}  {'gnorm':>6}  "
      f"{'cos(g,E)':>9}  {'sheet':>5}  {'Hessian + ridge':>30}  twist  kv_err")
print("  "+"-"*73)

torch.manual_seed(99)
gd=LM(); gd.te.weight.data.copy_(torch.tensor(E_init))
theta_init=None   # reset for GD arm
record("GD_init", gd, 0)
theta_prev=gd.flat_params().clone()

opt=torch.optim.AdamW(gd.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
gd_vals={}
for step in range(1,51):
    gd.train(); x,y=get_batch(); _,l=gd(x,y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(gd.parameters(),1.0); opt.step()
    if step in {1,5,10,15,20,25,30,35,40,45,50}:
        theta_cur=gd.flat_params().clone()
        delta=theta_cur-theta_prev
        v=record(f"GD_step{step}", gd, step, delta_theta=delta)
        gd_vals[step]=v
        theta_prev=theta_cur.clone()

# ═══════════════════════════════════════════════════════════════
# ARM B: COMPILER ALGEBRAIC PASSES
# ═══════════════════════════════════════════════════════════════
print()
print("─"*75)
print("ARM B: COMPILER ALGEBRAIC PASSES")
print("─"*75)
print(f"  {'Label':<22} {'step':>4}  {'val':>6}  {'gnorm':>6}  "
      f"{'cos(g,E)':>9}  {'sheet':>5}  {'Hessian + ridge':>30}  twist  kv_err")
print("  "+"-"*73)

torch.manual_seed(99)
comp=LM(); comp.te.weight.data.copy_(torch.tensor(E_init))
theta_init=None   # reset for compiler arm

record("Pass0_spectral", comp, 0, extra="E₀+prebake")
theta_after_init=comp.flat_params().clone()

# Pass 3: TopoGate FIRST (confirmed missing from compiler — sheet=-1 throughout)
theta_before=comp.flat_params().clone()
with torch.no_grad():
    for l_idx in [1, 2]:
        comp.blocks[l_idx].attn.WV.weight.mul_(-1)
        comp.blocks[l_idx].attn.op.weight.mul_(-1)
delta_topo=comp.flat_params()-theta_before
record("PassTopo_gate", comp, 0, delta_theta=delta_topo, extra="Z/2Z sign flip")

# Pass J14: W_K* = scale×I (after TopoGate)
WK_star=scale_wk*torch.eye(D)
theta_before=comp.flat_params().clone()
with torch.no_grad():
    for bl in comp.blocks:
        bl.attn.WK.weight.data.copy_(WK_star)
        bl.attn.WQ.weight.data.copy_(WK_star)
delta_wk=comp.flat_params()-theta_before
record("PassJ14_WK*", comp, 0, delta_theta=delta_wk, extra="W_K*=scale×I")

# Pass MF: E-descent / W_K-ascent ×10
print("  [Running MF pump ×10 — 200 batches each...]")
n_sub=200
for mf_i in range(1,11):
    for _ in range(n_sub):
        comp.train(); x,y=get_batch(); _,loss=comp(x,y)
        comp.zero_grad(); loss.backward()
        with torch.no_grad():
            if comp.te.weight.grad is not None:
                comp.te.weight.data -= ETA_MF*comp.te.weight.grad
    for _ in range(n_sub):
        comp.train(); x,y=get_batch(); _,loss=comp(x,y)
        comp.zero_grad(); loss.backward()
        with torch.no_grad():
            for bl in comp.blocks:
                if bl.attn.WK.weight.grad is not None:
                    bl.attn.WK.weight.data += ETA_MF*bl.attn.WK.weight.grad
    if mf_i in {5,10}:
        delta_mf=comp.flat_params()-theta_before
        record(f"PassMF_{mf_i}", comp, mf_i*2, delta_theta=delta_mf,
               extra=f"MF iter {mf_i}")
        theta_before=comp.flat_params().clone()

# Pass Basin: 33 CE at LR×5 with warmup
print("  [Running basin selector 33 CE...]")
theta_before=comp.flat_params().clone()
opt_b=torch.optim.AdamW(comp.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,34):
    lr_cur=LR*5*min(step,10)/10
    for pg in opt_b.param_groups: pg['lr']=lr_cur
    comp.train(); x,y=get_batch(); _,l=comp(x,y)
    opt_b.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(comp.parameters(),1.0); opt_b.step()
    if step in {10,20,33}:
        delta_b=comp.flat_params()-theta_before
        record(f"PassBasin_{step}CE", comp, step, delta_theta=delta_b,
               extra=f"Basin CE {step}")
        theta_before=comp.flat_params().clone()

# Pass LM: 8 iterations matching build_pass6 exactly
print("  [Running LM 8 iters...]")
mu=0.95; N_HVP=12; N_CG=6; N_GRAD=25
for lm_i in range(1,9):
    theta_before=comp.flat_params().clone()
    comp.zero_grad()
    loss=sum(comp(*get_batch())[1] for _ in range(N_GRAD))/N_GRAD
    loss.backward()
    g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                 for p in comp.parameters()]).detach()
    comp.zero_grad()
    def _hvp(v):
        comp.zero_grad()
        loss2=sum(comp(*get_batch())[1] for _ in range(N_HVP))/N_HVP
        grads=torch.autograd.grad(loss2,list(comp.parameters()),create_graph=True)
        gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
        hv=torch.cat([h.flatten() for h in
                      torch.autograd.grad(gv,list(comp.parameters()),retain_graph=False)])
        comp.zero_grad(); return hv.detach()
    d=torch.zeros_like(g); r=-g.clone(); p_cg=r.clone(); rr=float((r*r).sum())
    for _ in range(N_CG):
        Hp=_hvp(p_cg)+mu*p_cg; alpha=rr/max(float((p_cg*Hp).sum()),1e-10)
        d+=alpha*p_cg; r-=alpha*Hp; rr_new=float((r*r).sum())
        p_cg=r+(rr_new/max(rr,1e-10))*p_cg; rr=rr_new
    L0=eval_val(comp,n=8)
    comp.set_flat(theta_before+d); L_new=eval_val(comp,n=8)
    if L_new<L0:
        mu=max(mu*0.5,0.95); status='ACCEPT'
    else:
        comp.set_flat(theta_before); mu=min(mu*2,5.0); status='REJECT'
    delta_lm=comp.flat_params()-theta_before
    record(f"PassLM_{lm_i}", comp, lm_i, delta_theta=delta_lm,
           extra=f"LM {lm_i} {status} μ={mu:.3f}")

# ═══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════
print()
print("="*75)
print("SIDE-BY-SIDE: COMPILER POSITION vs GD AT EQUAL CE EQUIV")
print("="*75)
comp_recs=[r for r in records if 'Pass' in r['label']]
gd_recs  =[r for r in records if 'GD_'  in r['label']]

print(f"\n  {'Compiler pass':<22}  {'val':>6}  {'GD equiv step':>14}  "
      f"{'GD val':>7}  {'advantage':>9}")
print("  "+"-"*65)
ce_map={'Pass0_spectral':0,'PassJ14_WK*':0,'PassMF_5':10,'PassMF_10':20,
        'PassBasin_10CE':30,'PassBasin_20CE':40,'PassBasin_33CE':53}
for r in comp_recs:
    ce=ce_map.get(r['label'],None)
    if ce is None: continue
    gd_v=gd_vals.get(min(ce,50),gd_vals.get(50,0))
    adv=gd_v-r['val']
    print(f"  {r['label']:<22}  {r['val']:>6.4f}  "
          f"  GD@{ce:2d} CE: {gd_v:>6.4f}   {adv:>+9.4f}")

print()
print("="*75)
print("GEOMETRIC DIAGNOSTICS SUMMARY")
print("="*75)
print()
print("  TWIST ERRORS (cos(g,Δθ) > 0 = update aligns with gradient = wrong direction):")
twist_errs=[r for r in records if not math.isnan(r['twist']) and r['twist']>0.1]
if twist_errs:
    for r in twist_errs:
        print(f"    {r['label']}: twist={r['twist']:+.3f}  val={r['val']:.4f}")
else:
    print("    None detected (all updates oppose gradient correctly)")

print()
print("  RIDGE CROSSINGS (λ_min changes sign):")
prev_lmin=None
for r in records:
    if math.isnan(r['lmin']): continue
    if prev_lmin is not None and prev_lmin<0 and r['lmin']>-0.05:
        print(f"    {r['label']} step={r['step']}: "
              f"λ_min {prev_lmin:.3f}→{r['lmin']:.3f}  RIDGE CROSSED  val={r['val']:.4f}")
    prev_lmin=r['lmin']

print()
print("  SHEET CORRECTIONS:")
prev_sheet=None
for r in records:
    if prev_sheet is not None and r['sheet']!=prev_sheet:
        print(f"    {r['label']}: sheet {prev_sheet:+d}→{r['sheet']:+d}  val={r['val']:.4f}")
    prev_sheet=r['sheet']

print()
print("  KV ACCUMULATION ERROR (2nd-order defect, higher=worse):")
for r in records:
    if not math.isnan(r['kv_err']):
        flag='⚠' if r['kv_err']>2.0 else ' '
        print(f"    {flag} {r['label']:<22}: kv_err={r['kv_err']:.3f}  val={r['val']:.4f}")

print()
print("  COS(g,E) ANTI-ALIGNMENT TRAJECTORY (cos<-0.3 = anti-aligned):")
for r in records:
    flag='⚠ ANTI-ALIGNED' if r['cos_gE']<-0.3 else ('✓ aligned' if r['cos_gE']>0.2 else '~ neutral')
    print(f"    {r['label']:<22}: cos={r['cos_gE']:+.3f}  {flag}")

print()
print("  BASIN DEPTH (||θ - θ_init||):")
for r in records:
    print(f"    {r['label']:<22}: basin_depth={r['basin']:.2f}  val={r['val']:.4f}")
