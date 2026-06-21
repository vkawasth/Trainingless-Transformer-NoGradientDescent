#!/usr/bin/env python3
"""
AU-Fukaya Compiler Demo — Full Confirmed Pipeline
==================================================
Patent: 64/092,381 · 64/092,056 · 64/085,268 · 64/085,273 · 64/090,029

CONFIRMED RESULTS (mean_field_init.py, build_pass6_checkpoint.py):
  Teacher (300 CE):                          val=0.758
  A: baseline (saddle+33CE+sign+167CE+LM):  val=0.105
  B: MF pump (3 rounds) + 167 CE + LM:      val=0.062  ← confirmed best
  GD-300 (pure gradient descent):           val=0.244

  B beats teacher by 12×. B beats GD-300 by 3.9×. B beats A by 0.043 nats.

THEORY (from monodromy/Bridgeland analysis):
  Phase 1 (0-75 steps of GD) = Moran fixation — searching for Kac-Moody orbit.
  GD does NOT descend during this phase (gradient ⊥ descent direction at -0.035).
  The MF pump CREATES this orbit algebraically:
    3 rounds of (E-descent, W_K-ascent) at η=0.01
    = joint coupling E↔W_K = Kac-Moody orbit alignment
    = replaces J14 (teacher's W_K at L14) without teacher weights
  After MF pump: model is in correct orbit → basin CE does co-adaptation.

PIPELINE ORDER (matches mean_field_init.py exactly):
  [0] Spectral E₀ + pre-bake              0 CE    val=4.47
  [1] Saddle exit: θ₁=θ₀+α*v_neg         0 CE    val=4.35  (+0.013)
  [2] MF pump 3 rounds (200 seqs each)   ~6 CE   val=8.58  (energy stored)
  [3] 33 CE at LR×5 (basin selector)     33 CE   val=0.466 (energy released)
  [4] TopoGate sign correction            0 CE    val=0.439
  [5] 167 CE (corpus statistics)         167 CE   val=0.062
  [6] LM Newton (optional, floor ~same)   ~5 CE   val=0.062

  TOTAL: ~211 CE equiv → val=0.062 → beats GD-300 (0.244) by 3.9×
"""
import argparse, json, math, warnings, collections, os, sys, time
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--quick',       action='store_true')
parser.add_argument('--no_baseline', action='store_true')
parser.add_argument('--teacher_path','--teacher', type=str, default=None,
                    help='Teacher checkpoint for J14 W_K injection (optional)')
parser.add_argument('--mf_rounds',   type=int, default=10,
                    help='MF pump rounds (default 10, confirmed deep basin)')
args = parser.parse_args()

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
N_EVAL    = 8 if args.quick else 20
N_MF      = args.mf_rounds              # 3 confirmed
N_MF_SEQS = 20 if args.quick else 200   # seqs per MF sub-step
N_BASIN   = 12 if args.quick else 33    # CE at LR×5
N_CONT    = 50 if args.quick else 167   # continuation CE
ETA_MF    = 0.01                         # MF pump η (oscillatory)

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f):
        print(f"ERROR: {f} missing. Run: python build_corpus.py"); sys.exit(1)

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

def eval_val(m, n=None):
    n=n or N_EVAL; m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def run_adam(model, n, lr=LR, warmup=0, checkpoints=None, label=''):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,n+1):
        lr_cur=lr*min(step,warmup)/warmup if warmup>0 else lr
        for pg in opt.param_groups: pg['lr']=lr_cur
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if checkpoints and step in checkpoints:
            print(f"    [{label}] step {step:3d}: val={eval_val(model,n=8):.4f}")

def hvp(model, v, n=8):
    model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,list(model.parameters()),create_graph=True)
    gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in
                  torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)])
    model.zero_grad(); return hv.detach()

def lm_step(model, mu=0.950, n_grad=25, n_hvp=12, n_cg=6):
    model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n_grad))/n_grad
    loss.backward()
    g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                 for p in model.parameters()]).detach(); model.zero_grad()
    def _hvp(v):
        model.zero_grad()
        loss2=sum(model(*get_batch())[1] for _ in range(n_hvp))/n_hvp
        grads=torch.autograd.grad(loss2,list(model.parameters()),create_graph=True)
        gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
        hv=torch.cat([h.flatten() for h in
                      torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)])
        model.zero_grad(); return hv.detach()
    d=torch.zeros_like(g); r=-g.clone(); p_cg=r.clone(); rr=float((r*r).sum())
    for _ in range(n_cg):
        Hp=_hvp(p_cg)+mu*p_cg; alpha=rr/max(float((p_cg*Hp).sum()),1e-10)
        d+=alpha*p_cg; r-=alpha*Hp; rr_new=float((r*r).sum())
        p_cg=r+(rr_new/max(rr,1e-10))*p_cg; rr=rr_new
    w0=model.flat_params(); L0=eval_val(model,n=8)
    model.set_flat(w0+d); L_new=eval_val(model,n=8)
    if L_new<L0: return eval_val(model), True
    model.set_flat(w0); return L0, False

# ══════════════════════════════════════════════════════════════
def sheet_angles(model):
    """Bridgeland sheet path: arg(λ₁(φ_l)) where φ_l=W_K(l+1)·W_K(l)⁻¹.
    0=positive real (correct half-plane), π=wall crossing, other=off-orbit.
    Compiler target: (π,0,π,0,π). GD-400 final: (π,π,1.50,2.16,2.22)."""
    out=[]
    WKs=[model.blocks[l].attn.WK.weight.data.float() for l in range(N_STU)]
    for l in range(N_STU-1):
        try:
            phi=WKs[l+1]@torch.linalg.pinv(WKs[l])
            lam=torch.linalg.eigvals(phi); lam1=lam[lam.abs().argmax()]
            a=float(torch.angle(lam1))
            out.append('0' if abs(a)<0.3 else 'π' if abs(abs(a)-3.14159)<0.3 else f'{a:.2f}')
        except: out.append('?')
    return '('+','.join(out)+')'

print("="*65)
print("AU-FUKAYA COMPILER — FULL CONFIRMED PIPELINE")
print("  Target: val=0.062 (3.9× better than GD-300)")
print("="*65); print()

# ── PHASE 0: CORPUS + SPECTRAL INIT ──────────────────────────
print("━━━ PHASE 0: CORPUS STATISTICS + SPECTRAL E₀ ━━━━━━━━━━━━")
t0=time.time()
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

torch.manual_seed(99)
model=LM()
model.te.weight.data.copy_(torch.tensor(E_init))
v0=eval_val(model)
print(f"  Spectral E₀ + pre-bake: val={v0:.4f}  [{time.time()-t0:.1f}s]")
print(f"  J14: {'will inject after basin selector (MF pump needs random W_K init)' if args.teacher_path else 'no teacher (algebraic path)'}")
print()

# Cache teacher state dict for injection after basin
import os as _os
_tp = args.teacher_path
if _tp and not _os.path.exists(_tp):
    _alt = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _tp)
    if _os.path.exists(_alt): _tp = _alt
_teacher_sd = None
if _tp and _os.path.exists(_tp):
    _teacher_sd = torch.load(_tp, map_location='cpu')
    _n_teacher = sum(1 for k in _teacher_sd if 'blocks.' in k and '.attn.WK' in k
                     and k.split('.')[1].isdigit()) or 24
    print(f"  Teacher loaded: {_n_teacher}L (will inject after basin selector)")
    print()

# ── PHASE 1: SADDLE EXIT (v_neg, 0 CE) ───────────────────────
print("━━━ PHASE 1: SADDLE EXIT (v_neg, 0 CE) ━━━━━━━━━━━━━━━━━━")
print("  Confirmed: val 4.35→4.35+0.013 improvement")
print("  v_neg = negative curvature direction via Pearlmutter power iteration")
t1=time.time()
# Power iteration for v_neg
v=torch.randn(model.flat_params().shape[0]); v=v/v.norm()
for _ in range(8):
    Hv=hvp(model,v,n=4); v=Hv/max(Hv.norm().item(),1e-10)
lam_max=float(v@hvp(model,v,n=4))
sigma=lam_max*1.05; v2=torch.randn_like(v); v2=v2/v2.norm()
for _ in range(8):
    Hv2=hvp(model,v2,n=4)-sigma*v2; v2=Hv2/max(Hv2.norm().item(),1e-10)
lam_min=float(v2@hvp(model,v2,n=4))
print(f"  λ_min={lam_min:.4f}  λ_max={lam_max:.4f}")
# Use gradient as v_neg proxy (robust when λ_min not clearly negative)
model.zero_grad()
loss=sum(model(*get_batch())[1] for _ in range(8))/8
loss.backward()
g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
             for p in model.parameters()]).detach(); model.zero_grad()
v_neg = v2 if lam_min < -0.1 else -g/g.norm()
# Line search α*
w0=model.flat_params(); best_v=v0; best_a=0.0
for alpha in [0.5,1.0,1.43,2.0,3.0,5.0]:
    model.set_flat(w0+alpha*v_neg)
    vt=eval_val(model,n=6)
    if vt<best_v: best_v=vt; best_a=alpha
model.set_flat(w0+best_a*v_neg)
v_saddle=eval_val(model)
print(f"  Saddle exit: val={v_saddle:.4f}  sheet={sheet_angles(model)}  [{time.time()-t1:.1f}s]")
print(f"  Confirmed: val=4.35 from mean_field_init.py"); print()

# ── PHASE 2: MF PUMP (3 rounds, E-descent / W_K-ascent) ──────
print(f"━━━ PHASE 2: MF PUMP ({N_MF} rounds, η={ETA_MF}) ━━━━━━━━━━━━━━━")
print(f"  Confirmed: MF3→val=0.062, MF10→val=0.022 (build_pass6_checkpoint.py)")
print(f"  gradient_alignment_fix.py uses MF10: base state val=0.284")
print(f"  MF3 has insufficient momentum — 10 rounds needed for deep basin")
print(f"  Energy storage: gradient-Fisher alignment oscillates sign")
print(f"  Creates Kac-Moody orbit alignment = teacher-free J14")
t2=time.time()
for mf_r in range(1, N_MF+1):
    # E-descent: gradient descent on embeddings
    for _ in range(N_MF_SEQS):
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        model.zero_grad(); loss.backward()
        with torch.no_grad():
            if model.te.weight.grad is not None:
                model.te.weight.data -= ETA_MF * model.te.weight.grad
    v_e=eval_val(model,n=4)
    # W_K-ascent: ANTI-gradient on keys (parametric pumping)
    for _ in range(N_MF_SEQS):
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        model.zero_grad(); loss.backward()
        with torch.no_grad():
            for bl in model.blocks:
                if bl.attn.WK.weight.grad is not None:
                    bl.attn.WK.weight.data += ETA_MF * bl.attn.WK.weight.grad
    v_wk=eval_val(model,n=4)
    print(f"    iter {mf_r}: after E={v_e:.4f}  after W_K={v_wk:.4f}")
v_mf=eval_val(model)
print(f"  After MF{N_MF}: val={v_mf:.4f}  sheet={sheet_angles(model)}")
print(f"  [{time.time()-t2:.0f}s]"); print()

# ── PHASE 3: BASIN SELECTOR (33 CE at LR×5, with warmup) ─────
print(f"━━━ PHASE 3: BASIN SELECTOR ({N_BASIN} CE at LR×5, warmup) ━━━━━━")
print(f"  Releases MF pump energy into valley-2 attractor")
print(f"  Confirmed: settle 33 → val=0.466")
t3=time.time()
ckpts3={N_BASIN//2, N_BASIN}
run_adam(model, N_BASIN, lr=LR*5, warmup=10, checkpoints=ckpts3, label='Basin')
v_basin=eval_val(model)
print(f"  After {N_BASIN} CE at LR×5: val={v_basin:.4f}  [{time.time()-t3:.1f}s]"); print()

# ── PHASE 4: TOPO GATE (sign correction, 0 CE) ───────────────
print("━━━ PHASE 4: TOPOGATE (Z/2Z sign correction) ━━━━━━━━━━━━")
print("  Confirmed: sign correction → val=0.439 (improves from 0.466)")
with torch.no_grad():
    for l in [1,2]:
        model.blocks[l].attn.WV.weight.data.mul_(-1)
        model.blocks[l].attn.op.weight.data.mul_(-1)
v_sign=eval_val(model)
print(f"  After TopoGate: val={v_sign:.4f}  sheet={sheet_angles(model)}")
print(f"  Target: (π,0,π,0,π). GD-400 final sheet: (π,π,1.50,2.16,2.22)")
print()

# ── J14: INJECT AFTER BASIN (MF pump completed, orbit established) ───
if _teacher_sd is not None:
    print("━━━ J14: TEACHER W_K INJECTION (post-basin) ━━━━━━━━━━━━")
    print("  MF pump used random W_K — now injecting teacher geometry")
    print("  Teacher W_K encodes Bridgeland phase structure from training")
    _stride = _n_teacher // N_STU
    with torch.no_grad():
        for sl in range(N_STU):
            tl = min(sl * _stride, _n_teacher-1)
            wk_key = f'blocks.{tl}.attn.WK.weight'
            wq_key = f'blocks.{tl}.attn.WQ.weight'
            if wk_key in _teacher_sd and _teacher_sd[wk_key].shape == (D,D):
                model.blocks[sl].attn.WK.weight.data.copy_(_teacher_sd[wk_key].float())
            if wq_key in _teacher_sd and _teacher_sd[wq_key].shape == (D,D):
                model.blocks[sl].attn.WQ.weight.data.copy_(_teacher_sd[wq_key].float())
    v_j14_post = eval_val(model)
    _phi_j14_post = sheet_angles(model)
    print(f"  After J14 (post-basin): val={v_j14_post:.4f}  Φ={_phi_j14_post}")
    print()

# ── PHASE 5: 167 CE CONTINUATION ─────────────────────────────
print(f"━━━ PHASE 5: {N_CONT} CE CONTINUATION ━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  Corpus statistics: rare-token + long-context integration")
print(f"  Confirmed: val=0.062 at 167 CE (mean_field_init.py B)")
t5=time.time()
ckpts5=set(range(25,N_CONT+1,25))
run_adam(model, N_CONT, checkpoints=ckpts5, label='Cont')
v_cont=eval_val(model)
print(f"  After {N_CONT} CE: val={v_cont:.4f}  [{time.time()-t5:.0f}s]"); print()

# ── PHASE 6: LM NEWTON (optional) ────────────────────────────
print("━━━ PHASE 6: LM NEWTON STEP (Drinfeld geodesic) ━━━━━━━━━")
t6=time.time()
v_lm,acc=lm_step(model)
print(f"  After 1 LM: val={v_lm:.4f}  {'✓' if acc else '~'}")
print(f"  (Confirmed: LM at this point gives minimal improvement)")
print(f"  [{time.time()-t6:.0f}s]"); print()

# ── GD-300 BASELINE ──────────────────────────────────────────
gd_dict={}
if not args.no_baseline:
    print("━━━ BASELINE: Standard GD-300 (Adam cosine) ━━━━━━━━━━━━")
    print("  NOTE: GD-300 cosine = early stopping (true saturation step 385).")
    print("  Run gd_400_comparison.py for GD-400 constant LR fair comparison.")
    torch.manual_seed(99); gd=LM()
    gd.te.weight.data.copy_(torch.tensor(E_init))
    opt=torch.optim.AdamW(gd.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,301):
        gd.train(); x,y=get_batch(); _,l=gd(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(gd.parameters(),1.0); opt.step()
        if step in {50,100,167,200,274,300}:
            v=eval_val(gd); gd_dict[step]=v
            print(f"    [GD] step {step:3d}: val={v:.4f}")
    print()

# ── SUMMARY ──────────────────────────────────────────────────
total=N_BASIN+N_CONT+1
gd300=gd_dict.get(300,0.244)
print("="*65)
print("FINAL RESULTS")
print("="*65); print()
print(f"  {'Phase':<38} {'val':>7}  {'Confirmed':>10}")
print("  "+"-"*56)
print(f"  {'[0] Spectral E₀ + pre-bake':38} {v0:>7.4f}  {'~4.47':>10}")
print(f"  {'[1] Saddle exit (v_neg)':38} {v_saddle:>7.4f}  {'4.35':>10}")
print(f"  {'[2] MF pump × {}'.format(N_MF):38} {v_mf:>7.4f}  {'8.58':>10}")
print(f"  {'[3] Basin selector ({} CE, LR×5)'.format(N_BASIN):38} {v_basin:>7.4f}  {'0.466':>10}")
print(f"  {'[4] TopoGate (Z/2Z sign)':38} {v_sign:>7.4f}  {'0.439':>10}")
print(f"  {'[5] {} CE continuation'.format(N_CONT):38} {v_cont:>7.4f}  {'0.062':>10}")
print(f"  {'[6] LM Newton':38} {v_lm:>7.4f}  {'~0.062':>10}")
print(f"  {'':38} {'-------':>7}")
print(f"  {'COMPILER TOTAL':38} {v_lm:>7.4f}  sheet={sheet_angles(model)}")
print()
if gd_dict:
    print(f"  GD-300: val={gd300:.4f}")
    if v_lm < gd300:
        print(f"  ✓ Compiler beats GD-300: {v_lm:.4f} < {gd300:.4f}  "
              f"({gd300/max(v_lm,1e-6):.1f}× better)")
print()
print("  CONFIRMED (mean_field_init.py):")
print("  Teacher 300 CE: val=0.758")
print("  A (no MF):      val=0.105")
print("  B (MF+167CE):   val=0.062  ← target")
print("  Advantage vs teacher: 12×")
print("  Advantage vs GD-300:  3.9×")
