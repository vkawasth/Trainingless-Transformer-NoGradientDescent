#!/usr/bin/env python3
"""
MF Pump Sweep — Finding the Resonant Peak
==========================================
Tests N_MF ∈ {1,2,3,5,7,10,15} at fixed N_SUB=200, η=0.01
Tracks: post-basin val, Fisher condition number κ_F, gluing defect τ

THEORY: MF pump = topological pulse, not training
  Over-pumping: κ_F spikes (ill-conditioned natural gradient)
  Under-pumping: orbit not fully established (low Φ_clean)
  Resonant peak: lowest post-basin val, stable κ_F, τ decreasing

METRICS:
  κ_F = σ_max(F) / σ_min(F)   Fisher condition number
        F = E[g⊗g], diagonal approximation: κ_F = max(fish)/min(fish)
  τ    = ||∇_FF L|| / ||∇_Emb L||   gluing defect (K₀ extension class)
  Φ_cl = count of ϕ_l ∈ {0,π}       orbit cleanliness

ANNEALED PUMP (if peak < MF3):
  η(r) = η₀ × α^r   (decaying amplitude)
  Tested if flat sweep shows peak at r<3
"""
import json, math, warnings, collections, os, sys, time
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
ETA_MF=0.01; N_SUB=200

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f): print(f"ERROR: {f}"); sys.exit(1)

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

def sheet_angles(model):
    out=[]; WKs=[model.blocks[l].attn.WK.weight.data.float() for l in range(N_STU)]
    for l in range(N_STU-1):
        try:
            phi=WKs[l+1]@torch.linalg.pinv(WKs[l])
            lam=torch.linalg.eigvals(phi)
            most_neg=lam[lam.real.argmin()]
            lam1=most_neg if float(most_neg.real)<-0.9 else lam[lam.abs().argmax()]
            a=float(torch.angle(lam1))
            out.append('π' if abs(abs(a)-math.pi)<0.3 else '0' if abs(a)<0.3 else f'{a:.2f}')
        except: out.append('?')
    return out

def fisher_kappa(fish_diag):
    """κ_F = max(diag(F)) / min(diag(F)) — diagonal Fisher condition number."""
    f=fish_diag.flatten(); f=f[f>1e-10]
    if len(f)<2: return float('nan')
    return float(f.max()/f.min())

def gluing_defect(model, n=8):
    """τ = ||∇_FF|| / ||∇_Emb|| — K₀ extension class / H¹ defect proxy."""
    model.zero_grad()
    ls=[model(*get_batch())[1] for _ in range(n)]
    torch.stack(ls).mean().backward()
    g_ff=sum(p.grad.data.norm().item() for nm,p in model.named_parameters()
             if '.ff.' in nm and p.grad is not None)
    g_emb=model.te.weight.grad.data.norm().item() if model.te.weight.grad is not None else 1e-8
    model.zero_grad()
    return g_ff/max(g_emb,1e-8)

def run_mf_pump(model, n_mf, eta=ETA_MF, n_sub=N_SUB, annealed=False, anneal_rate=0.8):
    """
    Natural gradient MF pump.
    annealed=True: η(r) = η × anneal_rate^r  (decaying amplitude)
    Returns per-round diagnostics: (val_E, val_WK, kappa_F, tau, phi_clean)
    """
    diagnostics = []
    for mf_r in range(1, n_mf+1):
        eta_r = eta * (anneal_rate**(mf_r-1)) if annealed else eta

        # Step 1: Natural gradient E-descent, W_K frozen
        for l in range(N_STU):
            model.blocks[l].attn.WK.weight.requires_grad_(False)
            model.blocks[l].attn.WQ.weight.requires_grad_(False)
        emb_grad=torch.zeros(model.te.weight.shape)
        emb_fish=torch.zeros(model.te.weight.shape)
        torch.manual_seed(mf_r*1000)
        for i in range(n_sub):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            model.zero_grad(); _,loss=model(x,y); loss.backward()
            if model.te.weight.grad is not None:
                g=model.te.weight.grad.detach(); emb_grad+=g; emb_fish+=g**2
        emb_grad/=n_sub; emb_fish/=n_sub
        kappa_E = fisher_kappa(emb_fish)
        delta_E=-(emb_grad/(emb_fish+1e-4))
        with torch.no_grad(): model.te.weight.add_(eta_r*delta_E)
        for l in range(N_STU):
            model.blocks[l].attn.WK.weight.requires_grad_(True)
            model.blocks[l].attn.WQ.weight.requires_grad_(True)
        v_e=eval_val(model,n=6)

        # Step 2: Natural gradient W_K-descent, E frozen
        model.te.weight.requires_grad_(False)
        wk_grad=torch.zeros_like(model.blocks[0].attn.WK.weight)
        wk_fish=torch.zeros_like(model.blocks[0].attn.WK.weight)
        torch.manual_seed(mf_r*1000+500)
        for i in range(n_sub):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            model.zero_grad(); _,loss=model(x,y); loss.backward()
            g=torch.zeros_like(model.blocks[0].attn.WK.weight)
            for bl in model.blocks:
                if bl.attn.WK.weight.grad is not None: g+=bl.attn.WK.weight.grad/N_STU
            wk_grad+=g; wk_fish+=g**2
        wk_grad/=n_sub; wk_fish/=n_sub
        kappa_WK = fisher_kappa(wk_fish)
        delta_WK=-(wk_grad/(wk_fish+1e-4))
        with torch.no_grad():
            for l in range(N_STU):
                model.blocks[l].attn.WK.weight.add_(eta_r*delta_WK)
                model.blocks[l].attn.WQ.weight.add_(eta_r*(-(wk_grad.T/(wk_fish.T+1e-4))))
        model.te.weight.requires_grad_(True)
        v_wk=eval_val(model,n=6)

        # Diagnostics at end of round
        tau=gluing_defect(model,n=6)
        phi=sheet_angles(model)
        phi_clean=sum(1 for p in phi if p in ('0','π'))
        kappa_avg=(kappa_E+kappa_WK)/2

        diagnostics.append(dict(r=mf_r, v_e=v_e, v_wk=v_wk,
                                kappa=kappa_avg, tau=tau,
                                phi_clean=phi_clean, eta_r=eta_r))
        print(f"    MF{mf_r:2d}: E={v_e:.3f} WK={v_wk:.3f}  "
              f"κ_F={kappa_avg:.1f}  τ={tau:.3f}  Φ_cl={phi_clean}/5  η={eta_r:.4f}")
    return diagnostics

def run_basin_and_settle(model, n_basin=33):
    """Basin selector + TopoGate. Returns post-basin val."""
    opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,n_basin+1):
        lr_c=LR*5*min(step,10)/10
        for pg in opt.param_groups: pg['lr']=lr_c
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    # TopoGate
    with torch.no_grad():
        for l in [1,2]:
            model.blocks[l].attn.WV.weight.data.mul_(-1)
            model.blocks[l].attn.op.weight.data.mul_(-1)
    return eval_val(model)

# ── CORPUS + E₀ ───────────────────────────────────────────────
print("="*70)
print("MF PUMP SWEEP — Finding the Resonant Peak")
print("Tracking κ_F (Fisher condition), τ (gluing defect), Φ_clean")
print("="*70); print()

bigram=collections.Counter(); perm={}
for i in range(len(train_ids)-1):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB: bigram[(a,b)]+=1; perm.setdefault(a,b)
rows,cols,vv=[],[],[]
for (a,b),cnt in bigram.items(): rows.append(a);cols.append(b);vv.append(float(cnt))
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

# Pre-compute saddle exit (shared across all experiments)
print("Computing saddle exit (shared)...")
torch.manual_seed(99)
m_ref=LM(); m_ref.te.weight.data.copy_(torch.tensor(E_init))
m_ref.zero_grad()
ls=[m_ref(*get_batch())[1] for _ in range(8)]; torch.stack(ls).mean().backward()
g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
             for p in m_ref.parameters()]).detach(); m_ref.zero_grad()
v_neg=-g/g.norm()
w0=m_ref.flat_params(); best_v=eval_val(m_ref,n=8); best_a=0.
for alpha in [0.5,1.0,1.43,2.0,3.0]:
    m_ref.set_flat(w0+alpha*v_neg); vt=eval_val(m_ref,n=6)
    if vt<best_v: best_v=vt; best_a=alpha
theta_saddle=w0+best_a*v_neg
v_saddle=best_v
print(f"Saddle exit: val={v_saddle:.4f}  (α*={best_a})")
print()

# ── SWEEP ────────────────────────────────────────────────────
MF_ROUNDS = [1, 2, 3, 5, 7, 10, 15]
sweep_results = {}

for n_mf in MF_ROUNDS:
    print(f"━━━ MF{n_mf:2d} ROUNDS (η={ETA_MF}, N_SUB={N_SUB}) ━━━━━━━━━━━━━━━━━")
    torch.manual_seed(99)
    model=LM(); model.set_flat(theta_saddle)
    t_start=time.time()

    diags=run_mf_pump(model, n_mf)
    v_mf=eval_val(model)

    # Last-round diagnostics
    last=diags[-1]
    kappa_final=last['kappa']
    tau_final=last['tau']
    phi_final=last['phi_clean']

    # Basin + TopoGate
    v_basin=run_basin_and_settle(model)
    phi_basin=sheet_angles(model)
    phi_basin_clean=sum(1 for p in phi_basin if p in ('0','π'))

    elapsed=time.time()-t_start
    print(f"  After MF{n_mf}: val={v_mf:.4f}")
    print(f"  Post-basin:    val={v_basin:.4f}  Φ={phi_basin}  Φ_cl={phi_basin_clean}/5")
    print(f"  κ_F_final={kappa_final:.1f}  τ_final={tau_final:.3f}  [{elapsed:.0f}s]")
    print()

    sweep_results[n_mf] = dict(
        v_mf=v_mf, v_basin=v_basin, phi_basin=phi_basin,
        phi_clean_basin=phi_basin_clean,
        kappa_final=kappa_final, tau_final=tau_final,
        diags=diags
    )

# ── RESULTS TABLE ─────────────────────────────────────────────
print("="*70)
print("SWEEP RESULTS — POST-BASIN VAL vs MF ROUNDS")
print("="*70); print()
print(f"  {'MF rounds':>10} {'val_mf':>8} {'val_basin':>10} {'κ_F':>8} "
      f"{'τ':>6} {'Φ_cl':>6} {'cliff?'}")
print("  "+"-"*65)
prev_basin = None
for n_mf in MF_ROUNDS:
    r=sweep_results[n_mf]
    cliff=''
    if prev_basin is not None and r['v_basin'] > prev_basin:
        cliff='← OVER-PUMPED'
    elif prev_basin is not None and r['v_basin'] < prev_basin - 0.05:
        cliff='← DEEPER'
    print(f"  {n_mf:>10} {r['v_mf']:>8.4f} {r['v_basin']:>10.4f} "
          f"{r['kappa_final']:>8.1f} {r['tau_final']:>6.3f} "
          f"{r['phi_clean_basin']:>6} {cliff}")
    prev_basin=r['v_basin']

# Find optimal
best_mf=min(sweep_results, key=lambda k: sweep_results[k]['v_basin'])
best_v=sweep_results[best_mf]['v_basin']
print()
print(f"  OPTIMAL: MF{best_mf} → post-basin val={best_v:.4f}")
print()

# Cliff detection
vals=[sweep_results[n]['v_basin'] for n in MF_ROUNDS]
min_idx=vals.index(min(vals))
if min_idx < len(MF_ROUNDS)-1 and vals[min_idx+1] > vals[min_idx]+0.05:
    print(f"  ✓ CLIFF DETECTED after MF{MF_ROUNDS[min_idx]}")
    print(f"    Resonant peak is NARROW → try annealed pump")
    print(f"    Annealed: η(r) = {ETA_MF} × 0.8^r")
elif min_idx == 0:
    print(f"  MF1 is optimal → single pulse sufficient")
    print(f"  Resonant peak at r=1 → very narrow pulse needed")
else:
    print(f"  Monotone improvement up to MF{MF_ROUNDS[min_idx]}")
    print(f"  No cliff detected → pump is not over-driving")
print()
kappa_traj = [f"{sweep_results[n]['kappa_final']:.0f}" for n in MF_ROUNDS]
tau_traj   = [f"{sweep_results[n]['tau_final']:.3f}"   for n in MF_ROUNDS]
print(f"  kappa_F trajectory: {kappa_traj}")
print(f"  tau trajectory:     {tau_traj}")
print()
print(f"  κ_F spike = over-pumping signal (ill-conditioned Fisher)")
print(f"  τ increase = topological noise (orbit shattering)")
print(f"  τ decrease = orbit alignment improving")
print()
print(f"  CONFIRMED: MF3 → val=0.062 (mean_field_init.py B)")
print(f"  CONFIRMED: MF10 → val=0.022 (build_pass6_checkpoint.py)")
print(f"  TARGET: find cliff and optimal pulse duration")
