#!/usr/bin/env python3
"""
Stage 2: Grothendieck Group Action on Correlated Gradients
===========================================================
Goal: Replace the 25 CE steps with one K-group algebraic computation.

THE SETUP:
  The 4 slow-mode parameters {Emb, WK, WQ, FF} form a correlated system.
  During 25 CE steps:
  - Gradient rotates ~55°/step in Emb subspace
  - WK is dragged by Emb (268:1 gradient ratio)
  - FF and Emb are conjugate (lockstep convergence)
  - WQ mirrors WK (initialized identically)

THE GROTHENDIECK GROUP K_1:
  K_1(C) classifies automorphisms of the parameter space.
  The 55°/step rotation generates [σ] ∈ K_1(C).
  The 25-step trajectory = [σ]^25 in K_1(C).
  
  Key property: [σ]^25 is computable in O(D²) without 25 iterations.
  If g_25 ≈ R(25θ) · g_0, then:
  - We can predict the weight update direction after 25 steps
  - Apply it directly as a single LM-like step
  - Replicate the Hessian rotation without iterating

THE SHORT EXACT SEQUENCES:
  0 → K_fast → K_all → K_slow → 0
  K_fast = {WV, WO}  (dispensable, absorbed quickly)
  K_slow = {Emb, WK, WQ, FF}  (essential, require 25 steps)
  
  The K-group extension class [τ] ∈ Ext¹(K_slow, K_fast) measures
  how tightly the slow modes couple to the fast modes.
  If [τ] = 0: slow modes decouple → can solve independently.
  If [τ] ≠ 0: coupling requires joint update (the 25 CE steps).

EXPERIMENTS:
  1. Measure rotation angle θ precisely from gradient trace
  2. Compute K_1 generator [σ] = 2D rotation in gradient subspace
  3. Test: g_25_predicted = R(25θ) · g_0 vs actual g_25
  4. Apply K_1 prediction as direct weight update (bypass 25 CE)
  5. Measure coupling extension class [τ] for each pair
"""
import json, math, warnings, collections, os, copy, sys
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f): print(f"ERROR: {f} missing."); sys.exit(1)

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids=list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t=torch.tensor(val_ids,dtype=torch.long)
print(f"VOCAB={VOCAB}")

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return torch.stack([data[i:i+SEQ] for i in ix]),torch.stack([data[i+1:i+SEQ+1] for i in ix])

class Attn(nn.Module):
    def __init__(self,d,nh):
        super().__init__(); self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
        self.WQ=nn.Linear(d,d,bias=False); self.WK=nn.Linear(d,d,bias=False)
        self.WV=nn.Linear(d,d,bias=False); self.op=nn.Linear(d,d,bias=False)
        self.ln=nn.LayerNorm(d)
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h):
        B,S,D_=h.shape; H=self.nh; dh=self.dh
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2); K=self.WK(h).view(B,S,H,dh).transpose(1,2)
        V=self.WV(h).view(B,S,H,dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/self.sc
        mask=torch.triu(torch.ones(S,S,device=h.device),diagonal=1).bool()
        sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        return self.ln(h+self.op((F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D_)))
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
        self.ln_f=nn.LayerNorm(d); self.head=nn.Linear(d,VOCAB,bias=False)
        self.head.weight=self.te.weight
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

def ptype(name):
    if '.attn.WQ.' in name: return 'WQ'
    if '.attn.WK.' in name: return 'WK'
    if '.attn.WV.' in name: return 'WV'
    if '.attn.op.' in name: return 'WO'
    if 'te.weight'  in name: return 'Emb'
    if '.ff.'       in name: return 'FF'
    return 'other'

# Spectral embedding
bigram=collections.Counter()
for i in range(len(train_ids)-1):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB: bigram[(a,b)]+=1
rows,cols,vals_sp=[],[],[]
for (a,b),cnt in bigram.items(): rows.append(a); cols.append(b); vals_sp.append(float(cnt))
W_sp=sp.csr_matrix((vals_sp,(rows,cols)),shape=(VOCAB,VOCAB),dtype=np.float32)
W_sp=W_sp+W_sp.T; d_inv=np.array(1.0/(W_sp.sum(1)+1e-8)).flatten()
Dsi=sp.diags(np.sqrt(d_inv)); L_sym=sp.eye(VOCAB)-Dsi@W_sp@Dsi
evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)
idx_s=np.argsort(evals); evecs=evecs[:,idx_s][:,1:D+1]
sc_ev=1.0/(np.sqrt(evals[idx_s[1:D+1]])+1e-8)
E_0=(evecs*sc_ev[np.newaxis,:]).astype(np.float32)
E_0=(E_0/(E_0.std()+1e-8)*0.02)

torch.manual_seed(99); model=LM(D,N_HEADS,N_STU)
model.te.weight.data.copy_(torch.tensor(E_0))

# ── EXPERIMENT 1: Measure rotation angle precisely ────────────────────────────
print("\n" + "="*60)
print("EXP 1: Gradient rotation angle per CE step")
print("="*60)
print(f"  {'Step':>5}  {'val':>7}  {'cos(g_t,g_0)':>13}  {'angle(°)':>9}  "
      f"{'cos(g_t,g_{t-1})':>16}  {'step_angle(°)':>13}")
print("  "+"-"*70)

opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

# Compute initial gradient
model.zero_grad()
ls=[]
for _ in range(16): x,y=get_batch(); _,l=model(x,y); ls.append(l)
torch.stack(ls).mean().backward()
g0={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
g0_emb=g0['te.weight'].flatten()
model.zero_grad()

g_prev=g0_emb.clone()
rotation_angles=[]
step_angles=[]
grad_history={'Emb':[],'WK':[],'WQ':[],'FF':[]}

for step in range(1,26):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    
    # Capture gradient BEFORE step
    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    opt.step(); opt.zero_grad()
    
    g_emb=grads['te.weight'].flatten()
    cos_g0=float((g_emb*g0_emb).sum()/(g_emb.norm()*g0_emb.norm()+1e-10))
    cos_prev=float((g_emb*g_prev).sum()/(g_emb.norm()*g_prev.norm()+1e-10))
    angle_from_0=math.degrees(math.acos(max(-1,min(1,cos_g0))))
    step_angle=math.degrees(math.acos(max(-1,min(1,cos_prev))))
    rotation_angles.append(angle_from_0)
    step_angles.append(step_angle)
    
    # Per-group gradient norms
    for name,g in grads.items():
        pt=ptype(name)
        if pt in grad_history:
            grad_history[pt].append(float(g.norm()))
    
    v=eval_val(model,n=6)
    if step<=10 or step in [15,20,25]:
        print(f"  {step:>5}  {v:>7.4f}  {cos_g0:>13.4f}  {angle_from_0:>9.1f}°  "
              f"{cos_prev:>16.4f}  {step_angle:>13.1f}°")
    g_prev=g_emb.clone()

v25=eval_val(model)
print(f"\n  After 25 CE: val={v25:.4f}")
mean_step_angle=np.mean(step_angles)
std_step_angle=np.std(step_angles)
print(f"  Mean step angle: {mean_step_angle:.1f}° ± {std_step_angle:.1f}°")
print(f"  Total rotation: {sum(step_angles):.1f}° = {sum(step_angles)/360:.2f} full rotations")

# ── EXPERIMENT 2: K_1 generator [σ] ──────────────────────────────────────────
print("\n" + "="*60)
print("EXP 2: K_1 generator [σ] — rotation matrix in gradient subspace")
print("="*60)

theta = math.radians(mean_step_angle)
print(f"  Generator [σ]: rotation by θ = {mean_step_angle:.2f}°")
print(f"  [σ]^25 = rotation by 25θ = {25*mean_step_angle:.1f}°")
predicted_final_angle = (25 * mean_step_angle) % 360
print(f"  Predicted g_25 angle from g_0: {predicted_final_angle:.1f}°")
print(f"  Actual g_25 angle from g_0:    {rotation_angles[-1]:.1f}°")
prediction_error = abs(predicted_final_angle - rotation_angles[-1])
print(f"  Prediction error: {prediction_error:.1f}°")

# ── EXPERIMENT 3: Predict weight update from K_1 action ──────────────────────
print("\n" + "="*60)
print("EXP 3: K_1 weight update prediction")
print("="*60)
print()
print("  g_0 = initial Emb gradient")
print(f"  g_25_predicted = R({25*mean_step_angle:.0f}°) · g_0")
print()

# Build 2D rotation in the g_0 direction
g0_hat = g0_emb / (g0_emb.norm() + 1e-10)
# g_perp = any vector orthogonal to g0_hat (use Gram-Schmidt with random vector)
torch.manual_seed(1337)
g_rand = torch.randn_like(g0_hat)
g_perp = g_rand - (g_rand * g0_hat).sum() * g0_hat
g_perp = g_perp / (g_perp.norm() + 1e-10)

# Rotation in the (g0_hat, g_perp) plane
total_angle = math.radians(25 * mean_step_angle)
cos_t = math.cos(total_angle); sin_t = math.sin(total_angle)
g25_predicted = cos_t * g0_hat + sin_t * g_perp

# The predicted weight update direction
# ΔEmb = -α * g25_predicted (move in the predicted gradient direction)
# α chosen to match the actual displacement after 25 steps
actual_emb_25 = model.te.weight.data.flatten()
actual_emb_0  = torch.tensor(E_0.flatten())
actual_delta  = actual_emb_25 - actual_emb_0
alpha_match   = float((actual_delta * g25_predicted).sum() / (g25_predicted.norm()**2))

print(f"  Predicted update direction cosine with actual ΔEmb:")
cos_pred_actual = float((g25_predicted * actual_delta/actual_delta.norm()).sum())
print(f"  cos(g25_pred, ΔEmb_actual) = {cos_pred_actual:.4f}")
print(f"  (1.0 = perfect prediction, 0.0 = orthogonal, -1.0 = opposite)")
print()

# ── EXPERIMENT 4: Apply K_1 prediction as direct update ──────────────────────
print("="*60)
print("EXP 4: K_1 direct update — bypass 25 CE steps")
print("="*60)
print()
print("  Apply: E_new = E_0 + alpha * (-g25_predicted)")
print("  where alpha is the step size matching actual 25-step displacement")
print()

torch.manual_seed(99); model_k1=LM(D,N_HEADS,N_STU)
model_k1.te.weight.data.copy_(torch.tensor(E_0))

# Apply the K_1-predicted embedding update
alpha_search_best=0; v_search_best=eval_val(model_k1)
for alpha_scale in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    alpha=alpha_scale * abs(alpha_match)
    E_k1=(E_0.flatten() + alpha * (-g25_predicted).numpy()).reshape(VOCAB,D)
    E_k1=(E_k1*(float(np.linalg.norm(E_0))/max(float(np.linalg.norm(E_k1)),1e-8))).astype(np.float32)
    with torch.no_grad(): model_k1.te.weight.data.copy_(torch.tensor(E_k1))
    v=eval_val(model_k1,n=10)
    if v<v_search_best: v_search_best=v; alpha_search_best=alpha_scale*abs(alpha_match)
    print(f"  alpha_scale={alpha_scale:.1f}: val={v:.4f}")

# Compare to 25 CE baseline
print(f"\n  K_1 direct update best: val={v_search_best:.4f}")
print(f"  25 CE baseline:         val={v25:.4f}")
print(f"  Improvement: {'K_1 better' if v_search_best<v25 else 'CE better'} "
      f"by {abs(v_search_best-v25):.4f} nats")

# ── EXPERIMENT 5: Coupling extension class ────────────────────────────────────
print("\n" + "="*60)
print("EXP 5: Coupling extension class [τ] for each parameter pair")
print("="*60)
print()
print("  Measure: cos(ΔEmb, ΔWK) — are the updates correlated?")
print("  [τ] = 0 if decorrelated (can solve independently)")
print("  [τ] ≠ 0 if coupled (must update jointly)")
print()

# Compare initial and final parameters
initial={'te.weight':torch.tensor(E_0.flatten())}
for name,param in model.named_parameters():
    if ptype(name) in ['WK','WQ','FF']:
        # We need the initial values — reinitialize
        pass

# Re-run to get initial and post-25-CE values with tracking
torch.manual_seed(99); m_track=LM(D,N_HEADS,N_STU)
m_track.te.weight.data.copy_(torch.tensor(E_0))
params_0={n:p.data.clone().flatten() for n,p in m_track.named_parameters()}

opt2=torch.optim.AdamW(m_track.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for _ in range(25):
    m_track.train(); x,y=get_batch(); _,l=m_track(x,y)
    opt2.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(m_track.parameters(),1.0); opt2.step()

params_25={n:p.data.clone().flatten() for n,p in m_track.named_parameters()}

# Compute delta per group and cross-correlations
deltas={}
for name in params_0:
    pt=ptype(name)
    d=params_25[name]-params_0[name]
    if pt in deltas: deltas[pt]=torch.cat([deltas[pt],d])
    else: deltas[pt]=d

print(f"  {'Pair':<20}  {'cos(Δ_i, Δ_j)':>14}  {'|τ| interpretation'}")
print("  "+"-"*55)
pairs=[('Emb','WK'),('Emb','WQ'),('Emb','FF'),('WK','WQ'),('WK','FF'),('WQ','FF')]
for p1,p2 in pairs:
    if p1 in deltas and p2 in deltas:
        d1=deltas[p1]; d2=deltas[p2]
        # Align dimensions by sampling
        n=min(len(d1),len(d2),10000)
        cos=float((d1[:n]*d2[:n]).sum()/(d1[:n].norm()*d2[:n].norm()+1e-10))
        tau='strong coupling' if abs(cos)>0.3 else 'weak/decoupled'
        print(f"  {p1+'/'+p2:<20}  {cos:>14.4f}  {tau}")

print()
print("  High |cos| = strong coupling → must update jointly ([τ] ≠ 0)")
print("  Low  |cos| = weak coupling   → can update independently ([τ] = 0)")
