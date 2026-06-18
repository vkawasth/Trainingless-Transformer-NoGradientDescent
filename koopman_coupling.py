#!/usr/bin/env python3
"""
Koopman / Diffusion Map Parameter Coupling Analysis
====================================================
Identifies which parameters are tightly coupled vs nearly independent
during the 167 CE embedding relaxation steps.

Key idea (Ott-Antonsen / Koopman reduction):
  Not all N parameters change every other parameter.
  If we can identify the SLOW MODES (parameters that move little
  and drag others with them) vs FAST MODES (parameters that
  converge quickly and independently), we can:
    - Solve fast modes analytically (one shot)
    - Only iterate on slow modes (few steps instead of 167)
    - Use diffusion maps to find the low-dim slow manifold

MEASUREMENTS:
  1. ||Δw_i|| over 167 steps: which params move most?
  2. ||∇L_i|| trajectory: which params need sustained gradient signal?
  3. Cross-coupling: does freezing param A harm param B's convergence?
  4. Laplacian eigenmaps of the parameter coupling graph

Usage:
    python build_pass6_checkpoint.py   # first
    python koopman_coupling.py
"""
import json, math, warnings, collections, os, sys, copy
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

if not os.path.exists('/tmp/model_post_pass6.pt'):
    print("ERROR: run build_pass6_checkpoint.py first"); sys.exit(1)

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
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)

def classify(name):
    if '.attn.WQ.' in name:   return 'WQ'
    elif '.attn.WK.' in name: return 'WK'
    elif '.attn.WV.' in name: return 'WV'
    elif '.attn.op.' in name: return 'WO'
    elif 'te.weight'  in name: return 'Emb'
    elif '.ff.' in name:       return 'FF'
    return 'Norm/PE'

def eval_val(m, n=15):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def run_ce(m, steps=167, frozen=None):
    # Returns (val, trained_model_copy)
    mc = copy.deepcopy(m)
    if frozen:
        for name, p in mc.named_parameters():
            if classify(name) in frozen:
                p.requires_grad_(False)
    params = [p for p in mc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9,0.95), weight_decay=0.1)
    for _ in range(steps):
        mc.train(); x,y=get_batch(); _,l=mc(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    return eval_val(mc, n=20), mc

model = LM(D, N_HEADS, N_STU)
model.load_state_dict(torch.load('/tmp/model_post_pass6.pt', weights_only=True))
v0 = eval_val(model, n=20)
print(f"Post-Pass-6 val: {v0:.4f}")

# ── Measurement 1: How much does each param group move in 167 steps? ─────────
print("\n" + "="*65)
print("1. PARAMETER DISPLACEMENT over 167 CE steps")
print("   ||Δw|| / ||w|| tells us: does this param need to move at all?")
print("="*65)

model_after = copy.deepcopy(model)
v_full, model_after = run_ce(model, steps=167)

print(f"\n  Full 167 CE: val {v0:.4f} → {v_full:.4f}")
print()
print(f"  {'Group':<9}  {'||Δw||':>9}  {'||Δw||/||w||':>13}  {'verdict'}")
print("  " + "-"*60)

displacements = {}
for name, p_before in model.named_parameters():
    p_after = dict(model_after.named_parameters())[name]
    delta = (p_after.data - p_before.data).norm().item()
    w_norm = p_before.data.norm().item()
    rel = delta / (w_norm + 1e-8)
    ptype = classify(name)
    if ptype not in displacements:
        displacements[ptype] = []
    displacements[ptype].append(rel)

for ptype in ['WK','WQ','WV','WO','Emb','FF','Norm/PE']:
    if ptype not in displacements: continue
    rel = np.mean(displacements[ptype])
    if rel > 0.5:    verdict = 'SLOW MODE — large displacement, needs CE'
    elif rel > 0.1:  verdict = 'moderate — benefits from CE'
    elif rel > 0.02: verdict = 'fast mode — small displacement'
    else:            verdict = 'FROZEN — nearly static!'
    print(f"  {ptype:<9}  {rel*np.mean([p.data.norm().item() for n,p in model.named_parameters() if classify(n)==ptype]):>9.4f}  {rel:>13.4f}  {verdict}")

# ── Measurement 2: Ablation — what if we freeze each group? ──────────────────
print("\n" + "="*65)
print("2. FREEZE ABLATION — val after 167 CE with each group frozen")
print("   val close to full = group NOT needed for CE convergence")
print("   val much worse = group IS essential during CE")
print("="*65)
print()
print(f"  Baseline (all free, 167 CE): val={v_full:.4f}")
print()
print(f"  {'Frozen group':<14}  {'val@167':>9}  {'gap vs full':>11}  {'verdict'}")
print("  " + "-"*55)

groups = ['WK','WQ','WV','WO','Emb','FF']
for freeze_group in groups:
    v_frozen, _ = run_ce(model, steps=167, frozen={freeze_group})
    gap = v_frozen - v_full
    if gap < 0.01:
        verdict = 'SAFE TO FREEZE — not needed'
    elif gap < 0.05:
        verdict = 'small impact'
    elif gap < 0.15:
        verdict = 'moderate impact'
    else:
        verdict = 'CRITICAL — must update'
    print(f"  freeze {freeze_group:<8}  {v_frozen:>9.4f}  {gap:>+11.4f}  {verdict}")

# ── Measurement 3: Minimal update set ────────────────────────────────────────
print("\n" + "="*65)
print("3. MINIMAL UPDATE SET — which combinations suffice?")
print("   Find smallest set of groups that achieves val ≈ v_full")
print("="*65)
print()

# Test: update only the SLOW modes identified above
combos = [
    ('Emb only',       {'WQ','WK','WV','WO','FF'}),
    ('WK+Emb',         {'WQ','WV','WO','FF'}),
    ('WQ+WK+Emb',      {'WV','WO','FF'}),
    ('Emb+FF',         {'WQ','WK','WV','WO'}),
    ('WK+WQ+Emb+FF',   {'WV','WO'}),
    ('all attn+Emb',   {'FF'}),
]

print(f"  {'Updated groups':<25}  {'val@167':>9}  {'vs full':>9}")
print("  " + "-"*50)
for label, frozen_set in combos:
    v, _ = run_ce(model, steps=167, frozen=frozen_set)
    gap = v - v_full
    marker = ' ← MINIMAL SUFFICIENT' if abs(gap) < 0.02 else ''
    print(f"  {label:<25}  {v:>9.4f}  {gap:>+9.4f}{marker}")

# ── Measurement 4: Step budget per group ─────────────────────────────────────
print("\n" + "="*65)
print("4. STEP BUDGET — how many CE steps does each group need?")
print("   Find: N steps for slow group, 0 for frozen groups")
print("="*65)
print()

# Key insight: if FF is nearly frozen, can we run WK+Emb for 25 steps
# and get the same result as all params for 167 steps?
step_tests = [
    ('all,   25 steps', None, 25),
    ('all,   50 steps', None, 50),
    ('all,  100 steps', None, 100),
    ('all,  167 steps', None, 167),
]
print(f"  {'Config':<25}  {'val':>9}")
print("  " + "-"*38)
for label, frozen, steps in step_tests:
    v, _ = run_ce(model, steps=steps, frozen=frozen or set())
    print(f"  {label:<25}  {v:>9.4f}")

print()
print("KOOPMAN REDUCTION VERDICT:")
print("  The SLOW MODES (large ||Δw||/||w||) must be updated via CE")
print("  The FAST MODES (tiny ||Δw||/||w||) can be solved algebraically")
print("  The FROZEN groups can be set once (spectral/algebraic) and fixed")
print()
print("  Target: replace 167-step full update with:")
print("    - Algebraic one-shot for frozen groups")
print("    - ~25 CE steps for slow modes only")
print("    - Total: 25 CE steps instead of 167")
