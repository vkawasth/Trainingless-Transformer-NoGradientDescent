#!/usr/bin/env python3
"""
Gradient Orientation Checker
==============================
Measures gradient alignment per parameter type and checks whether
the algebraic J14 direction agrees with the CE gradient direction.

Key questions:
  1. Are any parameter gradients anti-aligned with their weights?
     (sign flip = gradient pointing wrong direction)
  2. Does the algebraic W_K move in the same direction as CE says to move?
  3. Are there per-layer gradient orientation inconsistencies?

Usage:
    python build_pass6_checkpoint.py   # first
    python check_gradient_orientation.py
"""
import json, math, warnings, collections
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import sys, os

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

if not os.path.exists('/tmp/model_post_pass6.pt'):
    print("ERROR: run build_pass6_checkpoint.py first")
    sys.exit(1)

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

model = LM(D, N_HEADS, N_STU)
model.load_state_dict(torch.load('/tmp/model_post_pass6.pt', weights_only=True))
print(f"Loaded post-Pass-6 model")

# ── Accumulate stable gradient ────────────────────────────────────────────────
model.train()
model.zero_grad()
for _ in range(100):
    x,y = get_batch()
    _,l = model(x,y)
    (l / 100).backward()

# ── Per parameter-type analysis ───────────────────────────────────────────────
print("\nGRADIENT ORIENTATION BY PARAMETER TYPE")
print("="*65)
print(f"  {'Type':<14} {'cos(g,w)':>10}  {'|g|/|w|':>9}  {'|g|mean':>9}  status")
print("  " + "-"*60)

type_stats = {}
for name, param in model.named_parameters():
    if param.grad is None: continue
    g = param.grad.detach().float()
    w = param.data.detach().float()
    cos = float((g*w).sum() / (g.norm()*w.norm() + 1e-10))
    ratio = float(g.norm() / (w.norm() + 1e-10))
    gnorm = float(g.norm())

    # Classify
    if '.attn.WQ.' in name:   ptype = 'WQ'
    elif '.attn.WK.' in name: ptype = 'WK'
    elif '.attn.WV.' in name: ptype = 'WV'
    elif '.attn.op.' in name: ptype = 'WO'
    elif 'te.weight' in name: ptype = 'Embedding'
    elif '.ff.' in name:      ptype = 'FF'
    elif 'ln' in name:        ptype = 'LayerNorm'
    elif 'pe.' in name:       ptype = 'PosEmb'
    else:                     ptype = 'Other'

    if ptype not in type_stats:
        type_stats[ptype] = {'cos': [], 'ratio': [], 'gnorm': []}
    type_stats[ptype]['cos'].append(cos)
    type_stats[ptype]['ratio'].append(ratio)
    type_stats[ptype]['gnorm'].append(gnorm)

for ptype, stats in sorted(type_stats.items()):
    mc = np.mean(stats['cos'])
    mr = np.mean(stats['ratio'])
    mg = np.mean(stats['gnorm'])
    if mc > 0.3:    status = '⚠ ANTI-ALIGNED (wrong direction!)'
    elif mc > 0.1:  status = '~ slight anti-align'
    elif mc < -0.3: status = '✓ strongly aligned'
    elif mc < -0.1: status = '✓ aligned'
    else:           status = '✓ orthogonal (good)'
    print(f"  {ptype:<14} {mc:>10.4f}  {mr:>9.4f}  {mg:>9.4f}  {status}")

# ── Per-layer WK orientation ──────────────────────────────────────────────────
print("\nPER-LAYER WK GRADIENT ORIENTATION:")
print(f"  {'Layer':<8} {'cos(g,w)':>10}  {'|g|/|w|':>9}  status")
print("  " + "-"*40)
for name, param in model.named_parameters():
    if '.attn.WK.weight' not in name: continue
    g = param.grad.detach().float()
    w = param.data.detach().float()
    parts = name.split('.')
    layer = int(parts[2]) if parts[2].isdigit() else int(parts[1])
    cos = float((g*w).sum()/(g.norm()*w.norm()+1e-10))
    ratio = float(g.norm()/(w.norm()+1e-10))
    status = '⚠ ANTI' if cos > 0.3 else ('~ slight' if cos > 0.1 else '✓ OK')
    print(f"  L{layer:<6}  {cos:>10.4f}  {ratio:>9.4f}  {status}")

# ── Check gradient sign consistency between early and late CE steps ───────────
print("\nGRADIENT SIGN CONSISTENCY (step 1 vs step 50 direction):")
print("  Do gradients point in consistent direction throughout training?")
import copy
model2 = copy.deepcopy(model)

# Get gradient at step 0
model2.zero_grad()
for _ in range(20): x,y=get_batch(); _,l=model2(x,y); (l/20).backward()
g_step0 = {n: p.grad.clone() for n,p in model2.named_parameters() if p.grad is not None}

# Train 25 steps
opt = torch.optim.AdamW(model2.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
for _ in range(25):
    model2.train(); x,y=get_batch(); _,l=model2(x,y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model2.parameters(),1.0); opt.step()

# Get gradient at step 25
model2.zero_grad()
for _ in range(20): x,y=get_batch(); _,l=model2(x,y); (l/20).backward()
g_step25 = {n: p.grad.clone() for n,p in model2.named_parameters() if p.grad is not None}

print(f"  {'Type':<14} {'cos(g0,g25)':>12}  interpretation")
print("  " + "-"*50)
type_cos = {}
for name in g_step0:
    if name not in g_step25: continue
    g0 = g_step0[name].float().flatten()
    g25 = g_step25[name].float().flatten()
    cos = float((g0*g25).sum()/(g0.norm()*g25.norm()+1e-10))
    if '.attn.WQ.' in name:   ptype = 'WQ'
    elif '.attn.WK.' in name: ptype = 'WK'
    elif '.attn.WV.' in name: ptype = 'WV'
    elif '.attn.op.' in name: ptype = 'WO'
    elif 'te.weight' in name: ptype = 'Embedding'
    elif '.ff.' in name:      ptype = 'FF'
    else:                     ptype = 'Other'
    if ptype not in type_cos: type_cos[ptype] = []
    type_cos[ptype].append(cos)

for ptype, vals in sorted(type_cos.items()):
    mc = np.mean(vals)
    if mc > 0.5:   interp = 'consistent direction (good)'
    elif mc > 0.1: interp = 'mostly consistent'
    elif mc < -0.1: interp = '⚠ DIRECTION REVERSAL — gradient flips sign!'
    else:           interp = 'noisy/orthogonal'
    print(f"  {ptype:<14} {mc:>12.4f}  {interp}")

# ── J14 vs CE gradient alignment ─────────────────────────────────────────────
print("\nALGEBRAIC J14 vs CE GRADIENT DIRECTION:")
bigram = collections.Counter()
for i in range(len(train_ids)-1):
    a,b = train_ids[i], train_ids[i+1]
    if a<VOCAB and b<VOCAB: bigram[(a,b)] += 1
A_np = np.zeros((VOCAB,VOCAB),dtype=np.float64)
for (a,b),cnt in bigram.items(): A_np[a,b]+=cnt
A_np /= (A_np.sum(1,keepdims=True)+1e-10)
log_A = np.where(A_np>1e-12, np.log(A_np), -30.0)
logit_A = (log_A - log_A.mean(1,keepdims=True)).astype(np.float32)
U,S,Vt = np.linalg.svd(logit_A, full_matrices=False)
scale = float(D**0.25)
WK_tgt = (scale * Vt[:D,:].T * np.sqrt(S[:D])).astype(np.float32)
E0 = model.te.weight.data.numpy()
WK_algebraic,_,_,_ = np.linalg.lstsq(E0, WK_tgt, rcond=None)
WK_algebraic = WK_algebraic.T.astype(np.float32)  # [D,D]

WK_current = model.blocks[0].attn.WK.weight.data.numpy()
WK_grad = model.blocks[0].attn.WK.weight.grad.numpy()
WK_delta_alg = (WK_algebraic - WK_current).flatten()
WK_delta_ce  = -WK_grad.flatten()  # CE says move in -gradient direction

cos_alg_ce = float(np.dot(WK_delta_alg, WK_delta_ce) /
                   (np.linalg.norm(WK_delta_alg)*np.linalg.norm(WK_delta_ce)+1e-10))
print(f"  cos(algebraic_J14_direction, CE_gradient_direction) = {cos_alg_ce:.4f}")
if cos_alg_ce > 0.1:
    print("  ✓ ALIGNED: algebraic J14 moves in same direction as CE gradient")
    print("    Problem is scale/magnitude, not direction")
elif cos_alg_ce < -0.1:
    print("  ⚠ ANTI-ALIGNED: algebraic J14 moves OPPOSITE to CE gradient")
    print("    This confirms: 1-gram logit target is WRONG direction for 6-layer model")
else:
    print("  ~ ORTHOGONAL: algebraic J14 is perpendicular to CE gradient")
    print("    J14 solves a different problem than what CE optimizes")
    print("    The 1-gram approximation misses the relevant subspace entirely")

# ── k-step A_corpus powers ────────────────────────────────────────────────────
print("\nK-STEP A_CORPUS POWERS (checking if A^k aligns better with CE):")
print("  cos(direction from logit(A^k), CE_gradient) for k=1..6")

def wk_from_Ak(A_power_k):
    log_Ak = np.where(A_power_k > 1e-12, np.log(A_power_k), -30.0)
    logit_Ak = (log_Ak - log_Ak.mean(1, keepdims=True)).astype(np.float32)
    _U, _S, _Vt = np.linalg.svd(logit_Ak, full_matrices=False)
    WK_tgt_k = (scale * _Vt[:D, :].T * np.sqrt(_S[:D])).astype(np.float32)
    WK_alg_k, _, _, _ = np.linalg.lstsq(E0, WK_tgt_k, rcond=None)
    return WK_alg_k.T.astype(np.float32)  # [D,D]

cos_by_k = []
A_power = A_np.copy()
for k in range(1, 7):
    if k > 1:
        A_power = A_power @ A_np
    WK_k = wk_from_Ak(A_power)
    delta_k = (WK_k - WK_current).flatten()
    cos_k = float(np.dot(delta_k, WK_delta_ce) /
                  (np.linalg.norm(delta_k) * np.linalg.norm(WK_delta_ce) + 1e-10))
    cos_by_k.append(cos_k)
    print(f"  A^{k}: cos={cos_k:.4f}")

best_k = int(np.argmax(cos_by_k)) + 1
print(f"  Best alignment at k={best_k} (cos={cos_by_k[best_k-1]:.4f})")
print(f"  → Use logit(A_corpus^{best_k}) for algebraic J14 instead of k=1")

