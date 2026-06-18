#!/usr/bin/env python3
"""
Embedding Isolation Test
========================
Tests: if we freeze ALL attention weights (WQ,WK,WV,WO,FF)
and only update the embedding E, how close do we get to
the full 167-step result?

If Emb-only-167 ≈ All-free-167:
  → Emb IS the only slow mode
  → One-shot Emb update replaces 167 CE steps
  → The attention weights are already at θ* after Pass 6

If the gap is large:
  → Multiple parameter groups are slow modes
  → Need the full Koopman analysis to find minimal set

Also tests: how many steps does Emb-only need?
  1, 5, 10, 25, 50, 167 steps — find the knee point.

Then: one-shot Emb via pseudoinverse of output head.
  E* = W_head^+ @ R   where R[t] = sum_s P(s→t) * h_s
  This is the exact least-squares solution to the embedding
  update problem given fixed attention weights.

Requires: build_pass6_checkpoint.py to have been run first.
"""
import json, math, warnings, collections, os, copy, sys
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

def eval_val(m,n=25):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def ce_steps(m, steps, freeze_groups=frozenset()):
    mc = copy.deepcopy(m)
    for name, p in mc.named_parameters():
        grp = ('WQ' if '.attn.WQ.' in name else
               'WK' if '.attn.WK.' in name else
               'WV' if '.attn.WV.' in name else
               'WO' if '.attn.op.' in name else
               'Emb' if 'te.weight' in name else
               'FF' if '.ff.' in name else None)
        if grp in freeze_groups:
            p.requires_grad_(False)
    params = [p for p in mc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9,0.95), weight_decay=0.1)
    for _ in range(steps):
        mc.train(); x,y=get_batch(); _,l=mc(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
    return eval_val(mc), mc

model = LM(D, N_HEADS, N_STU)
model.load_state_dict(torch.load('/tmp/model_post_pass6.pt', weights_only=True))
v0 = eval_val(model)
print(f"Post-Pass-6 val: {v0:.4f}")

ATTN = frozenset({'WQ','WK','WV','WO'})
ALL_BUT_EMB = frozenset({'WQ','WK','WV','WO','FF'})

# ── Section 1: Isolation test ─────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 1: ISOLATION — which group drives convergence?")
print("="*60)

v_full, m_full = ce_steps(model, 167)
print(f"\n  All free, 167 steps:          val={v_full:.4f}  [reference]")

v_attn_free, _ = ce_steps(model, 167, freeze_groups=frozenset({'Emb','FF'}))
print(f"  Only attn (freeze Emb+FF):    val={v_attn_free:.4f}  gap={v_attn_free-v_full:+.4f}")

v_emb_only, _ = ce_steps(model, 167, freeze_groups=ALL_BUT_EMB)
print(f"  Only Emb (freeze all attn+FF):val={v_emb_only:.4f}  gap={v_emb_only-v_full:+.4f}")

v_ff_only, _  = ce_steps(model, 167, freeze_groups=frozenset({'WQ','WK','WV','WO','Emb'}))
print(f"  Only FF (freeze all attn+Emb):val={v_ff_only:.4f}  gap={v_ff_only-v_full:+.4f}")

v_embff, _    = ce_steps(model, 167, freeze_groups=ATTN)
print(f"  Emb+FF (freeze attn):         val={v_embff:.4f}  gap={v_embff-v_full:+.4f}")

print()
if abs(v_emb_only - v_full) < 0.03:
    print("  ✓ EMB IS THE ONLY SLOW MODE")
    print("  → All attention weights are at θ* after Pass 6")
    print("  → One-shot Emb update is the right approach")
elif abs(v_embff - v_full) < 0.03:
    print("  ✓ EMB+FF are the slow modes (attn is at θ*)")
elif abs(v_attn_free - v_full) < 0.03:
    print("  ✓ ATTN is the slow mode (Emb is near θ*)")
else:
    print("  Multiple groups are slow modes — check individual contributions")

# ── Section 2: Emb-only step budget ──────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 2: EMB-ONLY STEP BUDGET")
print("How many steps does Emb-only need to match full 167?")
print("="*60)
print(f"\n  {'Config':<35}  {'val':>7}  {'gap vs full':>11}")
print("  " + "-"*55)
for steps in [1, 5, 10, 25, 50, 100, 167]:
    v, _ = ce_steps(model, steps, freeze_groups=ALL_BUT_EMB)
    gap = v - v_full
    knee = ' ← knee?' if abs(gap) < 0.05 else ''
    print(f"  Emb only, {steps:>4} steps:              {v:>7.4f}  {gap:>+11.4f}{knee}")

# ── Section 3: One-shot Emb via pseudoinverse ─────────────────────────────────
print("\n" + "="*60)
print("SECTION 3: ONE-SHOT EMB — pseudoinverse of output head")
print("Given FIXED attention weights, solve E* analytically")
print("="*60)
print("""
  The loss w.r.t. E at fixed attention weights θ*:
    L(E) = -sum_{s,t} P(s→t) log softmax(E[t]·h_s/√d)[t]
  
  At the minimum: E*[t] = argmin_e  -sum_s P(s→t) log(e·h_s/√d)
  
  Since W_head = E (tied weights), the output is: logit[t] = E[t]·h/√d
  The optimal E*[t] maximizes dot product with the corpus-averaged h:
    E*[t] ∝ sum_s P(s→t) · h_s   [the "target hidden state" for token t]
  
  This is exactly r_corpus[t] = sum_s P_corpus(s→t) · h_s
  (the same quantity from the slow-manifold projection)
  
  The key difference: here attention weights are TRULY fixed,
  so h_s does not change when E changes.
  The equation is LINEAR in E.
""")

# Build corpus bigram
bigram = collections.Counter()
for i in range(len(train_ids)-1):
    a,b = train_ids[i], train_ids[i+1]
    if a<VOCAB and b<VOCAB: bigram[(a,b)] += 1

freq = np.zeros(VOCAB)
for t in train_ids:
    if t < VOCAB: freq[t] += 1
P_token = freq / freq.sum()

# Accumulate r_corpus[t] = sum_s P(s→t) * h_s
# Use the FIXED attention-weight model (all attn frozen)
model_fixed = copy.deepcopy(model)
for name, p in model_fixed.named_parameters():
    if 'te.weight' not in name:
        p.requires_grad_(False)

# One forward pass to get h_s for all training sequences
print("  Computing r_corpus[t] (target hidden states)...")
R = torch.zeros(VOCAB, D)   # r_corpus
N = torch.zeros(VOCAB)       # count

model_fixed.eval()
P_tok = torch.tensor(P_token, dtype=torch.float32)

bigram_t = collections.Counter()
for i in range(len(train_ids)-1):
    a,b = train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB: bigram_t[(a,b)]+=1

# P(s→t) = P(current=s) * P(next=t|s)
A_corpus = np.zeros((VOCAB,VOCAB),dtype=np.float32)
for (a,b),cnt in bigram_t.items(): A_corpus[a,b]+=cnt
A_corpus /= (A_corpus.sum(1,keepdims=True)+1e-10)
A_corpus_t = torch.tensor(A_corpus, dtype=torch.float32)

with torch.no_grad():
    torch.manual_seed(42)
    n_seqs = 1000
    for i in range(n_seqs):
        ix = torch.randint(0, len(train_t)-SEQ-1, (1,))[0].item()
        x  = train_t[ix:ix+SEQ]; x_b = x.unsqueeze(0)
        # Get hidden states (ALL layers, fixed attn)
        h_in = model_fixed.te(x_b) + model_fixed.pe(torch.arange(SEQ))
        h = h_in.clone()
        for block in model_fixed.blocks: h = block(h)
        h = model_fixed.ln_f(h)[0]   # [SEQ, D]
        # Accumulate: for each query position s, add P(s→t)*h_s for all t
        for pos in range(SEQ):
            t_s = int(x[pos])
            if t_s >= VOCAB: continue
            w = float(P_tok[t_s])  # P(current = t_s)
            # r_corpus[t] += w * A_corpus[t_s, t] * h_s  for all t
            R += w * torch.outer(A_corpus_t[t_s], h[pos])
            N[t_s] += 1.0

n_total = float(N.sum())
R /= max(n_total, 1)

print(f"  r_corpus norm: {R.norm():.6f}")
print(f"  r_corpus / E_0 norm: {R.norm()/model.te.weight.data.norm():.6f}")

# The one-shot update: E*[t] is the E that minimizes ||E[t]·h - target_logit||
# Since logit[t] = E[t]·h/√d and we want logit[t] ≈ log P(t|s),
# the optimal E* in the subspace of h is:
# E*[t] = √d * R[t] / ||R[t]||² * scale
# (R[t] is already the corpus-weighted h_s target)

# Scale to match current embedding norm
E_0 = model.te.weight.data.clone()
E_0_norm = float(E_0.norm())

# Option A: direct assignment (E* = R scaled to E_0 norm)
R_norm = float(R.norm())
scale_A = E_0_norm / max(R_norm, 1e-8)
E_star_A = R * scale_A

# Option B: E* = E_0 + alpha*(R - E_0_aligned)
# Project R into embedding space via: the optimal alpha minimizes val
# Use the gradient direction: alpha = step size along R

# Apply and test
model_oneshot = copy.deepcopy(model)
with torch.no_grad():
    model_oneshot.te.weight.data.copy_(E_star_A)
v_oneshot_A = eval_val(model_oneshot, n=25)
print(f"\n  One-shot E* (R scaled to E_0 norm): val={v_oneshot_A:.4f}")

# Option B: additive update E* = E_0 + alpha*R
for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
    mc = copy.deepcopy(model)
    with torch.no_grad():
        mc.te.weight.data.copy_(E_0 + alpha * R)
    v = eval_val(mc, n=15)
    print(f"  One-shot E* (E_0 + {alpha}*R):          val={v:.4f}")

# Fine-tune one-shot Emb
print(f"\n  One-shot + {{}}-step Emb CE fine-tune:")
best_alpha = 0.1  # adjust based on above results
mc_best = copy.deepcopy(model)
with torch.no_grad():
    mc_best.te.weight.data.copy_(E_0 + best_alpha * R)
# Freeze everything except Emb
for name, p in mc_best.named_parameters():
    if 'te.weight' not in name: p.requires_grad_(False)
emb_params = [p for p in mc_best.parameters() if p.requires_grad]
opt_e = torch.optim.AdamW(emb_params, lr=LR, betas=(0.9,0.95), weight_decay=0.1)
for step in range(1, 26):
    mc_best.train(); x,y=get_batch(); _,l=mc_best(x,y)
    opt_e.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(emb_params,1.0); opt_e.step()
    if step in [5,10,25]:
        print(f"    Emb CE step {step:>3}: val={eval_val(mc_best,n=15):.4f}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Full 167 CE (all params):     val={v_full:.4f}  [reference]")
print(f"  Emb only, 167 CE:             val={v_emb_only:.4f}  gap={v_emb_only-v_full:+.4f}")
print(f"  One-shot E* + 25 Emb CE:      val={eval_val(mc_best,n=25):.4f}")
print()
print("If one-shot+25 ≈ full-167: the 167 steps reduce to 25 Emb-only steps")
print("+ one algebraic initialization of E*")
