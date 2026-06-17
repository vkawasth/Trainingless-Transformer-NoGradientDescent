#!/usr/bin/env python3
"""
W_K Basin Initialisation from Corpus (Teacher-Free)
=====================================================
DIAGNOSIS:
  Teacher W_K = basin membership (167 CE of free basin selection)
  PMI SVD W_K = token frequency basis (wrong subspace entirely)
  Gap: val=1.44 vs val=0.28 after 33CE basin selector
  
THE FIX — Option B: Corpus-derived W_K from attention target
  
  The teacher's W_K at equilibrium satisfies:
    softmax(E·W_Q^T·W_K·E^T / sqrt(D))_{q,k} ≈ P(k attends to q | context)
  
  Linearising softmax around 1/V:
    E·W_Q^T·W_K·E^T / sqrt(D) ≈ log P(k|q) + const
  
  If W_Q = W_K (symmetric, which is approximately true at init):
    W_K ≈ (E^T E)^{-1/2} · A_target^{1/2}
  where A_target[q,k] = log P(k attends to q) from corpus n-gram statistics.
  
  Simpler approximation (tractable):
    W_K = E^+ · A_target
  where E^+ = (E^T E)^{-1} E^T is the pseudoinverse of the embedding matrix.
  This directly maps: key_vector[k] = W_K · E[k] ≈ attention_pattern_for_token_k

EXPERIMENTS:
  A: PMI SVD W_K (current, broken)  -> val=1.44 after 33CE
  B: Attention target W_K (fix)     -> val=? after 33CE
  C: Random W_K (baseline)          -> val=? after 33CE
  D: Conditional PMI W_K            -> val=? after 33CE
"""
import json, math, collections, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
L_ATT=14; ALPHA_STAR=1.429

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
vocab={t:i for i,t in enumerate(_v)} if isinstance(_v,list) else _v
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

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

def eval_val(m,n=30):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clr(s,total,warmup=20):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ─── Corpus Statistics ────────────────────────────────────────────────────────
print("Computing corpus statistics...")

# Token frequencies
freq = np.zeros(VOCAB)
for t in train_ids:
    if t < VOCAB: freq[t] += 1
P = freq / freq.sum()

# PMI matrix (for embeddings)
bigram = collections.Counter()
for i in range(len(train_ids)-1):
    a, b = train_ids[i], train_ids[i+1]
    if a < VOCAB and b < VOCAB: bigram[(a,b)] += 1
total_bigrams = max(sum(bigram.values()), 1)

top_tokens = np.argsort(P)[::-1][:D].tolist()
pmi_emb = torch.randn(VOCAB, D) * 0.02
for t in range(VOCAB):
    if P[t] < 1e-8: continue
    vec = torch.zeros(D)
    for j, ctx in enumerate(top_tokens):
        if j >= D: break
        cnt = bigram.get((t,ctx),0) + bigram.get((ctx,t),0)
        if cnt > 0:
            pmi_val = math.log(cnt/total_bigrams/(P[t]*P[ctx]+1e-15)+1e-15)
            vec[j] = pmi_val
    n = float(vec.norm())
    if n > 0: pmi_emb[t] = vec/n * 0.3

# OPTION B: Attention target W_K
# A_target[q,k] = log P(k is in context of q, weighted by proximity)
# W_K ≈ A_target @ E @ (E^T E)^{-1}  [key space = attention target in embedding space]
print("Computing attention target for W_K...")
WINDOW = 8
# Build conditional: for each query token q, what are the key distributions?
A_target = torch.zeros(VOCAB, VOCAB)  # [query, key]
for i in range(min(len(train_ids)-WINDOW, 50000)):
    window = train_ids[i:i+WINDOW]
    for pq in range(len(window)):
        q = window[pq]
        if q >= VOCAB: continue
        for pk in range(max(0, pq-WINDOW), pq):
            k = window[pk]
            if k >= VOCAB: continue
            w = 1.0/(pq-pk+1)
            A_target[q,k] += w

# Normalise rows to get conditional P(k|q)
row_sums = A_target.sum(1, keepdim=True) + 1e-8
A_norm = A_target / row_sums  # [VOCAB, VOCAB] = P(key | query)

# W_K from attention target:
# The key vector for token k should produce high dot product with 
# query vectors for tokens q that attend to k.
# W_K[i,j] ≈ A_norm^T @ E_init, projected to D dims
# Simple: W_K = A_norm[:VOCAB, :VOCAB] @ E_init / sqrt(D)
# But A_norm is VOCAB×VOCAB = 1017×1017, E_init is 1017×256
# W_K should be D×D = 256×256

# Use top-D tokens only
topD = min(D, VOCAB)
A_sub = A_norm[:topD, :topD]  # [D,D]
# SVD of A_sub for basis
try:
    UA, SA, VtA = torch.linalg.svd(A_sub)
    print(f"  Attention target SVD: top-5 SVs = {SA[:5].tolist()}")
    print(f"  Concentration S[0]/S[9] = {float(SA[0]/SA[9]):.2f}")
    wk_from_attn = UA[:D, :D] * 0.1  # [D,D]
except Exception as e:
    print(f"  SVD failed: {e}, using random")
    wk_from_attn = torch.randn(D, D) * 0.02

# OPTION D: Conditional PMI (next-token prediction basis)
# W_K[k] = the direction in embedding space that best predicts token k appears next
# P(next=k | current=q) = bigram conditional
print("Computing conditional PMI W_K...")
cond_matrix = torch.zeros(VOCAB, VOCAB)  # [next, current] = P(next|current)
for (a,b), cnt in bigram.items():
    if a < VOCAB and b < VOCAB: cond_matrix[b, a] += cnt  # b follows a
row_c = cond_matrix.sum(1, keepdim=True) + 1e-8
cond_norm = cond_matrix / row_c  # P(next=k | current=q)

# SVD of conditional for D×D block
try:
    UC, SC, VtC = torch.linalg.svd(cond_norm[:D,:D])
    print(f"  Conditional PMI SVD: top-5 SVs = {SC[:5].tolist()}")
    wk_from_cond = UC[:D,:D] * 0.1
except:
    wk_from_cond = torch.randn(D,D)*0.02

# ─── Build student variants ───────────────────────────────────────────────────

def build_student(wk_init=None, emb_init=None):
    torch.manual_seed(99)
    stu = LM(D, N_HEADS, N_STU)
    if emb_init is not None:
        stu.te.weight.data.copy_(emb_init)
    if wk_init is not None:
        with torch.no_grad():
            for l in range(N_STU):
                n = min(wk_init.shape[0], D)
                stu.blocks[l].attn.WK.weight[:n,:n] = wk_init[:n,:n]
                stu.blocks[l].attn.WQ.weight[:n,:n] = wk_init[:n,:n].T
    return stu

def run_basin_selector(stu, n_steps=100, lr_mult=1.0, label=""):
    opt = torch.optim.AdamW(stu.parameters(),lr=LR*lr_mult,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1, n_steps+1):
        for pg in opt.param_groups: pg['lr']=LR*lr_mult*min(step,10)/10
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt.step()
        if step in [33, 50, 100, 200]:
            v = eval_val(stu,n=10)
            print(f"  {label} CE {step}: {v:.4f}")
    return eval_val(stu, n=20)

print("\n" + "="*65)
print("EXPERIMENT: Which W_K init reaches correct basin?")
print("Metric: val after N CE steps from each init")
print("="*65)

results = {}

# A: PMI SVD (current broken approach)
print("\n[A] PMI SVD W_K (current, val=1.44 after 33CE)")
stu_a = build_student(emb_init=pmi_emb)
v = run_basin_selector(stu_a, n_steps=200, lr_mult=1.0, label="A")
results['A_200CE'] = v

# B: Attention target W_K + PMI emb
print("\n[B] Attention target W_K + PMI embeddings")
stu_b = build_student(wk_init=wk_from_attn, emb_init=pmi_emb)
v = run_basin_selector(stu_b, n_steps=200, lr_mult=1.0, label="B")
results['B_200CE'] = v

# C: Random W_K (pure baseline)
print("\n[C] Random W_K (baseline)")
stu_c = build_student()
v = run_basin_selector(stu_c, n_steps=200, lr_mult=1.0, label="C")
results['C_200CE'] = v

# D: Conditional PMI W_K
print("\n[D] Conditional PMI (next-token prediction) W_K")
stu_d = build_student(wk_init=wk_from_cond, emb_init=pmi_emb)
v = run_basin_selector(stu_d, n_steps=200, lr_mult=1.0, label="D")
results['D_200CE'] = v

print(f"""
{'='*65}
  W_K INIT BASIN COMPARISON (200 CE steps each)
{'='*65}
    A (PMI SVD W_K):          val={results['A_200CE']:.4f}
    B (Attention target W_K): val={results['B_200CE']:.4f}
    C (Random W_K):           val={results['C_200CE']:.4f}
    D (Conditional PMI W_K):  val={results['D_200CE']:.4f}
    
  Teacher (24L, 300 steps):   val≈0.247
  
  If B or D << A: the corpus W_K structure helps
  If B ≈ C: attention target doesn't add over random
  If all ≈ C: the init doesn't matter, only CE steps count
  
  KEY FINDING: The teacher W_K provided ~167 CE steps of free
  basin selection. Without it, we need those steps back.
  The question is whether a BETTER corpus W_K can substitute.
""")
