#!/usr/bin/env python3
"""
Build Pass-6 Checkpoint
========================
Runs compiler passes 0-6 and saves the model to /tmp/model_post_pass6.pt.
This is the correct starting point for slow_manifold_pass11.py.

Faster than running the full spectral_compiler_v2.py (skips experiments A, B,
teacher verification, and Pass 7/8).

Usage:
    python build_pass6_checkpoint.py
    python slow_manifold_pass11.py   # then run this
"""
import json, math, warnings, collections
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

if VOCAB != 1017:
    print(f"ERROR: VOCAB={VOCAB}. Run: python build_corpus.py --out /tmp/ --loops 300")
    import sys; sys.exit(1)

print(f"VOCAB={VOCAB}, train={len(train_ids)} ({len(train_ids)//1364} loops)")

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
    def flat_params(self): return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat(self,f):
        idx=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(f[idx:idx+n].reshape(p.shape)); idx+=n

def eval_val(m,n=20):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clr(s, total=None, warmup=None):
    return LR  # constant LR

# ── Pass 0: Spectral embedding ────────────────────────────────────────────────
print("\nPass 0: Spectral embedding...")
bigram = collections.Counter()
for i in range(len(train_ids)-1):
    a,b = train_ids[i], train_ids[i+1]
    if a<VOCAB and b<VOCAB: bigram[(a,b)] += 1

rows_sp, cols_sp, vals_sp = [], [], []
for (a,b), cnt in bigram.items():
    rows_sp.append(a); cols_sp.append(b); vals_sp.append(float(cnt))
W = sp.csr_matrix((vals_sp,(rows_sp,cols_sp)), shape=(VOCAB,VOCAB), dtype=np.float32)
W = W + W.T
d_inv = np.array(1.0/(W.sum(1)+1e-8)).flatten()
D_sqrt_inv = sp.diags(np.sqrt(d_inv))
L_sym = sp.eye(VOCAB) - D_sqrt_inv @ W @ D_sqrt_inv
eigenvalues, eigenvectors = spla.eigsh(L_sym, k=D+1, which='SM', tol=1e-4, maxiter=2000)
idx = np.argsort(eigenvalues)
eigenvectors = eigenvectors[:,idx][:,1:D+1]
scales = 1.0/(np.sqrt(eigenvalues[idx[1:D+1]])+1e-8)
spectral_np = eigenvectors * scales[np.newaxis,:]
spectral_np = spectral_np / (spectral_np.std()+1e-8) * 0.02
spectral_emb = torch.tensor(spectral_np, dtype=torch.float32)
print(f"  Spectral emb: std={spectral_emb.std():.4f}")

# ── Build student ─────────────────────────────────────────────────────────────
torch.manual_seed(99)
stu = LM(D, N_HEADS, N_STU)
stu.te.weight.data.copy_(spectral_emb)
v0 = eval_val(stu, n=10)
print(f"  Pass 0 val: {v0:.4f}")

# ── Pass 2: Saddle exit ───────────────────────────────────────────────────────
print("\nPass 2: Saddle exit...")
n_p = sum(p.numel() for p in stu.parameters())
v_neg = torch.randn(n_p); v_neg = v_neg/v_neg.norm()
for _ in range(15):
    stu.zero_grad()
    loss = sum(stu(*get_batch())[1] for _ in range(15))/15
    grads = torch.autograd.grad(loss, list(stu.parameters()), create_graph=True)
    gv = (torch.cat([g.flatten() for g in grads])*v_neg.detach()).sum()
    hv = torch.cat([h.flatten() for h in
                    torch.autograd.grad(gv, list(stu.parameters()), retain_graph=False)]).detach()
    stu.zero_grad()
    neg=-hv; v_neg=neg/max(float(neg.norm()),1e-10)

ALPHA_STAR = 1.429
w0 = stu.flat_params()
stu.set_flat(w0 + ALPHA_STAR*(v_neg/v_neg.norm()))
v2 = eval_val(stu, n=10)
print(f"  Pass 2 val: {v2:.4f}")

# ── Pass 3: Sign correction ───────────────────────────────────────────────────
with torch.no_grad():
    for l in [1, 2]:
        stu.blocks[l].attn.WV.weight.mul_(-1)
        stu.blocks[l].attn.op.weight.mul_(-1)

# ── Pass 4: MF10 spectral pumping ─────────────────────────────────────────────
print("\nPass 4: MF10 spectral pumping (η=0.01)...")
eta_mf = 0.01
for mf_iter in range(1, 11):
    # E step: gradient descent on embeddings
    for _ in range(200):
        x,y = get_batch()
        stu.train(); _,loss = stu(x,y)
        stu.zero_grad(); loss.backward()
        with torch.no_grad():
            if stu.te.weight.grad is not None:
                stu.te.weight.data -= eta_mf * stu.te.weight.grad
    # W_K step: gradient ASCENT on keys (oscillatory pumping)
    for _ in range(200):
        x,y = get_batch()
        stu.train(); _,loss = stu(x,y)
        stu.zero_grad(); loss.backward()
        with torch.no_grad():
            for l in range(N_STU):
                if stu.blocks[l].attn.WK.weight.grad is not None:
                    stu.blocks[l].attn.WK.weight.data += eta_mf * stu.blocks[l].attn.WK.weight.grad
    if mf_iter in [5, 10]:
        print(f"  MF iter {mf_iter}: val={eval_val(stu,n=5):.4f}")

# ── Pass 5: Basin selector (33 CE, 5×LR) ─────────────────────────────────────
print("\nPass 5: Basin selector (33 CE)...")
opt5 = torch.optim.AdamW(stu.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
for step in range(1, 34):
    for pg in opt5.param_groups: pg['lr'] = LR*5*min(step,10)/10
    stu.train(); x,y=get_batch(); _,loss=stu(x,y)
    opt5.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt5.step()
v5 = eval_val(stu, n=15)
print(f"  Pass 5 val: {v5:.4f}")

# ── Pass 6: LM projection (8 iters) ──────────────────────────────────────────
print("\nPass 6: LM projection (8 iters)...")
mu = 0.95
N_HVP = 12; N_CG = 6; N_GRAD = 25

for it in range(1, 9):
    # Full gradient
    stu.zero_grad()
    loss = sum(stu(*get_batch())[1] for _ in range(N_GRAD))/N_GRAD
    loss.backward()
    g = torch.cat([p.grad.flatten() for p in stu.parameters()])
    stu.zero_grad()

    # HVP via CG
    def hvp(v):
        v_t = v.detach().requires_grad_(False)
        loss2 = sum(stu(*get_batch())[1] for _ in range(N_HVP))/N_HVP
        grads2 = torch.autograd.grad(loss2, list(stu.parameters()), create_graph=True)
        gv = (torch.cat([gr.flatten() for gr in grads2])*v_t).sum()
        hv = torch.cat([h.flatten() for h in
                        torch.autograd.grad(gv, list(stu.parameters()), retain_graph=False)])
        stu.zero_grad()
        return hv.detach()

    # CG solve (I + mu*H) d = -g
    d = torch.zeros_like(g); r = -g.clone(); p_cg = r.clone()
    rr = (r*r).sum()
    for _ in range(N_CG):
        Hp = hvp(p_cg) + mu * p_cg
        alpha_cg = rr / max(float((p_cg*Hp).sum()), 1e-10)
        d += alpha_cg * p_cg; r -= alpha_cg * Hp
        rr_new = (r*r).sum()
        p_cg = r + (rr_new/max(float(rr),1e-10)) * p_cg
        rr = rr_new

    # Line search
    w0_lm = stu.flat_params()
    L0 = eval_val(stu, n=8)
    stu.set_flat(w0_lm + d)
    L_new = eval_val(stu, n=8)
    if L_new < L0:
        mu = max(mu*0.5, 0.95)
        print(f"  LM {it}: ACCEPT μ={mu:.3f} val={L_new:.4f}")
    else:
        stu.set_flat(w0_lm)
        mu = min(mu*2, 5.0)
        print(f"  LM {it}: REJECT μ={mu:.3f}")

v6 = eval_val(stu, n=20)
print(f"\nPass 6 final val: {v6:.4f}")

# ── Save checkpoint ───────────────────────────────────────────────────────────
torch.save(stu.state_dict(), '/tmp/model_post_pass6.pt')
with open('/tmp/model_post_pass6_val.json','w') as f:
    json.dump({'val': v6, 'passes': '0-6', 'corpus_loops': len(train_ids)//1364}, f)

print(f"\n✓ Saved /tmp/model_post_pass6.pt  (val={v6:.4f})")
print(f"  Run: python slow_manifold_pass11.py")
