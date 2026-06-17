#!/usr/bin/env python3
"""
Spectral Compiler v2 — Correct VOCAB=1017, Full Pipeline
=========================================================

Previous experiments ran with VOCAB=675 (stale /tmp/ files).
This run uses the correct VOCAB=1017 corpus.

FINDING SO FAR:
  - PMI embeddings: HARMFUL (collapse all tokens to small magnitude cluster)
  - Spectral embeddings: 33% better than random on VOCAB=675 corpus
  - Random embeddings: best isotropic baseline

THIS EXPERIMENT: With correct VOCAB=1017
  A: Random init + 200CE (baseline, expected val≈0.51)
  B: Spectral init + 200CE (expected val≈0.34 if 33% transfers)
  C: Spectral init + full compiler pipeline
     (saddle exit + MF10 + basin selector + LM + 25CE)
     (expected: beats teacher val=0.247?)

The spectral embedding provides the token bigram graph Laplacian
eigenvectors — semantically structured AND isotropic (std≈0.02).

COMPILER PIPELINE (teacher-free):
  Pass 0: Corpus analysis → spectral embeddings, corpus stats
  Pass 1: Model init from spectral embeddings
  Pass 2: Saddle exit (v_neg from -H power iteration, α*=1.43)
  Pass 3: Sign correction (étale sheet assignment, blocks 1,2)
  Pass 4: MF10 spectral pumping (oscillatory, η=0.01)
  Pass 5: Basin selector (33 CE steps, 5× LR)
  Pass 6: LM projection (8 adaptive-μ iterations)
  Pass 7: Embedding relaxation (25 CE steps, cosine LR)

TEACHER: trained independently, used only for verification.
"""
import json, math, time, collections, warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; N_LAYERS_T=24; BATCH=8; SEQ=64; LR=3e-4
ALPHA_STAR=1.429

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

print(f"VOCAB={VOCAB}, corpus={len(train_ids)} tokens, val={len(val_ids)} tokens")
assert VOCAB==1017, f"Expected VOCAB=1017, got {VOCAB}. Run with fresh /tmp/ corpus."

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

def eval_val(m,n=30):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

def clr(s,total,warmup=20):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ─── PASS 0: Build Spectral Embeddings from corpus ────────────────────────────
print("\n" + "="*65)
print("PASS 0: Spectral Embedding from VOCAB=1017 bigram graph")
print("="*65)

# Build sparse bigram matrix
rows, cols, vals_list = [], [], []
for i in range(len(train_ids)-1):
    a,b = train_ids[i], train_ids[i+1]
    if a<VOCAB and b<VOCAB:
        rows.append(a); cols.append(b); vals_list.append(1.0)

W = sp.csr_matrix((vals_list,(rows,cols)), shape=(VOCAB,VOCAB), dtype=np.float32)
W = W + W.T  # symmetrise

# Normalised Laplacian
d_inv = np.array(1.0/(W.sum(1)+1e-8)).flatten()
d_sqrt_inv = np.sqrt(d_inv)
D_sqrt_inv = sp.diags(d_sqrt_inv)
L_sym = sp.eye(VOCAB) - D_sqrt_inv @ W @ D_sqrt_inv

print(f"  Graph: {VOCAB} nodes, {W.nnz//2} edges")
print(f"  Computing {D} eigenvectors of L_sym ({VOCAB}×{VOCAB})...")
t0=time.time()

eigenvalues, eigenvectors = spla.eigsh(L_sym, k=D+1, which='SM', tol=1e-4, maxiter=3000)
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]; eigenvectors = eigenvectors[:,idx]
eigenvectors = eigenvectors[:,1:D+1]  # skip constant eigenvector
scales = 1.0/(np.sqrt(eigenvalues[1:D+1])+1e-8)
spectral_np = eigenvectors * scales[np.newaxis,:]
spectral_np = spectral_np / (spectral_np.std()+1e-8) * 0.02
spectral_emb = torch.tensor(spectral_np, dtype=torch.float32)

print(f"  Computed in {time.time()-t0:.1f}s")
print(f"  Eigenvalue range: [{eigenvalues[1]:.4f}, {eigenvalues[D]:.4f}]")
print(f"  Spectral emb: std={spectral_emb.std():.4f}, "
      f"range=[{spectral_emb.min():.4f},{spectral_emb.max():.4f}]")

# ─── PASS 1: HVP saddle finder ────────────────────────────────────────────────
def hv_p(model,v,n=15):
    params=list(model.parameters()); model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad(); return hv

def find_saddle(model, n_iter=15, n_hvp=15):
    n_p=sum(p.numel() for p in model.parameters())
    v=torch.randn(n_p); v=v/v.norm()
    for _ in range(n_iter):
        Hv=hv_p(model,v,n_hvp); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
    return v

# ─── Full pipeline function ───────────────────────────────────────────────────
def run_compiler_pipeline(emb_init, label, n_mf=10, do_lm=True, n_ce_final=25):
    """Full compiler pipeline with given embedding initialisation."""
    torch.manual_seed(99)
    stu = LM(D, N_HEADS, N_STU)
    if emb_init is not None:
        stu.te.weight.data.copy_(emb_init)
    v_init = eval_val(stu, n=15)
    print(f"\n[{label}] init: val={v_init:.4f}")

    # PASS 2: Saddle exit
    print(f"  Pass 2: saddle finder...")
    v_neg = find_saddle(stu, n_iter=15, n_hvp=15)
    w0 = stu.flat_params()
    stu.set_flat(w0 + ALPHA_STAR*(v_neg/v_neg.norm()))
    v_saddle = eval_val(stu, n=10)
    print(f"  Pass 2: after saddle exit α*={ALPHA_STAR}: val={v_saddle:.4f}")

    # PASS 3: Sign correction (étale sheet, blocks 1,2)
    with torch.no_grad():
        for l in [1,2]:
            stu.blocks[l].attn.WV.weight.mul_(-1)
            stu.blocks[l].attn.op.weight.mul_(-1)

    # PASS 4: MF pumping (oscillatory, η=0.01)
    print(f"  Pass 4: MF{n_mf} spectral pumping (η=0.01)...")
    for it in range(n_mf):
        # E step
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.requires_grad_(False)
            stu.blocks[l].attn.WQ.weight.requires_grad_(False)
        eg=torch.zeros(VOCAB,D); ef=torch.zeros(VOCAB,D)
        for i in range(200):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            if stu.te.weight.grad is not None:
                g=stu.te.weight.grad.detach(); eg+=g; ef+=g**2
        eg/=200; ef/=200
        with torch.no_grad(): stu.te.weight.add_(-0.01*eg/(ef+1e-4))
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.requires_grad_(True)
            stu.blocks[l].attn.WQ.weight.requires_grad_(True)
        # WK step
        stu.te.weight.requires_grad_(False)
        wg=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        wf=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        for i in range(200):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            g=sum(stu.blocks[l].attn.WK.weight.grad/N_STU
                  for l in range(N_STU) if stu.blocks[l].attn.WK.weight.grad is not None)
            if isinstance(g, torch.Tensor): wg+=g; wf+=g**2
        wg/=200; wf/=200
        with torch.no_grad():
            for l in range(N_STU):
                stu.blocks[l].attn.WK.weight.add_(-0.01*wg/(wf+1e-4))
                stu.blocks[l].attn.WQ.weight.add_(-0.01*wg.T/(wf.T+1e-4))
        stu.te.weight.requires_grad_(True)
        if (it+1)%5==0:
            print(f"    MF iter {it+1}: val={eval_val(stu,n=5):.4f}")

    # PASS 5: Basin selector (33 CE steps, 5× LR)
    print(f"  Pass 5: basin selector (33 CE, 5×LR)...")
    opt5 = torch.optim.AdamW(stu.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,34):
        for pg in opt5.param_groups: pg['lr']=LR*5*min(step,10)/10
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt5.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt5.step()
    v5 = eval_val(stu, n=15)
    print(f"  Pass 5: val={v5:.4f}")

    # PASS 6: LM projection (8 adaptive-μ Newton steps)
    if do_lm:
        print(f"  Pass 6: LM projection (8 iters)...")
        mu=1.02
        def full_grad(n=25):
            stu.train(); stu.zero_grad()
            for _ in range(n): x,y=get_batch(); _,l=stu(x,y); (l/n).backward()
            g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                         for p in stu.parameters()]).detach().clone()
            stu.zero_grad(); return g
        def cg_step(g, mu, n_cg=6, n_hv=12):
            b=-g; delta=torch.zeros_like(b); r=b.clone(); p=r.clone()
            rsq=float((r*r).sum())
            for _ in range(n_cg):
                Hp=hv_p(stu,p,n_hv)+mu*p; pHp=float((p*Hp).sum())
                if pHp<=0: break
                a=rsq/pHp; delta=delta+a*p; r=r-a*Hp
                rsq_new=float((r*r).sum()); p=r+(rsq_new/max(rsq,1e-10))*p; rsq=rsq_new
                if rsq**0.5<1e-5: break
            return delta
        def L_now(): return eval_val(stu, n=15)
        L_prev=L_now()
        for it in range(8):
            g=full_grad(); delta=cg_step(g, mu)
            w0=stu.flat_params().clone(); stu.set_flat(w0+delta)
            L_new=L_now()
            if L_new<L_prev:
                L_prev=L_new; mu=max(mu*0.5,0.95)
                print(f"    LM {it+1}: ACCEPT μ={mu:.3f} val={L_new:.4f}")
            else:
                stu.set_flat(w0); mu=min(mu*2,5.0)
                print(f"    LM {it+1}: REJECT μ={mu:.3f}")

    # PASS 7: Embedding relaxation (25 CE steps, cosine LR)
    print(f"  Pass 7: embedding relaxation (25 CE)...")
    opt7 = torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,26):
        for pg in opt7.param_groups: pg['lr']=clr(step,25)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt7.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt7.step()
        if step in [10,25]:
            print(f"    CE {step}: val={eval_val(stu,n=8):.4f}")

    v_final = eval_val(stu, n=40)
    print(f"  [{label}] FINAL: val={v_final:.4f}")
    return v_final

# ─── Simple CE baseline ───────────────────────────────────────────────────────
def run_ce_only(emb_init, label, n_steps=200):
    torch.manual_seed(99)
    stu = LM(D, N_HEADS, N_STU)
    if emb_init is not None:
        stu.te.weight.data.copy_(emb_init)
    opt = torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,n_steps+1):
        for pg in opt.param_groups: pg['lr']=clr(step,n_steps)
        stu.train(); x,y=get_batch(); _,loss=stu(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt.step()
        if step in [50,100,150,200]:
            v=eval_val(stu,n=10); print(f"  {label} CE {step}: {v:.4f}")
    return eval_val(stu,n=30)

# ─── EXPERIMENTS ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("EXPERIMENT A: Random init + 200CE (baseline, VOCAB=1017)")
print("="*65)
vA = run_ce_only(None, "A")
print(f"A FINAL: {vA:.4f}")

print("\n" + "="*65)
print("EXPERIMENT B: Spectral init + 200CE")
print("="*65)
vB = run_ce_only(spectral_emb, "B")
print(f"B FINAL: {vB:.4f}")

print("\n" + "="*65)
print("EXPERIMENT C: Spectral init + FULL COMPILER PIPELINE")
print("="*65)
vC = run_compiler_pipeline(spectral_emb, "C")

print("\n" + "="*65)
print("VERIFICATION: Train teacher (300 steps)")
print("="*65)
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt_t=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,301):
    lr_now=LR*min(step,100)/100 if step<=100 else LR*0.5*(1+math.cos(math.pi*(step-100)/200))
    for pg in opt_t.param_groups: pg['lr']=lr_now
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt_t.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt_t.step()
    if step%100==0:
        teacher.eval()
        with torch.no_grad():
            vl=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  Teacher step {step}: val={vl:.4f}")
v_teacher = eval_val(teacher, n=40)

print(f"""
{'='*65}
  SPECTRAL COMPILER RESULTS (VOCAB=1017, correct corpus)
{'='*65}

  BASELINE (200 CE steps each):
    A (random init):      val={vA:.4f}
    B (spectral init):    val={vB:.4f}
    Spectral improvement: {(vA-vB)/vA*100:.1f}%

  FULL PIPELINE (spectral + compiler):
    C (spectral+compiler): val={vC:.4f}

  ORACLE:
    Teacher (24L,300 steps,VOCAB=1017): val={v_teacher:.4f}
    Serre cascade (teacher emb):        val≈0.187

  {'✓ COMPILER BEATS TEACHER' if vC < v_teacher else '✗ Below teacher'}
  {'✓ COMPILER BEATS SERRE'   if vC < 0.187 else '✗ Below Serre cascade'}

  TEACHER-FREE STATUS:
    A (random, 200CE):      {'beats teacher' if vA<v_teacher else 'below teacher'}
    B (spectral, 200CE):    {'beats teacher' if vB<v_teacher else 'below teacher'}
    C (spectral+pipeline):  {'BEATS TEACHER — compiler is teacher-free' if vC<v_teacher
                              else 'below teacher — need more passes'}
""")
