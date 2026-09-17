#!/usr/bin/env python3
"""
Geometry-Driven Compiler with Decorrelated Corpus Support
=========================================================
Adapted for build_corpus2.py generated data.
"""
import json, math, warnings, collections, os, sys, time, copy
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
ETA_MF=0.002; N_SUB=200
PHI_CLEAN_TARGET=5; TAU_MIN=1.5; TAU_MAX=5.7; VAL_FLOOR=0.062; ORBIT_TOLERANCE=0.3

for f in ['/tmp/train_ids.json', '/tmp/val_ids.json', '/tmp/vocab.json', '/tmp/corpus_meta.json']:
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} missing. Run: python build_corpus2.py --out /tmp --mode decorr")

with open('/tmp/train_ids.json')   as f: train_ids = list(map(int, json.load(f)))
with open('/tmp/val_ids.json')     as f: val_ids   = list(map(int, json.load(f)))
with open('/tmp/vocab.json')       as f: _v = json.load(f)
with open('/tmp/corpus_meta.json') as f: corpus_meta = json.load(f)

VOCAB = len(_v) if isinstance(_v, list) else len(_v)
H_BIGRAM = corpus_meta.get("H_bigram", 0.0147)
FLOOR_TARGET_VAL = H_BIGRAM

train_t = torch.tensor(train_ids, dtype=torch.long)
val_t   = torch.tensor(val_ids,   dtype=torch.long)

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
            lam=torch.linalg.eigvals(phi); lam1=lam[lam.abs().argmax()]
            a=float(torch.angle(lam1))
            out.append('π' if abs(abs(a)-math.pi)<0.3 else '0' if abs(a)<0.3 else f'{a:.2f}')
        except: out.append('?')
    return out

def phi_clean(model):
    return sum(1 for p in sheet_angles(model) if p in ('0','π'))

def gluing_defect(model, n=8):
    model.zero_grad()
    ls=[model(*get_batch())[1] for _ in range(n)]
    torch.stack(ls).mean().backward()
    g_ff=sum(p.grad.data.norm().item() for nm,p in model.named_parameters()
             if '.ff.' in nm and p.grad is not None)
    g_emb=model.te.weight.grad.data.norm().item() if model.te.weight.grad is not None else 1e-8
    model.zero_grad()
    return g_ff/max(g_emb,1e-8)

def gradient_alignment(model, g_floor, n=8):
    model.zero_grad()
    ls=[model(*get_batch())[1] for _ in range(n)]
    torch.stack(ls).mean().backward()
    g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                 for p in model.parameters()]).detach()
    model.zero_grad()
    return float((g*g_floor).sum()/(g.norm()*g_floor.norm()+1e-10))

def lm_step(model, mu=0.950, n_grad=25, n_hvp=12, n_cg=6):
    model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n_grad))/n_grad
    loss.backward()
    g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                 for p in model.parameters()]).detach(); model.zero_grad()
    def _hvp(v):
        model.zero_grad()
        ls=[model(*get_batch())[1] for _ in range(n_hvp)]; loss2=torch.stack(ls).mean()
        grads=torch.autograd.grad(loss2,list(model.parameters()),create_graph=True)
        gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
        hv=torch.cat([h.flatten() for h in
                      torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)])
        model.zero_grad(); return hv.detach()
    d=torch.zeros_like(g); r=-g.clone(); p=r.clone(); rr=float((r*r).sum())
    for _ in range(n_cg):
        Hp=_hvp(p)+mu*p; al=rr/max(float((p*Hp).sum()),1e-10)
        d+=al*p; r-=al*Hp; rr2=float((r*r).sum()); p=r+(rr2/max(rr,1e-10))*p; rr=rr2
    w0=model.flat_params(); v0=eval_val(model,n=8)
    model.set_flat(w0+d); v1=eval_val(model,n=8)
    if v1<v0: return v1, True
    model.set_flat(w0); return v0, False

# ── CORPUS + SPECTRAL E₀ ─────────────────────────────────────
print("="*65)
print("GEOMETRY-DRIVEN COMPILER (DECORRELATED CORPUS MODE)")
print(f"Target Floor (H_bigram): {H_BIGRAM:.4f} nats")
print("="*65); print()

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
E_init=(0.9*E_0+0.1*E_next); E_norm=float(np.linalg.norm(E_0))
E_init=(E_init*(E_norm/max(float(np.linalg.norm(E_init)),1e-8))).astype(np.float32)
print(f"Corpus: VOCAB={VOCAB}, nnz={len(bigram)}")

# Measure floor gradient
print("Measuring floor gradient (geometric anchor)...")
torch.manual_seed(42)
m_floor=LM(); m_floor.te.weight.data.copy_(torch.tensor(E_init))
opt_f=torch.optim.AdamW(m_floor.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for _ in range(200):
    m_floor.train(); x,y=get_batch(); _,l=m_floor(x,y)
    opt_f.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m_floor.parameters(),1.0); opt_f.step()
m_floor.zero_grad()
ls=[m_floor(*get_batch())[1] for _ in range(20)]; torch.stack(ls).mean().backward()
g_floor=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                   for p in m_floor.parameters()]).detach(); m_floor.zero_grad()
v_floor=eval_val(m_floor,n=20)
print(f"Floor gradient computed: val={v_floor:.4f}  ||g_floor||={float(g_floor.norm()):.4f}")
print()

# ── INIT MODEL ───────────────────────────────────────────────
torch.manual_seed(99)
model=LM(); model.te.weight.data.copy_(torch.tensor(E_init))
v0=eval_val(model)
print(f"Spectral E₀: val={v0:.4f}")
print()

# ── PHASE 1: SADDLE EXIT ─────────────────────────────────────
print("━━━ PHASE 1: SADDLE EXIT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
t1=time.time()
def hvp_s(v, n=8):
    model.zero_grad()
    ls=[model(*get_batch())[1] for _ in range(n)]; loss=torch.stack(ls).mean()
    grads=torch.autograd.grad(loss,list(model.parameters()),create_graph=True)
    gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in
                  torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)])
    model.zero_grad(); return hv.detach()

n_p=sum(p.numel() for p in model.parameters())
torch.manual_seed(42); v=torch.randn(n_p); v=v/v.norm()
for _ in range(15):
    Hv=hvp_s(v); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg=v.clone()
w0=model.flat_params(); best_v=eval_val(model,n=8); best_a=0.
for alpha in [0.5,1.0,1.429,2.0,3.0,4.0]:
    model.set_flat(w0+alpha*(v_neg/v_neg.norm())); vt=eval_val(model,n=6)
    if vt<best_v: best_v=vt; best_a=alpha
model.set_flat(w0+best_a*(v_neg/v_neg.norm()))
v_saddle=eval_val(model)
print(f"  α*={best_a:.3f}  val={v_saddle:.4f}  sheet={sheet_angles(model)}")
print(f"  [{time.time()-t1:.1f}s]"); print()

# ── PHASE 2: ADAPTIVE MF PUMP ────────────────────────────────
print("━━━ PHASE 2: ADAPTIVE MF PUMP ━━━━━━━━━━━━━━━━━━━━━━━━━")
best_phi = phi_clean(model); best_tau = gluing_defect(model)
tau_history = [best_tau]; phi_history = [best_phi]
mf_r = 0; tau_peaked = False

for mf_r in range(1, 16):
    for l in range(N_STU):
        model.blocks[l].attn.WK.weight.requires_grad_(False)
        model.blocks[l].attn.WQ.weight.requires_grad_(False)
    emb_grad=torch.zeros(model.te.weight.shape)
    emb_fish=torch.zeros(model.te.weight.shape)
    torch.manual_seed((mf_r-1)*1000)
    for i in range(N_SUB):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
        model.zero_grad(); _,loss=model(x,y); loss.backward()
        if model.te.weight.grad is not None:
            g=model.te.weight.grad.detach(); emb_grad+=g; emb_fish+=g**2
    emb_grad/=N_SUB; emb_fish/=N_SUB
    delta_E=-(emb_grad/(emb_fish+1e-4))
    with torch.no_grad(): model.te.weight.add_(ETA_MF*delta_E)
    for l in range(N_STU):
        model.blocks[l].attn.WK.weight.requires_grad_(True)
        model.blocks[l].attn.WQ.weight.requires_grad_(True)
    v_e=eval_val(model,n=4)

    model.te.weight.requires_grad_(False)
    wk_grad=torch.zeros_like(model.blocks[0].attn.WK.weight)
    wk_fish=torch.zeros_like(model.blocks[0].attn.WK.weight)
    torch.manual_seed((mf_r-1)*1000+500)
    for i in range(N_SUB):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
        model.zero_grad(); _,loss=model(x,y); loss.backward()
        g=torch.zeros_like(model.blocks[0].attn.WK.weight)
        for bl in model.blocks:
            if bl.attn.WK.weight.grad is not None: g+=bl.attn.WK.weight.grad/N_STU
        wk_grad+=g; wk_fish+=g**2
    wk_grad/=N_SUB; wk_fish/=N_SUB
    delta_WK=-(wk_grad/(wk_fish+1e-4))
    with torch.no_grad():
        for l in range(N_STU):
            model.blocks[l].attn.WK.weight.add_(ETA_MF*delta_WK)
            model.blocks[l].attn.WQ.weight.add_(ETA_MF*delta_WK.T)
    model.te.weight.requires_grad_(True)
    v_wk=eval_val(model,n=4)

    tau=gluing_defect(model,n=6); pc=phi_clean(model)
    tau_history.append(tau); phi_history.append(pc)
    print(f"  MF{mf_r:2d}: E={v_e:.3f} WK={v_wk:.3f}  Φ_cl={pc}/5  τ={tau:.2f}")

    if pc == N_STU-1:
        print(f"  ✓ STOP: Φ_clean=5/5 orbit established")
        break
    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:
        tau_peaked = True
        print(f"  ✓ STOP: τ rising ({tau_history[-3]:.2f}→{tau_history[-2]:.2f}→{tau:.2f}) — orbit shattering")
        break

v_mf=eval_val(model); n_mf_used=mf_r
print(f"  After MF{n_mf_used}: val={v_mf:.4f}  Φ={sheet_angles(model)}")
print()

def compute_rm2_sigma_inline(model, rank=6):
    wk_list = []
    for name, param in model.named_parameters():
        n = name.lower()
        if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n and param.ndim >= 2:
            wk_list.append(param.detach().float().cpu().numpy())
    if len(wk_list) < 2: return 0.0
    wk_list.sort(key=lambda w: w.shape[0])
    rm2_vals = []
    for k in range(len(wk_list)-1):
        W0, W1 = wk_list[k], wk_list[k+1]
        try:
            U0, s0, _ = np.linalg.svd(W0, full_matrices=False)
            U1, s1, _ = np.linalg.svd(W1, full_matrices=False)
            r = min(rank, U0.shape[1], U1.shape[1])
            Ur0, Ur1 = U0[:, :r], U1[:, :r]
            sv = np.linalg.svd(Ur0.T @ Ur1, compute_uv=False)
            sv = np.clip(sv, 1e-6, 1-1e-6)
            h_strip = sv / (1 - sv**2)**1.5
            h_loss  = s0[:r] / (np.linalg.norm(s0[:r]) + 1e-10)
            h_strip = h_strip / (np.linalg.norm(h_strip) + 1e-10)
            weights = 1.0 / (sv**2 + 1e-6)
            num  = np.dot(h_loss * weights, h_strip)
            den  = (np.sqrt(np.dot(h_loss**2, weights)) *
                    np.sqrt(np.dot(h_strip**2, weights)) + 1e-10)
            rm2_vals.append(float(num / den))
        except Exception: pass
    return float(np.mean(rm2_vals)) if rm2_vals else 0.0

class CompressedAdam:
    def __init__(self, params, lr, betas=(0.9,0.95), eps=1e-8,
                 weight_decay=0.1, mbits=4, vbits=4, vrow=False, reduced=False,
                 names=None):
        if names is not None:
            pn=[(n,q) for n,q in zip(names,params) if q.requires_grad]
            self.names=[n for n,_ in pn]; self.p=[q for _,q in pn]
        else:
            self.p=[q for q in params if q.requires_grad]
            self.names=[f"p{i}" for i in range(len(self.p))]
        self.lr=lr; self.b1,self.b2=betas; self.eps=eps; self.wd=weight_decay
        self.mbits=mbits; self.vbits=vbits; self.vrow=vrow; self.reduced=reduced
        self.m=[torch.zeros_like(q) for q in self.p]
        self.v=[torch.zeros_like(q) for q in self.p]
        self.t=0; self.cum=0.0; self._vfrozen=None
        self.bidx={}
        for _i,_nm in enumerate(self.names):
            self.bidx.setdefault(self.bucket_of(_nm),[]).append(_i)
        self.buckets=sorted(self.bidx)
        self.hist={k:[] for k in self.buckets}
        self.sub={k:None for k in self.buckets}
        self.track=True
        self._pg=[{"lr":lr}]
    @staticmethod
    def _q(x,bits):
        lv=torch.log(x.clamp_min(1e-20)); lo,hi=float(lv.min()),float(lv.max())
        if hi-lo<1e-12: return x
        s=(hi-lo)/(2**bits-1)
        return torch.exp(torch.round((lv-lo)/s).clamp(0,2**bits-1)*s+lo)
    def fire(self): self.reduced=True
    @staticmethod
    def bucket_of(name):
        if name.startswith("te"): return "EMB_tok"
        if name.startswith("pe"): return "EMB_pos"
        if "ln" in name.lower():  return "LN"
        if ".ff." in name:        return "FF"
        if ".WQ." in name or ".WK." in name: return "ATT_QK"
        return "ATT_VO"
    def freeze_buckets(self, buckets):
        b2=1-self.b2**max(self.t,1); n=0
        if self._vfrozen is None: self._vfrozen=[None]*len(self.p)
        for i,nm in enumerate(self.names):
            if self.bucket_of(nm) in buckets and self._vfrozen[i] is None:
                self._vfrozen[i]=(self.v[i]/b2).clone(); n+=self.p[i].numel()
        return n
    def freeze_v(self):
        b2=1-self.b2**max(self.t,1)
        self._vfrozen=[(x/b2).clone() for x in self.v]
    @property
    def param_groups(self): return self._pg
    def zero_grad(self, set_to_none=False):
        for q in self.p:
            if q.grad is not None:
                if set_to_none: q.grad=None
                else: q.grad.detach_(); q.grad.zero_()
    @torch.no_grad()
    def step(self):
        lr=self._pg[0]["lr"]; self.t+=1
        b1=1-self.b1**self.t; b2=1-self.b2**self.t
        alpha=lr*math.sqrt(b2)/b1; eps_t=self.eps*math.sqrt(b2)
        tot=0.0
        for k in self.buckets:
            acc=[]
            for i in self.bidx[k]:
                q=self.p[i]
                g=q.grad if q.grad is not None else torch.zeros_like(q)
                self.m[i].mul_(self.b1).add_(g,alpha=1-self.b1)
                self.v[i].mul_(self.b2).addcmul_(g,g,value=1-self.b2)
                _f=self._vfrozen
                vk=(_f[i] if (_f is not None and _f[i] is not None) else self.v[i])
                mk=self.m[i]
                if q.dim()==2 and q.shape[0]>1:
                    if self.vrow: vk=vk.mean(dim=-1,keepdim=True).expand_as(vk).contiguous()
                    vk=self._q(vk,self.vbits)
                    mk=(torch.sign(mk)*float(mk.abs().mean())) if self.reduced \
                       else torch.sign(mk)*self._q(mk.abs(),self.mbits)
                uk=mk/(vk.sqrt()+eps_t)
                if self.track: acc.append(uk.reshape(-1))
                d=-alpha*uk - lr*self.wd*q.data
                tot+=float((d*d).sum()); q.data.add_(d)
            if self.track and acc: self._push(k,torch.cat(acc))
        self.cum+=tot**0.5

    def _push(self,k,u):
        if self.sub[k] is None:
            g=torch.Generator().manual_seed(9)
            self.sub[k]=torch.randperm(u.numel(),generator=g)[:min(4000,u.numel())]
        h=self.hist[k]; h.append(u[self.sub[k]].clone())
        if len(h)>16: h.pop(0)

class StabDetector:
    def __init__(self, model, layer=3, sub=40000, delta=16):
        pre=f"blocks.{layer}."
        self.ref=[(n,p) for n,p in model.named_parameters()
                  if n.startswith(pre) and p.requires_grad]
        tot=sum(p.numel() for _,p in self.ref)
        g=torch.Generator().manual_seed(9)
        self.idx=torch.randperm(tot,generator=g)[:min(sub,tot)]
        self.W0=torch.cat([p.data.reshape(-1) for _,p in self.ref])[self.idx].clone()
        self.hist={}; self.delta=delta; self.trace=[]
    @torch.no_grad()
    def check(self, step):
        cur=torch.cat([p.data.reshape(-1) for _,p in self.ref])[self.idx]
        S=torch.sign(cur-self.W0); self.hist[step]=S
        prev=self.hist.get(step-self.delta)
        if prev is None: return None
        ss=float((S==prev).float().mean()); self.trace.append((step,ss)); return ss

# ── PHASE 3: BASIN SETTLE ──────────────────────────────────────
print("━━━ PHASE 3: BASIN SETTLE (GEO-STOP) ━━━━━━━━━━━━━━━━━━")
P3MIN           = 300
P3PLAT          = 0.0005
P3VALSTOP       = 0.0
_prev_small     = False
PRUNE_ATTN      = True

if PRUNE_ATTN:
    def _attn_probe(_mdl, _H=4):
        _x, _ = get_batch()
        _out = []
        with torch.no_grad():
            _h = _mdl.te(_x) + _mdl.pe(torch.arange(_x.shape[1]))
            for _bm in _mdl.blocks:
                _B, _T, _Dm = _h.shape; _dh = _Dm // _H
                _q = _bm.attn.WQ(_h).view(_B,_T,_H,_dh).transpose(1,2)
                _k = _bm.attn.WK(_h).view(_B,_T,_H,_dh).transpose(1,2)
                _s = (_q @ _k.transpose(-2,-1)) / math.sqrt(_dh)
                _s = _s.masked_fill(torch.triu(torch.ones(_T,_T),1).bool(), float("-inf"))
                _P = _s.softmax(-1).clamp_min(1e-12)
                _s2 = (_P**2).sum(-1); _s3 = (_P**3).sum(-1)
                _jf = float(torch.sqrt((_s2-2*_s3+_s2**2).clamp_min(0)).mean())
                _out.append((float(_P.max(-1).values.mean()), _jf/math.sqrt(_dh), _T))
                _h = _bm(_h)
        return _out
    model.zero_grad()
    _xx,_yy = get_batch(); model(_xx,_yy)[1].backward()
    _pr = _attn_probe(model)
    _N = _pr[0][2]
    _floor = sum(1.0/_i for _i in range(1,_N+1))/_N
    _pd = dict(model.named_parameters())
    _pruned = []
    for _bi,(_pm,_pred,_) in enumerate(_pr):
        _gq = _pd[f"blocks.{_bi}.attn.WQ.weight"].grad
        _gv = _pd[f"blocks.{_bi}.attn.WV.weight"].grad
        _r = (float(_gq.norm()) if _gq is not None else 0.0) / max(
              float(_gv.norm()) if _gv is not None else 1.0, 1e-30)
        _mp = (_pm-_floor)/_floor; _mr = _r/max(_pred,1e-30)
        if _mp < 0.10 and _mr < 1.0: _pruned.append(_bi)
    model.zero_grad()
    if _pruned:
        _np_ = 0
        for _bi in _pruned:
            for _W in (model.blocks[_bi].attn.WQ.weight, model.blocks[_bi].attn.WK.weight):
                _np_ += _W.numel()
                _W.register_hook(lambda g: torch.zeros_like(g))
        print(f"  [zoneadam] PRUNED attention QK in blocks {_pruned} ({_np_:,} params frozen)")

NO_GEOSTOP      = True
VFREEZE_AT      = 0
VFREEZE_BUCKETS = {'ATT_QK', 'ATT_VO'}
_vfroz_done     = False
opt_b = CompressedAdam(list(model.parameters()), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1,
                       names=[n for n,_ in model.named_parameters()])
_stab = StabDetector(model, layer=3)
val_history = [v_mf]
step_basin = 0
geo_stopped = False
geo_stop_step = None

for step in range(1, 300+1):
    if step <= 10:
        for pg in opt_b.param_groups: pg['lr'] = LR*5*step/10
    model.train(); x, y = get_batch(); _, l = model(x, y)
    opt_b.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_b.step()

    if step % 8 == 0:
        v = eval_val(model, n=8)
        delta = abs(v - val_history[-1]) / 8
        val_history.append(v)
        pc  = phi_clean(model)
        tau = gluing_defect(model, n=4)
        rm2 = compute_rm2_sigma_inline(model)
        _ss = _stab.check(step)
        
        if VFREEZE_AT and not _vfroz_done and step >= VFREEZE_AT:
            _n=opt_b.freeze_buckets(VFREEZE_BUCKETS); _vfroz_done=True

        print(f"  step {step:3d}: val={v:.4f}  Δ={delta:.4f}  Φ_cl={pc}/5  τ={tau:.2f}  rm2σ={rm2:+.3f}")

        _small = delta < P3PLAT
        if step >= P3MIN and _small and _prev_small:
            print(f"  ✓ Plateau (2 consecutive < {P3PLAT})"); break
        _prev_small = _small
        if P3VALSTOP > 0 and v < P3VALSTOP:
            print(f"  ✓ val={v:.4f} < {P3VALSTOP}"); break

step_basin = step
v_basin = eval_val(model); pc_b = phi_clean(model); tau_b = gluing_defect(model)
rm2_b = compute_rm2_sigma_inline(model)
print(f"  After {step}CE: val={v_basin:.4f}  Φ_cl={pc_b}/5  τ={tau_b:.2f}  rm2σ={rm2_b:+.3f}")

if pc_b < 3:
    print(f"  ⚠ Φ_cl={pc_b}/5 — extending 16CE")
    for _ in range(16):
        model.train(); x, y = get_batch(); _, l = model(x, y)
        opt_b.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_b.step()
    v_basin = eval_val(model); pc_b = phi_clean(model)
    step_basin += 16

torch.save(model.state_dict(), 'basin_entry_state.pt')

if tau_b > 5:
    n_retry = 25 if pc_b >= 5 else 75 if pc_b <= 2 else 50
    print(f"  ⚠ HIGH τ={tau_b:.2f}  Φ_cl={pc_b}/5 → τ-retry {n_retry}CE@LR×2")
    opt_retry = torch.optim.AdamW(model.parameters(), lr=LR*2, betas=(0.9,0.95), weight_decay=0.1)
    for _s in range(n_retry):
        lr_s = LR*2*0.5*(1+math.cos(math.pi*_s/n_retry))
        for pg in opt_retry.param_groups: pg['lr'] = lr_s
        model.train(); x, y = get_batch(); _, l = model(x, y)
        opt_retry.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_retry.step()
    v_basin = eval_val(model); pc_b = phi_clean(model); tau_b = gluing_defect(model)
    step_basin += n_retry

# ── SNAPPER POLYNOMIAL JUMP (SKIPPED ON DECORRELATED DATA) ───
print("━━━ SNAPPER POLYNOMIAL JUMP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  [zoneadam] SNAPPER SKIPPED -- decorrelated process landscape lacks non-convex memorization floor")
v_jump = v_basin

if v_jump <= FLOOR_TARGET_VAL:
    print(f"  ✓ REACHED FLOOR!")
    v_final = v_jump
else:
    print(f"  ⚠ Proceeding to TopoGate & Phase 5")
    
    # ── PHASE 4: TOPOGATE ──────────────────────────────────────────
    print("━━━ PHASE 4: TOPOGATE (geometry-checked) ━━━━━━━━━━━━━━")
    phi_before = sheet_angles(model)
    pc_before = phi_clean(model)
    v_before = eval_val(model, n=8)

    best_score = 0; best_layers = None; best_val = v_before
    for flip_layers in [[1,2],[0,1],[2,3],[0,2],[1,3],[0,3],[0,4],[1,4]]:
        with torch.no_grad():
            for l in flip_layers:
                model.blocks[l].attn.WV.weight.data.mul_(-1)
                model.blocks[l].attn.op.weight.data.mul_(-1)
        v_try = eval_val(model, n=6)
        pc_try = phi_clean(model)
        score = (v_before - v_try) + 0.3 * ((pc_try - pc_before)/5.0)
        if score > best_score:
            best_score = score; best_layers = flip_layers; best_val = v_try
        with torch.no_grad():
            for l in flip_layers:
                model.blocks[l].attn.WV.weight.data.mul_(-1)
                model.blocks[l].attn.op.weight.data.mul_(-1)

    if best_layers and best_score > 0:
        with torch.no_grad():
            for l in best_layers:
                model.blocks[l].attn.WV.weight.data.mul_(-1)
                model.blocks[l].attn.op.weight.data.mul_(-1)
        print(f"  ✓ TopoGate {best_layers}: val {v_before:.4f}→{best_val:.4f}")
    else:
        print(f"  ~ No TopoGate gain — proceeding")

    v_sign=eval_val(model)

    # ── PHASE 5: ALIGNMENT + LM + K₀ SPLIT ──────────────────────
    def k0_split_fn(base, n_steps, lr_emb_ff, lr_attn, w_ff, cosine_schedule=True):
        params_base={n:p.data.clone() for n,p in base.named_parameters()}
        def _ptype(name):
            if '.attn.WQ.' in name or '.attn.WK.' in name: return 'Attn'
            if 'te.weight' in name or '.ff.' in name: return 'EmbFF'
            return 'other'
        def get_lr_cos(step,n,base_lr):
            return base_lr*0.5*(1+math.cos(math.pi*step/n)) if cosine_schedule else base_lr

        m1=copy.deepcopy(base)
        for name,p in m1.named_parameters():
            if _ptype(name)!='EmbFF': p.requires_grad_(False)
        p1=[p for p in m1.parameters() if p.requires_grad]
        opt1=torch.optim.AdamW(p1,lr=lr_emb_ff,betas=(0.9,0.95),weight_decay=0.1)
        for s in range(1,n_steps+1):
            for pg in opt1.param_groups: pg['lr']=get_lr_cos(s,n_steps,lr_emb_ff)
            m1.train(); x,y=get_batch(); _,l=m1(x,y)
            opt1.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(p1,1.0); opt1.step()

        m2=copy.deepcopy(base)
        for name,p in m2.named_parameters():
            if _ptype(name)!='Attn': p.requires_grad_(False)
        p2=[p for p in m2.parameters() if p.requires_grad]
        opt2=torch.optim.AdamW(p2,lr=lr_attn,betas=(0.9,0.95),weight_decay=0.1)
        for s in range(1,n_steps+1):
            for pg in opt2.param_groups: pg['lr']=get_lr_cos(s,n_steps,lr_attn)
            m2.train(); x,y=get_batch(); _,l=m2(x,y)
            opt2.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(p2,1.0); opt2.step()

        m_out=copy.deepcopy(base)
        with torch.no_grad():
            for name,p in m_out.named_parameters():
                pt=_ptype(name)
                d1=dict(m1.named_parameters())[name].data-params_base[name]
                d2=dict(m2.named_parameters())[name].data-params_base[name]
                if pt=='EmbFF':
                    if 'te.weight' in name: p.data.add_(d1)
                    else: p.data.add_(w_ff*d1)
                elif pt=='Attn': p.data.add_(d2)
        return m_out

    print("━━━ PHASE 5: ALIGNMENT + LM + K₀ SPLIT DESCENT ━━━━━━━━━")
    tau_now = gluing_defect(model, n=8)
    w_ff_k0 = 3.5 * (1.5/max(tau_now, 0.5))**1.5
    cos_align=gradient_alignment(model, g_floor)
    v_pre_lm=eval_val(model,n=8)

    if v_pre_lm < H_BIGRAM + 0.05:
        print(f"  val={v_pre_lm:.4f} near floor — skipping LM")
        v_lm = v_pre_lm
    else:
        v_lm, acc = lm_step(model)

    tau_now2=gluing_defect(model,n=6)
    w_ff_k0_2=3.5*(1.5/max(tau_now2,0.5))**1.5

    if tau_now2 < 3.0:
        model=k0_split_fn(model, 25, LR, LR, w_ff_k0_2, cosine_schedule=True)
        v_final=eval_val(model,n=15)
    else:
        model_joint=copy.deepcopy(model)
        opt_j=torch.optim.AdamW(model_joint.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
        for _s in range(1,26):
            for pg in opt_j.param_groups: pg['lr']=LR*0.5*(1+math.cos(math.pi*_s/25))
            model_joint.train(); x,y=get_batch(); _,l=model_joint(x,y)
            opt_j.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model_joint.parameters(),1.0); opt_j.step()
        model=model_joint; v_final=eval_val(model,n=15)

    # ── LANCZOS TERMINAL PROJECTION ──────────────────────────────
    if v_final > H_BIGRAM + 0.05:
        print("━━━ LANCZOS TERMINAL PROJECTION ━━━━━━━━━━━━━━━━━━━━━━━━")
        def hvp_l(model, v, n=4):
            model.zero_grad()
            ls=[model(*get_batch())[1] for _ in range(n)]; loss=torch.stack(ls).mean()
            grads=torch.autograd.grad(loss,list(model.parameters()),create_graph=True)
            gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
            hv=torch.cat([h.flatten() for h in
                          torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)])
            model.zero_grad(); return hv.detach()

        n_p=sum(p.numel() for p in model.parameters())
        torch.manual_seed(7); q=torch.randn(n_p); q=q/q.norm()
        Q=[q]; alphas=[]; betas=[]
        for j in range(8):
            z=hvp_l(model,Q[j]); alpha=float((Q[j]*z).sum()); alphas.append(alpha)
            z=z-alpha*Q[j]
            if j>0: z=z-betas[-1]*Q[j-1]
            for qi in Q: z=z-float((qi*z).sum())*qi
            beta=float(z.norm()); betas.append(beta)
            if beta<1e-8: break
            Q.append(z/beta)
        n_l=len(alphas)
        T=torch.zeros(n_l,n_l)
        for i in range(n_l): T[i,i]=alphas[i]
        for i in range(n_l-1): T[i,i+1]=betas[i]; T[i+1,i]=betas[i]
        T_evals,T_evecs=torch.linalg.eigh(T)
        V=torch.stack(Q[:n_l],dim=1)@T_evecs

        mu=0.950
        for si in range(3):
            model.zero_grad()
            ls=[model(*get_batch())[1] for _ in range(25)]; torch.stack(ls).mean().backward()
            g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                         for p in model.parameters()]).detach(); model.zero_grad()
            g_proj=V.T@g; d_proj=g_proj/(T_evals+mu)
            g_res=g-V@(V.T@g); d=-(V@d_proj + g_res/mu)
            w0=model.flat_params(); v0=eval_val(model,n=8)
            model.set_flat(w0+d); v1=eval_val(model,n=8)
            if v1<v0:
                print(f"    Solve {si+1}: {v0:.4f}→{v1:.4f}")
            else:
                model.set_flat(w0); break
        v_final=eval_val(model)

# ── BASELINE COMPARISON ───────────────────────────────────────
print()
print("="*65)
print("BASELINE: GD-400 CONSTANT LR")
print("="*65)
torch.manual_seed(99)
gd=LM(); gd.te.weight.data.copy_(torch.tensor(E_init))
opt_gd=torch.optim.AdamW(gd.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

for gd_step in range(1,401):
    gd.train(); x,y=get_batch(); _,l=gd(x,y)
    opt_gd.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(gd.parameters(),1.0); opt_gd.step()

v_gd=eval_val(gd,n=20)
print(f"Final Compiler Loss: {v_final:.4f} nats")
print(f"Final GD-400 Loss:   {v_gd:.4f} nats")
print(f"Bigram Entropy Floor: {H_BIGRAM:.4f} nats")
print(f"Compiler Floor Gap:   {v_final - H_BIGRAM:+.4f} nats")
