#!/usr/bin/env python3
"""
MF10 Structure Analysis
========================
After 10 MF iterations, the model achieves val=0.0155 (vs teacher 0.247).
What has changed geometrically?

MEASUREMENTS (all pre-CE, no gradient descent after MF):
  1. Fisher alignment: do attention heads align with Fisher top eigenvectors?
  2. Hessian condition number: is the landscape better conditioned after MF?
  3. Im(z_l) sign pattern: which sheet of the étale cover is MF10 on?
  4. W_K-embedding alignment: has MF resolved the W_K misalignment?
  5. Valley comparison: does MF10 start in a qualitatively different basin?

KEY QUESTION:
  Is the oscillatory MF iteration equivalent to natural gradient descent
  in the joint (E, W_K) space?
  
  If yes: Fisher alignment should INCREASE with MF iterations.
  If no: Fisher alignment stays constant; MF explores orthogonal directions.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
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
val_t=torch.tensor(val_ids,dtype=torch.long)

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

class Attn(nn.Module):
    def __init__(self,d,nh):
        super().__init__()
        self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
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
    def get_flat_params(self): return torch.cat([p.data.flatten() for p in self.parameters()])
    def set_flat_params(self,flat):
        idx=0
        for p in self.parameters(): n=p.numel(); p.data.copy_(flat[idx:idx+n].reshape(p.shape)); idx+=n
    def flat_grad(self): return torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()) for p in self.parameters()])

def eval_val(m,n=30):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

print("Training teacher...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,301):
    lr_now=LR*min(step,100)/100 if step<=100 else LR*0.5*(1+math.cos(math.pi*(step-100)/200))
    for pg in opt.param_groups: pg['lr']=lr_now
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
    if step%100==0:
        teacher.eval()
        with torch.no_grad(): vl=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}")
teacher.eval(); val_teacher=eval_val(teacher)
print(f"  Teacher val={val_teacher:.4f}\n")

def build_student():
    torch.manual_seed(99); stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight); stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            for s,d in [(teacher.blocks[L_ATT].attn.WK,stu.blocks[l].attn.WK),
                        (teacher.blocks[L_ATT].attn.WQ,stu.blocks[l].attn.WQ),
                        (teacher.blocks[L_ATT].attn.WV,stu.blocks[l].attn.WV),
                        (teacher.blocks[L_ATT].attn.op,stu.blocks[l].attn.op),
                        (teacher.blocks[L_ATT].ff.g,stu.blocks[l].ff.g),
                        (teacher.blocks[L_ATT].ff.v,stu.blocks[l].ff.v),
                        (teacher.blocks[L_ATT].ff.o,stu.blocks[l].ff.o)]:
                d.weight.copy_(s.weight)
    return stu

def hv_product(model,v,n=15):
    params=list(model.parameters()); model.zero_grad()
    loss=sum(model(*get_batch())[1] for _ in range(n))/n
    grads=torch.autograd.grad(loss,params,create_graph=True)
    gv=(torch.cat([g.flatten() for g in grads])*v.detach()).sum()
    hv=torch.cat([h.flatten() for h in torch.autograd.grad(gv,params,retain_graph=False)]).detach()
    model.zero_grad(); return hv

print("Computing v_neg...")
stu_ref=build_student(); n_p=sum(p.numel() for p in stu_ref.parameters())
v=torch.randn(n_p); v=v/v.norm()
for _ in range(15): Hv=hv_product(stu_ref,v,15); neg=-Hv; v=neg/max(float(neg.norm()),1e-10)
v_neg=v.clone(); print("v_neg ready.\n")

def apply_mf(stu, n_iter, mf_lr=0.01, n_corpus=200):
    for it in range(n_iter):
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.requires_grad_(False)
            stu.blocks[l].attn.WQ.weight.requires_grad_(False)
        eg=torch.zeros(VOCAB,D); ef=torch.zeros(VOCAB,D)
        torch.manual_seed(it*1000)
        for i in range(n_corpus):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            if stu.te.weight.grad is not None:
                g=stu.te.weight.grad.detach(); eg+=g; ef+=g**2
        eg/=n_corpus; ef/=n_corpus
        with torch.no_grad(): stu.te.weight.add_(-mf_lr*eg/(ef+1e-4))
        for l in range(N_STU):
            stu.blocks[l].attn.WK.weight.requires_grad_(True)
            stu.blocks[l].attn.WQ.weight.requires_grad_(True)
        stu.te.weight.requires_grad_(False)
        wg=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        wf=torch.zeros_like(stu.blocks[0].attn.WK.weight)
        torch.manual_seed(it*1000+500)
        for i in range(n_corpus):
            ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
            x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
            stu.zero_grad(); _,loss=stu(x,y); loss.backward()
            g=torch.zeros_like(stu.blocks[0].attn.WK.weight)
            for l in range(N_STU):
                if stu.blocks[l].attn.WK.weight.grad is not None:
                    g+=stu.blocks[l].attn.WK.weight.grad/N_STU
            wg+=g; wf+=g**2
        wg/=n_corpus; wf/=n_corpus
        with torch.no_grad():
            for l in range(N_STU):
                stu.blocks[l].attn.WK.weight.add_(-mf_lr*wg/(wf+1e-4))
                stu.blocks[l].attn.WQ.weight.add_(-mf_lr*wg.T/(wf.T+1e-4))
        stu.te.weight.requires_grad_(True)

def analyze_model(stu, label, n_fisher=50):
    """Complete geometric analysis of model state."""
    print(f"\n  === {label} ===")
    v0=eval_val(stu,n=20); print(f"  val: {v0:.4f}")

    # 1. Fisher top eigenvector and alignment with gradient
    print(f"  Computing Fisher spectrum ({n_fisher} seqs)...")
    grads=[]
    torch.manual_seed(42)
    for i in range(n_fisher):
        ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
        x=train_t[ix:ix+SEQ].unsqueeze(0); y=train_t[ix+1:ix+SEQ+1].unsqueeze(0)
        stu.zero_grad(); _,loss=stu(x,y); loss.backward()
        grads.append(stu.flat_grad().detach().clone())
    G=torch.stack(grads)  # (n, params)

    # Fisher top eigenvector via power iteration
    vf=torch.randn(G.shape[1]); vf=vf/vf.norm()
    for _ in range(10):
        Gv=G@vf; Fv=(G.T@Gv)/n_fisher
        lam_f=float(Fv.norm()); vf=Fv/max(lam_f,1e-10)
    print(f"  Fisher lambda_1: {lam_f:.4f}")

    # Gradient alignment with Fisher v1
    mean_g=G.mean(0)
    align_fisher=float((mean_g*vf).sum())/(float(mean_g.norm())*float(vf.norm())+1e-10)
    print(f"  ||mean_grad||: {float(mean_g.norm()):.6f}")
    print(f"  Gradient alignment with Fisher v1: {align_fisher:.4f}")

    # 2. W_K-embedding alignment
    E=stu.te.weight.data.numpy()
    token_freq=torch.zeros(VOCAB)
    for t in train_ids: token_freq[t]+=1
    token_prob=(token_freq/token_freq.sum()).numpy()
    sqrt_p=np.sqrt(token_prob)[:,None]
    E_weighted=sqrt_p*E
    _,sv_e,Vt_e=np.linalg.svd(E_weighted,full_matrices=False)
    WK=stu.blocks[0].attn.WK.weight.data.numpy()
    alignments=[]
    for k in range(5):
        vk=Vt_e[k]; WKvk=WK@vk
        a=float(np.dot(WKvk,vk))/(np.linalg.norm(WKvk)*np.linalg.norm(vk)+1e-10)
        alignments.append(a)
    mean_align=np.mean(np.abs(alignments))
    print(f"  W_K-emb alignment (top-5 dirs): {mean_align:.4f}  "
          f"[baseline was 0.0456]")
    print(f"  Individual: {[f'{a:.3f}' for a in alignments]}")

    # 3. Min Hessian eigenvalue (saddle check)
    vhess=torch.randn(G.shape[1]); vhess=vhess/vhess.norm()
    for _ in range(8):
        Hv=hv_product(stu,vhess,10); neg=-Hv
        lam_min=-float(neg.norm()); vhess=neg/max(-lam_min,1e-10)
    print(f"  Hessian lambda_min: {lam_min:.4f}  "
          f"({'saddle' if lam_min<0 else 'convex'})")

    # 4. Embedding movement from init
    E_init=teacher.te.weight.data.numpy()
    emb_move=np.linalg.norm(E-E_init)
    print(f"  ||E - E_init||: {emb_move:.4f}  [baseline valley2 was 7.99]")

    # 5. WK movement from init
    WK_init=teacher.blocks[L_ATT].attn.WK.weight.data.numpy()
    wk_move=np.linalg.norm(WK-WK_init)
    print(f"  ||W_K - W_K_init||: {wk_move:.4f}")

    return {
        'val':v0, 'fisher_lam1':lam_f, 'fisher_align':align_fisher,
        'wk_emb_align':mean_align, 'hess_lam_min':lam_min,
        'emb_move':emb_move, 'wk_move':wk_move
    }

# Build models at different MF stages
print("="*65)
print("BUILDING MODELS AT DIFFERENT MF STAGES")
print("="*65)

# Baseline (no MF)
stu0=build_student()
w0=stu0.get_flat_params()
stu0.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
r0=analyze_model(stu0,"Baseline (saddle exit only)")

# MF3
stu3=build_student()
w0=stu3.get_flat_params()
stu3.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
apply_mf(stu3,n_iter=3,mf_lr=0.01)
r3=analyze_model(stu3,"MF3 (3 oscillatory iterations)")

# MF10
stu10=build_student()
w0=stu10.get_flat_params()
stu10.set_flat_params(w0+ALPHA_STAR*(v_neg/v_neg.norm()))
apply_mf(stu10,n_iter=10,mf_lr=0.01)
r10=analyze_model(stu10,"MF10 (10 oscillatory iterations)")

print(f"""
{'='*65}
  MF GEOMETRIC PROGRESSION
{'='*65}

  {'Metric':<30} {'Baseline':>10} {'MF3':>10} {'MF10':>10}
  {'-'*62}
  {'val':<30} {r0['val']:>10.4f} {r3['val']:>10.4f} {r10['val']:>10.4f}
  {'Fisher lambda_1':<30} {r0['fisher_lam1']:>10.4f} {r3['fisher_lam1']:>10.4f} {r10['fisher_lam1']:>10.4f}
  {'Grad-Fisher alignment':<30} {r0['fisher_align']:>10.4f} {r3['fisher_align']:>10.4f} {r10['fisher_align']:>10.4f}
  {'W_K-emb alignment':<30} {r0['wk_emb_align']:>10.4f} {r3['wk_emb_align']:>10.4f} {r10['wk_emb_align']:>10.4f}
  {'Hessian lambda_min':<30} {r0['hess_lam_min']:>10.4f} {r3['hess_lam_min']:>10.4f} {r10['hess_lam_min']:>10.4f}
  {'||E - E_init||':<30} {r0['emb_move']:>10.4f} {r3['emb_move']:>10.4f} {r10['emb_move']:>10.4f}
  {'||W_K - W_K_init||':<30} {r0['wk_move']:>10.4f} {r3['wk_move']:>10.4f} {r10['wk_move']:>10.4f}

  INTERPRETATION:
  IF Fisher lambda_1 INCREASES with MF: 
    MF is implementing natural gradient — aligning with Fisher directions
  IF Fisher lambda_1 DECREASES:
    MF is moving to flatter regions — reducing curvature ill-conditioning
    
  IF W_K-emb alignment INCREASES:
    MF resolves the W_K-embedding misalignment (the original saddle cause)
  IF unchanged:
    MF works through a different mechanism
    
  IF Hessian lambda_min INCREASES (less negative):
    MF is driving the system toward convexity
    The remaining CE steps are in a better-conditioned landscape
    
  IF ||E-E_init|| INCREASES monotonically:
    MF is continuously moving embeddings toward valley 2
    More iterations = closer to valley 2 floor
""")
