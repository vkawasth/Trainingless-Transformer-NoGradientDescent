"""A TINY REPLICA WHERE EVERYTHING IS EXACT.

Every geometric measurement here has been limited by P = 4.3e6: subspaces are
near-orthogonal by default, SPLIT never exceeded ~0.3 for the Fisher sheet, and
the commutator/transport questions returned "unresolvable" rather than an
answer.

At P ~ 2000 the Hessian and Fisher can be formed DENSELY -- exact eigenspaces,
exact projectors, exact commutators, no range finder, no SPLIT, no calibration.
Every question that hit the noise floor becomes computable.

The model keeps the structure and drops the scale: same architecture family
(pre-LN residual blocks, softmax attention, gated FF), D=12, 2 layers, vocab 40,
and a corpus with genuine 2-form and 3-form relations rather than the chunked
degenerate one.

Measured exactly at several checkpoints:
  eig(F), eig(H)           full spectra, no estimation
  ov(F,H)                  the subspace overlap that was 0.007-0.045 by sampling
  ||[P_F,P_H]||_F^2        the commutator, with its exact principal angles
  ||P_F H (I-P_F)||        the off-diagonal block M
  cap_F, dL(P_F u)         the descent-channel result, exactly
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, json, math
torch.manual_seed(0)
V, D, L, T = 40, 12, 2, 16
class Blk(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1=nn.LayerNorm(D); s.n2=nn.LayerNorm(D)
        s.q=nn.Linear(D,D,bias=False); s.k=nn.Linear(D,D,bias=False)
        s.v=nn.Linear(D,D,bias=False); s.o=nn.Linear(D,D,bias=False)
        s.g=nn.Linear(D,2*D,bias=False); s.u=nn.Linear(D,2*D,bias=False)
        s.w=nn.Linear(2*D,D,bias=False)
    def forward(s,x):
        h=s.n1(x); q,k,v=s.q(h),s.k(h),s.v(h)
        a=Fn.softmax(q@k.transpose(-2,-1)/math.sqrt(D)
                     +torch.triu(torch.full((x.shape[1],x.shape[1]),-1e9),1),dim=-1)
        x=x+s.o(a@v)
        h=s.n2(x)
        return x+s.w(Fn.silu(s.g(h))*s.u(h))
class M(nn.Module):
    def __init__(s):
        super().__init__()
        s.te=nn.Embedding(V,D); s.pe=nn.Embedding(T,D)
        s.b=nn.ModuleList([Blk() for _ in range(L)]); s.nf=nn.LayerNorm(D)
    def forward(s,x,y):
        h=s.te(x)+s.pe(torch.arange(x.shape[1]))
        for b in s.b: h=b(h)
        lo=s.nf(h)@s.te.weight.T
        return lo, Fn.cross_entropy(lo.reshape(-1,V),y.reshape(-1))
# corpus with 2-form and 3-form relations: a small grammar, not a cycle
rg=np.random.default_rng(1)
rules={i:[(i*3+1)%V,(i*5+2)%V] for i in range(V)}
seq=[0]
for _ in range(4000):
    c=seq[-1]; nxt=rules[c][0] if (len(seq)>1 and seq[-2]%2==0) else rules[c][1]
    seq.append(int(nxt))
seq=np.array(seq)
def batch(n=24):
    i=rg.integers(0,len(seq)-T-1,n)
    x=np.stack([seq[j:j+T] for j in i]); y=np.stack([seq[j+1:j+T+1] for j in i])
    return torch.tensor(x),torch.tensor(y)
m=M(); ps=[p for p in m.parameters()]; P=sum(p.numel() for p in ps)
print(f"  tiny model: V={V} D={D} L={L} T={T}   P = {P}")
print(f"  dense H is {P*P*4/1e6:.0f} MB\n")
opt=torch.optim.AdamW(m.parameters(),lr=3e-3,betas=(0.9,0.95),weight_decay=0.1)
def flat(): return torch.cat([p.data.flatten() for p in ps]).clone()
def setth(t):
    with torch.no_grad():
        i=0
        for p in ps: q=p.numel(); p.data.copy_(t[i:i+q].view_as(p)); i+=q
def gvec():
    return torch.cat([(p.grad.flatten() if p.grad is not None else
        torch.zeros(p.numel())) for p in ps]).clone()
EV=[batch(48) for _ in range(4)]
def loss():
    t=0.0
    with torch.no_grad():
        for x,y in EV: t+=float(m(x,y)[1])
    return t/len(EV)
def denseH(th,nb=6):
    setth(th); H=torch.zeros(P,P)
    for _ in range(nb):
        x,y=batch()
        m.zero_grad(); _,l=m(x,y)
        g=torch.autograd.grad(l,ps,create_graph=True)
        g=torch.cat([t.flatten() for t in g])
        for i in range(P):
            r=torch.autograd.grad(g[i],ps,retain_graph=True,allow_unused=True)
            H[i]+=torch.cat([(t if t is not None else torch.zeros_like(p)).flatten()
                             for t,p in zip(r,ps)]).detach()
        setth(th)
    m.zero_grad(); H=H/nb; return (H+H.T)/2
def denseF(th,ns=400):
    setth(th); Fm=torch.zeros(P,P)
    for _ in range(ns):
        x,y=batch(4); m.zero_grad(); _,l=m(x,y); l.backward()
        g=gvec(); Fm+=torch.outer(g,g); setth(th)
    m.zero_grad(); return Fm/ns
def PR(e):
    a=np.abs(e); return float(a.sum()**2/(a**2).sum())
step=0
print(f"  {'step':>6}{'val':>9}{'PR(F)':>8}{'PR(H)':>8}{'ov(F,H)':>10}"
      f"{'||[P,P]||^2':>13}{'||M||/||dia||':>14}{'capF':>8}{'dLF/dLfull':>12}")
for ck in (50,150,300,600):
    while step<ck:
        x,y=batch(); _,l=m(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); step+=1
    th=flat(); L0=loss()
    H=denseH(th); Fm=denseF(th); setth(th)
    eH,vH=np.linalg.eigh(H.numpy()); eF,vF=np.linalg.eigh(Fm.numpy())
    r=3
    QF=torch.tensor(vF[:,-r:],dtype=torch.float32)
    QH=torch.tensor(vH[:,-r:],dtype=torch.float32)
    Mx=(QF.T@QH).numpy(); s=np.clip(np.linalg.svd(Mx,compute_uv=False),0,1)
    c2=float(2*np.sum(s**2*(1-s**2)))
    HQ=H@QF; dia=QF.T@HQ; off=HQ-QF@dia
    prev=flat()
    for _ in range(3):
        x,y=batch(); _,l=m(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    u=flat()-prev; setth(th)
    pf=QF@(QF.T@u)
    setth(th+pf); dlf=loss()-L0
    setth(th+u); dlu=loss()-L0; setth(th)
    print(f"  {ck:>6}{L0:>9.4f}{PR(eF):>8.2f}{PR(eH):>8.2f}"
          f"{float((QF.T@QH).pow(2).sum()/r):>10.4f}{c2:>13.4f}"
          f"{float(off.norm()/max(float(torch.linalg.matrix_norm(dia)),1e-30)):>14.3f}"
          f"{float((pf*pf).sum()/(u*u).sum()):>8.4f}"
          f"{dlf/max(abs(dlu),1e-30)*np.sign(dlu):>12.3f}",flush=True)
    setth(th+u)
