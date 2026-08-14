"""IS THE LOW-RANK FISHER A REAL CHANNEL, OR THE MEAN GRADIENT?

F = E[g g'] = F_C + mu mu'  with mu = E[g]. The rank-one term mu mu' is the
mean descent direction, and it can single-handedly drive PR(F) down. So
PR(F) ~ 2.5-5.7 admits two readings:

  A  the information geometry is genuinely low dimensional
  B  PR is low because mu mu' overwhelms the fluctuation covariance

Decided exactly on the P=3672 replica, where F and F_C are both dense:
  PR(F) vs PR(F_C)      A -> F_C stays low;  B -> F_C expands toward PR(H)
  ov(F_C,H) vs ov(F,H)  A -> converges near the minimum;  B -> collapses
  ||mu mu'||/||F||      how much of F the mean actually is

SAMPLE CEILING: F built from n gradients has rank <= n, so PR cannot exceed n.
Using 600 samples so a value of 17-30 is not capped by the estimator.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, math, json
torch.manual_seed(0)
V,D,L,T=40,12,2,16
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
        x=x+s.o(a@v); h=s.n2(x)
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
        return lo,Fn.cross_entropy(lo.reshape(-1,V),y.reshape(-1))
rg=np.random.default_rng(1)
rules={i:[(i*3+1)%V,(i*5+2)%V] for i in range(V)}
s_=[0]
for _ in range(6000):
    c=s_[-1]; s_.append(int(rules[c][0] if (len(s_)>1 and s_[-2]%2==0) else rules[c][1]))
seq=np.array(s_)
def batch(n=24):
    i=rg.integers(0,len(seq)-T-1,n)
    return (torch.tensor(np.stack([seq[j:j+T] for j in i])),
            torch.tensor(np.stack([seq[j+1:j+T+1] for j in i])))
m=M(); ps=list(m.parameters()); P=sum(p.numel() for p in ps)
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
NS=600
def FandFc(th):
    setth(th); Gs=[]
    for _ in range(NS):
        x,y=batch(4); m.zero_grad(); _,l=m(x,y); l.backward()
        Gs.append(gvec()); setth(th)
    m.zero_grad(); Gm=torch.stack(Gs,1)          # P x NS
    mu=Gm.mean(1,keepdim=True)
    F=(Gm@Gm.T)/NS
    Gc=Gm-mu
    Fc=(Gc@Gc.T)/NS
    return F,Fc,mu.squeeze(1)
def denseH(th,nb=4):
    setth(th); H=torch.zeros(P,P)
    for _ in range(nb):
        x,y=batch(); m.zero_grad(); _,l=m(x,y)
        g=torch.autograd.grad(l,ps,create_graph=True)
        g=torch.cat([t.flatten() for t in g])
        for i in range(P):
            r=torch.autograd.grad(g[i],ps,retain_graph=True,allow_unused=True)
            H[i]+=torch.cat([(t if t is not None else torch.zeros_like(p)).flatten()
                             for t,p in zip(r,ps)]).detach()
        setth(th)
    m.zero_grad(); H=H/nb; return (H+H.T)/2
def PR(e):
    a=np.abs(e); return float(a.sum()**2/(a**2).sum())
print(f"  P={P}, {NS} gradient samples (PR ceiling {NS})\n")
print(f"  {'step':>6}{'val':>9}{'PR(F)':>8}{'PR(Fc)':>9}{'PR(H)':>8}"
      f"{'ov(F,H)':>10}{'ov(Fc,H)':>11}{'|mumu|/|F|':>12}")
step=0
for ck in (50,150,300):
    while step<ck:
        x,y=batch(); _,l=m(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); step+=1
    th=flat(); v=loss()
    F,Fc,mu=FandFc(th); H=denseH(th); setth(th)
    eF=np.linalg.eigvalsh(F.numpy()); eC=np.linalg.eigvalsh(Fc.numpy())
    eH,vH=np.linalg.eigh(H.numpy())
    _,vF=np.linalg.eigh(F.numpy()); _,vC=np.linalg.eigh(Fc.numpy())
    r=3
    QF=torch.tensor(vF[:,-r:],dtype=torch.float32)
    QC=torch.tensor(vC[:,-r:],dtype=torch.float32)
    QH=torch.tensor(vH[:,-r:],dtype=torch.float32)
    mm=float((mu@mu)/max(float(torch.linalg.matrix_norm(F)),1e-30))
    print(f"  {ck:>6}{v:>9.4f}{PR(eF):>8.2f}{PR(eC):>9.2f}{PR(eH):>8.2f}"
          f"{float((QF.T@QH).pow(2).sum()/r):>10.4f}"
          f"{float((QC.T@QH).pow(2).sum()/r):>11.4f}{mm:>12.4f}",flush=True)
    setth(th)
print(f"\n  PR(Fc) stays low + ov(Fc,H) high => a real information channel")
print(f"  PR(Fc) expands + ov(Fc,H) collapses => the 3-plane was the mean drift")
