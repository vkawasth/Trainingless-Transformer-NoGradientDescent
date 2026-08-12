"""SCALING LADDER: IS THE F/H DISJOINTNESS DIMENSIONAL OR PROXIMITY TO MINIMUM?

Exact at P=3672: ov(F,H) rises 0.256 -> 0.858. At P=4.3e6 by sampling: 0.007-0.045.
Two explanations, and the tiny run cannot separate them:

  DIMENSIONAL   subspaces are near-orthogonal by default at large P, and the
                range finder cannot resolve the true alignment
                -> ov falls with P AT MATCHED LOSS
  PROXIMITY     F -> H is a theorem near a well-specified minimum; the tiny
                model reaches val 0.036 while the large one sits at 0.06-0.09 on
                a far harder task
                -> ov tracks the LOSS and not P

Matched loss is therefore mandatory; matched step would confound them, which is
the confound that overturned four earlier results in this programme.

Three sizes, dense F and H where feasible, range finder at all three so the
dense-vs-sampled gap isolates where estimation starts biting.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, math, json
def make(V,D,L,T):
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
    return M()
rg=np.random.default_rng(1)
def corpus(V):
    rules={i:[(i*3+1)%V,(i*5+2)%V] for i in range(V)}
    s=[0]
    for _ in range(6000):
        c=s[-1]; s.append(int(rules[c][0] if (len(s)>1 and s[-2]%2==0) else rules[c][1]))
    return np.array(s)
TARGET=0.30
print(f"  matched LOSS: each size trained until val <= {TARGET}\n")
print(f"  {'P':>8}{'V':>5}{'D':>4}{'L':>3}{'step':>6}{'val':>8}"
      f"{'ov dense':>10}{'ov rangef':>11}{'PR(F)':>8}{'PR(H)':>8}")
out=[]
for V,D,L,T in ((40,12,2,16),(60,20,2,16),(80,32,3,16)):
    torch.manual_seed(0)
    m=make(V,D,L,T); ps=list(m.parameters()); P=sum(p.numel() for p in ps)
    seq=corpus(V)
    def batch(n=24):
        i=rg.integers(0,len(seq)-T-1,n)
        return (torch.tensor(np.stack([seq[j:j+T] for j in i])),
                torch.tensor(np.stack([seq[j+1:j+T+1] for j in i])))
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3,betas=(0.9,0.95),weight_decay=0.1)
    EV=[batch(48) for _ in range(4)]
    def loss():
        t=0.0
        with torch.no_grad():
            for x,y in EV: t+=float(m(x,y)[1])
        return t/len(EV)
    def flat(): return torch.cat([p.data.flatten() for p in ps]).clone()
    def setth(t):
        with torch.no_grad():
            i=0
            for p in ps: q=p.numel(); p.data.copy_(t[i:i+q].view_as(p)); i+=q
    def gvec():
        return torch.cat([(p.grad.flatten() if p.grad is not None else
            torch.zeros(p.numel())) for p in ps]).clone()
    step=0; v=9e9
    while step<3000 and v>TARGET:
        for _ in range(25):
            x,y=batch(); _,l=m(x,y); opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); step+=1
        v=loss()
    th=flat()
    def hvp(vv,nb=4):
        a=torch.zeros(P)
        for _ in range(nb):
            x,y=batch(); m.zero_grad(); _,l=m(x,y)
            g=torch.autograd.grad(l,ps,create_graph=True)
            g=torch.cat([t.flatten() for t in g])
            r=torch.autograd.grad((g*vv).sum(),ps,allow_unused=True)
            a+=torch.cat([(t if t is not None else torch.zeros_like(p)).flatten()
                          for t,p in zip(r,ps)]).detach(); setth(th)
        m.zero_grad(); return a/nb
    def fvp(vv,ns=120):
        a=torch.zeros(P)
        for _ in range(ns):
            x,y=batch(4); m.zero_grad(); _,l=m(x,y); l.backward()
            g=gvec(); a+=g*float((g*vv).sum()); setth(th)
        m.zero_grad(); return a/ns
    r=3
    # range finder
    gen=torch.Generator().manual_seed(7)
    QFr=torch.linalg.qr(torch.stack([fvp(torch.randn(P,generator=gen)) for _ in range(r)],1))[0]
    QHr=torch.linalg.qr(torch.stack([hvp(torch.randn(P,generator=gen)) for _ in range(r)],1))[0]
    ovr=float((QFr.T@QHr).pow(2).sum()/r)
    ovd=float('nan'); prF=float('nan'); prH=float('nan')
    if P<=9000:
        H=torch.zeros(P,P)
        for _ in range(4):
            x,y=batch(); m.zero_grad(); _,l=m(x,y)
            g=torch.autograd.grad(l,ps,create_graph=True)
            g=torch.cat([t.flatten() for t in g])
            for i in range(P):
                rr=torch.autograd.grad(g[i],ps,retain_graph=True,allow_unused=True)
                H[i]+=torch.cat([(t if t is not None else torch.zeros_like(p)).flatten()
                                 for t,p in zip(rr,ps)]).detach()
            setth(th)
        H=(H/4); H=(H+H.T)/2
        Fm=torch.zeros(P,P)
        for _ in range(300):
            x,y=batch(4); m.zero_grad(); _,l=m(x,y); l.backward()
            g=gvec(); Fm+=torch.outer(g,g); setth(th)
        Fm/=300
        eH,vH=np.linalg.eigh(H.numpy()); eF,vF=np.linalg.eigh(Fm.numpy())
        QF=torch.tensor(vF[:,-r:],dtype=torch.float32)
        QH=torch.tensor(vH[:,-r:],dtype=torch.float32)
        ovd=float((QF.T@QH).pow(2).sum()/r)
        pr=lambda e:(lambda a: float(a.sum()**2/(a**2).sum()))(np.abs(e))
        prF,prH=pr(eF),pr(eH)
    print(f"  {P:>8}{V:>5}{D:>4}{L:>3}{step:>6}{v:>8.4f}"
          f"{ovd:>10.4f}{ovr:>11.4f}{prF:>8.2f}{prH:>8.2f}",flush=True)
    out.append(dict(P=P,val=v,ovd=ovd,ovr=ovr,prF=prF,prH=prH))
    del m,ps
json.dump(out,open("/home/claude/work/res_ladder.json","w"),indent=2)
print(f"\n  ov falling with P at matched loss => dimensional")
print(f"  ov flat with P                     => proximity to minimum, not dimension")
print(f"  dense >> rangefinder               => estimation floor located")
