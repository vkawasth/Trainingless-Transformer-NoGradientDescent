"""ARE THE PER-BLOCK MOTION SPACES MUTUALLY ORTHOGONAL?

Each role's update sequence lives in a 1-5 dimensional manifold inside its own
block (k90 <= 5, E(top3) 0.68-0.99). The proposed reconciliation with the global
residual: the blocks are individually simple but their motion spaces are
mutually UNALIGNED, so the sum fails to collapse into one global direction.

The blocks partition the coordinates, so their motion spaces are orthogonal BY
CONSTRUCTION in parameter space -- comparing them there is vacuous. The question
only has content in a space they share. Two such spaces:

  FUNCTION   J du_i, the change in logits each block's motion induces. All
             blocks map into the same logit space, so overlap here is meaningful.
  CURVATURE  H^{1/2} du_i, via the Hessian inner product du_i' H du_j, which is
             the I_H object measured earlier at the 4-role granularity.

Measured for the top-3 motion directions of each role, at three windows, with a
random-direction control of matched dimension in each space.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, math, re, json
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
m=M(); named=[(n,p) for n,p in m.named_parameters()]; ps=[p for _,p in named]
P=sum(p.numel() for p in ps)
def role(nm):
    if nm.startswith("te") or nm.startswith("pe"): return "EMB"
    if "n1" in nm or "n2" in nm or nm.startswith("nf"): return "LN"
    for pat,lab in ((r"\.q\.","W_Q"),(r"\.k\.","W_K"),(r"\.v\.","W_V"),(r"\.o\.","W_O")):
        if re.search(pat,nm): return lab
    if ".g." in nm or ".u." in nm or ".w." in nm: return "FF"
    return "other"
ROLES=["EMB","LN","W_Q","W_K","W_V","W_O","FF"]
span={}; i=0
for nm,p in named: span[nm]=(i,i+p.numel()); i+=p.numel()
idx={r:[] for r in ROLES}
for nm,(a,b) in span.items():
    if role(nm) in idx: idx[role(nm)].append(np.arange(a,b))
idx={r:np.concatenate(v) for r,v in idx.items() if v}
opt=torch.optim.AdamW(m.parameters(),lr=3e-3,betas=(0.9,0.95),weight_decay=0.1)
def flat(): return torch.cat([p.data.flatten() for p in ps]).clone()
def setth(t):
    with torch.no_grad():
        i=0
        for p in ps: q=p.numel(); p.data.copy_(t[i:i+q].view_as(p)); i+=q
EVX=[batch(32) for _ in range(3)]
def logits(th):
    setth(th); o=[]
    with torch.no_grad():
        for x,y in EVX: o.append(m(x,y)[0].flatten().clone())
    return torch.cat(o)
def hvp(th,v,nb=4):
    a=torch.zeros(P)
    for _ in range(nb):
        x,y=batch(); m.zero_grad(); _,l=m(x,y)
        g=torch.autograd.grad(l,ps,create_graph=True)
        g=torch.cat([t.flatten() for t in g])
        r=torch.autograd.grad((g*v).sum(),ps,allow_unused=True)
        a+=torch.cat([(t if t is not None else torch.zeros_like(p)).flatten()
                      for t,p in zip(r,ps)]).detach(); setth(th)
    m.zero_grad(); return a/nb
WIN=[(20,60,"early"),(100,160,"mid"),(240,300,"late")]
step=0; prev=flat(); U={r:[] for r in idx}
for ck in range(1,301):
    x,y=batch(); _,l=m(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); step+=1
    cur=flat(); du=(cur-prev); prev=cur
    for r,ii in idx.items(): U[r].append(du[ii].clone())
th=flat()
for lo,hi,lab in WIN:
    print(f"\n  === {lab} (steps {lo}-{hi}) ===")
    B={}
    for r,ii in idx.items():
        A=torch.stack(U[r][lo-1:hi],1)
        Ub=torch.linalg.svd(A,full_matrices=False)[0][:,:3]
        full=torch.zeros(P,3); full[ii]=Ub
        B[r]=full
    # function space
    JB={}
    e=1.0; L0=logits(th)
    for r,Bm in B.items():
        cols=[]
        for j in range(3):
            v=Bm[:,j]
            cols.append(((logits(th+e*v)-logits(th-e*v))/(2*e)))
        setth(th)
        Jm=torch.stack(cols,1)
        JB[r]=torch.linalg.qr(Jm)[0]
    HB={}
    for r,Bm in B.items():
        HB[r]=torch.linalg.qr(torch.stack([hvp(th,Bm[:,j]) for j in range(3)],1))[0]
    ovf=lambda A,Bx: float((A.T@Bx).pow(2).sum()/3)
    rs=list(B)
    print(f"  FUNCTION-space overlap of the top-3 motion directions")
    print(f"  {'':>6}"+"".join(f"{r:>8}" for r in rs))
    for a_ in rs:
        print(f"  {a_:>6}"+"".join(f"{ovf(JB[a_],JB[b_]):>8.3f}" for b_ in rs))
    gen=torch.Generator().manual_seed(4)
    R1=torch.linalg.qr(torch.randn(JB[rs[0]].shape[0],3,generator=gen))[0]
    R2=torch.linalg.qr(torch.randn(JB[rs[0]].shape[0],3,generator=gen))[0]
    off=[ovf(JB[a_],JB[b_]) for a_ in rs for b_ in rs if a_!=b_]
    print(f"  mean off-diagonal {np.mean(off):.3f}   random-plane null {ovf(R1,R2):.4f}")
    offh=[ovf(HB[a_],HB[b_]) for a_ in rs for b_ in rs if a_!=b_]
    print(f"  CURVATURE-space mean off-diagonal {np.mean(offh):.3f}")
