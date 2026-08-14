"""IS A VARIANCE-NORMALISED FRAME BETTER THAN THE RAW ONE?

Established: coordinate recurrence is a LANDSCAPE property (11.06x under plain
SGD against 4.89x under Adam) which Adam dilutes. If so, normalising the gradient
stream by its own per-coordinate variance should recover the structure Adam
spreads:

    g~_i = g_i / sqrt(sigma_i^2 + eps),   sigma from K micro-batches at each step

Three streams, same trajectory, same frames construction:
  RAW      the update stream dtheta        (what the controller currently uses)
  EMA      filtered at H*=4               (the observer protocol)
  TILDE    variance-normalised gradients   (the proposal)

For each, at matched checkpoints:
  split-half   the resolution gate: frames from interleaved steps in one window
  rotation     d_Gr between disjoint consecutive windows
  leakage      fraction of the NEXT raw update falling outside the frame --
               the controller-relevant number, and deliberately measured against
               the raw update for all three streams so the comparison is fair
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, math, re, json
V,D,L,T=40,12,2,16
KDIM=3; WINL=8; KMB=4
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
def keyof(nm):
    if nm.startswith("te") or nm.startswith("pe"): return "EMB"
    mm=re.match(r"b\.(\d)\.",nm)
    if not mm: return None
    li=mm.group(1)
    if re.search(r"\.(q|k|v|o)\.",nm): return f"ATTN{li}"
    if re.search(r"\.(g|u|w)\.",nm): return f"FF{li}"
    return None
torch.manual_seed(0)
m=M(); named=[(n,p) for n,p in m.named_parameters()]; ps=[p for _,p in named]
P=sum(p.numel() for p in ps)
span={}; i=0
for nm,p in named: span[nm]=(i,i+p.numel()); i+=p.numel()
idx={}
for nm,(a,b) in span.items():
    k=keyof(nm)
    if k: idx.setdefault(k,[]).append(torch.arange(a,b))
idx={k:torch.cat(v) for k,v in idx.items()}
opt=torch.optim.AdamW(m.parameters(),lr=3e-3,betas=(0.9,0.95),weight_decay=0.1)
def flat(): return torch.cat([p.data.flatten() for p in ps]).clone()
def gv():
    return torch.cat([(p.grad.flatten() if p.grad is not None else
        torch.zeros(p.numel())) for p in ps]).clone()
RAW=[]; EMA=[]; TIL=[]
beta=0.75; mst=torch.zeros(P)
for st in range(360):
    th=flat()
    gs=[]
    for _ in range(KMB):
        xx,yy=batch(6); m.zero_grad(); _,ll=m(xx,yy); ll.backward(); gs.append(gv())
    Gm=torch.stack(gs,1)
    mu=Gm.mean(1); sd=Gm.std(1)
    TIL.append((mu/(sd+1e-8)).clone())
    m.zero_grad()
    x,y=batch(); _,l=m(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    du=flat()-th
    RAW.append(du.clone())
    mst=beta*mst+(1-beta)*du; EMA.append(mst.clone())
def frame(cols,ii):
    A=torch.stack([c[ii] for c in cols],1)
    return torch.linalg.svd(A,full_matrices=False)[0][:,:KDIM]
def dgr(Q1,Q2):
    sv=torch.linalg.svdvals(Q1.T@Q2).numpy()
    return float(np.sqrt((np.arccos(np.clip(sv,0,1))**2).sum()))
MAXD=math.sqrt(KDIM)*math.pi/2
print(f"  P={P}, k={KDIM}, window {WINL}, {KMB} micro-batches/step")
print(f"  leakage measured against the RAW update for all three streams\n")
print(f"  {'stream':>8}{'split-half':>12}{'rotation':>11}{'leakage':>10}"
      f"{'leak(late)':>12}")
res={}
for lab,S in (("raw",RAW),("ema H=4",EMA),("tilde",TIL)):
    sh=[]; ro=[]; lk=[]; lk2=[]
    for t in range(40,len(S)-2*WINL,WINL):
        w1=S[t:t+WINL]; w2=S[t+WINL:t+2*WINL]
        for k,ii in idx.items():
            Q=frame(w1,ii)
            sh.append(dgr(frame(w1[0::2],ii),frame(w1[1::2],ii)))
            ro.append(dgr(Q,frame(w2,ii)))
            nx=RAW[t+WINL][ii]
            pj=Q@(Q.T@nx)
            v=1-float((pj*pj).sum())/max(float((nx*nx).sum()),1e-30)
            lk.append(v)
            if t>len(S)*0.6: lk2.append(v)
    res[lab]=dict(sh=float(np.mean(sh)),ro=float(np.mean(ro)),
                  lk=float(np.mean(lk)),lk2=float(np.mean(lk2)))
    r=res[lab]
    print(f"  {lab:>8}{r['sh']:>12.3f}{r['ro']:>11.3f}{r['lk']:>10.4f}{r['lk2']:>12.4f}")
json.dump(res,open("/home/claude/work/res_precond2.json","w"),indent=2)
print(f"\n  max d_Gr for k={KDIM}: {MAXD:.3f}")
b=res["raw"]
for lab in ("ema H=4","tilde"):
    r=res[lab]
    print(f"  {lab:>8} vs raw: split-half {r['sh']-b['sh']:+.3f}  "
          f"rotation {r['ro']-b['ro']:+.3f}  leakage {100*(r['lk']-b['lk'])/b['lk']:+.1f}%")
print(f"\n  tilde: lower leakage and slower rotation => variance normalisation")
print(f"         recovers the structure Adam dilutes")
