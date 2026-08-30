"""UNPLUMBING ADAM: WHICH ELEMENT BUILDS THE SUPPORT OBJECT, AND IN WHICH ZONE?

Adam -> remove v -> coarsen v -> remove m -> SGD, with the trajectory split at
the developmental boundaries rather than pooled:

    zone I   0-40    formation
    zone II  40-100  consolidation
    zone III 100-200 late

Arms (EMB always per-coordinate v where v is present):
    full     m + coordinate v          Adam
    row      m + row-shared v
    tensor   m + one scalar per matrix
    no_v     m only                    momentum, no normalisation
    no_m     coordinate v, g in place of m   RMSProp
    sgd      neither

CONFOUND THAT MUST BE CARRIED. J is displacement-sensitive: r50(J) = 16 under
isotropic motion while r50(O3) = 32. Arms without v take wildly different step
sizes, so J is reported against CUMULATIVE DISPLACEMENT as well as against
step, and no claim is made from J alone at unmatched cum.

Measured per zone boundary: J(t, entry), J(t, prev boundary), O1/O2/O3, loss,
cumulative ||dtheta||. Cross-arm J at each boundary asks whether the arms build
the same support object.

D=128 deliberately: the container has 3 GB and a single D=256 arm was
OOM-killed mid-run. The reduced-width caveat is accepted and stated.
"""
import io, contextlib, subprocess, sys, math, json, time, gc
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F

OUT="res_unplumb.json"; RES={}
def flush(): json.dump(RES,open(OUT,"w"),indent=1,default=float)
subprocess.run([sys.executable,"build_corpus.py","--out","/tmp","--loops","300"],
               check=True,capture_output=True)
RAW=open("compiler_geometri_patched_86.py").read()
SRC=RAW[:RAW.find("# \u2500\u2500 PHASE 3")]
for o,n in [("D=256; N_HEADS=4","D=128; N_HEADS=4"),
            ("for mf_r in range(1, 16):","for mf_r in range(1, 3):"),
            ("    if pc == N_STU-1:","    if False:"),
            ("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
             "    if False:")]:
    assert SRC.count(o)==1, f"anchor {o!r}"; SRC=SRC.replace(o,n,1)
EO="evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)"
EN=("_v0=np.random.RandomState(7).randn(L_sym.shape[0])\n"
    "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000,v0=_v0)\n"
    "evecs=evecs*np.sign(evecs[np.argmax(np.abs(evecs),axis=0),np.arange(evecs.shape[1])])")
assert SRC.count(EO)==1; SRC=SRC.replace(EO,EN,1)

ids=json.load(open("/tmp/train_ids.json")); BASE=ids[:1364]; W=8
nxt=defaultdict(set)
for od in range(1,6):
    for i in range(len(BASE)-od): nxt[(od,tuple(BASE[i:i+od]))].add(BASE[i+od])
def bk(i):
    for od in range(1,6):
        if i-od>=0 and len(nxt[(od,tuple(BASE[i-od:i]))])==1: return od
    return 6
POS=list(range(W,len(BASE)-1)); np.random.default_rng(3).shuffle(POS)
BKT=defaultdict(list)
for i in POS: BKT[bk(i)].append(i)
BU=[b for b in sorted(BKT) if b<=3 and len(BKT[b])>=20]
HELD={b:BKT[b][:200] for b in BU}
def tens(idx):
    return (torch.tensor([[BASE[i-W+j] for j in range(W)] for i in idx]),
            torch.tensor([BASE[i] for i in idx]))
NS,B1,B2,EPS,WD,LRM=200,0.9,0.95,1e-8,0.1,5.0
ZONE=[0,40,100,200]; NU=24; PMAG=0.5
isEMB=lambda n: n.startswith("te") or n.startswith("pe")
def red(v,mode):
    if mode=="coord":  return v
    if mode=="row":    return v.mean(dim=-1,keepdim=True).expand_as(v)
    if mode=="tensor": return v.mean().expand_as(v)
    raise ValueError(mode)
ARMS={"full":("coord",True,True),"row":("row",True,True),
      "tensor":("tensor",True,True),"no_v":(None,True,False),
      "no_m":("coord",False,True),"sgd":(None,False,False)}

def run(arm):
    vmode,use_m,use_v=ARMS[arm]
    torch.manual_seed(1234); np.random.seed(1234)
    G={}; b_=io.StringIO()
    with contextlib.redirect_stdout(b_): exec(SRC,G)
    model=G["model"]; gb=G["get_batch"]; lr=G["LR"]*LRM
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    EV=[gb() for _ in range(6)]
    ff=model.blocks[0].ff; Wg=ff.g.weight; bx,_=gb()
    @torch.no_grad()
    def resp(x):
        h=model.te(x)+model.pe(torch.arange(x.shape[1]))
        h=model.blocks[0].attn(h)
        return ff.n(h+ff.o(F.silu(ff.g(h))*ff.v(h)))
    @torch.no_grad()
    def A():
        base=resp(bx).abs().mean(dim=(0,1))
        M=np.zeros((NU,base.shape[0])); gg=torch.Generator().manual_seed(5150)
        for i in range(NU):
            row=Wg[i].clone()
            d=torch.randn(row.shape,generator=gg); d=d/d.norm()*PMAG*row.norm()
            Wg[i].add_(d); M[i]=(resp(bx).abs().mean(dim=(0,1))-base).numpy()
            Wg[i].copy_(row)
        return M
    @torch.no_grad()
    def caps():
        model.eval(); a={}
        for b in BU:
            X,Y=tens(HELD[b]); a[b]=float((model(X)[0][:,-1,:].argmax(-1)==Y).float().mean())
        L=sum(float(model(x_,y_)[1]) for x_,y_ in EV)/len(EV); model.train(); return a,L
    torch.manual_seed(17)
    m={n:torch.zeros_like(p) for n,p in named} if use_m else None
    v={n:torch.zeros_like(p) for n,p in named} if use_v else None
    cum=0.0; out={}
    a0,L0=caps(); out[0]=dict(A=A().tolist(),acc=a0,L=L0,cum=0.0)
    for st in range(1,NS+1):
        x,y=gb(); _,l=model(x,y); model.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        b1,b2=1-B1**st,1-B2**st; tot=0.0
        with torch.no_grad():
            for n,p in named:
                g=p.grad if p.grad is not None else torch.zeros_like(p)
                num=g
                if use_m:
                    m[n].mul_(B1).add_(g,alpha=1-B1); num=m[n]/b1
                den=1.0
                if use_v:
                    v[n].mul_(B2).addcmul_(g,g,value=1-B2)
                    mo="coord" if (isEMB(n) or p.dim()!=2) else vmode
                    den=red(v[n]/b2,mo).sqrt()+EPS
                u=-lr*(num/den+WD*p.data)
                tot+=float((u*u).sum()); p.data.add_(u)
        if not math.isfinite(tot): break
        cum+=math.sqrt(tot)
        if st in ZONE:
            a,L=caps(); out[st]=dict(A=A().tolist(),acc=a,L=L,cum=cum)
            print(f"    {arm:>7} t={st:>3} L {L:>8.4f}  O "
                  f"{'/'.join(f'{a[b]:.3f}' for b in BU)}  cum {cum:>7.2f}",flush=True)
    RES[arm]={str(k):val for k,val in out.items()}; flush()
    del G,model,m,v; gc.collect()

for arm in ARMS: run(arm)
NT=np.array(RES["full"]["0"]["A"]).shape[1]
def sp(M,q=0.25):
    X=np.abs(np.array(M)); kk=max(1,int(NT*q))
    return [set(np.argsort(-X[i])[:kk]) for i in range(NU)]
JJ=lambda M,N: float(np.mean([len(a&b)/max(len(a|b),1) for a,b in zip(sp(M),sp(N))]))
print(f"\n  J against ENTRY (t=0), by zone.  row-permuted null ~0.18\n")
print(f"  {'arm':>8}"+"".join(f"{'t'+str(t):>9}" for t in ZONE[1:])+f"{'cum200':>9}")
for arm in ARMS:
    r=RES[arm]
    if "200" not in r: print(f"  {arm:>8}  diverged"); continue
    print(f"  {arm:>8}"+"".join(f"{JJ(r[str(t)]['A'],r['0']['A']):>9.3f}" for t in ZONE[1:])
          +f"{r['200']['cum']:>9.2f}")
print(f"\n  cross-arm J at t=200 (same init, indices correspond)\n")
ok=[a for a in ARMS if "200" in RES[a]]
print(f"  {'':>8}"+"".join(f"{a:>9}" for a in ok))
for a in ok:
    print(f"  {a:>8}"+"".join(f"{JJ(RES[a]['200']['A'],RES[b]['200']['A']):>9.3f}" for b in ok))
flush()
