"""HOW MUCH OF THE ZONE-II PATH IS ADAM'S NORMALISATION, AND HOW MUCH IS THE
GRADIENT IT IS APPLIED TO?

Adam's step is u = mhat / (sqrt(vhat)+eps). Two factors: a direction supplied
by the gradient history, and a per-coordinate rescaling supplied by the second
moment. The question is which one the zone-II chord is aligned with.

Decompose each step three ways and accumulate over 40->120:
    RAW      the plain gradient g            -- no Adam at all
    MOM      mhat                            -- momentum, no normalisation
    ADAM     mhat/(sqrt(vhat)+eps)           -- the realised step
and additionally the pure rescaling factor
    SCALE    1/(sqrt(vhat)+eps)              -- what the norm contributes

Then measure, against the chord CH = theta_120 - theta_40:
  cos(sum RAW, CH), cos(sum MOM, CH), cos(sum ADAM, CH)
which says how much of the destination each ingredient alone would have found.

And the counterfactual that actually answers "how much is the norm": rebuild
the chord with the normalisation FROZEN at its step-40 value (vhat fixed), and
with it REMOVED (vhat = 1). If the frozen-norm chord still points at the true
chord, the norm is a static rescaling; if only the live one does, the norm is
tracking something that changes.

NULL for every cosine: a random direction of matched norm, which on 1.18M
dimensions gives ~0.
"""
import io, contextlib, subprocess, sys, math, json
import numpy as np, torch, torch.nn.functional as F
subprocess.run([sys.executable,"build_corpus.py","--out","/tmp","--loops","300"],
               check=True,capture_output=True)
RAW_=open("compiler_geometri_patched_86.py").read()
SRC=RAW_[:RAW_.find("# \u2500\u2500 PHASE 3")]
for o,n in [("D=256; N_HEADS=4","D=128; N_HEADS=4"),
            ("for mf_r in range(1, 16):","for mf_r in range(1, 3):"),
            ("    if pc == N_STU-1:","    if False:"),
            ("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
             "    if False:")]:
    assert SRC.count(o)==1; SRC=SRC.replace(o,n,1)
EO="evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)"
EN=("_v0=np.random.RandomState(7).randn(L_sym.shape[0])\n"
    "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000,v0=_v0)\n"
    "evecs=evecs*np.sign(evecs[np.argmax(np.abs(evecs),axis=0),np.arange(evecs.shape[1])])")
assert SRC.count(EO)==1; SRC=SRC.replace(EO,EN,1)
B1,B2,EPS,WD,LRM=0.9,0.95,1e-8,0.1,5.0
torch.manual_seed(1234); np.random.seed(1234)
G={}; _b=io.StringIO()
with contextlib.redirect_stdout(_b): exec(SRC,G)
tm=G["model"]; gb=G["get_batch"]; lr0=G["LR"]
tn=[(n,p) for n,p in tm.named_parameters() if p.requires_grad]
EV=[gb() for _ in range(8)]
def _v():
    with torch.no_grad(): return sum(float(tm(x,y)[1]) for x,y in EV)/len(EV)
def flat(): return torch.cat([p.data.reshape(-1) for _,p in tn]).clone()
def setflat(v):
    with torch.no_grad():
        i=0
        for _,p in tn:
            q=p.numel(); p.data.copy_(v[i:i+q].view_as(p)); i+=q
BATCH=[gb() for _ in range(200)]
def q4(x):
    lv=torch.log(x.clamp_min(1e-20)); lo,hi=float(lv.min()),float(lv.max())
    if hi-lo<1e-12: return x
    s=(hi-lo)/15.0
    return torch.exp(torch.round((lv-lo)/s).clamp(0,15)*s+lo)
def walk(n,off,state=None,mode="adam",vfix=None,collect=False):
    lr=lr0*LRM
    if state is None:
        m={a:torch.zeros_like(p) for a,p in tn}; v={a:torch.zeros_like(p) for a,p in tn}
    else: m,v={k:t.clone() for k,t in state[0].items()},{k:t.clone() for k,t in state[1].items()}
    acc={k:[] for k in ("raw","mom","adam","scale")}
    for i,(x,y) in enumerate(BATCH[off:off+n],1):
        st=off+i
        _,l=tm(x,y); tm.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(tm.parameters(),1.0)
        b1,b2=1-B1**st,1-B2**st; w=lr*st/10 if st<=10 else lr
        pr={k:[] for k in acc}
        with torch.no_grad():
            for a,p in tn:
                g=p.grad if p.grad is not None else torch.zeros_like(p)
                m[a].mul_(B1).add_(g,alpha=1-B1); v[a].mul_(B2).addcmul_(g,g,value=1-B2)
                vh=(vfix[a] if vfix is not None else v[a]/b2)
                if mode=="novar": vh=torch.ones_like(vh)
                mh=m[a]/b1
                if p.dim()==2 and p.shape[0]>1:
                    vh=q4(vh); mh=torch.sign(mh)*q4(mh.abs())
                den=vh.sqrt()+EPS; u=mh/den
                if collect:
                    pr["raw"].append((-w*g).reshape(-1)); pr["mom"].append((-w*mh).reshape(-1))
                    pr["adam"].append((-w*u).reshape(-1)); pr["scale"].append((1.0/den).reshape(-1))
                p.data.add_(-w*(u+WD*p.data))
        if collect:
            for k in acc: acc[k].append(torch.cat(pr[k]))
    return (m,v),acc
st40,_=walk(40,0); TH40=flat(); V40=_v()
st40=({k:t.clone() for k,t in st40[0].items()},{k:t.clone() for k,t in st40[1].items()})
VFIX={a:(st40[1][a]/(1-B2**40)) for a,_ in tn}
_,acc=walk(80,40,state=st40,collect=True); TH120=flat()
CH=TH120-TH40; cn=float(CH.norm())
cos=lambda a,b: float((a@b)/(a.norm()*b.norm()+1e-30))
S={k:sum(acc[k]) for k in ("raw","mom","adam")}
print(f"  chord norm {cn:.2f}   zone II val {V40:.4f} -> {_v():.4f}\n")
print(f"  cos(accumulated ingredient, chord)")
for k in ("raw","mom","adam"):
    print(f"    {k:<6} {cos(S[k],CH):+.4f}")
g=torch.Generator().manual_seed(3)
r=torch.randn(cn and TH40.numel(),generator=g)
print(f"    {'random':<6} {cos(r,CH):+.4f}")
print(f"\n  norm contribution: |scale| stats over the zone")
sc=torch.stack([s for s in acc["scale"][::10]])
print(f"    mean 1/(sqrt(vhat)+eps) = {float(sc.mean()):.1f}   "
      f"sd {float(sc.std()):.1f}   max/min {float(sc.max())/max(float(sc.min()),1e-9):.0f}")
print(f"\n  counterfactual chords from theta_40")
for mode,vf,lab in (("adam",VFIX,"vhat FROZEN at t=40"),("novar",None,"vhat REMOVED (=1)")):
    setflat(TH40); _,_=walk(80,40,state=st40,mode=mode,vfix=vf)
    ch2=flat()-TH40
    print(f"    {lab:<22} val {_v():.4f}   cos to true chord {cos(ch2,CH):+.4f}"
          f"   |ch| {float(ch2.norm()):.2f}")
setflat(TH120)
