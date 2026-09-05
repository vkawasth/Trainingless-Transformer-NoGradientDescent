"""ROLE-DIFFERENTIATED vhat FREEZE: DOES THE SKELETON FINISH BEFORE ATTENTION?

The graph reading is that EMB/FF/LN form a skeleton perfected by ~k=40, after
which the W_Q/W_K/W_V/W_O paths are built on it. If true, the second moment for
the skeleton roles is learned earlier than for attention, and freezing them on
different schedules should beat a uniform freeze.

CAUTION ON THE PREMISE: the k90 measurement showed all seven roles reaching
their minimum at step 40 together (LN 2.7, FF 4.0, EMB 4.7, W_Q/W_K 4.3,
W_V/W_O 3.7), i.e. the contraction is global, not skeleton-first. So this
tests the hypothesis rather than assuming it.

Arms, all with m at 4 bits and v per-coordinate at 4 bits, run to 120:
    live          no freeze                              reference
    both40        every role frozen at 40                current shipped
    skel40        EMB/FF/LN at 40, attention live
    skel40_att80  EMB/FF/LN at 40, attention at 80       the graph reading
    att40_skel80  the REVERSE -- attention first         control for
                                                         "any split helps"
The reverse arm matters: if att40_skel80 does as well, the benefit is having
some roles live rather than the skeleton specifically.
"""
import io, contextlib, subprocess, sys, math, json
import numpy as np, torch, torch.nn.functional as F
subprocess.run([sys.executable,"build_corpus.py","--out","/tmp","--loops","300"],
               check=True,capture_output=True)
R=open("compiler_geometri_patched_86.py").read()
SRC=R[:R.find("# \u2500\u2500 PHASE 3")]
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
def setflat(x):
    with torch.no_grad():
        i=0
        for _,p in tn:
            q=p.numel(); p.data.copy_(x[i:i+q].view_as(p)); i+=q
TH0=flat(); BATCH=[gb() for _ in range(200)]
def role(a):
    if a.startswith(("te","pe")): return "EMB"
    if "ln" in a.lower(): return "LN"
    if ".ff." in a: return "FF"
    return "ATTN"
SKEL={"EMB","LN","FF"}
def q4(x):
    lv=torch.log(x.clamp_min(1e-20)); lo,hi=float(lv.min()),float(lv.max())
    if hi-lo<1e-12: return x
    s=(hi-lo)/15.0
    return torch.exp(torch.round((lv-lo)/s).clamp(0,15)*s+lo)
def run(sched,upto=120):
    setflat(TH0)
    m={a:torch.zeros_like(p) for a,p in tn}; v={a:torch.zeros_like(p) for a,p in tn}
    froz={}
    for i,(x,y) in enumerate(BATCH[:upto],1):
        _,l=tm(x,y); tm.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(tm.parameters(),1.0)
        b1,b2=1-B1**i,1-B2**i; w=lr0*LRM*(i/10 if i<=10 else 1)
        with torch.no_grad():
            for a,p in tn:
                g=p.grad if p.grad is not None else torch.zeros_like(p)
                m[a].mul_(B1).add_(g,alpha=1-B1); v[a].mul_(B2).addcmul_(g,g,value=1-B2)
                k=sched.get(role(a))
                if k is not None and i==k+1 and a not in froz: froz[a]=(v[a]/b2).clone()
                vh=froz.get(a, v[a]/b2); mh=m[a]/b1
                if p.dim()==2 and p.shape[0]>1:
                    vh=q4(vh); mh=torch.sign(mh)*q4(mh.abs())
                p.data.add_(-w*(mh/(vh.sqrt()+EPS)+WD*p.data))
    return _v(), flat()
ref,THr=run({})
print(f"  live (no freeze): val {ref:.4f}\n")
cos=lambda a,b: float((a@b)/(a.norm()*b.norm()+1e-30))
ARMS=[("both40",{"EMB":40,"LN":40,"FF":40,"ATTN":40}),
      ("skel40",{"EMB":40,"LN":40,"FF":40}),
      ("skel40_att80",{"EMB":40,"LN":40,"FF":40,"ATTN":80}),
      ("att40_skel80",{"ATTN":40,"EMB":80,"LN":80,"FF":80})]
print(f"  {'arm':<15}{'val@120':>10}{'vs live':>9}{'cos to live path':>19}")
for lab,sch in ARMS:
    v,TH=run(sch)
    print(f"  {lab:<15}{v:>10.4f}{v/ref:>9.3f}{cos(TH-TH0,THr-TH0):>19.4f}",flush=True)
