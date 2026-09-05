"""WHEN IS vhat FINISHED BEING LEARNED?

vhat frozen at step 40 and held through zone II gives cos 0.794 to the true
chord and val 0.4188 against 0.4118 -- but with chord norm 36.30 against 23.48,
a 55% overshoot. Frozen from step 1 it destroys the run. So the second moment
is learned early and held; the question is when "early" ends.

Sweep the freeze point k in {5,10,20,30,40,60,80}: run to k with live vhat,
freeze it, run to 120, and report

    val at 120        does the run still work
    cos to true chord is the direction right
    |chord|           is the SCALE right, which the k=40 result got wrong

against the live-vhat reference (val 0.4118, |chord| 23.48).

If val and cos plateau at some k while |chord| keeps changing, direction and
scale are learned on different schedules -- which is the distinction the k=40
result raised and could not settle.
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
def q4(x):
    lv=torch.log(x.clamp_min(1e-20)); lo,hi=float(lv.min()),float(lv.max())
    if hi-lo<1e-12: return x
    s=(hi-lo)/15.0
    return torch.exp(torch.round((lv-lo)/s).clamp(0,15)*s+lo)
def run(freeze_at=None,upto=120,start=None,state=None):
    if start is not None: setflat(start)
    if state is None:
        m={a:torch.zeros_like(p) for a,p in tn}; v={a:torch.zeros_like(p) for a,p in tn}
    else: m,v={k:t.clone() for k,t in state[0].items()},{k:t.clone() for k,t in state[1].items()}
    frozen=None
    for i,(x,y) in enumerate(BATCH[:upto],1):
        _,l=tm(x,y); tm.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(tm.parameters(),1.0)
        b1,b2=1-B1**i,1-B2**i; w=lr0*LRM*(i/10 if i<=10 else 1)
        if freeze_at is not None and i==freeze_at+1 and frozen is None:
            frozen={a:(v[a]/(1-B2**freeze_at)) for a,_ in tn}
        with torch.no_grad():
            for a,p in tn:
                g=p.grad if p.grad is not None else torch.zeros_like(p)
                m[a].mul_(B1).add_(g,alpha=1-B1); v[a].mul_(B2).addcmul_(g,g,value=1-B2)
                vh=frozen[a] if frozen is not None else v[a]/b2
                mh=m[a]/b1
                if p.dim()==2 and p.shape[0]>1:
                    vh=q4(vh); mh=torch.sign(mh)*q4(mh.abs())
                p.data.add_(-w*(mh/(vh.sqrt()+EPS)+WD*p.data))
    return (m,v)
setflat(TH0); run(None,40); TH40=flat()
setflat(TH0); run(None,120); TH120=flat()
CH=TH120-TH40; cn=float(CH.norm()); vref=_v()
cos=lambda a,b: float((a@b)/(a.norm()*b.norm()+1e-30))
print(f"  live-vhat reference: val {vref:.4f}  |chord| {cn:.2f}\n")
print(f"  {'freeze k':>9}{'val@120':>10}{'cos->chord':>13}{'|chord|':>10}{'ratio':>8}")
for k in (5,10,20,30,40,60,80):
    setflat(TH0); run(k,120)
    ch=flat()-TH40
    print(f"  {k:>9}{_v():>10.4f}{cos(ch,CH):>+13.4f}{float(ch.norm()):>10.2f}"
          f"{float(ch.norm())/cn:>8.2f}",flush=True)
