"""IS THE NEXT INTERVAL'S ORIENTATION PREDICTABLE FROM THE PREVIOUS ONE'S?

Precondition for any generator theta_100 -> R_120 that replaces Adam steps.
If S(100->120) is uncorrelated with S(40->100), an extrapolating extractor of
the form "continue the relational direction" cannot exist, and no training is
needed to find that out.

Computed on teacher snapshots only, layer 3:
    S(a,b) = sgn(W_b - W_a)
    agree(S(40,100), S(100,120))     can the past interval predict the next
    agree(S(40,100), S(100,200))     over a longer horizon
    agree(S(0,100),  S(100,120))     cumulative vs incremental
NULL for all: 0.5, plus a shuffled-coordinate control.

Also reported: agree(S(a,b), sgn(-m_b)) for each interval, since if momentum
predicted the NEXT interval that would be a generator too -- a different one
from the transfer question already settled at 0.601.
"""
import io, contextlib, subprocess, sys, math, json
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F

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
torch.manual_seed(1234); np.random.seed(1234)
G={}; _b=io.StringIO()
with contextlib.redirect_stdout(_b): exec(SRC,G)
model=G["model"]; gb=G["get_batch"]; lr=G["LR"]*5
named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
LAYER=3; pre=f"blocks.{LAYER}."
B1,B2,EPS,WD=0.9,0.95,1e-8,0.1
torch.manual_seed(17)
m={n:torch.zeros_like(p) for n,p in named}; v={n:torch.zeros_like(p) for n,p in named}
SNAP={0:{n:p.data.clone() for n,p in named if n.startswith(pre)}}
MSNAP={}
for st in range(1,201):
    x,y=gb(); _,l=model(x,y); model.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b1,b2=1-B1**st,1-B2**st
    with torch.no_grad():
        for n,p in named:
            g=p.grad if p.grad is not None else torch.zeros_like(p)
            m[n].mul_(B1).add_(g,alpha=1-B1); v[n].mul_(B2).addcmul_(g,g,value=1-B2)
            p.data.add_(-lr*((m[n]/b1)/((v[n]/b2).sqrt()+EPS)+WD*p.data))
    if st in (40,60,80,100,120,200):
        SNAP[st]={n:p.data.clone() for n,p in named if n.startswith(pre)}
        MSNAP[st]={n:m[n].clone() for n,_ in named if n.startswith(pre)}
def S(a,b): return {n:torch.sign(SNAP[b][n]-SNAP[a][n]) for n in SNAP[b]}
def agree(A,B):
    num=sum(float((A[n]==B[n]).sum()) for n in A)
    den=sum(A[n].numel() for n in A); return num/den
rng=torch.Generator().manual_seed(0)
def shuf(A):
    return {n:A[n].reshape(-1)[torch.randperm(A[n].numel(),generator=rng)].view_as(A[n])
            for n in A}
print(f"\n  orientation predictability, layer {LAYER}.  chance = 0.500\n")
pairs=[(("80","100"),("100","120")),   # short past -> short next: the gate
       (("60","80"),("80","100")),
       (("40","100"),("100","120")),   # long past, for comparison
       (("0","100"),("100","120")),
       (("100","120"),("120","200"))]
print(f"  {'past interval':>16}{'next interval':>16}{'agree':>9}{'shuffled':>10}")
for (a1,b1_),(a2,b2_) in pairs:
    A=S(int(a1),int(b1_)); B=S(int(a2),int(b2_))
    print(f"  {a1+'->'+b1_:>16}{a2+'->'+b2_:>16}{agree(A,B):>9.4f}{agree(shuf(A),B):>10.4f}")
print(f"\n  does a SHORT past interval beat the LONG one?")
A1=agree(S(80,100),S(100,120)); A2=agree(S(40,100),S(100,120))
print(f"    S(80,100) -> S(100,120): {A1:.4f}")
print(f"    S(40,100) -> S(100,120): {A2:.4f}   difference {A1-A2:+.4f}")
print(f"\n  momentum as a predictor of the NEXT interval\n")
print(f"  {'m at':>8}{'predicts':>16}{'agree':>9}")
for t,(a2,b2_) in ((80,("80","100")),(100,("100","120")),(120,("120","200"))):
    Mn={n:torch.sign(-MSNAP[t][n]) for n in MSNAP[t]}
    print(f"  {t:>8}{a2+'->'+b2_:>16}{agree(Mn,S(int(a2),int(b2_))):>9.4f}")
