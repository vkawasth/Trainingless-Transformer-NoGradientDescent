"""RELIABILITY GATE FOR THE CURVATURE AND FISHER FIELDS.

d_H(C3,C3) = 0.395 -- the field differs from ITSELF by 0.4 under one HVP on one
batch, so nothing below that scale was interpretable and r50(dH)=0.25 was an
artefact of the floor. d_F sat at 0.005 out to r=16, which says the estimator
cannot resolve rather than that Fisher is invariant.

Both go through the same gate that passed A and failed Phi, with the batch
count B as the knob:

    ceiling(B)  rho(h^a, h^b) at the SAME state, two independent draws of B
                batches each. Must approach 1 as B grows.
    floor(B)    rho(h(C3), h(theta_r)) at a displacement large enough that the
                fields should genuinely differ (r=16).
    null        shuffled.

The usable B is the smallest with ceiling - floor well separated. Only then is
d_H(C3, theta_r) a measurement rather than noise.

Same for F, where the question is the opposite: does ANY displacement move it,
or is the estimator simply blind? Reported as the ratio of the r=16 distance to
the same-state distance -- if that ratio is ~1, F cannot discriminate.
"""
import io, contextlib, subprocess, sys, json, time
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F

OUT="res_hgate.json"; RES={}
def flush(): json.dump(RES,open(OUT,"w"),indent=1,default=float)
subprocess.run([sys.executable,"build_corpus.py","--out","/tmp","--loops","300"],
               check=True,capture_output=True)
RAW=open("compiler_geometri_patched_86.py").read()
SRC=RAW[:RAW.find("# \u2500\u2500 PHASE 3")]
for o,n in [("D=256; N_HEADS=4","D=128; N_HEADS=4"),
            ("for mf_r in range(1, 16):","for mf_r in range(1, 3):"),
            ("    if pc == N_STU-1:","    if False:"),
            ("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:","    if False:")]:
    assert SRC.count(o)==1; SRC=SRC.replace(o,n,1)
EO="evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)"
EN=("_v0=np.random.RandomState(7).randn(L_sym.shape[0])\n"
    "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000,v0=_v0)\n"
    "evecs=evecs*np.sign(evecs[np.argmax(np.abs(evecs),axis=0),np.arange(evecs.shape[1])])")
assert SRC.count(EO)==1; SRC=SRC.replace(EO,EN,1)
def build():
    torch.manual_seed(1234); np.random.seed(1234)
    G={}; b=io.StringIO()
    with contextlib.redirect_stdout(b): exec(SRC,G)
    return G
ids=json.load(open("/tmp/train_ids.json")); BASE=ids[:1364]; W=8
nxt=defaultdict(set)
for od in range(1,6):
    for i in range(len(BASE)-od): nxt[(od,tuple(BASE[i:i+od]))].add(BASE[i+od])
def bk(i):
    for od in range(1,6):
        if i-od>=0 and len(nxt[(od,tuple(BASE[i-od:i]))])==1: return od
    return 6
POS=list(range(W,len(BASE)-1)); np.random.default_rng(3).shuffle(POS)
ALL=defaultdict(list)
for i in POS: ALL[bk(i)].append(i)
FIT3=ALL[3][:13]
def tens(idx):
    return (torch.tensor([[BASE[i-W+j] for j in range(W)] for i in idx]),
            torch.tensor([BASE[i] for i in idx]))
SUB=200000; BS=[1,4,16,64]

G=build(); model=G["model"]; gb=G["get_batch"]; LR=G["LR"]
torch.manual_seed(17)
ps=[p for p in model.parameters() if p.requires_grad]
P=sum(p.numel() for p in ps)
sub=torch.randperm(P,generator=torch.Generator().manual_seed(9))[:SUB]
Xf,Yf=tens(FIT3)
flat=lambda: torch.cat([p.data.reshape(-1) for p in ps]).clone()
def setflat(v):
    with torch.no_grad():
        i=0
        for p in ps:
            q=p.numel(); p.data.copy_(v[i:i+q].view_as(p)); i+=q
def g3unit():
    model.zero_grad(); F.cross_entropy(model(Xf)[0][:,-1,:],Yf).backward()
    v=torch.cat([(p.grad.reshape(-1) if p.grad is not None else
                  torch.zeros(p.numel())) for p in ps]).clone()
    return v/max(float(v.norm()),1e-30)
def hfield(B,v):
    acc=torch.zeros(P)
    for _ in range(B):
        x,y=gb(); model.zero_grad(); l=model(x,y)[1]
        gr=torch.autograd.grad(l,ps,create_graph=True)
        gf=torch.cat([q.reshape(-1) for q in gr])
        hh=torch.autograd.grad((gf*v).sum(),ps)
        acc+=torch.cat([q.reshape(-1) for q in hh]).detach()
    return (acc/B)[sub].numpy()
def ffield(B):
    acc=torch.zeros(P)
    for _ in range(B):
        x,y=gb(); model.zero_grad(); model(x,y)[1].backward()
        acc+=torch.cat([(p.grad.reshape(-1) if p.grad is not None
                         else torch.zeros(p.numel())) for p in ps])**2
    return torch.log(acc/B+1e-30)[sub].numpy()
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
st=0
while st<160:
    x,y=gb(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); st+=1
th0=flat(); v0=g3unit()
gh=torch.Generator().manual_seed(77)
z=torch.randn(P,generator=gh); z=z/z.norm()*16.0
c=lambda a,b: float(np.corrcoef(a,b)[0,1])
print(f"  curvature field h = H g3, subsample {SUB} coords\n")
print(f"  {'B':>4}{'ceiling':>10}{'floor(r=16)':>13}{'separation':>12}{'sec':>7}")
for B in BS:
    t0=time.time()
    setflat(th0); ha=hfield(B,v0); hb=hfield(B,v0)
    setflat(th0+z); hr=hfield(B,v0)
    setflat(th0)
    ceil=c(ha,hb); flo=c(ha,hr)
    RES[f"h_B{B}"]=dict(ceiling=ceil,floor=flo); flush()
    print(f"  {B:>4}{ceil:>10.3f}{flo:>13.3f}{ceil-flo:>12.3f}{time.time()-t0:>7.0f}")
print(f"\n  Fisher field log F\n")
print(f"  {'B':>4}{'ceiling':>10}{'floor(r=16)':>13}{'separation':>12}")
for B in (4,16,64):
    setflat(th0); fa=ffield(B); fb=ffield(B)
    setflat(th0+z); fr=ffield(B)
    setflat(th0)
    ceil=c(fa,fb); flo=c(fa,fr)
    RES[f"F_B{B}"]=dict(ceiling=ceil,floor=flo); flush()
    print(f"  {B:>4}{ceil:>10.4f}{flo:>13.4f}{ceil-flo:>12.4f}")
print(f"\n  usable B = smallest with ceiling near 1 and clear separation.")
print(f"  if separation stays ~0 the field cannot discriminate at r=16.")
flush()
