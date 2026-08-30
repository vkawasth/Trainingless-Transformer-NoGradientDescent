"""STABILITY RADII: DO SUBSTRATE, CAPABILITY AND GEOMETRY DECOUPLE UNDER
CONTROLLED DISPLACEMENT?

The intervention matrix so far:
    training     O changes, J evolves, F modest, H changes
    unlearning   O collapses, J inert, F modest, H strongly changes
    frozen K     O changes, J unchanged, F modest, H changes
    fixed attn   O changes, J unchanged, F modest, H changes
    new seed     O same, J collapses to near-null

So J looks like an initialisation-specific substrate rather than the learned
function. The sharp version of that claim is a difference in STABILITY RADIUS:
from one state C3, displace by r and track all four together.

    r in {0.25 .. 32}, isotropic, matched norm
    J(r)      support similarity to C3        substrate
    O3(r)     capability                       function
    dF(r)     1 - corr(log F_i(r), log F_i(0)) first-order metric
    dH(r)     1 - corr(h_i(r), h_i(0))         curvature field, h = H g3
    kappa(r)

Define r50(X) as the radius where observable X loses half its range. If
r50(J) >> r50(O3), the substrate is stable where the function is not, and the
separation is quantitative rather than anecdotal.

Isotropic displacement only: this is the null direction, so any decoupling seen
cannot be attributed to a semantic direction of motion.
"""
import io, contextlib, subprocess, sys, json, time
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F

OUT="res_radii.json"; RES={}
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
FIT3=ALL[3][:13]; HELD3=ALL[3][len(ALL[3])//2:][:60]
def tens(idx):
    return (torch.tensor([[BASE[i-W+j] for j in range(W)] for i in idx]),
            torch.tensor([BASE[i] for i in idx]))
NU=24; PMAG=0.5; SUB=200000
RADII=[0.0,0.25,0.5,1.0,2.0,4.0,8.0,16.0,32.0]

G=build(); model=G["model"]; gb=G["get_batch"]; LR=G["LR"]
torch.manual_seed(17)
ps=[p for p in model.parameters() if p.requires_grad]
P=sum(p.numel() for p in ps)
sub=torch.randperm(P,generator=torch.Generator().manual_seed(9))[:SUB]
ff=model.blocks[0].ff; Wg=ff.g.weight
bx1,_=gb(); X3,Y3=tens(HELD3); Xf,Yf=tens(FIT3)
flat=lambda: torch.cat([p.data.reshape(-1) for p in ps]).clone()
def setflat(v):
    with torch.no_grad():
        i=0
        for p in ps:
            q=p.numel(); p.data.copy_(v[i:i+q].view_as(p)); i+=q
@torch.no_grad()
def resp(x):
    h=model.te(x)+model.pe(torch.arange(x.shape[1]))
    h=model.blocks[0].attn(h)
    return ff.n(h+ff.o(F.silu(ff.g(h))*ff.v(h)))
@torch.no_grad()
def A():
    base=resp(bx1).abs().mean(dim=(0,1))
    M=np.zeros((NU,base.shape[0])); gg=torch.Generator().manual_seed(5150)
    for i in range(NU):
        row=Wg[i].clone()
        d=torch.randn(row.shape,generator=gg); d=d/d.norm()*PMAG*row.norm()
        Wg[i].add_(d); M[i]=(resp(bx1).abs().mean(dim=(0,1))-base).numpy()
        Wg[i].copy_(row)
    return M
@torch.no_grad()
def acc3():
    model.eval(); r=float((model(X3)[0][:,-1,:].argmax(-1)==Y3).float().mean())
    model.train(); return r
def fh():
    Fd=torch.zeros(P)
    for _ in range(8):
        x,y=gb(); model.zero_grad(); model(x,y)[1].backward()
        Fd+=torch.cat([(p.grad.reshape(-1) if p.grad is not None
                        else torch.zeros(p.numel())) for p in ps])**2
    Fd/=8
    model.zero_grad(); F.cross_entropy(model(Xf)[0][:,-1,:],Yf).backward()
    v=torch.cat([(p.grad.reshape(-1) if p.grad is not None else
                  torch.zeros(p.numel())) for p in ps]).clone()
    vn=max(float(v.norm()),1e-30); v=v/vn
    x,y=gb(); model.zero_grad(); l=model(x,y)[1]
    gr=torch.autograd.grad(l,ps,create_graph=True)
    gf=torch.cat([q.reshape(-1) for q in gr])
    hh=torch.autograd.grad((gf*v).sum(),ps)
    h=torch.cat([q.reshape(-1) for q in hh]).detach()
    return (torch.log(Fd[sub]+1e-30).numpy(), h[sub].numpy(), float(h@v))
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
st=0
while st<160:
    x,y=gb(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); st+=1
th0=flat(); A0=A(); F0,h0,k0=fh(); o0=acc3()
nt=A0.shape[1]
def sp(M,q=0.25):
    X=np.abs(M); kk=max(1,int(nt*q))
    return [set(np.argsort(-X[i])[:kk]) for i in range(NU)]
S0=sp(A0)
Jf=lambda M: float(np.mean([len(a&b)/max(len(a|b),1) for a,b in zip(sp(M),S0)]))
gh=torch.Generator().manual_seed(77)
print(f"  C3: O3 {o0:.3f}  kappa {k0:+.4f}\n")
print(f"  {'r':>7}{'J':>8}{'O3':>8}{'dF':>8}{'dH':>8}{'kappa':>10}")
for r in RADII:
    if r==0: setflat(th0)
    else:
        z=torch.randn(P,generator=gh); setflat(th0+z/z.norm()*r)
    Ar=A(); Fr,hr,kr=fh(); orr=acc3()
    dF=1-float(np.corrcoef(Fr,F0)[0,1]); dH=1-float(np.corrcoef(hr,h0)[0,1])
    RES[f"r{r}"]=dict(J=Jf(Ar),o3=orr,dF=dF,dH=dH,kappa=kr); flush()
    print(f"  {r:>7.2f}{Jf(Ar):>8.3f}{orr:>8.3f}{dF:>8.3f}{dH:>8.3f}{kr:>+10.4f}")
setflat(th0)
print(f"\n  row-permuted J null 0.181+-0.035.  r50 = radius at half range")
def r50(key,lo=None):
    v=[(r,RES[f'r{r}'][key]) for r in RADII]
    a,b=v[0][1],(lo if lo is not None else v[-1][1]); mid=(a+b)/2
    for r,x in v:
        if (a>b and x<=mid) or (a<b and x>=mid): return r
    return None
for k in ("J","o3","dF","dH"):
    print(f"    r50({k}) = {r50(k)}")
flush()
