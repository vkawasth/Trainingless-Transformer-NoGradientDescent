"""DOES UNLEARNING PRESERVE SUPPORT, OR IS SUPPORT JUST LOCALLY INERT?

J(U3,C3)=0.812 against a size-matched null of 0.144, robust to threshold
(+0.67 excess at q=0.10..0.25), present in 24/24 rows, and the temporal
version reproduces across seeds (0.429 vs 0.423). But |d_unlearn| = 3.25
against |d_learn| = 23.6, so the live alternative is LOCAL SUPPORT INERTIA:
any displacement that small leaves the support graph alone.

THREE STATES AT MATCHED ||dtheta|| = ||d_U3||, reference held at C3:
    U3     ascent on the O3 bucket
    U1     ascent on the O1 bucket -- directed, different capability
    N      isotropic noise -- undirected

  all ~0.8            -> local inertia; unlearning is not special
  U3 >> U1, N         -> O3-specific preservation
  U3 ~ U1 >> N        -> directed intervention preserves topology generally
  U3 ~ U1 ~ N, high   -> support is a locally stable invariant

Plus the nearly-free fiber comparison: J(U3,C2) and J(C2,C3), where
O3(U3) = O3(C2) by construction. If J separates them, the capability
observable and the support observable have different fibers.

NULLS: size-matched random supports, AND a row-permuted null that keeps every
support set intact while destroying which unit owns it -- the stronger null,
since a size-matched one reproduces only cardinality.
"""
import io, contextlib, subprocess, sys, json, time
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F

OUT="res_jctl.json"; RES={}
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
def bkt(i):
    for od in range(1,6):
        if i-od>=0 and len(nxt[(od,tuple(BASE[i-od:i]))])==1: return od
    return 6
POS=list(range(W,len(BASE)-1)); np.random.default_rng(3).shuffle(POS)
ALL=defaultdict(list)
for i in POS: ALL[bkt(i)].append(i)
FIT={b:ALL[b][:13] for b in (1,2,3)}
HELD3=ALL[3][len(ALL[3])//2:][:60]
def tens(idx):
    return (torch.tensor([[BASE[i-W+j] for j in range(W)] for i in idx]),
            torch.tensor([BASE[i] for i in idx]))
NU=24; PMAG=0.5; EPS_ASC,CAP,POLL=0.02,900,5

G=build(); model=G["model"]; gb=G["get_batch"]; LR=G["LR"]
torch.manual_seed(17)
ps=[p for p in model.parameters() if p.requires_grad]
P=sum(p.numel() for p in ps)
ff=model.blocks[0].ff; Wg=ff.g.weight
bx1,_=gb(); X3,Y3=tens(HELD3)
def flat(): return torch.cat([p.data.reshape(-1) for p in ps]).clone()
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
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
st=0; SD={}
for T in (60,160):
    while st<T:
        x,y=gb(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); st+=1
    SD[T]={k:v.clone() for k,v in model.state_dict().items()}
    RES[f"A_{T}"]=A().tolist(); RES[f"o3_{T}"]=acc3(); flush()
model.load_state_dict(SD[160]); th160=flat()
def ascend(b,stop):
    model.load_state_dict(SD[160]); Xf,Yf=tens(FIT[b]); n=0
    while n<CAP:
        if n%POLL==0 and acc3()<=stop: break
        model.zero_grad(); F.cross_entropy(model(Xf)[0][:,-1,:],Yf).backward()
        g=torch.cat([(p.grad.reshape(-1) if p.grad is not None else
                      torch.zeros(p.numel())) for p in ps]); gn=float(g.norm())
        with torch.no_grad():
            for p in ps:
                if p.grad is not None: p.data.add_(p.grad*(EPS_ASC/max(gn,1e-30)))
        n+=1
    return float((flat()-th160).norm())
dU3=ascend(3,RES["o3_60"]); RES["A_U3"]=A().tolist(); RES["o3_U3"]=acc3(); flush()
print(f"  U3: O3 {acc3():.3f}, ||d|| {dU3:.3f}",flush=True)
# U1 at MATCHED displacement
model.load_state_dict(SD[160]); Xf,Yf=tens(FIT[1]); n=0
while n<CAP and float((flat()-th160).norm())<dU3:
    model.zero_grad(); F.cross_entropy(model(Xf)[0][:,-1,:],Yf).backward()
    g=torch.cat([(p.grad.reshape(-1) if p.grad is not None else
                  torch.zeros(p.numel())) for p in ps]); gn=float(g.norm())
    with torch.no_grad():
        for p in ps:
            if p.grad is not None: p.data.add_(p.grad*(EPS_ASC/max(gn,1e-30)))
    n+=1
dU1=float((flat()-th160).norm()); RES["A_U1"]=A().tolist(); RES["o3_U1"]=acc3(); flush()
print(f"  U1: O3 {acc3():.3f}, ||d|| {dU1:.3f}",flush=True)
# noise at MATCHED displacement
gh=torch.Generator().manual_seed(77)
z=torch.randn(P,generator=gh); z=z/z.norm()*dU3
model.load_state_dict(SD[160]); setflat(th160+z)
dN=float((flat()-th160).norm()); RES["A_N"]=A().tolist(); RES["o3_N"]=acc3(); flush()
print(f"  N : O3 {acc3():.3f}, ||d|| {dN:.3f}",flush=True)

rows=lambda k: np.array(RES[k]).reshape(NU,-1)
n_t=rows("A_160").shape[1]
def supp(k,q=0.25):
    M=np.abs(rows(k)); kk=max(1,int(n_t*q))
    return [set(np.argsort(-M[i])[:kk]) for i in range(NU)]
def J(a,b): return float(np.mean([len(x&y)/max(len(x|y),1) for x,y in zip(supp(a),supp(b))]))
rng=np.random.default_rng(0); kk=max(1,int(n_t*0.25))
rand=np.mean([len(set(rng.choice(n_t,kk,0==1))&set(rng.choice(n_t,kk,0==1)))/
              max(len(set(rng.choice(n_t,kk,0==1))|set(rng.choice(n_t,kk,0==1))),1)
              for _ in range(400)])
S160=supp("A_160")
perm=[]
for _ in range(200):
    p_=rng.permutation(NU)
    perm.append(np.mean([len(S160[i]&S160[p_[i]])/max(len(S160[i]|S160[p_[i]]),1)
                         for i in range(NU)]))
print(f"\n  reference C3 (O3={RES['o3_160']:.3f}).  matched ||dtheta|| = {dU3:.3f}\n")
print(f"  {'state':>8}{'O3':>8}{'||d||':>9}{'J vs C3':>10}")
for k,lab in (("A_U3","U3"),("A_U1","U1"),("A_N","N"),("A_60","C2")):
    d={"A_U3":dU3,"A_U1":dU1,"A_N":dN,"A_60":float((flat()*0).norm())}[k]
    ds=f"{d:.3f}" if k!="A_60" else "--"
    print(f"  {lab:>8}{RES[k.replace('A_','o3_')]:>8.3f}{ds:>9}{J(k,'A_160'):>10.3f}")
print(f"\n  J(U3,C2) = {J('A_U3','A_60'):.3f}   [O3(U3)=O3(C2) by construction]")
print(f"  nulls: size-matched random {rand:.3f}   row-permuted "
      f"{np.mean(perm):.3f}+-{np.std(perm):.3f}")
flush()
