"""CALIBRATION TRIANGLE: WHAT IS rho WORTH WHEN THE ANSWER IS KNOWN?

The fixed-family matrix gave rho(U_O2,U_O3)=+0.609 against +0.088 and +0.254
for the U_O1 pairs. Whether +0.609 means anything is undecidable without
knowing the observable's ceiling and floor. So, before any interpretation:

  CEILING   rho(C160^a, C160^b) -- the SAME state, probed twice with
            independent eval batches. Anything below this is measurement noise.
  FLOOR     rho(C160^seed1, C160^seed2) -- the same checkpoint of an
            independently trained model. Two realisations of the same
            functional object; how much of Phi is realisation-specific?
  NULL      rho(C160, shuffled C160) -- zero-structure reference.

Then the U_O2/U_O3 number is placed on that scale.

Probe apparatus is EXTERNAL to every state, per the rule the last run
violated: one direction family drawn once from a fixed seed, one eps, applied
identically everywhere. The eval batches used to compute CE are what varies
between the two ceiling measurements, so the ceiling includes probe noise but
not basis change.

Reported with K=16 and K=32 directions, since the correlation's own error bar
scales as 1/sqrt(K) and 16 may simply be too few: for K=16 the 95% interval on
a true rho of 0 is roughly +-0.5, which would make +0.609 unremarkable.
"""
import io, contextlib, subprocess, sys, json, time
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F

OUT="res_calib.json"; RES={}
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
BU=[b for b in (1,2,3) if len(ALL.get(b,[]))>=16]
FIT={b:ALL[b][:13] for b in BU}
# two DISJOINT held sets for bucket 3: the ceiling measurement varies only this
H3=ALL[3][len(ALL[3])//2:]
HELD_A=H3[:len(H3)//2] or H3[:1]; HELD_B=H3[len(H3)//2:] or H3[:1]
def tens(idx):
    return (torch.tensor([[BASE[i-W+j] for j in range(W)] for i in idx]),
            torch.tensor([BASE[i] for i in idx]))
EPS=0.25; KMAX=32; EPS_ASC,CAP,POLL,CE_TGT=0.02,900,5,26.0

def run(seed,tag):
    t0=time.time(); G=build(); model=G["model"]; gb=G["get_batch"]; LR=G["LR"]
    torch.manual_seed(seed)
    ps=[p for p in model.parameters() if p.requires_grad]
    P=sum(p.numel() for p in ps)
    def setflat(v):
        with torch.no_grad():
            i=0
            for p in ps:
                q=p.numel(); p.data.copy_(v[i:i+q].view_as(p)); i+=q
    flat=lambda: torch.cat([p.data.reshape(-1) for p in ps]).clone()
    XA,YA=tens(HELD_A); XB,YB=tens(HELD_B)
    @torch.no_grad()
    def ce(X,Y):
        model.eval(); r=float(F.cross_entropy(model(X)[0][:,-1,:],Y)); model.train(); return r
    # ONE external family, drawn from a fixed seed, shared by every state and seed
    gg=torch.Generator().manual_seed(4242)
    FAM=[d/d.norm() for d in (torch.randn(P,generator=gg) for _ in range(KMAX))]
    def phi(X,Y):
        base=flat(); b=ce(X,Y); out=[]
        for d in FAM:
            setflat(base+EPS*d); out.append(ce(X,Y)-b)
        setflat(base); return out
    opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
    st=0
    while st<160:
        x,y=gb(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); st+=1
    sd160={k:v.clone() for k,v in model.state_dict().items()}
    RES[f"C160A_{tag}"]=phi(XA,YA); flush()
    RES[f"C160B_{tag}"]=phi(XB,YB); flush()
    print(f"  {tag}: ceiling pair done ({time.time()-t0:.0f}s)",flush=True)
    if tag=="s17":
        for b in BU:
            Xf,Yf=tens(FIT[b]); model.load_state_dict(sd160); n=0
            while n<CAP:
                if n%POLL==0 and ce(XA,YA)>=CE_TGT: break
                model.zero_grad(); F.cross_entropy(model(Xf)[0][:,-1,:],Yf).backward()
                g=torch.cat([(p.grad.reshape(-1) if p.grad is not None else
                              torch.zeros(p.numel())) for p in ps]); gn=float(g.norm())
                with torch.no_grad():
                    for p in ps:
                        if p.grad is not None: p.data.add_(p.grad*(EPS_ASC/max(gn,1e-30)))
                n+=1
            RES[f"U_O{b}_{tag}"]=phi(XA,YA); flush()
            print(f"    U_O{b} done ({time.time()-t0:.0f}s)",flush=True)
    del G,model; import gc; gc.collect()

run(17,"s17"); run(23,"s23")
def c(a,b,K): return float(np.corrcoef(np.array(a)[:K],np.array(b)[:K])[0,1])
for K in (16,32):
    print(f"\n  === K = {K} directions ===")
    print(f"  ceiling  same state, two probe batches : "
          f"{c(RES['C160A_s17'],RES['C160B_s17'],K):+.3f}   "
          f"(seed 23: {c(RES['C160A_s23'],RES['C160B_s23'],K):+.3f})")
    print(f"  floor    step 160, different seed      : "
          f"{c(RES['C160A_s17'],RES['C160A_s23'],K):+.3f}")
    rng=np.random.default_rng(0)
    sh=[float(np.corrcoef(rng.permutation(np.array(RES['C160A_s17'])[:K]),
                          np.array(RES['C160B_s17'])[:K])[0,1]) for _ in range(400)]
    print(f"  null     shuffled                      : {np.mean(sh):+.3f} +- {np.std(sh):.3f}")
    U=[f"U_O{b}_s17" for b in BU if f"U_O{b}_s17" in RES]
    if len(U)>=2:
        print(f"  ascent-state pairs:")
        for i,a in enumerate(U):
            for b in U[i+1:]:
                print(f"    {a[:5]} vs {b[:5]}                        : {c(RES[a],RES[b],K):+.3f}")
flush()
