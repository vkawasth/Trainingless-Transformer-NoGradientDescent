"""DOES THE PRECEDING MORPHISM PREDICT THE NEXT ONE, AND DOES THE PREDICTION
TRANSFER? PREREGISTERED, NULL IN THE SAME PROCESS.

Pure dynamical question, no corpus extractor and no optimiser-state reading:

    S(100,120) = sgn(W120 - W100)        the preceding morphism
    S(120,130) = sgn(W130 - W120)        the next morphism

  PART A (no training): agreement of the predictor with the actual field,
    against two nulls -- random signs, and S(100,120) with coordinates
    shuffled (preserves the sign multiset, destroys ownership).

  PART B (transplant): the predicted field, scaled to ||W130 - W120||, is
    added to a student's block-3-equivalent weights in every block, exactly as
    every other ladder arm. Its dose-matched null is a random orientation at
    the same norm, run in THIS process so a truncated run still yields the
    falsifying half.

PREREGISTERED READING, fixed before the numbers exist:
    dO3 = O3(predicted) - O3(random at same norm)
    dO3 <= 0            -> prediction is not functional; generator branch closed
    dO3 ~ that of the TRUE field (+0.230 at t=120)  -> forward relational
                           dynamics is computable
    0 < dO3 < +0.230    -> partial; quantitative target for any generator
No post-hoc thresholds. Loss and O3 must agree in sign for a positive call.
"""
import io, contextlib, subprocess, sys, math, json, time, gc
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F

OUT="res_markov.json"; RES={}
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
B1,B2,EPS,WD,LRM,LAYER=0.9,0.95,1e-8,0.1,5.0,3
pre=f"blocks.{LAYER}."
def newmodel(seed):
    torch.manual_seed(1234); np.random.seed(1234)
    G={}; b=io.StringIO()
    with contextlib.redirect_stdout(b): exec(SRC,G)
    return G
def adam_step(model,named,m,v,st,lr):
    b1,b2=1-B1**st,1-B2**st
    with torch.no_grad():
        for n,p in named:
            g=p.grad if p.grad is not None else torch.zeros_like(p)
            m[n].mul_(B1).add_(g,alpha=1-B1); v[n].mul_(B2).addcmul_(g,g,value=1-B2)
            p.data.add_(-lr*((m[n]/b1)/((v[n]/b2).sqrt()+EPS)+WD*p.data))
def probe(model,EV):
    with torch.no_grad():
        L=sum(float(model(x,y)[1]) for x,y in EV)/len(EV)
        model.eval(); a={}
        for b in BU:
            X,Y=tens(HELD[b]); a[b]=float((model(X)[0][:,-1,:].argmax(-1)==Y).float().mean())
        model.train()
    return L,a

# ---------- teacher to 130 ----------
t0=time.time(); G=newmodel(17); tm=G["model"]; gb=G["get_batch"]; lr=G["LR"]*LRM
tn=[(n,p) for n,p in tm.named_parameters() if p.requires_grad]
EV=[gb() for _ in range(6)]
torch.manual_seed(17)
m={n:torch.zeros_like(p) for n,p in tn}; v={n:torch.zeros_like(p) for n,p in tn}
SN={}
for st in range(1,131):
    x,y=gb(); _,l=tm(x,y); tm.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(tm.parameters(),1.0)
    adam_step(tm,tn,m,v,st,lr)
    if st in (100,120,130):
        SN[st]={n:p.data.clone() for n,p in tn if n.startswith(pre)}
        L,a=probe(tm,EV)
        print(f"  teacher t={st}: L {L:.4f}  O {'/'.join(f'{a[b]:.3f}' for b in BU)}",flush=True)
NB=len(tm.blocks); del G,tm,m,v; gc.collect()

# ---------- PART A ----------
PRED={n:torch.sign(SN[120][n]-SN[100][n]) for n in SN[120]}
ACT ={n:torch.sign(SN[130][n]-SN[120][n]) for n in SN[130]}
DW  ={n:SN[130][n]-SN[120][n] for n in SN[130]}
def agree(A,B):
    return sum(float((A[n]==B[n]).sum()) for n in A)/sum(A[n].numel() for n in A)
rg=torch.Generator().manual_seed(0)
SHUF={n:PRED[n].reshape(-1)[torch.randperm(PRED[n].numel(),generator=rg)].view_as(PRED[n])
      for n in PRED}
RAND={n:torch.where(torch.rand(PRED[n].shape,generator=rg)<0.5,
                    -torch.ones_like(PRED[n]),torch.ones_like(PRED[n])) for n in PRED}
RES["A"]=dict(pred=agree(PRED,ACT),shuf=agree(SHUF,ACT),rand=agree(RAND,ACT)); flush()
print(f"\n  PART A  agreement with the actual S(120,130)")
print(f"    S(100,120)  predicted : {RES['A']['pred']:.4f}")
print(f"    shuffled ownership    : {RES['A']['shuf']:.4f}")
print(f"    random signs          : {RES['A']['rand']:.4f}   (chance 0.500)\n",flush=True)

# ---------- PART B ----------
def student(label,field):
    G=newmodel(23); sm=G["model"]; gbs=G["get_batch"]; lrs=G["LR"]*LRM
    sn=[(n,p) for n,p in sm.named_parameters() if p.requires_grad]
    EVs=[gbs() for _ in range(6)]
    torch.manual_seed(23)
    ms={n:torch.zeros_like(p) for n,p in sn}; vs={n:torch.zeros_like(p) for n,p in sn}
    for st in range(1,31):
        x,y=gbs(); _,l=sm(x,y); sm.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(sm.parameters(),1.0)
        adam_step(sm,sn,ms,vs,st,lrs)
    if field is not None:
        with torch.no_grad():
            for n,p in sn:
                for bi in range(NB):
                    tg=f"blocks.{bi}."
                    if not n.startswith(tg): continue
                    src=pre+n[len(tg):]
                    if src not in field or p.data.shape!=field[src].shape: continue
                    sg=field[src]
                    p.data.add_(sg*(DW[src].norm()/max(float(sg.norm()),1e-30)))
    rec=[]
    for k in range(0,31):
        if k>0:
            x,y=gbs(); _,l=sm(x,y); sm.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(sm.parameters(),1.0)
            adam_step(sm,sn,ms,vs,30+k,lrs)
        if k in (0,10,20,30):
            L,a=probe(sm,EVs); rec.append((k,L,a))
            print(f"  {label:>10} k={k:>2}: L {L:>8.4f}  O "
                  f"{'/'.join(f'{a[b]:.3f}' for b in BU)}",flush=True)
    RES[label]=[(k,L,{str(x):y for x,y in a.items()}) for k,L,a in rec]; flush()
    del G,sm,ms,vs; gc.collect()

student("random",RAND)      # NULL FIRST
student("predicted",PRED)
student("actual",ACT)
student("control",None)
print(f"\n  done ({time.time()-t0:.0f}s)"); flush()
