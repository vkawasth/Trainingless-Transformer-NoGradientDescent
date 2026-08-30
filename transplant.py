"""TEACHER LAYER-3 -> STUDENT ALL LAYERS, THREE DONOR ZONES, FOUR-WAY SEPARATION.

Replicates the reported result -- teacher trained, its constriction-layer
weights copied to ALL student blocks, student continues and converges -- and
separates what carries it.

  teacher   trained to 200, snapshots at 40 / 100 / 120 / 200
  student   independent seed, trained to 30, then transplanted

DONOR ZONES: teacher layer 3 at t = 40, 100, 120. If the late donor works and
the early one does not, the transplanted object is built during consolidation;
if all three work, it is present from formation.

FOUR-WAY, at the t=120 donor:
    W        block-3 weights only
    S        block-3 Adam (m,v) only
    W+S      both
    ctrl     student continues untouched
Weights alone -> layer-level structural state. Adam state alone ->
optimizer-mediated. Only together -> coupled object. Neither -> distributed.

Evaluated at 0,1,2,5,10,20,30 steps after the transplant, because an immediate
discontinuity is qualitatively different from ordinary learning over 30 steps.
Tracked: L, O1/O2/O3, ||dW|| in block 3 and in the rest, so "state
transplantation" is separable from "a better initialisation".

D=128: a single D=256 arm was OOM-killed in this 3 GB container.
"""
import io, contextlib, subprocess, sys, math, json, time, gc, copy
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F

OUT="res_transplant.json"; RES={}
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
B1,B2,EPS,WD,LRM=0.9,0.95,1e-8,0.1,5.0
DONORS=[40,100,120]; EVAL=[0,1,2,5,10,20,30]; LAYER=3

def newmodel(seed):
    torch.manual_seed(1234); np.random.seed(1234)
    G={}; b=io.StringIO()
    with contextlib.redirect_stdout(b): exec(SRC,G)
    return G

def adam_step(model,named,m,v,st,lr):
    b1,b2=1-B1**st,1-B2**st; tot=0.0
    with torch.no_grad():
        for n,p in named:
            g=p.grad if p.grad is not None else torch.zeros_like(p)
            m[n].mul_(B1).add_(g,alpha=1-B1); v[n].mul_(B2).addcmul_(g,g,value=1-B2)
            u=-lr*((m[n]/b1)/((v[n]/b2).sqrt()+EPS)+WD*p.data)
            tot+=float((u*u).sum()); p.data.add_(u)
    return math.sqrt(tot)

def probe(model,EV):
    with torch.no_grad():
        L=sum(float(model(x,y)[1]) for x,y in EV)/len(EV)
        model.eval(); a={}
        for b in BU:
            X,Y=tens(HELD[b]); a[b]=float((model(X)[0][:,-1,:].argmax(-1)==Y).float().mean())
        model.train()
    return L,a

# ---------------- teacher ----------------
t0=time.time()
G=newmodel(17); tm=G["model"]; gb=G["get_batch"]; lr=G["LR"]*LRM
tnamed=[(n,p) for n,p in tm.named_parameters() if p.requires_grad]
EV=[gb() for _ in range(6)]
torch.manual_seed(17)
m={n:torch.zeros_like(p) for n,p in tnamed}; v={n:torch.zeros_like(p) for n,p in tnamed}
SNAP={}
for st in range(1,201):
    x,y=gb(); _,l=tm(x,y); tm.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(tm.parameters(),1.0)
    adam_step(tm,tnamed,m,v,st,lr)
    if st in DONORS+[200]:
        L,a=probe(tm,EV)
        pre=f"blocks.{LAYER}."
        SNAP[st]=dict(W={n:p.data.clone() for n,p in tnamed if n.startswith(pre)},
                      m={n:m[n].clone() for n,_ in tnamed if n.startswith(pre)},
                      v={n:v[n].clone() for n,_ in tnamed if n.startswith(pre)})
        RES[f"teacher_{st}"]=dict(L=L,acc=a); flush()
        print(f"  teacher t={st:>3}: L {L:.4f}  O "
              f"{'/'.join(f'{a[b]:.3f}' for b in BU)}  ({time.time()-t0:.0f}s)",flush=True)
NB=len(tm.blocks)
del G,tm,m,v; gc.collect()

# ---------------- student ----------------
def student_run(label,donor,what):
    G=newmodel(23); sm=G["model"]; gbs=G["get_batch"]; lrs=G["LR"]*LRM
    snamed=[(n,p) for n,p in sm.named_parameters() if p.requires_grad]
    EVs=[gbs() for _ in range(6)]
    torch.manual_seed(23)
    ms={n:torch.zeros_like(p) for n,p in snamed}
    vs={n:torch.zeros_like(p) for n,p in snamed}
    for st in range(1,31):
        x,y=gbs(); _,l=sm(x,y); sm.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(sm.parameters(),1.0)
        adam_step(sm,snamed,ms,vs,st,lrs)
    pre=f"blocks.{LAYER}."
    before={n:p.data.clone() for n,p in snamed}
    if donor is not None:
        S=SNAP[donor]
        with torch.no_grad():
            for n,p in snamed:
                for bi in range(NB):
                    tgt=f"blocks.{bi}."
                    if not n.startswith(tgt): continue
                    src=pre+n[len(tgt):]
                    if src not in S["W"]: continue
                    if p.data.shape!=S["W"][src].shape: continue
                    if "W" in what: p.data.copy_(S["W"][src])
                    if "S" in what:
                        ms[n].copy_(S["m"][src]); vs[n].copy_(S["v"][src])
    rec=[]
    L,a=probe(sm,EVs)
    d3=math.sqrt(sum(float(((p.data-before[n])**2).sum()) for n,p in snamed if n.startswith(pre)))
    dr=math.sqrt(sum(float(((p.data-before[n])**2).sum()) for n,p in snamed if not n.startswith(pre)))
    rec.append(dict(k=0,L=L,acc=a,d3=d3,drest=dr))
    print(f"  {label:>14} k= 0: L {L:>9.4f}  O {'/'.join(f'{a[b]:.3f}' for b in BU)}"
          f"  dW3 {d3:>6.2f} dWrest {dr:>7.2f}",flush=True)
    for st in range(31,31+30):
        x,y=gbs(); _,l=sm(x,y); sm.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(sm.parameters(),1.0)
        adam_step(sm,snamed,ms,vs,st,lrs)
        k=st-30
        if k in EVAL:
            L,a=probe(sm,EVs)
            d3=math.sqrt(sum(float(((p.data-before[n])**2).sum()) for n,p in snamed if n.startswith(pre)))
            dr=math.sqrt(sum(float(((p.data-before[n])**2).sum()) for n,p in snamed if not n.startswith(pre)))
            rec.append(dict(k=k,L=L,acc=a,d3=d3,drest=dr))
            print(f"  {label:>14} k={k:>2}: L {L:>9.4f}  O "
                  f"{'/'.join(f'{a[b]:.3f}' for b in BU)}  dW3 {d3:>6.2f} dWrest {dr:>7.2f}",
                  flush=True)
    RES[label]=rec; flush()
    del G,sm,ms,vs; gc.collect()

student_run("ctrl",None,"")
for d in DONORS: student_run(f"W_donor{d}",d,"W")
student_run("S_donor120",120,"S")
student_run("WS_donor120",120,"WS")
print(f"\n  done ({time.time()-t0:.0f}s)")
flush()
