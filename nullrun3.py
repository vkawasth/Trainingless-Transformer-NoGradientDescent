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
import io, contextlib, subprocess, sys, math, json, time, gc, copy, os
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F

OUT=f"res_tp_{sys.argv[1] if len(sys.argv)>1 else 'ctrl'}.json"; RES={}
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
DONOR_FOR_G=120
DONORS=[40,55,60,70,80,100,120]; EVAL=[0,1,2,5,10,20,30]; LAYER=3

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
SNAP={}; TSTATE={}; GSIGN={}; NACC=64
TMAX=int(os.environ.get("TMAX","200"))
for st in range(1,TMAX+1):
    x,y=gb(); _,l=tm(x,y); tm.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(tm.parameters(),1.0)
    adam_step(tm,tnamed,m,v,st,lr)
    if st in DONORS+[TMAX]:
        L,a=probe(tm,EV)
        pre=f"blocks.{LAYER}."
        TSTATE[st]={n:p.data.clone() for n,p in tnamed}
        if st==DONOR_FOR_G:
            _acc={n:torch.zeros_like(p) for n,p in tnamed if n.startswith(f"blocks.{LAYER}.")}
            for _ in range(NACC):
                _x,_y=gb(); tm.zero_grad(); tm(_x,_y)[1].backward()
                for n,p in tnamed:
                    if n.startswith(f"blocks.{LAYER}.") and p.grad is not None:
                        _acc[n]+=p.grad.detach()
            for n in _acc: GSIGN[n]=-_acc[n]/NACC      # descent direction
            _ag=torch.cat([v.reshape(-1) for v in _acc.values()])
            print(f"  gsign built at t={st} from {NACC} batches, |g| {float(_ag.norm()):.4f}",
                  flush=True)
        SNAP[st]=dict(W={n:p.data.clone() for n,p in tnamed if n.startswith(pre)},
                      m={n:m[n].clone() for n,_ in tnamed if n.startswith(pre)},
                      v={n:v[n].clone() for n,_ in tnamed if n.startswith(pre)})
        RES[f"teacher_{st}"]=dict(L=L,acc=a); flush()
        print(f"  teacher t={st:>3}: L {L:.4f}  O "
              f"{'/'.join(f'{a[b]:.3f}' for b in BU)}  ({time.time()-t0:.0f}s)",flush=True)
NB=len(tm.blocks)
# --- Q2: temporal emergence of the sign field, computed from snapshots only ---
if "--signtime" in sys.argv:
    G2={}; _b2=io.StringIO()
    torch.manual_seed(1234); np.random.seed(1234)
    with contextlib.redirect_stdout(_b2): exec(SRC,G2)
    sm0=G2["model"]; gb0=G2["get_batch"]
    sn0=[(n,p) for n,p in sm0.named_parameters() if p.requires_grad]
    torch.manual_seed(23)
    m0={n:torch.zeros_like(p) for n,p in sn0}; v0={n:torch.zeros_like(p) for n,p in sn0}
    for st_ in range(1,31):
        x,y=gb0(); _,l=sm0(x,y); sm0.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(sm0.parameters(),1.0)
        adam_step(sm0,sn0,m0,v0,st_,G2["LR"]*LRM)
    SW={n:p.data.clone() for n,p in sn0 if n.startswith(pre)}
    SG={}
    for t_ in DONORS:
        SG[t_]={n:torch.sign(SNAP[t_]["W"][n]-SW[n]) for n in SW}
    def agree(a,b):
        num=sum(float((a[n]==b[n]).sum()) for n in a)
        den=sum(a[n].numel() for n in a); return num/den
    print(f"\n  sign-field agreement between donor times (0.5 = chance)\n",flush=True)
    ts=sorted(SG)
    print("  "+"".join(f"{t_:>9}" for t_ in ts),flush=True)
    for a_ in ts:
        print(f"  t{a_:<4}"+"".join(f"{agree(SG[a_],SG[b_]):>9.3f}" for b_ in ts),flush=True)
    sys.exit(0)
# UNLEARNED DONOR: from t=120, ascend on the O3 bucket until O3 falls to its
# t=40 level. Same checkpoint, same layer, one difference -- capability removed.
ONLY3=False
SSEED=int(os.environ.get("SSEED","23")); RSEED=int(os.environ.get("RSEED","999"))
MASKSPEC=(sys.argv[2] if len(sys.argv)>2 else "mag", float(sys.argv[3]) if len(sys.argv)>3 else 1.0)
FIT3=BKT[3][:13]
Xf=torch.tensor([[BASE[i-W+j] for j in range(W)] for i in FIT3])
Yf=torch.tensor([BASE[i] for i in FIT3])
DO_UNLEARN = (len(sys.argv)>1 and sys.argv[1] in ("WU","WU3")) and 120 in TSTATE
if DO_UNLEARN:
  with torch.no_grad():
    for n,p in tnamed: p.data.copy_(TSTATE[120][n])
  tgt=RES["teacher_40"]["acc"][3]; nasc=0
  while nasc<4000:
      if nasc%5==0:
          _,aa=probe(tm,EV)
          if aa[3]<=tgt: break
      tm.zero_grad(); F.cross_entropy(tm(Xf)[0][:,-1,:],Yf).backward()
      # SURGICAL: ascent restricted to block LAYER. The whole-network version
      # spread its 0.734 displacement almost uniformly (block 3 share 0.313,
      # ||dW3||/||W3|| = 1.01%), so it never challenged the donor layer.
      if ONLY3:
          with torch.no_grad():
              for n,p in tnamed:
                  if p.grad is not None and not n.startswith(pre): p.grad.zero_()
      g=torch.cat([(p.grad.reshape(-1) if p.grad is not None else
                    torch.zeros(p.numel())) for _,p in tnamed]); gn=float(g.norm())
      with torch.no_grad():
          for _,p in tnamed:
              if p.grad is not None: p.data.add_(p.grad*(0.02/max(gn,1e-30)))
      nasc+=1
  Lu,au=probe(tm,EV)
  pre=f"blocks.{LAYER}."
  SNAP["U"]=dict(W={n:p.data.clone() for n,p in tnamed if n.startswith(pre)},
                 m={n:torch.zeros_like(p) for n,p in tnamed if n.startswith(pre)},
                 v={n:torch.ones_like(p) for n,p in tnamed if n.startswith(pre)})
  # WHERE DID THE ASCENT ACT? If block 3 barely moved, the null transfer result
  # is about where unlearning acted, not about the object's robustness.
  _d3=0.0; _dall=0.0; _per={}
  with torch.no_grad():
      for n,p in tnamed:
          dd=float(((p.data-TSTATE[120][n])**2).sum()); _dall+=dd
          blk=n.split(".")[1] if n.startswith("blocks.") else n.split(".")[0]
          _per[blk]=_per.get(blk,0.0)+dd
          if n.startswith(pre): _d3+=dd
  _d3=math.sqrt(_d3); _dall=math.sqrt(_dall)
  _share={k:math.sqrt(v)/max(_dall,1e-30) for k,v in _per.items()}
  _n3=math.sqrt(sum(float((TSTATE[120][n]**2).sum()) for n,_ in tnamed if n.startswith(pre)))
  RES["teacher_U"]=dict(L=Lu,acc=au,nasc=nasc,dW3=_d3,dAll=_dall,
                        frac=_d3/max(_dall,1e-30),relW3=_d3/max(_n3,1e-30),
                        share={k:round(v,4) for k,v in sorted(_share.items())}); flush()
  print(f"  ascent displacement: ||dtheta|| {_dall:.3f}  ||dW3|| {_d3:.3f}  "
        f"share {_d3/max(_dall,1e-30):.3f}  ||dW3||/||W3|| {_d3/max(_n3,1e-30):.4f}",flush=True)
  print(f"  per-block share of ||dtheta||: "
        +"  ".join(f"{k}:{v:.3f}" for k,v in sorted(_share.items()) if v>0.01),flush=True)
  _nb=sum(1 for k in _share if k.isdigit())
  print(f"  uniform expectation for one of {_nb} blocks: {1/math.sqrt(max(_nb,1)):.3f}",
        flush=True)
  print(f"  teacher UNLEARNED from 120: {nasc} steps, L {Lu:.4f}  O "
        f"{'/'.join(f'{au[b]:.3f}' for b in BU)}",flush=True)
TSTATE.clear(); del G,tm,m,v; gc.collect()

# ---------------- student ----------------
def student_run(label,donor,what):
    G=newmodel(SSEED); sm=G["model"]; gbs=G["get_batch"]; lrs=G["LR"]*LRM
    snamed=[(n,p) for n,p in sm.named_parameters() if p.requires_grad]
    EVs=[gbs() for _ in range(6)]
    torch.manual_seed(SSEED)
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
        _ag=_n=0
        for _k in S["W"]:
            if _k in S["m"]:
                _a=torch.sign(S["W"][_k]-0*S["W"][_k])  # placeholder, replaced below
        # agreement of sgn(W_T - W_S) with sgn(-m_T), computed on matched keys
        with torch.no_grad():
            for n_,p_ in snamed:
                if not n_.startswith(f"blocks.{LAYER}."): continue
                if n_ not in S["W"] or n_ not in S["m"]: continue
                _dw=S["W"][n_]-p_.data
                _ag+=int((torch.sign(_dw)==torch.sign(-S["m"][n_])).sum()); _n+=_dw.numel()
        if _n: print(f"    agree(sgn(dW), sgn(-m)) = {_ag/_n:.4f}  (chance 0.5)",flush=True)
        crit,frac=MASKSPEC
        kept=tot=0
        with torch.no_grad():
            for n,p in snamed:
                for bi in range(NB):
                    tgt=f"blocks.{bi}."
                    if not n.startswith(tgt): continue
                    src=pre+n[len(tgt):]
                    if src not in S["W"]: continue
                    if p.data.shape!=S["W"][src].shape: continue
                    dW=S["W"][src]-p.data          # teacher CONTRIBUTION
                    if crit in ("lowrank","resid","sign","signN","gsign","msign","randN","rowscale","colscale","spectrum"):
                        if dW.dim()!=2: continue
                        if crit=="sign":
                            p.data.add_(torch.sign(dW)*dW.abs().mean())
                        elif crit=="signN":
                            # NORM-MATCHED: ||scale*sign(dW)|| == ||dW|| exactly
                            sg=torch.sign(dW)
                            sc=dW.norm()/max(float(sg.norm()),1e-30)
                            p.data.add_(sg*sc)
                        elif crit=="gsign":
                            gg=GSIGN.get(src)
                            if gg is None: continue
                            sg=torch.sign(gg)
                            p.data.add_(sg*(dW.norm()/max(float(sg.norm()),1e-30)))
                        elif crit=="randN":
                            # PREREGISTERED NULL: random orientation, norm-matched
                            # to ||dW|| exactly. Same dose, no directional content.
                            g_=torch.Generator().manual_seed(RSEED+hash(n)%10000)
                            sg=torch.where(torch.rand(dW.shape,generator=g_)<0.5,
                                           -torch.ones_like(dW),torch.ones_like(dW))
                            p.data.add_(sg*(dW.norm()/max(float(sg.norm()),1e-30)))
                        elif crit=="msign":
                            mm=S["m"].get(src)
                            if mm is None: continue
                            sg=torch.sign(-mm)      # descent orientation
                            p.data.add_(sg*(dW.norm()/max(float(sg.norm()),1e-30)))
                        elif crit=="rowscale":
                            # keep only per-row scale: reproduce each row's norm
                            # along the student's own row direction
                            sr=dW.norm(dim=1,keepdim=True)
                            base=p.data/(p.data.norm(dim=1,keepdim=True)+1e-12)
                            p.data.add_(base*sr)
                        elif crit=="colscale":
                            sc=dW.norm(dim=0,keepdim=True)
                            base=p.data/(p.data.norm(dim=0,keepdim=True)+1e-12)
                            p.data.add_(base*sc)
                        else:
                            r=max(1,int(frac))
                            U,S_,Vt=torch.linalg.svd(dW,full_matrices=False)
                            LR=(U[:,:r]*S_[:r])@Vt[:r]
                            p.data.add_(dW-LR if crit=="resid" else LR)
                        kept+=dW.numel(); tot+=dW.numel(); continue
                    if frac>=1.0: M=torch.ones_like(dW)
                    elif crit=="mag":
                        k=max(1,int(dW.numel()*frac))
                        idx=torch.topk(dW.abs().reshape(-1),k).indices
                        M=torch.zeros_like(dW).reshape(-1); M[idx]=1.0; M=M.view_as(dW)
                    elif crit=="rand":
                        k=max(1,int(dW.numel()*frac))
                        g_=torch.Generator().manual_seed(hash(n)%(2**31))
                        idx=torch.randperm(dW.numel(),generator=g_)[:k]
                        M=torch.zeros_like(dW).reshape(-1); M[idx]=1.0; M=M.view_as(dW)
                    elif crit=="row" and dW.dim()==2:
                        k=max(1,int(dW.shape[0]*frac))
                        idx=torch.topk(dW.pow(2).sum(1),k).indices
                        M=torch.zeros_like(dW); M[idx]=1.0
                    else:
                        M=torch.ones_like(dW)
                    kept+=int(M.sum()); tot+=dW.numel()
                    p.data.add_(M*dW)
        print(f"    mask {crit} frac={frac} kept {kept}/{tot} = {kept/max(tot,1):.4f}",
              flush=True)
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

ARM=sys.argv[1] if len(sys.argv)>1 else "ctrl"
SPEC={"ctrl":(None,"")}
for _d in DONORS: SPEC[f"W{_d}"]=(_d,"W")
SPEC["WU"]=("U","W"); SPEC["WU3"]=("U","W")
for _d in (40,60,80,100,120): SPEC[f"W{_d}"]=(_d,"W"); SPEC["S120"]=(120,"S"); SPEC["WS120"]=(120,"WS")
d,what=SPEC[ARM]
student_run(ARM,d,what)
print(f"\n  {ARM} done ({time.time()-t0:.0f}s)")
flush()
