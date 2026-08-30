"""CONSTRUCTION-NULL ARM: IS THE STRUCTURE FORCED BY QKV -> ATTN?

Every observable so far was measured on one architecture, so nothing separates
"learned relation" from "inherited from the pathway". This adds construction
interventions that keep the optimiser, token stream, step count, LR and
parameter budget fixed and break only the QKV->ATTN coupling.

  base       normal
  frozenK    W_K held at initialisation (no gradient). The key side cannot
             learn, so attention routing cannot be shaped by the data through
             K, while Q, V, O and everything downstream train normally.
  fixedattn  attention weights replaced by a FROZEN random causal pattern.
             Q and K still compute and still receive gradient through the
             residual path but no longer determine routing at all.

Measured in every arm:
    O1,O2,O3 and the formation times tau_n  -- WHEN the object forms
    J(60,160) support persistence           -- WHAT object forms
    reliability of J's underlying A         -- the gate, re-run per arm
    F_diag, kappa_3                         -- local geometry alongside

Three outcomes, per the plan:
  J_null ~ random   -> the pathway is a necessary generator
  J_null ~ J_base   -> QKV->ATTN is not responsible
  J differs but tau matches, or vice versa -> construction changes the
        REPRESENTATION without changing the relation, or the timing without
        the object. That third case is the interesting one.

Parameter count is held equal in frozenK (weights exist, gradient masked) so
capacity is not confounded with coupling.
"""
import io, contextlib, subprocess, sys, json, time
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F

OUT="res_construct.json"; RES={}
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
SC_OLD="        sc=Q@K.transpose(-2,-1)/self.sc"
SC_NEW=("        sc=Q@K.transpose(-2,-1)/self.sc\n"
        "        if getattr(self,'_fixed',False):\n"
        "            if not hasattr(self,'_fx') or self._fx.shape[-1]!=S:\n"
        "                _g=torch.Generator().manual_seed(self._fxseed)\n"
        "                self._fx=torch.randn(1,self.nh,S,S,generator=_g)\n"
        "            sc=sc*0.0+self._fx")
assert SRC.count(SC_OLD)==1, "attn score anchor"

def build(fixed=False):
    src=SRC.replace(SC_OLD,SC_NEW,1) if fixed else SRC
    torch.manual_seed(1234); np.random.seed(1234)
    G={}; b=io.StringIO()
    with contextlib.redirect_stdout(b): exec(src,G)
    if fixed:
        for li,blk in enumerate(G["model"].blocks):
            blk.attn._fixed=True; blk.attn._fxseed=900+li
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
BU=[b for b in (1,2,3) if len(ALL.get(b,[]))>=16]
FIT3=ALL[3][:13]; HELD={b:ALL[b][len(ALL[b])//2:][:60] for b in BU}
def tens(idx):
    return (torch.tensor([[BASE[i-W+j] for j in range(W)] for i in idx]),
            torch.tensor([BASE[i] for i in idx]))
NU=24; PMAG=0.5; CKS=[20,40,60,80,100,120,160]

def run(arm,seed):
    t0=time.time(); G=build(fixed=(arm=="fixedattn")); model=G["model"]; gb=G["get_batch"]; LR=G["LR"]
    torch.manual_seed(seed)
    at=model.blocks[0].attn
    if arm=="frozenK":
        for blk in model.blocks: blk.attn.WK.weight.requires_grad_(False)
    if arm=="XXX_disabled":
        for li,blk in enumerate(model.blocks):
            a=blk.attn
            gg=torch.Generator().manual_seed(900+li)
            def mk(a,gg):
                Afix={}
                def f(h):
                    B,S,_=h.shape; nh,dh=a.nh,a.dh; D=nh*dh
                    if S not in Afix:
                        sc=torch.randn(1,nh,S,S,generator=gg)
                        m=torch.triu(torch.ones(S,S),diagonal=1).bool()
                        Afix[S]=F.softmax(sc.masked_fill(m,float("-inf")),-1)
                    v=a.WV(h).view(B,S,nh,dh).transpose(1,2)
                    q=a.WQ(h); k=a.WK(h)          # computed, gradient flows, unused
                    z=(Afix[S].expand(B,-1,-1,-1)@v).transpose(1,2).reshape(B,S,D)
                    return a.ln(h+a.op(z)+0.0*(q+k).mean())
                return f
            blk.attn.forward=mk(a,gg)
    ps=[p for p in model.parameters() if p.requires_grad]
    ff=model.blocks[0].ff; Wg=ff.g.weight
    bx1,_=gb(); bx2,_=gb(); Xf,Yf=tens(FIT3)
    @torch.no_grad()
    def resp(x):
        h=model.te(x)+model.pe(torch.arange(x.shape[1]))
        h=model.blocks[0].attn(h)
        return ff.n(h+ff.o(F.silu(ff.g(h))*ff.v(h)))
    @torch.no_grad()
    def A(x):
        base=resp(x).abs().mean(dim=(0,1))
        M=np.zeros((NU,base.shape[0])); gg=torch.Generator().manual_seed(5150)
        for i in range(NU):
            row=Wg[i].clone()
            d=torch.randn(row.shape,generator=gg); d=d/d.norm()*PMAG*row.norm()
            Wg[i].add_(d); M[i]=(resp(x).abs().mean(dim=(0,1))-base).numpy()
            Wg[i].copy_(row)
        return M
    @torch.no_grad()
    def caps():
        model.eval(); o={}
        for b in BU:
            X,Y=tens(HELD[b]); o[b]=float((model(X)[0][:,-1,:].argmax(-1)==Y).float().mean())
        model.train(); return o
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
    st=0; store={}
    for T in CKS:
        while st<T:
            x,y=gb(); _,l=model(x,y); opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],1.0)
            opt.step(); st+=1
        store[T]=(A(bx1),A(bx2)); c=caps()
        RES[f"cap_{arm}_{seed}_{T}"]=c; flush()
    Fd=torch.zeros(sum(p.numel() for p in ps))
    for _ in range(8):
        x,y=gb(); model.zero_grad(); model(x,y)[1].backward()
        Fd+=torch.cat([(p.grad.reshape(-1) if p.grad is not None
                        else torch.zeros(p.numel())) for p in ps])**2
    Fd/=8
    model.zero_grad(); F.cross_entropy(model(Xf)[0][:,-1,:],Yf).backward()
    v=torch.cat([(p.grad.reshape(-1) if p.grad is not None else
                  torch.zeros(p.numel())) for p in ps]).clone()
    v=v/max(float(v.norm()),1e-30)
    x,y=gb(); model.zero_grad(); l=model(x,y)[1]
    gr=torch.autograd.grad(l,ps,create_graph=True)
    gf=torch.cat([q.reshape(-1) for q in gr])
    hh=torch.autograd.grad((gf*v).sum(),ps)
    h=torch.cat([q.reshape(-1) for q in hh]).detach()
    RES[f"geo_{arm}_{seed}"]=dict(F=float(Fd.mean()),kappa=float(h@v)); flush()
    n_t=store[160][0].shape[1]
    def supp(M,q=0.25):
        X=np.abs(M); kk=max(1,int(n_t*q))
        return [set(np.argsort(-X[i])[:kk]) for i in range(NU)]
    Jf=lambda M,N: float(np.mean([len(a&b)/max(len(a|b),1)
                                  for a,b in zip(supp(M),supp(N))]))
    c=lambda a,b: float(np.corrcoef(a.reshape(-1),b.reshape(-1))[0,1])
    RES[f"A160_{arm}_{seed}"]=store[160][0].tolist()
    RES[f"J_{arm}_{seed}"]=dict(J60_160=Jf(store[60][0],store[160][0]),
        rel160=c(store[160][0],store[160][1]),rel60=c(store[60][0],store[60][1]))
    flush()
    print(f"  {arm} s{seed}: O3 {caps()[3]:.3f}  J(60,160) "
          f"{RES[f'J_{arm}_{seed}']['J60_160']:.3f}  ({time.time()-t0:.0f}s)",flush=True)
    del G,model; import gc; gc.collect()

for arm in ("base","frozenK","fixedattn"):
    for s in (17,18): run(arm,s)
print(f"\n  {'arm':>11}{'O1':>7}{'O2':>7}{'O3':>7}{'J(60,160)':>12}"
      f"{'rel(A)':>9}{'F_diag':>11}{'kappa':>9}")
for arm in ("base","frozenK","fixedattn"):
    cs=[RES[f"cap_{arm}_{s}_160"] for s in (17,18)]
    J=[RES[f"J_{arm}_{s}"]["J60_160"] for s in (17,18)]
    r=[RES[f"J_{arm}_{s}"]["rel160"] for s in (17,18)]
    g=[RES[f"geo_{arm}_{s}"] for s in (17,18)]
    print(f"  {arm:>11}"+"".join(f"{np.mean([c[str(b)] if str(b) in c else c[b] for c in cs]):>7.3f}"
          for b in BU)+f"{np.mean(J):>12.3f}{np.mean(r):>9.3f}"
          f"{np.mean([x['F'] for x in g]):>11.2e}{np.mean([x['kappa'] for x in g]):>+9.4f}")
NUt=NU
A160=lambda a,s_: np.array(RES[f"A160_{a}_{s_}"])
nt=A160("base",17).shape[1]
def sp(M,q=0.25):
    X=np.abs(M); kk=max(1,int(nt*q))
    return [set(np.argsort(-X[i])[:kk]) for i in range(NUt)]
JJ=lambda M,N: float(np.mean([len(a&b)/max(len(a|b),1) for a,b in zip(sp(M),sp(N))]))
print(f"\n  CROSS-ARM J at step 160 (same init, so unit indices correspond)\n")
arms=[a for a in ("base","frozenK","fixedattn") if f"A160_{a}_17" in RES]
print(f"  {'':>11}"+"".join(f"{a:>11}" for a in arms))
for a in arms:
    print(f"  {a:>11}"+"".join(f"{np.mean([JJ(A160(a,s_),A160(b_,s_)) for s_ in (17,18)]):>11.3f}"
          for b_ in arms))
print(f"\n  within-arm across seeds:")
for a in arms:
    print(f"    {a:>11}: J(s17,s18) {JJ(A160(a,17),A160(a,18)):.3f}")
print(f"\n  random-support null 0.144; row-permuted 0.180")
flush()
