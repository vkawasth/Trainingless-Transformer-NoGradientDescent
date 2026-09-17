"""GRADIENT INTERFERENCE BETWEEN CE AND THE BOUNDARY TERM, AND WHETHER
THE TRAJECTORY HAS BASINS TO REWIND TO.

rho = cos(grad CE, grad boundary). rho >= 0 means the shared feature space
accommodates tail suppression without fighting target prediction; rho << 0
means structural interference and the two objectives are competing for the
same directions.

Also recorded, because a retroactive rewind needs somewhere to rewind TO:
    val(t)              is it monotone? if so the bottom is always the end
    KL, margin, prec    do any of them turn over, i.e. is there an interval
                        where the model was better by some measure than it is
                        later?
A rewind-and-branch search is only defined if some tracked quantity is
non-monotone. If everything falls monotonically the trajectory has no basins
and "go back to [120,140]" has no referent.
"""
import io, contextlib, math, json, collections
import numpy as np, torch, torch.nn.functional as F
R=open("compiler_geometri_patched_86.py").read()
SRC=R[:R.find("# \u2500\u2500 PHASE 3")]
for o,n in [("for mf_r in range(1, 16):","for mf_r in range(1, 3):"),
            ("    if pc == N_STU-1:","    if False:"),("ETA_MF=0.01","ETA_MF=0.002"),
            ("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
             "    if False:")]:
    assert SRC.count(o)==1,o; SRC=SRC.replace(o,n,1)
EO="evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)"
EN=("_v0=np.random.RandomState(7).randn(L_sym.shape[0])\n"
    "evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000,v0=_v0)\n"
    "evecs=evecs*np.sign(evecs[np.argmax(np.abs(evecs),axis=0),np.arange(evecs.shape[1])])")
assert SRC.count(EO)==1; SRC=SRC.replace(EO,EN,1)
torch.manual_seed(1234); np.random.seed(1234)
G={}; _b=io.StringIO()
with contextlib.redirect_stdout(_b): exec(SRC,G)
model=G["model"]; gb=G["get_batch"]; LR=G["LR"]*5; V=G["VOCAB"]
named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
ids=json.load(open("/tmp/train_ids.json"))
succ=collections.defaultdict(set)
for a,b in zip(ids,ids[1:]): succ[a].add(b)
cnt=collections.Counter(ids)
CTX=sorted([x for x in succ if cnt[x]>=8])
CTXT=torch.tensor(CTX,dtype=torch.long)
SUPM=torch.zeros(V,V,dtype=torch.bool)
for x,S in succ.items(): SUPM[x,torch.tensor(sorted(S),dtype=torch.long)]=True
MASK=SUPM[CTXT]
OBS=torch.zeros(V,V,dtype=torch.bool)
_cv=torch.tensor([max(cnt.get(i,0),1) for i in CTX],dtype=torch.float32)
LOGPX=(_cv/_cv.sum()).log()
import sys
# fork at the rho crossing: raise the margin weight and/or the target gap
# once the objectives stop competing (rho ~ 0 from about step 140)
LAM2 =float(sys.argv[1]) if len(sys.argv)>1 else 0.5
GAM2 =float(sys.argv[2]) if len(sys.argv)>2 else 2.0
FORK =int(sys.argv[3])   if len(sys.argv)>3 else 140
BDMODE=sys.argv[4] if len(sys.argv)>4 else 'max'
DUAL=float(sys.argv[5]) if len(sys.argv)>5 else 0.0
RELEASE=float(sys.argv[6]) if len(sys.argv)>6 else 0.0
RELTHR =float(sys.argv[7]) if len(sys.argv)>7 else 2.0
MARGIN,LAMR=2.0,0.5
EV=[gb() for _ in range(4)]
def vl():
    with torch.no_grad(): return sum(float(model(a,b)[1]) for a,b in EV)/len(EV)
def strands():
    with torch.no_grad():
        lg,_=model(CTXT.unsqueeze(1)); lg=lg[:,-1,:].double()
        p=torch.softmax(lg,-1)
    b=MASK.sum(1).double(); eps=1.0/(2*V)
    above=p>eps
    prec=float(((above&MASK).sum(1).double()/above.sum(1).clamp_min(1).double()).mean())
    lp=p.clamp_min(1e-30).log()
    kl=float((-b.log()-(lp*MASK.double()).sum(1)/b).mean())
    zn=torch.where(~MASK,lg,torch.full_like(lg,-1e30))
    mv=(lg*MASK.double()).sum(1)/b
    return prec,kl,float((mv-zn.max(1).values).mean())
from bregman_dualflat import PythagoreanStop, TemporalComplex
STOP=PythagoreanStop(patience=3)
TC=TemporalComplex(n_landmarks=5,proj_dim=64)
opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
print(f"  boundary={BDMODE}  fork at {FORK}: lambda {LAMR}->{LAM2}, gamma {MARGIN}->{GAM2}")
print(f"  {'step':>5}{'val':>9}{'KL':>9}{'prec':>8}{'margin':>9}{'rho':>8}{'cover':>8}{'rev':>8}{'shape':>8}{'leak':>8}{'b1':>5}  stop")
for t in range(1,701):
    x,y=gb(); logits,_=model(x,y)
    lg=logits.reshape(-1,V); tgt=y.reshape(-1); ctx=x.reshape(-1)
    OBS[ctx,tgt]=True; M=OBS[ctx]
    ce=F.cross_entropy(lg,tgt)
    minv=torch.where(M,lg,torch.full_like(lg,float("inf"))).min(-1).values
    maxi=torch.where(~M,lg,torch.full_like(lg,float("-inf"))).max(-1).values
    ok=torch.isfinite(minv)&torch.isfinite(maxi)
    _g = GAM2 if t>=FORK else MARGIN
    _l = LAM2 if t>=FORK else LAMR
    if RELEASE>0 and t>FORK:
        # RELEASE the boundary term once the margin is comfortably satisfied.
        # Measured: precision and reverse recovery keep improving to step 680
        # (0.911, 0.971) while KL bottoms at ~300 (0.906) and then worsens to
        # 1.001. The constraint has done its work by then; continuing to
        # tighten it costs mass allocation inside the support, which is what
        # KL measures. Scale lambda down as the achieved margin exceeds the
        # target, so the term fades rather than fighting.
        _ach=float((minv[ok]-torch.logsumexp(
            torch.where(M,torch.full_like(lg,-1e30),lg),dim=-1)[ok]).mean()) \
            if ok.any() else 0.0
        _l=_l/(1.0+RELEASE*max(0.0,_ach-RELTHR))
    if BDMODE=="lse":
        # LogSumExp over the WHOLE invalid tail rather than its max. The
        # aggregate gradient does not vanish as individual tail probabilities
        # shrink -- which is exactly the CE failure mode, dL/dz_k = p(k|x) ~
        # 1e-4. The max-margin form constrains one token per context and
        # leaves ~950 others just under threshold, which is why precision
        # plateaued at 0.29 while the head was fixed.
        lse=torch.logsumexp(torch.where(M,torch.full_like(lg,-1e30),lg),dim=-1)
        bd=F.softplus(_g-(minv[ok]-lse[ok])).mean() if ok.any() else lg.sum()*0
    else:
        bd=F.relu(_g-(minv[ok]-maxi[ok])).mean() if ok.any() else lg.sum()*0
    rho=float("nan")
    if t%20==0:
        gc=torch.autograd.grad(ce,[p for _,p in named],retain_graph=True,
                               allow_unused=True)
        gb_=torch.autograd.grad(bd,[p for _,p in named],retain_graph=True,
                                allow_unused=True)
        a_=torch.cat([g.reshape(-1) for g in gc if g is not None]).double()
        b_=torch.cat([g.reshape(-1) for g in gb_ if g is not None]).double()
        rho=float((a_@b_)/(a_.norm()*b_.norm()+1e-30))
    loss=ce+_l*bd
    if DUAL>0 and t%10==0:
        # GLUING: forward and reverse are two normalisations of ONE joint
        #     J(x,y) = p_theta(y|x) P(x)
        # rows give p(y|x) -- trained; columns give q(x|y) -- never trained.
        # The contravariant test showed columns recover only 0.542 of the true
        # predecessor sets, which is the failure to glue. Apply the same LSE
        # boundary to the columns: for each y, the true predecessors must
        # outrank the aggregate of all non-predecessors under q(.|y).
        lgc,_=model(CTXT.unsqueeze(1))
        Jl=torch.log_softmax(lgc[:,-1,:],dim=-1)+LOGPX.unsqueeze(1)   # (C,V)
        Q=torch.log_softmax(Jl,dim=0)                                  # columns
        PM=MASK                                                        # (C,V) x->y
        minp=torch.where(PM,Q,torch.full_like(Q,float("inf"))).min(0).values
        lsen=torch.logsumexp(torch.where(PM,torch.full_like(Q,-1e30),Q),dim=0)
        okc=torch.isfinite(minp)&torch.isfinite(lsen)&(PM.sum(0)>0)
        if okc.any():
            dual=F.softplus(_g-(minp[okc]-lsen[okc])).mean()
            loss=loss+DUAL*dual
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if t%20==0:
        # how good is the self-supervised support estimate? precision is
        # plateauing at 0.29 and the proxy is a SUBSET of the truth, so the
        # constraint being enforced is partly wrong. Measure the gap.
        _cov=float((OBS[CTXT]&MASK).sum())/max(float(MASK.sum()),1)
        _fp=float((OBS[CTXT]&~MASK).sum())/max(float(OBS[CTXT].sum()),1)
        # PYTHAGOREAN SPLIT.  p* is supported on S, so projecting p_theta
        # (not p*) onto the support face gives p_S = p_theta|_S / Z_S and
        #     KL(p*||p_theta) = KL(p*||p_S) + (-log Z_S)
        # exactly: the first term is in-support SHAPE error, the second is
        # mass LEAKED outside the support. Verified against the direct KL.
        with torch.no_grad():
            _lgp,_=model(CTXT.unsqueeze(1))
            _pp=torch.softmax(_lgp[:,-1,:].double(),-1)
        _b=MASK.sum(1).double()
        _ZS=(_pp*MASK.double()).sum(1).clamp_min(1e-30)
        _pS=(_pp*MASK.double())/_ZS.unsqueeze(1)
        _shape=float((-_b.log()-(_pS.clamp_min(1e-30).log()*MASK.double()).sum(1)/_b).mean())
        _leak=float((-_ZS.log()).mean())
        _sp={"kl":0.0,"shape":_shape,"leak":_leak}
        _fire,_why=STOP.update(_sp)
        TC.push(torch.cat([p.data.reshape(-1) for _,p in named]))
        _b1,_contr=TC.b1_max()
        pr,kl,mg=strands()
        with torch.no_grad():
            _lg,_=model(CTXT.unsqueeze(1))
            _J=torch.log_softmax(_lg[:,-1,:].double(),-1)+LOGPX.double().unsqueeze(1)
            _hit=0; _tot=0
            for _j in range(V):
                _tp=int(MASK[:,_j].sum())
                if _tp==0: continue
                _top=torch.topk(_J[:,_j],_tp).indices
                _hit+=int(MASK[_top,_j].sum()); _tot+=_tp
            _rev=_hit/max(_tot,1)
        print(f"  {t:>5}{vl():>9.4f}{kl:>9.4f}{pr:>8.4f}{mg:>+9.4f}{rho:>+8.3f}"
              f"{_cov:>8.3f}{_rev:>8.3f}"
              f"{_shape:>8.3f}{_leak:>8.3f}{_b1:>5d}  {_why}",
              flush=True)
