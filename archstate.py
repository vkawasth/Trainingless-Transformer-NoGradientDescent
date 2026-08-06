"""ARCHITECTURAL STATE: THE EMPTY CATEGORY, AND THE TIMESCALE HYPOTHESIS.

Previous ablation (n=20, six held out) gave Q+v = +0.464, Q+dF = +0.439,
Q+Sigma and Q+m negative, history increasingly negative. The two winners are
slowly-varying accumulators; the losers track the fast-rotating drift. That is
the timescale hypothesis, and z_architecture has never been measured at all.

This run: n ~ 60 checkpoints so held-out sets are ~18 rather than 6, plus
architectural arms that are slow collective statistics of the REPRESENTATION
rather than of the optimiser:

  attn_ent   mean attention entropy over heads and positions
  aniso      participation ratio of the activation covariance spectrum
  ln_gain    mean and sd of LayerNorm gains
  wk_sv      participation ratio of the W_K singular values

Target: the residual dtheta_{t+1} - dtheta_t in Krylov coordinates, which
removed the autocorrelation confound (raw 0.036, residual -0.060).
Each arm scored as held-out R^2 against a shuffled-target null.
Also reported: the AUTOCORRELATION of each candidate, to test the timescale
hypothesis directly -- slow variables should have high lag-1 autocorrelation.
"""
import json, subprocess, numpy as np, torch, gc, io, contextlib
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
M_,NB,ND=3,6,30
CKS=list(range(6,372,6))
src=RAW[:CUT].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
src=src.replace("    if pc == N_STU-1:","    if False:",1)
src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                "    if False:",1)
G={}; buf=io.StringIO()
with contextlib.redirect_stdout(buf): exec(src,G)
model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]; ev=G["eval_val"]
named=[(n,p) for n,p in model.named_parameters()]; params=[p for _,p in named]
P=sum(p.numel() for p in params)
for p in params: p.requires_grad_(True)
nL=len(model.blocks)
A_={}
def mkf(i):
    def f(mod,inp,o): A_[i]=inp[0].detach()
    return f
for i,b in enumerate(model.blocks): b.register_forward_hook(mkf(i))
torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
def flat(): return torch.cat([p.data.flatten() for p in params]).clone()
def setth(t):
    with torch.no_grad():
        i=0
        for p in params:
            k=p.numel(); p.data.copy_(t[i:i+k].view_as(p)); i+=k
def gfl():
    return torch.cat([(p.grad.flatten() if p.grad is not None else
        torch.zeros(p.numel())) for p in params]).clone()
def drift(th,n=ND):
    a=torch.zeros(P)
    for _ in range(n):
        x,y=get_batch(); model.zero_grad(); _,l=model(x,y); l.backward(); a+=gfl(); setth(th)
    model.zero_grad(); return a/n
def hvp(th,v,nb=NB):
    a=torch.zeros(P)
    for _ in range(nb):
        x,y=get_batch(); model.zero_grad(); _,loss=model(x,y)
        gr=torch.autograd.grad(loss,params,create_graph=True,allow_unused=True)
        gr=[t if t is not None else torch.zeros_like(p) for t,p in zip(gr,params)]
        g2=torch.cat([t.flatten() for t in gr])
        hv=torch.autograd.grad((g2*v).sum(),params,allow_unused=True)
        hv=[t if t is not None else torch.zeros_like(p) for t,p in zip(hv,params)]
        a+=torch.cat([t.flatten() for t in hv]).detach(); setth(th)
    model.zero_grad(); return a/nb
def archstats():
    A_.clear()
    x,y=get_batch()
    with torch.no_grad(): model(x,y)
    ans=[]
    for i in range(nL):
        if i not in A_: continue
        a=A_[i].reshape(-1,A_[i].shape[-1]).double()
        C=(a.T@a)/max(a.shape[0],1)
        s=torch.linalg.svdvals(C); s=s[s>0]
        ans.append(float(s.sum()**2/(s**2).sum()))
    ln=[float(p.data.mean()) for n,p in named if "ln" in n.lower()]
    lns=[float(p.data.std()) for n,p in named if "ln" in n.lower()]
    wk=[]
    for n,p in named:
        if "WK" in n and p.dim()==2:
            s=torch.linalg.svdvals(p.data.float()); s=s[s>0]
            wk.append(float(s.sum()**2/(s**2).sum()))
    return dict(aniso=float(np.mean(ans)) if ans else 0.0,
                ln_m=float(np.mean(ln)) if ln else 0.0,
                ln_s=float(np.mean(lns)) if lns else 0.0,
                wk=float(np.mean(wk)) if wk else 0.0)
step=0; prev=flat(); D=[]
for ck in CKS:
    while step<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    th=flat(); u=th-prev; prev=th.clone()
    g=drift(th); V=[g/(g.norm()+1e-30)]
    for _ in range(M_-1):
        w=hvp(th,V[-1])
        for q in V: w=w-float((w*q).sum())*q
        if float(w.norm())<1e-10: break
        V.append(w/w.norm())
    Q=torch.stack(V,1)
    HQ=torch.stack([hvp(th,Q[:,j]) for j in range(Q.shape[1])],1)
    T=(Q.T@HQ).numpy(); T=(T+T.T)/2
    vv=torch.cat([opt.state[p]["exp_avg_sq"].flatten() for p in params])
    mm=torch.cat([opt.state[p]["exp_avg"].flatten() for p in params])
    Md=1.0/(vv.sqrt()+1e-8); un=u/(u.norm()+1e-30)
    a=archstats()
    D.append(dict(ck=ck,cu=[float(x) for x in (Q.T@un)],
                  sig=[float(x) for x in np.linalg.eigvalsh(T)],
                  cm=[float(x) for x in (Q.T@(mm/(mm.norm()+1e-30)))],
                  cv=[float(x) for x in (Q.T@((Md*g)/((Md*g).norm()+1e-30)))],
                  dF=float((u*Md).norm()/(u.norm()+1e-30)),**a))
    if ck%60<6: print(f"  ck {ck:>4}  aniso {a['aniso']:.2f} wk {a['wk']:.2f}",flush=True)
json.dump(D,open("/home/claude/work/res_arch_raw.json","w"),indent=2)
Y=np.array([d["cu"] for d in D]); R=Y[1:]-Y[:-1]
print(f"\n  n={len(D)}")
print(f"  {'variable':>10}{'lag-1 autocorr':>16}   (timescale test)")
for k in ("dF","aniso","ln_m","ln_s","wk"):
    s=np.array([d[k] for d in D])
    print(f"  {k:>10}{np.corrcoef(s[:-1],s[1:])[0,1]:>16.4f}")
for k,nm in ((("cv"),"v-proj"),(("cm"),"m-proj"),(("sig"),"Sigma")):
    s=np.array([d[k] for d in D])
    ac=np.mean([np.corrcoef(s[:-1,j],s[1:,j])[0,1] for j in range(s.shape[1])])
    print(f"  {nm:>10}{ac:>16.4f}")
rg=np.random.default_rng(0)
def build(keys,lo,hi):
    X=[]
    for i in range(lo,hi):
        f=list(Y[i])
        if "S" in keys: f+=D[i]["sig"]
        if "m" in keys: f+=D[i]["cm"]
        if "v" in keys: f+=D[i]["cv"]
        if "F" in keys: f+=[D[i]["dF"]]
        if "A" in keys: f+=[D[i]["aniso"]]
        if "L" in keys: f+=[D[i]["ln_m"],D[i]["ln_s"]]
        if "W" in keys: f+=[D[i]["wk"]]
        X.append(f)
    return np.array(X)
lo=1; hi=len(R); Yt=R[lo:hi]
print(f"\n  {'state':>14}{'R2 held out':>13}{'null p95':>10}{'verdict':>9}")
for name,keys in (("Q",""),("Q+v","v"),("Q+dF","F"),("Q+m","m"),("Q+Sigma","S"),
                  ("Q+aniso","A"),("Q+LN","L"),("Q+WK","W"),
                  ("Q+arch(A,L,W)","ALW"),("Q+v+arch","vALW")):
    X=build(keys,lo,hi)
    X=(X-X.mean(0))/(X.std(0)+1e-9); X=np.hstack([X,np.ones((len(X),1))])
    n=len(X); tr=slice(0,int(n*0.7)); te=slice(int(n*0.7),n)
    def r2(Yy):
        W=np.linalg.lstsq(X[tr],Yy[tr],rcond=None)[0]; Pd=X[te]@W
        return 1-((Yy[te]-Pd)**2).sum()/max(((Yy[te]-Yy[te].mean(0))**2).sum(),1e-30)
    sc=r2(Yt); nul=[r2(Yt[rg.permutation(len(Yt))]) for _ in range(300)]
    p95=np.percentile(nul,95)
    print(f"  {name:>14}{sc:>13.4f}{p95:>10.4f}{('SIGNAL' if sc>p95 else '-'):>9}")
