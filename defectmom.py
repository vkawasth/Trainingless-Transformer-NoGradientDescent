"""IS THE DEFECT DYNAMICS JUST MOMENTUM?

Backward-state defects are self-predictive per component (EMB .208, LN .260,
FF .161, ATTN .259) with a purely diagonal cross matrix. Adam's first moment
with beta1=0.9 gives ~10 steps of memory, so dtheta_t already contains a
weighted average of past gradients -- consecutive updates are correlated BY
CONSTRUCTION and their second differences may inherit that.

Three arms, same trajectory, same construction:
  beta1=0.9   the measured case
  beta1=0.0   no first moment: if the defect dynamics vanish, they were momentum
  SGD         no adaptive state at all

Plus, within the beta1=0.9 arm, the conditional test: does d_t still predict
d_t+1 after regressing out m_t? If the signal survives conditioning on the
optimiser state, it is not momentum.
"""
import json, subprocess, numpy as np, torch, gc, io, contextlib
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
def build():
    src=RAW[:CUT].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
    src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
    src=src.replace("    if pc == N_STU-1:","    if False:",1)
    src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                    "    if False:",1)
    G={}; buf=io.StringIO()
    with contextlib.redirect_stdout(buf): exec(src,G)
    return G
rr=np.random.default_rng(0)
def selfpred(d,extra=None):
    X=d[:-1] if extra is None else np.hstack([d[:-1],extra[:-1]])
    Y=d[1:]
    Xs=(X-X.mean(0))/(X.std(0)+1e-12); Xs=np.hstack([Xs,np.ones((len(Xs),1))])
    m=len(Xs); tr=slice(0,int(m*.7)); te=slice(int(m*.7),m)
    def fit(Yy):
        M=Xs[tr]; W=np.linalg.solve(M.T@M+1.0*np.eye(M.shape[1]),M.T@Yy[tr])
        return 1-((Yy[te]-Xs[te]@W)**2).sum()/max(((Yy[te]-Yy[te].mean(0))**2).sum(),1e-30)
    sc=fit(Y); nl=[]
    for _ in range(200):
        F=np.fft.rfft(Y,axis=0); ph=np.exp(1j*rr.uniform(0,2*np.pi,F.shape)); ph[0]=1
        nl.append(fit(np.fft.irfft(np.abs(F)*ph,n=len(Y),axis=0)))
    return sc,np.percentile(nl,95)
print(f"  {'arm':>10}{'|d|/|dx|':>11}{'R2 self':>10}{'null p95':>10}{'R2 | m':>10}")
for arm in ("b1=0.9","b1=0.0","sgd"):
    G=build(); model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]
    params=[p for _,p in model.named_parameters()]; P=sum(p.numel() for p in params)
    for p in params: p.requires_grad_(True)
    torch.manual_seed(17)
    if arm=="b1=0.9":
        opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
    elif arm=="b1=0.0":
        opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.0,0.95),weight_decay=0.1)
    else:
        opt=torch.optim.SGD(model.parameters(),lr=4.0)
    def flat(): return torch.cat([p.data.flatten() for p in params]).clone()
    K=12
    gen=torch.Generator().manual_seed(5)
    R=torch.linalg.qr(torch.randn(P,K,generator=gen))[0]
    U=[]; M_=[]; prev=None; step=0
    for ck in range(100,440):
        while step<ck:
            x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
        th=flat()
        if prev is not None: U.append((R.T@(th-prev)).numpy())
        prev=th.clone()
        if arm!="sgd":
            mm=torch.cat([opt.state[p]["exp_avg"].flatten() for p in params])
            M_.append((R.T@mm).numpy())
        else: M_.append(np.zeros(K))
    A=np.array(U); A=(A-A.mean(0))/(A.std(0)+1e-12)
    d=A[1:-1]-(A[:-2]+A[2:])/2
    dx=np.diff(A,axis=0)
    Mm=np.array(M_[1:]); Mm=(Mm-Mm.mean(0))/(Mm.std(0)+1e-12)
    sc,nu=selfpred(d)
    sc2,_=selfpred(d,extra=Mm[1:-1])
    print(f"  {arm:>10}{np.linalg.norm(d)/max(np.linalg.norm(dx[:-1]),1e-12):>11.3f}"
          f"{sc:>10.3f}{nu:>10.3f}{sc2:>10.3f}"
          f"   {'SIGNAL' if sc>nu else '-'}",flush=True)
    del G,model,params; gc.collect()
print(f"\n  signal vanishes at b1=0.0  -> the defect dynamics are momentum")
print(f"  signal survives            -> not momentum; a property of the flow")
