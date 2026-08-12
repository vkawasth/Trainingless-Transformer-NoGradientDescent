"""LAG SWEEP AND NESTED CONDITIONING.

Two questions the single k=1 number cannot answer:

 A IS IT DYNAMICAL OR ALGEBRAIC?  d_t is a second difference, so overlapping
   terms can manufacture a k=1 correlation. A genuine short-memory process shows
   a DECAY R2(k); an artefact shows an isolated spike at k=1 and nothing after.
   Swept k=1..8, with three nulls: phase randomisation, block permutation
   (preserves local structure, destroys long-range order), and a shift control.

 B IS IT OPTIMISER STATE?  The incremental contribution
     dR2 = R2(d_t+1 | d_t, m_t, v_t) - R2(d_t+1 | m_t, v_t)
   isolates what the defect adds over the full optimiser state, rather than
   testing whether m alone explains it. Adam's m,v are deterministic functions
   of past gradients, so conditioning on both is the strong form.

Run at beta1=0 where the effect was largest (R2 0.355), so the test has headroom.
"""
import json, subprocess, numpy as np, torch, gc, io, contextlib
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
src=RAW[:CUT].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
src=src.replace("    if pc == N_STU-1:","    if False:",1)
src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                "    if False:",1)
G={}; buf=io.StringIO()
with contextlib.redirect_stdout(buf): exec(src,G)
model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]
params=[p for _,p in model.named_parameters()]; P=sum(p.numel() for p in params)
for p in params: p.requires_grad_(True)
torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.0,0.95),weight_decay=0.1)
def flat(): return torch.cat([p.data.flatten() for p in params]).clone()
K=12
gen=torch.Generator().manual_seed(5)
R=torch.linalg.qr(torch.randn(P,K,generator=gen))[0]
U=[]; MV=[]; prev=None; step=0
for ck in range(100,520):
    while step<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    th=flat()
    if prev is not None: U.append((R.T@(th-prev)).numpy())
    prev=th.clone()
    mm=torch.cat([opt.state[p]["exp_avg"].flatten() for p in params])
    vv=torch.cat([opt.state[p]["exp_avg_sq"].flatten() for p in params])
    MV.append(np.concatenate([(R.T@mm).numpy(),(R.T@vv).numpy()]))
A=np.array(U); A=(A-A.mean(0))/(A.std(0)+1e-12)
d=A[1:-1]-(A[:-2]+A[2:])/2
MV=np.array(MV[2:]); MV=(MV-MV.mean(0))/(MV.std(0)+1e-12)
n=len(d); rr=np.random.default_rng(0)
def r2(X,Y,lam=1.0):
    Xs=np.hstack([X,np.ones((len(X),1))])
    m=len(Xs); tr=slice(0,int(m*.7)); te=slice(int(m*.7),m)
    M=Xs[tr]; W=np.linalg.solve(M.T@M+lam*np.eye(M.shape[1]),M.T@Y[tr])
    return 1-((Y[te]-Xs[te]@W)**2).sum()/max(((Y[te]-Y[te].mean(0))**2).sum(),1e-30)
print(f"  n={n} defects, dim {K}, beta1=0\n")
print(f"  (A) lag sweep")
print(f"  {'k':>4}{'R2':>9}{'phase null':>12}{'block null':>12}")
for k in range(1,9):
    X=d[:-k]; Y=d[k:]
    sc=r2(X,Y)
    ph=[]
    for _ in range(200):
        F=np.fft.rfft(Y,axis=0); p=np.exp(1j*rr.uniform(0,2*np.pi,F.shape)); p[0]=1
        ph.append(r2(X,np.fft.irfft(np.abs(F)*p,n=len(Y),axis=0)))
    bl=[]
    B=20
    for _ in range(200):
        nb=len(Y)//B
        order=rr.permutation(nb)
        Ys=np.vstack([Y[o*B:(o+1)*B] for o in order])
        bl.append(r2(X[:len(Ys)],Ys))
    print(f"  {k:>4}{sc:>9.3f}{np.percentile(ph,95):>12.3f}{np.percentile(bl,95):>12.3f}"
          f"   {'SIGNAL' if sc>max(np.percentile(ph,95),np.percentile(bl,95)) else ''}")
print(f"\n  (B) incremental contribution over the full optimiser state")
Y=d[1:]; X1=d[:-1]; X2=MV[1:len(d)][:len(Y)]
if len(X2)<len(Y): Y=Y[:len(X2)]; X1=X1[:len(X2)]
a=r2(X2,Y); b=r2(np.hstack([X1,X2]),Y); c=r2(X1,Y)
print(f"    R2(d_t+1 | m,v)        = {a:+.3f}")
print(f"    R2(d_t+1 | d_t)        = {c:+.3f}")
print(f"    R2(d_t+1 | d_t, m, v)  = {b:+.3f}")
print(f"    incremental dR2 of d_t = {b-a:+.3f}")
print(f"\n  dR2 > 0 => dynamical structure beyond optimiser state")
