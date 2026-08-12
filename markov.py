"""DOES THE DEFECT COORDINATE CLOSE THE DYNAMICS?

The defect's self-prediction was an overlap artefact (spike at k=1, negative for
k>=2). This asks a structurally different question, immune to that artefact:
does adding d_t to the state SCREEN OFF history when predicting x_{t+1}?

  M0  x_t+1 ~ x_t                      instantaneous state
  M1  x_t+1 ~ x_t, x_t-1               one lag of history
  M2  x_t+1 ~ x_t, x_t-1, x_t-2        two lags
  D0  x_t+1 ~ x_t, d_t                 state augmented by the defect
  D1  x_t+1 ~ x_t, d_t, x_t-1          does history STILL add after d?

  M1 >> M0  and  D1 ~ D0   ->  the trajectory is non-Markovian in x alone and
                               becomes Markovian after adding d: the state space
                               was wrong, not the dynamics
  D1 >> D0                 ->  d does not close it; genuine memory remains
  M1 ~ M0                  ->  already Markovian; nothing to close

Also a memory-order sweep, out of sample with ridge, since in-sample fit always
improves with lags. All arms share the same held-out split and regulariser so
the comparison is like-for-like.
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
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
def flat(): return torch.cat([p.data.flatten() for p in params]).clone()
K=12
gen=torch.Generator().manual_seed(5)
R=torch.linalg.qr(torch.randn(P,K,generator=gen))[0]
U=[]; MV=[]; prev=None; step=0
for ck in range(100,540):
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
MV=np.array(MV[1:]); MV=(MV-MV.mean(0))/(MV.std(0)+1e-12)
d=A[1:-1]-(A[:-2]+A[2:])/2          # defect, aligned to index t+1 of A
n=len(A); rr=np.random.default_rng(0)
LAM=3.0
def r2(cols,Y,lo,hi):
    X=np.hstack([c[lo:hi] for c in cols]); Yy=Y[lo:hi]
    X=np.hstack([X,np.ones((len(X),1))])
    m=len(X); tr=slice(0,int(m*.7)); te=slice(int(m*.7),m)
    M=X[tr]; W=np.linalg.solve(M.T@M+LAM*np.eye(M.shape[1]),M.T@Yy[tr])
    return 1-((Yy[te]-X[te]@W)**2).sum()/max(((Yy[te]-Yy[te].mean(0))**2).sum(),1e-30)
# align: predict A[t+1] from A[t], A[t-1], A[t-2], d aligned so d_t uses A[t-1..t+1]
lo,hi=3,n-2
Y=A[1:]                                  # A[t+1] indexed by t
cols_x  =[A]                             # A[t]
cols_x1 =[A,np.roll(A,1,axis=0)]
cols_x2 =[A,np.roll(A,1,axis=0),np.roll(A,2,axis=0)]
dpad=np.vstack([np.zeros((1,K)),d,np.zeros((1,K))])   # d aligned to t
cols_d  =[A,dpad]
cols_d1 =[A,dpad,np.roll(A,1,axis=0)]
print(f"  n={n} steps, dim {K}, ridge lam={LAM}, held-out 30%\n")
print(f"  {'model':<34}{'R2':>9}{'vs prev':>10}")
r_M0=r2(cols_x,Y,lo,hi);            print(f"  {'M0  x_t+1 ~ x_t':<34}{r_M0:>9.4f}")
r_M1=r2(cols_x1,Y,lo,hi);           print(f"  {'M1  + x_t-1':<34}{r_M1:>9.4f}{r_M1-r_M0:>+10.4f}")
r_M2=r2(cols_x2,Y,lo,hi);           print(f"  {'M2  + x_t-2':<34}{r_M2:>9.4f}{r_M2-r_M1:>+10.4f}")
r_D0=r2(cols_d,Y,lo,hi);            print(f"  {'D0  x_t+1 ~ x_t, d_t':<34}{r_D0:>9.4f}{r_D0-r_M0:>+10.4f}")
r_D1=r2(cols_d1,Y,lo,hi);           print(f"  {'D1  + x_t-1':<34}{r_D1:>9.4f}{r_D1-r_D0:>+10.4f}")
print(f"\n  history gain without d : {r_M1-r_M0:+.4f}")
print(f"  history gain WITH d    : {r_D1-r_D0:+.4f}")
print(f"  -> d screens off history if the second is much smaller than the first")
print(f"\n  memory-order sweep, out of sample")
print(f"  {'K lags':>7}{'R2':>10}")
for kk in range(0,7):
    cols=[np.roll(A,j,axis=0) for j in range(kk+1)]
    print(f"  {kk:>7}{r2(cols,Y,max(lo,kk+1),hi):>10.4f}")
print(f"\n  with optimiser state (m,v):")
print(f"    x_t+1 ~ x_t, m, v        {r2([A,MV[:len(A)]],Y,lo,hi):>8.4f}")
print(f"    x_t+1 ~ x_t, m, v, d_t   {r2([A,MV[:len(A)],dpad],Y,lo,hi):>8.4f}")
