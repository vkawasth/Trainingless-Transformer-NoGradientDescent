"""IS THERE A COHERENT LOCAL TIMESCALE, OR IS DECORRELATION SCALE-FREE?

The two-timescale reading -- fast local dynamics finding a path, slow global
rotation setting the sweep -- requires the local scale to be COHERENT.
Measured at 20-step resolution, consecutive signals sit at 74-79 degrees, i.e.
cos 0.20-0.26. That is barely aligned. Distant pairs are at 87 degrees,
cos 0.05.

So there is a gradient of decorrelation, not obviously a separation of scales.
This measures the autocorrelation of the update direction at lags from 1 to 40
steps. Two outcomes:

  cos rises sharply below some lag   -> a coherent local scale exists and the
                                        two-timescale picture holds
  cos ~ flat across lags             -> scale-free decorrelation, and "local
                                        finds the path" has no support

Reported against the null for independent directions, which on 1.18M
dimensions is ~0.
"""
import io, contextlib, math
import numpy as np, torch
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
model=G["model"]; gb=G["get_batch"]; LR=G["LR"]*5
named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
SUB=200000
tot=sum(p.numel() for _,p in named)
idx=torch.randperm(tot,generator=torch.Generator().manual_seed(9))[:SUB]
import sys
B1=float(sys.argv[1]) if len(sys.argv)>1 else 0.9
print(f"  beta1 = {B1}   (0.9 = Adam, 0.0 = preconditioned SGD, no momentum)")
if B1>0:
    print(f"  momentum horizon 1/(1-b1) = {1/(1-B1):.1f} steps")
else:
    print(f"  no optimiser memory in the numerator; any coherence is landscape")
opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(B1,0.95),weight_decay=0.1)
U=[]
for t in range(1,241):
    prev=torch.cat([p.data.reshape(-1) for _,p in named])[idx].clone()
    x,y=gb(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    cur=torch.cat([p.data.reshape(-1) for _,p in named])[idx]
    U.append((cur-prev).double())
print(f"  {len(U)} single-step directions, {SUB} coords\n")
print(f"  autocorrelation of the STEP direction by lag")
print(f"  {'lag':>5}{'cos':>9}{'deg':>8}")
for lag in (1,2,3,5,8,12,20,30,40,60):
    v=[float((U[i]@U[i+lag])/(U[i].norm()*U[i+lag].norm()+1e-30))
       for i in range(0,len(U)-lag,3)]
    c=float(np.mean(v))
    print(f"  {lag:>5}{c:>9.4f}{np.degrees(np.arccos(np.clip(c,-1,1))):>8.1f}")
_cs=[]
for lag in range(1,41):
    v=[float((U[i]@U[i+lag])/(U[i].norm()*U[i+lag].norm()+1e-30))
       for i in range(0,len(U)-lag,3)]
    _cs.append(float(np.mean(v)))
_pos=[c for c in _cs if c>0]
print(f"\n  integrated correlation time tau_c = sum of positive cos = "
      f"{sum(_pos):.2f} steps")
_h=[i+1 for i,c in enumerate(_cs) if c<0.5]
print(f"  half-coherence lag (first cos<0.5): {_h[0] if _h else '>40'} steps")
print(f"\n  now the same for AGGREGATED windows, which is what the 20-step")
print(f"  measurement saw: sum W consecutive steps, then correlate neighbours")
print(f"  {'window':>7}{'cos':>9}{'deg':>8}")
for W in (1,5,10,20,40):
    segs=[sum(U[i:i+W]) for i in range(0,len(U)-W,W)]
    v=[float((segs[i]@segs[i+1])/(segs[i].norm()*segs[i+1].norm()+1e-30))
       for i in range(len(segs)-1)]
    c=float(np.mean(v))
    print(f"  {W:>7}{c:>9.4f}{np.degrees(np.arccos(np.clip(c,-1,1))):>8.1f}")
