"""SPATIAL vs TEMPORAL RANK, SIGNAL vs NOISE.

Two distinct low-dimensionality claims have been conflated:

  SPATIAL   GaLore/SubTrack factor the gradient MATRIX G (d_out x d_in) by SVD
            and keep r directions. Low rank here is a property of one matrix.
  TEMPORAL  PR over a window of flattened update vectors. Measured 2.5 with
            momentum and 15.9 without, i.e. the collapse is the beta1 filter.

They are independent: G can be full rank while the update sequence spans two
directions, and vice versa. This measures both on the same run, and splits each
into signal and noise using two independent batch draws at the same theta:

    sig = (g_A + g_B)/2      common to both draws
    noi = (g_A - g_B)/2      differs

Reported per layer:
    PR_spatial(sig)   is the SIGNAL matrix low rank?
    PR_spatial(noi)   is the NOISE matrix low rank? -- if both are, a spatial
                      projection keeps the noise too
    PR_temporal       over the window, for reference
    r64_sig, r64_noi  energy captured by the signal's top-64 subspace
"""
import io, contextlib, math, sys
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
B1=float(sys.argv[1]) if len(sys.argv)>1 else 0.9
torch.manual_seed(1234); np.random.seed(1234)
G={}; _b=io.StringIO()
with contextlib.redirect_stdout(_b): exec(SRC,G)
model=G["model"]; gb=G["get_batch"]; LR=G["LR"]*5
named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
P_={n:p for n,p in named}
TGT=["blocks.3.attn.WQ.weight","blocks.3.attn.WV.weight",
     "blocks.3.ff.g.weight","te.weight"]
def pr(sv):
    e=sv**2; return float(e.sum()**2/max(float((e**2).sum()),1e-30))
def grad_at():
    model.zero_grad(); x,y=gb(); model(x,y)[1].backward()
    return {n:P_[n].grad.clone() for n in TGT}
opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(B1,0.95),weight_decay=0.1)
sub=torch.randperm(sum(p.numel() for _,p in named),
                   generator=torch.Generator().manual_seed(42))[:40000]
buf=[]
for t in range(1,201):
    prev=torch.cat([p.data.reshape(-1) for _,p in named])[sub].clone()
    x,y=gb(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    cur=torch.cat([p.data.reshape(-1) for _,p in named])[sub]
    buf.append((cur-prev).double())
    if len(buf)>16: buf.pop(0)
M=torch.stack(buf,1)
pt=pr(torch.linalg.svdvals(M).numpy())
A=grad_at(); B=grad_at()
print(f"  beta1={B1}   PR_temporal (window of 16 updates) = {pt:.2f}\n")
print(f"  {'layer':<26}{'shape':>12}{'PR_sig':>8}{'PR_noi':>8}"
      f"{'r64_sig':>9}{'r64_noi':>9}")
for n in TGT:
    sig=((A[n]+B[n])/2).double(); noi=((A[n]-B[n])/2).double()
    Us,Ss,_=torch.linalg.svd(sig,full_matrices=False)
    Sn=torch.linalg.svdvals(noi)
    r=min(64,min(sig.shape))
    cs=float((Us[:,:r].T@sig).pow(2).sum()/max(float(sig.pow(2).sum()),1e-30))
    cn=float((Us[:,:r].T@noi).pow(2).sum()/max(float(noi.pow(2).sum()),1e-30))
    print(f"  {n[:26]:<26}{str(tuple(sig.shape)):>12}{pr(Ss.numpy()):>8.1f}"
          f"{pr(Sn.numpy()):>8.1f}{cs:>9.3f}{cn:>9.3f}")
