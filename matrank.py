"""IS THE GRADIENT LOW-RANK AS A MATRIX?

Everything in this programme flattened parameters into one vector and measured
subspaces of R^P. That destroys the structure: a weight matrix W in R^{m x n} has
an m x n grid, and flattening throws it away. GaLore projects P^T G Q with P in
R^{m x r} and Q in R^{n x r}, costing r(m+n) rather than r*mn -- for our FF block
that is 37,888 floats against my flat frame's 4,718,592, a factor of 125. My
construction failed for a reason GaLore does not share.

But GaLore's PREMISE is that the gradient matrix is low rank, and this programme
measured something that sounds contradictory: a fresh gradient is only 5-17%
captured by a frame built from update HISTORY (gamma = 0.83-0.95). Those are
different objects -- history frame versus the gradient's own SVD -- and the
second has never been measured here.

THE TEST, per weight matrix, no flattening:
    G = dL/dW  in R^{m x n}
    SVD, and report the energy in the top r singular directions for r = 1..32
    against the null for a random m x n matrix of the same shape

  top-8 energy ~ 0.9   GaLore's premise holds here and the rewrite is worth doing
  top-8 energy ~ 0.15  it does not, and the method would not transfer

Also measured, because it is the quantity that actually matters for the
optimiser and is cheap once the SVD is in hand:
  STABILITY   overlap of the top-r left singular subspace between consecutive
              steps. GaLore refreshes its projector every 50-100 steps, which
              only works if the subspace persists. Our update frames rotated to
              near the random ceiling, so this cannot be assumed.
"""
import json, subprocess, numpy as np, torch, io, contextlib, math
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
src=RAW[:CUT].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
src=src.replace("    if pc == N_STU-1:","    if False:",1)
src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:","    if False:",1)
G={}; b=io.StringIO()
with contextlib.redirect_stdout(b): exec(src,G)
model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]
MATS=[(n,p) for n,p in model.named_parameters() if p.dim()==2 and p.requires_grad]
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
RS=(1,2,4,8,16,32)
def gradmats():
    x,y=get_batch(); model.zero_grad(); _,l=model(x,y); l.backward()
    out={n:p.grad.detach().clone() for n,p in MATS}
    model.zero_grad(); return out
print(f"  {len(MATS)} weight matrices, no flattening\n")
step=0; prevU={}; rows=[]
for ck in (30,90,150):
    while step<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    Gm=gradmats()
    print(f"  === step {ck}")
    print(f"  {'matrix':>22}{'shape':>13}"+"".join(f"{'r='+str(r):>8}" for r in RS)
          +f"{'null r=8':>10}{'stab r=8':>10}")
    for n,p in MATS:
        A=Gm[n]
        U,S,Vt=torch.linalg.svd(A,full_matrices=False)
        e=(S**2).numpy(); cum=np.cumsum(e)/e.sum()
        m_,n_=A.shape; mn=min(m_,n_)
        vals=[float(cum[min(r,mn)-1]) for r in RS]
        gen=torch.Generator().manual_seed(3)
        R=torch.randn(m_,n_,generator=gen)
        sr=torch.linalg.svdvals(R).numpy()**2
        null=float(np.cumsum(sr)[min(8,mn)-1]/sr.sum())
        k=min(8,mn); Uk=U[:,:k]
        st=float("nan")
        if n in prevU and prevU[n].shape==Uk.shape:
            sv=torch.linalg.svdvals(prevU[n].T@Uk).numpy()
            st=float((np.clip(sv,0,1)**2).mean())
        prevU[n]=Uk.clone()
        short=n.replace("blocks.","b").replace(".weight","")
        print(f"  {short:>22}{str(tuple(A.shape)):>13}"
              +"".join(f"{v:>8.3f}" for v in vals)+f"{null:>10.3f}{st:>10.3f}")
        rows.append(dict(ck=ck,name=n,shape=list(A.shape),cum=vals,null=null,stab=st))
    print()
json.dump(rows,open("/home/claude/work/res_matrank.json","w"),indent=2)
r8=np.array([r["cum"][3] for r in rows]); nl=np.array([r["null"] for r in rows])
sb=np.array([r["stab"] for r in rows if r["stab"]==r["stab"]])
print(f"  top-8 energy: mean {r8.mean():.3f}   null {nl.mean():.3f}   "
      f"excess {r8.mean()/max(nl.mean(),1e-9):.1f}x")
print(f"  subspace stability between checkpoints (mean cos^2): {sb.mean():.3f}")
print(f"\n  top-8 >> null => the gradient IS low rank as a matrix, and the")
print(f"  per-matrix projector is the right object")
print(f"  stability high => the projector can be refreshed rarely, as GaLore does")
