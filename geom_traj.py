"""SAME STORAGE, DIFFERENT GEOMETRY, TRAJECTORY TEST.

Local result: at nominally equal budget, sign+top retained 0.79 of the descent
against low-rank's 0.44. But the accounting was WRONG -- I never charged for the
sign field itself, which costs numel/32 floats-equivalent. At r=4 that is 2,048
against a low-rank budget of 2,080, so the sign field alone nearly exhausts the
budget and my arm used 1,040 full-precision coordinates ON TOP: 2x over. At
r=32 the overspend was 1.11x.

Corrected here: k is solved so that

    2k  +  numel/32  +  1   ==   r(d1+d2) + 2r^2
    (value+index)   (signs)  (scale)      (low-rank cost)

which gives k = 15 at r=4 and 8,191 at r=32.

AND THE TRAJECTORY TEST, which is the part that matters. Every local win today
has failed to transfer: sign was 20.8x better per step and 2.44x worse as an
optimizer. So retained descent at a step says nothing until the arms are run.

    THREE ARMS, identical initialisation, optimizer, batches, and horizon:
      none        AdamW, uncompressed
      lowrank     u -> P P^T u Q Q^T from the gradient SVD
      signtop     u -> top-k at full precision, rest as sign x block mean

Reported: L(t), and the cumulative sum of -g^T C(u), which separates
compression-induced local descent loss from the nonlinear trajectory outcome.

SUPERSEDED IN ONE RESPECT, and --normmatch fixes it. This script as first
written did NOT rescale the compressed update, and an orthogonal projection can
only REMOVE norm: low rank retained 0.51 of |u|, so that arm ran at half the
intended learning rate, drifting with the spectrum. The comparison was therefore
partly a learning-rate comparison. Rescaling u_hat to |u| moves low rank from
1.527 to 0.782 and sign+top from 0.319 to 0.283 -- roughly half the apparent gap
was the artefact. The residual 2.77x is the directional effect.

Run WITH --normmatch for the defensible number; without it to reproduce the
confound. Both are kept because the correction has more methodological value
than the inflated result did.
"""
import sys as _sys
NORMMATCH = "--normmatch" in _sys.argv
import json, subprocess, numpy as np, torch, io, contextlib, math, copy, sys
subprocess.run(["python3","build_corpus.py","--out","/tmp","--loops","300"],
               check=True,capture_output=True)
RAW=open("compiler_geometri_patched_86.py").read()
SRC=RAW[:RAW.find("# \u2500\u2500 PHASE 3")].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
SRC=SRC.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
SRC=SRC.replace("    if pc == N_STU-1:","    if False:",1)
SRC=SRC.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:","    if False:",1)
def build():
    torch.manual_seed(1234); np.random.seed(1234)
    G={}; b=io.StringIO()
    with contextlib.redirect_stdout(b): exec(SRC,G)
    return G
NS=140
def kbudget(p,r):
    """k such that 2k + numel/32 + 1 == r(d1+d2) + 2r^2"""
    d1,d2=p.shape
    lr=r*(d1+d2)+2*r*r
    return max(0,int((lr-p.numel()/32.0-1)//2))
def run(mode,r):
    G=build(); model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]
    MATS=[(n,p) for n,p in model.named_parameters() if p.dim()==2 and p.requires_grad]
    EV=[get_batch() for _ in range(8)]
    def L():
        t=0.0
        with torch.no_grad():
            for x,y in EV: t+=float(model(x,y)[1])
        return t/len(EV)
    torch.manual_seed(17)
    opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
    cum=0.0; curve=[]
    for st in range(NS):
        th={n:p.data.clone() for n,p in MATS}
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        g={n:p.grad.detach().clone() for n,p in MATS}
        opt.step()
        u={n:(p.data-th[n]).clone() for n,p in MATS}
        if mode!="none":
            c={}
            for n,p in MATS:
                if mode=="lowrank":
                    rr=min(r,min(p.shape))
                    U,S,Vt=torch.linalg.svd(g[n],full_matrices=False)
                    P_,Q_=U[:,:rr],Vt[:rr].T
                    c[n]=P_@(P_.T@u[n]@Q_)@Q_.T
                else:
                    k=kbudget(p,r)
                    fl=u[n].reshape(-1)
                    z=torch.zeros_like(fl)
                    if k>0:
                        idx=torch.topk(fl.abs(),min(k,fl.numel()),sorted=False).indices
                        m=torch.ones_like(fl,dtype=torch.bool); m[idx]=False
                        z[idx]=fl[idx]
                    else:
                        m=torch.ones_like(fl,dtype=torch.bool)
                    if m.any():
                        z[m]=torch.sign(fl[m])*float(fl[m].abs().mean())
                    c[n]=z.view_as(u[n])
            if NORMMATCH:
                un=math.sqrt(sum(float((u[n]*u[n]).sum()) for n,_ in MATS))
                cn=math.sqrt(sum(float((c[n]*c[n]).sum()) for n,_ in MATS))
                if cn>1e-30: c={n:c[n]*(un/cn) for n,_ in MATS}
            with torch.no_grad():
                for n,p in MATS: p.data.copy_(th[n]+c[n])
        else:
            c=u
        cum+=sum(float((-g[n]*c[n]).sum()) for n,_ in MATS)
        if (st+1)%20==0: curve.append((st+1,L(),cum))
    v=L()
    del G,model
    import gc; gc.collect()
    return dict(final=v,curve=curve,cum=cum)
print(f"  matched storage: 2k + numel/32 + 1 == r(d1+d2) + 2r^2")
print(f"  norm matching: {'ON' if NORMMATCH else 'OFF -- confounded, see docstring'}\n")
for r in (4,32):
    p=torch.zeros(256,256)
    print(f"    r={r:>2}: low-rank {r*512+2*r*r:>7,} floats, sign+top k={kbudget(p,r):>6,}")
print()
res={}
for r in (4,32):
    for m in ("lowrank","signtop"):
        res[(m,r)]=run(m,r); print(f"  {m} r={r} done",flush=True)
res[("none",0)]=run("none",0); print("  uncompressed done",flush=True)
arms=[("none",0),("lowrank",4),("signtop",4),("lowrank",32),("signtop",32)]
print(f"\n  {'step':>6}"+"".join(f"{m[:7]+str(r):>12}" for m,r in arms))
for i in range(len(res[("none",0)]["curve"])):
    print(f"  {res[('none',0)]['curve'][i][0]:>6}"
          +"".join(f"{res[(m,r)]['curve'][i][1]:>12.4f}" for m,r in arms))
print(f"\n  {'arm':>14}{'final':>10}{'vs none':>10}{'cumulative -g.C(u)':>20}")
b=res[("none",0)]["final"]
for m,r in arms:
    d=res[(m,r)]
    print(f"  {m+' r='+str(r):>14}{d['final']:>10.4f}{d['final']/b:>10.3f}{d['cum']:>20.4f}")
json.dump({f"{m}_{r}":{"final":res[(m,r)]["final"],"cum":res[(m,r)]["cum"]}
           for m,r in arms},open("res_geom_traj.json","w"),indent=2)
print(f"\n  reference, both norm-matched at r=32: lowrank 0.782, signtop 0.283")
print(f"  reference, unmatched:                lowrank 1.527, signtop 0.319")
print(f"\n  signtop < lowrank in final loss => the geometry pivot survives")
print(f"  signtop ~ lowrank              => the local win did not transfer,")
print(f"     as sign's 20.8x per step became 2.44x worse as a trajectory")
