"""IS SIGN+TOP'S WIN DIRECTION, OR JUST STEP SIZE?

The trajectory table reports low rank 1.502 against sign+top 0.310 at matched
storage. But the operator diagnostics show |u_hat|/|u| equals cos in every row,
because <u,u_hat> = |u_hat|^2 for both operators, so the two columns are ONE
column:

    low rank   cos 0.568, norm retained 0.568
    sign+top   cos 0.949, norm retained 0.949

An orthogonal projection can only REMOVE norm. So the low-rank arm was running
at roughly 57% of the intended learning rate -- and at a step-dependent one,
since the retention drifts with the spectrum. That is the step-norm collapse of
Correction 2 returning through the projection itself rather than through an
explicit scale factor.

The comparison as it stands cannot separate "wrong direction" from "smaller
step". This adds one arm:

    lowrank_nm    project, then rescale u_hat to |u|

  lowrank_nm ~ 0.31  the result is a learning-rate artefact and there is
                     nothing to replicate
  lowrank_nm ~ 1.5   directional fidelity survives its most obvious
                     alternative explanation

sign+top is also rescaled, for symmetry -- it retains 0.949 rather than 1.000,
so it is mildly under-stepped too and the same correction should apply to both.

All arms: identical initialisation, batches, optimizer, and 140 steps. No
plateau rule in this script, so every arm runs the full budget.
"""
import io, contextlib, subprocess, sys, math, json
import numpy as np, torch
subprocess.run([sys.executable,"build_corpus.py","--out","/tmp","--loops","300"],
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
def kbud(p,r):
    return max(0,int(((r*(p.shape[0]+p.shape[1])+2*r*r)-p.numel()/32.0-1)//2))
def run(mode,r,normmatch):
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
    curve=[]; rets=[]
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
                    k=kbud(p,r); fl=u[n].reshape(-1); z=torch.zeros_like(fl)
                    m=torch.ones_like(fl,dtype=torch.bool)
                    if k>0:
                        idx=torch.topk(fl.abs(),min(k,fl.numel()),sorted=False).indices
                        m[idx]=False; z[idx]=fl[idx]
                    if m.any(): z[m]=torch.sign(fl[m])*float(fl[m].abs().mean())
                    c[n]=z.view_as(u[n])
            un=math.sqrt(sum(float((u[n]*u[n]).sum()) for n,_ in MATS))
            cn=math.sqrt(sum(float((c[n]*c[n]).sum()) for n,_ in MATS))
            rets.append(cn/max(un,1e-30))
            if normmatch and cn>1e-30:
                sc=un/cn
                c={n:c[n]*sc for n,_ in MATS}
            with torch.no_grad():
                for n,p in MATS: p.data.copy_(th[n]+c[n])
        if (st+1)%28==0: curve.append((st+1,L()))
    v=L()
    del G,model
    import gc; gc.collect()
    return dict(final=v,curve=curve,ret=float(np.mean(rets)) if rets else 1.0)
arms=[("none",0,False),("lowrank",32,False),("lowrank",32,True),
      ("signtop",32,False),("signtop",32,True),
      ("lowrank",4,True),("signtop",4,True)]
res={}
for m,r,nm in arms:
    res[(m,r,nm)]=run(m,r,nm)
    print(f"  {m} r={r} normmatch={nm} done",flush=True)
lab=lambda m,r,nm: f"{m} r={r}"+(" NM" if nm else "")
print(f"\n  {'step':>6}"+"".join(f"{lab(*a)[:13]:>14}" for a in arms))
for i in range(len(res[arms[0]]["curve"])):
    print(f"  {res[arms[0]]['curve'][i][0]:>6}"
          +"".join(f"{res[a]['curve'][i][1]:>14.4f}" for a in arms))
print(f"\n  {'arm':>16}{'final':>10}{'vs none':>10}{'mean |uh|/|u|':>15}")
b=res[arms[0]]["final"]
for a in arms:
    d=res[a]
    print(f"  {lab(*a):>16}{d['final']:>10.4f}{d['final']/b:>10.3f}{d['ret']:>15.3f}")
json.dump({lab(*a):res[a]["final"] for a in arms},open("res_normmatch.json","w"),indent=2)
lr_nm=res[("lowrank",32,True)]["final"]; st_nm=res[("signtop",32,True)]["final"]
print(f"\n  low rank r=32 norm-matched: {lr_nm:.4f}   sign+top norm-matched: {st_nm:.4f}")
print(f"  ratio {lr_nm/st_nm:.2f}x")
print(f"\n  ratio ~ 1 => the headline was a learning-rate artefact")
print(f"  ratio >> 1 => directional fidelity survives the obvious alternative")
