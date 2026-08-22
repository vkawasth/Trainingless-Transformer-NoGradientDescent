"""ARE THE HIGH-CURVATURE COORDINATES ROW-CLUSTERED, OR SCATTERED?

Two structures have been measured and they are in tension:

  SPARSE   diag(H) has PR = 2% of P, and the top 1% of coordinates hold 38-49%
           of the diagonal mass. Permuting D cut the [D,N] commutator to 0.35x,
           so those coordinates sit exactly where the off-diagonal couplings are
           strongest.
  LOW-RANK the per-matrix gradient has E_8 = 0.785, and a two-sided projector
           P^T G Q captures a SUBSPACE of row and column space.

A rank-r projector cannot select individual entries. A scattered sparse set is
generically full rank as a matrix, so if the 12,000 high-curvature coordinates
are spread uniformly, low-rank projection is blind to exactly the structure that
carries the curvature -- and the two pictures are different decompositions, not
complementary ones.

But if they CLUSTER in a few rows or columns, a row/column-selecting projector
would capture them, and the two reconcile.

THE TEST. Take the top-1% of diag(H) by magnitude, map each flat index back to
its (matrix, row, col), and compare the row and column marginals against
uniform:

    concentration = fraction of the hub coordinates in the top 10% of rows
    uniform null  = 0.10
    Gini          = inequality of the per-row hub counts

  concentration >> 0.10 => row-clustered, and a row-selecting projector reaches
                           them
  concentration ~ 0.10  => scattered, and low-rank cannot represent them

Also reported: whether the SAME rows stay hot across checkpoints, since a
clustered but drifting set is no more usable than a scattered one.
"""
import json, subprocess, numpy as np, torch, io, contextlib, math
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
src=RAW[:RAW.find("# \u2500\u2500 PHASE 3")].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
src=src.replace("    if pc == N_STU-1:","    if False:",1)
src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:","    if False:",1)
G={}; b=io.StringIO()
with contextlib.redirect_stdout(b): exec(src,G)
model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]
named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
ps=[p for _,p in named]; P=sum(p.numel() for p in ps)
span={}; i=0
for nm,p in named: span[nm]=(i,i+p.numel()); i+=p.numel()
EV=[get_batch() for _ in range(8)]
def flat(): return torch.cat([p.data.reshape(-1) for p in ps]).clone()
def setth(t):
    with torch.no_grad():
        j=0
        for p in ps:
            q=p.numel(); p.data.copy_(t[j:j+q].view_as(p)); j+=q
def Hv(v):
    acc=torch.zeros(P)
    for x,y in EV:
        model.zero_grad(); _,l=model(x,y)
        gr=torch.autograd.grad(l,ps,create_graph=True)
        gf=torch.cat([t.reshape(-1) for t in gr])
        hv=torch.autograd.grad((gf*v).sum(),ps,allow_unused=True)
        acc+=torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1)
                        for t,p in zip(hv,ps)]).detach()
    model.zero_grad(); return acc/len(EV)
def gini(x):
    x=np.sort(np.asarray(x,float)); n=len(x)
    if n==0 or x.sum()==0: return float("nan")
    return float((2*np.arange(1,n+1)-n-1).dot(x)/(n*x.sum()))
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
NP=20
step=0; prev_hot={}
print(f"  P={P:,}   top-1% of diag(H) mapped back to (matrix,row,col)\n")
for ck in (40,100,160):
    while step<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    th=flat(); g=torch.Generator().manual_seed(60+ck)
    d=torch.zeros(P)
    for _ in range(NP):
        z=(torch.randint(0,2,(P,),generator=g).float()*2-1)
        d+=z*Hv(z); setth(th)
    d/=NP
    k=max(1,P//100)
    hubs=torch.topk(d.abs(),k).indices
    print(f"  === step {ck}   {k:,} hub coordinates")
    print(f"  {'matrix':>20}{'shape':>12}{'hubs':>8}{'top10% rows':>13}"
          f"{'gini rows':>11}{'top10% cols':>13}{'stab rows':>11}")
    for nm,p in named:
        if p.dim()!=2: continue
        a,bb=span[nm]
        sel=hubs[(hubs>=a)&(hubs<bb)]-a
        if len(sel)<20: continue
        m_,n_=p.shape
        r=(sel//n_).numpy(); c=(sel%n_).numpy()
        rc=np.bincount(r,minlength=m_); cc=np.bincount(c,minlength=n_)
        tr=int(max(1,m_//10)); tc=int(max(1,n_//10))
        hot=np.argsort(rc)[::-1][:tr]
        conc_r=rc[hot].sum()/max(rc.sum(),1)
        conc_c=np.sort(cc)[::-1][:tc].sum()/max(cc.sum(),1)
        st=float("nan")
        if nm in prev_hot:
            st=len(set(hot.tolist())&set(prev_hot[nm]))/max(tr,1)
        prev_hot[nm]=hot.tolist()
        short=nm.replace("blocks.","b").replace(".weight","")
        print(f"  {short:>20}{str((m_,n_)):>12}{len(sel):>8}{conc_r:>13.3f}"
              f"{gini(rc):>11.3f}{conc_c:>13.3f}{st:>11.3f}")
    print()
print(f"  uniform null for 'top10%' columns is 0.100")
print(f"  >> 0.100 => row-clustered, reachable by a row-selecting projector")
print(f"  ~ 0.100  => scattered, and low-rank projection cannot represent them")
