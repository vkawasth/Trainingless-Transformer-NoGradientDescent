"""P(r) SHAPE AND A SECOND-ORDER PREDICTOR.

E1. P_t(r) = ||Q(r)Q(r)^T dtheta||^2 / ||dtheta||^2 for r = 1..32. The SHAPE
    decides more than any single value:
      saturates fast   -> a genuinely low-dimensional privileged subspace
      grows ~linearly  -> the Krylov basis is no better than adding directions
    Control: the same projection onto a RANDOM r-dim subspace, which grows
    exactly linearly at r/P, so the comparison is against the right null.
    (Note the earlier 7.6% figure was the randomized range finder, not a Krylov
    space; the Krylov capture of Hv is 85-97%, so that prior does not carry.)

E2. Second-order Grassmann predictor. The first-order model beat persistence
    6/9 intervals then failed late. If the late failure is trajectory curvature,
    a second-order model should recover it; if it is missing state, it will not.
    Compared against persistence and against a random-tangent control of matched
    step size, which the first-order test lacked.
"""
import json, subprocess, numpy as np, torch, gc, io, contextlib
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
NB,ND=10,50
RS=[1,2,3,4,6,8,12,16,24,32]
CKS=[6,10,16,24,36,52,76,110]
src=RAW[:CUT].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
src=src.replace("    if pc == N_STU-1:","    if False:",1)
src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                "    if False:",1)
G={}; buf=io.StringIO()
with contextlib.redirect_stdout(buf): exec(src,G)
model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]; ev=G["eval_val"]
params=[p for _,p in model.named_parameters()]; P=sum(p.numel() for p in params)
for p in params: p.requires_grad_(True)
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
def kry(th,r):
    g=drift(th); V=[g/(g.norm()+1e-30)]
    for _ in range(r-1):
        w=hvp(th,V[-1])
        for u in V: w=w-float((w*u).sum())*u
        if float(w.norm())<1e-10: break
        V.append(w/w.norm())
    return torch.stack(V,1)
def ov(A,B):
    k=min(A.shape[1],B.shape[1])
    return float((A[:,:k].T@B[:,:k]).pow(2).sum()/k)
def logmap(Q0,Q1):
    Qp=Q1-Q0@(Q0.T@Q1)
    U,S,Vt=torch.linalg.svd(Qp@torch.linalg.pinv(Q0.T@Q1),full_matrices=False)
    return U@torch.diag(torch.atan(torch.clamp(S,-1+1e-7,1-1e-7)))@Vt
def expmap(Q0,H):
    U,S,Vt=torch.linalg.svd(H,full_matrices=False)
    return Q0@Vt.T@torch.diag(torch.cos(S))@Vt+U@torch.diag(torch.sin(S))@Vt
rg=np.random.default_rng(0)
step=0; prev=flat(); Q3s=[]; rows=[]
print("=== E1: P(r), capture of dtheta by the top-r Krylov space ===")
print(f"  {'s':>5}{'val':>8}"+"".join(f"{'r='+str(r):>9}" for r in RS))
for ck in CKS:
    while step<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    th=flat(); u=th-prev; prev=th.clone()
    Qmax=kry(th,max(RS)); un=float((u*u).sum())
    ps=[float((Qmax[:,:r].T@u).pow(2).sum()/un) for r in RS]
    Q3s.append(Qmax[:,:3].clone())
    v=float(ev(model,n=5)); model.train()
    rows.append(dict(ck=ck,val=v,P=ps))
    print(f"  {ck:>5}{v:>8.3f}"+"".join(f"{p:>9.4f}" for p in ps),flush=True)
    json.dump(rows,open("/home/claude/work/res_pofr.json","w"),indent=2)
print(f"\n  random-subspace null at r=32: {32/P:.2e}")
ps=np.array([r["P"] for r in rows]).mean(0)
print(f"  mean P(r): "+"  ".join(f"r{r}:{p:.4f}" for r,p in zip(RS,ps)))
print(f"  P(32)/P(3) = {ps[-1]/ps[2]:.2f}  (linear growth would give {32/3:.2f})")
print("\n=== E2: second-order vs first-order vs persistence vs random tangent ===")
print(f"  {'from->to':>12}{'PERSIST':>10}{'1st':>9}{'2nd':>9}{'random':>9}")
for i in range(2,len(Q3s)):
    Qa,Qb,Qc=Q3s[i-2],Q3s[i-1],Q3s[i]
    pers=ov(Qb,Qc)
    H1=logmap(Qa,Qb); d1=CKS[i-1]-CKS[i-2]; d2=CKS[i]-CKS[i-1]
    p1=ov(expmap(Qb,H1*(d2/d1)),Qc)
    # second order: extrapolate the velocity itself
    if i>=3:
        H0=logmap(Q3s[i-3],Qa); d0=CKS[i-2]-CKS[i-3]
        acc=(H1/d1-H0/d0)
        p2=ov(expmap(Qb,(H1/d1+acc*d2)*d2),Qc)
    else: p2=float('nan')
    R=torch.randn(P,3); R=R-Qb@(Qb.T@R); R,_=torch.linalg.qr(R)
    Hr=logmap(Qb,expmap(Qb,R@torch.diag(torch.linalg.svdvals(H1*(d2/d1)))[:3,:3]))
    pr=ov(expmap(Qb,Hr),Qc)
    print(f"  {CKS[i-1]:>5}->{CKS[i]:<5}{pers:>10.4f}{p1:>9.4f}{p2:>9.4f}{pr:>9.4f}")
