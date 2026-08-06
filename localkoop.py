"""LOCAL KOOPMAN: DENSE SAMPLING ON A NARROW WINDOW.

Global Koopman failed with R2 < 0 for every observable set, and the diagnosis was
non-autonomy: 61 checkpoints spanning val 6 -> 0.1. The sliding-window test then
showed relative error falling monotonically with window length (8.96, 8.64, 2.37,
1.06 at windows 8,16,24,40) but never below 1, because a longer window on that
ladder dissolves the local assumption.

The fix is dense sampling on a NARROW interval, where the system is closer to
stationary: steps 200-400, val roughly 0.3 -> 0.1, Hessian positive definite
throughout, transport slowly decaying. Every 3 steps gives ~67 checkpoints over
an interval a fifth as wide in loss, so a window of 40-50 is both long and local.

Controls, as before and for the same reason: the target is the INCREMENT so
persistence scores zero by construction, plus a time-shuffled null. Reported as
relative error, where 1.0 = predicting no change.
"""
import json, subprocess, numpy as np, torch, gc, io, contextlib
subprocess.run(["python3","/mnt/user-data/uploads/build_corpus.py","--out","/tmp",
                "--loops","300"],check=True,capture_output=True)
RAW=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT=RAW.find("# \u2500\u2500 PHASE 3")
M_,NB,ND=3,4,15
CKS=list(range(200,404,3))
src=RAW[:CUT].replace("D=256; N_HEADS=4","D=128; N_HEADS=4",1)
src=src.replace("for mf_r in range(1, 16):","for mf_r in range(1, 3):",1)
src=src.replace("    if pc == N_STU-1:","    if False:",1)
src=src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                "    if False:",1)
G={}; buf=io.StringIO()
with contextlib.redirect_stdout(buf): exec(src,G)
model=G["model"]; get_batch=G["get_batch"]; LR=G["LR"]; ev=G["eval_val"]
named=[(n,p) for n,p in model.named_parameters()]; params=[p for _,p in named]
P=sum(p.numel() for p in params)
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
step=0; prev=flat(); D=[]
for ck in CKS:
    while step<ck:
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    th=flat(); u=th-prev; prev=th.clone()
    g=drift(th); V=[g/(g.norm()+1e-30)]
    for _ in range(M_-1):
        w=hvp(th,V[-1])
        for q in V: w=w-float((w*q).sum())*q
        if float(w.norm())<1e-10: break
        V.append(w/w.norm())
    Q=torch.stack(V,1)
    HQ=torch.stack([hvp(th,Q[:,j]) for j in range(Q.shape[1])],1)
    T=(Q.T@HQ).numpy(); T=(T+T.T)/2
    vv=torch.cat([opt.state[p]["exp_avg_sq"].flatten() for p in params])
    Md=1.0/(vv.sqrt()+1e-8); un=u/(u.norm()+1e-30)
    ln=[float(p.data.mean()) for n,p in named if "ln" in n.lower()]
    D.append(dict(ck=ck,val=float(ev(model,n=4)),
                  cu=[float(x) for x in (Q.T@un)],
                  sig=[float(x) for x in np.linalg.eigvalsh(T)],
                  cv=[float(x) for x in (Q.T@((Md*g)/((Md*g).norm()+1e-30)))],
                  dF=float((u*Md).norm()/(u.norm()+1e-30)),
                  ln=float(np.mean(ln)) if ln else 0.0)); model.train()
    if ck%30<3: print(f"  ck {ck:>4} val {D[-1]['val']:.4f}",flush=True)
json.dump(D,open("/home/claude/work/res_localkoop.json","w"),indent=2)
n=len(D); v0,v1=D[0]["val"],D[-1]["val"]
print(f"\n  n={n} checkpoints, val {v0:.3f} -> {v1:.3f} "
      f"(global run spanned 6.0 -> 0.1)")
X=np.array([d["cu"]+d["sig"]+d["cv"]+[d["dF"],d["ln"]] for d in D])
Z=(X-X.mean(0))/(X.std(0)+1e-9)
rg=np.random.default_rng(0)
print(f"\n  {'window':>8}{'n_pred':>8}{'rel err':>10}{'null':>9}{'ratio':>8}{'verdict':>9}")
for W in (16,24,32,40,50,60):
    if n-W-1<6: continue
    e=[];nu=[]
    for s in range(0,n-W-1):
        A=Z[s:s+W]; dB=Z[s+1:s+W+1]-A
        K=np.linalg.lstsq(A,dB,rcond=None)[0]
        pred=Z[s+W]@K; true=Z[s+W+1]-Z[s+W]
        e.append(((true-pred)**2).sum()/max((true**2).sum(),1e-30))
        p=rg.permutation(W)
        Kn=np.linalg.lstsq(A,dB[p],rcond=None)[0]
        nu.append(((true-Z[s+W]@Kn)**2).sum()/max((true**2).sum(),1e-30))
    e=np.array(e); nu=np.array(nu)
    print(f"  {W:>8}{len(e):>8}{e.mean():>10.3f}{nu.mean():>9.3f}"
          f"{e.mean()/nu.mean():>8.3f}{('BEATS ZERO' if e.mean()<1 else '-'):>9}")
print(f"\n  rel err < 1 => the local linear operator predicts better than no change")
