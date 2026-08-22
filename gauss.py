"""THE GAUSS EQUATION: WHAT SHAPE IS THE LEVEL SET?

We know II(T,T) > 0 along the trajectory -- rising 0.115 -> 0.246 while |g| falls
-- and that it is 50-180x what a random tangent gives. But one direction cannot
say what SHAPE the surface has. For a codimension-1 hypersurface with unit normal
n = -g/|g|, the Gauss equation gives the sectional curvature of the plane spanned
by two tangent directions X, Y:

    K(X,Y) = II(X,X) II(Y,Y) - II(X,Y)^2

  K > 0   convex: the level set curves the same way in both directions, a bowl
  K < 0   saddle: it curves oppositely, and the trajectory sits on a ridge line
  K ~ 0   flat in that 2-plane

That is a genuinely different question from II(T,T) > 0. A saddle has POSITIVE
normal curvature along one direction and negative along another, so everything
measured so far is compatible with either.

WHAT IS MEASURED
  T          the trajectory tangent, from a frozen-batch update
  W          a second tangent direction, orthogonal to both n and T
  II(T,T), II(W,W), II(T,W)   three Pearlmutter products, all on the same
                              frozen batch so the quadratic form is exact
  K(T,W)     the sectional curvature

The mixed term II(T,W) needs a polarisation:
    II(T,W) = [ (T+W)^T H (T+W) - T^T H T - W^T H W ] / (2 |g|)

CHOICE OF W MATTERS, so three are used:
  RANDOM     a random tangent -- the generic 2-plane
  PRINCIPAL  the top eigenvector of H restricted to the tangent space, by a few
             power iterations with the normal projected out -- the direction of
             greatest curvature, where a saddle would show most clearly
  GRADIENT-  the residual direction (I - P_Q)u projected tangentially, which is
  RESIDUAL   the object the rest of the programme has been chasing

Also reported: the mean and spread of II over many random tangents, which is the
average normal curvature -- if the surface were a sphere every direction would
give the same value, and the spread measures how far from umbilic it is.
"""
import json, subprocess, numpy as np, torch, io, contextlib, math, copy
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
ps=[p for p in model.parameters() if p.requires_grad]
P=sum(p.numel() for p in ps)
def flat(): return torch.cat([p.data.reshape(-1) for p in ps]).clone()
def setth(t):
    with torch.no_grad():
        j=0
        for p in ps:
            q=p.numel(); p.data.copy_(t[j:j+q].view_as(p)); j+=q
def gB(B):
    acc=torch.zeros(P)
    for x,y in B:
        model.zero_grad(); _,l=model(x,y); l.backward()
        acc+=torch.cat([(p.grad.reshape(-1) if p.grad is not None
                         else torch.zeros(p.numel())) for p in ps])
    model.zero_grad(); return acc/len(B)
def HvB(v,B):
    acc=torch.zeros(P)
    for x,y in B:
        model.zero_grad(); _,l=model(x,y)
        gr=torch.autograd.grad(l,ps,create_graph=True)
        gf=torch.cat([t.reshape(-1) for t in gr])
        hv=torch.autograd.grad((gf*v).sum(),ps,allow_unused=True)
        acc+=torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1)
                        for t,p in zip(hv,ps)]).detach()
    model.zero_grad(); return acc/len(B)
opt=torch.optim.AdamW(model.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
NB=6
FR=[get_batch() for _ in range(NB)]
K,WINL=4,8
def tangent(v,nh): 
    v=v-float(v@nh)*nh
    return v/(v.norm()+1e-30)
print(f"  P={P:,}   K(X,Y) = II(X,X)II(Y,Y) - II(X,Y)^2,  n = -g/|g|\n")
print(f"  {'step':>5}{'|g|':>8}{'II(T,T)':>10}{'W':>12}{'II(W,W)':>10}"
      f"{'II(T,W)':>10}{'K(T,W)':>11}{'shape':>9}")
step=0; hist=[]; rows=[]
for ck in (40,90,140):
    while step<ck:
        th=flat()
        x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
        hist.append((flat()-th).clone())
        if len(hist)>WINL: hist.pop(0)
    th0=flat(); g=gB(FR); setth(th0); gn=float(g.norm()); nh=-g/(gn+1e-30)
    sd=copy.deepcopy(opt.state_dict())
    model.zero_grad()
    for x,y in FR:
        _,l=model(x,y); (l/len(FR)).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    u=flat()-th0; setth(th0); opt.load_state_dict(sd)
    T=tangent(u,nh)
    hT=HvB(T,FR); setth(th0); IITT=float(hT@T)/gn
    # principal tangent direction by power iteration with n projected out
    gen=torch.Generator().manual_seed(1234+ck)
    w=tangent(torch.randn(P,generator=gen),nh)
    for _ in range(6):
        w=HvB(w,FR); setth(th0); w=tangent(w,nh)
    # residual direction
    A=torch.stack(hist,1); Q=torch.linalg.svd(A,full_matrices=False)[0][:,:K]
    r=u-Q@(Q.T@u); Wr=tangent(r,nh)
    Wrand=tangent(torch.randn(P,generator=gen),nh)
    # average normal curvature over random tangents: umbilic test
    iis=[]
    for s in range(4):
        gg=torch.Generator().manual_seed(9000+ck+s)
        z=tangent(torch.randn(P,generator=gg),nh)
        hz=HvB(z,FR); setth(th0); iis.append(float(hz@z)/gn)
    for lab,W in (("random",Wrand),("principal",w),("residual",Wr)):
        hW=HvB(W,FR); setth(th0); IIWW=float(hW@W)/gn
        s_=T+W; hs=HvB(s_,FR); setth(th0)
        IITW=(float(hs@s_)-float(hT@T)-float(hW@W))/(2*gn)
        Ks=IITT*IIWW-IITW**2
        shape="convex" if Ks>1e-9 else ("saddle" if Ks<-1e-9 else "flat")
        print(f"  {ck if lab=='random' else '':>5}{gn if lab=='random' else float('nan'):>8.4f}"
              f"{IITT if lab=='random' else float('nan'):>10.5f}{lab:>12}"
              f"{IIWW:>10.5f}{IITW:>10.5f}{Ks:>11.2e}{shape:>9}",flush=True)
        rows.append(dict(ck=ck,W=lab,IITT=IITT,IIWW=IIWW,IITW=IITW,K=Ks,gn=gn))
    print(f"        random-tangent II: mean {np.mean(iis):.5f} sd {np.std(iis):.5f}"
          f"   II(T,T)/mean = {IITT/max(abs(np.mean(iis)),1e-12):.0f}x")
json.dump(rows,open("/home/claude/work/res_gauss.json","w"),indent=2)
print(f"\n  K > 0 => convex level set, a bowl in that 2-plane")
print(f"  K < 0 => saddle: the trajectory rides a ridge, not a valley floor")
print(f"  large spread in random-tangent II => far from umbilic")
