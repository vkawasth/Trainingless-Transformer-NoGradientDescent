"""
Refinement operators as objects: spectrum, eigenvectors, normality, conditioning.
Adds the control the previous run lacked: if a group's stalk cloud is nearly
degenerate, the 2x2 fit is ill-posed and can return near-identity by default.
"""
import re, sys, time, pickle, numpy as np, torch
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 17
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
if SEED==17: model.load_state_dict(torch.load("init.pt"))
named=list(model.named_parameters())
LAY=["L0","L1","L2","L3","L4","L5"]; BWD={"L0":1024,"L1":512,"L2":256,"L3":64,"L4":32,"L5":16}
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
def idx_for(pred):
    out={}
    for L in range(6):
        sel=[torch.arange(*SPAN[n]) for n,_ in named
             if (m:=re.match(r"blocks\.(\d+)\.",n)) and int(m.group(1))==L and pred(n)]
        if sel: out[f"L{L}"]=torch.cat(sel).numpy()
    return out
isln=lambda n:(".ln." in n.lower() or ".n." in n.lower())
IDX={}
IDX["LN"]=idx_for(isln)
IDX["LN_w"]=idx_for(lambda n: isln(n) and n.endswith("weight"))
IDX["LN_b"]=idx_for(lambda n: isln(n) and n.endswith("bias"))
FFa=idx_for(lambda n:".ff." in n.lower()); IDX["FF"]=FFa
IDX["FF1024"]={L:v[:1024] for L,v in FFa.items()}
IDX["W_V"]=idx_for(lambda n:"wv" in n.lower())
def bnd(nel,nt,ov=0.10):
    nt=min(nt,nel)
    if nt<=1: return np.array([0]),np.array([nel])
    st=nel/nt; h=ov*st/2
    a=np.clip(np.floor(np.arange(nt)*st-h).astype(int),0,nel-1)
    z=np.clip(np.ceil((np.arange(nt)+1)*st+h).astype(int),1,nel)
    return a,np.maximum(z,a+1)
B={(g,L):bnd(len(IDX[g][L]),BWD[L]) for g in IDX for L in LAY}
def flat(): return torch.cat([p.data.flatten() for _,p in named]).numpy().copy()
torch.manual_seed(SEED)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
REC={(g,L):[] for g in IDX for L in LAY}; LOSS=[]
for s in range(1,201):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); opt.step(); af=flat(); u=af-b4; LOSS.append(float(l))
    for g in IDX:
        for L in LAY:
            v=np.abs(u[IDX[g][L]]); a,z=B[(g,L)]
            REC[(g,L)].append(np.array([v[a[i]:z[i]].max() for i in range(len(a))],dtype=np.float32))
    del b4,af,u
LOSS=np.array(LOSS); H=0.062
def stalk(g,L):
    A=np.stack(REC[(g,L)]).astype(np.float64); ls=LOSS-H; ok=ls>1e-6
    x=np.log(ls[ok]); Y=np.log(np.maximum(A[ok],1e-30))
    xm=x-x.mean(); al=(Y-Y.mean(0)).T@xm/float((xm*xm).sum())
    return np.stack([al, Y.mean(0)-al*x.mean()],1)
X={g:{L:stalk(g,L) for L in LAY} for g in IDX}
pickle.dump({"X":X,"seed":SEED}, open(f"stalks_{SEED}.pkl","wb"))
print(f"seed {SEED}  ({time.time()-t0:.0f}s)")
print("="*82); print("  (0) CONDITIONING CONTROL: is the stalk cloud degenerate?"); print("="*82)
print("  cov eigenvalues of the (alpha,beta) cloud per layer; ratio<<1 => 2x2 fit ill-posed")
print(f"  {'group':>8}" + "".join(f"{L:>11}" for L in LAY))
for g in IDX:
    row=[]
    for L in LAY:
        C=np.cov(X[g][L].T); w=np.sort(np.abs(np.linalg.eigvalsh(C)))[::-1]
        row.append(w[1]/max(w[0],1e-300))
    print(f"  {g:>8}" + "".join(f"{v:>11.2e}" for v in row))
print("\n"+"="*82); print("  (1) THE REFINEMENT OPERATOR AS AN OBJECT"); print("="*82)
def rungs(g):
    order=LAY[::-1]; out=[]
    for a,b in zip(order[:-1],order[1:]):
        Xa,Xb=X[g][a],X[g][b]
        if Xb.shape[0]%Xa.shape[0]: continue
        r=Xb.shape[0]//Xa.shape[0]
        Xp=np.repeat(Xa,r,axis=0)
        M,_,_,_=np.linalg.lstsq(Xp,Xb,rcond=None)
        out.append((f"{a}->{b}",M.T,np.linalg.cond(Xp)))
    return out
print(f"  {'group':>8}{'|l1|':>8}{'|l2|':>8}{'l2/l1':>8}{'det':>8}{'complex?':>10}"
      f"{'non-normality':>15}{'gamma=(a,b)':>18}")
RES={}
for g in ["FF","FF1024","W_V","LN","LN_w","LN_b"]:
    Rs=rungs(g); C=np.eye(2)
    for _,M,_ in Rs: C=M@C
    w,V=np.linalg.eig(C); o=np.argsort(-np.abs(w)); w=w[o]; V=V[:,o]
    nn=np.linalg.norm(C.T@C-C@C.T)/max(np.linalg.norm(C)**2,1e-300)
    gam=np.real(V[:,1]); gam=gam/np.linalg.norm(gam); gam*=np.sign(gam[np.argmax(np.abs(gam))])
    RES[g]=dict(C=C,w=w,V=V,gamma=gam)
    print(f"  {g:>8}{abs(w[0]):>8.4f}{abs(w[1]):>8.4f}{abs(w[1])/abs(w[0]):>8.4f}"
          f"{np.real(np.linalg.det(C)):>8.4f}{'yes' if abs(np.imag(w[0]))>1e-9 else 'no':>10}"
          f"{nn:>15.4f}   ({gam[0]:+.3f},{gam[1]:+.3f})")
print("\n  beta (dominant) eigenvector per group:")
for g in RES:
    b=np.real(RES[g]["V"][:,0]); b=b/np.linalg.norm(b); b*=np.sign(b[1] if abs(b[1])>1e-12 else 1)
    print(f"    {g:>8}  beta=({b[0]:+.4f},{b[1]:+.4f})   angle(beta,gamma)="
          f"{np.degrees(np.arccos(min(1,abs(float(b@RES[g]['gamma']))))):.1f} deg")
print("\n"+"="*82); print("  (2) WHAT IS THE GAMMA COORDINATE?  gamma-field vs observables"); print("="*82)
FIS=pickle.load(open("fisher.pkl","rb"))
print(f"  {'group':>8}{'layer':>7}{'corr(g-field, log Fisher)':>27}{'corr(g-field, tile pos)':>25}")
for g in ["LN_w","LN_b","FF"]:
    gam=RES[g]["gamma"]
    for L in ["L2","L0"]:
        f=X[g][L]@gam
        key=f"{L}|"+("LN" if g.startswith("LN") else "FF")
        fv=FIS[key]; nt=len(f)
        e=np.linspace(0,len(fv),nt+1).astype(int)
        w=np.array([fv[e[i]:e[i+1]].sum() for i in range(nt)])
        lw=np.log(np.maximum(w,1e-30))
        c1=np.corrcoef(f,lw)[0,1]; c2=np.corrcoef(f,np.arange(nt))[0,1]
        print(f"  {g:>8}{L:>7}{c1:>27.3f}{c2:>25.3f}")
