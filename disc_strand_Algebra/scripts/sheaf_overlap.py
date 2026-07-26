"""
THE SHEAF, BUILT ON OVERLAPS.
Vertex stalk  : (alpha,beta) of a tile
Edge stalk    : (alpha,beta) of the OVERLAP region between two adjacent tiles
Restriction   : F_L, F_R  (2x2 per layer) mapping left/right tile stalk -> overlap stalk
Edges exist ONLY where tiles overlap -> the walk on the complex is directionless
                                        but confined to the nerve.
Vertical edges: parent -> child refinement.
"""
import re, time, pickle, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
model.load_state_dict(torch.load("init.pt"))
named=list(model.named_parameters())
LAY=["L0","L1","L2","L3","L4","L5"]; BWD=[1024,512,256,64,32,16]; OV=0.25
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
def idx_for(pred):
    out={}
    for L in range(6):
        sel=[torch.arange(*SPAN[n]) for n,_ in named
             if (m:=re.match(r"blocks\.(\d+)\.",n)) and int(m.group(1))==L and pred(n)]
        if sel: out[f"L{L}"]=torch.cat(sel).numpy()
    return out
IDX={"FF":idx_for(lambda n:".ff." in n.lower()),
     "LN":idx_for(lambda n:(".ln." in n.lower() or ".n." in n.lower()))}
def tiles(nel,nt,ov=OV):
    nt=min(nt,nel); st=nel/nt; h=ov*st/2
    a=np.clip(np.floor(np.arange(nt)*st-h).astype(int),0,nel-1)
    z=np.clip(np.ceil((np.arange(nt)+1)*st+h).astype(int),1,nel)
    return a,np.maximum(z,a+1)
REG={}   # (group,layer) -> (tile_bounds, overlap_bounds)
for g in IDX:
    for L,nt in zip(LAY,BWD):
        nel=len(IDX[g][L]); a,z=tiles(nel,nt)
        ova=[];ovz=[]
        for i in range(len(a)-1):
            lo=max(a[i+1],a[i]); hi=min(z[i],z[i+1])
            if hi>lo: ova.append(lo); ovz.append(hi)
            else: ova.append(a[i+1]); ovz.append(min(a[i+1]+1,nel))
        REG[(g,L)]=((a,z),(np.array(ova),np.array(ovz)))
print("  region counts:", {(g,L):(len(REG[(g,L)][0][0]),len(REG[(g,L)][1][0])) for g in IDX for L in ["L0","L5"]}, flush=True)
def flat(): return torch.cat([p.data.flatten() for _,p in named]).numpy().copy()
torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
REC={k:{"T":[],"O":[]} for k in REG}; LOSS=[]
for s in range(200):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); opt.step(); af=flat(); u=af-b4; LOSS.append(float(l))
    for (g,L),((a,z),(oa,oz)) in REG.items():
        v=np.abs(u[IDX[g][L]])
        REC[(g,L)]["T"].append(np.array([v[a[i]:z[i]].mean() for i in range(len(a))],dtype=np.float32))
        REC[(g,L)]["O"].append(np.array([v[oa[i]:oz[i]].mean() for i in range(len(oa))],dtype=np.float32))
    del b4,af,u
    if (s+1)%100==0: print(f"    step {s+1} ({time.time()-t0:.0f}s)", flush=True)
LOSS=np.array(LOSS); H=0.062; ls=LOSS-H; ok=ls>1e-6
xx=np.log(ls[ok]); xm=xx-xx.mean(); den=float((xm*xm).sum())
def stalk(M):
    Y=np.log(np.maximum(np.stack(M)[ok],1e-30))
    al=(Y-Y.mean(0)).T@xm/den
    return np.stack([al,Y.mean(0)-al*xx.mean()],1)
ST={k:{"T":stalk(v["T"]),"O":stalk(v["O"])} for k,v in REC.items()}
print(f"  stalks fitted ({time.time()-t0:.0f}s)\n", flush=True)
print("="*84); print("  RESTRICTION MAPS ON OVERLAPS  F_L, F_R : tile stalk -> overlap stalk"); print("="*84)
print(f"  {'group':>6}{'layer':>6}{'tiles':>7}{'edges':>7}{'resid F_L':>11}{'resid F_R':>11}"
      f"{'||F_L-F_R||':>13}")
FLR={}
for (g,L) in ST:
    T=ST[(g,L)]["T"]; O=ST[(g,L)]["O"]; n=T.shape[0]
    if O.shape[0]<3: continue
    XL=T[:-1]; XR=T[1:]
    FL,_,_,_=np.linalg.lstsq(XL,O,rcond=None); FR,_,_,_=np.linalg.lstsq(XR,O,rcond=None)
    rl=np.linalg.norm(XL@FL-O)/max(np.linalg.norm(O),1e-30)
    rr=np.linalg.norm(XR@FR-O)/max(np.linalg.norm(O),1e-30)
    FLR[(g,L)]=(FL.T,FR.T)
    print(f"  {g:>6}{L:>6}{n:>7}{O.shape[0]:>7}{rl:>11.4f}{rr:>11.4f}"
          f"{np.linalg.norm(FL-FR):>13.4f}")
print("\n"+"="*84); print("  SHEAF COHOMOLOGY, HORIZONTAL (overlap) EDGES ONLY"); print("="*84)
print(f"  {'group':>6}{'layer':>6}{'2V':>7}{'2E':>7}{'rank d':>8}{'dim H0':>8}{'dim H1':>8}"
      f"{'lambda_1(L)':>13}")
for (g,L) in sorted(FLR):
    T=ST[(g,L)]["T"]; n=T.shape[0]; m=n-1
    FL,FR=FLR[(g,L)]
    d=np.zeros((2*m,2*n))
    for e in range(m):
        d[2*e:2*e+2, 2*e:2*e+2]     =  FL
        d[2*e:2*e+2, 2*(e+1):2*e+4] = -FR
    sv=np.linalg.svd(d,compute_uv=False)
    tol=1e-8*max(sv[0],1e-30); r=int((sv>tol).sum())
    Lap=d.T@d; ev=np.linalg.eigvalsh(Lap)
    lam1=ev[ev>1e-10][0] if (ev>1e-10).any() else 0.0
    print(f"  {g:>6}{L:>6}{2*n:>7}{2*m:>7}{r:>8}{2*n-r:>8}{2*m-r:>8}{lam1:>13.3e}")
print("\n  (dim H0 = 2 means one global section per stalk coordinate: the sheaf glues)")
pickle.dump({"ST":ST,"FLR":FLR,"REG":{k:(v[0][0].tolist(),v[0][1].tolist()) for k,v in REG.items()}},
            open("sheaf_overlap.pkl","wb"))
print(f"\n  time {time.time()-t0:.0f}s")
