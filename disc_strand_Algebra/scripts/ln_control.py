"""
Two controls for the LN/bwd anomaly.
 (A) split LN into LN_w (gains) and LN_b (offsets)  -> tests weight/bias concatenation
 (B) FF1024: a 1024-element slice of FF tiled with the SAME backward ladder
     -> occupancy-matched to LN (1..64 elements/tile). Tests whether the anomaly
        is about LayerNorm at all, or just about tiles with few elements.
"""
import re, time, pickle, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
model.load_state_dict(torch.load("init.pt"))
named=list(model.named_parameters())
LAY=["L0","L1","L2","L3","L4","L5"]
BWD={"L0":1024,"L1":512,"L2":256,"L3":64,"L4":32,"L5":16}
# ---- group definitions ----
off=0; SPAN={}
for n,p in named:
    SPAN[n]=(off,off+p.numel()); off+=p.numel()
P=off
def idx_for(pred):
    out={}
    for L in range(6):
        sel=[]
        for n,_ in named:
            m=re.match(r"blocks\.(\d+)\.",n)
            if not m or int(m.group(1))!=L: continue
            if pred(n): sel.append(torch.arange(*SPAN[n]))
        if sel: out[f"L{L}"]=torch.cat(sel).numpy()
    return out
IDX={}
IDX["LN_w"]=idx_for(lambda n: (".ln." in n.lower() or ".n." in n.lower()) and n.endswith("weight"))
IDX["LN_b"]=idx_for(lambda n: (".ln." in n.lower() or ".n." in n.lower()) and n.endswith("bias"))
IDX["LN"]  =idx_for(lambda n: (".ln." in n.lower() or ".n." in n.lower()))
FFall      =idx_for(lambda n: ".ff." in n.lower())
IDX["FF"]  =FFall
IDX["FF1024"]={L:v[:1024] for L,v in FFall.items()}      # occupancy-matched slice
for k in IDX: print(f"  {k}: elements/layer = {len(IDX[k]['L0'])}", flush=True)
def bounds(nel,nt,ov=0.10):
    nt=min(nt,nel)
    if nt<=1: return np.array([0]),np.array([nel])
    st=nel/nt; half=ov*st/2
    a=np.clip(np.floor(np.arange(nt)*st-half).astype(int),0,nel-1)
    z=np.clip(np.ceil((np.arange(nt)+1)*st+half).astype(int),1,nel); z=np.maximum(z,a+1)
    return a,z
BND={(g,L):bounds(len(IDX[g][L]),BWD[L]) for g in IDX for L in LAY}
def flat(): return torch.cat([p.data.flatten() for _,p in named]).numpy().copy()
torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
REC={(g,L):[] for g in IDX for L in LAY}; LOSS=[]
for s in range(1,201):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    b4=flat(); opt.step(); af=flat(); upd=af-b4; LOSS.append(float(l))
    for g in IDX:
        for L in LAY:
            v=np.abs(upd[IDX[g][L]]); a,z=BND[(g,L)]
            REC[(g,L)].append(np.array([v[a[i]:z[i]].max() for i in range(len(a))],dtype=np.float32))
    del b4,af,upd
    if s%50==0: print(f"    step {s} ({time.time()-t0:.0f}s)", flush=True)
LOSS=np.array(LOSS); H=0.062
def stalks(g,L):
    A=np.stack(REC[(g,L)]).astype(np.float64)
    ls=LOSS-H; ok=ls>1e-6
    x=np.log(ls[ok]); Y=np.log(np.maximum(A[ok],1e-30))
    xm=x-x.mean(); den=float((xm*xm).sum())
    al=(Y-Y.mean(0)).T@xm/den; be=Y.mean(0)-al*x.mean()
    return np.stack([al,be],1)
def rungs(g):
    order=LAY[::-1]; X={L:stalks(g,L) for L in order}
    Ms=[]
    for a,b in zip(order[:-1],order[1:]):
        na,nb=X[a].shape[0],X[b].shape[0]
        if nb%na: continue
        r=nb//na
        Xp=np.repeat(X[a],r,axis=0); Xc=X[b]
        M,_,_,_=np.linalg.lstsq(Xp,Xc,rcond=None); M=M.T
        Ms.append((f"{a}->{b}",M))
    return Ms
print("\n"+"="*80); print("  CONTROLS FOR THE LN/bwd ANOMALY"); print("="*80)
print(f"  {'group':>8}{'el/tile L0':>12}{'det range':>20}{'mean|det-1|':>13}{'|ev2/ev1| comp':>16}")
for g in ["LN","LN_w","LN_b","FF","FF1024"]:
    Ms=rungs(g); dets=np.array([np.linalg.det(M) for _,M in Ms])
    C=np.eye(2)
    for _,M in Ms: C=M@C
    w=np.sort(np.abs(np.linalg.eigvals(C)))[::-1]
    ept=len(IDX[g]["L0"])/BWD["L0"]
    print(f"  {g:>8}{ept:>12.2f}   [{dets.min():+.3f},{dets.max():+.3f}]"
          f"{np.abs(dets-1).mean():>13.3f}{w[1]/max(w[0],1e-300):>16.4f}")
print("\n  per-rung determinants:")
for g in ["LN","LN_w","LN_b","FF1024","FF"]:
    Ms=rungs(g)
    print(f"    {g:>8}  " + "  ".join(f"{r}:{np.linalg.det(M):+.3f}" for r,M in Ms))
print("\n  READING: if FF1024 reproduces LN's det range and non-reduction, the anomaly")
print("  is tile occupancy, not LayerNorm. If LN_w and LN_b each collapse, it was")
print("  weight/bias concatenation.")
print(f"\n  time {time.time()-t0:.0f}s")
