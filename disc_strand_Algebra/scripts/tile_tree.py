"""
Tile-tree sensor over weights (fwd) and applied updates (bwd).
Forward  resolution by depth : L1=16 L2=32 L3=64 L4=256 L5=512 L6=1024
Backward resolution mirrored : L6=16 L5=32 L4=64 L3=256 L2=512 L1=1024
EMB uses resolution 16 (fwd) / 1024 (bwd), treated as the layer-0 boundary group.
Overlap between adjacent tiles along the flattened per-(layer,group) vector,
E(64)-gated per step: E>0.05 -> 10%, else 25%.
Tracked mean & max per tile per step.
Groups: steps 1-120 -> LN,FF,EMB ; steps 121-200 -> +W_Q,W_K,W_V,W_O.
Values: fwd tile = weights AFTER the step ; bwd tile = applied update (theta_t - theta_{t-1}).
"""
import re, json, time, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
model.load_state_dict(torch.load("init.pt"))
named=list(model.named_parameters())

def keyLT(n):
    m=re.match(r"blocks\.(\d+)\.",n)
    if not m: return ("EMB", "EMB")
    l=int(m.group(1)); nl=n.lower()
    t=("W_Q" if "wq" in nl else "W_K" if "wk" in nl else "W_V" if "wv" in nl else
       "W_O" if ".op." in nl else "LN" if (".ln." in nl or ".n." in nl) else "FF")
    return (f"L{l}", t)

# flat index ranges per (layer,group)
off=0; GRP={}
for n,p in named:
    k=keyLT(n); GRP.setdefault(k,[]).append((off,off+p.numel())); off+=p.numel()
P=off
IDX={k: torch.cat([torch.arange(a,b) for a,b in v]) for k,v in GRP.items()}
NEL={k: int(len(IDX[k])) for k in IDX}

# resolution ladder by depth
FWD_RES={"L0":16,"L1":32,"L2":64,"L3":256,"L4":512,"L5":1024,"EMB":16}
BWD_RES={"L5":16,"L4":32,"L3":64,"L2":256,"L1":512,"L0":1024,"EMB":1024}

def tile_bounds(nel, ntile, overlap):
    """contiguous tiles with fractional overlap; caps ntile at nel."""
    ntile=min(ntile, nel)
    if ntile<=1: return [(0,nel)]
    step=nel/ntile
    half=overlap*step/2.0
    b=[]
    for i in range(ntile):
        c0=i*step; c1=(i+1)*step
        a=max(0,int(np.floor(c0-half))); z=min(nel,int(np.ceil(c1+half)))
        if z<=a: z=min(nel,a+1)
        b.append((a,z))
    return b

def tstats(vals, bounds):
    m=np.empty(len(bounds),dtype=np.float32); mx=np.empty(len(bounds),dtype=np.float32)
    for i,(a,z) in enumerate(bounds):
        seg=vals[a:z]; m[i]=float(seg.mean()); mx[i]=float(seg.abs().max())
    return m, mx

def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()

# groups active by phase
def active_groups(step):
    base={"LN","FF","EMB"}
    if step>=121: base|= {"W_Q","W_K","W_V","W_O"}
    return base

# E(64) gate needs snapshots
torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
LAYERS=["EMB"]+[f"L{i}" for i in range(6)]
keys=[k for k in IDX.keys()]
DATA={ (lk,gk,side): {"mean":[], "max":[], "bounds_overlap":[]}
       for (lk,gk) in keys for side in ("fwd","bwd") }
STEP_META=[]
prevflat=flat()
snaps={}
T=200
for s in range(1,T+1):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    before=flat()
    opt.step()
    after=flat()
    upd=after-before                     # applied update = theta_t - theta_{t-1}
    snaps[s]=after.clone()
    E=float('nan'); overlap=0.10
    if s>=65 and (s-64) in snaps and (s-32) in snaps:
        S1=snaps[s-32]-snaps[s-64]; SW=snaps[s]-snaps[s-64]
        E=float(S1@SW/(S1.norm()*SW.norm()+1e-12))-float(np.sqrt(0.5))
        overlap=0.10 if E>0.05 else 0.25
    ag=active_groups(s)
    for (lk,gk) in keys:
        if gk not in ag: 
            continue
        idx=IDX[(lk,gk)]; nel=NEL[(lk,gk)]
        wv=after[idx]; uv=upd[idx]
        for side,res_map,vals in (("fwd",FWD_RES,wv),("bwd",BWD_RES,uv)):
            nt=res_map[lk]
            bnds=tile_bounds(nel, nt, overlap)
            mm,mx=tstats(vals, bnds)
            d=DATA[(lk,gk,side)]
            d["mean"].append(mm); d["max"].append(mx); d["bounds_overlap"].append((len(bnds),overlap))
    STEP_META.append({"step":s,"loss":float(l),"E64":E,"overlap":overlap,
                      "val":(float(eval_val(model,n=6)) if s%20==0 else None)})
    # free old snaps (keep >= s-64)
    for k in list(snaps):
        if k < s-64: del snaps[k]
    if s%40==0: print(f"  step {s:>3} loss {float(l):.4f} E64 {E:+.3f} overlap {overlap} ({time.time()-t0:.0f}s)", flush=True)

# save compactly
out={}
for (lk,gk,side),d in DATA.items():
    if not d["mean"]: continue
    out[f"{lk}|{gk}|{side}"]={
        "mean":[a.tolist() for a in d["mean"]],
        "max":[a.tolist() for a in d["max"]],
        "ntiles":[b[0] for b in d["bounds_overlap"]],
        "overlap":[b[1] for b in d["bounds_overlap"]],
        "first_step": 1 if gk in ("LN","FF","EMB") else 121,
    }
np.save("tilemeta.npy", {"step_meta":STEP_META}, allow_pickle=True)
import pickle
pickle.dump(out, open("tiletree.pkl","wb"))
print(f"\n  built {len(out)} (layer,group,side) series over {T} steps ({time.time()-t0:.0f}s)")
print(f"  P={P:,}  groups={len(keys)}")
