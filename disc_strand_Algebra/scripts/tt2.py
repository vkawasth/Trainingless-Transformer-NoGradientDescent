import re, time, pickle, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
model.load_state_dict(torch.load("init.pt"))
named=list(model.named_parameters())
def keyLT(n):
    m=re.match(r"blocks\.(\d+)\.",n)
    if not m: return ("EMB","EMB")
    l=int(m.group(1)); nl=n.lower()
    t=("W_Q" if "wq" in nl else "W_K" if "wk" in nl else "W_V" if "wv" in nl else
       "W_O" if ".op." in nl else "LN" if (".ln." in nl or ".n." in nl) else "FF")
    return (f"L{l}",t)
off=0; GRP={}
for n,p in named:
    k=keyLT(n); GRP.setdefault(k,[]).append((off,off+p.numel())); off+=p.numel()
P=off
IDX={k: torch.cat([torch.arange(a,b) for a,b in v]).numpy() for k,v in GRP.items()}
NEL={k:int(len(IDX[k])) for k in IDX}
FWD={"L0":16,"L1":32,"L2":64,"L3":256,"L4":512,"L5":1024,"EMB":16}
BWD={"L5":16,"L4":32,"L3":64,"L2":256,"L1":512,"L0":1024,"EMB":1024}
keys=list(IDX.keys())
def bounds(nel,nt,ov):
    nt=min(nt,nel)
    if nt<=1: return np.array([0]),np.array([nel])
    step=nel/nt; half=ov*step/2
    a=np.clip(np.floor(np.arange(nt)*step-half).astype(int),0,nel-1)
    z=np.clip(np.ceil((np.arange(nt)+1)*step+half).astype(int),1,nel)
    z=np.maximum(z,a+1)
    return a,z
BND={}
for (lk,gk) in keys:
    for side,res in (("fwd",FWD),("bwd",BWD)):
        for ov in (0.10,0.25):
            BND[(lk,gk,side,ov)]=bounds(NEL[(lk,gk)],res[lk],ov)
def seg(vals,a,z):
    cs=np.concatenate([[0.0],np.cumsum(vals,dtype=np.float64)])
    means=((cs[z]-cs[a])/(z-a)).astype(np.float32)
    av=np.abs(vals); mx=np.array([av[a[i]:z[i]].max() for i in range(len(a))],dtype=np.float32)
    return means,mx
def flat(): return torch.cat([p.data.flatten() for _,p in named]).numpy().copy()
def active(step):
    b={"LN","FF","EMB"}
    if step>=121: b|={"W_Q","W_K","W_V","W_O"}
    return b
torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
DATA={(lk,gk,side):{"mean":[],"max":[]} for (lk,gk) in keys for side in ("fwd","bwd")}
META=[]; snaps={}; LASTE=[float('nan')]; LASTOV=[0.10]; T=200
for s in range(1,T+1):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    before=flat(); opt.step(); after=flat(); upd=after-before
    if s % 8 == 0: snaps[s]=after.copy()
    if s % 8 == 0 and (s-64) in snaps and (s-32) in snaps:
        S1=snaps[s-32]-snaps[s-64]; SW=snaps[s]-snaps[s-64]
        d1=float(np.dot(S1,SW)); n1=float(np.linalg.norm(S1)*np.linalg.norm(SW))
        E=d1/(n1+1e-12)-float(np.sqrt(0.5)); LASTE[0]=E; LASTOV[0]=0.10 if E>0.05 else 0.25
    E=LASTE[0]; ov=LASTOV[0]
    ag=active(s)
    for (lk,gk) in keys:
        if gk not in ag: continue
        idx=IDX[(lk,gk)]
        for side,vals in (("fwd",after[idx]),("bwd",upd[idx])):
            a,z=BND[(lk,gk,side,ov)]; mm,mx=seg(vals,a,z)
            d=DATA[(lk,gk,side)]; d["mean"].append(mm); d["max"].append(mx)
    META.append({"step":s,"loss":float(l),"E64":E,"overlap":ov})
    del before, after, upd
    for k in list(snaps):
        if k < s-64: del snaps[k]
    if s%20==0: print(f"step {s} loss {float(l):.4f} E64 {E:+.3f} ov {ov} t{time.time()-t0:.0f}",flush=True)
out={}
for (lk,gk,side),d in DATA.items():
    if not d["mean"]: continue
    out[f"{lk}|{gk}|{side}"]={"mean":np.stack(d["mean"]),"max":np.stack(d["max"]),
        "nt":int(d["mean"][0].shape[0]),"first_step":1 if gk in ("LN","FF","EMB") else 121}
pickle.dump({"series":out,"meta":META,"P":P}, open("tiletree.pkl","wb"))
print(f"built {len(out)} series t{time.time()-t0:.0f}",flush=True)
