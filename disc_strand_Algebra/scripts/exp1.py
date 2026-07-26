import time, json, math, copy, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
model.load_state_dict(torch.load("/home/claude/w/init.pt"))
named=list(model.named_parameters())
def grp(n):
    n=n.lower()
    if ".ln." in n or ".n." in n or n.startswith("ln_f"): return "LN"
    if n.startswith("te") or n.startswith("pe") or n.startswith("head"): return "EMB"
    if ".ff." in n: return "FF"
    if "wk" in n: return "W_K"
    if "wq" in n: return "W_Q"
    if "wv" in n: return "W_V"
    if ".op." in n: return "W_O"
    return "other"
GR=["EMB","LN","FF","W_Q","W_K","W_V","W_O"]
sl={}; off=0
for n,p in named:
    sl.setdefault(grp(n),[]).append((off,off+p.numel())); off+=p.numel()
P=off
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def blkvec(v,g): return torch.cat([v[a:b] for a,b in sl[g]])

torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
T=200
prev=flat(); dprev=None; Tprev=None
hist=[]; DHIST={}   # store some displacement dirs for lag-cosines
KEEP=set(range(0,T,1))
recent=[]
for s in range(1,T+1):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward()
    gn=float(torch.cat([p.grad.flatten() for _,p in named]).norm())
    prof={}
    for n,p in named:
        if p.grad is not None:
            gg=grp(n); prof[gg]=prof.get(gg,0.0)+float(p.grad.pow(2).sum())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    cur=flat(); d=cur-prev; prev=cur
    nd=float(d.norm()); Tv=d/(nd+1e-12)
    cos1=float(Tv@Tprev) if Tprev is not None else float('nan')
    kap=float((Tv-Tprev).norm()/(0.5*(nd+ndprev))) if Tprev is not None else float('nan')
    rec={"s":s,"loss":float(l),"gn":gn,"nd":nd,"cos1":cos1,"kap":kap,
         "theta":float(cur.norm())}
    pv=np.array([math.sqrt(prof.get(k,0.0)) for k in GR]); pv=pv/(pv.sum()+1e-12)
    rec["prof"]=pv.tolist()
    # per-block step share
    rec["blk_nd"]={g: float(blkvec(d,g).norm()) for g in GR}
    if s%5==0 or s<=10: rec["val"]=float(eval_val(model,n=8))
    hist.append(rec)
    recent.append(Tv.clone())
    if len(recent)>65: recent.pop(0)
    if s%8==0 and len(recent)>=65:
        rec["lag"]={str(k): float(recent[-1]@recent[-1-k]) for k in (1,4,8,16,32,64)}
    Tprev=Tv; ndprev=nd
    if s in (40,120,200):
        torch.save({"sd":copy.deepcopy(model.state_dict()),
                    "od":copy.deepcopy(opt.state_dict())}, f"/home/claude/w/ck{s}.pt")
    if s%25==0: print(f"  step {s:>3} loss {float(l):.4f} val {rec.get('val',float('nan')):.4f} |d| {nd:.4f} cos1 {cos1:.3f}", flush=True)
json.dump(hist, open("/home/claude/w/hist.json","w"))
print("done %.1fs"%(time.time()-t0))
