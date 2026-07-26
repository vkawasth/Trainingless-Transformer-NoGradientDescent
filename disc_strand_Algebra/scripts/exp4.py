import time, copy, json, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def V(n=12): return float(eval_val(model,n=n))
W=20   # window
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
buf={}; EX=[]; CKS={}
def onestep():
    model.train(); x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
print("="*72); print("  ONLINE PERSISTENCE EXCESS   E = cos(S_half, S_win) - sqrt(1/2)"); print("="*72)
print(f"  window W={W}; null for a driftless walk is 0.707 -> E=0")
print(f"\n  {'step':>6}{'val':>9}{'cos':>8}{'E':>8}{'|d_win|':>10}{'verdict':>14}")
for s in range(1,201):
    onestep()
    if s%(W//2)==0: buf[s]=flat()
    if s>=W and s%(W//2)==0:
        a=buf.get(s-W); b=buf.get(s-W//2); c=buf.get(s)
        if a is not None:
            S1=b-a; SW=c-a
            cos=float(S1@SW/(S1.norm()*SW.norm()+1e-12)); E=cos-np.sqrt(0.5)
            EX.append((s,cos,E,float(SW.norm())))
            if s%20==0:
                v=V(n=8)
                print(f"  {s:>6}{v:>9.4f}{cos:>8.3f}{E:>+8.3f}{float(SW.norm()):>10.3f}"
                      f"{'DRIFT' if E>0.06 else 'walk':>14}", flush=True)
        for k in list(buf):
            if k < s-W: del buf[k]
    if s in (40,60,80,100,120,140,160,180):
        CKS[s]=(copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), flat().clone(), buf.get(s-W).clone() if buf.get(s-W) is not None else None)
json.dump([(a,b,c,d) for a,b,c,d in EX], open("EX.json","w"))
torch.save({k:(v[2],v[3]) for k,v in CKS.items()}, "ckvecs.pt")

print("\n"+"="*72); print("  JUMP-WITH-LIFT: theta += beta*(theta_t - theta_{t-20}), then k corrector steps"); print("="*72)
print("  'steps saved' = plain steps from t needed to match the jumped val, minus k")
res={}
for t in [40,60,80,100,120,140,160,180]:
    sd,od,th,thw = CKS[t]
    if thw is None: continue
    d = th-thw
    def restore(): model.load_state_dict(copy.deepcopy(sd)); opt.load_state_dict(copy.deepcopy(od))
    restore(); v0=V()
    # baseline: plain steps
    base=[]; restore(); torch.manual_seed(1000+t)
    for k in range(1,41):
        onestep()
        if k%5==0: base.append((k,V()))
    def plain_steps_to(target):
        for k,v in base:
            if v<=target: return k
        return 99
    Emeas=[e for (s,c,e,n) in EX if s==t]
    Em=Emeas[0] if Emeas else float('nan')
    best=None
    for beta in [0.5,1.0,2.0]:
        for corr in [0,10]:
            restore(); setflat(th+beta*d)
            if corr:
                torch.manual_seed(1000+t)
                for _ in range(corr): onestep()
            v=V(); need=plain_steps_to(v); saved=need-corr
            if best is None or saved>best[0]: best=(saved,beta,corr,v,need)
    print(f"  t={t:>4} E={Em:+.3f} val={v0:.4f} | best: beta={best[1]}, corr={best[2]}, "
          f"val={best[3]:.4f} = {best[4]} plain steps -> SAVED {best[0]:+d} steps", flush=True)
    res[t]=dict(E=Em,v0=v0,saved=best[0],beta=best[1],corr=best[2],val=best[3])
json.dump(res,open("jump.json","w"))
print("\n  time %.1fs"%(time.time()-t0))
