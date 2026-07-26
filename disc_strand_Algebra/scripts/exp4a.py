import time, copy, json, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def V(n=8): return float(eval_val(model,n=n))
W=20
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
def onestep():
    model.train(); x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
buf={}; EX=[]
for s in range(1,201):
    onestep()
    if s%(W//2)==0: buf[s]=flat()
    if s>=W and s%(W//2)==0:
        a=buf.get(s-W); b=buf.get(s-W//2); c=buf.get(s)
        if a is not None:
            S1=b-a; SW=c-a
            cos=float(S1@SW/(S1.norm()*SW.norm()+1e-12))
            EX.append((s,cos,cos-float(np.sqrt(0.5)),float(SW.norm()),V() if s%20==0 else None))
        for k in list(buf):
            if k<s-W: del buf[k]
    if s in (40,80,120,160):
        torch.save({"sd":model.state_dict(),"od":opt.state_dict(),
                    "th":flat(),"thw":buf[s-W]}, f"J{s}.pt")
json.dump(EX,open("EX.json","w"))
print("="*70); print("  ONLINE PERSISTENCE EXCESS  E = cos(S_10, S_20) - 0.707"); print("="*70)
print(f"  {'step':>6}{'val':>9}{'cos':>8}{'E':>9}{'|d_win|':>10}  verdict")
for s,c,e,n,v in EX:
    if s%20==0:
        print(f"  {s:>6}{v:>9.4f}{c:>8.3f}{e:>+9.3f}{n:>10.3f}  {'DRIFT' if e>0.06 else 'walk'}")
print("\n  time %.1fs"%(time.time()-t0))
