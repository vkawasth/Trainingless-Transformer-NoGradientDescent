"""
(a) TRAIN loss for the deterministic arms: stuck (optimisation failure) or
    memorised (generalisation failure)?
(b) WHEN do the trajectories separate?  Track val for stochastic vs deterministic
    step by step, and the angle between the two trajectories from a shared start.
(c) RESCUE: run deterministic for K steps, then switch to stochastic. How much
    of the deterministic prefix is recoverable?
"""
import json, time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
ids=torch.tensor(json.load(open("/tmp/train_ids.json")),dtype=torch.long)
SEQ=64; NW=1364//SEQ
WIN=[(ids[i*SEQ:i*SEQ+SEQ], ids[i*SEQ+1:i*SEQ+SEQ+1]) for i in range(NW)]
CH=[(torch.stack([w[0] for w in WIN[i:i+8]]), torch.stack([w[1] for w in WIN[i:i+8]]))
    for i in range(0,NW,8)]
def trainloss():
    model.eval(); L=[]
    with torch.no_grad():
        for x,y in CH: _,l=model(x,y); L.append(float(l))
    model.train(); return float(np.mean(L))
def step(o,mode):
    model.train()
    if mode=="full":
        o.zero_grad()
        for x,y in CH: _,l=model(x,y); (l/len(CH)).backward()
    else:
        x,y=get_batch(); o.zero_grad(); _,l=model(x,y); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
print("="*82); print("  (a)+(b) WHEN AND HOW DO THE TWO REGIMES SEPARATE?"); print("="*82)
print(f"  {'step':>6}{'stoch val':>11}{'stoch train':>13}{'det val':>10}{'det train':>12}{'||dtheta||':>12}")
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); oS=newopt(); S=[]
model2_sd=torch.load("init.pt")
TR={}
for tag,mode,opt_ in [("S","stoch",None),("D","full",None)]: pass
# run both, interleaved snapshots
def run_track(mode,seed=17):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(seed); o=newopt()
    rec={}
    for s in range(1,201):
        step(o,mode)
        if s in (5,10,20,30,50,80,120,200):
            rec[s]=(float(eval_val(model,n=12)), trainloss(), flat().clone())
    return rec
import pickle,os
if os.path.exists("rt.pkl"): RS,RD=pickle.load(open("rt.pkl","rb"))
else:
    RS=run_track("stoch"); print(f"    stoch done ({time.time()-t0:.0f}s)",flush=True)
    RD=run_track("full");  print(f"    det done ({time.time()-t0:.0f}s)",flush=True)
    pickle.dump((RS,RD),open("rt.pkl","wb"))
for s in sorted(RS):
    dv=float((RS[s][2]-RD[s][2]).norm())
    print(f"  {s:>6}{RS[s][0]:>11.4f}{RS[s][1]:>13.4f}{RD[s][0]:>10.4f}{RD[s][1]:>12.4f}{dv:>12.2f}")
print("\n  det train << det val  => memorised (generalisation failure)")
print("  det train ~  det val  => stuck      (optimisation failure)")
print("\n"+"="*82); print("  (c) RESCUE: K deterministic steps, then switch to stochastic"); print("="*82)
print(f"  {'K det steps':>13}{'val at switch':>15}{'val at 200':>12}{'vs pure stoch':>15}")
pure=RS[200][0]
import sys
for K in [int(a) for a in (sys.argv[1:] or ["50"])]:
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    for s in range(K): step(o,"full")
    vsw=float(eval_val(model,n=12))
    for s in range(200-K): step(o,"stoch")
    v=float(eval_val(model,n=16))
    print(f"  {K:>13}{vsw:>15.4f}{v:>12.4f}{v/pure:>14.2f}x", flush=True)
print(f"\n  time {time.time()-t0:.0f}s")
