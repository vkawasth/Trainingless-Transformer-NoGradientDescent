"""
GEOMETRY-GUIDED FLOW WITH FULL COVERAGE.
Fixed-batch gave smooth motion (cos_consec 0.996) but val 6.81 -- because one
batch covers only 512 of the 1364 base tokens.
Arm C: DETERMINISTIC FULL COVERAGE -- accumulate the gradient over every window
of the base sentence, so the step is smooth AND sees the whole corpus.
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
BASE=1364; SEQ=64; NW=BASE//SEQ
WIN=[(ids[i*SEQ:i*SEQ+SEQ], ids[i*SEQ+1:i*SEQ+SEQ+1]) for i in range(NW)]
print(f"  full coverage: {NW} windows x {SEQ} = {NW*SEQ} of {BASE} base tokens", flush=True)
CH=[(torch.stack([w[0] for w in WIN[i:i+8]]), torch.stack([w[1] for w in WIN[i:i+8]]))
    for i in range(0,NW,8)]
torch.manual_seed(99); FIXED=get_batch()
def gradstep(o, mode):
    model.train()
    if mode=="full":
        o.zero_grad()
        for x,y in CH:
            _,l=model(x,y); (l/len(CH)).backward()
    else:
        x,y = FIXED if mode=="fixed" else get_batch()
        o.zero_grad(); _,l=model(x,y); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
def run(mode):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    th0=flat()
    for s in range(200): gradstep(o,mode)
    return flat()-th0, float(eval_val(model,n=16))
def measure(mode, Dg):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    thF=flat()+Dg; net=torch.zeros_like(thF); path=torch.zeros_like(thF)
    tot=0.0; cs=[]; c1=[]; dp=None
    for s in range(200):
        b4=flat(); gradstep(o,mode); af=flat(); d=af-b4
        nd=float(d.norm()); tot+=nd; net+=d; path+=d.abs()
        rem=thF-b4; cs.append(float(d@rem)/max(nd*float(rem.norm()),1e-30))
        if dp is not None: c1.append(float(d@dp)/max(nd*float(dp.norm()),1e-30))
        dp=d.clone(); del b4,af,d
        if s%50==49: gc.collect()
    return dict(cosf=np.mean(cs), cons=np.mean(c1), l1=float(net.abs().sum()/path.sum()),
                cp=float(Dg.norm())/tot, path=tot)
R={}
import sys
_A={"stoch":"stochastic (baseline)","fixed":"fixed single batch","full":"deterministic FULL coverage"}
for mode in (sys.argv[1:] or ["full"]):
    lab=_A[mode]
    Dg,v=run(mode); m=measure(mode,Dg); m["val"]=v; m["chord"]=float(Dg.norm()); R[lab]=m
    print(f"  {lab:<30} val {v:.4f}  ({time.time()-t0:.0f}s)", flush=True)
print("\n"+"="*86); print("  GEOMETRY-GUIDED FLOW: does full coverage buy smoothness AND accuracy?"); print("="*86)
print(f"  {'arm':>30}{'val':>9}{'cos(d,to-final)':>17}{'cos(d_t,d_t+1)':>16}"
      f"{'net/path':>10}{'path':>9}")
for lab,m in R.items():
    print(f"  {lab:>30}{m['val']:>9.4f}{m['cosf']:>17.3f}{m['cons']:>16.3f}"
          f"{m['l1']:>10.3f}{m['path']:>9.1f}")
b=R.get("stochastic (baseline)", list(R.values())[0])
print(f"\n  vs baseline:")
for lab,m in R.items():
    if m is b: continue
    print(f"    {lab:<32} val {m['val']/b['val']:>6.2f}x   path {m['path']/b['path']:>5.2f}x"
          f"   alignment {m['cosf']/b['cosf']:>5.2f}x")
print(f"\n  time {time.time()-t0:.0f}s")
