import time, copy, numpy as np, torch
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
CK={t: torch.load(f"J{t}.pt") for t in (40,80,120,160)}
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
def onestep():
    model.train(); x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
print("="*76); print("  JUMP-WITH-LIFT, W=40 window, gated by E(64)"); print("="*76)
print("  jump: theta_t += beta*(theta_t - theta_{t-40});  then k corrector steps")
print("  'plain-equiv' = number of ordinary steps from t reaching the same val\n")
E64={80:"+0.16 (DRIFT)",120:"~+0.09",160:"+0.01 (walk)"}
for t,tp in [(80,40),(120,80),(160,120)]:
    ck=CK[t]; th=ck["th"]; d=th-CK[tp]["th"]
    def restore():
        model.load_state_dict(copy.deepcopy(ck["sd"])); opt.load_state_dict(copy.deepcopy(ck["od"]))
    restore(); v0=V()
    base=[(0,v0)]; restore(); torch.manual_seed(1000+t)
    for k in range(1,41):
        onestep()
        if k%5==0: base.append((k,V()))
    def equiv(v):
        for k,bv in base:
            if bv<=v: return k
        return 99
    print(f"  --- t={t}  val={v0:.4f}   E(64)={E64[t]}   ||d_40||={float(d.norm()):.2f}")
    print(f"      {'beta':>6}{'corr':>6}{'val':>10}{'plain-equiv':>13}{'net saved':>11}")
    for beta in [0.5,1.0,2.0]:
        for corr in [0,10]:
            restore(); setflat(th+beta*d)
            if corr:
                torch.manual_seed(1000+t)
                for _ in range(corr): onestep()
            v=V(); e=equiv(v)
            print(f"      {beta:>6.1f}{corr:>6}{v:>10.4f}{e:>13}{e-corr:>+11}", flush=True)
    print(f"      baseline curve: " + "  ".join(f"{k}:{bv:.4f}" for k,bv in base))
    print()
print("  time %.1fs"%(time.time()-t0))
