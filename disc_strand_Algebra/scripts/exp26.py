import time, copy, numpy as np, torch
t0=time.time()
src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
PRE=src[:src.find("# ── PHASE 1")]
SEEDS=[17,23,31,47]; T=80; W=40; BETAS=[0.25,0.5,0.75]; CORR=10; BASE_N=40
ROWS=[]
for si,sd in enumerate(SEEDS):
    g_={}; torch.manual_seed(3000+sd); np.random.seed(3000+sd)
    exec(PRE,g_)
    model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
    named=list(model.named_parameters())
    def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
    def setflat(v):
        i=0
        with torch.no_grad():
            for _,p in named:
                k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
    def V(n=12): return float(eval_val(model,n=n))
    torch.manual_seed(sd)
    opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
    SNAP={}
    def onestep():
        model.train(); x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    for s in range(1,T+1):
        onestep()
        if s in (T-64,T-32,T-W,T): SNAP[s]=flat()
    S0=copy.deepcopy(model.state_dict()); O0=copy.deepcopy(opt.state_dict())
    S1=SNAP[T-32]-SNAP[T-64]; SW=SNAP[T]-SNAP[T-64]
    E64=float(S1@SW/(S1.norm()*SW.norm()))-float(np.sqrt(0.5))
    d=SNAP[T]-SNAP[T-W]; th=SNAP[T]
    def restore(): model.load_state_dict(copy.deepcopy(S0)); opt.load_state_dict(copy.deepcopy(O0))
    restore(); v0=V()
    restore(); torch.manual_seed(9000+sd); base=[(0,v0)]
    for k in range(1,BASE_N+1):
        onestep()
        if k<=24 or k%4==0: base.append((k,V()))
    def equiv(v):
        for k,bv in base:
            if bv<=v: return k
        return None
    res={}
    for b in BETAS:
        restore(); setflat(th+b*d); torch.manual_seed(9000+sd)
        for _ in range(CORR): onestep()
        vv=V(); e=equiv(vv); res[b]=(vv,e,(e-CORR) if e else None)
    b5=res[0.5]
    ROWS.append(dict(seed=sd,E=E64,v0=v0,vj=b5[0],eq=b5[1],net=b5[2],
                     all={b:res[b] for b in BETAS},
                     b10=dict(base)[10] if 10 in dict(base) else None))
    print(f"  seed {sd}: E(64)={E64:+.3f}  val@80={v0:.4f}  "
          f"jump+{CORR} -> {b5[0]:.4f} = {b5[1]} plain steps, net {b5[2]:+d}" if b5[2] is not None
          else f"  seed {sd}: E(64)={E64:+.3f}  val@80={v0:.4f}  jump+{CORR} -> {b5[0]:.4f} (>{BASE_N})", flush=True)
    del g_,model,opt; import gc; gc.collect()
print("\n"+"="*78); print("  SEED REPLICATION: E(64)-GATED JUMP AT t=80  (beta=0.5, 10 corrector steps)"); print("="*78)
print(f"  {'seed':>6}{'E(64)':>9}{'val@80':>9}{'10 plain':>10}{'jump+10':>10}{'plain-equiv':>13}{'net saved':>11}")
for r in ROWS:
    print(f"  {r['seed']:>6}{r['E']:>+9.3f}{r['v0']:>9.4f}"
          f"{(r['b10'] if r['b10'] else float('nan')):>10.4f}{r['vj']:>10.4f}"
          f"{(r['eq'] if r['eq'] else -1):>13}{(r['net'] if r['net'] is not None else -99):>+11}")
nets=[r['net'] for r in ROWS if r['net'] is not None]
Es=[r['E'] for r in ROWS]
print(f"\n  E(64): mean {np.mean(Es):+.3f} +/- {np.std(Es):.3f}   (gate fires if > 0.05: "
      f"{sum(1 for e in Es if e>0.05)}/{len(Es)} seeds)")
if nets:
    print(f"  net steps saved: {nets}   mean {np.mean(nets):+.1f} +/- {np.std(nets):.1f}"
          f"   positive in {sum(1 for n in nets if n>0)}/{len(nets)} seeds")
print(f"\n  beta sweep (net steps saved):")
print(f"  {'seed':>6}" + "".join(f"{'b='+str(b):>10}" for b in BETAS))
for r in ROWS:
    print(f"  {r['seed']:>6}" + "".join(
        f"{(r['all'][b][2] if r['all'][b][2] is not None else -99):>+10}" for b in BETAS))
print("\n  time %.0fs"%(time.time()-t0))
