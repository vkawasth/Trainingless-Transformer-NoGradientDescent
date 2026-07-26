import time, copy, gc, numpy as np, torch
t0=time.time()
src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
PRE=src[:src.find("# ── PHASE 1")]
SEEDS=[17,23,31,47,59]; T=80; W=40; CORR=10; BASE_N=32
print("seed,E64,val80,val_10plain,val_jump,plain_equiv,net", flush=True)
for sd in SEEDS:
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
    def V(): return float(eval_val(model,n=8))
    torch.manual_seed(sd)
    opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
    def onestep():
        model.train(); x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    SNAP={}
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
        if k in (2,4,6,8,10,12,14,16,20,24,28,32): base.append((k,V()))
    v10=dict(base).get(10,float('nan'))
    restore(); setflat(th+0.5*d); torch.manual_seed(9000+sd)
    for _ in range(CORR): onestep()
    vj=V()
    eq=next((k for k,bv in base if bv<=vj), None)
    net = (eq-CORR) if eq is not None else None
    print(f"{sd},{E64:+.4f},{v0:.4f},{v10:.4f},{vj:.4f},{eq if eq else -1},"
          f"{net if net is not None else -99}", flush=True)
    del g_,model,opt; gc.collect()
print(f"# total {time.time()-t0:.0f}s", flush=True)
