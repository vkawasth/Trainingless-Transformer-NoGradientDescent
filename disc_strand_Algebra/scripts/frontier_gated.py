"""
MOMENTUM-GATED BACKWARD: does sign(m)!=s find the subset where exact gradient is
necessary? Train under each gating rule; compare val. Controls: random, r-gated.
Gate F_t; update = true g on F_t, persist (a*s_prev) elsewhere.
"""
import time, gc, numpy as np, torch
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
def stt(o,k):
    out=[]
    for _,p in named:
        s=o.state.get(p,{}); out.append(s[k].flatten() if k in s else torch.zeros(p.numel()))
    return torch.cat(out)
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
def V(n=16): return float(eval_val(model,n=n))
P=flat().numel()
def run(gate, frac=0.20):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    prev=flat(); s_prev=None; a_prev=0.0
    rng=torch.Generator().manual_seed(0)
    for step in range(120):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        o.zero_grad(); l.backward()
        b4=flat(); m=stt(o,"exp_avg")
        o.step(); af=flat(); u_true=af-b4
        if gate=="full" or s_prev is None:
            newth=af
        else:
            r=(stt(o,"exp_avg").abs()/(stt(o,"exp_avg_sq").sqrt()+1e-12))
            if gate=="momentum": F=(torch.sign(m)!=s_prev)
            elif gate=="random":
                F=torch.zeros(P,dtype=torch.bool); idx=torch.randperm(P,generator=rng)[:int(P*frac)]; F[idx]=True
            elif gate=="r": F=(r<torch.quantile(r[torch.randperm(P)[:100000]],frac))
            elif gate=="strand": F=torch.zeros(P,dtype=torch.bool)
            # update: exact where F, persist (reuse last sign * current mean|u|) elsewhere
            a_now=float(u_true.abs().mean())
            persist=s_prev*a_now
            u=torch.where(F,u_true,persist)
            newth=b4+u
        setflat(newth); cur=flat()
        s_prev=torch.sign(cur-b4); prev=cur; del b4,af,u_true,m
        if step%40==39: gc.collect()
    return V()
print("="*72); print("  FRONTIER-GATED BACKWARD: train under each gating rule"); print("="*72)
vb=run("full")
print(f"  {'gate':>16}{'backward frac':>15}{'val':>10}{'vs full':>10}")
print(f"  {'full':>16}{'100%':>15}{vb:>10.4f}{'1.00x':>10}")
# measure momentum frontier fraction
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
prev=flat(); sp=None; fr=[]
for step in range(30):
    model.train(); x,y=get_batch(); _,l=model(x,y); o.zero_grad(); l.backward()
    m=stt(o,"exp_avg"); b4=flat(); o.step(); af=flat()
    s=torch.sign(af-b4)
    if sp is not None: fr.append(float((torch.sign(m)!=sp).double().mean()))
    sp=s; prev=af
mfrac=np.mean(fr)
print(f"  {'momentum':>16}{100*mfrac:>13.0f}%{run('momentum'):>10.4f}{run('momentum')/vb:>9.2f}x", flush=True)
for gate,lab in [("strand","strand(0%)"),("random","random 20%"),("r","r-gated 20%")]:
    v=run(gate,0.20 if gate!="strand" else 0.0)
    fc="0%" if gate=="strand" else "20%"
    print(f"  {lab:>16}{fc:>15}{v:>10.4f}{v/vb:>9.2f}x", flush=True)
print(f"\n  momentum frontier fraction = {100*mfrac:.0f}%")
print("  if momentum << random,r at same/less backward => it finds the RIGHT subset.")
print("  if momentum ~ random => volatility not information (coupling wall).")
print(f"\n  time {time.time()-t0:.0f}s")
