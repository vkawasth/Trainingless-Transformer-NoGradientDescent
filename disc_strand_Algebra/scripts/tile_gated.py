"""
TILE-GATED BACKWARD: gate whole NEURONS (coupling-respecting units), not coords.
A neuron is computed exact (whole row) or persisted whole. Tests whether keeping
coupled coordinates together rescues gated backward.
Gates: momentum (tile flagged if its mean |sign(m)!=s| high), r (low mean-r tiles),
random, all at matched backward fraction. Metric: val after training.
"""
import re, time, gc, numpy as np, torch
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
# neuron tiling: each 2-D weight row = one tile; 1-D params = one tile each
off=0; TILE=torch.zeros(P,dtype=torch.long); tid=0; SPAN={}
for n,p in named:
    a=off; b=off+p.numel(); SPAN[n]=(a,b)
    if p.dim()==2:
        R,C=p.shape
        for r_ in range(R): TILE[a+r_*C:a+(r_+1)*C]=tid; tid+=1
    else:
        TILE[a:b]=tid; tid+=1
    off=b
NT=tid; cnt=torch.bincount(TILE,minlength=NT).float()
print(f"  {NT} neuron-tiles over {P} params ({time.time()-t0:.0f}s)", flush=True)
def run(gate, frac=0.20):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    prev=flat(); s_prev=None; rng=torch.Generator().manual_seed(0)
    for step in range(120):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        o.zero_grad(); l.backward(); b4=flat(); m=stt(o,"exp_avg")
        o.step(); af=flat(); u_true=af-b4
        if gate=="full" or s_prev is None:
            setflat(af)
        else:
            r=(stt(o,"exp_avg").abs()/(stt(o,"exp_avg_sq").sqrt()+1e-12))
            # tile scores
            if gate=="momentum":
                dis=(torch.sign(m)!=s_prev).float()
                ts=torch.zeros(NT).index_add_(0,TILE,dis)/cnt      # frac disagreeing per tile
                thr=torch.quantile(ts,1-frac); tsel=(ts>=thr)
            elif gate=="r":
                tr=torch.zeros(NT).index_add_(0,TILE,r)/cnt
                thr=torch.quantile(tr,frac); tsel=(tr<thr)         # low-r tiles
            elif gate=="random":
                tsel=torch.zeros(NT,dtype=torch.bool)
                tsel[torch.randperm(NT,generator=rng)[:int(NT*frac)]]=True
            F=tsel[TILE]
            a_now=float(u_true.abs().mean())
            u=torch.where(F,u_true,s_prev*a_now)
            setflat(b4+u)
        cur=flat(); s_prev=torch.sign(cur-b4); prev=cur; del b4,af,u_true,m
        if step%40==39: gc.collect()
    return V()
print("="*70); print("  TILE(NEURON)-GATED BACKWARD, 20% of tiles"); print("="*70)
vb=run("full")
print(f"  {'gate':>16}{'val':>10}{'vs full':>10}")
print(f"  {'full (100%)':>16}{vb:>10.4f}{'1.00x':>10}")
for g in ["momentum","r","random"]:
    v=run(g,0.20); print(f"  {g+' 20%':>16}{v:>10.4f}{v/vb:>9.2f}x", flush=True)
print("\n  compare to COORDINATE gating (prev): random 3.2x, r-gated 6.9x.")
print("  if tile-momentum << those, coupling-respecting gating helps.")
print("  if all ~ same, coupling is diffuse across tiles too (dense-graph result).")
print(f"\n  time {time.time()-t0:.0f}s")
