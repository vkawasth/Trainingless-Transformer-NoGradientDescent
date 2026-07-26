"""
HOW MUCH OF THE NET 1->200 DISPLACEMENT ACTUALLY MATTERS?
 net  = theta_200 - theta_0        path = sum_t |theta_t - theta_{t-1}|
Truncation test: theta_0 + (top-k% of net, rest zeroed) -> val loss.
Report the fraction of the full loss improvement retained.
Granularities: per parameter, per tile (backward ladder), and a random control.
"""
import re, time, copy, numpy as np, torch
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
def V(n=16): return float(eval_val(model,n=n))
model.load_state_dict(torch.load("init.pt")); th0=flat(); torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
setflat(th0); v_init=V()
path=torch.zeros_like(th0); prev=th0.clone()
for s in range(200):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    cur=flat(); path+=(cur-prev).abs(); prev=cur
th200=flat(); v_full=V(); D=th200-th0; P=D.numel()
print("="*82); print("  NET DISPLACEMENT, STEPS 1-200"); print("="*82)
print(f"  val: init {v_init:.4f} -> final {v_full:.4f}   improvement {v_init-v_full:.4f}")
print(f"  ||net||_2 = {float(D.norm()):.2f}   sum|net| = {float(D.abs().sum()):.1f}"
      f"   path = {float(path.sum()):.1f}")
print(f"  cancellation: {100*(1-float(D.abs().sum()/path.sum())):.1f}% of motion undoes itself")
a=D.abs(); srt,_=torch.sort(a,descending=True); cum=torch.cumsum(srt,0)/srt.sum()
cum2=torch.cumsum(srt**2,0)/ (srt**2).sum()
print(f"\n  CONCENTRATION of |net|")
print(f"  {'top k%':>9}{'params':>12}{'% of sum|net|':>15}{'% of ||net||^2':>16}")
for k in (1,5,10,25,50):
    i=int(P*k/100)-1
    print(f"  {k:>8}%{i+1:>12,}{100*float(cum[i]):>14.1f}%{100*float(cum2[i]):>15.1f}%")
def frac(v): return 100*(v_init-v)/max(v_init-v_full,1e-12)
print("\n"+"="*82); print("  TRUNCATION TEST: theta_0 + top-k% of net (rest zeroed)"); print("="*82)
print(f"  {'top k%':>8}{'val (per-param)':>17}{'% kept':>9}{'val (random k%)':>18}{'% kept':>9}")
rng=torch.Generator().manual_seed(3)
for k in (1,5,10,25,50,75,100):
    thr=srt[int(P*k/100)-1]
    m=(a>=thr).float()
    setflat(th0+D*m); v1=V()
    perm=torch.randperm(P,generator=rng)[:int(P*k/100)]
    m2=torch.zeros_like(D); m2[perm]=1.0
    setflat(th0+D*m2); v2=V()
    print(f"  {k:>7}%{v1:>17.4f}{frac(v1):>8.1f}%{v2:>18.4f}{frac(v2):>8.1f}%", flush=True)
# ---- tile granularity ----
LAY=["L0","L1","L2","L3","L4","L5"]; BWD=[1024,512,256,64,32,16]
off=0; SPAN={}
for n,p in named: SPAN[n]=(off,off+p.numel()); off+=p.numel()
def lay(n):
    m=re.match(r"blocks\.(\d+)\.",n); return f"L{m.group(1)}" if m else None
GI={}
for n,_ in named:
    L=lay(n)
    if L: GI.setdefault(L,[]).append(torch.arange(*SPAN[n]))
GI={k:torch.cat(v) for k,v in GI.items()}
tiles=[]
for L,nt in zip(LAY,BWD):
    idx=GI[L]; nel=len(idx); e=np.linspace(0,nel,nt+1).astype(int)
    for i in range(nt): tiles.append(idx[e[i]:e[i+1]])
mass=torch.tensor([float(D[t].abs().sum()) for t in tiles])
order=torch.argsort(mass,descending=True); cm=torch.cumsum(mass[order],0)/mass.sum()
print(f"\n  tiles: {len(tiles)}  (backward ladder, all 6 layers)")
print("="*82); print("  TRUNCATION TEST AT TILE GRANULARITY"); print("="*82)
print(f"  {'top k% tiles':>14}{'tiles':>8}{'% of |net|':>12}{'val':>10}{'% kept':>9}")
for k in (1,5,10,25,50,100):
    nk=max(1,int(len(tiles)*k/100)); keep=order[:nk]
    m=torch.zeros_like(D)
    for j in keep: m[tiles[j]]=1.0
    setflat(th0+D*m); v=V()
    print(f"  {k:>13}%{nk:>8}{100*float(cm[nk-1]):>11.1f}%{v:>10.4f}{frac(v):>8.1f}%", flush=True)
setflat(th0)
print(f"\n  time {time.time()-t0:.0f}s")
