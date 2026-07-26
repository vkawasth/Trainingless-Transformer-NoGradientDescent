"""
REPRESENT EACH STEP BY ONE CONSTANT PER STRAND.
Partition all 4,330,240 parameters into T contiguous chunks ("strands").
At every step replace the applied update d by a quantised version:
  A) sign(d) * c_chunk        per-parameter sign, one magnitude per chunk
  B) uniform  c_chunk         one signed constant per chunk (no per-param sign)
c = mean|d|, max|d|, or signed mean.  Compression = 4.33M -> T numbers per step.
"""
import sys, time, numpy as np, torch
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
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
model.load_state_dict(torch.load("init.pt")); P=flat().numel(); v0=V()
def run(T=None, mode="none"):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
    o=newopt(); prev=flat()
    if T:
        e=np.linspace(0,P,T+1).astype(int)
        seg=torch.zeros(P,dtype=torch.long)
        for i in range(T): seg[e[i]:e[i+1]]=i
        cnt=torch.bincount(seg,minlength=T).double()
    for s in range(200):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
        cur=flat(); d=cur-prev
        if mode=="none": d2=d
        else:
            if mode.startswith("sgn"):
                stat=torch.zeros(T,dtype=torch.float64)
                if "max" in mode:
                    stat=torch.zeros(T).scatter_reduce(0,seg,d.abs(),reduce="amax",include_self=False).double()
                else:
                    stat=(torch.zeros(T,dtype=torch.float64).index_add_(0,seg,d.abs().double())/cnt)
                d2=torch.sign(d)*stat[seg].float()
            else:  # uniform signed constant per chunk
                stat=(torch.zeros(T,dtype=torch.float64).index_add_(0,seg,d.double())/cnt)
                d2=stat[seg].float()
        setflat(prev+d2); prev=flat()
    return V()
vb=run(); print(f"  baseline  val {vb:.4f}   (init {v0:.4f})   [{time.time()-t0:.0f}s]", flush=True)
pct=lambda v: 100*(v0-v)/max(v0-vb,1e-12)
print("="*84); print("  ONE CONSTANT PER STRAND PER STEP"); print("="*84)
print(f"  {'T (chunks)':>11}{'numbers/step':>14}{'compression':>13}{'scheme':>22}{'val':>9}{'% kept':>9}")
import sys
ARMS=[a.split(":") for a in sys.argv[1:]] or [["1024","sgn_mean"]]
LAB={"sgn_mean":"sign(d) x mean|d|","sgn_max":"sign(d) x max|d|","uniform":"uniform signed const"}
for Ts,mode in ARMS:
    if True:
        T=int(Ts); lab=LAB[mode]
        v=run(T,mode)
        print(f"  {T:>11}{T:>14}{P//T:>12}x{lab:>22}{v:>9.4f}{pct(v):>8.1f}%", flush=True)
print(f"\n  time {time.time()-t0:.0f}s")
