import time, copy, json, itertools, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def grp(n):
    n=n.lower()
    if ".ln." in n or ".n." in n or n.startswith("ln_f"): return "LN"
    if n.startswith("te") or n.startswith("pe") or n.startswith("head"): return "EMB"
    if ".ff." in n: return "FF"
    if "wk" in n: return "W_K"
    if "wq" in n: return "W_Q"
    if "wv" in n: return "W_V"
    if ".op." in n: return "W_O"
    return "other"
NODES=["EMB","LN","FF","W_Q","W_K","W_V","W_O"]
sl={g:[] for g in NODES}; off=0
for n,p in named:
    sl[grp(n)].append((off,off+p.numel())); off+=p.numel()
P=off
IDX={g: torch.cat([torch.arange(a,b) for a,b in sl[g]]) for g in NODES}
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def sub(v,gs):
    m=torch.zeros_like(v)
    for g in gs:
        for a,b in sl[g]: m[a:b]=v[a:b]
    return m
def V(n=12): return float(eval_val(model,n=n))

# ---- count sketch per node (m buckets) ----
M=512; rg=np.random.default_rng(0)
HSH={g: torch.tensor(rg.integers(0,M,size=len(IDX[g])),dtype=torch.long) for g in NODES}
SGN={g: torch.tensor(rg.choice([-1.0,1.0],size=len(IDX[g])),dtype=torch.float32) for g in NODES}
def sketch(d):
    out={}
    for g in NODES:
        v=d[IDX[g]]*SGN[g]
        b=torch.zeros(M); b.index_add_(0,HSH[g],v); out[g]=b.numpy().copy()
    return out

model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
for _ in range(120):
    model.train(); x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
th120=flat(); S120=copy.deepcopy(model.state_dict()); O120=copy.deepcopy(opt.state_dict())
v120=V(n=16)
SK={g:[] for g in NODES}; prev=th120.clone(); path=torch.zeros_like(prev)
torch.manual_seed(11)
for s in range(80):
    model.train(); x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    cur=flat(); d=cur-prev; path+=d.abs(); prev=cur
    sk=sketch(d)
    for g in NODES: SK[g].append(sk[g])
th200=flat(); v200=V(n=16); DT=th200-th120
print("="*76); print("  SEGMENT 120 -> 200 : DENSE-INTERACTION REGIME"); print("="*76)
print(f"  val(120)={v120:.4f}  val(200)={v200:.4f}  ||Delta||={float(DT.norm()):.3f}"
      f"  = {float(DT.norm()/th120.norm()):.3f}*||theta||")
print(f"  per-parameter cancellation over segment: {100*(1-float(DT.abs().sum()/path.sum())):.1f}%")
np.save("SK.npy", np.array([np.stack(SK[g]) for g in NODES]))   # (7,80,M)
torch.save({"DT":DT,"th120":th120}, "seg.pt")

print("\n  NODE DECOMPOSITION")
print(f"  {'node':>6}{'params':>11}{'%par':>7}{'||d||':>9}{'%energy':>9}")
tot=float(DT.norm())**2
for g in NODES:
    dv=sub(DT,[g]); npar=len(IDX[g])
    print(f"  {g:>6}{npar:>11,}{100*npar/P:>6.1f}%{float(dv.norm()):>9.3f}{100*float(dv.norm())**2/tot:>8.1f}%")

def restore(): model.load_state_dict(copy.deepcopy(S120)); opt.load_state_dict(copy.deepcopy(O120))
def jump(gs, corr=0):
    restore(); setflat(th120+sub(DT,gs)); v=V()
    if corr:
        torch.manual_seed(11)
        for _ in range(corr):
            model.train(); x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        v=V()
    return v
den=(v120-v200)
rec=lambda v: 100*(v120-v)/den
R1={}
print("\n  SINGLE-NODE ORACLE JUMPS (recovery of the 120->200 progress)")
for g in NODES:
    R1[g]=rec(jump([g])); print(f"    {g:>6}  val {jump([g]):.4f}   rec {R1[g]:>6.1f}%", flush=True)
print(f"    {'ALL':>6}  rec {rec(jump(NODES)):>6.1f}%   (sanity: should be 100)")
print("\n  PAIRWISE SYNERGY  S_ij = rec(i,j) - rec(i) - rec(j)   [edge weights]")
SYN=np.zeros((7,7))
for i,j in itertools.combinations(range(7),2):
    a,b=NODES[i],NODES[j]; rij=rec(jump([a,b])); s=rij-R1[a]-R1[b]
    SYN[i,j]=SYN[j,i]=s
print("        "+"".join(f"{g:>8}" for g in NODES))
for i,g in enumerate(NODES):
    print(f"  {g:>6}"+"".join(("      . " if i==j else f"{SYN[i,j]:>8.1f}") for j in range(7)))
np.save("SYN.npy",SYN); json.dump(R1,open("R1.json","w"))
print("\n  time %.1fs"%(time.time()-t0))
