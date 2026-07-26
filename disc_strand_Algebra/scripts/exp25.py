import time, re, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
LO,HI=100,116; TH={}
for s in range(1,HI+1):
    model.train(); x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if s>=LO: TH[s]=flat()
print(f"  {len(TH)} checkpoints at 1-step spacing ({time.time()-t0:.0f}s)", flush=True)
torch.manual_seed(3); XB=[get_batch() for _ in range(2)]
model=model.double(); named=list(model.named_parameters())
def keyL(n):
    m=re.match(r"blocks\.(\d+)\.",n); return f"L{m.group(1)}" if m else "EMB"
ks=[];sl={};off=0
for n,p in named:
    k=keyL(n)
    if k not in sl: sl[k]=[];ks.append(k)
    sl[k].append((off,off+p.numel())); off+=p.numel()
IDX=[torch.cat([torch.arange(a,b) for a,b in sl[k]]) for k in ks]; K=len(ks)
NB=[len(I) for I in IDX]
def setflat(v):
    i=0
    with torch.no_grad():
        for _,p in named:
            k=p.numel(); p.data.copy_(v[i:i+k].view_as(p)); i+=k
def G(v):
    setflat(v); model.train(); model.zero_grad(set_to_none=True)
    for x,y in XB:
        _,l=model(x,y); (l/len(XB)).backward()
    g=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(),dtype=torch.float64))
                 for _,p in named]).clone()
    model.zero_grad(set_to_none=True); return g
H=1e-4; U={}; GAP={}
for s in sorted(TH):
    th=TH[s].double(); g0=G(th); nV=[float(g0[I].norm()) for I in IDX]
    A=[]
    for m in range(K):
        d=torch.zeros_like(th); d[IDX[m]]=-g0[IDX[m]]/max(nV[m],1e-30)
        A.append((G(th+H*d)-g0)/H)
    us=[]; gaps=[]
    for l in range(K):
        Mt=torch.stack([A[m][IDX[l]] for m in range(K)])
        _,S,Vt=torch.linalg.svd(Mt,full_matrices=False)
        us.append(Vt[0].float()); gaps.append(float(S[1]/S[0]))
    U[s]=us; GAP[s]=np.mean(gaps)
    del A,g0,th; gc.collect()
print(f"  u_l computed at every step ({time.time()-t0:.0f}s)", flush=True)
SS=sorted(TH); NULL=[1.0/np.sqrt(n) for n in NB]
print("\n"+"="*78)
print("  PROJECTION PATH:  ||e_l(t) - e_l(t+D)|| = sin(angle),  criterion for a")
print("  well-defined K_0 class along the path is  ||e-f|| < 1")
print("="*78)
print(f"  {'lag D':>7}{'n':>4}" + "".join(f"{k:>9}" for k in ks))
print(f"  {'':>11}" + "".join(f"{'cos/||e-f||':>9}" for _ in ks[:1]) + "  (cos on first line, ||e-f|| on second)")
for D in [1,2,4,8,16]:
    prs=[(a,a+D) for a in SS if a+D in U]
    if not prs: continue
    cs=[[abs(float(U[a][l].double()@U[b][l].double())) for a,b in prs] for l in range(K)]
    mc=[np.mean(c) for c in cs]
    print(f"  {D:>7}{len(prs):>4}" + "".join(f"{x:>9.3f}" for x in mc))
    print(f"  {'  ||e-f||':>11}" + "".join(f"{np.sqrt(max(1-x*x,0)):>9.4f}" for x in mc))
print(f"\n  null cos for random directions per block: "
      + " ".join(f"{ks[l]}:{NULL[l]:.4f}" for l in range(K)))
print(f"  mean singular gap S2/S1 over the window: {np.mean([GAP[s] for s in SS]):.4f}"
      f"  (small gap => u_l well separated, path meaningful)")
print("\n  READING: if ||e-f|| falls well below 1 as D->0 the projections form a")
print("  continuous path and the K-class is defined; if it stays ~1 at every")
print("  spacing the path is discontinuous at all scales.")
print("\n  time %.0fs"%(time.time()-t0))
