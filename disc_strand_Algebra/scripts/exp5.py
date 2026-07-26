import time, copy, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
NODES=["EMB","LN","FF","W_Q","W_K","W_V","W_O"]
def grp(n):
    n=n.lower()
    if ".ln." in n or ".n." in n or n.startswith("ln_f"): return "LN"
    if n.startswith("te") or n.startswith("pe") or n.startswith("head"): return "EMB"
    if ".ff." in n: return "FF"
    if "wk" in n: return "W_K"
    if "wq" in n: return "W_Q"
    if "wv" in n: return "W_V"
    if ".op." in n: return "W_O"
sl={g:[] for g in NODES}; off=0
for n,p in named: sl[grp(n)].append((off,off+p.numel())); off+=p.numel()
IDX={g: torch.cat([torch.arange(a,b) for a,b in sl[g]]) for g in NODES}
M=512; rg=np.random.default_rng(0)
HSH={g: torch.tensor(rg.integers(0,M,size=len(IDX[g])),dtype=torch.long) for g in NODES}
SGN={g: torch.tensor(rg.choice([-1.0,1.0],size=len(IDX[g])),dtype=torch.float32) for g in NODES}
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def sk(d):
    out=[]
    for g in NODES:
        v=d[IDX[g]]*SGN[g]; b=torch.zeros(M); b.index_add_(0,HSH[g],v); out.append(b.numpy())
    return np.concatenate(out)
ck=torch.load("J40.pt")
model.load_state_dict(ck["sd"])
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
opt.load_state_dict(ck["od"]); torch.manual_seed(11)
prev=flat(); Gs=[]
for _ in range(80):
    model.train(); x,y=get_batch(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    cur=flat(); Gs.append(sk(cur-prev)); prev=cur
np.save("SK_40_120.npy", np.stack(Gs))
print("collected 40-120 in %.1fs"%(time.time()-t0))
def Ecurve(G,name):
    T=G.shape[0]; out=[]
    for W in [4,8,16,32,64]:
        vals=[]
        for st in range(0,T-W+1,2):
            S1=G[st:st+W//2].sum(0); SW=G[st:st+W].sum(0)
            vals.append(float(S1@SW/(np.linalg.norm(S1)*np.linalg.norm(SW)+1e-12)))
        out.append((W,np.mean(vals),np.std(vals)/np.sqrt(len(vals))))
    print(f"\n  {name}")
    print(f"  {'W':>5}{'cos(S_W/2,S_W)':>18}{'null=0.707':>12}{'E':>9}{'sem':>8}")
    for W,m,se in out:
        print(f"  {W:>5}{m:>18.4f}{0.7071:>12.4f}{m-0.7071:>+9.4f}{se:>8.4f}")
    return out
A=np.load("SK_40_120.npy"); B=np.concatenate([np.load("SK.npy")[i] for i in range(7)],axis=1)
print("="*70); print("  MULTI-SCALE PERSISTENCE EXCESS  E(W)"); print("="*70)
Ecurve(A,"SEGMENT 40->120  (phase 1-2)")
Ecurve(B,"SEGMENT 120->200 (phase 3)")
print("\n  time %.1fs"%(time.time()-t0))
