"""Diagonal Fisher per tile (true Fisher: labels sampled from the model)."""
import re, pickle, time, numpy as np, torch, torch.nn.functional as F
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]
named=list(model.named_parameters())
def keyLT(n):
    m=re.match(r"blocks\.(\d+)\.",n)
    if not m: return ("EMB","EMB")
    l=int(m.group(1)); nl=n.lower()
    t=("W_Q" if "wq" in nl else "W_K" if "wk" in nl else "W_V" if "wv" in nl else
       "W_O" if ".op." in nl else "LN" if (".ln." in nl or ".n." in nl) else "FF")
    return (f"L{l}",t)
off=0; GRP={}
for n,p in named:
    k=keyLT(n); GRP.setdefault(k,[]).append((off,off+p.numel())); off+=p.numel()
P=off
IDX={k: torch.cat([torch.arange(a,b) for a,b in v]).numpy() for k,v in GRP.items()}
# train to step 200 to match the tile run
torch.manual_seed(17)
model.load_state_dict(torch.load("init.pt"))
opt=torch.optim.AdamW(model.parameters(), lr=g_["LR"]*5, betas=(0.9,0.95), weight_decay=0.1)
for s in range(200):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
print(f"  trained to 200 ({time.time()-t0:.0f}s)", flush=True)
fis=torch.zeros(P)
NB=12
model.eval()
for b in range(NB):
    x,_=get_batch()
    logits,_=model(x)
    lp=F.log_softmax(logits.reshape(-1,logits.shape[-1]),dim=-1)
    ys=torch.multinomial(lp.exp(),1).squeeze(1)          # sample from model = TRUE Fisher
    model.zero_grad(set_to_none=True)
    F.nll_loss(lp,ys).backward()
    gv=torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()))
                  for _,p in named])
    fis += gv*gv
fis/=NB
out={f"{lk}|{gk}": fis[IDX[(lk,gk)]].numpy() for (lk,gk) in IDX}
pickle.dump(out, open("fisher.pkl","wb"))
tot=float(fis.sum())
print(f"  Fisher trace {tot:.4e}   ({time.time()-t0:.0f}s)")
for k in sorted(out): print(f"    {k:<12} share {100*out[k].sum()/tot:6.2f}%  n={len(out[k]):>7,}")
