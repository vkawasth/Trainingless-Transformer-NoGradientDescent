"""Is L1-cancellation = L2-chord/path an identity? Test the shape factors
   kappa = E|x| / sqrt(E[x^2]).  Gaussian: 0.7979.  If kappa_net = kappa_step,
   the two measures coincide by construction, not by absence of rotation."""
import time, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def kap(x): return float(x.abs().mean()/max(float((x*x).mean())**0.5,1e-30))
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17)
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
th0=flat(); prev=th0.clone(); net=torch.zeros_like(th0); path=torch.zeros_like(th0)
l2num=0.0; ks=[]
for s in range(200):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    cur=flat(); d=cur-prev; prev=cur
    net+=d; path+=d.abs(); l2num+=float(d.norm()); ks.append(kap(d))
D=net
L1=float(D.abs().sum()/path.sum()); L2=float(D.norm())/l2num
kn=kap(D); kd=float(np.mean(ks))
print("="*76); print("  IS L1 == L2 AN IDENTITY?"); print("="*76)
print(f"  L1 (per-coordinate)  sum|net|/sum|path| = {L1:.4f}")
print(f"  L2 (whole-vector)    chord/path         = {L2:.4f}")
print(f"  measured gap                            = {abs(L1-L2):.4f}\n")
print(f"  shape factor of the NET displacement   kappa_net  = {kn:.4f}")
print(f"  shape factor of a STEP (mean over 200) kappa_step = {kd:.4f}")
print(f"  Gaussian reference                                = {np.sqrt(2/np.pi):.4f}")
print(f"\n  identity predicts  L1/L2 = kappa_net/kappa_step = {kn/kd:.4f}")
print(f"  measured           L1/L2                        = {L1/L2:.4f}")
print(f"  agreement: {100*abs(1-(kn/kd)/(L1/L2)):.2f}% discrepancy")
print("\n  If these agree, L1~L2 reflects matched distribution shape, NOT the")
print("  absence of rotation, and carries no dynamical content.")
print(f"\n  time {time.time()-t0:.0f}s")
