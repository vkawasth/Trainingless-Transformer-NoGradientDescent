import time, torch, copy
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]
torch.save(copy.deepcopy(model.state_dict()), "/home/claude/w/init.pt")
print("prefix time %.1fs  params %d"%(time.time()-t0, sum(p.numel() for p in model.parameters())))
# time one step
gb=g_["get_batch"]; ev=g_["eval_val"]; LR=g_["LR"]
opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
t1=time.time()
for _ in range(5):
    model.train(); x,y=gb(); _,l=model(x,y); opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
print("per-step %.3fs"%((time.time()-t1)/5))
t2=time.time(); v=float(ev(model,n=8)); print("eval n=8 %.3fs val %.4f"%(time.time()-t2,v))
