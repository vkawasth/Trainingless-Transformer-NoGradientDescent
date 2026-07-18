"""
What is the slow mode computing?
 - save layer-0 QK circuit M = WQ^T WK at checkpoints
 - (A) extrapolate the exponential relaxation from EARLY steps -> predict the limit
 - (B) identify the limit: positional vs token subspace, and its action on positions
"""
import numpy as np, torch, json
g_={}; src=open("compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
blocks=[m for n,m in model.named_modules() if n.count(".")==1 and n.startswith("blocks.")]

def QK(layer=0):
    a=blocks[layer].attn
    return (a.WQ.weight.detach().T@a.WK.weight.detach()).cpu().numpy().copy()

opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
ck=[0,20,40,60,80,100,120,140,160,180,200]
snaps={}; vals={}
step=0
for tgt in ck:
    while step<tgt:
        model.train(); x,y=get_batch(); _,l=model(x,y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
    snaps[step]=QK(0); vals[step]=float(eval_val(model,n=4))
    print(f"  step {step:>3}  val {vals[step]:.4f}", flush=True)

np.savez("qk_snaps.npz", **{str(k):v for k,v in snaps.items()},
         pe=model.pe.weight.detach().cpu().numpy(),
         te=model.te.weight.detach().cpu().numpy(),
         steps=np.array(ck), vals=np.array([vals[k] for k in ck]))
print("saved qk_snaps.npz")
