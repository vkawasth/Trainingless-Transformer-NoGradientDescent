"""
CAN A REDUCED-PRECISION BACKWARD PASS RECOVER THE UPDATE SIGN?
sign(d) can't be backprop'd directly (discontinuous), but the backward pass
could be run cheaply. Test: how well does the sign from a NOISY/quantized
gradient match the sign from the full fp32 gradient -- and does training on the
noisy-sign update still work?
Emulate low precision by adding relative noise to g (proxy for bf16/int8 error)
and by stochastic rounding, then measure sign agreement and end-to-end val.
"""
import time, gc, numpy as np, torch
t0=time.time()
g_={}; src=open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
named=list(model.named_parameters())
def flat(): return torch.cat([p.data.flatten() for _,p in named]).clone()
def gradv():
    return torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())) for _,p in named]).clone()
def newopt(): return torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
def V(n=16): return float(eval_val(model,n=n))
# relative-noise levels approximating precision (bf16 ~ 8-bit mantissa ~ 0.4% ;
# int8 per-tensor ~ few % ; aggressive ~ 10-25%)
LEVELS={"fp32 (0%)":0.0,"bf16 ~0.4%":0.004,"int8 ~3%":0.03,"~10%":0.10,"~25%":0.25}
# (1) sign agreement of a noisy gradient vs clean gradient
model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
agree={k:[] for k in LEVELS}
for s in range(60):
    model.train(); x,y=get_batch(); _,l=model(x,y)
    o.zero_grad(); l.backward()
    g=gradv()
    for k,eta in LEVELS.items():
        gn=g*(1+eta*torch.randn_like(g)) if eta>0 else g
        agree[k].append(float((torch.sign(gn)==torch.sign(g)).double().mean()))
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
print("="*72); print("  (1) GRADIENT-SIGN AGREEMENT UNDER PRECISION NOISE"); print("="*72)
for k in LEVELS: print(f"  {k:>14}: sign agreement {100*np.mean(agree[k]):.1f}%")
# (2) end-to-end: train with the NOISY gradient fed to Adam, measure val
print("\n"+"="*72); print("  (2) TRAIN WITH NOISY-PRECISION GRADIENT (full pipeline)"); print("="*72)
def run(eta):
    model.load_state_dict(torch.load("init.pt")); torch.manual_seed(17); o=newopt()
    for s in range(120):
        model.train(); x,y=get_batch(); _,l=model(x,y)
        o.zero_grad(); l.backward()
        if eta>0:
            i=0
            with torch.no_grad():
                for _,p in named:
                    if p.grad is not None: p.grad.mul_(1+eta*torch.randn_like(p.grad))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); o.step()
    return V()
vb=run(0.0)
print(f"  {'precision':>14}{'val':>10}{'vs fp32':>10}")
import sys
for k in (sys.argv[1:] or ["bf16 ~0.4%"]):
    v=run(LEVELS[k]); print(f"  {k:>14}{v:>10.4f}{v/vb:>9.2f}x  (fp32={vb:.4f})", flush=True)
print("\n  if noisy-gradient training holds up, a cheap low-precision backward pass")
print("  recovers usable signs => 'backprop for sign' saves compute.")
print(f"\n  time {time.time()-t0:.0f}s")
