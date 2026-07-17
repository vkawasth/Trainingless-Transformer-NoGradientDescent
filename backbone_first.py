"""r-triggered backbone-first schedule on the real model.
Backbone-only (attention skipped, ~60% FLOPs/step) until |g_att|/|g_ff| turns
up, then instantiate the periphery. Tallies real FLOPs vs a full-model run."""
import torch, math
g_={}; src=open("compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]; eval_val=g_["eval_val"]; LR=g_["LR"]
blocks=[m for n,m in model.named_modules() if n.count(".")==1 and n.startswith("blocks.")]

class PassThrough(torch.nn.Module):
    def forward(self,h,*a,**k): return h
def backbone_only(on):
    for b in blocks:
        if on:
            if not hasattr(b,"_sa"): b._sa=b.attn
            b.attn=PassThrough()
        elif hasattr(b,"_sa"): b.attn=b._sa; del b._sa

def gname(n):
    n=n.lower()
    if "wq" in n or "wk" in n or "wv" in n or ".op." in n: return "att"
    if ".ff." in n: return "ff"
    return "other"
def probe_r():
    backbone_only(False)
    x,y=get_batch(); out=model(x,y); loss=out[1]; model.zero_grad(set_to_none=True); loss.backward()
    ga=gf=0.0
    for n,p in model.named_parameters():
        if p.grad is None: continue
        g=gname(n); s=float(p.grad.pow(2).sum())
        if g=="att": ga+=s
        elif g=="ff": gf+=s
    model.zero_grad(set_to_none=True)
    return math.sqrt(ga)/(math.sqrt(gf)+1e-12)

FULL_F=13_483_376_640; BB_F=8_047_558_656   # measured earlier
def run(N=80, probe_every=8):
    opt=torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9,0.95), weight_decay=0.1)
    backbone_only(True)
    flops=0.0; on=False; rhist=[]; onset=None
    for step in range(1,N+1):
        if not on and step%probe_every==0:
            r=probe_r(); flops+=FULL_F; rhist.append((step,r))    # probe = 1 full fwd+bwd
            # onset: r risen for 2 consecutive probes after its dip
            if len(rhist)>=3:
                r0,r1,r2=rhist[-3][1],rhist[-2][1],rhist[-1][1]
                if r2>r1>r0 and r2>min(x[1] for x in rhist)*1.1:
                    onset=step; on=True; backbone_only(False)
                    print(f"  onset at step {step}: r rising {r0:.2f}<{r1:.2f}<{r2:.2f} -> periphery ON")
        backbone_only(not on)
        x,y=get_batch(); out=model(x,y); loss=out[1]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        flops += FULL_F if on else BB_F
    backbone_only(False)
    return flops, onset, eval_val(model,n=6), rhist

print("="*70); print("  BACKBONE-FIRST vs FULL (real model, FLOP-accounted)"); print("="*70)
N=80
f_bf, onset, v_bf, rhist = run(N=N)
f_full = N*FULL_F
print(f"\n  r probes: " + " ".join(f"{s}:{r:.2f}" for s,r in rhist))
print(f"  onset step: {onset}   final val: {v_bf:.4f}")
print(f"\n  FLOPs over {N} steps:")
print(f"    full-model     : {f_full/1e12:.2f} TFLOP")
print(f"    backbone-first : {f_bf/1e12:.2f} TFLOP")
print(f"    saved          : {100*(1-f_bf/f_full):.0f}%  (window = steps 1..{onset})")
