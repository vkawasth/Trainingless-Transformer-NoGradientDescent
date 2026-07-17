import torch, math
g_={}; src=open("compiler_geometri_patched_86.py").read()
exec(src[:src.find("# ── PHASE 1")], g_)
model=g_["model"]; get_batch=g_["get_batch"]
# inspect block structure
blocks=[m for n,m in model.named_modules() if n.count(".")==1 and n.startswith("blocks.")]
b0=blocks[0]
print("block children:", [n for n,_ in b0.named_children()])
print("attn children:", [n for n,_ in b0.attn.named_children()] if hasattr(b0,"attn") else "NO .attn")
print("n_blocks:", len(blocks), " P:", sum(p.numel() for p in model.parameters()))

# passthrough to skip attention entirely
class PassThrough(torch.nn.Module):
    def forward(self, h, *a, **k): return h

def set_backbone_only(on):
    for b in blocks:
        if on:
            if not hasattr(b,"_saved_attn"): b._saved_attn=b.attn
            b.attn=PassThrough()
        else:
            if hasattr(b,"_saved_attn"): b.attn=b._saved_attn; del b._saved_attn

def step_flops(backbone_only):
    set_backbone_only(backbone_only)
    x,y=get_batch()
    try:
        from torch.utils.flop_counter import FlopCounterMode
        fc=FlopCounterMode(display=False)
        with fc:
            out=model(x,y); loss=out[1] if isinstance(out,tuple) else out
            loss.backward()
        tot=fc.get_total_flops()
    except Exception as e:
        tot=None; print("flopcounter err:",e)
    model.zero_grad(set_to_none=True)
    set_backbone_only(False)
    return tot

full=step_flops(False)
bb  =step_flops(True)
print(f"\nFLOPs/step (fwd+bwd):")
print(f"  full        : {full:,}")
print(f"  backbone-only: {bb:,}")
print(f"  ratio bb/full: {bb/full:.3f}   -> backbone step costs {100*bb/full:.0f}% of full")
print(f"  saving/step while backbone-only: {100*(1-bb/full):.0f}%")
