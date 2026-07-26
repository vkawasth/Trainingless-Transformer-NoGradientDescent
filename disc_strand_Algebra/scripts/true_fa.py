"""
TRUE FEEDBACK ALIGNMENT THROUGH THE RECURSION.
Custom autograd: forward uses W; backward propagates delta with a FIXED RANDOM
matrix B (not W^T). Error compounds through depth. Measure weighted sign
agreement of the resulting weight-gradient vs exact backprop, per layer and
overall, and train on it.
Build a small MLP mirroring the transformer's FF depth (6 layers) on the same
sign-prediction target so depth-compounding is exercised.
"""
import time, gc, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
t0=time.time()
torch.manual_seed(0)

class FALinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, B):
        ctx.save_for_backward(x, W, B)
        return x @ W.t()
    @staticmethod
    def backward(ctx, gout):
        x, W, B = ctx.saved_tensors
        # gradient to INPUT uses B (random), not W  -> feedback alignment
        gin = gout @ B
        # gradient to WEIGHT is exact (delta^T x), delta arriving via B chain
        gW = gout.t() @ x
        return gin, gW, None

class Layer(nn.Module):
    def __init__(s, d_in, d_out, fa):
        super().__init__()
        s.W = nn.Parameter(torch.randn(d_out, d_in)*0.05)
        s.register_buffer("B", torch.randn(d_out, d_in)*0.05)  # fixed random feedback
        s.fa = fa
    def forward(s, x):
        if s.fa: return FALinear.apply(x, s.W, s.B)
        return x @ s.W.t()

class Net(nn.Module):
    def __init__(s, fa, D=128, L=6, V=64):
        super().__init__()
        s.emb=nn.Embedding(V,D)
        s.layers=nn.ModuleList([Layer(D,D,fa) for _ in range(L)])
        s.head=nn.Linear(D,V)
    def forward(s,x):
        h=s.emb(x).mean(1)
        for l in s.layers: h=F.gelu(l(h))
        return s.head(h)

# task: predict a fixed random target class per input pattern (memorization, matches setup)
V=64; N=256
torch.manual_seed(1)
Xd=torch.randint(0,V,(N,8)); Yd=torch.randint(0,V,(N,))
def batch(bs=32):
    i=torch.randint(0,N,(bs,)); return Xd[i],Yd[i]

def flatg(net): return torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])
def gradsign_agreement():
    """one exact net and one FA net at identical weights; compare weight-grad signs per layer"""
    torch.manual_seed(5)
    ex=Net(False); fa=Net(True)
    fa.load_state_dict(ex.state_dict(), strict=False)   # same W; B are buffers (random)
    x,y=batch(128)
    for net in (ex,fa):
        net.zero_grad(); F.cross_entropy(net(x),y).backward()
    print("  weighted sign agreement of weight-gradient, exact vs FA, per layer:")
    print(f"  {'layer':>8}{'wtd sign agree':>16}{'cos':>8}")
    for i,(pe,pf) in enumerate(zip(ex.parameters(),fa.parameters())):
        if pe.grad is None or pe.dim()<2: continue
        ge=pe.grad.flatten(); gf=pf.grad.flatten()
        w=ge.abs()
        aw=float((w*(torch.sign(ge)==torch.sign(gf)).double()).sum()/w.sum())
        cs=float(ge@gf/(ge.norm()*gf.norm()+1e-30))
        print(f"  {i:>8}{aw:>16.3f}{cs:>8.3f}")
print("="*64); print("  TRUE FA: SIGN AGREEMENT THROUGH DEPTH"); print("="*64)
gradsign_agreement()
def train(fa,steps=400):
    torch.manual_seed(5); net=Net(fa)
    o=torch.optim.AdamW(net.parameters(),lr=3e-3,betas=(0.9,0.95),weight_decay=0.1)
    for s in range(steps):
        x,y=batch(); o.zero_grad(); l=F.cross_entropy(net(x),y); l.backward(); o.step()
    with torch.no_grad(): acc=float((net(Xd).argmax(1)==Yd).float().mean())
    return float(l), acc
print("\n"+"="*64); print("  TRAINING: exact backprop vs true FA"); print("="*64)
le,ae=train(False); lf,af=train(True)
print(f"  exact : final loss {le:.4f}  train acc {ae:.3f}")
print(f"  FA    : final loss {lf:.4f}  train acc {af:.3f}")
print(f"\n  if FA sign agreement stays >0.95 through depth AND FA trains, the door")
print(f"  is open. if agreement decays with depth or FA fails to fit, it is shut.")
print(f"\n  time {time.time()-t0:.0f}s")
