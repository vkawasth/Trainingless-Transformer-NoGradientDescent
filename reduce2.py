"""DOES BLOCK MISALIGNMENT CAUSE THE OTHER RESIDUALS? (CORRECTED)

The previous attempt had a coding error: the rotation
    nu = blk*(1-a) + a*sign(c)*blk
interpolates a block with a signed copy of ITSELF, so it never rotates toward
the shared direction. The random control destroyed the model (val 8.21 vs 0.030),
which was the signal that the manipulation was wrong rather than informative.

Corrected construction. The shared direction w lives in FUNCTION space (the
logits), where the blocks actually meet. To move a block toward it we need the
pullback into that block's parameter coordinates, which is a Jacobian TRANSPOSE
product -- obtained exactly by one backward pass of <logits, w>:

    z_i = P_i J^T w        one vjp, restricted to block i

Then the block's update is rotated toward z_i by angle alpha, with its norm
preserved exactly:

    nu_i = normalise( (1-a) * du_i/|du_i| + a * z_i/|z_i| ) * |du_i|

Controls:
  a=0        baseline
  a=0.5,1.0  toward the shared direction
  rand       same rotation toward a RANDOM function-space direction, which
             preserves the norm and the construction but not the alignment

Diagnostics: R3 (mean pairwise block overlap in function space), R1 (fwd/bwd
alignment), R2 (fraction of update energy outside the top-3 Fisher sheet), and
val, all at the end of a matched number of steps.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, math, re, json
V, D, L, T = 40, 12, 2, 16


class Blk(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1 = nn.LayerNorm(D); s.n2 = nn.LayerNorm(D)
        s.q = nn.Linear(D, D, bias=False); s.k = nn.Linear(D, D, bias=False)
        s.v = nn.Linear(D, D, bias=False); s.o = nn.Linear(D, D, bias=False)
        s.g = nn.Linear(D, 2 * D, bias=False); s.u = nn.Linear(D, 2 * D, bias=False)
        s.w = nn.Linear(2 * D, D, bias=False)

    def forward(s, x):
        h = s.n1(x); q, k, v = s.q(h), s.k(h), s.v(h)
        a = Fn.softmax(q @ k.transpose(-2, -1) / math.sqrt(D)
                       + torch.triu(torch.full((x.shape[1], x.shape[1]), -1e9), 1), dim=-1)
        x = x + s.o(a @ v); h = s.n2(x)
        return x + s.w(Fn.silu(s.g(h)) * s.u(h))


class M(nn.Module):
    def __init__(s):
        super().__init__()
        s.te = nn.Embedding(V, D); s.pe = nn.Embedding(T, D)
        s.b = nn.ModuleList([Blk() for _ in range(L)]); s.nf = nn.LayerNorm(D)

    def forward(s, x, y):
        h = s.te(x) + s.pe(torch.arange(x.shape[1]))
        for b in s.b:
            h = b(h)
        lo = s.nf(h) @ s.te.weight.T
        return lo, Fn.cross_entropy(lo.reshape(-1, V), y.reshape(-1))


rg = np.random.default_rng(1)
rules = {i: [(i * 3 + 1) % V, (i * 5 + 2) % V] for i in range(V)}
s_ = [0]
for _ in range(6000):
    c = s_[-1]
    s_.append(int(rules[c][0] if (len(s_) > 1 and s_[-2] % 2 == 0) else rules[c][1]))
seq = np.array(s_)


def batch(n=24):
    i = rg.integers(0, len(seq) - T - 1, n)
    return (torch.tensor(np.stack([seq[j:j + T] for j in i])),
            torch.tensor(np.stack([seq[j + 1:j + T + 1] for j in i])))


def role(nm):
    if nm.startswith("te") or nm.startswith("pe"):
        return "EMB"
    if "n1" in nm or "n2" in nm or nm.startswith("nf"):
        return "LN"
    for pat, lab in ((r"\.q\.", "W_Q"), (r"\.k\.", "W_K"),
                     (r"\.v\.", "W_V"), (r"\.o\.", "W_O")):
        if re.search(pat, nm):
            return lab
    if ".g." in nm or ".u." in nm or ".w." in nm:
        return "FF"
    return "other"


ROLES = ["EMB", "LN", "W_Q", "W_K", "W_V", "W_O", "FF"]


def run(alpha, mode="sync", steps=260, seed=0):
    torch.manual_seed(seed)
    m = M(); named = [(n, p) for n, p in m.named_parameters()]
    ps = [p for _, p in named]
    P = sum(p.numel() for p in ps)
    span = {}; i = 0
    for nm, p in named:
        span[nm] = (i, i + p.numel()); i += p.numel()
    idx = {r: [] for r in ROLES}
    for nm, (a, b) in span.items():
        if role(nm) in idx:
            idx[role(nm)].append(torch.arange(a, b))
    idx = {r: torch.cat(v) for r, v in idx.items() if v}
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)

    def flat():
        return torch.cat([p.data.flatten() for p in ps]).clone()

    def setth(t):
        with torch.no_grad():
            j = 0
            for p in ps:
                q = p.numel(); p.data.copy_(t[j:j + q].view_as(p)); j += q

    def vjp(x, y, w):
        """J^T w exactly: one backward of <logits, w>."""
        m.zero_grad()
        lo, _ = m(x, y)
        (lo.flatten() * w).sum().backward()
        z = torch.cat([(p.grad.flatten() if p.grad is not None
                        else torch.zeros(p.numel())) for p in ps]).clone()
        m.zero_grad(); return z

    gen = torch.Generator().manual_seed(11)
    XW, YW = batch(16)
    nlog = m(XW, YW)[0].numel()
    for st in range(steps):
        th = flat()
        x, y = batch(); _, l = m(x, y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        du = flat() - th
        if alpha > 0:
            if mode == "sync":
                # shared direction: the mean function-space image of the blocks,
                # obtained without forming J -- use the loss gradient's image,
                # which is the direction all blocks are pushing the logits along
                m.zero_grad()
                lo, ll = m(XW, YW)
                gl = torch.autograd.grad(ll, lo)[0].flatten()
                w = -gl / (gl.norm() + 1e-30)
                m.zero_grad()
            else:
                w = torch.randn(nlog, generator=gen); w = w / w.norm()
            setth(th)
            z = vjp(XW, YW, w)
            nu = du.clone()
            for r, ii in idx.items():
                blk = du[ii]; n0 = blk.norm()
                zi = z[ii]
                if float(zi.norm()) < 1e-20 or float(n0) < 1e-20:
                    continue
                d = blk / n0; e = zi / zi.norm()
                mix = (1 - alpha) * d + alpha * e
                nu[ii] = mix / (mix.norm() + 1e-30) * n0
            setth(th + nu)
    th = flat()
    # R3: pairwise block overlap in function space, from the gradient's blocks
    x, y = batch(); m.zero_grad(); _, l = m(x, y); l.backward()
    g = torch.cat([(p.grad.flatten() if p.grad is not None
                    else torch.zeros(p.numel())) for p in ps]).clone()
    m.zero_grad()
    EVX = [batch(24) for _ in range(2)]

    def lg(t):
        setth(t); o = []
        with torch.no_grad():
            for xx, yy in EVX:
                o.append(m(xx, yy)[0].flatten().clone())
        return torch.cat(o)

    eps = 1.0; Js = {}
    for r, ii in idx.items():
        v = torch.zeros(P); v[ii] = g[ii]
        if float(v.norm()) < 1e-20:
            continue
        v = v / v.norm()
        Js[r] = (lg(th + eps * v) - lg(th - eps * v)) / (2 * eps)
    setth(th)
    rs = list(Js)
    ovs = [abs(float((Js[rs[a]] / (Js[rs[a]].norm() + 1e-30)) @
                     (Js[rs[b]] / (Js[rs[b]].norm() + 1e-30))))
           for a in range(len(rs)) for b in range(a + 1, len(rs))]
    # R1 and R2
    def gv():
        return torch.cat([(p.grad.flatten() if p.grad is not None
                           else torch.zeros(p.numel())) for p in ps]).clone()
    Gs = []
    for _ in range(150):
        xx, yy = batch(4); m.zero_grad(); _, ll = m(xx, yy); ll.backward()
        Gs.append(gv()); setth(th)
    m.zero_grad()
    Gm = torch.stack(Gs, 1)
    QF = torch.linalg.svd(Gm - Gm.mean(1, keepdim=True), full_matrices=False)[0][:, :3]
    prev = flat()
    for _ in range(3):
        xx, yy = batch(); _, ll = m(xx, yy); opt.zero_grad(); ll.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    u = flat() - prev; setth(th)
    r1 = abs(float((th * u).sum() / (th.norm() * u.norm() + 1e-30)))
    pf = QF @ (QF.T @ u)
    r2 = 1 - float((pf * pf).sum() / ((u * u).sum() + 1e-30))
    vl = float(np.mean([float(m(xx, yy)[1].detach()) for xx, yy in EVX]))
    return dict(val=vl, R3=float(np.mean(ovs)), R1=r1, R2=r2)


print(f"  {'arm':>14}{'val':>9}{'R3 blocks':>12}{'R1 fwd.bwd':>12}{'R2 resid':>11}")
res = {}
for nm, a, md in (("baseline", 0.0, "sync"), ("sync 0.3", 0.3, "sync"),
                  ("sync 0.6", 0.6, "sync"), ("random 0.6", 0.6, "rand")):
    r = run(a, md); res[nm] = r
    print(f"  {nm:>14}{r['val']:>9.4f}{r['R3']:>12.4f}{r['R1']:>12.4f}{r['R2']:>11.4f}",
          flush=True)
json.dump(res, open("/home/claude/work/res_reduce.json", "w"), indent=2)
b = res["baseline"]
print(f"\n  vs baseline: sync 0.6 changes R3 by "
      f"{res['sync 0.6']['R3']-b['R3']:+.4f}, R1 by {res['sync 0.6']['R1']-b['R1']:+.4f}, "
      f"R2 by {res['sync 0.6']['R2']-b['R2']:+.4f}")
print(f"  random 0.6 : R3 {res['random 0.6']['R3']-b['R3']:+.4f}, "
      f"R1 {res['random 0.6']['R1']-b['R1']:+.4f}, R2 {res['random 0.6']['R2']-b['R2']:+.4f}")
print(f"\n  sync raises R3 and moves R1/R2 while random does not => mechanism")
print(f"  sync and random move them alike => the residuals are independent")
