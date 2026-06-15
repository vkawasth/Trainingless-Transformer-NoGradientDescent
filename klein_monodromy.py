#!/usr/bin/env python3
"""
Klein Quadric Monodromy Tracker
=================================
Track the hidden state trajectory as a path on the Klein quadric K ⊂ P^5.

SETUP:
  At each layer l, the active 2-plane of δJ_l is a point on Gr(2,4) = Klein quadric.
  Gr(2,4) ↪ P^5 via Plücker: p = [p12:p13:p14:p23:p24:p34]
  Klein relation: p12*p34 - p13*p24 + p14*p23 = 0  (the quadric constraint)
  
  This reduces 6 projective coordinates to 4 intrinsic DOF.
  The path l → p(l) on K traces the monodromy trajectory.

COMPRESSION:
  Stereographic projection from P^5 → R^4 (local chart on K).
  The 4D compressed path is the structural skeleton of the trajectory.

HOLONOMY DETECTION:
  Factual text: path closes in a clean loop on K (holonomy ≈ identity)
  Fabricated:   path diverges or collapses (p → 0, Levi collapse)

ALSO: Run spectral universality comparison inline.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4
PROJ=32

print(f"\n{'='*65}")
print(f"  KLEIN QUADRIC MONODROMY TRACKER")
print(f"  Gr(2,4) ↪ P^5  →  compress to 4D via stereographic projection")
print(f"  Track: factual vs fabricated path geometry on Klein quadric")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=json.load(f)
with open('/tmp/val_ids.json')   as f: val_ids=json.load(f)
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

def get_batch(split='train'):
    data=train_t if split=='train' else val_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ]   for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

class Attn(nn.Module):
    def __init__(self,d,nh):
        super().__init__()
        self.nh=nh; self.dh=d//nh; self.sc=math.sqrt(d//nh)
        self.WQ=nn.Linear(d,d,bias=False); self.WK=nn.Linear(d,d,bias=False)
        self.WV=nn.Linear(d,d,bias=False); self.op=nn.Linear(d,d,bias=False)
        self.ln=nn.LayerNorm(d)
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h):
        B,S,D_=h.shape; H=self.nh; dh=self.dh
        Q=self.WQ(h).view(B,S,H,dh).transpose(1,2)
        K=self.WK(h).view(B,S,H,dh).transpose(1,2)
        V=self.WV(h).view(B,S,H,dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/self.sc
        mask=torch.triu(torch.ones(S,S,device=h.device),diagonal=1).bool()
        sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        out=(F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D_)
        return self.ln(h+self.op(out))

class FF(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.g=nn.Linear(d,d*2,bias=False); self.v=nn.Linear(d,d*2,bias=False)
        self.o=nn.Linear(d*2,d,bias=False); self.n=nn.LayerNorm(d)
        for w in [self.g,self.v,self.o]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h): return self.n(h+self.o(F.silu(self.g(h))*self.v(h)))

class Block(nn.Module):
    def __init__(self,d,nh): super().__init__(); self.attn=Attn(d,nh); self.ff=FF(d)
    def forward(self,h): return self.ff(self.attn(h))

class LM(nn.Module):
    def __init__(self,d,nh,nl):
        super().__init__()
        self.te=nn.Embedding(VOCAB,d); self.pe=nn.Embedding(512,d)
        self.blocks=nn.ModuleList([Block(d,nh) for _ in range(nl)])
        self.ln_f=nn.LayerNorm(d)
        self.head=nn.Linear(d,VOCAB,bias=False); self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)
    def hidden_states(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs

def clr(s,total=300,warmup=100):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def train_model(seed, steps=300):
    torch.manual_seed(seed)
    m=LM(D,N_HEADS,N_LAYERS)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    for step in range(1,steps+1):
        for pg in opt.param_groups: pg['lr']=clr(step,total=steps)
        m.train(); x,y=get_batch(); _,loss=m(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval()
    with torch.no_grad():
        vl=float(np.mean([m(*get_batch('val'))[1].item() for _ in range(30)]))
    return m, vl

# ── Jacobian ──────────────────────────────────────────────────────────────────
def layer_jacobian(block, h_in, pos, m):
    seq,d_=h_in.shape; m=min(m,seq,d_)
    _,_,Vt=torch.linalg.svd(h_in,full_matrices=False)
    U=Vt[:m,:].T.detach(); J=np.zeros((m,m))
    with torch.enable_grad():
        for i in range(m):
            h=h_in.clone().unsqueeze(0).detach().requires_grad_(True)
            ho=block(h)
            v=(ho[0,pos,:] if ho.dim()==3 else ho[pos,:])
            (v*U[:,i]).sum().backward()
            g=h.grad; g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
            J[:,i]=(U.T@g).numpy()
    return J.T, U.detach().numpy(), m

# ── Klein quadric tools ───────────────────────────────────────────────────────
def plucker_2plane(U2):
    """
    U2: [m, 2] — two orthonormal vectors spanning the 2-plane.
    Returns Plücker coordinates for all 2x2 minors of top-4 rows.
    p = [p01, p02, p03, p12, p13, p23]  (6 coords from top-4 rows)
    """
    V = U2[:4, :]   # [4, 2] — top-4 rows (sufficient for Klein embedding)
    p01 = float(V[0,0]*V[1,1] - V[0,1]*V[1,0])
    p02 = float(V[0,0]*V[2,1] - V[0,1]*V[2,0])
    p03 = float(V[0,0]*V[3,1] - V[0,1]*V[3,0])
    p12 = float(V[1,0]*V[2,1] - V[1,1]*V[2,0])
    p13 = float(V[1,0]*V[3,1] - V[1,1]*V[3,0])
    p23 = float(V[2,0]*V[3,1] - V[2,1]*V[3,0])
    return np.array([p01, p02, p03, p12, p13, p23])

def klein_relation(p):
    """Klein relation: p01*p23 - p02*p13 + p03*p12 = 0 for valid 2-plane."""
    return p[0]*p[5] - p[1]*p[4] + p[2]*p[3]

def normalize_plucker(p):
    n = np.linalg.norm(p)
    return p/n if n > 1e-10 else p

def stereographic_project(p):
    """
    Project P^5 → R^4 via stereographic projection from north pole n=(0,0,0,0,0,1).
    Valid when p[5] ≠ -1. Maps Klein quadric (4-dim) to R^4.
    p = [p0,...,p5] normalized to ||p||=1.
    Projection: q_i = p_i / (1 + p[5])  for i=0..3
    This gives 4 coordinates on the Klein quadric chart.
    """
    p_n = normalize_plucker(p)
    denom = 1.0 + p_n[5]
    if abs(denom) < 1e-8:
        # Near south pole — use different chart
        denom = 1.0 - p_n[5]
        if abs(denom) < 1e-8: return np.zeros(4)
        return p_n[:4] / denom
    return p_n[:4] / denom

def path_closure(coords_4d):
    """
    Measure how well the 4D path closes.
    closure = ||first_point - last_point|| / path_length
    0 = perfect loop, 1 = open path
    """
    path = np.array(coords_4d)   # [L, 4]
    start = path[0]; end = path[-1]
    closure_dist = np.linalg.norm(end - start)
    path_length = sum(np.linalg.norm(path[i+1]-path[i]) for i in range(len(path)-1))
    if path_length < 1e-8: return 1.0
    return float(closure_dist / path_length)

def path_collapse(coords_4d, norms_6d):
    """
    Detect Levi collapse: does ||p|| drop toward 0?
    Returns fraction of layers where ||p|| < 0.1 * mean(||p||)
    """
    norms = np.array(norms_6d)
    threshold = norms.mean() * 0.1
    return float(np.mean(norms < threshold))

# ── Extract Klein path for one text ──────────────────────────────────────────
def extract_klein_path(model, text_ids, pos, m=PROJ):
    """Extract the 4D Klein quadric path across all layers."""
    x = text_ids.unsqueeze(0)
    with torch.no_grad():
        hs = model.hidden_states(x); hs = [h[0] for h in hs]

    plueckers_6d = []   # [L, 6]
    coords_4d    = []   # [L, 4]
    klein_resids = []   # Klein relation residuals
    norms_6d     = []   # ||p|| at each layer

    for l in range(N_LAYERS):
        J, U_basis, ma = layer_jacobian(model.blocks[l], hs[l], pos, m)
        dJ = J - np.eye(ma)
        sv = np.linalg.svd(dJ, compute_uv=False)
        U_sv, _, _ = np.linalg.svd(dJ)
        U2 = U_sv[:, :2]   # top-2 singular vectors → 2-plane

        p6 = plucker_2plane(U2)         # 6D Plücker
        kr = klein_relation(p6)         # should be ~0
        c4 = stereographic_project(p6)  # 4D chart
        n6 = float(np.linalg.norm(p6))

        plueckers_6d.append(p6)
        coords_4d.append(c4)
        klein_resids.append(abs(kr))
        norms_6d.append(n6)

    return {
        'p6':     np.array(plueckers_6d),   # [L, 6]
        'c4':     np.array(coords_4d),      # [L, 4]
        'klein':  np.array(klein_resids),   # [L]
        'norms':  np.array(norms_6d),       # [L]
    }

def text_to_ids(text):
    words = text.lower().split()
    if isinstance(vocab, dict):
        ids = [vocab.get(w, hash(w)%VOCAB) for w in words[:SEQ]]
    else:
        # vocab is a list — build lookup once
        v2i = {w:i for i,w in enumerate(vocab)}
        ids = [v2i.get(w, hash(w)%VOCAB) for w in words[:SEQ]]
    return torch.tensor(ids, dtype=torch.long)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training source model (seed=42, 300 steps)...")
t0=time.time()
modelA, valA = train_model(42, 300)
print(f"  Model A: val={valA:.4f}  t={time.time()-t0:.0f}s")

print("\nTraining model B (seed=137, 300 steps) for universality test...")
t0=time.time()
modelB, valB = train_model(137, 300)
print(f"  Model B: val={valB:.4f}  t={time.time()-t0:.0f}s\n")

# ── Test texts ────────────────────────────────────────────────────────────────
TEXTS = [
    ("factual",    "Albert Einstein was born in Ulm Germany in 1879 and developed the theory of relativity"),
    ("fabricated", "Albert Einstein invented quantum teleportation in 1923 while working at MIT on neural networks"),
    ("factual",    "Napoleon Bonaparte was defeated at the Battle of Waterloo in 1815 by the Duke of Wellington"),
    ("fabricated", "Napoleon Bonaparte invented the internet in 1799 during his campaign in Silicon Valley"),
    ("structural", "The transformer architecture uses multi-head self-attention to process sequential data efficiently"),
]

pos = SEQ // 2

# ── Klein paths on Model A ────────────────────────────────────────────────────
print(f"{'='*65}")
print(f"  KLEIN QUADRIC PATHS — MODEL A")
print(f"  path_closure: 0=loop, 1=open  |  collapse: fraction of layers with ||p||≈0")
print("="*65)

paths_A = {}
print(f"\n  {'label':>12}  {'closure':>9}  {'collapse':>9}  "
      f"{'mean||p||':>10}  {'mean|K|':>9}  {'verdict'}")
print("  "+"-"*62)

for label, text in TEXTS:
    ids = text_to_ids(text)
    path = extract_klein_path(modelA, ids, min(pos, len(ids)-1))
    closure  = path_closure(path['c4'])
    collapse = path_collapse(path['c4'], path['norms'])
    mean_n   = float(path['norms'].mean())
    mean_k   = float(path['klein'].mean())
    paths_A[text[:20]] = path

    verdict = "✓ STABLE" if (closure < 0.5 and collapse < 0.1) else \
              "✗ COLLAPSE" if collapse > 0.3 else \
              "~ DRIFT"
    print(f"  {label:>12}  {closure:>9.4f}  {collapse:>9.4f}  "
          f"{mean_n:>10.4f}  {mean_k:>9.6f}  {verdict}  '{text[:25]}...'")

# ── Layer-by-layer path for factual vs fabricated ─────────────────────────────
print(f"\n{'='*65}")
print(f"  LAYER-BY-LAYER: FACTUAL vs FABRICATED (first pair)")
print(f"  ||p|| and Klein residual at each layer")
print("="*65)

text_f = TEXTS[0][1]; text_b = TEXTS[1][1]
ids_f = text_to_ids(text_f); ids_b = text_to_ids(text_b)
path_f = extract_klein_path(modelA, ids_f, min(pos, len(ids_f)-1))
path_b = extract_klein_path(modelA, ids_b, min(pos, len(ids_b)-1))

print(f"\n  {'L':>3}  {'||p_fact||':>11}  {'||p_fabr||':>11}  "
      f"{'ratio':>7}  {'K_fact':>9}  {'K_fabr':>9}  {'4D dist':>9}")
print("  "+"-"*68)

for l in range(N_LAYERS):
    nf = path_f['norms'][l]; nb = path_b['norms'][l]
    ratio = nf/nb if nb>1e-8 else float('inf')
    kf = path_f['klein'][l]; kb = path_b['klein'][l]
    # 4D chordal distance between paths at this layer
    cf = path_f['c4'][l]; cb = path_b['c4'][l]
    dist4 = float(np.linalg.norm(cf - cb))
    marker = " ←" if abs(l-14)<=1 else ""
    print(f"  L{l:>2}  {nf:>11.4f}  {nb:>11.4f}  "
          f"{ratio:>7.3f}  {kf:>9.6f}  {kb:>9.6f}  {dist4:>9.4f}{marker}")

# ── Spectral universality via Plücker ─────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  SPECTRAL UNIVERSALITY — Plücker similarity A vs B")
print(f"  Same data, different seeds. Are the Klein paths the same?")
print("="*65)

x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
ids_ref=x_ref[0]

pathA_ref = extract_klein_path(modelA, ids_ref, pos)
pathB_ref = extract_klein_path(modelB, ids_ref, pos)

cos_AB = [float(np.dot(normalize_plucker(pathA_ref['p6'][l]),
                        normalize_plucker(pathB_ref['p6'][l])))
          for l in range(N_LAYERS)]

print(f"\n  cos(p_A, p_B) per layer (1=identical Klein point, 0=orthogonal):")
print(f"  {'L':>3}  {'cos(pA,pB)':>12}  {'||pA||':>8}  {'||pB||':>8}  {'same?'}")
print("  "+"-"*48)
for l in range(N_LAYERS):
    c=cos_AB[l]; na=pathA_ref['norms'][l]; nb=pathB_ref['norms'][l]
    marker=" ←L14" if l==14 else ""
    same="≈" if abs(c)>0.8 else ("~" if abs(c)>0.5 else "≠")
    print(f"  L{l:>2}  {c:>12.4f}  {na:>8.4f}  {nb:>8.4f}  {same}{marker}")

mean_cos=float(np.mean(np.abs(cos_AB)))
print(f"\n  Mean |cos(pA,pB)| = {mean_cos:.4f}")

if mean_cos > 0.8:
    verdict = "SPECTRAL UNIVERSALITY CONFIRMED\n  The Klein paths are nearly identical across seeds.\n  W* is determined by (architecture, data), not initialization.\n  Inverse scattering bypass of gradient descent is possible."
elif mean_cos > 0.5:
    verdict = "PARTIAL UNIVERSALITY\n  The Klein paths are similar but not identical.\n  Approximate initialization via IST would help but not replace training."
else:
    verdict = "UNIVERSALITY REFUTED\n  The Klein paths differ significantly across seeds.\n  Multiple attractors exist. Gradient descent is necessary."

print(f"\n  VERDICT: {verdict}")

# ── 4D path for 2-layer model ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  THE 4D COMPRESSION FOR A 2-LAYER MODEL")
print(f"  The Klein path {N_LAYERS}L → {N_LAYERS}×4 = {N_LAYERS*4}D compressed trajectory")
print("="*65)
print(f"""
  The full Klein path of Model A on the reference input:
  Shape: [{N_LAYERS} layers × 4 coordinates] = {N_LAYERS*4}D vector

  This {N_LAYERS*4}D vector is the compressed monodromy trajectory.
  It encodes the full structural skeleton of the 24-layer forward pass
  in {N_LAYERS*4} real numbers (vs {N_LAYERS}×{32}×{32} = {N_LAYERS*32*32} Jacobian entries).

  A 2-layer model trained to reproduce this 4D path at L1 and L2
  would be learning to match the Klein quadric geometry of the
  24-layer model in {N_LAYERS*4}D instead of matching hidden states in {N_LAYERS}×{256}D.

  Compression ratio: {N_LAYERS*256}/{N_LAYERS*4} = {256//4}x smaller target.

  The 4D coordinates at L14 (attractor):
  Model A: {pathA_ref['c4'][14].round(4)}
  Model B: {pathB_ref['c4'][14].round(4)}
  Distance: {np.linalg.norm(pathA_ref['c4'][14]-pathB_ref['c4'][14]):.4f}

  If this distance is small: L14 is a universal fixed point in Klein space.
  The 2-layer model just needs to hit this 4-point target.
""")
