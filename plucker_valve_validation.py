#!/usr/bin/env python3
"""
Plücker L20 Valve Validation
==============================
Test ||p(L20)|| as hallucination detector across 50 sentence pairs.

Two versions:
  A. Jacobian-based: full vjp at L20 → top-2 singular vectors → ||p||
  B. Weight-based:   SVD of Σ_h W_O_h W_V_h (no backward pass) → ||p||

Measures:
  - Distribution of ||p(L20)|| for factual vs fabricated
  - Threshold that best separates the classes
  - Whether weight-based approximation matches Jacobian-based
  - Comparison with σ₁(T₁₄) valve

n=25 factual + 25 fabricated sentences generated from templates.
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4
L_VALVE=20   # the layer where ||p|| gives 6x separation
L_ATT=14     # attractor center for σ₁ valve

print(f"\n{'='*65}")
print(f"  PLÜCKER L{L_VALVE} VALVE VALIDATION")
print(f"  50 sentence pairs. Jacobian vs weight-based ||p||.")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=json.load(f)
with open('/tmp/val_ids.json')   as f: val_ids=json.load(f)
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

# Build vocab lookup
if isinstance(vocab, list):
    v2i = {w:i for i,w in enumerate(vocab)}
else:
    v2i = vocab

def get_batch(split='train'):
    data=train_t if split=='train' else val_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
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
    def hidden_to(self,x,stop):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for l,b in enumerate(self.blocks):
            h=b(h)
            if l==stop: return h
        return h

def clr(s,total=300,warmup=100):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

# ── Sentence templates ────────────────────────────────────────────────────────
# 25 factual + 25 fabricated, diverse topics
FACTUAL = [
    "the earth orbits the sun once every three hundred sixty five days",
    "water freezes at zero degrees celsius at standard atmospheric pressure",
    "albert einstein was born in ulm germany in eighteen seventy nine",
    "the speed of light in a vacuum is approximately three hundred thousand kilometers per second",
    "william shakespeare was born in stratford upon avon in fifteen sixty four",
    "the french revolution began in seventeen eighty nine with the storming of the bastille",
    "dna is a double helix structure discovered by watson and crick in nineteen fifty three",
    "the great wall of china was built over many centuries to protect against invasions",
    "napoleon bonaparte was exiled to saint helena after his defeat at waterloo",
    "the pythagorean theorem states that the square of the hypotenuse equals the sum of squares",
    "oxygen has atomic number eight and is essential for human respiration",
    "the internet was developed from arpanet which began in the nineteen sixties",
    "charles darwin published on the origin of species in eighteen fifty nine",
    "the roman empire fell in four seventy six with the abdication of romulus augustulus",
    "the moon is approximately three hundred eighty four thousand kilometers from earth",
    "beethoven composed his ninth symphony while completely deaf in eighteen twenty four",
    "the magna carta was signed by king john of england in twelve fifteen",
    "penicillin was discovered by alexander fleming in nineteen twenty eight",
    "the amazon river is the largest river by water flow in the world",
    "the human genome contains approximately three billion base pairs of dna",
    "isaac newton formulated the laws of motion and universal gravitation in the seventeenth century",
    "the eiffel tower was constructed between eighteen eighty seven and eighteen eighty nine",
    "the first world war began in nineteen fourteen following the assassination of archduke franz ferdinand",
    "the mitochondria is the powerhouse of the cell and produces atp",
    "socrates was a greek philosopher who was sentenced to death in three ninety nine bc",
]

FABRICATED = [
    "the earth orbits jupiter once every forty two days at a distance of five million kilometers",
    "water freezes at fifty degrees celsius and boils at minus ten degrees under normal pressure",
    "albert einstein invented quantum teleportation in nineteen twenty three while working at mit",
    "the speed of light changes to fifty thousand kilometers per second inside water at room temperature",
    "william shakespeare was born in london in fifteen eighty two and attended oxford university",
    "the french revolution began in seventeen fifty with the invention of the printing press",
    "dna is a triple helix discovered by franklin and wilkins in nineteen forty seven",
    "the great wall of china was built in three years by emperor qin to attract tourists",
    "napoleon bonaparte won the battle of waterloo and was crowned emperor of europe in eighteen fifteen",
    "the pythagorean theorem states that the cube of the hypotenuse equals the product of the sides",
    "oxygen has atomic number twelve and was first synthesized in a laboratory in eighteen fifty",
    "the internet was invented by bill gates in nineteen eighty five as a commercial product",
    "charles darwin published on the origin of species in eighteen thirty two after visiting australia",
    "the roman empire fell in three hundred twenty with the conversion of emperor constantine to buddhism",
    "the moon is approximately four million kilometers from earth and has its own magnetic field",
    "beethoven composed his ninth symphony at age twenty while studying in paris in seventeen ninety",
    "the magna carta was signed by king richard the third of england in fourteen fifteen",
    "penicillin was discovered by louis pasteur in nineteen zero two during his work on fermentation",
    "the nile river is the largest river by water flow and spans twelve countries in asia",
    "the human genome contains approximately forty billion base pairs organized into twenty chromosomes",
    "isaac newton discovered gravity after observing a solar eclipse in the eighteenth century",
    "the eiffel tower was constructed between nineteen ten and nineteen fifteen as a war memorial",
    "the first world war began in nineteen twenty following the signing of the treaty of versailles",
    "the mitochondria produces glucose through photosynthesis using chlorophyll in animal cells",
    "socrates was a roman philosopher who founded the academy in athens in two fifty bc",
]

def encode(text):
    words = text.lower().split()
    ids = [v2i.get(w, hash(w)%VOCAB) for w in words[:SEQ]]
    if len(ids) < 4: ids = ids + [0]*(4-len(ids))
    return torch.tensor(ids, dtype=torch.long)

# ── Train model ───────────────────────────────────────────────────────────────
print("Training model (300 steps)...")
torch.manual_seed(42)
model = LM(D,N_HEADS,N_LAYERS)
opt = torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    for pg in opt.param_groups: pg['lr']=clr(step)
    model.train(); x,y=get_batch(); _,loss=model(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if step%100==0:
        model.eval()
        with torch.no_grad():
            vl=float(np.mean([model(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
        model.train()
model.eval()
print(f"  Done. t={time.time()-t0:.0f}s\n")

# ── Method A: Jacobian-based ||p(L20)|| ──────────────────────────────────────
def plucker_norm_jacobian(model, ids, l=L_VALVE, m=16):
    """Compute ||p(L_valve)|| via Jacobian vjp. Top-2 singular vectors of δJ_l."""
    x = ids.unsqueeze(0)
    with torch.no_grad():
        h = model.hidden_to(x, l)   # [1,S,D]
    h0 = h[0]   # [S,D]
    pos = min(len(ids)//2, h0.shape[0]-1)
    seq,d_ = h0.shape; m_=min(m,seq,d_)
    _,_,Vt = torch.linalg.svd(h0,full_matrices=False)
    U = Vt[:m_,:].T.detach()
    J = np.zeros((m_,m_))
    with torch.enable_grad():
        for i in range(m_):
            hh=h0.clone().unsqueeze(0).detach().requires_grad_(True)
            ho=model.blocks[l](hh)
            v=(ho[0,pos,:] if ho.dim()==3 else ho[pos,:])
            (v*U[:,i]).sum().backward()
            g=hh.grad; g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
            J[:,i]=(U.T@g).numpy()
    dJ=J.T-np.eye(m_)
    U2,_,_=np.linalg.svd(dJ); u=U2[:,:2]
    # Plücker: top-4 rows
    V4=u[:4,:]
    p=[V4[i,0]*V4[j,1]-V4[j,0]*V4[i,1] for i in range(4) for j in range(i+1,4)]
    return float(np.linalg.norm(p))

# ── Method B: Weight-based ||p(L20)|| ────────────────────────────────────────
def plucker_norm_weights(model, l=L_VALVE):
    """
    Compute ||p|| from W_O_h W_V_h without any forward pass.
    Σ_h (W_O_h W_V_h) is the weight-space analog of the attention Jacobian.
    Top-2 singular vectors of this sum → Plücker norm.
    """
    blk = model.blocks[l].attn
    dh = blk.dh; d = D
    # Sum of W_O_h @ W_V_h over heads
    WO = blk.op.weight.data   # [d, d]
    WV = blk.WV.weight.data   # [d, d]
    S = torch.zeros(d, d)
    for h in range(blk.nh):
        Wo_h = WO[:, h*dh:(h+1)*dh]   # [d, dh]
        Wv_h = WV[h*dh:(h+1)*dh, :]   # [dh, d]
        S = S + Wo_h @ Wv_h            # [d, d]
    S_np = S.detach().numpy()
    U2,_,_ = np.linalg.svd(S_np); u = U2[:,:2]
    V4 = u[:4,:]
    p = [V4[i,0]*V4[j,1]-V4[j,0]*V4[i,1] for i in range(4) for j in range(i+1,4)]
    return float(np.linalg.norm(p))

# ── Method C: σ₁(T₁₄) valve (existing) ──────────────────────────────────────
def sigma1_valve(model, ids, l=L_ATT, m=16):
    x=ids.unsqueeze(0)
    with torch.no_grad():
        h=model.hidden_to(x,l); h0=h[0]
    pos=min(len(ids)//2,h0.shape[0]-1)
    seq,d_=h0.shape; m_=min(m,seq,d_)
    _,_,Vt=torch.linalg.svd(h0,full_matrices=False)
    U=Vt[:m_,:].T.detach(); J=np.zeros((m_,m_))
    with torch.enable_grad():
        for i in range(m_):
            hh=h0.clone().unsqueeze(0).detach().requires_grad_(True)
            ho=model.blocks[l](hh)
            v=(ho[0,pos,:] if ho.dim()==3 else ho[pos,:])
            (v*U[:,i]).sum().backward()
            g=hh.grad; g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
            J[:,i]=(U.T@g).numpy()
    sv=np.linalg.svd(J.T,compute_uv=False)
    return float(sv[0])

# ── Run all 50 sentences ──────────────────────────────────────────────────────
print(f"Computing valves for {len(FACTUAL)+len(FABRICATED)} sentences...")
print("(Jacobian method: ~16 vjp passes per sentence per layer)\n")

results = []
t0=time.time()
for label,sentences in [("factual",FACTUAL),("fabricated",FABRICATED)]:
    for text in sentences:
        ids = encode(text)
        pJ  = plucker_norm_jacobian(model, ids)
        pW  = plucker_norm_weights(model, L_VALVE)
        s1  = sigma1_valve(model, ids)
        results.append({'label':label,'text':text[:40],'pJ':pJ,'pW':pW,'s1':s1})

print(f"  Done. t={time.time()-t0:.0f}s\n")

# ── Analysis ──────────────────────────────────────────────────────────────────
fact  = [r for r in results if r['label']=='factual']
fabr  = [r for r in results if r['label']=='fabricated']

pJ_f = np.array([r['pJ'] for r in fact]);  pJ_b = np.array([r['pJ'] for r in fabr])
pW_f = np.array([r['pW'] for r in fact]);  pW_b = np.array([r['pW'] for r in fabr])
s1_f = np.array([r['s1'] for r in fact]);  s1_b = np.array([r['s1'] for r in fabr])

print(f"{'='*65}")
print(f"  RESULTS ACROSS 50 SENTENCES")
print("="*65)

def stats(arr): return f"mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}"
def separation(a,b):
    # Cohen's d
    pooled=np.sqrt((a.std()**2+b.std()**2)/2)
    return abs(a.mean()-b.mean())/max(pooled,1e-8)
def best_threshold(a,b,label_a=1,label_b=0):
    # sweep threshold, find best accuracy
    all_vals=np.concatenate([a,b])
    all_labs=np.concatenate([np.ones(len(a)),np.zeros(len(b))])
    best_acc=0; best_t=0
    for t in np.linspace(all_vals.min(),all_vals.max(),200):
        pred=(all_vals>t).astype(int)  # >t → factual(1)
        acc=float(np.mean(pred==all_labs))
        if acc>best_acc: best_acc=acc; best_t=t
    return best_t, best_acc

print(f"\n  A. JACOBIAN-BASED ||p(L{L_VALVE})||:")
print(f"     factual:    {stats(pJ_f)}")
print(f"     fabricated: {stats(pJ_b)}")
print(f"     ratio means: {pJ_f.mean()/pJ_b.mean():.2f}x  |  Cohen d: {separation(pJ_f,pJ_b):.3f}")
tA,accA=best_threshold(pJ_f,pJ_b)
print(f"     Best threshold: {tA:.4f}  accuracy: {accA:.1%}")
print(f"     @ threshold 0.15: acc={float(np.mean(np.concatenate([(pJ_f>0.15),(pJ_b<=0.15)]))):.1%}")

print(f"\n  B. WEIGHT-BASED ||p(L{L_VALVE})|| (NO FORWARD PASS):")
print(f"     factual:    {stats(pW_f)}")
print(f"     fabricated: {stats(pW_b)}")
print(f"     (Weight-based is input-independent — same value for all sentences)")
print(f"     Single value: {pW_f[0]:.4f}  (cannot separate factual from fabricated)")

print(f"\n  C. σ₁(T₁₄) VALVE (existing):")
print(f"     factual:    {stats(s1_f)}")
print(f"     fabricated: {stats(s1_b)}")
print(f"     ratio means: {s1_f.mean()/s1_b.mean():.2f}x  |  Cohen d: {separation(s1_f,s1_b):.3f}")
tC,accC=best_threshold(s1_f,s1_b)
print(f"     Best threshold: {tC:.4f}  accuracy: {accC:.1%}")
print(f"     @ threshold 1.0: acc={float(np.mean(np.concatenate([(s1_f>1.0),(s1_b<=1.0)]))):.1%}")

# Per-sentence table (sample)
print(f"\n  Sample results (first 5 each class):")
print(f"  {'label':>12}  {'||p_J||':>9}  {'σ₁':>8}  {'text'}")
print("  "+"-"*60)
for r in fact[:5]+fabr[:5]:
    verdict_p="✓" if (r['label']=='factual')==(r['pJ']>tA) else "✗"
    verdict_s="✓" if (r['label']=='factual')==(r['s1']>1.0) else "✗"
    print(f"  {r['label']:>12}  {r['pJ']:>9.4f}{verdict_p}  {r['s1']:>8.4f}{verdict_s}  '{r['text']}'")

print(f"\n{'='*65}")
print(f"  COMPARISON SUMMARY")
print("="*65)
print(f"""
  Method A (Jacobian ||p(L{L_VALVE})||):
    Cohen's d = {separation(pJ_f,pJ_b):.3f}
    Best accuracy = {accA:.1%} at threshold {tA:.4f}
    Compute: ~{16} vjp passes at L{L_VALVE}

  Method C (σ₁ at L{L_ATT}):
    Cohen's d = {separation(s1_f,s1_b):.3f}
    Best accuracy = {accC:.1%} at threshold {tC:.4f}
    Compute: ~{16} vjp passes at L{L_ATT}

  Method B (weight-based):
    Input-independent → cannot detect hallucination.
    Weight-only ||p|| is the same for all inputs.
    A forward pass is required for input-dependent detection.

  WINNER: {'Method A' if separation(pJ_f,pJ_b) > separation(s1_f,s1_b) else 'Method C'}
  The better valve for production deployment.
""")
