#!/usr/bin/env python3
"""
Teacher Profiler
================
Continuous measurement of key parameters during teacher training.
Not pass/fail — quantitative profiles that reveal the structure
as it crystallizes.

WHAT WE PROFILE (every 25 steps):

1. SPECTRAL PROFILE
   - sv(M_fwd): monodromy amplification at each step
   - ||δJ_l|| per layer: which layers are active
   - rank profile: how the filtration is building

2. ORIENTATION PROFILE
   - Grassmannian distance between consecutive snapshots
   - How fast the embedding orientation is changing
   - When orientation crystallizes (rate of change → 0)

3. CASCADE ALIGNMENT
   - corr(ad(J14)^l, ad(J14_step0)^l): does the cascade structure
     persist from early training?
   - Which Serre level crystallizes first

4. EMBEDDING PROFILE
   - gram_align with corpus Laplacian: how fast does the
     teacher embedding approach the Laplacian structure?
   - row_cos trajectory: which tokens align first

5. SECTORIAL ASCENT/DESCENT
   - At each step: which layers are ascending, which descending
   - Does the 8-sector structure appear before val converges?

6. TOKEN FILTRATION DEPTH
   - For each token: at which training step does its embedding
     orientation stabilize (derivative → 0)?
   - Do high-frequency tokens stabilize first?

This answers: where does the 45% missing orientation come from,
and when does it appear during training?
"""
import json, math, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS=24; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14; PROFILE_EVERY=25

print(f"\n{'='*65}")
print(f"  TEACHER PROFILER")
print(f"  Continuous measurement every {PROFILE_EVERY} steps")
print(f"{'='*65}\n")

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: vocab=json.load(f)
VOCAB=len(vocab)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

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
    def hidden_states(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs

def clr(s,total=300,warmup=100):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model,n=30):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(l.item())
    return float(np.mean(ls))

def layer_jac_fast(block,h_in,pos,m):
    """Fast Jacobian — single reference, no averaging."""
    seq,d_=h_in.shape; m=min(m,seq,d_)
    _,_,Vt=torch.linalg.svd(h_in,full_matrices=False)
    U=Vt[:m,:].T.detach(); J=np.zeros((m,m))
    with torch.enable_grad():
        for i in range(m):
            hh=h_in.clone().unsqueeze(0).detach().requires_grad_(True)
            ho=block(hh)
            v=(ho[0,pos,:] if ho.dim()==3 else ho[pos,:])
            (v*U[:,i]).sum().backward()
            g=hh.grad; g=(g[0,pos,:] if g.dim()==3 else g[pos,:]).detach()
            J[:,i]=(U.T@g).numpy()
    return J.T,U.detach().numpy()

def extract_profile(model, x_ref, pos, m):
    """Extract key parameters at current training state."""
    with torch.no_grad():
        hs=model.hidden_states(x_ref); hs=[h[0] for h in hs]

    profile={}

    # Jacobians at key layers (fast: L0, L7, L14, L21, L23)
    key_layers=[0,7,14,21,23]
    Js={}; Us={}
    for l in key_layers:
        J,U=layer_jac_fast(model.blocks[l],hs[l],pos,m)
        Js[l]=J; Us[l]=U

    # 1. SPECTRAL: ||δJ|| and rank at key layers
    profile['norms']={l:float(np.linalg.norm(Js[l]-np.eye(m))) for l in key_layers}
    profile['ranks']={l:int(np.sum(np.linalg.svd(Js[l]-np.eye(m),
                      compute_uv=False)>0.1)) for l in key_layers}

    # 2. MONODROMY at L14
    Mf=np.eye(m)
    for l in range(15):
        J,_=layer_jac_fast(model.blocks[l],hs[l],pos,m)
        Mf=J@Mf
    sv_mf=np.linalg.svd(Mf,compute_uv=False)
    profile['sv_mfwd']=float(sv_mf[0])
    profile['sv_mfwd_mean']=float(sv_mf[:4].mean())

    # 3. EMBEDDING orientation
    E=model.te.weight.data.numpy()
    profile['emb_norm']=float(np.linalg.norm(E,'fro'))

    # 4. CASCADE ALIGNMENT: how aligned is J14 with initial J14?
    J14=Js[14]; dJ14=J14-np.eye(m)
    profile['dJ14_norm']=float(np.linalg.norm(dJ14))

    # 5. SECTORIAL: ascent/descent counts
    norms_all={}
    for l in range(N_LAYERS):
        J,_=layer_jac_fast(model.blocks[l],hs[l],pos,m)
        norms_all[l]=float(np.linalg.norm(J-np.eye(m)))
    ascents=sum(1 for l in range(1,N_LAYERS)
                if norms_all[l]>norms_all[l-1])
    profile['ascents']=ascents
    profile['descents']=N_LAYERS-1-ascents
    profile['norm_profile']=[norms_all[l] for l in range(N_LAYERS)]

    # 6. Gram alignment with corpus Laplacian
    profile['gram_align_lap']=gram_align_corpus(E)

    # 7. Token orientation change rate (vs previous snapshot)
    profile['E']=E.copy()

    return profile

def gram_align_corpus(E):
    """Measure gram alignment with Laplacian embedding."""
    G1=(E@E.T).flatten()
    G2=(E_lap@E_lap.T).flatten()
    return float(np.corrcoef(G1,G2)[0,1])

def grassmannian_dist(E1,E2,k=16):
    """Distance between top-k subspaces of two embedding matrices."""
    U1,_,_=np.linalg.svd(E1,full_matrices=False); U1=U1[:,:k]
    U2,_,_=np.linalg.svd(E2,full_matrices=False); U2=U2[:,:k]
    sv=np.linalg.svd(U1.T@U2,compute_uv=False)
    sv=np.clip(sv,0,1)
    angles=np.arccos(sv)
    return float(np.sqrt(np.sum(angles**2)))

# ── Build corpus structures ───────────────────────────────────────────────────
print("Building corpus structures...")

# Token frequencies
freq=np.zeros(VOCAB)
for t in train_ids:
    if 0<=t<VOCAB: freq[t]+=1
freq=freq/freq.sum()

# Laplacian embedding
bigram=np.zeros((VOCAB,VOCAB),dtype=np.float32)
for k in range(len(train_ids)-1):
    a,b=train_ids[k],train_ids[k+1]
    if 0<=a<VOCAB and 0<=b<VOCAB: bigram[a,b]+=1
A=(bigram+bigram.T)/2
deg=np.maximum(A.sum(1),1e-10)
D_inv_sqrt=sp.diags(1/np.sqrt(deg))
L_norm_sp=sp.eye(VOCAB)-D_inv_sqrt@sp.csr_matrix(A)@D_inv_sqrt
vals_l,vecs_l=spla.eigsh(L_norm_sp,k=min(D+2,VOCAB-1),which='SM')
idx_l=np.argsort(vals_l); vals_l=vals_l[idx_l]; vecs_l=vecs_l[:,idx_l]
skip_l=int(np.sum(vals_l<1e-8))
E_lap=vecs_l[:,skip_l:skip_l+D].astype(np.float32)
E_lap=E_lap*(13.78/max(float(np.linalg.norm(E_lap,'fro')),1e-8))
print(f"  Laplacian embedding built: {E_lap.shape}")

# Corpus cascade: ad(P)^l applied to token rows
P=bigram.copy()
rs=P.sum(1,keepdims=True); rs[rs==0]=1; P=P/rs

def corpus_cascade_depth(P,n_levels=6):
    """Token cascade depth via matrix powers P^l.
    coord[t,l] = how broadly token t spreads influence at depth l.
    High values = token reaches many contexts at this depth.
    """
    coords=np.zeros((VOCAB,n_levels),dtype=np.float32)
    current=P.copy()
    for l in range(n_levels):
        coords[:,l]=current.sum(axis=1).astype(np.float32)  # total outflow at depth l
        current=current@P  # P^{l+1}
        # Renormalize rows
        rs=current.sum(axis=1,keepdims=True); rs[rs==0]=1
        current=current/rs
    return coords

print("  Computing corpus cascade coordinates...")
cascade_coords=corpus_cascade_depth(P)
print(f"  Cascade coords: {cascade_coords.shape}")
print(f"  Mean per level: {cascade_coords.mean(0).round(4)}\n")

# ── Train teacher with profiling ──────────────────────────────────────────────
print(f"Training teacher (300 steps, profiling every {PROFILE_EVERY})...")
torch.manual_seed(42)
model=LM(D,N_HEADS,N_LAYERS)
opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)

torch.manual_seed(0)
x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
pos=SEQ//2; m=min(PROJ,SEQ,D)

profiles=[]
E_prev=None

print(f"\n  {'step':>5}  {'val':>7}  {'sv_mfwd':>9}  {'ascents':>8}  "
      f"{'gram_lap':>9}  {'grass_dist':>11}  {'norm_L14':>9}")
print("  "+"-"*68)

for step in range(0,301):
    if step>0:
        for pg in opt.param_groups: pg['lr']=clr(step)
        model.train(); x,y=get_batch(); _,loss=model(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

    if step%PROFILE_EVERY==0:
        model.eval()
        vl=eval_val(model,n=20)
        prof=extract_profile(model,x_ref,pos,m)
        prof['step']=step; prof['val']=vl

        # Grassmannian distance from previous snapshot
        if E_prev is not None:
            gdist=grassmannian_dist(prof['E'],E_prev)
        else:
            gdist=float('nan')
        prof['grass_dist']=gdist
        E_prev=prof['E'].copy()

        profiles.append(prof)
        print(f"  {step:>5}  {vl:>7.4f}  {prof['sv_mfwd']:>9.3f}  "
              f"{prof['ascents']:>8}  {prof['gram_align_lap']:>9.4f}  "
              f"{gdist if not np.isnan(gdist) else float('nan'):>11.4f}  "
              f"{prof['dJ14_norm']:>9.4f}")
        model.train()

# ── Analysis ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PROFILING ANALYSIS")
print("="*65)

steps=[p['step'] for p in profiles]
vals=[p['val'] for p in profiles]
sv_mfwds=[p['sv_mfwd'] for p in profiles]
gram_laps=[p['gram_align_lap'] for p in profiles]
grass_dists=[p['grass_dist'] for p in profiles[1:]]
ascents=[p['ascents'] for p in profiles]
dJ14_norms=[p['dJ14_norm'] for p in profiles]

print(f"""
  1. MONODROMY BUILDUP: sv(M_fwd) over training
     step 0:   {sv_mfwds[0]:.3f}  (random init)
     step 100: {sv_mfwds[4]:.3f}  (mid training)
     step 300: {sv_mfwds[-1]:.3f}  (converged)
     → Amplification builds as training progresses

  2. GRAM ALIGNMENT WITH LAPLACIAN over training
     step 0:   {gram_laps[0]:.4f}  (random emb)
     step 100: {gram_laps[4]:.4f}
     step 300: {gram_laps[-1]:.4f}  (converged)
     → Does the teacher approach the Laplacian structure?

  3. EMBEDDING ORIENTATION CHANGE RATE (Grassmannian distance)
     per {PROFILE_EVERY} steps:""")

for i,(s,gd) in enumerate(zip(steps[1:],grass_dists)):
    print(f"     step {s:>3}: {gd:.4f}")

print(f"""
  4. SECTORIAL STRUCTURE (ascents per profile)
     step 0:   {ascents[0]} ascents  (random)
     step 100: {ascents[4]} ascents
     step 300: {ascents[-1]} ascents  (converged)
     8-sector prediction: ~12 ascents in 23 transitions

  5. L14 ATTRACTOR FORMATION (||δJ14|| norm)
     step 0:   {dJ14_norms[0]:.4f}  (random)
     step 100: {dJ14_norms[4]:.4f}
     step 300: {dJ14_norms[-1]:.4f}  (attractor crystallized)

  6. CORPUS CASCADE VS EMBEDDING ALIGNMENT""")

# How well do cascade coordinates predict embedding coordinates?
E_final=model.te.weight.data.numpy()
# Correlation of cascade_coords with E_final projected onto top directions
U_E,s_E,_=np.linalg.svd(E_final,full_matrices=False)
for l in range(6):
    corr=float(np.corrcoef(cascade_coords[:,l],
                           np.abs(U_E[:,l]))[0,1])
    print(f"     Level {l+1} cascade coord corr with emb PC{l+1}: {corr:.4f}")

print(f"""
  7. WHEN DOES ORIENTATION CRYSTALLIZE?
     (Grassmannian distance rate → 0 means orientation is fixed)""")

for i in range(1,len(grass_dists)):
    if i>0:
        rate=grass_dists[i]-grass_dists[i-1]
        s=steps[i+1]
        if abs(grass_dists[i])<0.05:
            print(f"     → Orientation crystallized around step {s} "
                  f"(dist={grass_dists[i]:.4f})")
            break

print(f"""
  8. TOKEN FILTRATION DEPTH FROM CASCADE COORDS
     High cascade_depth tokens (appear late in Serre levels):""")
depths=cascade_coords.sum(1)
top_tokens=np.argsort(depths)[::-1][:10]
# Map back to token strings if possible
tok_list=list(range(VOCAB))
print(f"     Top-10 high-depth tokens (ids): {top_tokens.tolist()}")
print(f"     Their cascade depths: {depths[top_tokens].round(4)}")
print(f"     Bottom-10 low-depth (peripheral): {np.argsort(depths)[:10].tolist()}")

print(f"""
  CONCLUSION:
  The profiler reveals which aspects of the teacher's structure
  appear early (and are thus accessible from corpus statistics)
  vs late (and require the CE prediction signal).

  Early crystallizers → accessible without gradient descent
  Late crystallizers  → require training

  The gram_align trajectory shows whether the teacher's embedding
  is moving toward or away from the Laplacian structure during
  training — if it moves toward it, the Laplacian is the
  correct initialization; if away, it is diverging from corpus.
""")
