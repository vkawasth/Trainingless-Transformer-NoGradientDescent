#!/usr/bin/env python3
"""
Moran-Seidel Hybrid: Sector Identification + Moran Gradient Weighting
=======================================================================
PRECISE CONSTRUCTION:

1. SECTOR IDENTIFICATION (one forward pass, no gradient):
   For each corpus sequence x, project h_L14(x) onto the
   dominant SV subspace of M_fwd.
   Sign of projection = sector assignment (+1 or -1).
   
   Sequences with sector = +1: majority sector (near fixed point)
   Sequences with sector = -1: minority sector (far from fixed point)
   
   This is the "Seidel HMM state" — the topological sector.

2. MORAN GRADIENT WEIGHTING:
   Standard CE: all batches weighted equally.
   Moran-weighted CE: minority-sector sequences get weight
     w = N / (N - k)  where k = number of majority-sector blocks
   Majority-sector sequences get weight
     w = 1 / N
   
   This is the Moran model selection pressure:
   the minority drives fixation, majority drifts.

3. FIXATION CRITERION:
   Check sector of each student block's WK matrix.
   If all blocks have the same sector: fixation achieved.
   After fixation: switch to standard CE for within-sector alignment.

4. MORAN FIXATION TIME PREDICTION:
   For effective population N_eff ~ embedding dimension m = 48:
   T_fix ~ m * log(m) ~ 48 * ln(48) ~ 180 steps.
   This matches observed 200-step convergence.
   
   With Moran weighting favoring minority-sector sequences:
   T_fix ~ m * log(m) / selection_advantage
   where selection_advantage = fitness differential between sectors.
   Measured from the data: cos(s_3, s_{2,5,6}) ~ -0.42 vs cos=+0.87
   within cluster. Selection advantage ~ (0.87 - (-0.42)) / 2 ~ 0.65.
   Predicted T_fix ~ 180 / 0.65 ~ 277... but with 6 blocks N_eff ~ 6*48/6 = 48.
   
   Actually: with Moran weighting concentrated on minority sector,
   each gradient step is 1/k more informative for fixation.
   Predicted speedup: 1/fraction_minority ~ 1/0.15 ~ 7x for the fixation phase.
"""
import json, math, time, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_LAYERS_T=24; N_STU=6; BATCH=8; SEQ=64; LR=3e-4; PROJ=48
L_ATT=14; DEHN_GAP=1.4; N_SECTOR_SEQS=200

print(f"\n{'='*65}")
print(f"  MORAN-SEIDEL HYBRID")
print(f"  Sector identification + Moran gradient weighting")
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
    def forward(self,x,y=None,seq_weights=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        if y is None: return logits,None
        losses=F.cross_entropy(logits.view(-1,VOCAB),y.view(-1),reduction='none')
        losses=losses.view(x.shape[0],-1).mean(dim=1)  # (B,) per-sequence losses
        if seq_weights is not None:
            loss=(losses*seq_weights).sum()/seq_weights.sum()
        else:
            loss=losses.mean()
        return logits,loss
    def hidden_states(self,x):
        hs=[]; h=self.te(x)+self.pe(torch.arange(x.shape[1])); hs.append(h.detach())
        for b in self.blocks: h=b(h); hs.append(h.detach())
        return hs

def clr(s,total=200,warmup=50):
    if s<=warmup: return LR*s/warmup
    return LR*0.5*(1+math.cos(math.pi*(s-warmup)/(total-warmup)))

def eval_val(model,n=60):
    model.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get_batch('val'); _,l=model(x,y); ls.append(float(l))
    return float(np.mean(ls))

def layer_jac(block,h_in,pos,m):
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

def comm(A,B): return A@B-B@A
def l3(a,b,c): return comm(comm(a,b),c)-comm(a,comm(b,c))
def N(A): return float(np.linalg.norm(A))
def lift_to_d(C,U,scale=0.01):
    UU=U@U.T
    return (U@C@U.T+(np.eye(D)-UU)*scale).astype(np.float32)
def mu4(a,b,c,d):
    return -(comm(l3(a,b,c),d)-l3(comm(a,b),c,d)+l3(a,comm(b,c),d)
            -l3(a,b,comm(c,d))+comm(a,l3(b,c,d)))
def mu5(a,b,c,d,e):
    return -(l3(l3(a,b,c),d,e)-l3(a,l3(b,c,d),e)+l3(a,b,l3(c,d,e))
            +comm(mu4(a,b,c,d),e)+comm(a,mu4(b,c,d,e))
            -mu4(comm(a,b),c,d,e)+mu4(a,b,c,comm(d,e)))
def mu6(a,b,c,d,e,f):
    m5ab=mu5(a,b,c,d,e); m5bc=mu5(b,c,d,e,f)
    m4ab=mu4(a,b,c,d); m4bc=mu4(b,c,d,e); m4cd=mu4(c,d,e,f)
    m3ab=l3(a,b,c); m3bc=l3(b,c,d); m3cd=l3(c,d,e); m3de=l3(d,e,f)
    return -(comm(m5ab,f)-comm(a,m5bc)+l3(m4ab,e,f)-l3(a,m4bc,f)
            +l3(a,b,m4cd)+mu4(m3ab,d,e,f)-mu4(a,m3bc,e,f)
            +mu4(a,b,m3cd,f)-mu4(a,b,c,m3de))

# ════════════════════════════════════════════════════
# Train teacher + extract operators
# ════════════════════════════════════════════════════
print("Training teacher (300 steps)...")
torch.manual_seed(42)
teacher=LM(D,N_HEADS,N_LAYERS_T)
opt=torch.optim.AdamW(teacher.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
t0=time.time()
for step in range(1,301):
    lr_now=LR*min(step,100)/100 if step<=100 else \
           LR*0.5*(1+math.cos(math.pi*(step-100)/200))
    for pg in opt.param_groups: pg['lr']=lr_now
    teacher.train(); x,y=get_batch(); _,loss=teacher(x,y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
    if step%100==0:
        teacher.eval()
        with torch.no_grad():
            vl=float(np.mean([teacher(*get_batch('val'))[1].item() for _ in range(10)]))
        print(f"  step {step}  val={vl:.4f}  t={time.time()-t0:.0f}s")
        teacher.train()
teacher.eval(); val_teacher=eval_val(teacher)
print(f"  Teacher val={val_teacher:.4f}\n")

torch.manual_seed(0); pos=SEQ//2; m=min(PROJ,SEQ,D); ma=None
J_acc=[[] for _ in range(N_LAYERS_T)]; U_acc=[[] for _ in range(N_LAYERS_T)]
for ref in range(5):
    x_ref,_=get_batch('val'); x_ref=x_ref[0:1]
    with torch.no_grad(): hs=teacher.hidden_states(x_ref); hs=[h[0] for h in hs]
    for l in range(N_LAYERS_T):
        J,U=layer_jac(teacher.blocks[l],hs[l],pos,m)
        J_acc[l].append(J); U_acc[l].append(U)
        if ma is None: ma=J.shape[0]
    if (ref+1)%3==0: print(f"  ref {ref+1}/5...",flush=True)
Js=[np.mean(J_acc[l],axis=0) for l in range(N_LAYERS_T)]
Us=[np.mean(U_acc[l],axis=0) for l in range(N_LAYERS_T)]
J14=Js[L_ATT]; U14=Us[L_ATT]

# Monodromy stable subspace — defines the sectors
M_fwd=np.eye(ma)
for l in range(L_ATT+1): M_fwd=Js[l]@M_fwd
U_M,sv_M,_=np.linalg.svd(M_fwd)
sector_axis=U_M[:,0]  # dominant SV direction = sector separator
print(f"  Monodromy sv[:3]: {sv_M[:3].round(3)}")
print(f"  Sector axis ||u_1|| = {np.linalg.norm(sector_axis):.4f}\n")

# Cascades
cascade_serre=[]
for l in range(1,N_STU+1):
    C=Js[min(L_ATT+l,N_LAYERS_T-1)].copy()
    for _ in range(l): C=comm(J14,C)
    cascade_serre.append(C/max(N(C),1e-8))

att_basin=[l for l in range(8,21) if N(Js[l]-np.eye(ma))<0.75]
combos=list(itertools.combinations(att_basin,6))
scored=sorted([(c,N(mu6(*[Js[i] for i in c]))) for c in combos],key=lambda x:-x[1])
cascade_prime=[mu6(*[Js[i] for i in c])/max(N(mu6(*[Js[i] for i in c])),1e-8)
               for c,_ in scored[:N_STU]]

# ════════════════════════════════════════════════════
# STEP 1: SECTOR IDENTIFICATION
# ════════════════════════════════════════════════════
print("="*65)
print(f"STEP 1: SECTOR IDENTIFICATION ({N_SECTOR_SEQS} sequences)")
print("  Project h_L14 onto monodromy dominant SV direction")
print("  Sign = sector (+1 majority / -1 minority)")
print("="*65)

sector_labels=[]  # +1 or -1 per sequence
sector_positions=[]  # corpus positions

torch.manual_seed(0)
for i in range(N_SECTOR_SEQS):
    ix=torch.randint(0,len(train_t)-SEQ-1,(1,))[0].item()
    x_seq=train_t[ix:ix+SEQ].unsqueeze(0)
    teacher.eval()
    with torch.no_grad():
        hs=teacher.hidden_states(x_seq)
        h_att=hs[L_ATT][0,pos,:ma].numpy()
    proj=float(h_att@sector_axis)
    sector=+1 if proj>=0 else -1
    sector_labels.append(sector)
    sector_positions.append(ix)

sector_labels=np.array(sector_labels)
n_pos=int(np.sum(sector_labels>0))
n_neg=int(np.sum(sector_labels<0))
frac_minority=min(n_pos,n_neg)/N_SECTOR_SEQS

print(f"\n  Sector distribution:")
print(f"  Majority (+1): {max(n_pos,n_neg)}/{N_SECTOR_SEQS} "
      f"({100*max(n_pos,n_neg)/N_SECTOR_SEQS:.1f}%)")
print(f"  Minority (-1): {min(n_pos,n_neg)}/{N_SECTOR_SEQS} "
      f"({100*frac_minority:.1f}%)")
print(f"\n  Moran fixation time prediction:")
print(f"  N_eff ~ m * log(m) = {ma} * ln({ma}) = {ma*math.log(ma):.0f} steps")
print(f"  Selection advantage ~ (cos_intra - cos_inter)/2 = "
      f"({0.87-(-0.42)}/2) = 0.65")
print(f"  T_fix ~ N_eff / s = {ma*math.log(ma)/0.65:.0f} steps")
print(f"  With Moran weighting: T_fix / (1/f_min) = "
      f"{ma*math.log(ma)/0.65*frac_minority:.0f} steps")

# Precompute sector labels for training pool
majority_sign=+1 if n_pos>n_neg else -1
minority_sign=-majority_sign

# Moran weights: minority gets N/(N-k) weight, majority gets 1/N
# where k = number of majority-sector blocks (approximated as N_STU-1)
k_majority=N_STU-1
moran_weight_minority=N_STU/(N_STU-k_majority)  # = N_STU
moran_weight_majority=1.0/N_STU
print(f"\n  Moran weights:")
print(f"  Minority weight: {moran_weight_minority:.2f}x")
print(f"  Majority weight: {moran_weight_majority:.4f}x")

# ════════════════════════════════════════════════════
# STEP 2: SECTOR-AWARE BATCH SAMPLING
# ════════════════════════════════════════════════════
minority_positions=np.array([p for p,s in zip(sector_positions,sector_labels)
                              if s==minority_sign])
majority_positions=np.array([p for p,s in zip(sector_positions,sector_labels)
                              if s==majority_sign])
print(f"\n  Minority positions: {len(minority_positions)}")
print(f"  Majority positions: {len(majority_positions)}")

def get_moran_batch():
    """
    Moran-weighted batch: oversample minority sector.
    Half the batch from minority, half from majority.
    (Even split maximizes information about fixation direction)
    """
    half=BATCH//2
    if len(minority_positions)>=half:
        min_idx=np.random.choice(len(minority_positions),half,replace=True)
        min_pos=minority_positions[min_idx]
    else:
        min_pos=np.random.choice(len(train_t)-SEQ-1,half)

    maj_idx=np.random.choice(len(majority_positions),BATCH-half,replace=True)
    maj_pos=majority_positions[maj_idx]

    all_pos=np.concatenate([min_pos,maj_pos])
    x=torch.stack([train_t[i:i+SEQ] for i in all_pos])
    y=torch.stack([train_t[i+1:i+SEQ+1] for i in all_pos])

    # Moran weights: minority gets higher weight
    weights=torch.ones(BATCH)
    weights[:half]=moran_weight_minority
    weights[half:]=moran_weight_majority
    weights=weights/weights.sum()
    return x,y,weights

def get_sector_of_block(stu, block_idx):
    """Check which sector a student block's WK is in."""
    WK=stu.blocks[block_idx].attn.WK.weight.data.numpy()
    # Project WK onto sector axis
    proj=float(WK[:ma,:ma]@sector_axis@sector_axis)
    return +1 if proj>=0 else -1

def check_fixation(stu):
    """All blocks in same sector = fixation achieved."""
    sectors=[get_sector_of_block(stu,i) for i in range(N_STU)]
    return len(set(sectors))==1, sectors

# ════════════════════════════════════════════════════
# STEP 3: STUDENT EXPERIMENTS
# ════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 3: STUDENT EXPERIMENTS")
print("  A: Serre + standard Adam (baseline)")
print("  B: Prime + standard Adam (best confirmed)")
print("  C: Prime + Moran-weighted Adam (minority sector oversampled)")
print("  D: Prime + Moran-weighted until fixation, then standard")
print("="*65)

def build_student(cascade):
    torch.manual_seed(99)
    stu=LM(D,N_HEADS,N_STU)
    stu.te.weight.data.copy_(teacher.te.weight.data)
    with torch.no_grad():
        stu.pe.weight.copy_(teacher.pe.weight)
        stu.ln_f.weight.copy_(teacher.ln_f.weight)
        stu.ln_f.bias.copy_(teacher.ln_f.bias)
        for l in range(N_STU):
            W_d=lift_to_d(cascade[l],U14,scale=0.01)
            W_t=torch.tensor(W_d,dtype=torch.float32)
            stu.blocks[l].attn.WK.weight.copy_(W_t)
            stu.blocks[l].attn.WQ.weight.copy_(W_t.T)
            stu.blocks[l].attn.WV.weight.copy_(teacher.blocks[L_ATT].attn.WV.weight)
            stu.blocks[l].attn.op.weight.copy_(teacher.blocks[L_ATT].attn.op.weight)
            stu.blocks[l].ff.g.weight.copy_(teacher.blocks[L_ATT].ff.g.weight)
            stu.blocks[l].ff.v.weight.copy_(teacher.blocks[L_ATT].ff.v.weight)
            stu.blocks[l].ff.o.weight.copy_(teacher.blocks[L_ATT].ff.o.weight)
    return stu

def run_standard(cascade,label,steps=200):
    stu=build_student(cascade)
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={}
    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)
        stu.train(); x,y=get_batch()
        _,loss=stu(x,y)
        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()
        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck

def run_moran(cascade,label,steps=200,switch_after_fixation=False):
    stu=build_student(cascade)
    v0=eval_val(stu,n=20); print(f"\n  [{label}] zero-shot={v0:.4f}")
    opt_s=torch.optim.AdamW(stu.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
    ck={}; fixation_step=None; mode='moran'

    for step in range(1,steps+1):
        for pg in opt_s.param_groups: pg['lr']=clr(step,steps)

        # Check fixation
        if mode=='moran' and step%10==0:
            fixed,sectors=check_fixation(stu)
            if fixed:
                fixation_step=step
                if switch_after_fixation:
                    mode='standard'
                    print(f"  [{label}] FIXATION at step {step}  "
                          f"sectors={sectors}  → switching to standard")

        stu.train()
        if mode=='moran':
            x,y,w=get_moran_batch()
            _,loss=stu(x,y,seq_weights=w)
        else:
            x,y=get_batch(); _,loss=stu(x,y)

        opt_s.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(stu.parameters(),1.0); opt_s.step()

        if step in [25,50,75,100,125,150,200]:
            v=eval_val(stu,n=20); ck[step]=v
            _,sects=check_fixation(stu)
            print(f"  [{label}] step {step:>4}  val={v:.4f}"
                  f"  sectors={sects}"
                  f"{' ✓' if v<val_teacher else ''}")
    return eval_val(stu),ck,fixation_step

vA,ckA=run_standard(cascade_serre,"A-Serre-std")
vB,ckB=run_standard(cascade_prime,"B-Prime-std")
vC,ckC,fx_C=run_moran(cascade_prime,"C-Prime-Moran",switch_after_fixation=False)
vD,ckD,fx_D=run_moran(cascade_prime,"D-Prime-Moran-switch",switch_after_fixation=True)

print(f"\n{'='*65}")
print("  MORAN-SEIDEL RESULTS")
print("="*65)
print(f"""
  SECTOR ANALYSIS:
    Minority sector fraction: {100*frac_minority:.1f}%
    Predicted Moran fixation time: ~{int(ma*math.log(ma)/0.65*frac_minority)} steps
    Observed fixation step C: {fx_C or 'not reached'}
    Observed fixation step D: {fx_D or 'not reached'}

  CONVERGENCE:
  {'step':>6}  {'A-Serre':>8}  {'B-Prime':>8}  {'C-Moran':>8}  {'D-M-swch':>9}""")
for s in [25,50,75,100,125,150,200]:
    a=ckA.get(s); b=ckB.get(s); c=ckC.get(s); d=ckD.get(s)
    row=f"  {s:>6}"
    for v in [a,b,c,d]:
        row+=f"  {v:>8.4f}" if v else f"  {'---':>8}"
    best=min((v for v in [b,c,d] if v),default=99)
    if a and best<a-0.003: row+=" ←"
    print(row)

print(f"""
  FINAL:
    Teacher:              val={val_teacher:.4f}
    A (Serre+std):        val={vA:.4f}
    B (Prime+std):        val={vB:.4f}  diff={vA-vB:+.4f}
    C (Moran-weighted):   val={vC:.4f}  diff={vA-vC:+.4f}
    D (Moran→standard):   val={vD:.4f}  diff={vA-vD:+.4f}

  IF C or D beats A at steps 25-75:
    Moran weighting accelerates fixation — the minority sector
    sequences carry disproportionate gradient information.
    The Seidel sector identification correctly separates the
    corpus into majority (near fixed point) and minority
    (driving fixation) subsets.

  IF fixation is observed:
    The Moran model is the correct dynamics for this system.
    Fixation step ~ predicted T_fix validates the population genetics
    interpretation of transformer training.

  IF C ≈ D ≈ A:
    The sector identification is not discriminating — all sequences
    activate both sectors with equal frequency. The corpus is uniform
    with respect to the monodromy sectors. Fixation is not a useful
    concept here because the minority fraction is too small or
    the selection advantage is too weak for the gradient to detect.
""")
