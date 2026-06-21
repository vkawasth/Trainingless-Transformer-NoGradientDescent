#!/usr/bin/env python3
"""
Path Characterizer — Geometric Instrumentation of Optimization Paths
=====================================================================
Maps any optimization path into (Φ, σ, τ) phase-spectrum-defect space.

COORDINATE SYSTEM:
  Φ = (ϕ₀,...,ϕ₄)  Bridgeland sheet angles at each layer transition
  σ = (σ₀,...,σ₅)  log singular values of W_K at each layer
  τ = [τ]          K₀ extension class / gluing defect

DERIVED COORDINATES (for path comparison):
  x = val_t / val_final          normalized progress toward entropy floor
  y = ||σ_t - σ_Serre||₂        spectral distance from Serre decay
  z = ||∇_FF L|| / ||∇_Emb L||  dynamic w_FF ratio (gluing defect proxy)

ANCHORS (topology-change events):
  P0: Saddle Exit   — λ_min(H) crosses 0 (neg→pos curvature)
  P1: TopoGate      — sheet angles snap to {0,π} from messy
  P2: Orbit Lock    — Φ stabilizes (dΦ/dt < ε)
  P3: Serre Point   — d log σ/dl ≈ -0.843 (Serre decay rate fit)
  P4: Saturation    — dval/dt → 0 (entropy floor)

USAGE:
  pc = PathCharacterizer(model, val_fn)
  pc.snapshot("init")
  ... training ...
  pc.snapshot("step_25")
  pc.print_path()
  pc.compare(other_pc, label="GD-400 vs Compiler")
"""
import json, math, warnings, collections, os, sys, time, copy
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D=256; N_HEADS=4; N_STU=6; BATCH=8; SEQ=64; LR=3e-4
SERRE_SLOPE = -0.843   # document: log||ad(J14)^k|| = -0.843k + 0.665

for f in ['/tmp/train_ids.json','/tmp/val_ids.json','/tmp/vocab.json']:
    if not os.path.exists(f):
        print(f"ERROR: {f} missing."); sys.exit(1)

with open('/tmp/train_ids.json') as f: train_ids=list(map(int,json.load(f)))
with open('/tmp/val_ids.json')   as f: val_ids  =list(map(int,json.load(f)))
with open('/tmp/vocab.json')     as f: _v=json.load(f)
VOCAB=len(_v) if isinstance(_v,list) else len(_v)
train_t=torch.tensor(train_ids,dtype=torch.long)
val_t  =torch.tensor(val_ids,  dtype=torch.long)

class Attn(nn.Module):
    def __init__(self):
        super().__init__(); dh=D//N_HEADS
        self.WQ=nn.Linear(D,D,bias=False); self.WK=nn.Linear(D,D,bias=False)
        self.WV=nn.Linear(D,D,bias=False); self.op=nn.Linear(D,D,bias=False)
        self.ln=nn.LayerNorm(D); self.sc=math.sqrt(dh); self.nh=N_HEADS; self.dh=dh
        for w in [self.WQ,self.WK,self.WV,self.op]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h):
        B,S,_=h.shape
        Q=self.WQ(h).view(B,S,self.nh,self.dh).transpose(1,2)
        K=self.WK(h).view(B,S,self.nh,self.dh).transpose(1,2)
        V=self.WV(h).view(B,S,self.nh,self.dh).transpose(1,2)
        sc=Q@K.transpose(-2,-1)/self.sc
        mask=torch.triu(torch.ones(S,S),diagonal=1).bool()
        sc=sc.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        return self.ln(h+self.op((F.softmax(sc,dim=-1)@V).transpose(1,2).reshape(B,S,D)))
class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.g=nn.Linear(D,D*2,bias=False); self.v=nn.Linear(D,D*2,bias=False)
        self.o=nn.Linear(D*2,D,bias=False); self.n=nn.LayerNorm(D)
        for w in [self.g,self.v,self.o]: nn.init.normal_(w.weight,std=0.02)
    def forward(self,h): return self.n(h+self.o(F.silu(self.g(h))*self.v(h)))
class Block(nn.Module):
    def __init__(self): super().__init__(); self.attn=Attn(); self.ff=FF()
    def forward(self,h): return self.ff(self.attn(h))
class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.te=nn.Embedding(VOCAB,D); self.pe=nn.Embedding(512,D)
        self.blocks=nn.ModuleList([Block() for _ in range(N_STU)])
        self.ln_f=nn.LayerNorm(D); self.head=nn.Linear(D,VOCAB,bias=False)
        self.head.weight=self.te.weight
        nn.init.normal_(self.te.weight,std=0.02); nn.init.normal_(self.pe.weight,std=0.02)
    def forward(self,x,y=None):
        h=self.te(x)+self.pe(torch.arange(x.shape[1]))
        for b in self.blocks: h=b(h)
        logits=self.head(self.ln_f(h))
        return logits,(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)) if y is not None else None)

def get_batch(split='train'):
    data=val_t if split=='val' else train_t
    ix=torch.randint(0,len(data)-SEQ-1,(BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

def eval_val(m, n=12):
    m.eval(); ls=[]
    with torch.no_grad():
        for _ in range(n): x,y=get_batch('val'); _,l=m(x,y); ls.append(l.item())
    return float(np.mean(ls))

# ── PATH CHARACTERIZER ────────────────────────────────────────
class PathCharacterizer:
    """Records (Φ, σ, τ, val, step) snapshots and detects anchor events."""

    def __init__(self, name="path"):
        self.name = name
        self.records = []      # list of dicts
        self.anchors = {}      # event_name → record index
        self._prev_phi = None  # for orbit-lock detection

    def snapshot(self, model, label, step=0, val=None):
        """Capture full geometric state of model."""
        if val is None: val = eval_val(model)
        phi   = self._sheet_angles(model)
        sigma = self._log_svs(model)
        tau   = self._gluing_defect(model)
        serre_dist = self._serre_distance(sigma)
        serre_slope, serre_r2 = self._serre_fit(sigma)
        phi_clean = sum(1 for p in phi if abs(p) < 0.3 or abs(abs(p)-math.pi) < 0.3)

        rec = dict(
            label=label, step=step, val=val,
            phi=phi, sigma=sigma, tau=tau,
            serre_dist=serre_dist,
            serre_slope=serre_slope, serre_r2=serre_r2,
            phi_clean=phi_clean,   # how many layers have {0,π} angles
        )
        self.records.append(rec)

        # Auto-detect anchors
        idx = len(self.records) - 1
        if self._prev_phi is not None:
            dphi = sum(abs(a-b) for a,b in zip(phi, self._prev_phi))
            if dphi < 0.1 and phi_clean >= 4 and 'orbit_lock' not in self.anchors:
                self.anchors['P2_orbit_lock'] = idx
        if phi_clean >= 4 and 'topo_gate' not in self.anchors and idx > 0:
            self.anchors['P1_topo_gate'] = idx
        if abs(serre_slope - SERRE_SLOPE) < 0.2 and serre_r2 > 0.8:
            if 'serre_point' not in self.anchors:
                self.anchors['P3_serre_point'] = idx
        self._prev_phi = phi
        return rec

    def print_path(self, show_sigma=False):
        """Print path as table."""
        print(f"\n  PATH: {self.name}")
        print(f"  {'Label':<22} {'step':>5} {'val':>7} "
              f"{'Φ':>32} {'τ':>6} {'Serre_d':>8} {'clean':>5}")
        print("  "+"-"*85)
        for r in self.records:
            phi_str = '(' + ','.join(
                '0' if abs(p)<0.3 else 'π' if abs(abs(p)-math.pi)<0.3
                else f'{p:.2f}' for p in r['phi']
            ) + ')'
            anchor = next((k for k,v in self.anchors.items() if v==self.records.index(r)),'')
            print(f"  {r['label']:<22} {r['step']:>5} {r['val']:>7.4f} "
                  f"{phi_str:>32} {r['tau']:>6.3f} {r['serre_dist']:>8.3f} "
                  f"{r['phi_clean']}/5 {anchor}")
        if self.anchors:
            print(f"\n  ANCHORS detected: {list(self.anchors.keys())}")

    def compare(self, other, label=None):
        """Compare two paths in (x,y,z) normalized coordinates."""
        if not self.records or not other.records: return
        v_final_a = self.records[-1]['val']
        v_final_b = other.records[-1]['val']
        print(f"\n  PATH COMPARISON: {self.name} vs {other.name}")
        print(f"  Coordinate space: x=val_progress, y=Serre_dist, z=defect τ")
        print(f"  {'Label':<20} {'x (progress)':>14} {'y (Serre_d)':>12} {'z (τ)':>8} {'Φ_clean':>8}")
        print("  "+"-"*66)
        # Match by val level
        val_targets = [4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.3, 0.2, 0.1]
        for vt in val_targets:
            ra = next((r for r in self.records if r['val']<=vt), None)
            rb = next((r for r in other.records if r['val']<=vt), None)
            if ra is None and rb is None: continue
            xa = ra['val']/v_final_a if ra else float('nan')
            xb = rb['val']/v_final_b if rb else float('nan')
            ya = ra['serre_dist'] if ra else float('nan')
            yb = rb['serre_dist'] if rb else float('nan')
            za = ra['tau'] if ra else float('nan')
            zb = rb['tau'] if rb else float('nan')
            ca = ra['phi_clean'] if ra else -1
            cb = rb['phi_clean'] if cb else -1
            print(f"  val≤{vt:.1f}:")
            if ra: print(f"    {self.name:<18} step={ra['step']:>4} x={xa:.3f} y={ya:.3f} z={za:.3f} Φ={ca}/5")
            if rb: print(f"    {other.name:<18} step={rb['step']:>4} x={xb:.3f} y={yb:.3f} z={zb:.3f} Φ={cb}/5")

    # ── GEOMETRIC PRIMITIVES ──────────────────────────────────
    def _sheet_angles(self, model):
        angles=[]
        WKs=[model.blocks[l].attn.WK.weight.data.float() for l in range(N_STU)]
        for l in range(N_STU-1):
            try:
                phi=WKs[l+1]@torch.linalg.pinv(WKs[l])
                lam=torch.linalg.eigvals(phi); lam1=lam[lam.abs().argmax()]
                angles.append(float(torch.angle(lam1)))
            except: angles.append(float('nan'))
        return angles

    def _log_svs(self, model):
        svs=[]
        for l in range(N_STU):
            sv=float(torch.linalg.svdvals(model.blocks[l].attn.WK.weight.data)[0])
            svs.append(math.log(max(sv,1e-8)))
        return svs

    def _gluing_defect(self, model):
        """τ = ||∇_FF L|| / ||∇_Emb L||  (K₀ extension class proxy)."""
        model.zero_grad()
        ls=[model(*get_batch())[1] for _ in range(6)]
        torch.stack(ls).mean().backward()
        g_ff=sum(p.grad.data.norm().item()
                 for n,p in model.named_parameters() if '.ff.' in n and p.grad is not None)
        g_emb=model.te.weight.grad.data.norm().item() if model.te.weight.grad is not None else 1e-8
        model.zero_grad()
        return g_ff / max(g_emb, 1e-8)

    def _serre_distance(self, log_svs):
        """||σ_t - σ_Serre||₂ where σ_Serre(l) = σ₀ + SERRE_SLOPE·l."""
        if not log_svs or all(math.isnan(x) for x in log_svs): return float('nan')
        ls=list(range(len(log_svs)))
        sigma0=log_svs[0]
        serre=[sigma0+SERRE_SLOPE*l for l in ls]
        return float(np.sqrt(sum((a-b)**2 for a,b in zip(log_svs,serre))))

    def _serre_fit(self, log_svs):
        """Fit log σ(l) = slope·l + intercept. Return slope, R²."""
        n=len(log_svs); ls=list(range(n))
        if n<2: return float('nan'), float('nan')
        A=np.vstack([ls,np.ones(n)]).T
        try:
            slope,intercept=np.linalg.lstsq(A,log_svs,rcond=None)[0]
            fitted=[slope*l+intercept for l in ls]
            res=[log_svs[l]-fitted[l] for l in range(n)]
            r2=1-np.var(res)/max(np.var(log_svs),1e-10)
            return float(slope), float(r2)
        except: return float('nan'), float('nan')




# ══════════════════════════════════════════════════════════════
# SYMPLECTIC ACTION FUNCTIONAL — ENERGY FIELD
# ══════════════════════════════════════════════════════════════

class SymplecticEnergyField:
    """
    S(u) = ω_kinetic + H_potential

    ω_kinetic  = ||Δθ||² / |ΔL|   [strip area proxy: displacement per loss drop]
    H_potential = Σ_l min(|ϕ_l|, |ϕ_l - π|) [Bridgeland wall energy]
    S_total    = ω + H              [total action]

    Additional Stokes invariants:
    stokes_rate:  ϕ_l crossings of {0,π} per step   [chamber transitions]
    m2_density:   ||∇²L|| / ||∇L||                  [topological loop density]
    hess_sign:    sign(λ_min(H))                     [valley vs saddle indicator]

    GD path:       high ω, high stokes_rate, high m2_density  → dissipative
    Compiler path: low ω,  low stokes_rate,  low m2_density   → adiabatic
    """

    def __init__(self, name="energy"):
        self.name = name
        self.history = []          # (step, val, S, omega, H_pot, stokes, m2)
        self._prev_theta = None
        self._prev_val   = None
        self._prev_phi   = None

    def measure(self, model, step, val, n_hvp=4):
        """Measure full energy field at current state."""
        phi = self._sheet_angles(model)

        # ω kinetic: ||Δθ||² / |ΔL|  (strip area)
        theta = torch.cat([p.data.flatten() for p in model.parameters()])
        if self._prev_theta is not None and self._prev_val is not None:
            dtheta = float((theta - self._prev_theta).norm())
            dval   = abs(val - self._prev_val)
            omega  = (dtheta**2) / max(dval, 1e-6)
        else:
            omega = float('nan')

        # H potential: Bridgeland wall energy (distance from {0,π})
        H_pot = sum(min(abs(p), abs(abs(p)-math.pi)) for p in phi if not math.isnan(p))

        # S total
        S = (omega if not math.isnan(omega) else 0) + H_pot

        # Stokes rate: chamber transitions since last snapshot
        stokes = 0
        if self._prev_phi is not None:
            for p_now, p_prev in zip(phi, self._prev_phi):
                if math.isnan(p_now) or math.isnan(p_prev): continue
                # transition = crossed 0 or π boundary
                def chamber(p):
                    if abs(p) < 0.5: return 'A'           # near 0
                    elif abs(abs(p)-math.pi) < 0.5: return 'B'  # near π
                    else: return 'C'                        # intermediate
                if chamber(p_now) != chamber(p_prev): stokes += 1

        # m₂ loop density: ||Hv|| / ||g|| where v = normalized gradient
        try:
            model.zero_grad()
            ls=[model(*get_batch())[1] for _ in range(n_hvp)]
            loss=torch.stack(ls).mean(); loss.backward()
            g=torch.cat([p.grad.flatten() if p.grad is not None
                         else torch.zeros(p.numel()) for p in model.parameters()]).detach()
            gnorm=float(g.norm()); model.zero_grad()
            if gnorm > 1e-6:
                v=g/gnorm
                model.zero_grad()
                ls2=[model(*get_batch())[1] for _ in range(n_hvp)]
                loss2=torch.stack(ls2).mean()
                grads=torch.autograd.grad(loss2,list(model.parameters()),create_graph=True)
                gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
                hv=torch.cat([h.flatten() for h in
                              torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)])
                model.zero_grad()
                m2_density = float(hv.norm()) / max(gnorm, 1e-6)
            else:
                m2_density = float('nan')
        except:
            m2_density = float('nan')

        rec = dict(step=step, val=val, S=S, omega=omega, H_pot=H_pot,
                   stokes=stokes, m2_density=m2_density, phi=phi)
        self.history.append(rec)

        # Update state
        self._prev_theta = theta.clone()
        self._prev_val   = val
        self._prev_phi   = phi
        return rec

    def print_field(self):
        print(f"\n  ENERGY FIELD: {self.name}")
        print(f"  {'step':>5} {'val':>7} {'S(u)':>8} {'ω_kin':>8} "
              f"{'H_pot':>7} {'Stokes':>7} {'m₂_dens':>9} {'chamber'}") 
        print("  "+"-"*80)
        for r in self.history:
            phi=r['phi']
            ch='ORBIT' if sum(1 for p in phi if abs(p)<0.5 or abs(abs(p)-math.pi)<0.5)>=4 \
               else 'MIXED' if sum(1 for p in phi if abs(p)<0.5 or abs(abs(p)-math.pi)<0.5)>=2 \
               else 'OFF'
            omega_s = f"{r['omega']:>8.2f}" if not math.isnan(r.get('omega',float('nan'))) else f"{'---':>8}"
            m2_s = f"{r['m2_density']:>9.3f}" if r['m2_density'] and not math.isnan(r['m2_density']) else f"{'---':>9}"
            print(f"  {r['step']:>5} {r['val']:>7.4f} {r['S']:>8.3f} "
                  f"{omega_s} {r['H_pot']:>7.3f} {r['stokes']:>7d} {m2_s} {ch}")

    def _sheet_angles(self, model):
        angles=[]
        WKs=[model.blocks[l].attn.WK.weight.data.float() for l in range(N_STU)]
        for l in range(N_STU-1):
            try:
                phi=WKs[l+1]@torch.linalg.pinv(WKs[l])
                lam=torch.linalg.eigvals(phi); lam1=lam[lam.abs().argmax()]
                angles.append(float(torch.angle(lam1)))
            except: angles.append(float('nan'))
        return angles

    @staticmethod
    def compare_fields(ef_a, ef_b):
        """Compare two energy fields at equal val levels. Predicts which path wins."""
        print(f"\n  ENERGY FIELD COMPARISON: {ef_a.name} vs {ef_b.name}")
        print(f"  Prediction: lower S(u) + lower stokes_rate = adiabatic path = lower final val")
        print()
        print(f"  {'val≤':>6} | {'S_A':>8} {'ω_A':>8} {'H_A':>7} {'m2_A':>7} "
              f"| {'S_B':>8} {'ω_B':>8} {'H_B':>7} {'m2_B':>7} | winner")
        print("  "+"-"*80)
        for vt in [4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.3]:
            ra = next((r for r in ef_a.history if r['val']<=vt), None)
            rb = next((r for r in ef_b.history if r['val']<=vt), None)
            if ra is None and rb is None: continue
            def fmt(r, key):
                if r is None: return f"{'---':>8}"
                v=r.get(key,float('nan'))
                if math.isnan(v if v is not None else float('nan')): return f"{'---':>8}"
                return f"{v:>8.3f}"
            Sa=ra['S'] if ra else float('inf')
            Sb=rb['S'] if rb else float('inf')
            winner = ef_a.name if Sa<Sb else (ef_b.name if Sb<Sa else '=')
            print(f"  {vt:>6.1f} | {fmt(ra,'S')} {fmt(ra,'omega')} {fmt(ra,'H_pot'):>7} {fmt(ra,'m2_density'):>7} "
                  f"| {fmt(rb,'S')} {fmt(rb,'omega')} {fmt(rb,'H_pot'):>7} {fmt(rb,'m2_density'):>7} | {winner}")
        print()
        # Prediction
        S_gd_final = ef_a.history[-1]['S'] if ef_a.history else float('nan')
        S_co_final = ef_b.history[-1]['S'] if ef_b.history else float('nan')
        stokes_gd = sum(r['stokes'] for r in ef_a.history)
        stokes_co = sum(r['stokes'] for r in ef_b.history)
        print(f"  TOTAL Stokes crossings: {ef_a.name}={stokes_gd}  {ef_b.name}={stokes_co}")
        print(f"  Final S(u):             {ef_a.name}={S_gd_final:.3f}  {ef_b.name}={S_co_final:.3f}")
        if stokes_gd > stokes_co:
            print(f"  → {ef_a.name} is DISSIPATIVE (more chamber crossings)")
            print(f"  → {ef_b.name} is ADIABATIC (fewer crossings, stays in orbit)")
        print(f"  PREDICTION: {'adiabatic path finds deeper structural basin' if stokes_co<stokes_gd else 'paths comparable'}")




# ════════════════════════════════════════════════════════════════
# MAIN: Run both paths, compare
# ════════════════════════════════════════════════════════════════
print("="*70)
print("PATH CHARACTERIZER — GD-400 vs MF COMPILER")
print("Mapping optimization paths into (Φ, σ, τ) phase-spectrum-defect space")
print("="*70)

# Build corpus + E₀
bigram=collections.Counter(); perm={}
for i in range(len(train_ids)-1):
    a,b=train_ids[i],train_ids[i+1]
    if a<VOCAB and b<VOCAB:
        bigram[(a,b)]+=1
        if a not in perm: perm[a]=b
rows,cols,vv=[],[],[]
for (a,b),cnt in bigram.items(): rows.append(a); cols.append(b); vv.append(float(cnt))
W_sp=sp.csr_matrix((vv,(rows,cols)),shape=(VOCAB,VOCAB),dtype=np.float32)
W_sp=W_sp+W_sp.T; d_inv=np.array(1.0/(W_sp.sum(1)+1e-8)).flatten()
Dsi=sp.diags(np.sqrt(d_inv)); L_sym=sp.eye(VOCAB)-Dsi@W_sp@Dsi
evals,evecs=spla.eigsh(L_sym,k=D+1,which='SM',tol=1e-4,maxiter=2000)
idx_s=np.argsort(evals); evecs=evecs[:,idx_s][:,1:D+1]
E_0=(evecs/(np.sqrt(evals[idx_s[1:D+1]])+1e-8)[np.newaxis,:]).astype(np.float32)
E_0=(E_0/(E_0.std()+1e-8)*0.02)
E_next=np.array([E_0[perm.get(t,t)] for t in range(VOCAB)],dtype=np.float32)
E_init=(0.9*E_0+0.1*E_next)
E_norm=float(np.linalg.norm(E_0))
E_init=(E_init*(E_norm/max(float(np.linalg.norm(E_init)),1e-8))).astype(np.float32)
print(f"Corpus ready. VOCAB={VOCAB}, nnz={len(bigram)}")

# ── PATH A: GD-400 CONSTANT LR ───────────────────────────────
print("\n━━━ PATH A: GD-400 CONSTANT LR ━━━━━━━━━━━━━━━━━━━━━━━━━")
torch.manual_seed(99)
gd=LM(); gd.te.weight.data.copy_(torch.tensor(E_init))
pc_gd = PathCharacterizer("GD-400")
pc_gd.snapshot(gd, "init", step=0)
opt=torch.optim.AdamW(gd.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
LOG_STEPS={35,70,100,140,167,200,250,274,300,350,385,400}
ef_gd = SymplecticEnergyField("GD-400")
ef_gd.measure(gd, 0, eval_val(gd))
for step in range(1,401):
    gd.train(); x,y=get_batch(); _,l=gd(x,y)
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(gd.parameters(),1.0); opt.step()
    if step in LOG_STEPS:
        v=eval_val(gd)
        rec=pc_gd.snapshot(gd, f"step_{step}", step=step, val=v)
        ef_rec=ef_gd.measure(gd, step, v)
        phi_str='('+','.join('0' if abs(p)<0.3 else 'π' if abs(abs(p)-math.pi)<0.3
                              else f'{p:.2f}' for p in rec['phi'])+')'
        print(f"  step {step:3d}: val={v:.4f}  Φ={phi_str}  "
              f"S={ef_rec['S']:.2f}  ω={ef_rec['omega']:.2f}  H={ef_rec['H_pot']:.2f}  "
              f"Stokes={ef_rec['stokes']}")

# ── PATH B: MF COMPILER ──────────────────────────────────────
print("\n━━━ PATH B: MF COMPILER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
torch.manual_seed(99)
comp=LM(); comp.te.weight.data.copy_(torch.tensor(E_init))
pc_comp = PathCharacterizer("Compiler")
pc_comp.snapshot(comp, "init", step=0)
ef_comp = SymplecticEnergyField("Compiler")
ef_comp.measure(comp, 0, eval_val(comp))

# Saddle exit
comp.zero_grad()
loss=sum(comp(*get_batch())[1] for _ in range(6))/6
loss.backward()
g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
             for p in comp.parameters()]).detach(); comp.zero_grad()
v_neg=-g/g.norm()
w0=comp.flat_params() if hasattr(comp,'flat_params') else None
# inline flat_params/set_flat
def flat_p(m): return torch.cat([p.data.flatten() for p in m.parameters()])
def set_p(m,v):
    i=0
    for p in m.parameters(): n=p.numel(); p.data.copy_(v[i:i+n].reshape(p.shape)); i+=n
w0=flat_p(comp); best_v=eval_val(comp,n=6); best_a=0.0
for alpha in [0.5,1.0,1.43,2.0,3.0]:
    set_p(comp,w0+alpha*v_neg); vt=eval_val(comp,n=6)
    if vt<best_v: best_v=vt; best_a=alpha
set_p(comp,w0+best_a*v_neg)
rec=pc_comp.snapshot(comp, "saddle_exit", step=0)
pc_comp.anchors['P0_saddle_exit']=len(pc_comp.records)-1
print(f"  Saddle exit: val={rec['val']:.4f}  Φ_clean={rec['phi_clean']}/5")

# MF pump 3 rounds
ETA_MF=0.01; N_SUB=200
for mf_r in range(1,4):
    for _ in range(N_SUB):
        comp.train(); x,y=get_batch(); _,loss=comp(x,y)
        comp.zero_grad(); loss.backward()
        with torch.no_grad():
            if comp.te.weight.grad is not None:
                comp.te.weight.data -= ETA_MF*comp.te.weight.grad
    for _ in range(N_SUB):
        comp.train(); x,y=get_batch(); _,loss=comp(x,y)
        comp.zero_grad(); loss.backward()
        with torch.no_grad():
            for bl in comp.blocks:
                if bl.attn.WK.weight.grad is not None:
                    bl.attn.WK.weight.data += ETA_MF*bl.attn.WK.weight.grad
rec=pc_comp.snapshot(comp, f"MF_{mf_r}", step=mf_r*2)
print(f"  MF round {mf_r}: val={rec['val']:.4f}  Φ_clean={rec['phi_clean']}/5  "
      f"τ={rec['tau']:.2f}")

# Basin 33 CE
opt_b=torch.optim.AdamW(comp.parameters(),lr=LR*5,betas=(0.9,0.95),weight_decay=0.1)
for step in range(1,34):
    lr_cur=LR*5*min(step,10)/10
    for pg in opt_b.param_groups: pg['lr']=lr_cur
    comp.train(); x,y=get_batch(); _,l=comp(x,y)
    opt_b.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(comp.parameters(),1.0); opt_b.step()
    if step in {10,20,33}:
        rec=pc_comp.snapshot(comp, f"basin_{step}", step=step)
        print(f"  Basin CE {step}: val={rec['val']:.4f}  Φ_clean={rec['phi_clean']}/5")

# TopoGate
with torch.no_grad():
    for l in [1,2]:
        comp.blocks[l].attn.WV.weight.data.mul_(-1)
        comp.blocks[l].attn.op.weight.data.mul_(-1)
rec=pc_comp.snapshot(comp, "topo_gate", step=33)
pc_comp.anchors['P1_topo_gate']=len(pc_comp.records)-1
print(f"  TopoGate: val={rec['val']:.4f}  Φ_clean={rec['phi_clean']}/5")

# 167 CE continuation
opt_c=torch.optim.AdamW(comp.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
LOG_COMP={25,50,75,100,125,141,167}
for step in range(1,168):
    comp.train(); x,y=get_batch(); _,l=comp(x,y)
    opt_c.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(comp.parameters(),1.0); opt_c.step()
    if step in LOG_COMP:
        rec=pc_comp.snapshot(comp, f"cont_{step}", step=33+step)
        ef_rec=ef_comp.measure(comp, 33+step, rec['val'])
        print(f"  Cont CE {step:3d}: val={rec['val']:.4f}  "
              f"Φ_clean={rec['phi_clean']}/5  "
              f"S={ef_rec['S']:.2f}  ω={ef_rec['omega'] if not math.isnan(ef_rec['omega']) else 0:.2f}  "
              f"Stokes={ef_rec['stokes']}")

# ── PATH SUMMARIES + COMPARISON ──────────────────────────────
print("\n"+"="*70)
pc_gd.print_path()
pc_comp.print_path()
ef_gd.print_field()
ef_comp.print_field()
SymplecticEnergyField.compare_fields(ef_gd, ef_comp)

# Side-by-side at equal val levels
print("\n"+"="*70)
print("PATH COMPARISON IN (Progress, Serre_dist, Defect) SPACE")
print("="*70)
v_final_gd=pc_gd.records[-1]['val']
v_final_comp=pc_comp.records[-1]['val']
targets=[4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.3, 0.2]
print(f"\n  {'val≤':>6}  {'GD step':>8} {'GD Φ_cl':>8} {'GD τ':>6} {'GD Sd':>7} |"
      f" {'Comp step':>9} {'Co Φ_cl':>8} {'Co τ':>6} {'Co Sd':>7}")
print("  "+"-"*80)
for vt in targets:
    rg=next((r for r in pc_gd.records if r['val']<=vt), None)
    rc=next((r for r in pc_comp.records if r['val']<=vt), None)
    if rg is None and rc is None: continue
    gs=f"{rg['step']:>8}" if rg else f"{'---':>8}"
    gc=f"{rg['phi_clean']:>8}" if rg else f"{'---':>8}"
    gt=f"{rg['tau']:>6.2f}" if rg else f"{'---':>6}"
    gd_=f"{rg['serre_dist']:>7.2f}" if rg else f"{'---':>7}"
    cs=f"{rc['step']:>9}" if rc else f"{'---':>9}"
    cc=f"{rc['phi_clean']:>8}" if rc else f"{'---':>8}"
    ct=f"{rc['tau']:>6.2f}" if rc else f"{'---':>6}"
    cd=f"{rc['serre_dist']:>7.2f}" if rc else f"{'---':>7}"
    print(f"  {vt:>6.1f}  {gs} {gc} {gt} {gd_} | {cs} {cc} {ct} {cd}")

print("\n  ANCHOR COMPARISON:")
print(f"  GD-400 anchors:   {pc_gd.anchors}")
print(f"  Compiler anchors: {pc_comp.anchors}")
print()
print("  GEOMETRIC INTERPRETATION:")
print("  Φ_clean: how many layer transitions are in {0,π} (Bridgeland orbit)")
print("  τ: gluing defect = FF/Emb gradient ratio (K₀ extension class)")
print("  Sd: Serre distance = ||log σ - σ_Serre||₂ from Kac-Moody target")
print()
print("  At equal val:")
print("  Higher Φ_clean = cleaner Bridgeland orbit")
print("  Lower Sd = closer to Serre decay (correct monodromy scaling)")
print("  τ ≈ 2: in K₀ correction regime (FF needs amplification)")


# ══════════════════════════════════════════════════════════════
# HESSIAN ACCUMULATOR — 2nd order curvature tracking
# Connects Python training to Julia curved_cup_product structure
# ══════════════════════════════════════════════════════════════

class HessianAccumulator:
    """
    Tracks 2nd-order Hessian accumulation during training.
    Corresponds to the m₀ curvature term in Julia's curved_cup_product:
      curved_cup(f,g) = cup(f,g) + [m₀, cup(f,g)]
    where m₀ = accumulated Hessian drift from the reference point.

    THREE MEASUREMENTS:
    1. kv_err(t) = ||g(t) - H(t)·Δθ(t)|| / ||g(t)||
       = how much the Taylor expansion has drifted from reference θ₀
       = the [m₀, f⌣g] term magnitude relative to f⌣g

    2. m0_drift(t) = ||H(t)·v - H(0)·v|| / ||H(0)·v||
       = how much the Hessian kernel has rotated
       = the m₀ itself (Hessian change from reference)

    3. mc_curv(t) = ||H(t)·g(t)|| / ||g(t)||²
       = Maurer-Cartan curvature proxy
       = ||[φ,φ]|| in Julia's curvature(phi_map)
       = non-commutativity of gradient: how much H rotates g
       GD prediction: mc_curv oscillates (chaotic regime)
       Compiler prediction: mc_curv monotone decreasing (adiabatic)
    """

    def __init__(self, name="hessian"):
        self.name = name
        self.history = []
        self._theta_ref = None   # reference point for kv_err and m0_drift
        self._Hv_ref   = None    # H(θ₀)·v_ref for m0_drift

    def set_reference(self, model, n=6):
        """Set θ₀ reference point. Call once after init/MF pump."""
        self._theta_ref = torch.cat([p.data.flatten() for p in model.parameters()]).clone()
        # Compute H(θ₀)·v where v = normalized gradient
        g = self._get_grad(model, n)
        v = g / max(float(g.norm()), 1e-10)
        self._Hv_ref = self._hvp(model, v, n)

    def measure(self, model, step, theta_start=None, n=6):
        """Measure all three curvature quantities at current model state."""
        g = self._get_grad(model, n)
        gnorm = float(g.norm())
        if gnorm < 1e-10:
            return dict(step=step, kv_err=0., m0_drift=0., mc_curv=0.)

        # 1. kv_err: ||g - H·Δθ|| / ||g||
        if theta_start is not None:
            delta = torch.cat([p.data.flatten() for p in model.parameters()]) - theta_start
            if float(delta.norm()) > 1e-8:
                Hd = self._hvp(model, delta / delta.norm(), n) * float(delta.norm())
                kv = float((g - Hd).norm()) / max(gnorm, 1e-10)
            else:
                kv = 0.
        else:
            kv = float('nan')

        # 2. m0_drift: ||H(t)·v - H(0)·v|| / ||H(0)·v||
        if self._Hv_ref is not None and self._theta_ref is not None:
            v_ref = g / gnorm
            Hv_now = self._hvp(model, v_ref, n)
            m0 = float((Hv_now - self._Hv_ref).norm()) / max(float(self._Hv_ref.norm()), 1e-10)
        else:
            m0 = float('nan')

        # 3. mc_curv: ||H·g|| / ||g||² (Maurer-Cartan proxy)
        v_g = g / gnorm
        Hg  = self._hvp(model, v_g, n)
        mc  = float(Hg.norm()) / max(gnorm, 1e-10)

        rec = dict(step=step, kv_err=kv, m0_drift=m0, mc_curv=mc, gnorm=gnorm)
        self.history.append(rec)
        return rec

    def print_history(self):
        print(f"\n  HESSIAN ACCUMULATOR: {self.name}")
        print(f"  {'step':>5} {'kv_err':>10} {'m0_drift':>10} {'mc_curv':>10} {'||g||':>8}")
        print("  "+"-"*50)
        for r in self.history:
            kv  = f"{r['kv_err']:>10.3f}"  if not math.isnan(r.get('kv_err',float('nan'))) else f"{'---':>10}"
            m0  = f"{r['m0_drift']:>10.3f}" if not math.isnan(r.get('m0_drift',float('nan'))) else f"{'---':>10}"
            mc  = f"{r['mc_curv']:>10.3f}"
            print(f"  {r['step']:>5} {kv} {m0} {mc} {r['gnorm']:>8.4f}")

    @staticmethod
    def compare(ha_a, ha_b):
        """Compare two accumulators at equal steps."""
        print(f"\n  HESSIAN COMPARISON: {ha_a.name} vs {ha_b.name}")
        print(f"  kv_err:   how much Taylor expansion has drifted (= |m₀ correction|)")
        print(f"  m0_drift: how much Hessian kernel has rotated (= m₀ itself)")
        print(f"  mc_curv:  Maurer-Cartan curvature ||H·g||/||g||²")
        print(f"            = Julia's curvature(phi) = [φ,φ] in optimizer space")
        print(f"  Prediction: GD has higher mc_curv (β₁=5 from Lyapunov exp)")
        print()
        print(f"  {'step':>5} {'kv_A':>8} {'kv_B':>8} {'m0_A':>8} {'m0_B':>8} "
              f"{'mc_A':>8} {'mc_B':>8} {'winner'}")
        print("  "+"-"*70)
        steps_a = {r['step']:r for r in ha_a.history}
        steps_b = {r['step']:r for r in ha_b.history}
        all_steps = sorted(set(steps_a) | set(steps_b))
        for s in all_steps:
            ra = steps_a.get(s); rb = steps_b.get(s)
            def f(r,k): return f"{r[k]:.3f}" if r and not math.isnan(r.get(k,float('nan'))) else '---'
            mc_a = ra['mc_curv'] if ra else float('nan')
            mc_b = rb['mc_curv'] if rb else float('nan')
            winner = ha_a.name if mc_a < mc_b else (ha_b.name if mc_b < mc_a else '=')
            print(f"  {s:>5} {f(ra,'kv_err'):>8} {f(rb,'kv_err'):>8} "
                  f"{f(ra,'m0_drift'):>8} {f(rb,'m0_drift'):>8} "
                  f"{f(ra,'mc_curv'):>8} {f(rb,'mc_curv'):>8} {winner}")

    def _get_grad(self, model, n):
        model.zero_grad()
        ls=[model(*get_batch())[1] for _ in range(n)]
        torch.stack(ls).mean().backward()
        g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                     for p in model.parameters()]).detach()
        model.zero_grad(); return g

    def _hvp(self, model, v, n):
        model.zero_grad()
        ls=[model(*get_batch())[1] for _ in range(n)]
        loss=torch.stack(ls).mean()
        grads=torch.autograd.grad(loss,list(model.parameters()),create_graph=True)
        gv=(torch.cat([gr.flatten() for gr in grads])*v.detach()).sum()
        hv=torch.cat([h.flatten() for h in
                      torch.autograd.grad(gv,list(model.parameters()),retain_graph=False)])
        model.zero_grad(); return hv.detach()

print("[HessianAccumulator added to path_characterizer.py]")
print("Usage:")
print("  ha = HessianAccumulator('GD-400')")
print("  ha.set_reference(model)              # at init")
print("  ha.measure(model, step, theta_start) # at each log step")
print("  HessianAccumulator.compare(ha_gd, ha_comp)")
