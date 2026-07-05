#!/usr/bin/env python3
"""
Geodesic vs GD Comparison with Curvature Detection
==================================================
1. Run GD-400 (standard) and track its path
2. Compute analytical geodesic jump (single step)
3. Compare the two paths
4. Detect curvature using Hessian eigenvalues
"""
import json, math, warnings, collections, os, sys, time, copy
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

# ============================================================================
# CONFIGURATION
# ============================================================================
D = 256
N_HEADS = 4
N_STU = 6
BATCH = 8
SEQ = 64
LR = 3e-4
VOCAB = 1017
PHI_CLEAN_TARGET = 5
VAL_FLOOR = 0.062
ORBIT_TOLERANCE = 0.3

# ============================================================================
# DATA LOADING
# ============================================================================
for f in ['/tmp/train_ids.json', '/tmp/val_ids.json', '/tmp/vocab.json']:
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} missing. Run: python build_corpus.py")

with open('/tmp/train_ids.json') as f: train_ids = list(map(int, json.load(f)))
with open('/tmp/val_ids.json') as f: val_ids = list(map(int, json.load(f)))
with open('/tmp/vocab.json') as f: _v = json.load(f)
VOCAB = len(_v) if isinstance(_v, list) else len(_v)
train_t = torch.tensor(train_ids, dtype=torch.long)
val_t = torch.tensor(val_ids, dtype=torch.long)

# ============================================================================
# MODEL DEFINITION
# ============================================================================
class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        dh = D // N_HEADS
        self.WQ = nn.Linear(D, D, bias=False)
        self.WK = nn.Linear(D, D, bias=False)
        self.WV = nn.Linear(D, D, bias=False)
        self.op = nn.Linear(D, D, bias=False)
        self.ln = nn.LayerNorm(D)
        self.sc = math.sqrt(dh)
        self.nh = N_HEADS
        self.dh = dh
        for w in [self.WQ, self.WK, self.WV, self.op]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, h):
        B, S, _ = h.shape
        Q = self.WQ(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        K = self.WK(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        V = self.WV(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        sc = Q @ K.transpose(-2, -1) / self.sc
        mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
        sc = sc.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        return self.ln(h + self.op((F.softmax(sc, dim=-1) @ V).transpose(1, 2).reshape(B, S, D)))

class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Linear(D, D * 2, bias=False)
        self.v = nn.Linear(D, D * 2, bias=False)
        self.o = nn.Linear(D * 2, D, bias=False)
        self.n = nn.LayerNorm(D)
        for w in [self.g, self.v, self.o]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, h):
        return self.n(h + self.o(F.silu(self.g(h)) * self.v(h)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.ff = FF()

    def forward(self, h):
        return self.ff(self.attn(h))

class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = nn.Embedding(VOCAB, D)
        self.pe = nn.Embedding(512, D)
        self.blocks = nn.ModuleList([Block() for _ in range(N_STU)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)
        self.head.weight = self.te.weight
        nn.init.normal_(self.te.weight, std=0.02)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x, y=None):
        h = self.te(x) + self.pe(torch.arange(x.shape[1]))
        for b in self.blocks:
            h = b(h)
        logits = self.head(self.ln_f(h))
        if y is not None:
            return logits, F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        return logits, None

    def flat_params(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_flat(self, v):
        i = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(v[i:i + n].reshape(p.shape))
            i += n

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_batch(split='train'):
    data = val_t if split == 'val' else train_t
    ix = torch.randint(0, len(data) - SEQ - 1, (BATCH,))
    return (torch.stack([data[i:i + SEQ] for i in ix]),
            torch.stack([data[i + 1:i + SEQ + 1] for i in ix]))

def eval_val(m, n=8):
    m.eval()
    ls = []
    with torch.no_grad():
        for _ in range(n):
            x, y = get_batch('val')
            _, l = m(x, y)
            ls.append(l.item())
    return float(np.mean(ls))

def sheet_angles(model):
    out = []
    WKs = [model.blocks[l].attn.WK.weight.data.float() for l in range(N_STU)]
    for l in range(N_STU - 1):
        try:
            phi = WKs[l + 1] @ torch.linalg.pinv(WKs[l])
            lam = torch.linalg.eigvals(phi)
            lam1 = lam[lam.abs().argmax()]
            a = float(torch.angle(lam1))
            while a > math.pi:
                a -= 2 * math.pi
            while a < -math.pi:
                a += 2 * math.pi
            out.append(a)
        except:
            out.append(float('nan'))
    return out

def phi_clean(model):
    angles = sheet_angles(model)
    return sum(1 for a in angles if not np.isnan(a) and 
               (abs(a) < ORBIT_TOLERANCE or abs(abs(a) - math.pi) < ORBIT_TOLERANCE))

def gluing_defect(model, n=6):
    model.zero_grad()
    ls = [model(*get_batch())[1] for _ in range(n)]
    torch.stack(ls).mean().backward()
    g_ff = sum(p.grad.data.norm().item() for nm, p in model.named_parameters()
               if '.ff.' in nm and p.grad is not None)
    g_emb = model.te.weight.grad.data.norm().item() if model.te.weight.grad is not None else 1e-8
    model.zero_grad()
    return float(g_ff / max(g_emb, 1e-8))

def compute_E_init():
    bigram = collections.Counter()
    perm = {}
    for i in range(len(train_ids) - 1):
        a, b = train_ids[i], train_ids[i + 1]
        if a < VOCAB and b < VOCAB:
            bigram[(a, b)] += 1
            perm.setdefault(a, b)

    rows, cols, vv = [], [], []
    for (a, b), cnt in bigram.items():
        rows.append(a); cols.append(b); vv.append(float(cnt))

    W_sp = sp.csr_matrix((vv, (rows, cols)), shape=(VOCAB, VOCAB), dtype=np.float32)
    W_sp = W_sp + W_sp.T
    d_inv = np.array(1.0 / (W_sp.sum(1) + 1e-8)).flatten()
    Dsi = sp.diags(np.sqrt(d_inv))
    L_sym = sp.eye(VOCAB) - Dsi @ W_sp @ Dsi
    evals, evecs = spla.eigsh(L_sym, k=D + 1, which='SM', tol=1e-4, maxiter=2000)
    idx_s = np.argsort(evals)
    evecs = evecs[:, idx_s][:, 1:D + 1]
    E_0 = (evecs / (np.sqrt(evals[idx_s[1:D + 1]]) + 1e-8)[np.newaxis, :]).astype(np.float32)
    E_0 = (E_0 / (E_0.std() + 1e-8) * 0.02)
    E_next = np.array([E_0[perm.get(t, t)] for t in range(VOCAB)], dtype=np.float32)
    E_init = (0.9 * E_0 + 0.1 * E_next)
    E_norm = float(np.linalg.norm(E_0))
    E_init = (E_init * (E_norm / max(float(np.linalg.norm(E_init)), 1e-8))).astype(np.float32)
    return E_init

# ============================================================================
# GRADIENT AND HESSIAN COMPUTATION
# ============================================================================
def compute_gradient(model):
    """Compute the gradient of the loss."""
    model.zero_grad()
    ls = [model(*get_batch())[1] for _ in range(6)]
    loss = torch.stack(ls).mean()
    loss.backward()
    g = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                   for p in model.parameters()]).detach()
    model.zero_grad()
    return g

def hessian_spectrum(model, n_eigs=2, n_hvp=4):
    """
    Compute the smallest and largest Hessian eigenvalues and eigenvectors.
    """
    n_p = sum(p.numel() for p in model.parameters())
    
    def hvp(v, n=4):
        model.zero_grad()
        ls = [model(*get_batch())[1] for _ in range(n)]
        loss = torch.stack(ls).mean()
        grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
        gv = (torch.cat([gr.flatten() for gr in grads]) * v.detach()).sum()
        hv = torch.cat([h.flatten() for h in
                        torch.autograd.grad(gv, list(model.parameters()), retain_graph=False)])
        model.zero_grad()
        return hv.detach()
    
    # Largest eigenvalue
    torch.manual_seed(42)
    v = torch.randn(n_p)
    v = v / v.norm()
    for _ in range(10):
        Hv = hvp(v, n_hvp)
        v = Hv / max(float(Hv.norm()), 1e-10)
    lambda_max = float((v * hvp(v, n_hvp)).sum())
    v_max = v.clone()
    
    # Smallest eigenvalue
    torch.manual_seed(43)
    v = torch.randn(n_p)
    v = v / v.norm()
    for _ in range(10):
        Hv = hvp(v, n_hvp)
        v = -Hv / max(float((-Hv).norm()), 1e-10)
    lambda_min = float((v * hvp(v, n_hvp)).sum())
    v_min = v.clone()
    
    curvature = lambda_max - lambda_min
    is_curved = curvature > 0.01
    
    return {
        'lambda_max': lambda_max,
        'lambda_min': lambda_min,
        'curvature': curvature,
        'is_curved': is_curved,
        'v_max': v_max,
        'v_min': v_min
    }

def hessian_pseudo_inverse(model, reg=1e-4):
    """
    Compute H⁺ · g using conjugate gradient (pseudo-inverse).
    This gives the geodesic jump direction.
    """
    n_p = sum(p.numel() for p in model.parameters())
    g = compute_gradient(model)
    
    def hvp(v, n=4):
        model.zero_grad()
        ls = [model(*get_batch())[1] for _ in range(n)]
        loss = torch.stack(ls).mean()
        grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
        gv = (torch.cat([gr.flatten() for gr in grads]) * v.detach()).sum()
        hv = torch.cat([h.flatten() for h in
                        torch.autograd.grad(gv, list(model.parameters()), retain_graph=False)])
        model.zero_grad()
        return hv.detach() + reg * v  # Regularization
    
    # Conjugate gradient to solve (H + reg*I) * d = g
    d = torch.zeros_like(g)
    r = g.clone()
    p = g.clone()
    rr = float((r * r).sum())
    
    for _ in range(20):
        Hp = hvp(p)
        alpha = rr / max(float((p * Hp).sum()), 1e-10)
        d += alpha * p
        r -= alpha * Hp
        rr_new = float((r * r).sum())
        beta = rr_new / max(rr, 1e-10)
        p = r + beta * p
        rr = rr_new
    
    return d

# ============================================================================
# ANALYTICAL GEODESIC JUMP
# ============================================================================
def analytic_geodesic_jump(model):
    """
    Compute the analytical geodesic jump to the floor.
    θ* = θ - H⁺ · g
    """
    g = compute_gradient(model)
    d = hessian_pseudo_inverse(model)
    
    w0 = model.flat_params()
    w_star = w0 - d
    
    return w_star, d

# ============================================================================
# MAIN COMPARISON
# ============================================================================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 70)
    print("GEODESIC VS GD COMPARISON WITH CURVATURE DETECTION")
    print("=" * 70)
    
    # Build model
    E_init = compute_E_init()
    model = LM()
    model.te.weight.data.copy_(torch.tensor(E_init))
    
    # Get to basin (simplified basin settle)
    print("\n[1] Getting to basin edge...")
    opt = torch.optim.AdamW(model.parameters(), lr=LR*10, betas=(0.9, 0.95), weight_decay=0.1)
    for step in range(1, 41):
        if step <= 10:
            for pg in opt.param_groups:
                pg['lr'] = LR*10*step/10
        model.train()
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    
    opt5 = torch.optim.AdamW(model.parameters(), lr=LR*5, betas=(0.9, 0.95), weight_decay=0.1)
    for step in range(1, 31):
        if step <= 10:
            for pg in opt5.param_groups:
                pg['lr'] = LR*5*step/10
        model.train()
        x, y = get_batch()
        _, loss = model(x, y)
        opt5.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt5.step()
    
    opt3 = torch.optim.AdamW(model.parameters(), lr=LR*3, betas=(0.9, 0.95), weight_decay=0.1)
    for step in range(1, 21):
        if step <= 10:
            for pg in opt3.param_groups:
                pg['lr'] = LR*3*step/10
        model.train()
        x, y = get_batch()
        _, loss = model(x, y)
        opt3.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt3.step()
    
    v0 = eval_val(model, n=8)
    phi0 = phi_clean(model)
    tau0 = gluing_defect(model, n=4)
    print(f"  Basin edge: val={v0:.4f}, phi={phi0}/5, tau={tau0:.2f}")
    
    # ====================================================================
    # PART 1: Hessian Spectrum (Curvature Detection)
    # ====================================================================
    print("\n[2] Hessian Spectrum at Basin Edge")
    print("-" * 50)
    
    spec = hessian_spectrum(model)
    print(f"  λ_max = {spec['lambda_max']:.6f}")
    print(f"  λ_min = {spec['lambda_min']:.6f}")
    print(f"  Curvature = {spec['curvature']:.6f}")
    print(f"  Is curved: {spec['is_curved']}")
    
    # ====================================================================
    # PART 2: Analytical Geodesic Jump
    # ====================================================================
    print("\n[3] Analytical Geodesic Jump")
    print("-" * 50)
    
    w0 = model.flat_params()
    w_star, d = analytic_geodesic_jump(model)
    
    model_geodesic = copy.deepcopy(model)
    model_geodesic.set_flat(w_star)
    
    v_geodesic = eval_val(model_geodesic, n=8)
    phi_geodesic = phi_clean(model_geodesic)
    tau_geodesic = gluing_defect(model_geodesic, n=4)
    
    print(f"  Jump direction norm: {d.norm():.6f}")
    print(f"  Geodesic jump result: val={v_geodesic:.4f}, phi={phi_geodesic}/5, tau={tau_geodesic:.2f}")
    print(f"  Improvement: {v0 - v_geodesic:.4f}")
    
    # ====================================================================
    # PART 3: GD-400 Path
    # ====================================================================
    print("\n[4] GD-400 Path")
    print("-" * 50)
    
    model_gd = copy.deepcopy(model)
    opt_gd = torch.optim.AdamW(model_gd.parameters(), lr=0.003,
                                betas=(0.9, 0.95), weight_decay=0.1)
    
    gd_path = []
    gd_directions = []
    grad_directions = []
    hessian_directions = []
    
    # Initial state
    w0_gd = model_gd.flat_params()
    g0 = compute_gradient(model_gd)
    g0_dir = g0 / max(g0.norm(), 1e-10)
    spec0 = hessian_spectrum(model_gd)
    
    gd_path.append(w0_gd.clone())
    grad_directions.append(g0_dir)
    hessian_directions.append(spec0['v_min'])
    
    print(f"  Step    0: val={eval_val(model_gd, n=4):.4f}")
    
    for step in range(1, 401):
        model_gd.train()
        x, y = get_batch()
        _, loss = model_gd(x, y)
        opt_gd.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_gd.parameters(), 1.0)
        opt_gd.step()
        
        if step % 50 == 0:
            v = eval_val(model_gd, n=4)
            print(f"  Step {step:4d}: val={v:.4f}")
            
            # Track direction
            curr_w = model_gd.flat_params()
            g = compute_gradient(model_gd)
            g_dir = g / max(g.norm(), 1e-10)
            spec = hessian_spectrum(model_gd)
            
            gd_path.append(curr_w.clone())
            grad_directions.append(g_dir)
            hessian_directions.append(spec['v_min'])
    
    v_gd_final = eval_val(model_gd, n=12)
    phi_gd_final = phi_clean(model_gd)
    tau_gd_final = gluing_defect(model_gd, n=6)
    print(f"\n  GD-400 final: val={v_gd_final:.4f}, phi={phi_gd_final}/5, tau={tau_gd_final:.2f}")
    
    # ====================================================================
    # PART 4: Direction Change Analysis (Geodesic Detection)
    # ====================================================================
    print("\n[5] Direction Change Analysis (Geodesic Detection)")
    print("-" * 50)
    
    # Compute cos(grad_t, grad_{t-1}) for each step
    grad_changes = []
    for i in range(1, len(grad_directions)):
        cos_sim = float((grad_directions[i] * grad_directions[i-1]).sum())
        grad_changes.append(cos_sim)
    
    # Compute cos(grad, Hessian) for each step
    grad_hess_align = []
    for i in range(len(grad_directions)):
        cos_sim = float((grad_directions[i] * hessian_directions[i]).sum())
        grad_hess_align.append(cos_sim)
    
    # Print direction change table
    print("\n  Step | cos(grad_t, grad_{t-1}) | cos(grad, Hessian)")
    print("  -----|--------------------------|------------------")
    for i in range(len(grad_changes)):
        step = (i + 1) * 50
        print(f"  {step:5d} | {grad_changes[i]:+.4f}                    | {grad_hess_align[i+1]:+.4f}")
    
    # Detect geodesic
    avg_dir_change = np.mean(grad_changes) if grad_changes else 0.0
    avg_hess_align = np.mean(grad_hess_align) if grad_hess_align else 0.0
    
    print(f"\n  Average cos(grad_t, grad_{{t-1}}): {avg_dir_change:.4f}")
    print(f"  Average cos(grad, Hessian): {avg_hess_align:.4f}")
    
    if avg_dir_change < 0.9 and abs(avg_hess_align) < 0.1:
        print("\n  ✓ GEODESIC DETECTED")
        print("    - Direction changes over time (cos < 0.9)")
        print("    - Gradient is orthogonal to Hessian (cos ≈ 0)")
        print("    → The path follows the geodesic on the curved manifold")
    elif avg_dir_change < 0.9:
        print("\n  ⚠ PARTIAL GEODESIC")
        print("    - Direction changes, but Hessian alignment is not orthogonal")
    else:
        print("\n  ✗ STRAIGHT PATH")
        print("    - Direction does not change (cos ≈ 1.0)")
        print("    → The path is a straight line in Euclidean space")
    
    # ====================================================================
    # PART 6: Curvature Detection
    # ====================================================================
    print("\n[6] Curvature Detection")
    print("-" * 50)
    
    # Measure curvature at multiple points along the GD path
    curvature_data = []
    model_curv = copy.deepcopy(model)
    opt_curv = torch.optim.AdamW(model_curv.parameters(), lr=0.003,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    
    for step in range(1, 401):
        model_curv.train()
        x, y = get_batch()
        _, loss = model_curv(x, y)
        opt_curv.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_curv.parameters(), 1.0)
        opt_curv.step()
        
        if step % 50 == 0:
            spec = hessian_spectrum(model_curv)
            v = eval_val(model_curv, n=4)
            tau = gluing_defect(model_curv, n=4)
            curvature_data.append((step, v, tau, spec['curvature'], spec['lambda_max'], spec['lambda_min']))
    
    print("\n  Step |   val   |  τ   | Curvature | λ_max | λ_min")
    print("  -----|---------|------|-----------|-------|------")
    for step, v, tau, curv, lamax, lamin in curvature_data:
        print(f"  {step:5d} | {v:.4f} | {tau:.2f} | {curv:.6f} | {lamax:.6f} | {lamin:.6f}")
    
    # Detect curvature trend
    if len(curvature_data) >= 2:
        start_curv = curvature_data[0][3]
        end_curv = curvature_data[-1][3]
        print(f"\n  Curvature: {start_curv:.6f} → {end_curv:.6f}")
        if end_curv < start_curv * 0.5:
            print("  ✓ CURVATURE DECREASING")
            print("    → The manifold is flattening as we approach the floor")
        elif end_curv > start_curv * 2.0:
            print("  ✗ CURVATURE INCREASING")
            print("    → The manifold is becoming more curved")
        else:
            print("  ~ CURVATURE STABLE")
            print("    → The manifold has constant curvature")
    
    # ====================================================================
    # PART 7: Comparison Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Method        | Val      | Phi  | Tau")
    print(f"  --------------|----------|------|-------")
    print(f"  Basin edge    | {v0:.4f}   | {phi0}/5  | {tau0:.2f}")
    print(f"  Geodesic jump | {v_geodesic:.4f}   | {phi_geodesic}/5  | {tau_geodesic:.2f}")
    print(f"  GD-400        | {v_gd_final:.4f}   | {phi_gd_final}/5  | {tau_gd_final:.2f}")
    
    print(f"\n  Geodesic detected: {'YES' if avg_dir_change < 0.9 and abs(avg_hess_align) < 0.1 else 'NO'}")
    if len(curvature_data) >= 2:
        print(f"  Curvature: {curvature_data[0][3]:.6f} → {curvature_data[-1][3]:.6f}")
    print(f"  Direction change: {avg_dir_change:.4f}")
    print(f"  Hessian alignment: {avg_hess_align:.4f}")
    
    # ====================================================================
    # PART 8: Save Analysis
    # ====================================================================
    torch.save({
        'basin_edge': {'val': v0, 'phi': phi0, 'tau': tau0},
        'geodesic_jump': {'val': v_geodesic, 'phi': phi_geodesic, 'tau': tau_geodesic, 'direction': d},
        'gd_final': {'val': v_gd_final, 'phi': phi_gd_final, 'tau': tau_gd_final},
        'direction_changes': grad_changes,
        'hessian_alignment': grad_hess_align,
        'curvature_data': curvature_data,
        'geodesic_detected': avg_dir_change < 0.9 and abs(avg_hess_align) < 0.1
    }, 'geodesic_analysis_complete.pt')
    print("\n  Saved: geodesic_analysis_complete.pt")

if __name__ == '__main__':
    main()
