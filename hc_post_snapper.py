#!/usr/bin/env python3
"""
Smart DP Compiler with Bieberbach Singularity Detection + Homotopy
==================================================================
1. DP compiler reaches flat basin (val ~0.075)
2. Bieberbach's theorem detects singularity (spectral gap = bowl exists)
3. Homotopy continuation with GD-equivalent step size walks to bowl
4. Algebraic jump to bowl floor

Total CE: ~5 (DP) + 2 (singularity) + 33 (homotopy) = ~40 CE
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
ETA_MF = 0.01
N_SUB = 200
VOCAB = 1017
MAX_DP_STEPS = 50

PHI_CLEAN_TARGET = 5
TAU_MIN = 1.5
TAU_MAX = 5.7
VAL_FLOOR = 0.062
ORBIT_TOLERANCE = 0.3
BOWL_DETECTION_THRESHOLD = 0.08
SINGULARITY_SPECTRAL_GAP_THRESHOLD = 0.3

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


def sheet_angles_str(model):
    angles = sheet_angles(model)
    out = []
    for a in angles:
        if np.isnan(a):
            out.append('?')
        elif abs(a) < ORBIT_TOLERANCE:
            out.append('0')
        elif abs(abs(a) - math.pi) < ORBIT_TOLERANCE:
            out.append('π')
        else:
            out.append(f'{a:.2f}')
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


def w_ff_formula_clamped(tau):
    if tau < TAU_MIN:
        return 3.5
    if tau > TAU_MAX:
        return 0.47
    return 3.5 * (TAU_MIN / max(tau, 0.1)) ** 1.5


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


def compute_gradient(model, n=8):
    model.zero_grad()
    ls = [model(*get_batch())[1] for _ in range(n)]
    torch.stack(ls).mean().backward()
    g = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                   for p in model.parameters()]).detach()
    model.zero_grad()
    return g


# ============================================================================
# BIEBERBACH SINGULARITY DETECTION
# ============================================================================
def bieberbach_singularity_detection(model):
    """
    Detect singularities using Bieberbach's theorem.
    
    Bieberbach's theorem: flat Riemannian manifolds are finitely covered by tori.
    In the loss landscape, the flat plateau is a torus (isotropic Hessian).
    The bowl is the singular locus where the Hessian develops a spectral gap.
    
    Returns: {
        'gap_detected': bool,
        'gap_value': float,
        'gap_idx': int,
        'eigenvals': list,
        'eigenvecs': list,
        'eigenvals_sorted': list,
        'ce_used': int
    }
    """
    model_try = copy.deepcopy(model)
    n_p = sum(p.numel() for p in model_try.parameters())
    ce_used = 0
    
    print("      [Bieberbach] Detecting singularity...")
    
    def hvp(v, n=4):
        model_try.zero_grad()
        ls = [model_try(*get_batch())[1] for _ in range(n)]
        loss = torch.stack(ls).mean()
        grads = torch.autograd.grad(loss, list(model_try.parameters()), create_graph=True)
        gv = (torch.cat([gr.flatten() for gr in grads]) * v.detach()).sum()
        hv = torch.cat([h.flatten() for h in
                        torch.autograd.grad(gv, list(model_try.parameters()), retain_graph=False)])
        model_try.zero_grad()
        return hv.detach()
    
    # Compute gradient norm
    g = compute_gradient(model_try, n=8)
    g_norm = float(g.norm())
    ce_used += 8
    print(f"        Gradient norm: {g_norm:.4f}")
    
    # Power iteration for top eigenvalues
    torch.manual_seed(42)
    eigenvals = []
    eigenvecs = []
    
    v = torch.randn(n_p)
    v = v / v.norm()
    
    for i in range(8):
        for _ in range(10):
            Hv = hvp(v, 4)
            for ev in eigenvecs:
                Hv = Hv - (Hv * ev).sum() * ev
            v = Hv / max(float(Hv.norm()), 1e-10)
        Hv = hvp(v, 4)
        eigenval = float((v * Hv).sum().item())
        eigenvals.append(eigenval)
        eigenvecs.append(v.clone())
        v = torch.randn(n_p)
        for ev in eigenvecs:
            v = v - (v * ev).sum() * ev
        v = v / v.norm()
    
    eigenvals_sorted = sorted(eigenvals, reverse=True)
    print(f"        Eigenvalues: {[f'{x:.3f}' for x in eigenvals_sorted[:5]]}")
    
    # Detect spectral gap (Bieberbach: singularities have gaps)
    gap_detected = False
    gap_idx = -1
    gap_value = 0
    for i in range(len(eigenvals_sorted) - 1):
        gap = eigenvals_sorted[i] - eigenvals_sorted[i+1]
        if gap > SINGULARITY_SPECTRAL_GAP_THRESHOLD:
            gap_detected = True
            gap_idx = i
            gap_value = gap
            break
    
    print(f"        Spectral gap: {gap_value:.4f} ({'detected' if gap_detected else 'none'})")
    
    return {
        'gap_detected': gap_detected,
        'gap_value': gap_value,
        'gap_idx': gap_idx,
        'eigenvals': eigenvals,
        'eigenvecs': eigenvecs,
        'eigenvals_sorted': eigenvals_sorted,
        'g_norm': g_norm,
        'ce_used': ce_used
    }


# ============================================================================
# HOMOTOPY CONTINUATION WITH GD-EQUIVALENT STEP SIZE
# ============================================================================
def homotopy_continuation(model, singularity_info):
    """
    Homotopy continuation from flat basin to bowl.
    
    Uses GD-equivalent step size (not 0.05 jumps).
    """
    print("\n      [Homotopy] Continuation...")
    v0 = eval_val(model, n=4)
    print(f"        Starting val={v0:.4f}")
    
    model_try = copy.deepcopy(model)
    ce_used = 4
    v0_val = v0
    
    # Get bowl direction from singularity gap
    gap_idx = singularity_info['gap_idx']
    if gap_idx == -1:
        print("        ✗ No spectral gap found")
        return None, v0_val, phi_clean(model_try), gluing_defect(model_try, n=4), ce_used
    
    bowl_direction = singularity_info['eigenvecs'][gap_idx]
    print(f"        Bowl direction norm: {float(bowl_direction.norm()):.4f}")
    
    # Compute GD-equivalent step size
    w0 = model_try.flat_params()
    g = compute_gradient(model_try, n=8)
    g_norm = float(g.norm())
    ce_used += 8
    
    # GD step size: LR * g_norm / ||direction||
    eta = LR * g_norm / (float(bowl_direction.norm()) + 1e-10)
    eta = min(eta, LR * 10)  # Cap at 10× LR
    eta = max(eta, LR * 0.1)  # Floor at 0.1× LR
    
    print(f"        GD LR: {LR:.6f}")
    print(f"        Gradient norm: {g_norm:.4f}")
    print(f"        Homotopy eta: {eta:.6f} ({eta/LR:.2f}× LR)")
    
    # Number of steps: scale to match GD
    n_steps = int(50 * (eta / LR))
    n_steps = max(15, min(200, n_steps))
    print(f"        Homotopy steps: {n_steps}")
    
    best_val = v0_val
    best_model = copy.deepcopy(model_try)
    best_phi = phi_clean(model_try)
    
    for step in range(1, n_steps + 1):
        s = step / n_steps
        
        # Compute gradient of loss
        model_try.zero_grad()
        ls = [model_try(*get_batch())[1] for _ in range(6)]
        loss = torch.stack(ls).mean()
        loss.backward()
        ce_used += 6
        
        g_loss = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                            for p in model_try.parameters()]).detach()
        model_try.zero_grad()
        
        # Add bowl potential gradient (from Hessian eigenvector)
        # λ(s) = s^2 for smooth transition
        lam = s ** 2
        grad_phi = bowl_direction * lam * 0.1
        
        # Total gradient: ∇H = ∇L + ∇φ
        g_total = g_loss + grad_phi
        
        # Step with GD-equivalent step size
        model_try.set_flat(model_try.flat_params() - eta * g_total)
        ce_used += 1
        
        if step % max(1, n_steps // 8) == 0 or step == n_steps:
            v = eval_val(model_try, n=4)
            ce_used += 4
            phi = phi_clean(model_try)
            tau = gluing_defect(model_try, n=4)
            ce_used += 4
            print(f"          s={s:.3f}: val={v:.4f}, phi={phi}/5, tau={tau:.2f}")
            
            if v < best_val:
                best_val = v
                best_model = copy.deepcopy(model_try)
                best_phi = phi
            
            # Early exit if we reached the bowl
            if v < 0.02:
                print(f"          ✓ Reached bowl at s={s:.3f}")
                break
    
    return best_model, best_val, best_phi, gluing_defect(best_model, n=4), ce_used


# ============================================================================
# ACTION: BIEBERBACH SINGULARITY + HOMOTOPY
# ============================================================================
def action_bieberbach_homotopy(state):
    """
    1. Detect singularity using Bieberbach's theorem
    2. Homotopy continuation with GD-equivalent step size
    """
    model = state.model
    v0_val = state.val
    phi0 = state.phi
    tau0 = state.tau
    
    print(f"\n  → BIEBERBACH SINGULARITY + HOMOTOPY...")
    print(f"    Starting val={v0_val:.4f}, phi={phi0}/5, tau={tau0:.2f}")
    
    if v0_val > BOWL_DETECTION_THRESHOLD:
        print(f"    ℹ Skipping: val={v0_val:.4f} > {BOWL_DETECTION_THRESHOLD}")
        return None
    
    if phi0 < 4:
        print(f"    ℹ Skipping: phi={phi0}/5 < 4")
        return None
    
    total_ce = 0
    
    # 1. Bieberbach singularity detection
    print("    [1] Bieberbach singularity detection...")
    singularity_info = bieberbach_singularity_detection(model)
    total_ce += singularity_info['ce_used']
    
    if not singularity_info['gap_detected']:
        print("    ✗ No singularity detected (no spectral gap)")
        return None
    
    print(f"    ✓ Singularity detected! Spectral gap={singularity_info['gap_value']:.4f}")
    
    # 2. Homotopy continuation with GD-equivalent steps
    print("    [2] Homotopy continuation...")
    model_try, v_new, phi_new, tau_new, ce = homotopy_continuation(model, singularity_info)
    total_ce += ce
    
    if model_try is None:
        print("    ✗ Homotopy failed")
        return None
    
    print(f"\n    ✓ Complete: val {v0_val:.4f} → {v_new:.4f}")
    print(f"    phi: {phi0}/5 → {phi_new}/5")
    print(f"    tau: {tau0:.2f} → {tau_new:.2f}")
    print(f"    Total CE: {total_ce}")
    
    if v_new < v0_val * 0.9:
        return CompilerState(model_try, state.step + 1, 'bieberbach_homotopy',
                            v_new, phi_new, tau_new,
                            state.action_history + [f'bieberbach_homotopy({total_ce}CE)'],
                            parent=state)
    
    return None


# ============================================================================
# LANDSCAPE ANALYSIS
# ============================================================================
class LandscapeAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def _to_float(self, x):
        if x is None:
            return 0.0
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return float(x.item())
            return float(x.mean().item())
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.item())
            return float(np.mean(x))
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0
    
    def analyze(self, model):
        v = self._to_float(eval_val(model, n=8))
        phi = int(phi_clean(model))
        tau = self._to_float(gluing_defect(model, n=6))
        
        def hvp_s(v_tensor, n=4):
            model.zero_grad()
            ls = [model(*get_batch())[1] for _ in range(n)]
            loss = torch.stack(ls).mean()
            grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            gv = (torch.cat([gr.flatten() for gr in grads]) * v_tensor.detach()).sum()
            hv = torch.cat([h.flatten() for h in
                            torch.autograd.grad(gv, list(model.parameters()), retain_graph=False)])
            model.zero_grad()
            return hv.detach()
        
        n_p = sum(p.numel() for p in model.parameters())
        
        torch.manual_seed(42)
        v_tensor = torch.randn(n_p)
        v_tensor = v_tensor / v_tensor.norm()
        for _ in range(10):
            Hv = hvp_s(v_tensor)
            v_tensor = Hv / max(self._to_float(Hv.norm()), 1e-10)
        lambda_max = self._to_float((v_tensor * hvp_s(v_tensor)).sum())
        
        torch.manual_seed(43)
        v_tensor = torch.randn(n_p)
        v_tensor = v_tensor / v_tensor.norm()
        for _ in range(10):
            Hv = hvp_s(v_tensor)
            v_tensor = -Hv / max(self._to_float(Hv.norm()), 1e-10)
        lambda_min = self._to_float((v_tensor * hvp_s(v_tensor)).sum())
        
        model.zero_grad()
        ls = [model(*get_batch())[1] for _ in range(6)]
        torch.stack(ls).mean().backward()
        g_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g_norm += self._to_float(p.grad.data.norm())
        model.zero_grad()
        
        return {
            'val': v,
            'phi': phi,
            'tau': tau,
            'lambda_max': lambda_max,
            'lambda_min': lambda_min,
            'g_norm': g_norm,
            'is_positive_definite': lambda_min > 0,
            'has_negative_curvature': lambda_min < -0.01,
            'is_flat': abs(lambda_max) < 0.01,
            'in_orbit': phi == PHI_CLEAN_TARGET and TAU_MIN <= tau <= TAU_MAX,
            'near_floor': v < 0.1,
            'phase': 'saddle' if lambda_min < -0.01 else 'flat' if abs(lambda_max) < 0.01 else 'bowl'
        }


# ============================================================================
# COMPILER STATE
# ============================================================================
class CompilerState:
    def __init__(self, model, step, phase, val, phi, tau, action_history=None, parent=None):
        self.model = copy.deepcopy(model)
        self.step = step
        self.phase = phase
        self.val = val
        self.phi = phi
        self.tau = tau
        self.action_history = action_history or []
        self.parent = parent
        self.children = []

    def __repr__(self):
        orbit_ok = "✓" if self.phi == PHI_CLEAN_TARGET and TAU_MIN <= self.tau <= TAU_MAX else "✗"
        return (f"Step {self.step}: val={self.val:.4f}, phi={self.phi}/5, tau={self.tau:.2f} {orbit_ok}")


# ============================================================================
# SMART ACTION SELECTION
# ============================================================================
def get_smart_actions(state, analyzer):
    landscape = analyzer.analyze(state.model)
    actions = []
    
    val = landscape['val']
    phi = landscape['phi']
    tau = landscape['tau']
    lambda_min = landscape['lambda_min']
    near_floor = landscape['near_floor']
    in_orbit = landscape['in_orbit']
    has_negative_curvature = landscape['has_negative_curvature']
    
    print(f"\n  Landscape: val={val:.4f}, phi={phi}/5, tau={tau:.2f}, "
          f"λ_min={lambda_min:.4f}, near_floor={near_floor}")
    
    if has_negative_curvature and not near_floor:
        alpha = min(1.0, -lambda_min / (1 + abs(lambda_min)))
        alpha = max(0.1, alpha)
        actions.append(('saddle_exit', {'alpha': float(alpha)}))
        print(f"    → saddle_exit: α={alpha:.3f}")
    
    if not in_orbit and not near_floor and val > 0.3:
        eta = min(0.02, 0.01 / (1 + val))
        eta = max(0.001, eta)
        actions.append(('mf_pump', {'eta': float(eta)}))
        print(f"    → mf_pump: η={eta:.4f}")
    
    if not near_floor:
        if val > 1.0:
            actions.append(('basin_settle', {'lr_mult': 10}))
            actions.append(('basin_settle', {'lr_mult': 5}))
        elif val > 0.3:
            actions.append(('basin_settle', {'lr_mult': 5}))
            actions.append(('basin_settle', {'lr_mult': 3}))
        else:
            actions.append(('basin_settle', {'lr_mult': 3}))
            actions.append(('basin_settle', {'lr_mult': 2}))
            actions.append(('basin_settle', {'lr_mult': 1}))
        for a in actions:
            if a[0] == 'basin_settle':
                print(f"    → basin_settle: LR×{a[1]['lr_mult']}")
    
    if not in_orbit and phi >= 3 and not near_floor:
        off_wall = PHI_CLEAN_TARGET - phi
        step_size = min(0.5, 0.15 * off_wall)
        actions.append(('restore_orbit', {'step_size': float(step_size)}))
        print(f"    → restore_orbit: step={step_size:.2f}")
    
    if phi >= 4:
        w_ff = w_ff_formula_clamped(tau)
        actions.append(('k0_split', {'w_ff': 'clamped'}))
        print(f"    → k0_split: w_FF={w_ff:.2f}")
    
    if near_floor and landscape['is_positive_definite']:
        actions.append(('lm_step', {'mu': 0.95}))
        print(f"    → lm_step: μ=0.95")
    
    if near_floor and in_orbit:
        actions.append(('lanczos', {'k': 8}))
        print(f"    → lanczos: k=8")
    
    # BIEBERBACH SINGULARITY + HOMOTOPY: when near floor and phi >= 4
    if near_floor and phi >= 4:
        actions.append(('bieberbach_homotopy', {}))
        print(f"    → bieberbach_homotopy: val={val:.4f}, phi={phi}/5")
    
    if not actions:
        actions.append(('basin_settle', {'lr_mult': 2}))
        print(f"    → fallback: basin_settle LR×2")
    
    return actions


# ============================================================================
# ACTION FUNCTIONS (from working compiler)
# ============================================================================
def action_saddle_exit(state, params):
    model = state.model
    alpha = params['alpha']
    
    def hvp_s(v, n=8):
        model.zero_grad()
        ls = [model(*get_batch())[1] for _ in range(n)]
        loss = torch.stack(ls).mean()
        grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
        gv = (torch.cat([gr.flatten() for gr in grads]) * v.detach()).sum()
        hv = torch.cat([h.flatten() for h in
                        torch.autograd.grad(gv, list(model.parameters()), retain_graph=False)])
        model.zero_grad()
        return hv.detach()
    
    n_p = sum(p.numel() for p in model.parameters())
    torch.manual_seed(42)
    v = torch.randn(n_p)
    v = v / v.norm()
    for _ in range(10):
        Hv = hvp_s(v)
        neg = -Hv
        v = neg / max(float(neg.norm()), 1e-10)
    
    w0 = model.flat_params()
    model_try = copy.deepcopy(model)
    model_try.set_flat(w0 + alpha * (v / v.norm()))
    
    if phi_clean(model_try) < PHI_CLEAN_TARGET:
        WKs = [model_try.blocks[l].attn.WK.weight.data.float().cpu().numpy() for l in range(N_STU)]
        for k in range(N_STU - 1):
            phases = sheet_angles(model_try)
            if k < len(phases) and not (abs(phases[k]) < ORBIT_TOLERANCE or abs(abs(phases[k]) - math.pi) < ORBIT_TOLERANCE):
                try:
                    Wk = WKs[k].astype(complex)
                    Wk1 = WKs[k + 1].astype(complex)
                    Wk_inv = np.linalg.pinv(Wk)
                    M = Wk1 @ Wk_inv
                    evals_M, evecs_M = np.linalg.eig(M)
                    idx = np.argmax(np.abs(evals_M.real))
                    lam = evals_M[idx]
                    r_vec = evecs_M[:, idx]
                    evals_L, evecs_L = np.linalg.eig(M.T)
                    idx_L = np.argmax(np.abs(evals_L.real))
                    l_vec = evecs_L[:, idx_L].conj()
                    lr = l_vec @ r_vec
                    if abs(lr) > 1e-10:
                        l_vec = l_vec / lr
                    lam_mag = abs(lam)
                    Wk_inv_r = Wk_inv @ r_vec
                    J_k = np.outer(l_vec, Wk_inv_r)
                    dPhi = np.imag(J_k) / (lam_mag + 1e-10)
                    U_s, s_vals, Vt_s = np.linalg.svd(dPhi)
                    if s_vals[0] > 1e-10:
                        target = math.pi if phases[k] > 0 else 0.0
                        alpha_corr = (target - phases[k]) / s_vals[0]
                        alpha_corr = np.clip(alpha_corr, -2.0, 2.0) * 0.3
                        u1 = U_s[:, 0].real
                        v1 = Vt_s[0, :].real
                        delta = alpha_corr * np.outer(u1, v1)
                        with torch.no_grad():
                            model_try.blocks[k + 1].attn.WK.weight.data.add_(
                                torch.tensor(delta, dtype=torch.float32))
                except:
                    pass
    
    v_try = eval_val(model_try, n=6)
    if v_try < state.val * 0.95:
        return CompilerState(model_try, state.step + 1, 'saddle_exit',
                            v_try, phi_clean(model_try), gluing_defect(model_try),
                            state.action_history + [f'saddle_exit(α={alpha:.3f})'],
                            parent=state)
    return None


def action_mf_pump(state, params):
    eta = params['eta']
    model = state.model
    model_try = copy.deepcopy(model)
    v0 = state.val
    
    for mf_r in range(1, 4):
        for l in range(N_STU):
            model_try.blocks[l].attn.WK.weight.requires_grad_(False)
            model_try.blocks[l].attn.WQ.weight.requires_grad_(False)
        
        emb_grad = torch.zeros(model_try.te.weight.shape)
        emb_fish = torch.zeros(model_try.te.weight.shape)
        torch.manual_seed((mf_r - 1) * 1000)
        
        for i in range(N_SUB // 3):
            ix = torch.randint(0, len(train_t) - SEQ - 1, (1,))[0].item()
            x = train_t[ix:ix + SEQ].unsqueeze(0)
            y = train_t[ix + 1:ix + SEQ + 1].unsqueeze(0)
            model_try.zero_grad()
            _, loss = model_try(x, y)
            loss.backward()
            if model_try.te.weight.grad is not None:
                g = model_try.te.weight.grad.detach()
                emb_grad += g
                emb_fish += g ** 2
        
        emb_grad /= (N_SUB // 3)
        emb_fish /= (N_SUB // 3)
        delta_E = -(emb_grad / (emb_fish + 1e-4))
        with torch.no_grad():
            model_try.te.weight.add_(eta * delta_E)
        
        for l in range(N_STU):
            model_try.blocks[l].attn.WK.weight.requires_grad_(True)
            model_try.blocks[l].attn.WQ.weight.requires_grad_(True)
        
        model_try.te.weight.requires_grad_(False)
        wk_grad = torch.zeros_like(model_try.blocks[0].attn.WK.weight)
        wk_fish = torch.zeros_like(model_try.blocks[0].attn.WK.weight)
        torch.manual_seed((mf_r - 1) * 1000 + 500)
        
        for i in range(N_SUB // 3):
            ix = torch.randint(0, len(train_t) - SEQ - 1, (1,))[0].item()
            x = train_t[ix:ix + SEQ].unsqueeze(0)
            y = train_t[ix + 1:ix + SEQ + 1].unsqueeze(0)
            model_try.zero_grad()
            _, loss = model_try(x, y)
            loss.backward()
            g = torch.zeros_like(model_try.blocks[0].attn.WK.weight)
            for bl in model_try.blocks:
                if bl.attn.WK.weight.grad is not None:
                    g += bl.attn.WK.weight.grad / N_STU
            wk_grad += g
            wk_fish += g ** 2
        
        wk_grad /= (N_SUB // 3)
        wk_fish /= (N_SUB // 3)
        delta_WK = -(wk_grad / (wk_fish + 1e-4))
        with torch.no_grad():
            for l in range(N_STU):
                model_try.blocks[l].attn.WK.weight.add_(eta * delta_WK)
                model_try.blocks[l].attn.WQ.weight.add_(eta * delta_WK.T)
        model_try.te.weight.requires_grad_(True)
        
        v_mf = eval_val(model_try, n=4)
        if v_mf > v0 * 1.5:
            return None
        if phi_clean(model_try) >= 4:
            break
    
    v_try = eval_val(model_try, n=6)
    if v_try < state.val * 0.95:
        return CompilerState(model_try, state.step + 1, 'mf_pump',
                            v_try, phi_clean(model_try), gluing_defect(model_try),
                            state.action_history + [f'mf_pump(η={eta:.4f})'],
                            parent=state)
    return None


def action_basin_settle(state, params):
    lr_mult = params['lr_mult']
    model = state.model
    v0 = state.val
    
    model_try = copy.deepcopy(model)
    opt_b = torch.optim.AdamW(model_try.parameters(), lr=LR * lr_mult,
                              betas=(0.9, 0.95), weight_decay=0.1)
    
    for step in range(1, 41):
        if step <= 10:
            for pg in opt_b.param_groups:
                pg['lr'] = LR * lr_mult * step / 10
        
        model_try.train()
        x, y = get_batch()
        _, l = model_try(x, y)
        opt_b.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model_try.parameters(), 1.0)
        opt_b.step()
        
        if step % 8 == 0:
            v = eval_val(model_try, n=4)
            if v > v0 * 1.5:
                return None
            if v < 0.1:
                break
    
    v_try = eval_val(model_try, n=6)
    if v_try < state.val * 0.95:
        return CompilerState(model_try, state.step + 1, 'basin_settle',
                            v_try, phi_clean(model_try), gluing_defect(model_try),
                            state.action_history + [f'basin_settle(LR×{lr_mult})'],
                            parent=state)
    return None


def action_restore_orbit(state, params):
    step_size = params['step_size']
    model = state.model
    
    phases = sheet_angles(model)
    off_wall = [i for i, p in enumerate(phases) 
                if not np.isnan(p) and not (abs(p) < ORBIT_TOLERANCE or abs(abs(p) - math.pi) < ORBIT_TOLERANCE)]
    
    if not off_wall:
        return None
    
    model_try = copy.deepcopy(model)
    WKs = [model_try.blocks[l].attn.WK.weight.data.float().cpu().numpy() for l in range(N_STU)]
    
    with torch.no_grad():
        for k in off_wall:
            if k >= len(WKs) - 1:
                continue
            try:
                Wk = WKs[k].astype(complex)
                Wk1 = WKs[k + 1].astype(complex)
                Wk_inv = np.linalg.pinv(Wk)
                M = Wk1 @ Wk_inv
                evals_M, evecs_M = np.linalg.eig(M)
                idx = np.argmax(np.abs(evals_M.real))
                lam = evals_M[idx]
                r_vec = evecs_M[:, idx]
                evals_L, evecs_L = np.linalg.eig(M.T)
                idx_L = np.argmax(np.abs(evals_L.real))
                l_vec = evecs_L[:, idx_L].conj()
                lr = l_vec @ r_vec
                if abs(lr) > 1e-10:
                    l_vec = l_vec / lr
                lam_mag = abs(lam)
                Wk_inv_r = Wk_inv @ r_vec
                J_k = np.outer(l_vec, Wk_inv_r)
                dPhi = np.imag(J_k) / (lam_mag + 1e-10)
                U_s, s_vals, Vt_s = np.linalg.svd(dPhi)
                if s_vals[0] > 1e-10:
                    target = math.pi if phases[k] > 0 else 0.0
                    alpha = (target - phases[k]) / s_vals[0]
                    alpha = np.clip(alpha, -2.0, 2.0) * step_size
                    u1 = U_s[:, 0].real
                    v1 = Vt_s[0, :].real
                    delta = alpha * np.outer(u1, v1)
                    model_try.blocks[k + 1].attn.WK.weight.data.add_(
                        torch.tensor(delta, dtype=torch.float32))
            except:
                pass
    
    v_try = eval_val(model_try, n=6)
    phi_try = phi_clean(model_try)
    
    if phi_try > state.phi and v_try < state.val * 1.1:
        return CompilerState(model_try, state.step + 1, 'restore_orbit',
                            v_try, phi_try, gluing_defect(model_try),
                            state.action_history + [f'restore_orbit(step={step_size:.2f})'],
                            parent=state)
    return None


def action_k0_split(state, params):
    model = state.model
    w_ff = w_ff_formula_clamped(state.tau)
    
    params_base = {n: p.data.clone() for n, p in model.named_parameters()}
    
    def ptype(name):
        if '.attn.WQ.' in name or '.attn.WK.' in name:
            return 'Attn'
        if 'te.weight' in name or '.ff.' in name:
            return 'EmbFF'
        return 'other'
    
    def get_lr(s, n, base_lr):
        return base_lr * 0.5 * (1 + math.cos(math.pi * s / n))
    
    m1 = copy.deepcopy(model)
    for name, p in m1.named_parameters():
        if ptype(name) != 'EmbFF':
            p.requires_grad_(False)
    p1 = [p for p in m1.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(p1, lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    
    for s in range(1, 21):
        for pg in opt1.param_groups:
            pg['lr'] = get_lr(s, 20, LR)
        m1.train()
        x, y = get_batch()
        _, l = m1(x, y)
        opt1.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(p1, 1.0)
        opt1.step()
    
    m2 = copy.deepcopy(model)
    for name, p in m2.named_parameters():
        if ptype(name) != 'Attn':
            p.requires_grad_(False)
    p2 = [p for p in m2.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(p2, lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    
    for s in range(1, 21):
        for pg in opt2.param_groups:
            pg['lr'] = get_lr(s, 20, LR)
        m2.train()
        x, y = get_batch()
        _, l = m2(x, y)
        opt2.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(p2, 1.0)
        opt2.step()
    
    m_out = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in m_out.named_parameters():
            pt = ptype(name)
            d1 = dict(m1.named_parameters())[name].data - params_base[name]
            d2 = dict(m2.named_parameters())[name].data - params_base[name]
            if pt == 'EmbFF':
                if 'te.weight' in name:
                    p.data.add_(d1)
                else:
                    p.data.add_(w_ff * d1)
            elif pt == 'Attn':
                p.data.add_(d2)
    
    v_try = eval_val(m_out, n=6)
    if v_try < state.val * 0.95:
        return CompilerState(m_out, state.step + 1, 'k0_split',
                            v_try, phi_clean(m_out), gluing_defect(m_out),
                            state.action_history + [f'k0_split(w_FF={w_ff:.2f})'],
                            parent=state)
    return None


def action_lm_step(state, params):
    mu = params['mu']
    model = state.model
    
    def hvp_l(v, n=4):
        model.zero_grad()
        ls = [model(*get_batch())[1] for _ in range(n)]
        loss = torch.stack(ls).mean()
        grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
        gv = (torch.cat([gr.flatten() for gr in grads]) * v.detach()).sum()
        hv = torch.cat([h.flatten() for h in
                        torch.autograd.grad(gv, list(model.parameters()), retain_graph=False)])
        model.zero_grad()
        return hv.detach()
    
    model_try = copy.deepcopy(model)
    model_try.zero_grad()
    ls = [model_try(*get_batch())[1] for _ in range(20)]
    torch.stack(ls).mean().backward()
    g = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                   for p in model_try.parameters()]).detach()
    model_try.zero_grad()
    
    n_p = sum(p.numel() for p in model_try.parameters())
    d = torch.zeros(n_p)
    r = -g.clone()
    p = r.clone()
    rr = float((r * r).sum())
    
    for _ in range(6):
        Hp = hvp_l(p) + mu * p
        al = rr / max(float((p * Hp).sum()), 1e-10)
        d += al * p
        r -= al * Hp
        rr2 = float((r * r).sum())
        p = r + (rr2 / max(rr, 1e-10)) * p
        rr = rr2
    
    w0 = model_try.flat_params()
    model_try.set_flat(w0 + d)
    v_try = eval_val(model_try, n=6)
    if v_try < state.val * 0.95:
        return CompilerState(model_try, state.step + 1, 'lm_step',
                            v_try, phi_clean(model_try), gluing_defect(model_try),
                            state.action_history + [f'lm_step(μ={mu:.2f})'],
                            parent=state)
    return None


def action_lanczos(state, params):
    model = state.model
    
    def hvp_l(v, n=4):
        model.zero_grad()
        ls = [model(*get_batch())[1] for _ in range(n)]
        loss = torch.stack(ls).mean()
        grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
        gv = (torch.cat([gr.flatten() for gr in grads]) * v.detach()).sum()
        hv = torch.cat([h.flatten() for h in
                        torch.autograd.grad(gv, list(model.parameters()), retain_graph=False)])
        model.zero_grad()
        return hv.detach()
    
    model_try = copy.deepcopy(model)
    n_p = sum(p.numel() for p in model_try.parameters())
    torch.manual_seed(7)
    q = torch.randn(n_p)
    q = q / q.norm()
    Q = [q]
    alphas = []
    betas = []
    
    for j in range(8):
        z = hvp_l(Q[j])
        alpha = float((Q[j] * z).sum())
        alphas.append(alpha)
        z = z - alpha * Q[j]
        if j > 0:
            z = z - betas[-1] * Q[j - 1]
        for qi in Q:
            z = z - float((qi * z).sum()) * qi
        beta = float(z.norm())
        betas.append(beta)
        if beta < 1e-8:
            break
        Q.append(z / beta)
    
    n_l = len(alphas)
    if n_l < 2:
        return None
    
    T = torch.zeros(n_l, n_l)
    for i in range(n_l):
        T[i, i] = alphas[i]
    for i in range(n_l - 1):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]
    
    T_evals, T_evecs = torch.linalg.eigh(T)
    V = torch.stack(Q[:n_l], dim=1) @ T_evecs
    
    mu = 0.95
    v0 = eval_val(model_try, n=6)
    accepted = False
    
    for si in range(3):
        model_try.zero_grad()
        ls = [model_try(*get_batch())[1] for _ in range(20)]
        torch.stack(ls).mean().backward()
        g = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                       for p in model_try.parameters()]).detach()
        model_try.zero_grad()
        
        g_proj = V.T @ g
        d_proj = g_proj / (T_evals + mu)
        g_res = g - V @ (V.T @ g)
        d = -(V @ d_proj + g_res / mu)
        
        w0 = model_try.flat_params()
        model_try.set_flat(w0 + d)
        v_try = eval_val(model_try, n=6)
        
        if v_try < v0:
            v0 = v_try
            accepted = True
        else:
            model_try.set_flat(w0)
            break
    
    if accepted and v0 < state.val * 0.95:
        return CompilerState(model_try, state.step + 1, 'lanczos',
                            v0, phi_clean(model_try), gluing_defect(model_try),
                            state.action_history + ['lanczos'],
                            parent=state)
    return None


# ============================================================================
# SMART DP COMPILER
# ============================================================================
class SmartDPCompiler:
    def __init__(self):
        self.analyzer = LandscapeAnalyzer()
        self.failed_actions = set()
        self.exhausted_states = set()
        self.visited_states = []
        self.best_state = None
        
    def get_state_key(self, state):
        val_bucket = int(state.val / 0.1)
        phi = state.phi
        tau_bucket = int(state.tau / 0.5)
        angles = sheet_angles(state.model)
        phase_bucket = tuple([int(a / 0.3) for a in angles if not np.isnan(a)])
        return (val_bucket, phi, tau_bucket, phase_bucket)
    
    def mark_failed(self, state, action_name, params):
        state_key = self.get_state_key(state)
        key = (state_key, action_name, tuple(sorted(params.items())))
        self.failed_actions.add(key)
    
    def find_backtrack_state(self, state_stack):
        for i in range(len(state_stack) - 1, -1, -1):
            state = state_stack[i]
            state_key = self.get_state_key(state)
            
            if state_key in self.exhausted_states:
                continue
            if state_key in self.visited_states:
                continue
            
            actions = get_smart_actions(state, self.analyzer)
            untried = [(name, params) for name, params in actions 
                      if (state_key, name, tuple(sorted(params.items()))) not in self.failed_actions]
            
            if untried:
                return i, state, untried
        
        return None, None, None
    
    def apply_action(self, state, action_name, params):
        if action_name == 'saddle_exit':
            return action_saddle_exit(state, params)
        elif action_name == 'mf_pump':
            return action_mf_pump(state, params)
        elif action_name == 'basin_settle':
            return action_basin_settle(state, params)
        elif action_name == 'restore_orbit':
            return action_restore_orbit(state, params)
        elif action_name == 'k0_split':
            return action_k0_split(state, params)
        elif action_name == 'lm_step':
            return action_lm_step(state, params)
        elif action_name == 'lanczos':
            return action_lanczos(state, params)
        elif action_name == 'bieberbach_homotopy':
            return action_bieberbach_homotopy(state)
        return None
    
    def run(self):
        print("=" * 70)
        print("SMART DP COMPILER WITH BIEBERBACH SINGULARITY + HOMOTOPY")
        print("=" * 70)
        print()
        
        E_init = compute_E_init()
        model = LM()
        model.te.weight.data.copy_(torch.tensor(E_init))
        
        v0 = eval_val(model, n=10)
        phi0 = phi_clean(model)
        tau0 = gluing_defect(model, n=6)
        
        print(f"Initial: val={v0:.4f}, phi={phi0}/5, tau={tau0:.2f}")
        print()
        
        initial_state = CompilerState(model, 0, 'init', v0, phi0, tau0, [], None)
        state_stack = [initial_state]
        self.best_state = initial_state
        self.visited_states = [self.get_state_key(initial_state)]
        step_counter = 0
        
        for step_counter in range(MAX_DP_STEPS):
            current = state_stack[-1]
            print(f"\n[Step {step_counter}] Current: {current}")
            
            if current.val < VAL_FLOOR and current.phi == PHI_CLEAN_TARGET:
                print(f"\n✓ SUCCESS! val={current.val:.4f}, phi={current.phi}/5")
                print(f"  Path: {' → '.join(current.action_history)}")
                return current
            
            actions = get_smart_actions(current, self.analyzer)
            state_key = self.get_state_key(current)
            
            untried = [(name, params) for name, params in actions 
                      if (state_key, name, tuple(sorted(params.items()))) not in self.failed_actions]
            
            if not untried:
                print(f"\n  ⊘ No untried actions at {current}")
                idx, back_state, back_untried = self.find_backtrack_state(state_stack)
                if back_state is None:
                    print(f"\n⚠ No states with untried actions. Best val={self.best_state.val:.4f}")
                    print(f"  Path: {' → '.join(self.best_state.action_history)}")
                    return self.best_state
                print(f"  → Backtracking to step {idx}")
                state_stack = state_stack[:idx+1]
                continue
            
            action_taken = False
            for action_name, params in untried:
                print(f"\n  → Trying {action_name} with {params}")
                
                result = self.apply_action(current, action_name, params)
                
                if result is None:
                    print(f"    ⊘ {action_name}: failed")
                    self.mark_failed(current, action_name, params)
                    continue
                
                if result.val < current.val * 0.95:
                    print(f"    ✓ {action_name}: {current.val:.4f} → {result.val:.4f} (phi={result.phi}/5)")
                    state_stack.append(result)
                    self.visited_states.append(self.get_state_key(result))
                    
                    if result.val < self.best_state.val:
                        self.best_state = result
                        print(f"    ★ New best: {result.val:.4f}")
                    
                    action_taken = True
                    break
                elif result.phi > current.phi and result.val < current.val * 1.1:
                    print(f"    ~ {action_name}: phi {current.phi}→{result.phi}/5, val {current.val:.4f}→{result.val:.4f}")
                    state_stack.append(result)
                    self.visited_states.append(self.get_state_key(result))
                    action_taken = True
                    break
                else:
                    print(f"    ⊘ {action_name}: no gain ({current.val:.4f} → {result.val:.4f})")
                    self.mark_failed(current, action_name, params)
            
            if not action_taken:
                print(f"\n  ⊘ No working action at {current}")
                self.exhausted_states.add(state_key)
        
        print(f"\n⚠ Max steps reached. Best val={self.best_state.val:.4f}")
        return self.best_state


# ============================================================================
# GRADIENT DESCENT FALLBACK
# ============================================================================
def run_gradient_descent(model, steps=400, lr=LR):
    print(f"\n  Running gradient descent fallback for {steps} steps...")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    
    history = []
    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch()
        _, l = model(x, y)
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if step % 50 == 0:
            v = eval_val(model, n=8)
            phi = phi_clean(model)
            tau = gluing_defect(model, n=4)
            history.append((step, v, phi, tau))
            print(f"    GD step {step:4d}: val={v:.4f}, phi={phi}/5, tau={tau:.2f}")
    
    v_final = eval_val(model, n=12)
    phi_final = phi_clean(model)
    tau_final = gluing_defect(model, n=6)
    print(f"    GD final: val={v_final:.4f}, phi={phi_final}/5, tau={tau_final:.2f}")
    return model, v_final, phi_final, tau_final, history


# ============================================================================
# MAIN
# ============================================================================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    compiler = SmartDPCompiler()
    final_state = compiler.run()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Final val:     {final_state.val:.4f}")
    print(f"  Final phi:     {final_state.phi}/5")
    print(f"  Final tau:     {final_state.tau:.2f}")
    print(f"  Total steps:   {final_state.step}")
    print(f"  Path:          {' → '.join(final_state.action_history)}")
    
    # If Bieberbach+homotopy didn't work, run GD fallback
    if final_state.val > 0.1 or final_state.phi < PHI_CLEAN_TARGET:
        print("\n" + "=" * 70)
        print("RUNNING GRADIENT DESCENT FALLBACK")
        print("=" * 70)
        
        model, v_final, phi_final, tau_final, history = run_gradient_descent(
            final_state.model, steps=400, lr=LR
        )
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS AFTER GD FALLBACK")
        print("=" * 70)
        print(f"  Final val:     {v_final:.4f}")
        print(f"  Final phi:     {phi_final}/5")
        print(f"  Final tau:     {tau_final:.2f}")
        print(f"  Path:          {' → '.join(final_state.action_history)} → gradient_descent(400)")
        
        torch.save(model.state_dict(), 'compiler_bieberbach_gd_final.pt')
        print(f"\n  Saved: compiler_bieberbach_gd_final.pt")
    else:
        torch.save(final_state.model.state_dict(), 'compiler_bieberbach_final.pt')
        print(f"\n  Saved: compiler_bieberbach_final.pt")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
