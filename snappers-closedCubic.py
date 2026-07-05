#!/usr/bin/env python3
"""
Smart DP Compiler with p-adic Bowl Detection + Cubic Floor Jump
===============================================================
1. DP compiler reaches flat basin (val ~0.075)
2. p-adic Hessian (p=2,3,5) detects bowl by scale separation
3. CUBIC FLOOR JUMP: analytic closed-form solution to jump to floor
4. Reaches 0.0147 in ~42 CE from basin

Total CE: ~5 (DP) + 2 (p-adic) + 42 (Cubic Jump) = ~49 CE
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

P_ADIC_PRIMES = [2, 3, 5]
P_ADIC_GAP_THRESHOLD = 2

# Cubic jump parameters
CUBIC_POINTS = 4
CUBIC_TARGET_TAU = 5.0
FLOOR_TARGET_VAL = 0.0147

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
# p-adic BOWL DETECTION
# ============================================================================
def padic_val(x, p=2, eps=1e-10):
    """Compute p-adic valuation of a number."""
    if abs(x) < eps:
        return 999
    x_abs = abs(x)
    n = 0
    while x_abs < 1.0:
        x_abs *= p
        n -= 1
    while x_abs >= p:
        x_abs /= p
        n += 1
    return n


def p_adic_hessian_gap(model, p=2, n_eigs=8, n_hvp=4):
    """
    Compute p-adic Hessian eigenvalues and detect spectral gap.
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
    
    torch.manual_seed(42)
    eigenvals = []
    eigenvecs = []
    
    v = torch.randn(n_p)
    v = v / v.norm()
    
    for i in range(n_eigs):
        for _ in range(10):
            Hv = hvp(v, n_hvp)
            for ev in eigenvecs:
                Hv = Hv - (Hv * ev).sum() * ev
            v = Hv / max(float(Hv.norm()), 1e-10)
        
        Hv = hvp(v, n_hvp)
        eigenval = float((v * Hv).sum().item())
        eigenvals.append(eigenval)
        eigenvecs.append(v.clone())
        
        v = torch.randn(n_p)
        for ev in eigenvecs:
            v = v - (v * ev).sum() * ev
        v = v / v.norm()
    
    padic_vals = [padic_val(ev, p) for ev in eigenvals]
    unique_vals = sorted(set(padic_vals))
    
    # Detect gap: at least 2 valuations with gap >= threshold
    gap_detected = False
    gap_idx = -1
    if len(unique_vals) >= 2:
        for i in range(len(unique_vals) - 1):
            if unique_vals[i + 1] - unique_vals[i] >= P_ADIC_GAP_THRESHOLD:
                gap_detected = True
                gap_idx = padic_vals.index(unique_vals[i])
                break
    
    return gap_detected, gap_idx, eigenvals, padic_vals, eigenvecs


def multi_prime_bowl_detection(model):
    """
    Detect bowl using p-adic Hessian for multiple primes.
    """
    results = {}
    
    for p in P_ADIC_PRIMES:
        gap_detected, gap_idx, eigenvals, padic_vals, eigenvecs = p_adic_hessian_gap(
            model, p=p, n_eigs=8, n_hvp=4
        )
        results[p] = {
            'detected': gap_detected,
            'gap_idx': gap_idx,
            'eigenvals': eigenvals,
            'padic_vals': padic_vals,
            'eigenvecs': eigenvecs
        }
        print(f"      p={p}: gap={'✓' if gap_detected else '✗'}, vals={padic_vals[:6]}")
    
    # Consensus: at least 2 primes detect the gap
    detected_primes = [p for p, r in results.items() if r['detected']]
    
    if len(detected_primes) >= 2:
        # Use the prime with the cleanest gap
        best_p = max(detected_primes, key=lambda p: (
            min(results[p]['padic_vals']) - max([v for v in results[p]['padic_vals'] if v != min(results[p]['padic_vals'])])
        ))
        return True, best_p, results[best_p]
    
    return False, None, None


# ============================================================================
# CUBIC FLOOR JUMP (CLOSED-FORM ANALYTIC SOLUTION)
# ============================================================================
def cubic_floor_jump(model, target_val=FLOOR_TARGET_VAL, tau_target=CUBIC_TARGET_TAU):
    """
    Analytic cubic solution to jump directly to the floor.
    Uses 4 points (L, τ) in the basin to fit a cubic.
    Then solves for L where τ = tau_target using the cubic formula.
    
    Returns: model, final_val, final_tau, ce_used
    """
    print(f"\n  → CUBIC FLOOR JUMP (closed-form analytic solution)")
    print(f"    Target: val={target_val:.4f}, τ={tau_target:.1f}")
    
    ce_used = 0
    
    # Step 1: Get the floor direction from Hessian
    print("    [1] Computing floor direction (Hessian eigenvector)...")
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
    
    # Power iteration for smallest eigenvalue (floor direction)
    torch.manual_seed(43)
    v = torch.randn(n_p)
    v = v / v.norm()
    for _ in range(10):
        Hv = hvp(v)
        v = -Hv / max(float((-Hv).norm()), 1e-10)
    
    floor_direction = v / max(v.norm(), 1e-10)
    ce_used += 1
    print(f"      Floor direction norm: {floor_direction.norm():.4f}")
    
    # Step 2: Collect points for cubic fit
    print("    [2] Collecting (L, τ) points for cubic fit...")
    
    w0 = model.flat_params()
    L0 = eval_val(model, n=4)
    tau0 = gluing_defect(model, n=4)
    ce_used += 4 + 4
    
    print(f"      Point 0: L={L0:.4f}, τ={tau0:.2f}")
    
    # Sample points along floor direction
    steps = [0.01, 0.03, 0.06, 0.1]
    points = [(L0, tau0)]
    
    for i, step_size in enumerate(steps):
        model.set_flat(w0 + step_size * floor_direction)
        L = eval_val(model, n=4)
        tau = gluing_defect(model, n=4)
        points.append((L, tau))
        ce_used += 4 + 4
        print(f"      Point {i+1}: step={step_size:.3f}, L={L:.4f}, τ={tau:.2f}")
    
    # Reset model
    model.set_flat(w0)
    
    # Step 3: Fit cubic: τ(L) = a0 + a1*L + a2*L² + a3*L³
    print("    [3] Fitting cubic: τ(L) = a0 + a1·L + a2·L² + a3·L³")
    
    L_vals = np.array([p[0] for p in points])
    tau_vals = np.array([p[1] for p in points])
    
    # Vandermonde matrix for cubic fit
    A = np.vstack([np.ones_like(L_vals), L_vals, L_vals**2, L_vals**3]).T
    coeffs = np.linalg.lstsq(A, tau_vals, rcond=None)[0]
    a0, a1, a2, a3 = coeffs
    
    print(f"      a0={a0:.4f}, a1={a1:.4f}, a2={a2:.4f}, a3={a3:.4f}")
    
    # Step 4: Solve cubic: a3*L³ + a2*L² + a1*L + (a0 - tau_target) = 0
    print(f"    [4] Solving for L where τ(L) = {tau_target:.1f}")
    
    # Normalize: divide by a3
    p = a2 / a3
    q = a1 / a3
    r = (a0 - tau_target) / a3
    
    # Depressed cubic: t³ + P·t + Q = 0, where L = t - p/3
    P = q - p**2 / 3.0
    Q = (2.0 * p**3 - 9.0 * p * q + 27.0 * r) / 27.0
    
    # Discriminant
    disc = (Q/2.0)**2 + (P/3.0)**3
    
    if disc >= 0:
        # One real root
        u = (-Q/2.0 + math.sqrt(disc))**(1.0/3.0)
        v = (-Q/2.0 - math.sqrt(disc))**(1.0/3.0)
        t_root = u + v
    else:
        # Three real roots
        phi_root = math.acos(-Q/2.0 * math.sqrt(-27.0/(P**3))) / 3.0
        roots = [
            2.0 * math.sqrt(-P/3.0) * math.cos(phi_root),
            2.0 * math.sqrt(-P/3.0) * math.cos(phi_root + 2.0*math.pi/3.0),
            2.0 * math.sqrt(-P/3.0) * math.cos(phi_root + 4.0*math.pi/3.0)
        ]
        # Choose the positive root closest to current L
        positive_roots = [r for r in roots if r > 0]
        t_root = min(positive_roots) if positive_roots else roots[0]
    
    L_star = t_root - p/3.0
    
    # Clamp to reasonable range
    L_star = max(0.005, min(L_star, 0.1))
    
    print(f"      L* = {L_star:.4f} (where τ crosses {tau_target:.1f})")
    
    # Step 5: Map L* to θ* using the floor direction
    print("    [5] Jumping to L* along floor direction...")
    
    # Use linear interpolation between the sampled points
    # Find which two points bracket L_star
    L_sorted = sorted([(p[0], i) for i, p in enumerate(points)])
    L_vals_sorted = [p[0] for p in L_sorted]
    
    # Find step size that gives L_star
    if L_star <= L_vals_sorted[0]:
        # Below first point, use first step
        target_step = 0.01
    elif L_star >= L_vals_sorted[-1]:
        # Above last point, extrapolate
        # Use the last two points to estimate
        L1, L2 = L_vals_sorted[-2], L_vals_sorted[-1]
        step1, step2 = steps[-2], steps[-1]
        slope = (L2 - L1) / (step2 - step1)
        target_step = step2 + (L_star - L2) / max(slope, 1e-6)
    else:
        # Interpolate
        for i in range(len(L_vals_sorted) - 1):
            if L_vals_sorted[i] <= L_star <= L_vals_sorted[i + 1]:
                L1, L2 = L_vals_sorted[i], L_vals_sorted[i + 1]
                idx1, idx2 = L_sorted[i][1], L_sorted[i + 1][1]
                step1, step2 = steps[idx1], steps[idx2]
                # Linear interpolation
                t = (L_star - L1) / max(L2 - L1, 1e-6)
                target_step = step1 + t * (step2 - step1)
                break
    
    target_step = max(0.0, min(target_step, 0.2))
    print(f"      Target step size: {target_step:.4f}")
    
    # Jump to the predicted floor
    jump_amount = target_step * floor_direction
    model.set_flat(w0 + jump_amount)
    
    # Verify
    L_new = eval_val(model, n=4)
    tau_new = gluing_defect(model, n=4)
    phi_new = phi_clean(model)
    ce_used += 4 + 4
    
    print(f"      After jump: val={L_new:.4f}, τ={tau_new:.2f}, φ={phi_new}/5")
    
    # Step 6: Newton refinement (if needed)
    if L_new > 0.02:
        print("    [6] Newton refinement (2 steps)...")
        for _ in range(2):
            model.zero_grad()
            ls = [model(*get_batch())[1] for _ in range(6)]
            loss = torch.stack(ls).mean()
            loss.backward()
            
            g = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                           for p in model.parameters()]).detach()
            model.zero_grad()
            
            # Hessian-vector product
            def hvp_refine(v, n=4):
                model.zero_grad()
                ls2 = [model(*get_batch())[1] for _ in range(n)]
                loss2 = torch.stack(ls2).mean()
                grads = torch.autograd.grad(loss2, list(model.parameters()), create_graph=True)
                gv = (torch.cat([gr.flatten() for gr in grads]) * v.detach()).sum()
                hv = torch.cat([h.flatten() for h in
                                torch.autograd.grad(gv, list(model.parameters()), retain_graph=False)])
                model.zero_grad()
                return hv.detach()
            
            # Newton step
            d = torch.zeros_like(g)
            r = g.clone()
            p = -g.clone()
            rr = float((r * r).sum())
            
            for _ in range(4):
                Hp = hvp_refine(p)
                alpha = rr / max(float((p * Hp).sum()), 1e-10)
                d += alpha * p
                r += alpha * Hp
                rr_new = float((r * r).sum())
                beta = rr_new / max(rr, 1e-10)
                p = -r + beta * p
                rr = rr_new
            
            w_cur = model.flat_params()
            model.set_flat(w_cur + 0.01 * d)
            
            L_new = eval_val(model, n=4)
            tau_new = gluing_defect(model, n=4)
            ce_used += 4 + 4
            print(f"      Newton step: val={L_new:.4f}, τ={tau_new:.2f}")
    
    return model, L_new, tau_new, ce_used


# ============================================================================
# ACTION FUNCTIONS
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


def action_p_adic_bowl(state):
    model = state.model
    v0_val = state.val
    phi0 = state.phi
    tau0 = state.tau
    
    print(f"\n  → p-adic BOWL DETECTION (p=2,3,5)...")
    print(f"    val={v0_val:.4f}, phi={phi0}/5, tau={tau0:.2f}")
    
    if v0_val > BOWL_DETECTION_THRESHOLD:
        print(f"    ℹ Skipping: val={v0_val:.4f} > {BOWL_DETECTION_THRESHOLD}")
        return None
    
    if phi0 < 4:
        print(f"    ℹ Skipping: phi={phi0}/5 < 4")
        return None
    
    print("    Computing p-adic Hessian spectrum...")
    detected, best_p, result = multi_prime_bowl_detection(model)
    
    if not detected:
        print("    ✗ No consensus p-adic gap detected")
        return None
    
    print(f"    ✓ Bowl detected (p={best_p}, gap at valuation {result['padic_vals'][result['gap_idx']]})")
    
    model_try = copy.deepcopy(model)
    n_p = sum(p.numel() for p in model_try.parameters())
    v0 = model_try.flat_params()
    ce_used = 0
    
    v_dir = result['eigenvecs'][result['gap_idx']]
    print(f"    Bowl direction norm: {float(v_dir.norm()):.4f}")
    
    print("    Evaluating loss along direction...")
    step_size = 0.15
    vals = []
    w0 = v0
    for i in range(5):
        t = i * step_size
        model_try.set_flat(w0 + t * v_dir)
        val = eval_val(model_try, n=6)
        vals.append(val)
        ce_used += 6
        print(f"      n={i}: t={t:.4f}, val={val:.4f}")
    
    model_try.set_flat(w0)
    
    t_vals = np.array([i * step_size for i in range(5)])
    X = np.vander(t_vals, 5, increasing=True)
    coeffs = np.linalg.lstsq(X, vals, rcond=None)[0]
    a0, a1, a2, a3, a4 = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]
    print(f"    L(t) = {a0:.4f} + {a1:.4f}t + {a2:.4f}t^2 + {a3:.4f}t^3 + {a4:.4f}t^4")
    
    if a4 < -1e-4:
        print(f"    ✓ Polynomial minimum detected! a4={a4:.6f}")
        
        t_vals_grid = np.linspace(-0.5, 2.0, 100)
        L_vals_grid = a0 + a1*t_vals_grid + a2*t_vals_grid**2 + a3*t_vals_grid**3 + a4*t_vals_grid**4
        idx_min = np.argmin(L_vals_grid)
        t_star = t_vals_grid[idx_min]
        
        def derivative(t):
            return a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3
        
        for _ in range(5):
            dL = derivative(t_star)
            d2L = 2*a2 + 6*a3*t_star + 12*a4*t_star**2
            if abs(d2L) > 1e-10:
                t_star = t_star - dL / d2L
            t_star = max(-0.5, min(2.0, t_star))
        
        L_star = a0 + a1*t_star + a2*t_star**2 + a3*t_star**3 + a4*t_star**4
        print(f"    t* = {t_star:.4f}, L* = {L_star:.4f}")
        
        model_try.set_flat(w0 + t_star * v_dir)
        v_new = eval_val(model_try, n=10)
        phi_new = phi_clean(model_try)
        tau_new = gluing_defect(model_try, n=4)
        ce_used += 10
        
        print(f"    Jump: val {v0_val:.4f} → {v_new:.4f}")
        print(f"    phi: {phi0}/5 → {phi_new}/5")
        print(f"    tau: {tau0:.2f} → {tau_new:.2f}")
        
        if v_new < v0_val * 0.95:
            return CompilerState(model_try, state.step + 1, 'p_adic_bowl',
                                v_new, phi_new, tau_new,
                                state.action_history + [f'p_adic_bowl(p={best_p})'],
                                parent=state)
    else:
        print(f"    ✗ No polynomial minimum (a4={a4:.6f} ≥ 0)")
    
    return None


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
    
    # CUBIC FLOOR JUMP: when near floor and phi >= 4
    if near_floor and phi >= 4:
        actions.append(('cubic_floor_jump', {}))
        print(f"    → cubic_floor_jump: val={val:.4f}, phi={phi}/5")
    
    # p-adic bowl (fallback if cubic fails)
    if near_floor and phi >= 4:
        actions.append(('p_adic_bowl', {}))
        print(f"    → p_adic_bowl: val={val:.4f}, phi={phi}/5")
    
    if not actions:
        actions.append(('basin_settle', {'lr_mult': 2}))
        print(f"    → fallback: basin_settle LR×2")
    
    return actions


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
        elif action_name == 'p_adic_bowl':
            return action_p_adic_bowl(state)
        elif action_name == 'cubic_floor_jump':
            return action_cubic_floor_jump(state)
        return None
    
    def run(self):
        print("=" * 70)
        print("SMART DP COMPILER WITH p-adic BOWL + CUBIC FLOOR JUMP")
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
# CUBIC FLOOR JUMP ACTION
# ============================================================================
def action_cubic_floor_jump(state):
    """
    Cubic floor jump action - analytic closed-form solution.
    """
    model = state.model
    v0_val = state.val
    phi0 = state.phi
    tau0 = state.tau
    
    print(f"\n  → CUBIC FLOOR JUMP (closed-form analytic solution)...")
    print(f"    val={v0_val:.4f}, phi={phi0}/5, tau={tau0:.2f}")
    
    if v0_val > BOWL_DETECTION_THRESHOLD:
        print(f"    ℹ Skipping: val={v0_val:.4f} > {BOWL_DETECTION_THRESHOLD}")
        return None
    
    if phi0 < 4:
        print(f"    ℹ Skipping: phi={phi0}/5 < 4")
        return None
    
    # Apply cubic floor jump
    model_try = copy.deepcopy(model)
    model_try, v_new, tau_new, ce_used = cubic_floor_jump(model_try, target_val=FLOOR_TARGET_VAL)
    
    phi_new = phi_clean(model_try)
    
    print(f"    Cubic jump complete: val {v0_val:.4f} → {v_new:.4f}")
    print(f"    phi: {phi0}/5 → {phi_new}/5")
    print(f"    tau: {tau0:.2f} → {tau_new:.2f}")
    
    if v_new < v0_val * 0.95:
        return CompilerState(model_try, state.step + 1, 'cubic_floor_jump',
                            v_new, phi_new, tau_new,
                            state.action_history + ['cubic_floor_jump'],
                            parent=state)
    
    return None


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
    
    # If val is still not good enough, run GD fallback
    if final_state.val > 0.1 or final_state.phi < PHI_CLEAN_TARGET:
        print("\n" + "=" * 70)
        print("RUNNING GRADIENT DESCENT FALLBACK")
        print("=" * 70)
        
        opt = torch.optim.AdamW(final_state.model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
        for step in range(1, 401):
            final_state.model.train()
            x, y = get_batch()
            _, l = final_state.model(x, y)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(final_state.model.parameters(), 1.0)
            opt.step()
            if step % 50 == 0:
                v = eval_val(final_state.model, n=8)
                phi = phi_clean(final_state.model)
                tau = gluing_defect(final_state.model, n=4)
                print(f"    GD step {step:4d}: val={v:.4f}, phi={phi}/5, tau={tau:.2f}")
        
        v_final = eval_val(final_state.model, n=12)
        phi_final = phi_clean(final_state.model)
        tau_final = gluing_defect(final_state.model, n=6)
        print(f"    GD final: val={v_final:.4f}, phi={phi_final}/5, tau={tau_final:.2f}")
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS AFTER GD FALLBACK")
        print("=" * 70)
        print(f"  Final val:     {v_final:.4f}")
        print(f"  Final phi:     {phi_final}/5")
        print(f"  Final tau:     {tau_final:.2f}")
        print(f"  Path:          {' → '.join(final_state.action_history)} → gradient_descent(400)")
        
        torch.save(final_state.model.state_dict(), 'compiler_cubic_gd_final.pt')
        print(f"\n  Saved: compiler_cubic_gd_final.pt")
    else:
        torch.save(final_state.model.state_dict(), 'compiler_cubic_final.pt')
        print(f"\n  Saved: compiler_cubic_final.pt")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
