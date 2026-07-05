#!/usr/bin/env python3
"""
Geometry-Driven Compiler with Spectral E0 + Optimized GD Fallback
================================================================
Total target: ~130 CE
  - Compiler: ~30-40 CE → val < 0.1
  - GD fallback: ~90-100 CE → val < 0.02

Spectral E0 provides the optimal initialization from corpus geometry.
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
MAX_COMPILER_STEPS = 30

PHI_CLEAN_TARGET = 5
TAU_MIN = 1.5
TAU_MAX = 5.7
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


def w_ff_formula_clamped(tau):
    if tau < TAU_MIN:
        return 3.5
    if tau > TAU_MAX:
        return 0.47
    return 3.5 * (TAU_MIN / max(tau, 0.1)) ** 1.5


def compute_spectral_E0():
    """Compute spectral embedding E0 from corpus Laplacian."""
    print("Computing spectral E0...")
    
    bigram = collections.Counter()
    perm = {}
    for i in range(len(train_ids) - 1):
        a, b = train_ids[i], train_ids[i + 1]
        if a < VOCAB and b < VOCAB:
            bigram[(a, b)] += 1
            perm.setdefault(a, b)

    rows, cols, vv = [], [], []
    for (a, b), cnt in bigram.items():
        rows.append(a)
        cols.append(b)
        vv.append(float(cnt))

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
    
    print(f"  Corpus: VOCAB={VOCAB}, nnz={len(bigram)}")
    return E_init, len(bigram)


# ============================================================================
# GD BASELINE (from E0)
# ============================================================================
def run_gd_from_E0(E_init, steps=400, lr=LR, log_every=50):
    """Run GD from E0 to establish baseline."""
    print("\n" + "=" * 70)
    print(f"GD-{steps} FROM E0 (BASELINE)")
    print("=" * 70)
    
    model = LM()
    model.te.weight.data.copy_(torch.tensor(E_init))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    
    history = []
    ce_used = 0
    
    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch()
        _, l = model(x, y)
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ce_used += 1
        
        if step % log_every == 0:
            v = eval_val(model, n=8)
            ce_used += 8
            phi = phi_clean(model)
            tau = gluing_defect(model, n=4)
            ce_used += 4
            history.append((step, v, phi, tau))
            print(f"  GD step {step:4d}: val={v:.4f}, phi={phi}/5, tau={tau:.2f}")
    
    v_final = eval_val(model, n=12)
    phi_final = phi_clean(model)
    tau_final = gluing_defect(model, n=6)
    
    print(f"\n  GD-{steps} final: val={v_final:.4f}, phi={phi_final}/5, tau={tau_final:.2f}")
    return model, v_final, phi_final, tau_final, ce_used, history


# ============================================================================
# OPTIMIZED GD FALLBACK (with cosine annealing, from compiler state)
# ============================================================================
def run_optimized_gd_fallback(model, steps=100, lr_start=LR, lr_end=LR*0.1):
    """
    Run optimized GD with cosine annealing.
    From compiler state (val ~0.075) to final (val < 0.02).
    """
    print("\n" + "=" * 70)
    print(f"OPTIMIZED GD FALLBACK ({steps} steps, cosine annealing)")
    print("=" * 70)
    
    model_try = copy.deepcopy(model)
    opt = torch.optim.AdamW(model_try.parameters(), lr=lr_start, betas=(0.9, 0.95), weight_decay=0.1)
    
    history = []
    ce_used = 0
    v0 = eval_val(model_try, n=6)
    ce_used += 6
    print(f"  Starting val: {v0:.4f}")
    
    for step in range(1, steps + 1):
        # Cosine annealing
        progress = step / steps
        lr_current = lr_end + (lr_start - lr_end) * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in opt.param_groups:
            pg['lr'] = lr_current
        
        model_try.train()
        x, y = get_batch()
        _, l = model_try(x, y)
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model_try.parameters(), 1.0)
        opt.step()
        ce_used += 1
        
        if step % 20 == 0 or step == steps:
            v = eval_val(model_try, n=8)
            ce_used += 8
            phi = phi_clean(model_try)
            tau = gluing_defect(model_try, n=4)
            ce_used += 4
            history.append((step, v, phi, tau))
            print(f"  GD step {step:4d}: val={v:.4f}, phi={phi}/5, tau={tau:.2f}, lr={lr_current:.6f}")
            
            # Early stopping if val < 0.015
            if v < 0.015:
                print(f"  ✓ Early stop at step {step}")
                break
    
    v_final = eval_val(model_try, n=12)
    ce_used += 12
    phi_final = phi_clean(model_try)
    tau_final = gluing_defect(model_try, n=6)
    ce_used += 6
    
    print(f"\n  GD final: val={v_final:.4f}, phi={phi_final}/5, tau={tau_final:.2f}")
    print(f"  GD CE used: {ce_used}")
    
    return model_try, v_final, phi_final, tau_final, ce_used, history


# ============================================================================
# COMPILER ACTIONS (simplified, from your working run)
# ============================================================================
class CompilerState:
    def __init__(self, model, step, phase, val, phi, tau, action_history=None, parent=None, ce_used=0):
        self.model = copy.deepcopy(model)
        self.step = step
        self.phase = phase
        self.val = val
        self.phi = phi
        self.tau = tau
        self.action_history = action_history or []
        self.parent = parent
        self.ce_used = ce_used
        self.total_ce = ce_used + (parent.total_ce if parent else 0)

    def __repr__(self):
        return f"Step {self.step}: val={self.val:.4f}, phi={self.phi}/5, tau={self.tau:.2f} (CE={self.total_ce})"


def action_mf_pump(state):
    """MF pump from the working run."""
    eta = min(0.02, 0.01 / (1 + state.val))
    eta = max(0.001, eta)
    
    model = state.model
    model_try = copy.deepcopy(model)
    v0 = state.val
    ce_used = 0
    
    for mf_r in range(1, 4):
        # E step
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
            ce_used += 1 / (N_SUB // 3)
        
        emb_grad /= (N_SUB // 3)
        emb_fish /= (N_SUB // 3)
        delta_E = -(emb_grad / (emb_fish + 1e-4))
        with torch.no_grad():
            model_try.te.weight.add_(eta * delta_E)
        
        for l in range(N_STU):
            model_try.blocks[l].attn.WK.weight.requires_grad_(True)
            model_try.blocks[l].attn.WQ.weight.requires_grad_(True)
        
        # WK step
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
            ce_used += 1 / (N_SUB // 3)
        
        wk_grad /= (N_SUB // 3)
        wk_fish /= (N_SUB // 3)
        delta_WK = -(wk_grad / (wk_fish + 1e-4))
        with torch.no_grad():
            for l in range(N_STU):
                model_try.blocks[l].attn.WK.weight.add_(eta * delta_WK)
                model_try.blocks[l].attn.WQ.weight.add_(eta * delta_WK.T)
        model_try.te.weight.requires_grad_(True)
        
        v_mf = eval_val(model_try, n=4)
        ce_used += 4
        if v_mf > v0 * 1.5:
            return None
        if phi_clean(model_try) >= 4:
            break
    
    v_try = eval_val(model_try, n=6)
    ce_used = int(ce_used + 6)
    if v_try < state.val * 0.95:
        return CompilerState(model_try, state.step + 1, 'mf_pump',
                            v_try, phi_clean(model_try), gluing_defect(model_try),
                            state.action_history + [f'mf_pump(η={eta:.4f}, {ce_used}CE)'],
                            parent=state, ce_used=ce_used)
    return None


def action_basin_settle(state, lr_mult):
    """Basin settle with given LR multiplier."""
    model = state.model
    v0 = state.val
    ce_used = 0
    max_steps = 80 if lr_mult == 1 else 40
    
    model_try = copy.deepcopy(model)
    opt_b = torch.optim.AdamW(model_try.parameters(), lr=LR * lr_mult,
                              betas=(0.9, 0.95), weight_decay=0.1)
    
    for step in range(1, max_steps + 1):
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
        ce_used += 1
        
        if step % 8 == 0:
            v = eval_val(model_try, n=4)
            ce_used += 4
            if v > v0 * 1.5:
                return None
            if v < 0.1 and lr_mult <= 2:
                break
    
    v_try = eval_val(model_try, n=6)
    ce_used += 6
    threshold = 0.98 if lr_mult <= 2 else 0.95
    if v_try < state.val * threshold:
        return CompilerState(model_try, state.step + 1, 'basin_settle',
                            v_try, phi_clean(model_try), gluing_defect(model_try),
                            state.action_history + [f'basin_settle(LR×{lr_mult}, {ce_used}CE)'],
                            parent=state, ce_used=ce_used)
    return None


# ============================================================================
# SIMPLE COMPILER (based on your working run pattern)
# ============================================================================
def run_compiler(E_init):
    """Run the compiler with the proven pattern from your working run."""
    print("=" * 70)
    print("GEOMETRY-DRIVEN COMPILER")
    print("=" * 70)
    print("  Pattern: mf_pump → basin_settle(LR×10) → basin_settle(LR×5) → basin_settle(LR×3)")
    print("=" * 70)
    print()
    
    # Initialize model with spectral E0
    model = LM()
    model.te.weight.data.copy_(torch.tensor(E_init))
    
    v0 = eval_val(model, n=10)
    phi0 = phi_clean(model)
    tau0 = gluing_defect(model, n=6)
    
    print(f"Initial: val={v0:.4f}, phi={phi0}/5, tau={tau0:.2f}")
    print()
    
    # Create initial state
    initial = CompilerState(model, 0, 'init', v0, phi0, tau0, [], None, 0)
    current = initial
    total_ce = 0
    
    # Step 1: MF pump
    print(f"\n[Step 1] Trying MF pump...")
    result = action_mf_pump(current)
    if result is not None:
        current = result
        total_ce = current.total_ce
        print(f"  ✓ MF pump: val={current.val:.4f}, phi={current.phi}/5 (CE={total_ce})")
    else:
        print("  ⊘ MF pump failed")
    
    # Step 2: Basin settle LR×10
    print(f"\n[Step 2] Trying basin_settle LR×10...")
    result = action_basin_settle(current, 10)
    if result is not None:
        current = result
        total_ce = current.total_ce
        print(f"  ✓ Basin settle LR×10: val={current.val:.4f}, phi={current.phi}/5 (CE={total_ce})")
    else:
        print("  ⊘ Basin settle LR×10 failed")
    
    # Step 3: Basin settle LR×5
    print(f"\n[Step 3] Trying basin_settle LR×5...")
    result = action_basin_settle(current, 5)
    if result is not None:
        current = result
        total_ce = current.total_ce
        print(f"  ✓ Basin settle LR×5: val={current.val:.4f}, phi={current.phi}/5 (CE={total_ce})")
    else:
        print("  ⊘ Basin settle LR×5 failed")
    
    # Step 4: Basin settle LR×3
    print(f"\n[Step 4] Trying basin_settle LR×3...")
    result = action_basin_settle(current, 3)
    if result is not None:
        current = result
        total_ce = current.total_ce
        print(f"  ✓ Basin settle LR×3: val={current.val:.4f}, phi={current.phi}/5 (CE={total_ce})")
    else:
        print("  ⊘ Basin settle LR×3 failed")
    
    # Step 5: Basin settle LR×1 (if val > 0.1)
    if current.val > 0.1:
        print(f"\n[Step 5] Trying basin_settle LR×1 (small steps)...")
        result = action_basin_settle(current, 1)
        if result is not None:
            current = result
            total_ce = current.total_ce
            print(f"  ✓ Basin settle LR×1: val={current.val:.4f}, phi={current.phi}/5 (CE={total_ce})")
        else:
            print("  ⊘ Basin settle LR×1 failed")
    
    print("\n" + "=" * 70)
    print("COMPILER RESULTS")
    print("=" * 70)
    print(f"  Final val:      {current.val:.4f}")
    print(f"  Final phi:      {current.phi}/5")
    print(f"  Final tau:      {current.tau:.2f}")
    print(f"  Total CE:       {current.total_ce}")
    print(f"  Path:           {' → '.join(current.action_history)}")
    
    return current


# ============================================================================
# MAIN
# ============================================================================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Compute spectral E0 from corpus
    E_init, nnz = compute_spectral_E0()
    print()
    
    # 2. Run GD-400 baseline from E0
    gd_model, gd_val, gd_phi, gd_tau, gd_ce, gd_history = run_gd_from_E0(
        E_init, steps=400, lr=LR, log_every=50
    )
    print()
    
    # 3. Run compiler
    compiler_state = run_compiler(E_init)
    print()
    
    # 4. Run optimized GD fallback from compiler state
    if compiler_state.val > 0.015:
        gd_opt_model, gd_opt_val, gd_opt_phi, gd_opt_tau, gd_opt_ce, gd_opt_history = run_optimized_gd_fallback(
            compiler_state.model, steps=100, lr_start=LR, lr_end=LR*0.1
        )
    else:
        gd_opt_model = compiler_state.model
        gd_opt_val = compiler_state.val
        gd_opt_phi = compiler_state.phi
        gd_opt_tau = compiler_state.tau
        gd_opt_ce = 0
    
    total_ce = compiler_state.total_ce + gd_opt_ce
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  {'Metric':<30} {'Compiler+GD':>14} {'GD-400':>12}")
    print("  " + "-" * 58)
    print(f"  {'Final val':<30} {gd_opt_val:>14.4f} {gd_val:>12.4f}")
    print(f"  {'Final phi':<30} {gd_opt_phi:>14}/5 {gd_phi:>12}/5")
    print(f"  {'Final tau':<30} {gd_opt_tau:>14.2f} {gd_tau:>12.2f}")
    print(f"  {'Compiler CE':<30} {compiler_state.total_ce:>14} {'0':>12}")
    print(f"  {'GD fallback CE':<30} {gd_opt_ce:>14} {'400':>12}")
    print(f"  {'Total CE':<30} {total_ce:>14} {'400':>12}")
    
    if gd_opt_val < gd_val:
        advantage = gd_val / gd_opt_val
        print(f"  {'Advantage vs GD-400':<30} {advantage:>13.2f}× {'1.0×':>12}")
    
    # Save models
    torch.save(gd_model.state_dict(), 'gd_baseline.pt')
    torch.save(gd_opt_model.state_dict(), 'compiler_gd_optimized.pt')
    print(f"\n  Saved: gd_baseline.pt, compiler_gd_optimized.pt")


if __name__ == '__main__':
    main()
