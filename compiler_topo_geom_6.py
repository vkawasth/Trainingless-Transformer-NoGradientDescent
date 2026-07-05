#!/usr/bin/env python3
"""
Compiler with Bowl Detection
=============================
Detects the small bowl in the flat basin and switches to small steps.

The bowl has geometric signatures:
  1. Val plateau (Δval < 0.01)
  2. phi=5/5 stable
  3. τ changing (spike or drop)
  4. Gradient norm increasing

Once detected, switch to LR×0.5 to carefully descend into the bowl.
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
GD_FALLBACK_STEPS = 200

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


def gradient_norm(model, n=6):
    model.zero_grad()
    ls = [model(*get_batch())[1] for _ in range(n)]
    torch.stack(ls).mean().backward()
    g_norm = float(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).norm())
    model.zero_grad()
    return g_norm


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


# ============================================================================
# BOWL DETECTION
# ============================================================================
class BowlDetector:
    def __init__(self, window=5):
        self.history = []
        self.window = window
        self.bowl_detected = False
    
    def update(self, val, phi, tau, grad_norm):
        self.history.append({
            'val': val,
            'phi': phi,
            'tau': tau,
            'grad_norm': grad_norm
        })
        if len(self.history) > 20:
            self.history = self.history[-20:]
    
    def detect(self):
        if len(self.history) < self.window:
            return False
        
        recent = self.history[-self.window:]
        vals = [h['val'] for h in recent]
        phis = [h['phi'] for h in recent]
        taus = [h['tau'] for h in recent]
        grad_norms = [h['grad_norm'] for h in recent]
        
        # 1. Val plateau: small change over window
        val_delta = max(vals) - min(vals)
        val_plateau = val_delta < 0.01
        
        # 2. phi stable at 5/5
        phi_stable = all(p == 5 for p in phis)
        
        # 3. tau changing (spike or drop) - indicating bowl boundary
        tau_range = max(taus) - min(taus)
        tau_changing = tau_range > 0.3
        
        # 4. Gradient norm increasing - entering curvature
        grad_increasing = grad_norms[-1] > grad_norms[0] * 1.3
        
        # 5. Val is below 0.15 (we're in the basin floor region)
        val_low = vals[-1] < 0.15
        
        # Combined signature
        if val_plateau and phi_stable and (tau_changing or grad_increasing) and val_low:
            self.bowl_detected = True
            print(f"\n  🎯 BOWL DETECTED!")
            print(f"    val: {vals[-1]:.4f}, phi={phis[-1]}/5")
            print(f"    τ range: {min(taus):.2f} → {max(taus):.2f} (Δ={tau_range:.2f})")
            print(f"    gradient norm: {grad_norms[0]:.4f} → {grad_norms[-1]:.4f}")
            return True
        
        return False


# ============================================================================
# COMPILER STATE
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
        orbit_ok = "✓" if self.phi == PHI_CLEAN_TARGET and TAU_MIN <= self.tau <= TAU_MAX else "✗"
        return (f"Step {self.step}: val={self.val:.4f}, phi={self.phi}/5, "
                f"tau={self.tau:.2f} {orbit_ok} (CE={self.total_ce})")


# ============================================================================
# ACTION FUNCTIONS
# ============================================================================
def action_mf_pump(state):
    eta = min(0.02, 0.01 / (1 + state.val))
    eta = max(0.001, eta)
    
    model = state.model
    model_try = copy.deepcopy(model)
    v0 = state.val
    ce_used = 0
    
    for mf_r in range(1, 3):
        for l in range(N_STU):
            model_try.blocks[l].attn.WK.weight.requires_grad_(False)
            model_try.blocks[l].attn.WQ.weight.requires_grad_(False)
        
        emb_grad = torch.zeros(model_try.te.weight.shape)
        emb_fish = torch.zeros(model_try.te.weight.shape)
        torch.manual_seed((mf_r - 1) * 1000)
        
        n_sub = N_SUB // 4
        for i in range(n_sub):
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
            ce_used += 1 / n_sub
        
        emb_grad /= n_sub
        emb_fish /= n_sub
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
        
        for i in range(n_sub):
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
            ce_used += 1 / n_sub
        
        wk_grad /= n_sub
        wk_fish /= n_sub
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


def action_basin_settle(state, lr_mult, steps=40):
    """Basin settle with given LR multiplier."""
    model = state.model
    v0 = state.val
    ce_used = 0
    
    model_try = copy.deepcopy(model)
    opt_b = torch.optim.AdamW(model_try.parameters(), lr=LR * lr_mult,
                              betas=(0.9, 0.95), weight_decay=0.1)
    
    for step in range(1, steps + 1):
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
# BOWL DESCENT (Small steps after bowl detection)
# ============================================================================
def bowl_descent(state, detector, steps=30):
    """Descent into the small bowl with LR×0.5 after detection."""
    model = state.model
    v0 = state.val
    ce_used = 0
    
    model_try = copy.deepcopy(model)
    opt_b = torch.optim.AdamW(model_try.parameters(), lr=LR * 0.5,
                              betas=(0.9, 0.95), weight_decay=0.1)
    
    best_val = v0
    best_model = copy.deepcopy(model_try)
    best_phi = phi_clean(model_try)
    
    for step in range(1, steps + 1):
        model_try.train()
        x, y = get_batch()
        _, l = model_try(x, y)
        opt_b.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model_try.parameters(), 1.0)
        opt_b.step()
        ce_used += 1
        
        if step % 5 == 0:
            v = eval_val(model_try, n=4)
            ce_used += 4
            phi = phi_clean(model_try)
            tau = gluing_defect(model_try, n=4)
            grad_norm = gradient_norm(model_try, n=4)
            
            print(f"    Bowl step {step:2d}: val={v:.4f}, phi={phi}/5, tau={tau:.2f}")
            
            if v < best_val:
                best_val = v
                best_model = copy.deepcopy(model_try)
                best_phi = phi
            
            # Update detector with bowl descent progress
            detector.update(v, phi, tau, grad_norm)
            
            if v < 0.02:
                break
    
    v_try = eval_val(best_model, n=6)
    ce_used += 6
    phi_try = phi_clean(best_model)
    
    if v_try < state.val * 0.95 or phi_try > state.phi:
        return CompilerState(best_model, state.step + 1, 'bowl_descent',
                            v_try, phi_try, gluing_defect(best_model),
                            state.action_history + [f'bowl_descent({ce_used}CE)'],
                            parent=state, ce_used=ce_used)
    return None


# ============================================================================
# SMART ACTION SELECTION WITH BOWL DETECTION
# ============================================================================
def get_smart_actions(state, detector):
    """Select actions based on landscape and bowl detection."""
    val = state.val
    phi = state.phi
    tau = state.tau
    
    actions = []
    
    print(f"\n  State: val={val:.4f}, phi={phi}/5, tau={tau:.2f}")
    
    # 1. MF pump if val > 0.3 and phi < 5
    if val > 0.3 and phi < PHI_CLEAN_TARGET:
        actions.append(('mf_pump', {}))
        print(f"    → mf_pump")
    
    # 2. Basin settle with progressive LR
    if val > 1.0:
        actions.append(('basin_settle', {'lr_mult': 10, 'steps': 40}))
        actions.append(('basin_settle', {'lr_mult': 5, 'steps': 40}))
    elif val > 0.3:
        actions.append(('basin_settle', {'lr_mult': 5, 'steps': 40}))
        actions.append(('basin_settle', {'lr_mult': 3, 'steps': 40}))
    else:
        actions.append(('basin_settle', {'lr_mult': 3, 'steps': 40}))
        actions.append(('basin_settle', {'lr_mult': 2, 'steps': 40}))
        actions.append(('basin_settle', {'lr_mult': 1, 'steps': 30}))
    
    # 3. Bowl descent if detector says we're in the bowl
    if detector.bowl_detected:
        actions = [('bowl_descent', {'steps': 30})]
        print(f"    → bowl_descent (bowl detected!)")
    
    return actions


# ============================================================================
# GRADIENT DESCENT FALLBACK (200 steps)
# ============================================================================
def run_gd_fallback(model, steps=200):
    print("\n" + "=" * 70)
    print(f"GD FALLBACK ({steps} steps)")
    print("=" * 70)
    
    model_try = copy.deepcopy(model)
    opt = torch.optim.AdamW(model_try.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    
    ce_used = 0
    best_val = eval_val(model_try, n=4)
    ce_used += 4
    best_model = copy.deepcopy(model_try)
    
    for step in range(1, steps + 1):
        lr_s = LR * 0.5 * (1 + math.cos(math.pi * step / steps))
        lr_s = max(lr_s, LR * 0.01)
        for pg in opt.param_groups:
            pg['lr'] = lr_s
        
        model_try.train()
        x, y = get_batch()
        _, l = model_try(x, y)
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model_try.parameters(), 1.0)
        opt.step()
        ce_used += 1
        
        if step % 50 == 0:
            v = eval_val(model_try, n=8)
            ce_used += 8
            phi = phi_clean(model_try)
            print(f"    GD step {step:4d}: val={v:.4f}, phi={phi}/5")
            if v < best_val:
                best_val = v
                best_model = copy.deepcopy(model_try)
    
    v_final = eval_val(best_model, n=12)
    ce_used += 12
    phi_final = phi_clean(best_model)
    tau_final = gluing_defect(best_model, n=6)
    ce_used += 6
    
    print(f"\n    GD final: val={v_final:.4f}, phi={phi_final}/5, tau={tau_final:.2f}")
    print(f"    GD CE: {ce_used}")
    
    return best_model, v_final, phi_final, tau_final, ce_used


# ============================================================================
# SMART DP COMPILER WITH BOWL DETECTION
# ============================================================================
class SmartDPCompiler:
    def __init__(self):
        self.detector = BowlDetector(window=5)
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
            
            actions = get_smart_actions(state, self.detector)
            untried = [(name, params) for name, params in actions 
                      if (state_key, name, tuple(sorted(params.items()))) not in self.failed_actions]
            
            if untried:
                return i, state, untried
        
        return None, None, None
    
    def apply_action(self, state, action_name, params):
        if action_name == 'mf_pump':
            return action_mf_pump(state)
        elif action_name == 'basin_settle':
            return action_basin_settle(state, params['lr_mult'], params.get('steps', 40))
        elif action_name == 'bowl_descent':
            return bowl_descent(state, self.detector, params.get('steps', 30))
        return None
    
    def run(self):
        print("=" * 70)
        print("COMPILER WITH BOWL DETECTION")
        print("=" * 70)
        print("  Detects the small bowl in the flat basin")
        print("  Switches to LR×0.5 when bowl is detected")
        print("=" * 70)
        print()
        
        E_init = compute_E_init()
        model = LM()
        model.te.weight.data.copy_(torch.tensor(E_init))
        
        v0 = eval_val(model, n=10)
        phi0 = phi_clean(model)
        tau0 = gluing_defect(model, n=6)
        grad0 = gradient_norm(model, n=6)
        
        print(f"Initial: val={v0:.4f}, phi={phi0}/5, tau={tau0:.2f}")
        print()
        
        initial_state = CompilerState(model, 0, 'init', v0, phi0, tau0, [], None, 0)
        state_stack = [initial_state]
        self.best_state = initial_state
        self.visited_states = [self.get_state_key(initial_state)]
        
        for step_counter in range(MAX_DP_STEPS):
            current = state_stack[-1]
            print(f"\n[Step {step_counter}] Current: {current}")
            
            if current.val < VAL_FLOOR and current.phi == PHI_CLEAN_TARGET:
                print(f"\n✓ SUCCESS! val={current.val:.4f}, phi={current.phi}/5")
                return current
            
            actions = get_smart_actions(current, self.detector)
            state_key = self.get_state_key(current)
            
            untried = [(name, params) for name, params in actions 
                      if (state_key, name, tuple(sorted(params.items()))) not in self.failed_actions]
            
            if not untried:
                print(f"\n  ⊘ No untried actions at {current}")
                idx, back_state, back_untried = self.find_backtrack_state(state_stack)
                if back_state is None:
                    print(f"\n⚠ No states with untried actions. Best val={self.best_state.val:.4f}")
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
                
                # Update detector with result
                grad_norm = gradient_norm(result.model, n=4)
                self.detector.update(result.val, result.phi, result.tau, grad_norm)
                
                if result.val < current.val * 0.95:
                    print(f"    ✓ {action_name}: {current.val:.4f} → {result.val:.4f} (phi={result.phi}/5)")
                    state_stack.append(result)
                    self.visited_states.append(self.get_state_key(result))
                    
                    if result.val < self.best_state.val:
                        self.best_state = result
                        print(f"    ★ New best: {result.val:.4f} (CE={result.total_ce})")
                    
                    action_taken = True
                    break
                elif result.phi > current.phi and result.val < current.val * 1.1:
                    print(f"    ~ {action_name}: phi {current.phi}→{result.phi}/5")
                    state_stack.append(result)
                    self.visited_states.append(self.get_state_key(result))
                    action_taken = True
                    break
                else:
                    print(f"    ⊘ {action_name}: no gain")
                    self.mark_failed(current, action_name, params)
            
            if not action_taken:
                print(f"\n  ⊘ No working action at {current}")
                self.exhausted_states.add(state_key)
        
        return self.best_state


# ============================================================================
# MAIN
# ============================================================================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    compiler = SmartDPCompiler()
    compiler_state = compiler.run()
    
    print("\n" + "=" * 70)
    print("COMPILER RESULTS (Before GD)")
    print("=" * 70)
    print(f"  Compiler val:      {compiler_state.val:.4f}")
    print(f"  Compiler phi:      {compiler_state.phi}/5")
    print(f"  Compiler CE:       {compiler_state.total_ce}")
    print(f"  Path:              {' → '.join(compiler_state.action_history)}")
    
    # Run GD fallback (200 steps)
    if compiler_state.val > 0.02:
        model_final, v_final, phi_final, tau_final, gd_ce = run_gd_fallback(
            compiler_state.model, steps=200
        )
    else:
        model_final = compiler_state.model
        v_final = compiler_state.val
        phi_final = compiler_state.phi
        tau_final = compiler_state.tau
        gd_ce = 0
    
    total_ce = compiler_state.total_ce + gd_ce
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Final val:         {v_final:.4f}")
    print(f"  Final phi:         {phi_final}/5")
    print(f"  Final tau:         {tau_final:.2f}")
    print(f"  Compiler CE:       {compiler_state.total_ce}")
    print(f"  GD CE:             {gd_ce}")
    print(f"  Total CE:          {total_ce}")
    print(f"  Path:              {' → '.join(compiler_state.action_history)} → gd_fallback({gd_ce}CE)")
    
    gd_val = 0.0914
    print("\n" + "=" * 70)
    print("COMPARISON: COMPILER+GD vs GD-400")
    print("=" * 70)
    print(f"  {'Metric':<30} {'Compiler+GD':>14} {'GD-400':>12}")
    print("  " + "-" * 58)
    print(f"  {'Final val':<30} {v_final:>14.4f} {gd_val:>12.4f}")
    print(f"  {'Final phi':<30} {phi_final:>14}/5 {'4/5':>12}")
    print(f"  {'Total CE':<30} {total_ce:>14} {'400':>12}")
    
    if v_final < gd_val:
        advantage = gd_val / v_final
        print(f"  {'Advantage':<30} {advantage:>13.2f}× {'1.0×':>12}")
    else:
        print(f"  {'Advantage':<30} {'1.0×':>12} {'1.0×':>12}")
    
    torch.save(model_final.state_dict(), 'bowl_detection_compiler.pt')
    print(f"\n  Saved: bowl_detection_compiler.pt")


if __name__ == '__main__':
    main()
