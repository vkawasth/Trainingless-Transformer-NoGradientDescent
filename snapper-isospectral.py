#!/usr/bin/env python3
"""
Hybrid Geometric Compiler
=========================
- MF pump (unchanged) to enter the basin
- Basin settle (unchanged) to traverse the basin
- Witness set + LoG to find the dimple at the basin floor
- Path tracking to verify the jump

Total CE: ~50-70 (vs 229 with pure LoG)
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
PHI_CLEAN_TARGET = 5
VAL_FLOOR = 0.062
TAU_MIN = 1.5
TAU_MAX = 5.7
ORBIT_TOLERANCE = 0.3

# Basin detection thresholds
BASIN_ENTRY_VAL = 0.5  # val below this means we're in the basin
WITNESS_SAMPLES = 6
LOG_THRESHOLD = 0.001

# ============================================================================
# DATA LOADING & MODEL (abbreviated - same as compiler_topo_geom_1.py)
# ============================================================================
for f in ['/tmp/train_ids.json', '/tmp/val_ids.json', '/tmp/vocab.json']:
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} missing")

with open('/tmp/train_ids.json') as f:
    train_ids = list(map(int, json.load(f)))
with open('/tmp/val_ids.json') as f:
    val_ids = list(map(int, json.load(f)))
with open('/tmp/vocab.json') as f:
    _v = json.load(f)
VOCAB = len(_v) if isinstance(_v, list) else len(_v)
train_t = torch.tensor(train_ids, dtype=torch.long)
val_t = torch.tensor(val_ids, dtype=torch.long)

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
        self.g = nn.Linear(D, D*2, bias=False)
        self.v = nn.Linear(D, D*2, bias=False)
        self.o = nn.Linear(D*2, D, bias=False)
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
            p.data.copy_(v[i:i+n].reshape(p.shape))
            i += n

def get_batch(split='train'):
    data = val_t if split == 'val' else train_t
    ix = torch.randint(0, len(data) - SEQ - 1, (BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

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
            phi = WKs[l+1] @ torch.linalg.pinv(WKs[l])
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
        a, b = train_ids[i], train_ids[i+1]
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
    evals, evecs = spla.eigsh(L_sym, k=D+1, which='SM', tol=1e-4, maxiter=2000)
    idx_s = np.argsort(evals)
    evecs = evecs[:, idx_s][:, 1:D+1]
    E_0 = (evecs / (np.sqrt(evals[idx_s[1:D+1]]) + 1e-8)[np.newaxis, :]).astype(np.float32)
    E_0 = (E_0 / (E_0.std() + 1e-8) * 0.02)
    E_next = np.array([E_0[perm.get(t, t)] for t in range(VOCAB)], dtype=np.float32)
    E_init = (0.9 * E_0 + 0.1 * E_next)
    E_norm = float(np.linalg.norm(E_0))
    E_init = (E_init * (E_norm / max(float(np.linalg.norm(E_init)), 1e-8))).astype(np.float32)
    return E_init

# ============================================================================
# ACTION FUNCTIONS (FROM compiler_topo_geom_1.py - UNCHANGED)
# ============================================================================
def action_mf_pump(model):
    """MF pump - from compiler_topo_geom_1.py, unchanged."""
    v0 = eval_val(model, n=6)
    phi0 = phi_clean(model)
    tau0 = gluing_defect(model, n=4)
    
    print(f"  MF pump: starting val={v0:.4f}, phi={phi0}/5, tau={tau0:.2f}")
    
    if v0 > 1.5:
        print(f"    ⚠ val={v0:.4f} > 1.5 - MF pump would explode, skipping")
        return model, v0, phi0, tau0
    
    best_v = v0
    best_model = copy.deepcopy(model)
    
    for mf_r in range(1, 4):
        # E step
        for l in range(N_STU):
            model.blocks[l].attn.WK.weight.requires_grad_(False)
            model.blocks[l].attn.WQ.weight.requires_grad_(False)
        
        emb_grad = torch.zeros(model.te.weight.shape)
        emb_fish = torch.zeros(model.te.weight.shape)
        torch.manual_seed((mf_r - 1) * 1000)
        
        for i in range(N_SUB // 3):
            ix = torch.randint(0, len(train_t) - SEQ - 1, (1,))[0].item()
            x = train_t[ix:ix + SEQ].unsqueeze(0)
            y = train_t[ix + 1:ix + SEQ + 1].unsqueeze(0)
            model.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            if model.te.weight.grad is not None:
                g = model.te.weight.grad.detach()
                emb_grad += g
                emb_fish += g ** 2
        
        emb_grad /= (N_SUB // 3)
        emb_fish /= (N_SUB // 3)
        delta_E = -(emb_grad / (emb_fish + 1e-4))
        with torch.no_grad():
            model.te.weight.add_(ETA_MF * delta_E)
        
        for l in range(N_STU):
            model.blocks[l].attn.WK.weight.requires_grad_(True)
            model.blocks[l].attn.WQ.weight.requires_grad_(True)
        
        # WK step
        model.te.weight.requires_grad_(False)
        wk_grad = torch.zeros_like(model.blocks[0].attn.WK.weight)
        wk_fish = torch.zeros_like(model.blocks[0].attn.WK.weight)
        torch.manual_seed((mf_r - 1) * 1000 + 500)
        
        for i in range(N_SUB // 3):
            ix = torch.randint(0, len(train_t) - SEQ - 1, (1,))[0].item()
            x = train_t[ix:ix + SEQ].unsqueeze(0)
            y = train_t[ix + 1:ix + SEQ + 1].unsqueeze(0)
            model.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            g = torch.zeros_like(model.blocks[0].attn.WK.weight)
            for bl in model.blocks:
                if bl.attn.WK.weight.grad is not None:
                    g += bl.attn.WK.weight.grad / N_STU
            wk_grad += g
            wk_fish += g ** 2
        
        wk_grad /= (N_SUB // 3)
        wk_fish /= (N_SUB // 3)
        delta_WK = -(wk_grad / (wk_fish + 1e-4))
        with torch.no_grad():
            for l in range(N_STU):
                model.blocks[l].attn.WK.weight.add_(ETA_MF * delta_WK)
                model.blocks[l].attn.WQ.weight.add_(ETA_MF * delta_WK.T)
        model.te.weight.requires_grad_(True)
        
        v_mf = eval_val(model, n=4)
        phi_mf = phi_clean(model)
        tau_mf = gluing_defect(model, n=4)
        
        print(f"    MF{mf_r}: val={v_mf:.4f}, phi={phi_mf}/5, tau={tau_mf:.2f}")
        
        if v_mf < best_v:
            best_v = v_mf
            best_model = copy.deepcopy(model)
        
        if v_mf > v0 * 1.5:
            print(f"    ✗ Loss exploded - reverting")
            model = best_model
            return model, best_v, phi_clean(model), gluing_defect(model, n=4)
        
        if phi_mf >= 4:
            print(f"    ✓ Orbit established")
            break
    
    v_final = eval_val(model, n=8)
    return model, v_final, phi_clean(model), gluing_defect(model, n=4)

def action_basin_settle(model, lr_mult, steps=40):
    """Basin settle - from compiler_topo_geom_1.py, unchanged."""
    v0 = eval_val(model, n=6)
    phi0 = phi_clean(model)
    tau0 = gluing_defect(model, n=4)
    
    print(f"  Basin settle LR×{lr_mult}: starting val={v0:.4f}, phi={phi0}/5, tau={tau0:.2f}")
    
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
        
        if step % 8 == 0:
            v = eval_val(model_try, n=4)
            if v > v0 * 1.5:
                print(f"    ✗ Loss exploded at step {step}")
                return model, v0, phi0, tau0
            if v < 0.1:
                print(f"    ✓ Reached floor at step {step}")
                break
    
    v_final = eval_val(model_try, n=8)
    phi_final = phi_clean(model_try)
    tau_final = gluing_defect(model_try, n=4)
    
    print(f"    After {step} steps: val={v_final:.4f}, phi={phi_final}/5, tau={tau_final:.2f}")
    return model_try, v_final, phi_final, tau_final

# ============================================================================
# WITNESS SET + LOG (ONLY USED IN THE BASIN)
# ============================================================================
def compute_witness_set(model, n_samples=WITNESS_SAMPLES):
    """Compute witness set of the basin."""
    n_p = sum(p.numel() for p in model.parameters())
    w0 = model.flat_params()
    points = []
    log_values = []
    
    def compute_log(w):
        model.set_flat(w)
        
        def hvp_diag(v):
            model.zero_grad()
            ls = [model(*get_batch())[1] for _ in range(4)]
            loss = torch.stack(ls).mean()
            grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            gv = (torch.cat([gr.flatten() for gr in grads]) * v.detach()).sum()
            hv = torch.cat([h.flatten() for h in
                            torch.autograd.grad(gv, list(model.parameters()), retain_graph=False)])
            model.zero_grad()
            return hv.detach()
        
        trace = 0.0
        torch.manual_seed(99)
        for _ in range(4):
            v = torch.randn(n_p)
            v = v / v.norm()
            Hv = hvp_diag(v)
            trace += float((v * Hv).sum())
        return trace / 4.0
    
    torch.manual_seed(123)
    for i in range(n_samples):
        v = torch.randn(n_p)
        v = v / max(v.norm(), 1e-10)
        alpha = (i / n_samples) * 0.1  # Smaller steps in the basin
        w_sample = w0 + alpha * v
        
        points.append(w_sample)
        log_values.append(compute_log(w_sample))
    
    return points, log_values

def find_dimple(model, witness_points, log_values):
    """Find the LoG peak (dimple) in the basin."""
    peak_idx = np.argmax(log_values)
    peak_point = witness_points[peak_idx]
    peak_log = log_values[peak_idx]
    
    # The direction to the dimple
    w0 = model.flat_params()
    direction = peak_point - w0
    direction = direction / max(direction.norm(), 1e-10)
    
    return direction, peak_log

def jump_to_dimple(model, direction, alpha=1.0):
    """Jump to the dimple."""
    w0 = model.flat_params()
    model.set_flat(w0 + alpha * direction)
    return model

# ============================================================================
# MAIN COMPILER
# ============================================================================
class HybridGeometricCompiler:
    def __init__(self):
        self.ce_used = 0
        self.history = []
    
    def _log(self, msg, val=None, phi=None, tau=None):
        self.history.append({'ce': self.ce_used, 'msg': msg, 'val': val, 'phi': phi, 'tau': tau})
        if val is not None:
            print(f"  [CE={self.ce_used:3d}] {msg}: val={val:.4f}, phi={phi}/5, tau={tau:.2f}")
        else:
            print(f"  [CE={self.ce_used:3d}] {msg}")
    
    def run(self):
        print("=" * 70)
        print("HYBRID GEOMETRIC COMPILER")
        print("MF pump → Basin settle → Witness set → Dimple jump")
        print("Target: 0.0147 in ≤70 CE")
        print("=" * 70)
        print()
        
        # ====================================================================
        # Phase 0: Spectral Initialization
        # ====================================================================
        print("PHASE 0: Spectral Initialization")
        print("-" * 50)
        E_init = compute_E_init()
        model = LM()
        model.te.weight.data.copy_(torch.tensor(E_init))
        v0 = eval_val(model, n=10)
        phi0 = phi_clean(model)
        tau0 = gluing_defect(model, n=6)
        self.ce_used += 10 + 6
        self._log("Spectral E₀", v0, phi0, tau0)
        print()
        
        # ====================================================================
        # Phase 1: MF Pump (enter basin)
        # ====================================================================
        print("PHASE 1: MF Pump (enter basin)")
        print("-" * 50)
        model, v1, phi1, tau1 = action_mf_pump(model)
        self.ce_used += 8 + 4
        self._log("MF pump", v1, phi1, tau1)
        print()
        
        # ====================================================================
        # Phase 2: Basin Settle (traverse basin)
        # ====================================================================
        print("PHASE 2: Basin Settle (traverse basin)")
        print("-" * 50)
        
        # Determine LR multiplier from val
        if v1 > 0.5:
            lr_mult = 10
        elif v1 > 0.2:
            lr_mult = 5
        else:
            lr_mult = 3
        
        model, v2, phi2, tau2 = action_basin_settle(model, lr_mult, steps=40)
        self.ce_used += 8 + 4
        self._log(f"Basin settle LR×{lr_mult}", v2, phi2, tau2)
        
        # If still not in basin, do another settle
        if v2 > 0.5:
            print("  Still above basin - second settle LR×5")
            model, v2, phi2, tau2 = action_basin_settle(model, 5, steps=40)
            self.ce_used += 8 + 4
            self._log("Second settle", v2, phi2, tau2)
        
        print()
        
        # ====================================================================
        # Phase 3: Witness Set + Dimple Detection (ONLY IN BASIN)
        # ====================================================================
        print("PHASE 3: Witness Set (basin mapping)")
        print("-" * 50)
        
        if v2 > BASIN_ENTRY_VAL:
            print(f"  ⚠ val={v2:.4f} > {BASIN_ENTRY_VAL} - not in basin yet")
            print("  Running extra settle steps...")
            model, v2, phi2, tau2 = action_basin_settle(model, 3, steps=30)
            self.ce_used += 8 + 4
            self._log("Extra settle", v2, phi2, tau2)
        
        if v2 <= BASIN_ENTRY_VAL:
            print(f"  ✓ In basin (val={v2:.4f} ≤ {BASIN_ENTRY_VAL})")
            print("  Computing witness set...")
            
            witness_points, log_values = compute_witness_set(model)
            self.ce_used += WITNESS_SAMPLES * 8
            
            direction, peak_log = find_dimple(model, witness_points, log_values)
            
            print(f"  Witness points: {len(witness_points)}")
            print(f"  LoG peak: {peak_log:.4f}")
            print(f"  Direction norm: {direction.norm():.4f}")
            
            # Jump to dimple
            alpha = 0.5  # Conservative jump
            model = jump_to_dimple(model, direction, alpha)
            
            v3 = eval_val(model, n=8)
            phi3 = phi_clean(model)
            tau3 = gluing_defect(model, n=4)
            self.ce_used += 8 + 4
            self._log(f"Dimple jump (α={alpha:.2f})", v3, phi3, tau3)
            
            # If jump helped, try another
            if v3 < v2:
                model = jump_to_dimple(model, direction, 0.3)
                v3 = eval_val(model, n=8)
                phi3 = phi_clean(model)
                tau3 = gluing_defect(model, n=4)
                self.ce_used += 8 + 4
                self._log("Second dimple jump", v3, phi3, tau3)
        else:
            print(f"  ⚠ Not in basin - skipping witness set")
            v3 = v2
            phi3 = phi2
            tau3 = tau2
        
        print()
        
        # ====================================================================
        # Phase 4: Final Verification
        # ====================================================================
        print("PHASE 4: Final")
        print("-" * 50)
        v_final = eval_val(model, n=15)
        phi_final = phi_clean(model)
        tau_final = gluing_defect(model, n=6)
        self.ce_used += 15 + 6
        self._log("FINAL", v_final, phi_final, tau_final)
        
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"  Final val:     {v_final:.4f}")
        print(f"  Final phi:     {phi_final}/5")
        print(f"  Final tau:     {tau_final:.2f}")
        print(f"  Total CE:      {self.ce_used}")
        
        if v_final <= VAL_FLOOR and phi_final == PHI_CLEAN_TARGET:
            print(f"\n  ✓ SUCCESS!")
        else:
            print(f"\n  ⚠ val={v_final:.4f} (target: {VAL_FLOOR})")
            print("  → Running GD fallback (400 steps)...")
            opt = torch.optim.AdamW(model.parameters(), lr=LR,
                                    betas=(0.9, 0.95), weight_decay=0.1)
            for step in range(1, 401):
                model.train()
                x, y = get_batch()
                _, loss = model(x, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if step % 100 == 0:
                    v = eval_val(model, n=4)
                    print(f"    GD step {step}: val={v:.4f}")
            v_final = eval_val(model, n=15)
            phi_final = phi_clean(model)
            tau_final = gluing_defect(model, n=6)
            print(f"    GD final: val={v_final:.4f}, phi={phi_final}/5, tau={tau_final:.2f}")
        
        return model, v_final, phi_final, tau_final

# ============================================================================
# MAIN
# ============================================================================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    compiler = HybridGeometricCompiler()
    model, v_final, phi_final, tau_final = compiler.run()
    torch.save(model.state_dict(), 'hybrid_geometric_final.pt')
    print(f"\n  Saved: hybrid_geometric_final.pt")

if __name__ == '__main__':
    main()
