#!/usr/bin/env python3
"""
Basin Witness Compiler
======================
1. MF pump enters basin (always run, no skip)
2. Witness set maps the basin floor (read-only probes)
3. Direct jump to floor (5-6 steps instead of 200)

Target: 0.0147 in ≤60 CE
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

# Basin thresholds
BASIN_VAL_THRESHOLD = 0.5
BASIN_PHI_THRESHOLD = 4

# Witness set in basin
WITNESS_LRS = [0.3, 0.1, 0.03, 0.01, 0.003]
WITNESS_STEPS = 3
FLOOR_PROBE_STEPS = 5

# ============================================================================
# DATA LOADING & MODEL
# ============================================================================
for f in ['/tmp/train_ids.json', '/tmp/val_ids.json', '/tmp/vocab.json']:
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} missing. Run: python build_corpus.py")

with open('/tmp/train_ids.json') as f:
    train_ids = list(map(int, json.load(f)))
with open('/tmp/val_ids.json') as f:
    val_ids = list(map(int, json.load(f)))
with open('/tmp/vocab.json') as f:
    _v = json.load(f)
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
# MF PUMP - ALWAYS RUN (NO SKIP)
# ============================================================================
def action_mf_pump(model):
    """MF pump - always run. This enters the basin."""
    v0 = eval_val(model, n=6)
    phi0 = phi_clean(model)
    tau0 = gluing_defect(model, n=4)
    
    print(f"  MF pump: starting val={v0:.4f}, phi={phi0}/5, tau={tau0:.2f}")
    print(f"  ALWAYS RUN - this enters the basin")
    
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
        
        for i in range(N_SUB // 2):
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
        
        emb_grad /= (N_SUB // 2)
        emb_fish /= (N_SUB // 2)
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
        
        for i in range(N_SUB // 2):
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
        
        wk_grad /= (N_SUB // 2)
        wk_fish /= (N_SUB // 2)
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
        
        # Track best
        if v_mf < best_v:
            best_v = v_mf
            best_model = copy.deepcopy(model)
        
        # Stop if orbit established and in basin
        if phi_mf >= 4 and v_mf < 0.5:
            print(f"    ✓ In basin! phi={phi_mf}/5, val={v_mf:.4f}")
            break
    
    v_final = eval_val(model, n=8)
    phi_final = phi_clean(model)
    tau_final = gluing_defect(model, n=4)
    
    return model, v_final, phi_final, tau_final

# ============================================================================
# BASIN WITNESS SET (READ-ONLY)
# ============================================================================
class BasinProbe:
    def __init__(self, lr, steps, val_start, val_end, phi, tau, is_stable):
        self.lr = lr
        self.steps = steps
        self.val_start = val_start
        self.val_end = val_end
        self.phi = phi
        self.tau = tau
        self.is_stable = is_stable
        self.delta = val_end - val_start
        self.improvement = val_start - val_end  # Positive = improvement
    
    def __repr__(self):
        status = "✓" if self.is_stable else "✗"
        imp = f"↓{self.improvement:.4f}" if self.improvement > 0 else f"↑{-self.improvement:.4f}"
        return f"LR={self.lr:.4f}: {self.val_start:.4f}→{self.val_end:.4f} ({imp}) {status}"

def probe_basin(model, lr, steps=3):
    """Read-only probe in the basin."""
    val_start = eval_val(model, n=4)
    
    model_copy = copy.deepcopy(model)
    opt = torch.optim.AdamW(model_copy.parameters(), lr=lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    
    for step in range(steps):
        model_copy.train()
        x, y = get_batch()
        _, loss = model_copy(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_copy.parameters(), 1.0)
        opt.step()
    
    val_end = eval_val(model_copy, n=4)
    phi = phi_clean(model_copy)
    tau = gluing_defect(model_copy, n=4)
    
    is_stable = val_end < val_start * 2.0 and val_end < 1.0
    
    return BasinProbe(lr, steps, val_start, val_end, phi, tau, is_stable)

def find_floor_probe(model, lrs=WITNESS_LRS):
    """
    Find the best floor probe in the basin.
    Returns the probe with lowest val_end.
    """
    print("  Witness set (basin probes):")
    probes = []
    for lr in lrs:
        probe = probe_basin(model, lr)
        probes.append(probe)
        print(f"    {probe}")
    
    stable = [p for p in probes if p.is_stable]
    if not stable:
        print("  ⚠ No stable probes!")
        return None
    
    best = min(stable, key=lambda p: p.val_end)
    print(f"\n  Best probe: LR={best.lr:.4f}, val={best.val_end:.4f}")
    print(f"    Improvement: {best.improvement:.4f}")
    
    return best

# ============================================================================
# DIRECT JUMP TO FLOOR
# ============================================================================
def jump_to_floor(model, best_probe, steps=FLOOR_PROBE_STEPS):
    """
    Jump directly to the floor using the best probe's LR.
    """
    print(f"\n  Jumping to floor with LR={best_probe.lr:.4f} ({steps} steps)")
    
    opt = torch.optim.AdamW(model.parameters(), lr=best_probe.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    
    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        v = eval_val(model, n=4)
        print(f"    Step {step}: val={v:.4f}")
    
    return model, eval_val(model, n=8), phi_clean(model), gluing_defect(model, n=4)

# ============================================================================
# MAIN COMPILER
# ============================================================================
class BasinWitnessCompiler:
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
        print("BASIN WITNESS COMPILER")
        print("1. MF pump enters basin (ALWAYS RUN)")
        print("2. Witness set maps basin floor")
        print("3. Direct jump to floor (5-6 steps)")
        print("Target: 0.0147 in ≤60 CE")
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
        # Phase 1: MF Pump (ALWAYS RUN)
        # ====================================================================
        print("PHASE 1: MF Pump (Enter Basin)")
        print("-" * 50)
        model, v1, phi1, tau1 = action_mf_pump(model)
        self.ce_used += 8 + 4
        self._log("MF pump", v1, phi1, tau1)
        
        # If not in basin after MF pump, run another round
        if v1 > 0.5 or phi1 < 4:
            print("  Running extra MF pump round...")
            model, v1, phi1, tau1 = action_mf_pump(model)
            self.ce_used += 8 + 4
            self._log("MF pump 2", v1, phi1, tau1)
        
        print()
        
        # ====================================================================
        # Phase 2: Verify we're in the basin
        # ====================================================================
        print("PHASE 2: Basin Verification")
        print("-" * 50)
        print(f"  val={v1:.4f}, phi={phi1}/5, tau={tau1:.2f}")
        
        if v1 <= BASIN_VAL_THRESHOLD and phi1 >= BASIN_PHI_THRESHOLD:
            print("  ✓ IN BASIN")
        else:
            print("  ⚠ Not in basin yet. Running additional MF pump...")
            model, v1, phi1, tau1 = action_mf_pump(model)
            self.ce_used += 8 + 4
            self._log("MF pump 3", v1, phi1, tau1)
        
        print()
        
        # ====================================================================
        # Phase 3: Witness Set in Basin (Find Floor)
        # ====================================================================
        print("PHASE 3: Witness Set (Find Floor)")
        print("-" * 50)
        print(f"  Current state: val={v1:.4f}, phi={phi1}/5")
        
        best_probe = find_floor_probe(model)
        self.ce_used += len(WITNESS_LRS) * (WITNESS_STEPS * 4 + 4 + 4)
        
        print()
        
        # ====================================================================
        # Phase 4: Direct Jump to Floor
        # ====================================================================
        print("PHASE 4: Direct Jump to Floor")
        print("-" * 50)
        
        if best_probe is not None:
            model, v2, phi2, tau2 = jump_to_floor(model, best_probe)
            self.ce_used += FLOOR_PROBE_STEPS * 4 + 8 + 4
            self._log("Jump to floor", v2, phi2, tau2)
        else:
            print("  No stable probe found - running GD fallback")
            opt = torch.optim.AdamW(model.parameters(), lr=0.003,
                                    betas=(0.9, 0.95), weight_decay=0.1)
            for step in range(1, 201):
                model.train()
                x, y = get_batch()
                _, loss = model(x, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if step % 50 == 0:
                    v = eval_val(model, n=4)
                    self.ce_used += 4
                    print(f"    GD step {step}: val={v:.4f}")
            v2 = eval_val(model, n=8)
            phi2 = phi_clean(model)
            tau2 = gluing_defect(model, n=4)
            self.ce_used += 8 + 4
            self._log("GD fallback", v2, phi2, tau2)
        
        print()
        
        # ====================================================================
        # Phase 5: Final
        # ====================================================================
        print("PHASE 5: Final")
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
        
        if best_probe is not None:
            print(f"  Best probe LR: {best_probe.lr:.4f}")
            print(f"  Probe val:     {best_probe.val_end:.4f}")
        
        if v_final <= VAL_FLOOR and phi_final == PHI_CLEAN_TARGET:
            print(f"\n  ✓ SUCCESS!")
        else:
            print(f"\n  ⚠ val={v_final:.4f} (target: {VAL_FLOOR})")
        
        return model, v_final, phi_final, tau_final

# ============================================================================
# MAIN
# ============================================================================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    compiler = BasinWitnessCompiler()
    model, v_final, phi_final, tau_final = compiler.run()
    
    torch.save(model.state_dict(), 'basin_witness_compiler.pt')
    print(f"\n  Saved: basin_witness_compiler.pt")

if __name__ == '__main__':
    main()
