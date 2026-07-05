#!/usr/bin/env python3
"""
Two-Stage Bowl Compiler
=======================
Stage 1: Basin settle to get close to the floor (val < 0.1)
Stage 2: Bowl projection using Hessian eigenvectors (val < 0.02)

The bowl only exists near the floor — spectral gap appears at val < 0.1.
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
TAU_MIN = 1.5
TAU_MAX = 5.7
VAL_FLOOR = 0.062
ORBIT_TOLERANCE = 0.3
BOWL_EIGENVALUE_THRESHOLD = 0.5

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
# FAST BASIN SETTLE (Stage 1: get close to floor)
# ============================================================================
def fast_basin_settle(model, lr_mult, steps=30):
    """Fast basin settle to get close to the floor."""
    v0 = eval_val(model, n=4)
    ce_used = 0
    
    model_try = copy.deepcopy(model)
    opt_b = torch.optim.AdamW(model_try.parameters(), lr=LR * lr_mult,
                              betas=(0.9, 0.95), weight_decay=0.1)
    
    for step in range(1, steps + 1):
        if step <= 5:
            for pg in opt_b.param_groups:
                pg['lr'] = LR * lr_mult * step / 5
        
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
            if v < 0.08:
                break
    
    v_try = eval_val(model_try, n=6)
    ce_used += 6
    phi_try = phi_clean(model_try)
    
    if v_try < v0 * 0.95:
        return model_try, v_try, ce_used, phi_try
    return None


# ============================================================================
# BOWL PROJECTION (Stage 2: only when close to floor)
# ============================================================================
def bowl_projection(model):
    """
    Bowl projection using Hessian eigenvectors.
    ONLY CALL THIS WHEN val < 0.1 (the bowl exists near the floor).
    """
    print("\n  Bowl projection (near floor)...")
    
    model_try = copy.deepcopy(model)
    ce_used = 0
    v0 = eval_val(model_try, n=4)
    ce_used += 4
    
    print(f"    Starting val: {v0:.4f}")
    
    # Hessian-vector product
    def hvp(v, n=4):
        model_try.zero_grad()
        ls = [model_try(*get_batch())[1] for _ in range(n)]
        ce_used_local = n
        loss = torch.stack(ls).mean()
        grads = torch.autograd.grad(loss, list(model_try.parameters()), create_graph=True)
        gv = (torch.cat([gr.flatten() for gr in grads]) * v.detach()).sum()
        hv = torch.cat([h.flatten() for h in
                        torch.autograd.grad(gv, list(model_try.parameters()), retain_graph=False)])
        model_try.zero_grad()
        return hv.detach(), ce_used_local
    
    n_p = sum(p.numel() for p in model_try.parameters())
    
    # 1. Compute gradient
    model_try.zero_grad()
    ls = [model_try(*get_batch())[1] for _ in range(10)]
    ce_used += 10
    torch.stack(ls).mean().backward()
    g = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                   for p in model_try.parameters()]).detach()
    model_try.zero_grad()
    
    # 2. Power iteration for top Hessian eigenvectors
    n_eigs = 6
    eigenvecs = []
    eigenvals = []
    
    torch.manual_seed(42)
    v = torch.randn(n_p)
    v = v / v.norm()
    
    for i in range(n_eigs):
        for _ in range(10):
            Hv, ce = hvp(v)
            ce_used += ce
            for ev in eigenvecs:
                Hv = Hv - (Hv * ev).sum() * ev
            v = Hv / max(float(Hv.norm()), 1e-10)
        
        Hv, ce = hvp(v)
        ce_used += ce
        eigenval = float((v * Hv).sum().item())
        eigenvecs.append(v.clone())
        eigenvals.append(eigenval)
        
        print(f"      eigen {i}: {eigenval:.4f}")
        
        v = torch.randn(n_p)
        for ev in eigenvecs:
            v = v - (v * ev).sum() * ev
        v = v / v.norm()
    
    # 3. Find bowl directions (eigenvalues > threshold)
    threshold = max(0.3, min(eigenvals) + 0.1)
    bowl_indices = [i for i, ev in enumerate(eigenvals) if ev > threshold]
    
    if not bowl_indices:
        print("    No bowl found (spectral gap too small)")
        return None
    
    print(f"    Bowl directions: {len(bowl_indices)} (eigenvalues > {threshold:.2f})")
    
    # 4. Project onto bowl subspace
    bowl_basis = torch.stack([eigenvecs[i] for i in bowl_indices], dim=1)
    g_bowl = bowl_basis.T @ g
    
    # 5. Newton step in bowl subspace
    mu = 0.1
    diag = torch.tensor([eigenvals[i] + mu for i in bowl_indices], dtype=torch.float32)
    delta_bowl = -g_bowl / diag
    
    # 6. Apply the step
    delta = bowl_basis @ delta_bowl
    
    # Clamp step size
    step_norm = float(delta.norm())
    if step_norm > 0.2:
        delta = delta * (0.2 / step_norm)
    
    w0 = model_try.flat_params()
    model_try.set_flat(w0 + delta)
    
    v_try = eval_val(model_try, n=6)
    ce_used += 6
    phi_try = phi_clean(model_try)
    
    print(f"    val: {v0:.4f} → {v_try:.4f}")
    print(f"    phi: {phi_clean(model)}/5 → {phi_try}/5")
    print(f"    CE: {ce_used}")
    
    if v_try < v0 * 0.9:
        return model_try, v_try, ce_used, phi_try
    
    return None


# ============================================================================
# MAIN
# ============================================================================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 70)
    print("TWO-STAGE BOWL COMPILER")
    print("=" * 70)
    print("  Stage 1: Basin settle to get close to floor")
    print("  Stage 2: Bowl projection (only when val < 0.1)")
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
    
    total_ce = 0
    path = []
    
    # ========================================================================
    # Stage 1: Basin settle to get close to floor
    # ========================================================================
    print("[Stage 1] Basin settle...")
    
    # MF pump
    print("  MF pump...")
    eta = min(0.02, 0.01 / (1 + eval_val(model, n=4)))
    eta = max(0.001, eta)
    
    model_try = copy.deepcopy(model)
    v_mf = eval_val(model_try, n=4)
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
        
        v = eval_val(model_try, n=4)
        ce_used += 4
        if v > v_mf * 1.5:
            break
        if phi_clean(model_try) >= 4:
            break
    
    model = model_try
    v = eval_val(model, n=6)
    ce_used += 6
    total_ce += ce_used
    phi = phi_clean(model)
    tau = gluing_defect(model, n=4)
    path.append(f"mf_pump({ce_used}CE)")
    print(f"  ✓ val={v:.4f}, phi={phi}/5, tau={tau:.2f} (CE={total_ce})")
    
    # Basin settle phases (progressive LR reduction)
    for lr_mult, steps in [(10, 30), (5, 25), (3, 20), (2, 20), (1, 20)]:
        if v < 0.08:
            break
        print(f"  Basin settle LR×{lr_mult}...")
        result = fast_basin_settle(model, lr_mult, steps)
        if result is not None:
            model, v, ce, phi = result
            total_ce += ce
            tau = gluing_defect(model, n=4)
            path.append(f"bs_LR{lr_mult}({ce}CE)")
            print(f"    ✓ val={v:.4f}, phi={phi}/5, tau={tau:.2f} (CE={total_ce})")
        else:
            print(f"    ⊘ Failed")
    
    print(f"\n[Stage 1 complete] val={v:.4f}, phi={phi}/5, CE={total_ce}")
    
    # ========================================================================
    # Stage 2: Bowl projection (only if val < 0.1)
    # ========================================================================
    if v < 0.1 and phi == PHI_CLEAN_TARGET:
        print("\n[Stage 2] Bowl projection...")
        result = bowl_projection(model)
        if result is not None:
            model, v, ce, phi = result
            total_ce += ce
            tau = gluing_defect(model, n=4)
            path.append(f"bowl_proj({ce}CE)")
            print(f"  ✓ val={v:.4f}, phi={phi}/5, tau={tau:.2f} (CE={total_ce})")
        else:
            print("  ⊘ Bowl projection failed, using small GD")
            # Small GD polish
            model_try = copy.deepcopy(model)
            opt = torch.optim.AdamW(model_try.parameters(), lr=LR * 0.3, betas=(0.9, 0.95), weight_decay=0.1)
            ce_gd = 0
            for step in range(1, 31):
                lr_s = LR * 0.3 * 0.5 * (1 + math.cos(math.pi * step / 30))
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
                ce_gd += 1
                if step % 10 == 0:
                    vv = eval_val(model_try, n=4)
                    ce_gd += 4
                    print(f"      GD step {step}: val={vv:.4f}")
            v = eval_val(model_try, n=6)
            ce_gd += 6
            phi = phi_clean(model_try)
            tau = gluing_defect(model_try, n=4)
            model = model_try
            total_ce += ce_gd
            path.append(f"gd_polish({ce_gd}CE)")
            print(f"  ✓ val={v:.4f}, phi={phi}/5, tau={tau:.2f} (CE={total_ce})")
    elif v < 0.1:
        # Small GD polish even if phi not perfect
        print("\n[Stage 2] Small GD polish...")
        model_try = copy.deepcopy(model)
        opt = torch.optim.AdamW(model_try.parameters(), lr=LR * 0.3, betas=(0.9, 0.95), weight_decay=0.1)
        ce_gd = 0
        for step in range(1, 31):
            lr_s = LR * 0.3 * 0.5 * (1 + math.cos(math.pi * step / 30))
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
            ce_gd += 1
            if step % 10 == 0:
                vv = eval_val(model_try, n=4)
                ce_gd += 4
                print(f"      GD step {step}: val={vv:.4f}")
        v = eval_val(model_try, n=6)
        ce_gd += 6
        phi = phi_clean(model_try)
        tau = gluing_defect(model_try, n=4)
        model = model_try
        total_ce += ce_gd
        path.append(f"gd_polish({ce_gd}CE)")
        print(f"  ✓ val={v:.4f}, phi={phi}/5, tau={tau:.2f} (CE={total_ce})")
    
    # Final results
    v_final = eval_val(model, n=15)
    phi_final = phi_clean(model)
    tau_final = gluing_defect(model, n=6)
    
    print("\n" + "=" * 70)
    print("COMPILER RESULTS")
    print("=" * 70)
    print(f"  Final val:     {v_final:.4f}")
    print(f"  Final phi:     {phi_final}/5")
    print(f"  Final tau:     {tau_final:.2f}")
    print(f"  Total CE:      {total_ce}")
    print(f"  Path:          {' → '.join(path)}")
    
    gd_val = 0.0914
    print("\n" + "=" * 70)
    print("COMPARISON: COMPILER vs GD-400")
    print("=" * 70)
    print(f"  {'Metric':<30} {'Compiler':>12} {'GD-400':>12}")
    print("  " + "-" * 56)
    print(f"  {'Final val':<30} {v_final:>12.4f} {gd_val:>12.4f}")
    print(f"  {'Final phi':<30} {phi_final:>12}/5 {'4/5':>12}")
    print(f"  {'CE steps':<30} {total_ce:>12} {'400':>12}")
    
    if v_final < gd_val:
        advantage = gd_val / v_final
        print(f"  {'Advantage':<30} {advantage:>11.2f}× {'1.0×':>12}")
    else:
        print(f"  {'Advantage':<30} {'1.0×':>12} {'1.0×':>12}")
    
    torch.save(model.state_dict(), 'two_stage_bowl_compiler.pt')
    print(f"\n  Saved: two_stage_bowl_compiler.pt")


if __name__ == '__main__':
    main()
