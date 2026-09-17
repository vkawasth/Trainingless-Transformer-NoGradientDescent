#!/usr/bin/env python3
"""
Integrated Geometry-Driven Compiler (final.py)
==============================================
Combines c_prune baseline, Spectral H1 homology witness, and Dually Flat Bregman geometry.
"""

import json, math, warnings, collections, os, sys, time, copy
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

# ── CONFIGURATION & HYPERPARAMETERS ──────────────────────────
D = 256; N_HEADS = 4; N_STU = 6; BATCH = 8; SEQ = 64; LR = 3e-4
ETA_MF = 0.002; N_SUB = 200
PHI_CLEAN_TARGET = 5; TAU_MIN = 1.5; TAU_MAX = 5.7

# Verify corpus files
for f in ['/tmp/train_ids.json', '/tmp/val_ids.json', '/tmp/vocab.json', '/tmp/corpus_meta.json']:
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} missing. Run: python build_corpus2.py --out /tmp --mode decorr")

with open('/tmp/train_ids.json')   as f: train_ids = list(map(int, json.load(f)))
with open('/tmp/val_ids.json')     as f: val_ids   = list(map(int, json.load(f)))
with open('/tmp/vocab.json')       as f: _v = json.load(f)
with open('/tmp/corpus_meta.json') as f: corpus_meta = json.load(f)

VOCAB = len(_v) if isinstance(_v, list) else len(_v)
H_BIGRAM = corpus_meta.get("H_bigram", 0.0147)
FLOOR_TARGET_VAL = H_BIGRAM

train_t = torch.tensor(train_ids, dtype=torch.long)
val_t   = torch.tensor(val_ids,   dtype=torch.long)

# ── MODEL DEFINITIONS WITH DYNAMIC ATTENTION FEEDBACK ─────────
class Attn(nn.Module):
    def __init__(self):
        super().__init__(); dh = D // N_HEADS
        self.WQ = nn.Linear(D, D, bias=False); self.WK = nn.Linear(D, D, bias=False)
        self.WV = nn.Linear(D, D, bias=False); self.op = nn.Linear(D, D, bias=False)
        self.ln = nn.LayerNorm(D); self.sc = math.sqrt(dh); self.nh = N_HEADS; self.dh = dh
        for w in [self.WQ, self.WK, self.WV, self.op]: nn.init.normal_(w.weight, std=0.02)
        self.beta = 1.0  # Dynamic Bregman temperature feedback

    def forward(self, h):
        B, S, _ = h.shape
        Q = self.WQ(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        K = self.WK(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        V = self.WV(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        # Apply Bregman temperature beta scaling to attention scores
        sc = (Q @ K.transpose(-2, -1)) / (self.sc * max(self.beta, 1e-4))
        mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(h.device)
        sc = sc.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        return self.ln(h + self.op((F.softmax(sc, dim=-1) @ V).transpose(1, 2).reshape(B, S, D)))

class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Linear(D, D*2, bias=False); self.v = nn.Linear(D, D*2, bias=False)
        self.o = nn.Linear(D*2, D, bias=False); self.n = nn.LayerNorm(D)
        for w in [self.g, self.v, self.o]: nn.init.normal_(w.weight, std=0.02)
    def forward(self, h): return self.n(h + self.o(F.silu(self.g(h)) * self.v(h)))

class Block(nn.Module):
    def __init__(self): super().__init__(); self.attn = Attn(); self.ff = FF()
    def forward(self, h): return self.ff(self.attn(h))

class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = nn.Embedding(VOCAB, D); self.pe = nn.Embedding(512, D)
        self.blocks = nn.ModuleList([Block() for _ in range(N_STU)])
        self.ln_f = nn.LayerNorm(D); self.head = nn.Linear(D, VOCAB, bias=False)
        self.head.weight = self.te.weight
        nn.init.normal_(self.te.weight, std=0.02); nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x, y=None):
        h = self.te(x) + self.pe(torch.arange(x.shape[1], device=x.device))
        for b in self.blocks: h = b(h)
        logits = self.head(self.ln_f(h))
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1)) if y is not None else None
        return logits, loss

    def flat_params(self):
        return torch.cat([p.data.flatten() for p in self.parameters() if p.requires_grad])

    def set_flat(self, v):
        i = 0
        for p in self.parameters():
            if p.requires_grad:
                n = p.numel()
                p.data.copy_(v[i:i+n].reshape(p.shape))
                i += n

def get_batch(split='train'):
    data = val_t if split == 'val' else train_t
    ix = torch.randint(0, len(data) - SEQ - 1, (BATCH,))
    return (torch.stack([data[i:i+SEQ] for i in ix]),
            torch.stack([data[i+1:i+SEQ+1] for i in ix]))

def eval_val(m, n=15):
    m.eval(); ls = []
    with torch.no_grad():
        for _ in range(n): x, y = get_batch('val'); _, l = m(x, y); ls.append(l.item())
    return float(np.mean(ls))

# ── HOMOLOGY & DYNAMIC SUBSPACE ENGINES ───────────────────────
class NativeHomologyWitnessEngine:
    """Computes exact 1st Betti numbers (b1) across a landmark point cloud."""
    def __init__(self, num_landmarks=16):
        self.num_landmarks = num_landmarks

    def compute_b1(self, point_cloud: torch.Tensor, eps: float = 0.5) -> int:
        pts = point_cloud.detach().cpu().numpy()
        N = pts.shape[0]
        if N < 4: return 0
        
        # Distance matrix
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        edges = [(i, j) for i in range(N) for j in range(i+1, N) if dists[i, j] <= eps]
        
        n_vertices = N
        n_edges = len(edges)
        if n_edges == 0: return 0

        # Boundary operator d1
        d1 = np.zeros((n_vertices, n_edges))
        for idx, (u, v) in enumerate(edges):
            d1[u, idx] = -1.0
            d1[v, idx] = 1.0

        # Build 2-simplices (triangles)
        triangles = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    if dists[i,j] <= eps and dists[j,k] <= eps and dists[i,k] <= eps:
                        triangles.append((i, j, k))
        
        n_triangles = len(triangles)
        if n_triangles == 0:
            rank_d1 = np.linalg.matrix_rank(d1)
            null_d1 = n_edges - rank_d1
            return max(0, null_d1)

        # Boundary operator d2
        d2 = np.zeros((n_edges, n_triangles))
        edge_map = {edge: idx for idx, edge in enumerate(edges)}
        for t_idx, (u, v, w) in enumerate(triangles):
            e1 = edge_map.get((u, v)) if (u, v) in edge_map else edge_map.get((v, u))
            e2 = edge_map.get((v, w)) if (v, w) in edge_map else edge_map.get((w, v))
            e3 = edge_map.get((u, w)) if (u, w) in edge_map else edge_map.get((w, u))
            d2[e1, t_idx] = 1.0; d2[e2, t_idx] = 1.0; d2[e3, t_idx] = -1.0

        rank_d1 = np.linalg.matrix_rank(d1)
        rank_d2 = np.linalg.matrix_rank(d2)
        dim_ker_d1 = n_edges - rank_d1
        b1 = dim_ker_d1 - rank_d2
        return max(0, int(b1))

class IntegratedGeometryEngine:
    """Manages dynamic projection matrix resizing and rolling N-point cloud buffer."""
    def __init__(self, model: nn.Module, proj_dim: int = 128, max_points: int = 16):
        self.model = model
        self.proj_dim = proj_dim
        self.max_points = max_points
        self.point_cloud_buffer = []
        self.proj_matrix = None
        self.homology_engine = NativeHomologyWitnessEngine(num_landmarks=max_points)

    def _ensure_projection_matrix(self, active_dim: int, device: torch.device):
        if self.proj_matrix is None or self.proj_matrix.shape[0] != active_dim:
            g = torch.Generator(device=device).manual_seed(42)
            self.proj_matrix = torch.randn(active_dim, self.proj_dim, generator=g, device=device) / (self.proj_dim ** 0.5)

    def update_and_evaluate_b1(self) -> int:
        active_params = torch.cat([p.data.flatten() for p in self.model.parameters() if p.requires_grad])
        self._ensure_projection_matrix(active_params.numel(), active_params.device)
        
        proj_state = torch.matmul(active_params, self.proj_matrix)
        self.point_cloud_buffer.append(proj_state.detach())
        if len(self.point_cloud_buffer) > self.max_points:
            self.point_cloud_buffer.pop(0)
            
        pts = torch.stack(self.point_cloud_buffer)
        return self.homology_engine.compute_b1(pts)

# ── HELPER METRICS & COMPRESSION OPTIMIZER ──────────────────
def sheet_angles(model):
    out = []; WKs = [model.blocks[l].attn.WK.weight.data.float() for l in range(N_STU)]
    for l in range(N_STU - 1):
        try:
            phi = WKs[l+1] @ torch.linalg.pinv(WKs[l])
            lam = torch.linalg.eigvals(phi); lam1 = lam[lam.abs().argmax()]
            a = float(torch.angle(lam1))
            out.append('π' if abs(abs(a) - math.pi) < 0.3 else '0' if abs(a) < 0.3 else f'{a:.2f}')
        except: out.append('?')
    return out

def phi_clean(model): return sum(1 for p in sheet_angles(model) if p in ('0', 'π'))

def gluing_defect(model, n=8):
    model.zero_grad()
    ls = [model(*get_batch())[1] for _ in range(n)]
    torch.stack(ls).mean().backward()
    g_ff = sum(p.grad.data.norm().item() for nm, p in model.named_parameters() if '.ff.' in nm and p.grad is not None)
    g_emb = model.te.weight.grad.data.norm().item() if model.te.weight.grad is not None else 1e-8
    model.zero_grad()
    return g_ff / max(g_emb, 1e-8)

def compute_amari_bregman(logits):
    P = F.softmax(logits, dim=-1).clamp_min(1e-12)
    eta = torch.log(P) + 1.0
    psi = (P * torch.log(P)).sum(dim=-1).mean()
    psi_star = (P * eta - P * torch.log(P)).sum(dim=-1).mean()
    duality_gap = float(torch.abs(psi + psi_star - (P * eta).sum(dim=-1).mean()))
    return P, eta, duality_gap

class CompressedAdam:
    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, names=None):
        pn = [(n, q) for n, q in zip(names, params) if q.requires_grad] if names else [(f"p{i}", q) for i, q in enumerate(params) if q.requires_grad]
        self.names = [n for n, _ in pn]; self.p = [q for _, q in pn]
        self.lr = lr; self.b1, self.b2 = betas; self.eps = eps; self.wd = weight_decay
        self.m = [torch.zeros_like(q) for q in self.p]; self.v = [torch.zeros_like(q) for q in self.p]
        self.t = 0; self._pg = [{"lr": lr}]

    @property
    def param_groups(self): return self._pg

    def zero_grad(self):
        for q in self.p:
            if q.grad is not None: q.grad.zero_()

    @torch.no_grad()
    def step(self):
        lr = self._pg[0]["lr"]; self.t += 1
        b1 = 1 - self.b1**self.t; b2 = 1 - self.b2**self.t
        alpha = lr * math.sqrt(b2) / b1; eps_t = self.eps * math.sqrt(b2)
        for i, q in enumerate(self.p):
            g = q.grad if q.grad is not None else torch.zeros_like(q)
            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            uk = self.m[i] / (self.v[i].sqrt() + eps_t)
            q.data.add_(-alpha * uk - lr * self.wd * q.data)

# ── MAIN COMPILER PIPELINE ────────────────────────────────────
print("="*65)
print("INTEGRATED GEOMETRY-DRIVEN COMPILER (FINAL.PY)")
print(f"Target Floor (H_bigram): {H_BIGRAM:.4f} nats")
print("="*65); print()

# ── STEP 1: NATIVE CORPUS FLOOR & GEOMETRIC ANCHOR SETUP ──────
print("━━━ STEP 1: CORPUS FLOOR & GEOMETRIC ANCHOR SETUP ━━━━━━")
bigram = collections.Counter(); perm = {}
for i in range(len(train_ids) - 1):
    a, b = train_ids[i], train_ids[i+1]
    if a < VOCAB and b < VOCAB: bigram[(a, b)] += 1; perm.setdefault(a, b)

rows, cols, vv = [], [], []
for (a, b), cnt in bigram.items(): rows.append(a); cols.append(b); vv.append(float(cnt))
W_sp = sp.csr_matrix((vv, (rows, cols)), shape=(VOCAB, VOCAB), dtype=np.float32)
W_sp = W_sp + W_sp.T; d_inv = np.array(1.0 / (W_sp.sum(1) + 1e-8)).flatten()
Dsi = sp.diags(np.sqrt(d_inv)); L_sym = sp.eye(VOCAB) - Dsi @ W_sp @ Dsi

evals, evecs = spla.eigsh(L_sym, k=D+1, which='SM', tol=1e-4, maxiter=2000)
idx_s = np.argsort(evals); evecs = evecs[:, idx_s][:, 1:D+1]
E_0 = (evecs / (np.sqrt(evals[idx_s[1:D+1]]) + 1e-8)[np.newaxis, :]).astype(np.float32)
E_0 = (E_0 / (E_0.std() + 1e-8) * 0.02)
E_next = np.array([E_0[perm.get(t, t)] for t in range(VOCAB)], dtype=np.float32)
E_init = (0.9 * E_0 + 0.1 * E_next); E_norm = float(np.linalg.norm(E_0))
E_init = (E_init * (E_norm / max(float(np.linalg.norm(E_init)), 1e-8))).astype(np.float32)

torch.manual_seed(42)
m_floor = LM(); m_floor.te.weight.data.copy_(torch.tensor(E_init))
opt_f = torch.optim.AdamW(m_floor.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
for _ in range(200):
    m_floor.train(); x, y = get_batch(); _, l = m_floor(x, y)
    opt_f.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m_floor.parameters(), 1.0); opt_f.step()

m_floor.zero_grad()
ls = [m_floor(*get_batch())[1] for _ in range(20)]; torch.stack(ls).mean().backward()
g_floor = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                   for p in m_floor.parameters()]).detach(); m_floor.zero_grad()
v_floor = eval_val(m_floor, n=20)
print(f"Floor gradient computed: val={v_floor:.4f}  ||g_floor||={float(g_floor.norm()):.4f}\n")

# ── STEP 2: CURVATURE SADDLE EXIT ────────────────────────────
print("━━━ STEP 2: CURVATURE SADDLE EXIT ━━━━━━━━━━━━━━━━━━━━━━")
torch.manual_seed(99)
model = LM(); model.te.weight.data.copy_(torch.tensor(E_init))

def hvp_s(v, n=8):
    model.zero_grad()
    ls = [model(*get_batch())[1] for _ in range(n)]; loss = torch.stack(ls).mean()
    grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
    gv = (torch.cat([gr.flatten() for gr in grads if gr is not None]) * v.detach()).sum()
    hv = torch.cat([h.flatten() for h in torch.autograd.grad(gv, list(model.parameters()), retain_graph=False)])
    model.zero_grad(); return hv.detach()

n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
torch.manual_seed(42); v = torch.randn(n_p); v = v / v.norm()
for _ in range(15):
    Hv = hvp_s(v); neg = -Hv; v = neg / max(float(neg.norm()), 1e-10)

v_neg = v.clone()
w0 = model.flat_params(); best_v = eval_val(model, n=8); best_a = 0.
for alpha in [0.5, 1.0, 1.429, 2.0, 3.0, 4.0]:
    model.set_flat(w0 + alpha * (v_neg / v_neg.norm())); vt = eval_val(model, n=6)
    if vt < best_v: best_v = vt; best_a = alpha

model.set_flat(w0 + best_a * (v_neg / v_neg.norm()))
print(f"  α*={best_a:.3f}  val={eval_val(model):.4f}  sheet={sheet_angles(model)}\n")

# ── STEP 3: DUALLY FLAT BREGMAN MF PUMP & FEEDBACK ────────────
print("━━━ STEP 3: DUALLY FLAT BREGMAN MF PUMP ━━━━━━━━━━━━━━━━")
for mf_r in range(1, 16):
    x, y = get_batch()
    logits, _ = model(x, y)
    P, eta, duality_gap = compute_amari_bregman(logits)
    
    # Compute Bregman divergence layer temperature scaling
    D_bregman = duality_gap + 1e-4
    for bl in model.blocks:
        bl.attn.beta = float(1.0 / (D_bregman + 1e-4))

    # Mean-field updates
    for l in range(N_STU):
        model.blocks[l].attn.WK.weight.requires_grad_(False)
        model.blocks[l].attn.WQ.weight.requires_grad_(False)
    
    emb_grad = torch.zeros_like(model.te.weight); emb_fish = torch.zeros_like(model.te.weight)
    for i in range(N_SUB // 10):
        model.zero_grad(); _, loss = model(*get_batch()); loss.backward()
        if model.te.weight.grad is not None:
            g = model.te.weight.grad.detach(); emb_grad += g; emb_fish += g**2
            
    delta_E = -(emb_grad / (emb_fish + 1e-4))
    with torch.no_grad(): model.te.weight.add_(ETA_MF * delta_E)
    for l in range(N_STU):
        model.blocks[l].attn.WK.weight.requires_grad_(True)
        model.blocks[l].attn.WQ.weight.requires_grad_(True)

    tau = gluing_defect(model, n=4); pc = phi_clean(model)
    print(f"  MF{mf_r:2d}: val={eval_val(model, n=2):.4f}  Φ_cl={pc}/5  τ={tau:.2f}  gap={duality_gap:.6f}")
    if pc == N_STU - 1:
        print("  ✓ STOP: Φ_clean=5/5 orbit established"); break

# ── STEP 4: ATTENTION PRUNING & DYNAMIC SUBSPACE RESIZING ─────
print("\n━━━ STEP 4: ATTENTION PRUNING & SUBSPACE RESIZING ━━━━━")
pruned_blocks = []
for bi, bl in enumerate(model.blocks):
    model.zero_grad()
    _, l = model(*get_batch()); l.backward()
    gq = bl.attn.WQ.weight.grad; gv = bl.attn.WV.weight.grad
    r = (gq.norm() if gq is not None else 0.0) / max(gv.norm() if gv is not None else 1.0, 1e-12)
    if r < 0.15:
        pruned_blocks.append(bi)
        bl.attn.WQ.weight.requires_grad_(False)
        bl.attn.WK.weight.requires_grad_(False)

geometry_engine = IntegratedGeometryEngine(model, proj_dim=128, max_points=16)
active_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Pruned QK blocks: {pruned_blocks} | Active Parameters: {active_p:,}")

# ── STEP 5: ROLLING N-POINT HOMOLOGY & DUALITY GAP GEO-STOP ───
print("\n━━━ STEP 5: ROLLING N-POINT HOMOLOGY GEO-STOP ━━━━━━━━━━")
opt_b = CompressedAdam(list(model.parameters()), lr=LR*5, weight_decay=0.1,
                       names=[n for n, _ in model.named_parameters()])

for step in range(1, 301):
    model.train(); x, y = get_batch(); logits, l = model(x, y)
    opt_b.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_b.step()

    if step % 8 == 0:
        b1 = geometry_engine.update_and_evaluate_b1()
        _, _, duality_gap = compute_amari_bregman(logits)
        tau = gluing_defect(model, n=4)
        v = eval_val(model, n=4)
        print(f"  step {step:3d}: val={v:.4f}  b1={b1}  τ={tau:.2f}  gap={duality_gap:.6f}")
        
        # Geo-stop condition
        if b1 == 0 and duality_gap < 1e-3 and (1.8 <= tau <= 4.5):
            print(f"  ✓ Geo-Stop triggered at step {step}!"); break

# ── STEP 6: TOPOGATE DISCRETE STRUCTURAL INVERSION ────────────
print("\n━━━ STEP 6: TOPOGATE DISCRETE STRUCTURAL INVERSION ━━━━━")
v_before = eval_val(model, n=8); pc_before = phi_clean(model)
best_score = 0; best_layers = None

for flip_layers in [[1,2], [0,1], [2,3], [0,2], [1,3], [0,3]]:
    with torch.no_grad():
        for l in flip_layers:
            model.blocks[l].attn.WV.weight.data.mul_(-1)
            model.blocks[l].attn.op.weight.data.mul_(-1)
    v_try = eval_val(model, n=4); pc_try = phi_clean(model)
    score = (v_before - v_try) + 0.3 * ((pc_try - pc_before) / 5.0)
    if score > best_score: best_score = score; best_layers = flip_layers
    with torch.no_grad():
        for l in flip_layers:
            model.blocks[l].attn.WV.weight.data.mul_(-1)
            model.blocks[l].attn.op.weight.data.mul_(-1)

if best_layers and best_score > 0:
    with torch.no_grad():
        for l in best_layers:
            model.blocks[l].attn.WV.weight.data.mul_(-1)
            model.blocks[l].attn.op.weight.data.mul_(-1)
    print(f"  ✓ Applied TopoGate flips to blocks {best_layers}")
else:
    print("  ~ TopoGate neutral — skipping flips")

# ── STEP 7: K₀ SPLIT DESCENT & AMARI-PRECONDITIONED LANCZOS ───
print("\n━━━ STEP 7: K₀ SPLIT DESCENT & PRECONDITIONED LANCZOS ━━")
def k0_split(base, steps=25):
    tau_now = gluing_defect(base, n=4)
    w_ff = 3.5 * (1.5 / max(tau_now, 0.5))**1.5
    m1 = copy.deepcopy(base); m2 = copy.deepcopy(base)
    
    # M1: EmbFF
    p1 = [p for n, p in m1.named_parameters() if ('te.' in n or '.ff.' in n) and p.requires_grad]
    opt1 = torch.optim.AdamW(p1, lr=LR)
    for _ in range(steps):
        m1.train(); x, y = get_batch(); _, l = m1(x, y)
        opt1.zero_grad(); l.backward(); opt1.step()

    # M2: Attn
    p2 = [p for n, p in m2.named_parameters() if '.attn.' in n and p.requires_grad]
    opt2 = torch.optim.AdamW(p2, lr=LR)
    for _ in range(steps):
        m2.train(); x, y = get_batch(); _, l = m2(x, y)
        opt2.zero_grad(); l.backward(); opt2.step()

    # Merge
    base_params = {n: p.data.clone() for n, p in base.named_parameters()}
    with torch.no_grad():
        for n, p in base.named_parameters():
            if p.requires_grad:
                d1 = dict(m1.named_parameters())[n].data - base_params[n]
                d2 = dict(m2.named_parameters())[n].data - base_params[n]
                if 'te.' in n or '.ff.' in n: p.data.add_(w_ff * d1)
                else: p.data.add_(d2)

k0_split(model, steps=20)
print(f"  Post K0-Split val: {eval_val(model):.4f}")

# Amari-Preconditioned Krylov Lanczos Terminal Step
def preconditioned_hvp(model, v, eta_metric):
    model.zero_grad()
    ls = [model(*get_batch())[1] for _ in range(4)]; loss = torch.stack(ls).mean()
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], create_graph=True)
    gv = (torch.cat([gr.flatten() for gr in grads if gr is not None]) * v.detach()).sum()
    hv = torch.cat([h.flatten() for h in torch.autograd.grad(gv, [p for p in model.parameters() if p.requires_grad], retain_graph=False)])
    model.zero_grad()
    return hv.detach() * eta_metric

n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
q = torch.randn(n_p, device=train_t.device); q = q / q.norm()
Q = [q]; alphas = []; betas = []
logits, _ = model(*get_batch())
_, eta, _ = compute_amari_bregman(logits)
eta_metric = float(eta.abs().mean())

for j in range(8):
    z = preconditioned_hvp(model, Q[j], eta_metric)
    alpha = float((Q[j] * z).sum()); alphas.append(alpha)
    z = z - alpha * Q[j]
    if j > 0: z = z - betas[-1] * Q[j-1]
    beta = float(z.norm()); betas.append(beta)
    if beta < 1e-8: break
    Q.append(z / beta)

T = torch.zeros(len(alphas), len(alphas))
for i in range(len(alphas)): T[i, i] = alphas[i]
for i in range(len(alphas)-1): T[i, i+1] = betas[i]; T[i+1, i] = betas[i]
T_evals, T_evecs = torch.linalg.eigh(T)
V = torch.stack(Q[:len(alphas)], dim=1) @ T_evecs

model.zero_grad()
ls = [model(*get_batch())[1] for _ in range(10)]; torch.stack(ls).mean().backward()
g = torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad and p.grad is not None])
g_proj = V.T @ g; d_proj = g_proj / (T_evals + 0.950)
d = -(V @ d_proj)

w0 = model.flat_params()
v_pre = eval_val(model)
model.set_flat(w0 + d)
v_post = eval_val(model)

if v_post > v_pre: model.set_flat(w0) # Revert if step didn't decrease loss

print("\n" + "="*65)
print(f"FINAL COMPILER LOSS:   {eval_val(model):.4f} nats")
print(f"BIGRAM ENTROPY FLOOR:  {H_BIGRAM:.4f} nats")
print(f"GAP TO FLOOR:          {eval_val(model) - H_BIGRAM:+.4f} nats")
print("="*65)
