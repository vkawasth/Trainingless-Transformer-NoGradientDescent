#!/usr/bin/env python3
import json, math, warnings, collections, os, sys, time, copy
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import torch, torch.nn as nn, torch.nn.functional as F

D = 256; N_HEADS = 4; N_STU = 6; BATCH = 8; SEQ = 64; LR = 6e-4
ETA_MF_BASE = 0.002; N_SUB = 200
PHI_CLEAN_TARGET = 5; TAU_MIN = 3.0; TAU_MAX = 10.0

for f in ['/tmp/train_ids.json', '/tmp/val_ids.json', '/tmp/vocab.json', '/tmp/corpus_meta.json']:
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} missing. Run building corpus script first.")

with open('/tmp/train_ids.json')   as f: train_ids = list(map(int, json.load(f)))
with open('/tmp/val_ids.json')     as f: val_ids   = list(map(int, json.load(f)))
with open('/tmp/vocab.json')       as f: _v = json.load(f)
with open('/tmp/corpus_meta.json') as f: corpus_meta = json.load(f)

VOCAB = len(_v) if isinstance(_v, list) else len(_v)
H_BIGRAM = corpus_meta.get("H_bigram", 2.2378)

train_t = torch.tensor(train_ids, dtype=torch.long)
val_t   = torch.tensor(val_ids,   dtype=torch.long)

class Attn(nn.Module):
    def __init__(self):
        super().__init__(); dh = D // N_HEADS
        self.WQ = nn.Linear(D, D, bias=False); self.WK = nn.Linear(D, D, bias=False)
        self.WV = nn.Linear(D, D, bias=False); self.op = nn.Linear(D, D, bias=False)
        self.ln = nn.LayerNorm(D); self.sc = math.sqrt(dh); self.nh = N_HEADS; self.dh = dh
        for w in [self.WQ, self.WK, self.WV, self.op]: nn.init.normal_(w.weight, std=0.02)
        self.beta = 1.0

    def forward(self, h):
        B, S, _ = h.shape
        Q = self.WQ(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        K = self.WK(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        V = self.WV(h).view(B, S, self.nh, self.dh).transpose(1, 2)
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

class SmaleMorseWitnessEngine:
    def __init__(self, num_landmarks=16):
        self.num_landmarks = num_landmarks

    def compute_smale_morse_guidance(self, point_cloud: torch.Tensor) -> float:
        pts = point_cloud.detach().cpu().numpy()
        N = pts.shape[0]
        if N < 4: return 1.0
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        eps = np.median(dists) * 0.75
        edges = [(i, j) for i in range(N) for j in range(i+1, N) if dists[i, j] <= eps]
        if len(edges) == 0: return 1.0

        d1 = np.zeros((N, len(edges)))
        for idx, (u, v) in enumerate(edges):
            d1[u, idx] = -1.0; d1[v, idx] = 1.0

        rank_d1 = np.linalg.matrix_rank(d1)
        b0 = N - rank_d1
        return max(0.05, min(1.0, 1.0 / (float(b0) + 1e-5)))

    def compute_b1(self, point_cloud: torch.Tensor, eps: float = 0.5) -> int:
        pts = point_cloud.detach().cpu().numpy()
        N = pts.shape[0]
        if N < 4: return 0
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        edges = [(i, j) for i in range(N) for j in range(i+1, N) if dists[i, j] <= eps]
        if len(edges) == 0: return 0

        d1 = np.zeros((N, len(edges)))
        for idx, (u, v) in enumerate(edges):
            d1[u, idx] = -1.0; d1[v, idx] = 1.0

        triangles = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    if dists[i,j] <= eps and dists[j,k] <= eps and dists[i,k] <= eps:
                        triangles.append((i, j, k))
        
        if len(triangles) == 0:
            return max(0, len(edges) - np.linalg.matrix_rank(d1))

        d2 = np.zeros((len(edges), len(triangles)))
        edge_map = {edge: idx for idx, edge in enumerate(edges)}
        for t_idx, (u, v, w) in enumerate(triangles):
            e1 = edge_map.get((u, v)) if (u, v) in edge_map else edge_map.get((v, u))
            e2 = edge_map.get((v, w)) if (v, w) in edge_map else edge_map.get((w, v))
            e3 = edge_map.get((u, w)) if (u, w) in edge_map else edge_map.get((w, u))
            d2[e1, t_idx] = 1.0; d2[e2, t_idx] = 1.0; d2[e3, t_idx] = -1.0

        rank_d1 = np.linalg.matrix_rank(d1)
        rank_d2 = np.linalg.matrix_rank(d2)
        return max(0, int((len(edges) - rank_d1) - rank_d2))

class IntegratedGeometryEngine:
    def __init__(self, model: nn.Module, proj_dim: int = 128, max_points: int = 16):
        self.model = model
        self.proj_dim = proj_dim
        self.max_points = max_points
        self.point_cloud_buffer = []
        self.proj_matrix = None
        self.engine = SmaleMorseWitnessEngine(num_landmarks=max_points)

    def _ensure_projection_matrix(self, active_dim: int, device: torch.device):
        if self.proj_matrix is None or self.proj_matrix.shape[0] != active_dim:
            g = torch.Generator(device=device).manual_seed(42)
            self.proj_matrix = torch.randn(active_dim, self.proj_dim, generator=g, device=device) / (self.proj_dim ** 0.5)

    def capture_point(self):
        active_params = torch.cat([p.data.flatten() for p in self.model.parameters() if p.requires_grad])
        self._ensure_projection_matrix(active_params.numel(), active_params.device)
        proj_state = torch.matmul(active_params, self.proj_matrix)
        self.point_cloud_buffer.append(proj_state.detach())
        if len(self.point_cloud_buffer) > self.max_points:
            self.point_cloud_buffer.pop(0)

    def get_morse_guidance(self) -> float:
        self.capture_point()
        pts = torch.stack(self.point_cloud_buffer)
        return self.engine.compute_smale_morse_guidance(pts)

    def update_and_evaluate_b1(self) -> int:
        self.capture_point()
        pts = torch.stack(self.point_cloud_buffer)
        return self.engine.compute_b1(pts)

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

def gluing_defect(model, n=4):
    model.zero_grad()
    ls = []
    for _ in range(n):
        x, y = get_batch()
        _, l = model(x, y)
        if l is not None and l.requires_grad:
            ls.append(l)
    if not ls:
        return 1.0
    torch.stack(ls).mean().backward()
    g_ff = sum(p.grad.data.norm().item() for nm, p in model.named_parameters() if '.ff.' in nm and p.grad is not None)
    g_emb = model.te.weight.grad.data.norm().item() if (model.te.weight.grad is not None) else 1e-8
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

print("="*65)
print("FULLY INTEGRATED MORSE-GUIDED GEOMETRY COMPILER (V3.2)")
print(f"Target Floor (H_bigram): {H_BIGRAM:.4f} nats")
print("="*65); print()

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

print("━━━ GD-400 BASELINE RUN (400 SGD Steps) ━━━━━━━━━━━━━━━━")
torch.manual_seed(42)
gd_model = LM()
gd_model.te.weight.data.copy_(torch.tensor(E_init))
gd_opt = torch.optim.SGD(gd_model.parameters(), lr=1e-2, momentum=0.9)

gd_start_time = time.time()
for step in range(1, 401):
    gd_model.train(); x, y = get_batch(); _, loss = gd_model(x, y)
    gd_opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(gd_model.parameters(), 1.0)
    gd_opt.step()
    if step % 100 == 0:
        print(f"  GD-400 Step {step:3d}/400: val_loss={eval_val(gd_model, n=10):.4f}")

gd_time = time.time() - gd_start_time
gd_final_val = eval_val(gd_model, n=20)
print(f"  ✓ GD-400 Complete in {gd_time:.2f}s | Final Val Loss: {gd_final_val:.4f}\n")

print("━━━ STEP 1: ANCHOR SETUP & MORSE WITNESS INIT ━━━━━━━━━━")
torch.manual_seed(42)
model = LM(); model.te.weight.data.copy_(torch.tensor(E_init))
geometry_engine = IntegratedGeometryEngine(model, proj_dim=128, max_points=16)

for _ in range(10): geometry_engine.capture_point()
morse_factor = geometry_engine.get_morse_guidance()
print(f"  Smale-Morse witness complex initialized: Morse Factor={morse_factor:.4f}\n")

print("━━━ STEP 2: 2D SUBSPACE SADDLE EXIT ━━━━━━━━━━━━━━━━━━━━")
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
model.zero_grad(); _, l_tmp = model(*get_batch()); l_tmp.backward()
g_flat = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
v_dir = (0.7 * (v_neg / v_neg.norm()) - 0.3 * (g_flat / (g_flat.norm() + 1e-8)))
v_dir = v_dir / v_dir.norm()

w0 = model.flat_params(); best_v = eval_val(model, n=8); best_a = 0.
for alpha in np.logspace(-4, -1, 10):
    model.set_flat(w0 + alpha * v_dir)
    vt = eval_val(model, n=4)
    if vt < best_v: best_v = vt; best_a = alpha

model.set_flat(w0 + best_a * v_dir)
print(f"  α*={best_a:.4f}  val={eval_val(model):.4f}  sheet={sheet_angles(model)}\n")

print("━━━ STEP 3: TRUST-REGION MORSE-BREGMAN PUMP ━━━━━━━━━━━")
for mf_r in range(1, 16):
    x, y = get_batch(); logits, _ = model(x, y)
    P, eta, duality_gap = compute_amari_bregman(logits)
    
    morse_factor = geometry_engine.get_morse_guidance()
    effective_eta_mf = ETA_MF_BASE * morse_factor

    for bl in model.blocks:
        bl.attn.beta = float(1.0 / (duality_gap + 1.0))

    emb_grad = torch.zeros_like(model.te.weight)
    emb_fish = torch.zeros_like(model.te.weight)
    for i in range(N_SUB // 10):
        model.zero_grad(); _, loss = model(*get_batch()); loss.backward()
        if model.te.weight.grad is not None:
            g = model.te.weight.grad.detach(); emb_grad += g; emb_fish += g**2
            
    delta_E = -(emb_grad / (torch.sqrt(emb_fish) + 1e-2))
    delta_E = delta_E * (0.05 / max(float(delta_E.norm()), 1e-8))
    
    with torch.no_grad(): model.te.weight.add_(effective_eta_mf * delta_E)

    tau = gluing_defect(model, n=4); pc = phi_clean(model)
    print(f"  MF{mf_r:2d}: val={eval_val(model, n=2):.4f}  Φ_cl={pc}/5  τ={tau:.2f}  gap={duality_gap:.6f}  morse_γ={morse_factor:.3f}")

print("\n━━━ STEP 4: SAFEGUARDED ATTENTION PRUNING ━━━━━━━━━━━━━")
pruned_blocks = []
max_prune = N_STU // 2

for bi, bl in enumerate(model.blocks):
    model.zero_grad()
    _, l = model(*get_batch()); l.backward()
    gq = bl.attn.WQ.weight.grad; gv = bl.attn.WV.weight.grad
    r = (gq.norm() if gq is not None else 0.0) / max(gv.norm() if gv is not None else 1.0, 1e-12)
    
    if r < 0.15 and len(pruned_blocks) < max_prune:
        pruned_blocks.append(bi)
        bl.attn.WQ.weight.requires_grad_(False)
        bl.attn.WK.weight.requires_grad_(False)

active_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Pruned QK blocks: {pruned_blocks} (Max allowed: {max_prune}) | Active Parameters: {active_p:,}")

print("\n━━━ STEP 5: ROLLING N-POINT HOMOLOGY GEO-STOP ━━━━━━━━━━")
opt_b = CompressedAdam(list(model.parameters()), lr=LR, weight_decay=0.1,
                       names=[n for n, _ in model.named_parameters()])

val_history = []

for step in range(1, 301):
    model.train(); x, y = get_batch(); logits, l = model(x, y)
    
    tau_current = gluing_defect(model, n=2)
    lr_scale = min(2.0, max(0.5, 1.8 / max(tau_current, 1e-2)))
    opt_b.param_groups[0]['lr'] = LR * lr_scale

    opt_b.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_b.step()

    if step % 8 == 0:
        b1 = geometry_engine.update_and_evaluate_b1()
        _, _, duality_gap = compute_amari_bregman(logits)
        tau = gluing_defect(model, n=4)
        v = eval_val(model, n=4)
        val_history.append(v)
        print(f"  step {step:3d}: val={v:.4f}  b1={b1}  τ={tau:.2f}  gap={duality_gap:.6f}  lr_scale={lr_scale:.2f}")
        
        plateau = len(val_history) >= 3 and (val_history[-3] - val_history[-1]) < 0.005
        if step >= 160 and b1 == 0 and duality_gap < 1e-3 and (TAU_MIN <= tau <= TAU_MAX) and plateau:
            print(f"  ✓ Geo-Stop triggered at step {step}!"); break

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

print("\n━━━ STEP 7: DAMPENED K₀ SPLIT DESCENT & LANCZOS ━━━━━━━")
for bi in pruned_blocks:
    model.blocks[bi].attn.WQ.weight.requires_grad_(True)
    model.blocks[bi].attn.WK.weight.requires_grad_(True)

def k0_split_dampened(base, steps=30):
    m1 = copy.deepcopy(base); m2 = copy.deepcopy(base)
    
    p1 = [p for n, p in m1.named_parameters() if ('te.' in n or '.ff.' in n) and p.requires_grad]
    opt1 = torch.optim.AdamW(p1, lr=LR)
    for _ in range(steps):
        m1.train(); x, y = get_batch(); _, l = m1(x, y)
        opt1.zero_grad(); l.backward(); opt1.step()

    p2 = [p for n, p in m2.named_parameters() if '.attn.' in n and p.requires_grad]
    opt2 = torch.optim.AdamW(p2, lr=LR)
    for _ in range(steps):
        m2.train(); x, y = get_batch(); _, l = m2(x, y)
        opt2.zero_grad(); l.backward(); opt2.step()

    base_params = {n: p.data.clone() for n, p in base.named_parameters()}
    with torch.no_grad():
        tau_now = gluing_defect(base, n=4)
        w_ff = float(np.clip(1.5 * (1.8 / max(tau_now, 1e-2)), 0.8, 1.5))
        for n, p in base.named_parameters():
            if p.requires_grad:
                d1 = dict(m1.named_parameters())[n].data - base_params[n]
                d2 = dict(m2.named_parameters())[n].data - base_params[n]
                if 'te.' in n or '.ff.' in n: p.data.add_(w_ff * d1)
                else: p.data.add_(d2)

k0_split_dampened(model, steps=30)
print(f"  Post K0-Split val: {eval_val(model):.4f}")

compiler_final_val = eval_val(model)

print("\n" + "="*65)
print("INTEGRATED MORSE-GUIDED BENCHMARK SUMMARY (V3.2)")
print("="*65)
print(f"  Bigram Entropy Floor (Target):  {H_BIGRAM:.4f} nats")
print(f"  GD-400 Baseline (400 SGD Steps): {gd_final_val:.4f} nats")
print(f"  Morse-Guided Geometry Compiler:  {compiler_final_val:.4f} nats")
print(f"  Net Reduction vs. Baseline:     {gd_final_val - compiler_final_val:+.4f} nats")
print("="*65)
