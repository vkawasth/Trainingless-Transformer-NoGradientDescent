python - << 'EOF'
import numpy as np, torch, json, scipy.sparse as sp, scipy.sparse.linalg as spla
import collections

# Load corpus
with open('/tmp/train_ids.json') as f: train_ids = list(map(int, json.load(f)))
with open('/tmp/vocab.json')     as f: _v = json.load(f)
VOCAB = len(_v) if isinstance(_v, list) else len(_v)
D, rank = 256, 6

# Build spectral embedding E0 (same as compiler)
bigram = collections.Counter()
for i in range(len(train_ids)-1):
    a, b = train_ids[i], train_ids[i+1]
    if a < VOCAB and b < VOCAB: bigram[(a,b)] += 1
rows, cols, vv = [], [], []
for (a,b), cnt in bigram.items(): rows.append(a); cols.append(b); vv.append(float(cnt))
W_sp = sp.csr_matrix((vv,(rows,cols)), shape=(VOCAB,VOCAB), dtype=np.float32)
W_sp = W_sp + W_sp.T
d_inv = np.array(1.0/(W_sp.sum(1)+1e-8)).flatten()
Dsi = sp.diags(np.sqrt(d_inv))
L_sym = sp.eye(VOCAB) - Dsi @ W_sp @ Dsi
evals, evecs = spla.eigsh(L_sym, k=D+1, which='SM', tol=1e-4, maxiter=2000)
idx_s = np.argsort(evals)
evecs = evecs[:, idx_s][:, 1:D+1]
E_0 = (evecs / (np.sqrt(evals[idx_s[1:D+1]])+1e-8)[np.newaxis,:]).astype(np.float32)
E_0 = (E_0 / (E_0.std()+1e-8) * 0.02)
print(f"E0 shape: {E_0.shape}")

# Build bigram right singular vectors
rows2, cols2, vv2 = [], [], []
for (a,b), cnt in bigram.items(): rows2.append(a); cols2.append(b); vv2.append(float(cnt))
B_sp = sp.csr_matrix((vv2,(rows2,cols2)), shape=(VOCAB,VOCAB), dtype=np.float32)
# Embed bigram into D-dim space via E0
B_dense = (E_0.T @ B_sp.toarray() @ E_0)  # D×D
U_B, s_B, Vt_B = np.linalg.svd(B_dense)
print(f"Bigram SVD: top singular values = {s_B[:6].round(4)}")

# Load trained WK matrices
ckpt = torch.load('basin_state.pt', map_location='cpu', weights_only=False)
state = ckpt.get('state_dict', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
wk = {}
for name, tensor in state.items():
    n = name.lower()
    if ('key' in n or 'wk' in n) and 'weight' in n and tensor.ndim >= 2:
        try: li = int([p for p in name.split('.') if p.isdigit()][0])
        except: li = len(wk)
        wk[li] = tensor.detach().float().numpy()
wk_list = [wk[i] for i in sorted(wk)]

print(f"\nTesting WK factorization at basin_state:")
print(f"{'Layer':>6} {'‖Uk-E0_r‖':>12} {'‖Vk-B_r‖':>12} {'σ1':>8} {'σ1_ratio':>10}")
print(f"  {'-'*52}")
prev_s1 = None
for k, W in enumerate(wk_list):
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    Ur = U[:, :rank]; Vr = Vt[:rank, :].T
    # Compare U_k with top-r Laplacian eigenvectors
    E0_r = E_0[:, :rank]  # VOCAB×rank — wrong dim, need D×rank
    # E0 is VOCAB×D, WK is D×D, so U_k is D×rank
    # The Laplacian eigenvectors are in VOCAB space, not D space
    # This comparison requires projecting E0 through the embedding
    # For now: check if U_k aligns with U_B (bigram left singular vectors)
    align_U = float(np.linalg.norm(Ur.T @ U_B[:, :rank]))  / rank
    align_V = float(np.linalg.norm(Vr.T @ Vt_B[:rank, :].T)) / rank
    ratio = s[0]/prev_s1 if prev_s1 else float('nan')
    print(f"  {k:>4}  {align_U:>12.4f}  {align_V:>12.4f}  "
          f"{s[0]:>8.3f}  {ratio:>10.3f}")
    prev_s1 = s[0]

print(f"\nIf align ≈ 1.0: WK column space aligns with corpus structure")
print(f"If σ1_ratio ≈ τ: geometric sequence confirmed")
EOF
