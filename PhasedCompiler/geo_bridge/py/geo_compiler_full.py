"""
geo_compiler_full.py
====================
Extends geo_compiler_surface with the actual geometry-compiler PHASES as
individual, schedule-callable ops:

    saddle()          -> Phase 1: negative-curvature saddle exit  (returns val)
    mfpump()          -> Phase 2: one mean-field pump round        (returns val)
    lanczos()         -> Lanczos terminal projection (3 solves)    (returns val)

plus the basics from the base surface (train_steps/eval_val/gluing_defect/mem).

These are faithful extractions of the algorithms in
compiler_geometri_patched_86_memfixed.py, refactored so the module-scope globals
they depend on (E_init spectral embedding, g_floor floor gradient) become
instance state built once in __init__. Each phase mutates self.model IN PLACE and
returns a scalar the schedule can branch on -- matching the "one op -> one float"
contract the C++ bridge expects.

Corpus: synthetic by default (imports instantly); pass use_real_corpus=True to
load /tmp/{train,val,vocab}.json and build the REAL spectral E_init + g_floor.
"""

import os, json, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# reuse model + hyperparameters + corpus helpers from the base surface
from geo_compiler_surface import (
    D, N_HEADS, N_STU, BATCH, SEQ, LR, ETA_MF, N_SUB,
    LM, _synthetic_corpus, _load_real_corpus,
)

# optional scipy (only needed for the REAL spectral E_0). Synthetic path skips it.
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


class GeometryCompilerFull:
    def __init__(self, seed: int = 99, use_real_corpus: bool = False,
                 build_floor: bool = True, floor_steps: int = 200):
        if use_real_corpus:
            self.train_t, self.val_t, self.vocab = _load_real_corpus()
        else:
            self.train_t, self.val_t, self.vocab = _synthetic_corpus()

        # ---- spectral E_init (Phase 0 anchor) --------------------------------
        # For the real corpus we build the graph-Laplacian spectral embedding
        # exactly as the original does. For synthetic we skip it (random init is
        # fine as a starting point; the phases still run and mutate the model).
        E_init = None
        if use_real_corpus and _HAVE_SCIPY:
            E_init = self._build_spectral_E0(seed)

        torch.manual_seed(seed)
        self.model = LM(self.vocab)
        if E_init is not None:
            with torch.no_grad():
                self.model.te.weight.data.copy_(torch.tensor(E_init))

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=LR,
                                     betas=(0.9, 0.95), weight_decay=0.1)
        self._step = 0

        # ---- floor gradient g_floor (geometric anchor for alignment) ---------
        self.g_floor = None
        if build_floor:
            self.g_floor = self._build_floor_gradient(seed=42, steps=floor_steps)

    # ======================= corpus-derived anchors =========================
    def _build_spectral_E0(self, seed):
        import collections
        train_ids = self.train_t.tolist()
        VOCAB = self.vocab
        bigram = collections.Counter(); perm = {}
        for i in range(len(train_ids) - 1):
            a, b = train_ids[i], train_ids[i + 1]
            if a < VOCAB and b < VOCAB:
                bigram[(a, b)] += 1; perm.setdefault(a, b)
        rows, cols, vv = [], [], []
        for (a, b), cnt in bigram.items():
            rows.append(a); cols.append(b); vv.append(float(cnt))
        W = sp.csr_matrix((vv, (rows, cols)), shape=(VOCAB, VOCAB), dtype=np.float32)
        W = W + W.T
        d_inv = np.array(1.0 / (W.sum(1) + 1e-8)).flatten()
        Dsi = sp.diags(np.sqrt(d_inv))
        L = sp.eye(VOCAB) - Dsi @ W @ Dsi
        k = min(D + 1, VOCAB - 1)
        evals, evecs = spla.eigsh(L, k=k, which='SM', tol=1e-4, maxiter=2000)
        idx = np.argsort(evals); evecs = evecs[:, idx][:, 1:D + 1]
        E0 = (evecs / (np.sqrt(evals[idx[1:D + 1]]) + 1e-8)[np.newaxis, :]).astype(np.float32)
        E0 = E0 / (E0.std() + 1e-8) * 0.02
        Enext = np.array([E0[perm.get(t, t)] for t in range(VOCAB)], dtype=np.float32)
        Einit = 0.9 * E0 + 0.1 * Enext
        norm = float(np.linalg.norm(E0))
        Einit = (Einit * (norm / max(float(np.linalg.norm(Einit)), 1e-8))).astype(np.float32)
        # pad/truncate rows to vocab (already vocab-sized) and cols to D
        return Einit

    def _build_floor_gradient(self, seed, steps):
        torch.manual_seed(seed)
        m = LM(self.vocab)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
        for _ in range(steps):
            m.train(); x, y = self._get_batch('train'); _, l = m(x, y)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.zero_grad()
        ls = [m(*self._get_batch())[1] for _ in range(20)]
        torch.stack(ls).mean().backward()
        g = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                       for p in m.parameters()]).detach()
        return g

    # ============================ helpers ===================================
    def _get_batch(self, split='train'):
        data = self.val_t if split == 'val' else self.train_t
        ix = torch.randint(0, len(data) - SEQ - 1, (BATCH,))
        return (torch.stack([data[i:i + SEQ] for i in ix]),
                torch.stack([data[i + 1:i + SEQ + 1] for i in ix]))

    def _flat_params(self):
        return torch.cat([p.data.flatten() for p in self.model.parameters()])

    def _set_flat(self, v):
        i = 0
        for p in self.model.parameters():
            n = p.numel(); p.data.copy_(v[i:i + n].reshape(p.shape)); i += n

    def _hvp(self, vec, n=8):
        self.model.zero_grad()
        ls = [self.model(*self._get_batch())[1] for _ in range(n)]
        loss = torch.stack(ls).mean()
        grads = torch.autograd.grad(loss, list(self.model.parameters()), create_graph=True)
        gv = (torch.cat([gr.flatten() for gr in grads]) * vec.detach()).sum()
        hv = torch.cat([h.flatten() for h in
                        torch.autograd.grad(gv, list(self.model.parameters()),
                                            retain_graph=False)])
        self.model.zero_grad()
        return hv.detach()

    # ============================ basic ops =================================
    def eval_val(self, n=12) -> float:
        self.model.eval(); ls = []
        with torch.no_grad():
            for _ in range(n):
                x, y = self._get_batch('val'); _, l = self.model(x, y); ls.append(l.item())
        self.model.train()
        return float(np.mean(ls))

    def train_steps(self, k=10) -> float:
        last = 0.0
        for _ in range(k):
            self.model.train(); x, y = self._get_batch('train'); _, l = self.model(x, y)
            self.opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.opt.step()
            self._step += 1; last = float(l.item())
        return last

    def gluing_defect(self, n=6) -> float:
        self.model.zero_grad()
        ls = [self.model(*self._get_batch())[1] for _ in range(n)]
        torch.stack(ls).mean().backward()
        g_ff = sum(p.grad.data.norm().item() for nm, p in self.model.named_parameters()
                   if '.ff.' in nm and p.grad is not None)
        g_emb = (self.model.te.weight.grad.data.norm().item()
                 if self.model.te.weight.grad is not None else 1e-8)
        self.model.zero_grad()
        return float(g_ff / max(g_emb, 1e-8))

    def mem_allocated_mib(self) -> float:
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024 * 1024))
        b = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return float(b / (1024 * 1024))

    def num_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def step_count(self) -> int:
        return self._step

    # -- Φ_clean sensor: count of layer-pair angles that are 0 or π ----------
    def phi_clean(self) -> int:
        import math as _m
        WKs = [self.model.blocks[l].attn.WK.weight.data.float() for l in range(N_STU)]
        clean = 0
        for l in range(N_STU - 1):
            try:
                phi = WKs[l + 1] @ torch.linalg.pinv(WKs[l])
                lam = torch.linalg.eigvals(phi)
                lam1 = lam[lam.abs().argmax()]
                a = float(torch.angle(lam1))
                if abs(abs(a) - _m.pi) < 0.3 or abs(a) < 0.3:
                    clean += 1
            except Exception:
                pass
        return int(clean)

    # ============================ PHASES ====================================
    # -- Phase 1: saddle exit (negative-curvature power iteration + line search)
    def saddle(self, n_iter: int = 15, hvp_n: int = 8) -> float:
        n_p = self.num_params()
        torch.manual_seed(42)
        v = torch.randn(n_p); v = v / v.norm()
        for _ in range(n_iter):
            Hv = self._hvp(v, hvp_n); neg = -Hv
            v = neg / max(float(neg.norm()), 1e-10)
        w0 = self._flat_params()
        best_v = self.eval_val(n=8); best_a = 0.0
        vn = v / v.norm()
        for alpha in [0.5, 1.0, 1.429, 2.0, 3.0, 4.0]:
            self._set_flat(w0 + alpha * vn); vt = self.eval_val(n=6)
            if vt < best_v: best_v = vt; best_a = alpha
        self._set_flat(w0 + best_a * vn)
        return self.eval_val(n=8)

    # -- Phase 2: one mean-field pump round (embedding Fisher + WK/WQ Fisher) --
    def mfpump(self, seed: int = 0) -> float:
        m = self.model
        # (a) embedding Fisher step, attention frozen
        for l in range(N_STU):
            m.blocks[l].attn.WK.weight.requires_grad_(False)
            m.blocks[l].attn.WQ.weight.requires_grad_(False)
        emb_grad = torch.zeros(m.te.weight.shape); emb_fish = torch.zeros(m.te.weight.shape)
        torch.manual_seed(seed * 1000)
        for _ in range(N_SUB):
            ix = torch.randint(0, len(self.train_t) - SEQ - 1, (1,))[0].item()
            x = self.train_t[ix:ix + SEQ].unsqueeze(0); y = self.train_t[ix + 1:ix + SEQ + 1].unsqueeze(0)
            m.zero_grad(); _, loss = m(x, y); loss.backward()
            if m.te.weight.grad is not None:
                g = m.te.weight.grad.detach(); emb_grad += g; emb_fish += g ** 2
        emb_grad /= N_SUB; emb_fish /= N_SUB
        with torch.no_grad():
            m.te.weight.add_(ETA_MF * (-(emb_grad / (emb_fish + 1e-4))))
        for l in range(N_STU):
            m.blocks[l].attn.WK.weight.requires_grad_(True)
            m.blocks[l].attn.WQ.weight.requires_grad_(True)

        # (b) WK/WQ Fisher step, embedding frozen
        m.te.weight.requires_grad_(False)
        wk_grad = torch.zeros_like(m.blocks[0].attn.WK.weight)
        wk_fish = torch.zeros_like(m.blocks[0].attn.WK.weight)
        torch.manual_seed(seed * 1000 + 500)
        for _ in range(N_SUB):
            ix = torch.randint(0, len(self.train_t) - SEQ - 1, (1,))[0].item()
            x = self.train_t[ix:ix + SEQ].unsqueeze(0); y = self.train_t[ix + 1:ix + SEQ + 1].unsqueeze(0)
            m.zero_grad(); _, loss = m(x, y); loss.backward()
            g = torch.zeros_like(m.blocks[0].attn.WK.weight)
            for bl in m.blocks:
                if bl.attn.WK.weight.grad is not None: g += bl.attn.WK.weight.grad / N_STU
            wk_grad += g; wk_fish += g ** 2
        wk_grad /= N_SUB; wk_fish /= N_SUB
        delta = -(wk_grad / (wk_fish + 1e-4))
        with torch.no_grad():
            for l in range(N_STU):
                m.blocks[l].attn.WK.weight.add_(ETA_MF * delta)
                m.blocks[l].attn.WQ.weight.add_(ETA_MF * delta.T)
        m.te.weight.requires_grad_(True)
        return self.eval_val(n=4)

    # -- Lanczos terminal projection: k=8 basis, 3 projected Newton solves -----
    def lanczos(self, k: int = 8, solves: int = 3, mu: float = 0.95) -> float:
        n_p = self.num_params()
        torch.manual_seed(7); q = torch.randn(n_p); q = q / q.norm()
        Q = [q]; alphas = []; betas = []
        for j in range(k):
            z = self._hvp(Q[j], n=4); alpha = float((Q[j] * z).sum()); alphas.append(alpha)
            z = z - alpha * Q[j]
            if j > 0: z = z - betas[-1] * Q[j - 1]
            for qi in Q: z = z - float((qi * z).sum()) * qi
            beta = float(z.norm()); betas.append(beta)
            if beta < 1e-8: break
            Q.append(z / beta)
        n_l = len(alphas)
        T = torch.zeros(n_l, n_l)
        for i in range(n_l): T[i, i] = alphas[i]
        for i in range(n_l - 1): T[i, i + 1] = betas[i]; T[i + 1, i] = betas[i]
        T_evals, T_evecs = torch.linalg.eigh(T)
        V = torch.stack(Q[:n_l], dim=1) @ T_evecs

        for _ in range(solves):
            self.model.zero_grad()
            ls = [self.model(*self._get_batch())[1] for _ in range(25)]
            torch.stack(ls).mean().backward()
            g = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                           for p in self.model.parameters()]).detach()
            self.model.zero_grad()
            g_proj = V.T @ g; d_proj = g_proj / (T_evals + mu)
            g_res = g - V @ (V.T @ g); d = -(V @ d_proj + g_res / mu)
            w0 = self._flat_params(); v0 = self.eval_val(n=8)
            self._set_flat(w0 + d); v1 = self.eval_val(n=8)
            if not (v1 < v0):
                self._set_flat(w0); break
        return self.eval_val(n=8)


# alias so the C++ bridge can construct either surface by the same class name
GeometryCompiler = GeometryCompilerFull


if __name__ == "__main__":
    gc = GeometryCompilerFull(use_real_corpus=False, build_floor=True, floor_steps=30)
    print(f"params={gc.num_params()} vocab={gc.vocab}")
    print(f"val0   = {gc.eval_val(4):.4f}")
    print(f"saddle = {gc.saddle():.4f}")
    print(f"mfpump = {gc.mfpump(seed=0):.4f}  tau={gc.gluing_defect():.3f}")
    print(f"lanczos= {gc.lanczos():.4f}")
