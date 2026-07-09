"""
geo_compiler_surface.py
=======================
An IMPORTABLE surface over the geometry-driven training logic.

The original compiler_geometri_patched_86_memfixed.py is a top-to-bottom
*script*: importing it would execute the entire pipeline (and sys.exit if the
/tmp corpus is missing). That makes it impossible to drive op-by-op from C++.

This module extracts the reusable pieces into a class whose methods are each a
single, submittable operation returning a plain float (so a C++ std::future can
read the result). It uses the REAL LM model and REAL eval/step logic; only the
corpus is swappable: by default it builds a small synthetic in-memory corpus so
the module imports instantly and every op does genuine PyTorch work. Point
USE_REAL_CORPUS=True (or call use_real_corpus()) to load /tmp/*.json instead.

Design contract for the bridge:
  - construct GeometryCompiler() once (heavy: builds model + corpus)
  - each method (eval_val, train_step, mf_pump_round, mem_sample) is one op
  - every method returns float (or a small tuple of floats) -> future-friendly
  - no printing to stdout from ops (the C++ side owns I/O ordering)
"""

import os, json, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ---- hyperparameters (mirrors the original) --------------------------------
D = 256; N_HEADS = 4; N_STU = 6; BATCH = 8; SEQ = 64; LR = 3e-4
ETA_MF = 0.01; N_SUB = 200

USE_REAL_CORPUS = False  # flip to True to load /tmp/{train,val,vocab}.json


# ---- model (verbatim structure from the original) --------------------------
class Attn(nn.Module):
    def __init__(self):
        super().__init__(); dh = D // N_HEADS
        self.WQ = nn.Linear(D, D, bias=False); self.WK = nn.Linear(D, D, bias=False)
        self.WV = nn.Linear(D, D, bias=False); self.op = nn.Linear(D, D, bias=False)
        self.ln = nn.LayerNorm(D); self.sc = math.sqrt(dh); self.nh = N_HEADS; self.dh = dh
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
        self.g = nn.Linear(D, D * 2, bias=False); self.v = nn.Linear(D, D * 2, bias=False)
        self.o = nn.Linear(D * 2, D, bias=False); self.n = nn.LayerNorm(D)
        for w in [self.g, self.v, self.o]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, h):
        return self.n(h + self.o(F.silu(self.g(h)) * self.v(h)))


class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.attn = Attn(); self.ff = FF()

    def forward(self, h):
        return self.ff(self.attn(h))


class LM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.te = nn.Embedding(vocab, D); self.pe = nn.Embedding(512, D)
        self.blocks = nn.ModuleList([Block() for _ in range(N_STU)])
        self.ln_f = nn.LayerNorm(D); self.head = nn.Linear(D, vocab, bias=False)
        self.head.weight = self.te.weight
        nn.init.normal_(self.te.weight, std=0.02); nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x, y=None):
        h = self.te(x) + self.pe(torch.arange(x.shape[1]))
        for b in self.blocks:
            h = b(h)
        logits = self.head(self.ln_f(h))
        loss = (F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                if y is not None else None)
        return logits, loss


# ---- corpus ----------------------------------------------------------------
def _load_real_corpus():
    for f in ['/tmp/train_ids.json', '/tmp/val_ids.json', '/tmp/vocab.json']:
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} missing. Run build_corpus.py first.")
    with open('/tmp/train_ids.json') as f: train_ids = list(map(int, json.load(f)))
    with open('/tmp/val_ids.json')   as f: val_ids   = list(map(int, json.load(f)))
    with open('/tmp/vocab.json')      as f: _v = json.load(f)
    vocab = len(_v)
    return (torch.tensor(train_ids, dtype=torch.long),
            torch.tensor(val_ids, dtype=torch.long), vocab)


def _synthetic_corpus(vocab=512, n_train=20000, n_val=4000, seed=0):
    # A learnable-but-cheap synthetic stream: a noisy order-1 Markov chain so
    # cross-entropy actually decreases with training (unlike pure random ids).
    g = torch.Generator().manual_seed(seed)
    trans = torch.randint(0, vocab, (vocab,), generator=g)  # each tok -> preferred next

    def gen(n):
        out = torch.empty(n, dtype=torch.long)
        out[0] = 0
        for i in range(1, n):
            if torch.rand(1, generator=g).item() < 0.8:
                out[i] = trans[out[i - 1]]
            else:
                out[i] = torch.randint(0, vocab, (1,), generator=g)
        return out

    return gen(n_train), gen(n_val), vocab


# ---- the importable surface ------------------------------------------------
class GeometryCompiler:
    """One instance = one training context. Methods are individual ops."""

    def __init__(self, seed: int = 99, use_real_corpus: bool | None = None):
        if use_real_corpus is None:
            use_real_corpus = USE_REAL_CORPUS
        if use_real_corpus:
            self.train_t, self.val_t, self.vocab = _load_real_corpus()
        else:
            self.train_t, self.val_t, self.vocab = _synthetic_corpus()
        torch.manual_seed(seed)
        self.model = LM(self.vocab)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=LR,
                                     betas=(0.9, 0.95), weight_decay=0.1)
        self._step = 0

    # -- batching --
    def _get_batch(self, split='train'):
        data = self.val_t if split == 'val' else self.train_t
        ix = torch.randint(0, len(data) - SEQ - 1, (BATCH,))
        x = torch.stack([data[i:i + SEQ] for i in ix])
        y = torch.stack([data[i + 1:i + SEQ + 1] for i in ix])
        return x, y

    # -- OP: evaluate validation loss (returns float) --
    def eval_val(self, n: int = 12) -> float:
        self.model.eval(); ls = []
        with torch.no_grad():
            for _ in range(n):
                x, y = self._get_batch('val'); _, l = self.model(x, y)
                ls.append(l.item())
        self.model.train()
        return float(np.mean(ls))

    # -- OP: one AdamW training step (returns train loss) --
    def train_step(self) -> float:
        self.model.train()
        x, y = self._get_batch('train')
        _, l = self.model(x, y)
        self.opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self._step += 1
        return float(l.item())

    # -- OP: run k training steps, return final train loss --
    def train_steps(self, k: int = 10) -> float:
        last = 0.0
        for _ in range(k):
            last = self.train_step()
        return last

    # -- OP: gluing defect (tau sensor) --
    def gluing_defect(self, n: int = 6) -> float:
        self.model.zero_grad()
        ls = [self.model(*self._get_batch())[1] for _ in range(n)]
        torch.stack(ls).mean().backward()
        g_ff = sum(p.grad.data.norm().item()
                   for nm, p in self.model.named_parameters()
                   if '.ff.' in nm and p.grad is not None)
        g_emb = (self.model.te.weight.grad.data.norm().item()
                 if self.model.te.weight.grad is not None else 1e-8)
        self.model.zero_grad()
        return float(g_ff / max(g_emb, 1e-8))

    # -- OP: current allocated memory in MiB (for the mem profile) --
    def mem_allocated_mib(self) -> float:
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024 * 1024))
        # CPU fallback: sum parameter + grad bytes as a proxy
        b = 0
        for p in self.model.parameters():
            b += p.numel() * p.element_size()
            if p.grad is not None:
                b += p.grad.numel() * p.grad.element_size()
        return float(b / (1024 * 1024))

    def num_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def step_count(self) -> int:
        return self._step


def use_real_corpus(flag: bool = True):
    global USE_REAL_CORPUS
    USE_REAL_CORPUS = flag


# Quick self-test when run directly (NOT on import).
if __name__ == "__main__":
    gc = GeometryCompiler()
    print(f"params={gc.num_params()} vocab={gc.vocab}")
    print(f"val before: {gc.eval_val(n=4):.4f}")
    for _ in range(3):
        print(f"  train_steps(10) -> loss {gc.train_steps(10):.4f}")
    print(f"val after:  {gc.eval_val(n=4):.4f}")
    print(f"tau={gc.gluing_defect():.3f}  mem={gc.mem_allocated_mib():.2f} MiB")
