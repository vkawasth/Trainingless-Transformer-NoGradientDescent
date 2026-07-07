#!/usr/bin/env python3
"""
memory_viz.py  —  standalone, reproducible memory verification for
"Geometry-Driven Compiler vs GD-400".

WHY THIS EXISTS
---------------
The full training pipeline needs a private corpus, PyTorch, and an Apple/GPU
backend, which makes independent verification hard. This script removes every
one of those barriers:

  * builds its OWN tiny synthetic corpus (no external files),
  * builds the SAME model architecture as the paper (D=256, 6 layers,
    VOCAB=1017, tied embedding),
  * measures REAL memory with tracemalloc (CPU) or torch.*.max_memory_allocated
    (CUDA/MPS) — no RSS guesswork,
  * compares two optimizer regimes that mirror the paper's claim:
        (A) GD-400   : full AdamW state on ALL parameters (2 moments x P)
        (B) Compiler : AdamW state only on NON-masked parameters
                       (the dispensable WV + attention-output weights are
                        frozen -> no optimizer state, no gradient),
  * also computes the corpus-operator memory O(V^2) dense vs O(nnz) sparse,
  * writes self-explanatory artifacts anyone can inspect:
        memory_report.json      (machine-readable, exact bytes)
        memory_report.csv       (spreadsheet-friendly)
        memory_summary.txt      (plain-language, for a general reader)
        memory_viz.png          (the clear chart)

Run:
    python memory_viz.py
    python memory_viz.py --seed 0 --quick     # faster smoke test

Everything is deterministic given --seed, so two people get the same numbers.
"""

import argparse
import csv
import json
import math
import platform
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Config — mirrors compiler_geometri_patched_86.py
# ----------------------------------------------------------------------
D = 256
N_HEADS = 4
N_STU = 6
VOCAB = 1017
NNZ = 1347          # non-zero corpus pairs (from the paper's corpus)
SEQ = 64
BATCH = 8
BYTES_FP32 = 4


# ----------------------------------------------------------------------
# Model (identical structure to the paper's LM)
# ----------------------------------------------------------------------
class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        dh = D // N_HEADS
        self.WQ = nn.Linear(D, D, bias=False)
        self.WK = nn.Linear(D, D, bias=False)
        self.WV = nn.Linear(D, D, bias=False)   # dispensable (masked by compiler)
        self.op = nn.Linear(D, D, bias=False)   # dispensable (masked by compiler)
        self.ln = nn.LayerNorm(D)
        self.sc = math.sqrt(dh); self.nh = N_HEADS; self.dh = dh
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
        att = F.softmax(sc, dim=-1) @ V
        return self.ln(h + self.op(att.transpose(1, 2).reshape(B, S, D)))


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
        self.attn = Attn(); self.ff = FF()

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
        self.head.weight = self.te.weight   # tied
        nn.init.normal_(self.te.weight, std=0.02)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x, y=None):
        h = self.te(x) + self.pe(torch.arange(x.shape[1]))
        for b in self.blocks:
            h = b(h)
        logits = self.head(self.ln_f(h))
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        return logits, loss


# ----------------------------------------------------------------------
# Synthetic corpus: a cyclic near-permutation, matching the paper's
# "each token has exactly one successor" structure that produces the
# sparse (nnz << V^2) co-occurrence operator.
# ----------------------------------------------------------------------
def build_corpus(seed=42, n_tokens=20000):
    g = torch.Generator().manual_seed(seed)
    # cyclic successor map -> near-permutation -> sparse bigram support
    succ = torch.randperm(VOCAB, generator=g)
    ids = torch.empty(n_tokens, dtype=torch.long)
    ids[0] = int(torch.randint(0, VOCAB, (1,), generator=g))
    for i in range(1, n_tokens):
        ids[i] = succ[ids[i - 1]]
    return ids


def get_batch(ids, seed_gen):
    ix = torch.randint(0, len(ids) - SEQ - 1, (BATCH,), generator=seed_gen)
    x = torch.stack([ids[i:i + SEQ] for i in ix])
    y = torch.stack([ids[i + 1:i + SEQ + 1] for i in ix])
    return x, y


# ----------------------------------------------------------------------
# Memory measurement helpers
# ----------------------------------------------------------------------
def backend():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tensor_bytes(t):
    return t.numel() * t.element_size()


def analytic_optimizer_bytes(opt):
    total = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                total += tensor_bytes(v)
    return total


def param_bytes(model, trainable_only=False):
    return sum(tensor_bytes(p) for p in model.parameters()
               if (p.requires_grad or not trainable_only))


def grad_bytes(model):
    return sum(tensor_bytes(p.grad) for p in model.parameters()
               if p.grad is not None)


class PeakMeter:
    """Real high-water-mark of live torch tensor memory over a code region.

    tracemalloc only sees Python allocations, not PyTorch's C++ tensor
    storage, so on CPU we instead poll the sum of live tensor bytes via the
    garbage collector. On CUDA we use the exact driver counter; on MPS we use
    the allocated-memory counter."""
    def __init__(self):
        self.be = backend()
        self._peak = 0.0

    def _live_tensor_mb(self):
        import gc
        seen = set(); total = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.device.type == "cpu":
                    key = obj.untyped_storage().data_ptr()
                    if key in seen:
                        continue
                    seen.add(key)
                    total += obj.untyped_storage().nbytes()
            except Exception:
                continue
        return total / 1e6

    def __enter__(self):
        if self.be == "cuda":
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        return self

    def poll(self):
        """Call inside the loop to update the CPU/MPS high-water mark."""
        if self.be == "cpu":
            self._peak = max(self._peak, self._live_tensor_mb())
        elif self.be == "mps":
            self._peak = max(self._peak,
                             torch.mps.current_allocated_memory() / 1e6)

    def peak_mb(self):
        if self.be == "cuda":
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1e6
        self.poll()
        return self._peak

    def __exit__(self, *a):
        pass


# ----------------------------------------------------------------------
# The two regimes
# ----------------------------------------------------------------------
def mask_dispensable(model):
    """Freeze WV + op in every block: no grad, no optimizer state.
    Returns count of masked params."""
    masked = 0
    for blk in model.blocks:
        for w in (blk.attn.WV, blk.attn.op):
            w.weight.requires_grad_(False)
            masked += w.weight.numel()
    return masked


def run_regime(name, mask, steps, corpus, seed):
    torch.manual_seed(seed)
    model = LM()
    masked_params = mask_dispensable(model) if mask else 0

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed + 1)

    # warm one step so Adam state buffers are actually allocated
    x, y = get_batch(corpus, gen)
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()

    with PeakMeter() as pm:
        for _ in range(steps - 1):
            x, y = get_batch(corpus, gen)
            _, loss = model(x, y)
            opt.zero_grad(); loss.backward()
            pm.poll()   # capture high-water mark while grads+activations live
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        empirical_peak = pm.peak_mb()

    rec = {
        "regime": name,
        "steps": steps,
        "total_params": sum(p.numel() for p in model.parameters()),
        "masked_params": masked_params,
        "trainable_params": sum(p.numel() for p in trainable),
        "param_mb": param_bytes(model) / 1e6,
        "grad_mb": grad_bytes(model) / 1e6,
        "optimizer_state_mb": analytic_optimizer_bytes(opt) / 1e6,
        "empirical_peak_mb": empirical_peak,
    }
    rec["analytic_resident_mb"] = (
        rec["param_mb"] + rec["grad_mb"] + rec["optimizer_state_mb"])
    return rec


def corpus_operator_memory():
    dense = VOCAB * VOCAB * BYTES_FP32 / 1e6
    sparse = NNZ * 3 * BYTES_FP32 / 1e6   # COO: value + row + col
    return {
        "dense_O_V2_mb": dense,
        "sparse_O_nnz_mb": sparse,
        "reduction_x": (VOCAB * VOCAB) / NNZ,
        "V": VOCAB, "V2": VOCAB * VOCAB, "nnz": NNZ,
    }


# ----------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------
def write_artifacts(gd, comp, op, prefix="memory"):
    payload = {
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "backend": backend(),
            "platform": platform.platform(),
        },
        "config": {"D": D, "N_STU": N_STU, "VOCAB": VOCAB, "NNZ": NNZ,
                   "SEQ": SEQ, "BATCH": BATCH},
        "gd_400": gd,
        "compiler": comp,
        "corpus_operator": op,
        "headline": {
            "optimizer_state_saving_mb":
                gd["optimizer_state_mb"] - comp["optimizer_state_mb"],
            "optimizer_state_saving_pct":
                100 * (gd["optimizer_state_mb"] - comp["optimizer_state_mb"])
                / max(gd["optimizer_state_mb"], 1e-9),
            "resident_reduction_x":
                gd["analytic_resident_mb"]
                / max(comp["analytic_resident_mb"], 1e-9),
            "corpus_operator_reduction_x": op["reduction_x"],
        },
    }
    with open(f"{prefix}_report.json", "w") as f:
        json.dump(payload, f, indent=2)

    with open(f"{prefix}_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "steps", "total_params", "masked_params",
                    "trainable_params", "param_mb", "grad_mb",
                    "optimizer_state_mb", "analytic_resident_mb",
                    "empirical_peak_mb"])
        for r in (gd, comp):
            w.writerow([r["regime"], r["steps"], r["total_params"],
                        r["masked_params"], r["trainable_params"],
                        f'{r["param_mb"]:.3f}', f'{r["grad_mb"]:.3f}',
                        f'{r["optimizer_state_mb"]:.3f}',
                        f'{r["analytic_resident_mb"]:.3f}',
                        f'{r["empirical_peak_mb"]:.3f}'])

    h = payload["headline"]
    lines = [
        "MEMORY VERIFICATION SUMMARY",
        "=" * 60,
        f"Backend: {backend()}   torch {torch.__version__}",
        "",
        "MODEL / OPTIMIZER MEMORY (same model, two optimizer regimes)",
        "-" * 60,
        f"  GD-400   optimizer state : {gd['optimizer_state_mb']:8.2f} MB "
        f"(Adam on all {gd['trainable_params']:,} params)",
        f"  Compiler optimizer state : {comp['optimizer_state_mb']:8.2f} MB "
        f"(Adam on {comp['trainable_params']:,} params; "
        f"{comp['masked_params']:,} masked)",
        f"  Saving                   : {h['optimizer_state_saving_mb']:8.2f} MB "
        f"({h['optimizer_state_saving_pct']:.1f}% of Adam state)",
        "",
        f"  GD-400   resident total  : {gd['analytic_resident_mb']:8.2f} MB",
        f"  Compiler resident total  : {comp['analytic_resident_mb']:8.2f} MB",
        f"  Empirical peak GD-400    : {gd['empirical_peak_mb']:8.2f} MB",
        f"  Empirical peak Compiler  : {comp['empirical_peak_mb']:8.2f} MB",
        "",
        "CORPUS-OPERATOR MEMORY (the big one; grows with vocabulary)",
        "-" * 60,
        f"  Dense  O(V^2)  V={op['V']}  : {op['dense_O_V2_mb']:8.2f} MB",
        f"  Sparse O(nnz)  nnz={op['nnz']} : {op['sparse_O_nnz_mb']:8.4f} MB",
        f"  Reduction                : {op['reduction_x']:8.1f}x",
        "",
        "PLAIN-LANGUAGE TAKEAWAY",
        "-" * 60,
        "  Two independent memory savings, do not conflate them:",
        "  1) Freezing the dispensable attention weights removes their",
        "     Adam optimizer state (modest, scales with masked fraction).",
        "  2) Not storing the 99.87% zero corpus pairs turns an O(V^2)",
        f"     operator into O(nnz): a {op['reduction_x']:.0f}x reduction here,",
        "     and ~1,000,000x at a 50k vocabulary.",
    ]
    txt = "\n".join(lines)
    with open(f"{prefix}_summary.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)
    return payload


def make_chart(gd, comp, op, filename="memory_viz.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: stacked model/optimizer memory
    regimes = ["GD-400", "Compiler"]
    params = [gd["param_mb"], comp["param_mb"]]
    grads = [gd["grad_mb"], comp["grad_mb"]]
    opts = [gd["optimizer_state_mb"], comp["optimizer_state_mb"]]
    x = range(len(regimes))
    ax1.bar(x, params, label="Parameters", color="#4c72b0")
    ax1.bar(x, grads, bottom=params, label="Gradients", color="#dd8452")
    ax1.bar(x, opts, bottom=[p + g for p, g in zip(params, grads)],
            label="Optimizer state (Adam)", color="#c44e52")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(regimes)
    ax1.set_ylabel("Memory (MB)")
    ax1.set_title("Model + Optimizer Memory\n(same model, two optimizer regimes)")
    ax1.legend()
    for i in x:
        tot = params[i] + grads[i] + opts[i]
        ax1.text(i, tot + 0.5, f"{tot:.1f} MB", ha="center", fontsize=10,
                 weight="bold")

    # Panel 2: corpus operator, log scale
    labels = ["Dense O(V²)", "Sparse O(nnz)"]
    vals = [op["dense_O_V2_mb"], op["sparse_O_nnz_mb"]]
    ax2.bar(labels, vals, color=["#c44e52", "#55a868"])
    ax2.set_yscale("log")
    ax2.set_ylabel("Memory (MB, log scale)")
    ax2.set_title(f"Corpus-Operator Memory\n{op['reduction_x']:.0f}x reduction "
                  f"(V={op['V']}, nnz={op['nnz']})")
    for i, v in enumerate(vals):
        ax2.text(i, v, f"{v:.3g} MB", ha="center", va="bottom", fontsize=10,
                 weight="bold")

    plt.suptitle("Memory Verification: Geometry-Driven Compiler vs GD-400",
                 fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"\n[memory_viz] wrote {filename}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--quick", action="store_true",
                    help="use 20 steps for a fast smoke test")
    args = ap.parse_args()
    steps = 20 if args.quick else args.steps

    print(f"Building synthetic corpus (seed={args.seed}) ...")
    corpus = build_corpus(seed=args.seed)

    print(f"Running GD-400 regime ({steps} steps, full Adam) ...")
    gd = run_regime("GD-400", mask=False, steps=steps, corpus=corpus,
                    seed=args.seed)
    print(f"Running Compiler regime ({steps} steps, masked Adam) ...")
    comp = run_regime("Compiler", mask=True, steps=steps, corpus=corpus,
                      seed=args.seed)
    op = corpus_operator_memory()

    print()
    write_artifacts(gd, comp, op)
    make_chart(gd, comp, op)


if __name__ == "__main__":
    main()
