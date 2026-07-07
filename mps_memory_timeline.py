#!/usr/bin/env python3
"""
mps_memory_timeline.py  —  stage-by-stage forward/backward memory timeline
that WORKS ON APPLE SILICON (MPS), unlike torch.cuda.memory._record_memory_history
(which is CUDA-only and produces nothing on a Mac).

WHAT IT DOES
------------
Registers nn.Module forward hooks and tensor autograd hooks so that memory is
sampled at every layer boundary as the graph executes:

    forward:  embed -> block0.attn -> block0.ff -> ... -> head -> loss
    backward: loss -> head -> ... -> block0.ff -> block0.attn -> embed
    step:     optimizer.step()  (Adam moment allocation)

At each hook it records the live allocated memory, producing a trace you can
read exactly like the CUDA pytorch.org/memory_viz timeline: activations pile
up on the way forward, get freed on the way back, and the optimizer state
appears at the step.

BACKENDS
--------
  MPS  : torch.mps.current_allocated_memory()      (the real device counter)
  CUDA : torch.cuda.memory_allocated()             (exact)
  CPU  : sum of live tensor storages via gc        (fallback so it runs anywhere)

OUTPUTS
-------
  mps_timeline.png        the timeline chart (fwd rise / bwd fall / step)
  mps_timeline.csv        event, phase, layer, allocated_mb, delta_mb
  mps_timeline.json       full trace + peak analysis

RUN
---
  python mps_memory_timeline.py                 # compiler-style (masked) + GD
  python mps_memory_timeline.py --regime gd     # just the GD-400 style
  python mps_memory_timeline.py --regime compiler
"""

import argparse
import csv
import gc
import json
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- config (matches the paper's model) ----
D = 256
N_HEADS = 4
N_STU = 6
VOCAB = 1017
SEQ = 64
BATCH = 8


# ---------------- backend-aware live memory ----------------
def backend():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_BE = backend()


def live_mb(device_type):
    if device_type == "cuda":
        return torch.cuda.memory_allocated() / 1e6
    if device_type == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    # cpu: sum unique live tensor storages
    seen = set(); total = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.device.type == "cpu":
                    k = obj.untyped_storage().data_ptr()
                    if k in seen:
                        continue
                    seen.add(k)
                    total += obj.untyped_storage().nbytes()
            except Exception:
                continue
    return total / 1e6


# ---------------- model ----------------
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
        mask = torch.triu(torch.ones(S, S, device=h.device), diagonal=1).bool()
        sc = sc.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        att = F.softmax(sc, dim=-1) @ V
        return self.ln(h + self.op(att.transpose(1, 2).reshape(B, S, D)))


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
    def __init__(self):
        super().__init__()
        self.te = nn.Embedding(VOCAB, D); self.pe = nn.Embedding(512, D)
        self.blocks = nn.ModuleList([Block() for _ in range(N_STU)])
        self.ln_f = nn.LayerNorm(D); self.head = nn.Linear(D, VOCAB, bias=False)
        self.head.weight = self.te.weight
        nn.init.normal_(self.te.weight, std=0.02); nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x, y=None):
        h = self.te(x) + self.pe(torch.arange(x.shape[1], device=x.device))
        for b in self.blocks:
            h = b(h)
        logits = self.head(self.ln_f(h))
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        return logits, loss


# ---------------- the timeline recorder ----------------
class MemoryTimeline:
    def __init__(self, device_type):
        self.dev = device_type
        self.events = []       # list of (order, phase, label, allocated_mb)
        self._order = 0
        self._hooks = []

    def _rec(self, phase, label):
        self._order += 1
        self.events.append({
            "order": self._order,
            "phase": phase,
            "label": label,
            "allocated_mb": live_mb(self.dev),
        })

    def attach(self, model):
        """Forward hooks record memory AFTER each submodule runs (activation
        growth). Full-backward hooks record memory during the backward pass."""
        named = [("embed", model.te)]
        for i, blk in enumerate(model.blocks):
            named.append((f"block{i}.attn", blk.attn))
            named.append((f"block{i}.ff", blk.ff))
        named.append(("head", model.head))

        for name, mod in named:
            def fwd_hook(m, inp, out, nm=name):
                self._rec("forward", nm)
            self._hooks.append(mod.register_forward_hook(fwd_hook))

            # full backward hook fires when this module's grads are computed
            if hasattr(mod, "register_full_backward_hook"):
                def bwd_hook(m, gin, gout, nm=name):
                    self._rec("backward", nm)
                self._hooks.append(mod.register_full_backward_hook(bwd_hook))

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def mark(self, phase, label):
        self._rec(phase, label)

    # ---- analysis / output ----
    def _with_deltas(self):
        out = []
        prev = None
        for e in self.events:
            d = 0.0 if prev is None else e["allocated_mb"] - prev
            prev = e["allocated_mb"]
            out.append({**e, "delta_mb": d})
        return out

    def save(self, prefix="mps_timeline"):
        ev = self._with_deltas()
        peak = max(ev, key=lambda e: e["allocated_mb"]) if ev else None
        payload = {
            "backend": self.dev,
            "peak_mb": peak["allocated_mb"] if peak else 0.0,
            "peak_at": {"phase": peak["phase"], "label": peak["label"]} if peak else None,
            "events": ev,
        }
        with open(f"{prefix}.json", "w") as f:
            json.dump(payload, f, indent=2)
        with open(f"{prefix}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["order", "phase", "label", "allocated_mb", "delta_mb"])
            for e in ev:
                w.writerow([e["order"], e["phase"], e["label"],
                            f'{e["allocated_mb"]:.4f}', f'{e["delta_mb"]:+.4f}'])
        return payload

    def plot(self, prefix="mps_timeline", title_extra=""):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ev = self._with_deltas()
        x = [e["order"] for e in ev]
        y = [e["allocated_mb"] for e in ev]
        colors = {"forward": "#4c72b0", "backward": "#dd8452",
                  "loss": "#8172b3", "step": "#c44e52", "init": "#55a868"}
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(x, y, color="#333", lw=1.2, zorder=1)
        for e in ev:
            ax.scatter(e["order"], e["allocated_mb"],
                       color=colors.get(e["phase"], "#333"), s=40, zorder=2)
        # phase bands
        for ph, col in colors.items():
            pts = [e for e in ev if e["phase"] == ph]
            if pts:
                ax.scatter([p["order"] for p in pts], [p["allocated_mb"] for p in pts],
                           color=col, s=40, label=ph)
        peak = max(ev, key=lambda e: e["allocated_mb"])
        ax.axhline(peak["allocated_mb"], ls="--", color="red", alpha=0.5)
        ax.annotate(f'peak {peak["allocated_mb"]:.1f} MB @ {peak["phase"]}:{peak["label"]}',
                    (peak["order"], peak["allocated_mb"]),
                    textcoords="offset points", xytext=(5, 8), color="red", fontsize=9)
        ax.set_xlabel("execution order (forward -> loss -> backward -> step)")
        ax.set_ylabel(f"live allocated memory (MB, {self.dev})")
        ax.set_title(f"Forward/Backward Memory Timeline {title_extra}".strip())
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{prefix}.png", dpi=200)
        print(f"[timeline] wrote {prefix}.png")


# ---------------- driver ----------------
def mask_dispensable(model):
    n = 0
    for blk in model.blocks:
        for w in (blk.attn.WV, blk.attn.op):
            w.weight.requires_grad_(False); n += w.weight.numel()
    return n


def run_timeline(regime, seed=42):
    """regime is 'gd' or 'compiler'."""
    dev = backend()
    torch.manual_seed(seed)
    model = LM()
    masked = mask_dispensable(model) if regime == "compiler" else 0
    if dev != "cpu":
        model = model.to(dev)

    tl = MemoryTimeline(dev)
    tl.mark("init", "params_resident")
    tl.attach(model)

    gen = torch.Generator().manual_seed(seed + 1)
    ix = torch.randint(0, 10000, (BATCH,), generator=gen)
    base = torch.arange(SEQ + 1)
    x = torch.stack([(base[:-1] + int(i)) % VOCAB for i in ix])
    y = torch.stack([(base[1:] + int(i)) % VOCAB for i in ix])
    if dev != "cpu":
        x = x.to(dev); y = y.to(dev)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4, betas=(0.9, 0.95))

    # forward
    _, loss = model(x, y)
    tl.mark("loss", "cross_entropy")
    # backward
    opt.zero_grad()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # quiet the full-backward-hook notice
        loss.backward()
    tl.mark("backward", "grads_complete")
    # optimizer step (allocates Adam moments on first call)
    opt.step()
    tl.mark("step", "adam_step")

    tl.detach()
    label = (f"(compiler, masked={masked:,} params)"
             if regime == "compiler" else "(GD-400 style)")
    payload = tl.save(prefix=f"mps_timeline_{regime}")
    tl.plot(prefix=f"mps_timeline_{regime}", title_extra=label)

    print(f"\n=== {regime} timeline (backend={dev}) ===")
    print(f"  peak {payload['peak_mb']:.2f} MB at "
          f"{payload['peak_at']['phase']}:{payload['peak_at']['label']}")
    print(f"  events: {len(payload['events'])}  "
          f"(wrote mps_timeline_{regime}.png / .csv / .json)")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["gd", "compiler", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(f"backend detected: {backend()}")
    if backend() == "cpu":
        print("  (CPU fallback; identical code uses the MPS counter on "
              "Apple Silicon)")
    regimes = ["gd", "compiler"] if a.regime == "both" else [a.regime]
    for r in regimes:
        run_timeline(r, seed=a.seed)
