"""
mem_tracker.py  —  correct memory instrumentation for the
Geometry-Driven Compiler vs GD-400 comparison.

Two independent measurements are provided:

  (1) ANALYTIC (exact, deterministic, backend-independent):
      parameter bytes + optimizer-state bytes + gradient bytes, computed
      directly from the model. This is what the paper's "Memory Reduction"
      section claims and is fully reproducible.

  (2) EMPIRICAL PEAK (what the machine actually used):
      - CUDA:  torch.cuda.max_memory_allocated()   (tensor memory only)
      - MPS:   torch.mps.current_allocated_memory() (sampled; no true peak API)
      - CPU:   tracemalloc peak of Python/torch allocations
      RSS (psutil) is offered ONLY as a coarse whole-process fallback and is
      NOT used for the headline number, because it is dominated by fixed
      interpreter/library overhead and cannot isolate optimizer/activation
      memory.

Usage sketch (see patch_instructions.md):

    from mem_tracker import MemTracker
    mt = MemTracker()
    ...
    mt.sample_compiler("Spectral E0", model)             # analytic snapshot
    mt.sample_compiler("Basin settle", model, opt=None)  # opt optional
    ...
    mt.sample_gd(gd_step, gd, opt_gd)                     # per logged step
    ...
    mt.report()                 # prints a text table
    mt.plot("compiler_vs_gd_memory.png")
"""

import os
import torch

BYTES = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
         torch.float64: 8, torch.int64: 8, torch.int32: 4}


def _tensor_bytes(t):
    return t.numel() * BYTES.get(t.dtype, 4)


def _backend():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def analytic_param_bytes(model):
    """Bytes of trainable parameters (resident weights)."""
    return sum(_tensor_bytes(p) for p in model.parameters())


def analytic_grad_bytes(model):
    """Bytes of gradient buffers currently allocated (0 if never backward'd)."""
    return sum(_tensor_bytes(p.grad) for p in model.parameters()
               if p.grad is not None)


def analytic_optimizer_bytes(opt):
    """Bytes held in the optimizer state (Adam: exp_avg + exp_avg_sq)."""
    if opt is None:
        return 0
    total = 0
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += _tensor_bytes(v)
    return total


def analytic_trainable_bytes(model, requires_grad_only=True):
    """Param bytes counting only params that will actually receive Adam state.
    Masked/frozen params (requires_grad=False) are excluded — this is the
    O(nnz)-style saving from freezing WV/op."""
    return sum(_tensor_bytes(p) for p in model.parameters()
               if (p.requires_grad or not requires_grad_only))


class MemTracker:
    def __init__(self):
        self.compiler_history = []
        self.gd_history = []
        self.backend = _backend()

    # ---- empirical live tensor memory (works on every backend) ----
    def _empirical_mb(self):
        if self.backend == "cuda":
            # exact driver counter; peak since last reset
            return torch.cuda.max_memory_allocated() / 1e6
        if self.backend == "mps":
            return torch.mps.current_allocated_memory() / 1e6
        # cpu: tracemalloc cannot see PyTorch's C++ storage, so sum live
        # tensor storages directly via the garbage collector.
        import gc
        import warnings
        seen = set()
        total = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for obj in gc.get_objects():
                try:
                    if isinstance(obj, torch.Tensor) and obj.device.type == "cpu":
                        key = obj.untyped_storage().data_ptr()
                        if key in seen:
                            continue
                        seen.add(key)
                        total += obj.untyped_storage().nbytes()
                except Exception:
                    continue
        return total / 1e6

    def _snapshot(self, model, opt):
        pb = analytic_param_bytes(model)
        gb = analytic_grad_bytes(model)
        ob = analytic_optimizer_bytes(opt)
        return {
            "param_mb": pb / 1e6,
            "grad_mb": gb / 1e6,
            "opt_mb": ob / 1e6,
            "analytic_total_mb": (pb + gb + ob) / 1e6,
            "empirical_mb": self._empirical_mb(),
        }

    def sample_compiler(self, phase_name, model, opt=None):
        s = self._snapshot(model, opt)
        s["phase"] = phase_name
        self.compiler_history.append(s)

    def sample_gd(self, step, model, opt=None):
        s = self._snapshot(model, opt)
        s["step"] = step
        self.gd_history.append(s)

    # ---- reporting ----
    def report(self):
        print("\n" + "=" * 72)
        print(f"MEMORY REPORT  (backend={self.backend})")
        print("=" * 72)

        def _emit(title, hist, keyname):
            if not hist:
                print(f"  [{title}] no samples recorded")
                return
            print(f"\n  {title}")
            print(f"  {'label':<22}{'param':>9}{'grad':>9}{'opt':>9}"
                  f"{'analytic':>10}{'empirical':>11}")
            print("  " + "-" * 70)
            for s in hist:
                label = str(s.get(keyname))
                print(f"  {label:<22}{s['param_mb']:>9.2f}{s['grad_mb']:>9.2f}"
                      f"{s['opt_mb']:>9.2f}{s['analytic_total_mb']:>10.2f}"
                      f"{s['empirical_mb']:>11.2f}")

        _emit("GEOMETRY-DRIVEN COMPILER", self.compiler_history, "phase")
        _emit("GD-400 BASELINE", self.gd_history, "step")

        # peaks
        if self.compiler_history and self.gd_history:
            c_peak = max(s["analytic_total_mb"] for s in self.compiler_history)
            g_peak = max(s["analytic_total_mb"] for s in self.gd_history)
            print("\n  " + "-" * 70)
            print(f"  Analytic peak — compiler: {c_peak:.2f} MB   "
                  f"GD-400: {g_peak:.2f} MB   "
                  f"reduction: {g_peak / max(c_peak, 1e-9):.2f}x")
            ce_peak = max(s["empirical_mb"] for s in self.compiler_history)
            ge_peak = max(s["empirical_mb"] for s in self.gd_history)
            print(f"  Empirical peak — compiler: {ce_peak:.2f} MB   "
                  f"GD-400: {ge_peak:.2f} MB")
        print("=" * 72)

    def plot(self, filename="compiler_vs_gd_memory.png"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        def _panel(ax, hist, keyname, title, color):
            if not hist:
                ax.set_title(title + " (no data)")
                return
            labels = [str(s[keyname]) for s in hist]
            x = range(len(labels))
            ax.plot(x, [s["analytic_total_mb"] for s in hist], marker="o",
                    color=color, lw=2, label="Analytic (param+opt+grad)")
            ax.plot(x, [s["param_mb"] for s in hist], marker="^",
                    color=color, lw=1, ls="--", alpha=0.6, label="Params only")
            ax.plot(x, [s["opt_mb"] for s in hist], marker="v",
                    color=color, lw=1, ls=":", alpha=0.8, label="Optimizer state")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        _panel(ax1, self.compiler_history, "phase",
               "Geometry-Driven Compiler", "#1f77b4")
        _panel(ax2, self.gd_history, "step", "GD-400 Baseline", "#ff7f0e")
        ax1.set_ylabel("Memory (MB)")
        plt.suptitle(f"Memory Profile (backend={self.backend}): "
                     "Geometric Compiler vs GD-400",
                     fontsize=14, weight="bold")
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        print(f"[mem_tracker] saved {filename}")
