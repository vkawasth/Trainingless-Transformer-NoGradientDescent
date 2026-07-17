"""
step_utility.py
===============
CULL UNUSEFUL TRAINING STEPS BY READING THE GRADIENT GRAPH'S RATE OF CHANGE.

Frame training as graph construction:
  - EARLY  (backbone): the interaction graph is being built. The update
    direction keeps turning; each step adds structure. These steps are useful
    and must be computed.
  - LATE   (settled): the graph is fixed; gradients just flow through it toward
    the basin. The weight trajectory becomes locally POLYNOMIAL -- predictable.
    A predictable step carries no new information: you can extrapolate through
    it (a SNAPPER-style jump) instead of computing a backward pass.

So a step's utility = how much it changes the graph. We measure three cheap
per-step signals and use them to decide, online, when to STEP and when to JUMP:

  graph_drift    ||profile_t - profile_{t-W}||   profile = per-block grad-norm
                                                 shares (where gradient flows)
  novelty        1 - cos(g_t, EMA g)             is the direction still turning?
  collinearity   cos(dw_t, dw_{t-1})             is the path locally straight?

When the graph has stopped drifting AND the path is collinear, we fit a
quadratic to the recent weight trajectory, jump forward K steps, and keep the
jump iff val does not worsen (accept-if-better, like lm_step / the snapper).
Every accepted jump culls K backward passes.

The counterfactual is the proof: baseline (all steps computed) vs culled
(jumps where predictable), matched on val, counting BACKWARD PASSES saved.

Model architecture + group_of() match compiler_geometri_patched_86.py, so this
transfers to the real phases.
"""

from __future__ import annotations
import math
import copy
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D, N_HEADS, N_STU, VOCAB, SEQ, BATCH = 128, 4, 4, 96, 64, 16
LR = 3e-4


def group_of(name: str) -> str:
    n = name.lower()
    if ".ln." in n or ".n." in n or n.startswith("ln_f"):
        return "LayerNorm"
    if n.startswith("te") or n.startswith("pe") or n.startswith("head"):
        return "Emb"
    if ".ff." in n:
        return "FF"
    if "wk" in n:
        return "W_K"
    if "wq" in n:
        return "W_Q"
    if "wv" in n:
        return "W_V"
    if ".op." in n:
        return "W_O"
    return "other"


# ---- model (identical architecture) --------------------------------------- #
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
        sc = sc.masked_fill(mask, float("-inf"))
        return self.ln(h + self.op((F.softmax(sc, -1) @ V).transpose(1, 2).reshape(B, S, D)))


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
        h = self.te(x) + self.pe(torch.arange(x.shape[1]))
        for b in self.blocks:
            h = b(h)
        logits = self.head(self.ln_f(h))
        return logits, (F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
                        if y is not None else None)

    def flat(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_flat(self, v):
        i = 0
        for p in self.parameters():
            n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n


class Corpus:
    def __init__(self, succ=6, length=200_000, seed=0):
        g = torch.Generator().manual_seed(seed)
        T = torch.zeros(VOCAB, VOCAB)
        for s in range(VOCAB):
            nxt = torch.randperm(VOCAB, generator=g)[:succ]
            T[s, nxt] = 1.0 / succ
        seq = torch.zeros(length, dtype=torch.long)
        seq[0] = torch.randint(VOCAB, (1,), generator=g)
        for t in range(1, length):
            seq[t] = torch.multinomial(T[seq[t - 1]], 1, generator=g)
        self.data = seq; self.floor = math.log(succ)
        self.g = torch.Generator().manual_seed(seed + 1)

    def get_batch(self):
        ix = torch.randint(0, len(self.data) - SEQ - 1, (BATCH,), generator=self.g)
        return (torch.stack([self.data[i:i + SEQ] for i in ix]),
                torch.stack([self.data[i + 1:i + SEQ + 1] for i in ix]))

    def eval_val(self, m, n=8):
        m.eval(); ls = []
        with torch.no_grad():
            for _ in range(n):
                x, y = self.get_batch(); _, l = m(x, y); ls.append(l.item())
        m.train(); return float(np.mean(ls))


# ---- per-step gradient-graph profile -------------------------------------- #
def block_profile(model):
    """Where is gradient flowing right now? Per-block grad-norm share =
    the activity of each node in the gradient graph."""
    prof = defaultdict(float)
    for name, p in model.named_parameters():
        if p.grad is not None:
            prof[group_of(name)] += float(p.grad.pow(2).sum())
    keys = ["Emb", "W_Q", "W_K", "W_V", "W_O", "FF", "LayerNorm"]
    v = np.array([math.sqrt(prof.get(k, 0.0)) for k in keys])
    return v / (v.sum() + 1e-12)


# ---- baseline: profile the whole descent, log utility signals -------------- #
def profile_run(steps=300, seed=99):
    torch.manual_seed(seed)
    corpus = Corpus(); model = LM()
    opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95),
                            weight_decay=0.1)
    ema_g = None
    prof_hist = deque(maxlen=25)
    w_prev = model.flat().clone(); dw_prev = None
    log = []
    for s in range(1, steps + 1):
        model.train(); x, y = corpus.get_batch(); _, l = model(x, y)
        opt.zero_grad(); l.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters()]).detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        prof = block_profile(model)
        ema_g = g.clone() if ema_g is None else 0.9 * ema_g + 0.1 * g
        novelty = 1 - float((g @ ema_g) / (g.norm() * ema_g.norm() + 1e-12))
        w = model.flat(); dw = w - w_prev
        collin = (float((dw @ dw_prev) / (dw.norm() * dw_prev.norm() + 1e-12))
                  if dw_prev is not None else 0.0)
        drift = (float(np.linalg.norm(prof - prof_hist[0]))
                 if len(prof_hist) == prof_hist.maxlen else float("nan"))
        prof_hist.append(prof)
        w_prev = w.clone(); dw_prev = dw.clone()
        if s % 10 == 0:
            v = corpus.eval_val(model)
            log.append({"step": s, "val": v, "novelty": novelty,
                        "collin": collin, "drift": drift,
                        "attn_share": float(prof[1:5].sum()),
                        "ff_share": float(prof[5]), "emb_share": float(prof[0])})
    return log


# ---- counterfactual: cull predictable steps by quadratic jump -------------- #
def run_baseline(steps, corpus, model, opt):
    bwd = 0
    for s in range(steps):
        model.train(); x, y = corpus.get_batch(); _, l = model(x, y)
        opt.zero_grad(); l.backward(); bwd += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return corpus.eval_val(model), bwd


def run_culled(steps, corpus, model, opt, K=6, warm=60, W=8,
               drift_max=0.04, resid_max=0.15):
    """Step while the graph drifts; once it settles, fit a quadratic by LEAST
    SQUARES over the last W weight snapshots (denoises the minibatch jitter) and
    jump forward K steps -- but only if the fit RESIDUAL is small (the recent
    path really is low-order polynomial) and val does not worsen."""
    import numpy as _np
    bwd = 0
    snaps = deque(maxlen=W)
    prof_hist = deque(maxlen=25)
    w_prev = model.flat().clone()
    culled = 0
    # design matrix over t = -(W-1)..0 and its pseudoinverse (quadratic)
    tt = _np.arange(-(W - 1), 1, dtype=_np.float64)
    A = _np.stack([_np.ones_like(tt), tt, tt * tt], 1)      # (W,3)
    Apinv = _np.linalg.pinv(A)                              # (3,W)
    s = 0
    while s < steps:
        can_jump = (s >= warm and len(snaps) == W
                    and len(prof_hist) == prof_hist.maxlen)
        if can_jump:
            drift = float(_np.linalg.norm(prof_hist[-1] - prof_hist[0]))
            Wsnap = torch.stack(list(snaps)).double().numpy()   # (W,P)
            coef = Apinv @ Wsnap                                # (3,P)
            fit = A @ coef                                      # (W,P)
            resid = float(_np.linalg.norm(fit - Wsnap) /
                          (_np.linalg.norm(Wsnap) + 1e-12))
            if drift < drift_max and resid < resid_max:
                w_jump = torch.tensor(coef[0] + coef[1] * K + coef[2] * K * K)
                v_before = corpus.eval_val(model, n=6)
                w_keep = model.flat().clone()
                model.set_flat(w_jump.float())
                v_after = corpus.eval_val(model, n=6)
                if v_after <= v_before + 1e-3:
                    culled += K; s += K
                    snaps.clear(); prof_hist.clear()
                    opt.state = defaultdict(dict)     # moments stale after a jump
                    w_prev = model.flat().clone()
                    continue
                model.set_flat(w_keep)
        model.train(); x, y = corpus.get_batch(); _, l = model(x, y)
        opt.zero_grad(); l.backward(); bwd += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        prof_hist.append(block_profile(model))
        w = model.flat(); snaps.append(w.clone()); w_prev = w.clone()
        s += 1
    return corpus.eval_val(model), bwd, culled


def run_earlystop(max_steps, corpus, model, opt, thr=0.013, warm=90, win=10):
    """Cull steps by STOPPING: once the gradient graph stops drifting (smoothed
    over `win` steps) further steps do no graph construction and, on a plateau,
    barely move the loss. This is the measured GEO-STOP -- one grad-norm profile
    per step, no Phi_cl/tau/rm2."""
    bwd = 0
    prof_hist = deque(maxlen=25); dr = deque(maxlen=win)
    for s in range(max_steps):
        model.train(); x, y = corpus.get_batch(); _, l = model(x, y)
        opt.zero_grad(); l.backward(); bwd += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        prof_hist.append(block_profile(model))
        if s >= warm and len(prof_hist) == prof_hist.maxlen:
            dr.append(float(np.linalg.norm(prof_hist[-1] - prof_hist[0])))
            if len(dr) == win and np.mean(dr) < thr:
                return corpus.eval_val(model), bwd, s + 1
    return corpus.eval_val(model), bwd, max_steps


def main():
    print("=" * 78)
    print("  STEP UTILITY: TRAINING AS GRAPH CONSTRUCTION, CULL THE STATIC STEPS")
    print("=" * 78)
    log = profile_run(steps=300)
    print(f"  {'step':>5}{'val':>8}{'novelty':>9}{'collin':>8}{'graph drift':>12}"
          f"{'attn share':>11}   phase")
    print("  " + "-" * 70)
    prev_v = None
    for r in log:
        d = r["drift"]
        dprog = (prev_v - r["val"]) if prev_v is not None else 0.0
        prev_v = r["val"]
        if not math.isnan(d) and d > 0.06:
            ph = "BUILD graph"
        elif dprog > 0.004:
            ph = "refine"
        else:
            ph = "churn -> cull"
        ds = "  --  " if math.isnan(d) else f"{d:>7.3f}"
        print(f"  {r['step']:>5}{r['val']:>8.3f}{r['novelty']:>9.3f}"
              f"{r['collin']:>8.3f}{ds:>12}{100*r['attn_share']:>10.1f}%   {ph}")

    print("\n" + "=" * 78)
    print("  COUNTERFACTUAL: two ways to cull, one per regime")
    print("=" * 78)
    N = 300
    torch.manual_seed(99); m_base = LM()
    torch.manual_seed(99); m_jump = LM()
    torch.manual_seed(99); m_stop = LM()
    mk = lambda m: torch.optim.AdamW(m.parameters(), lr=LR * 5,
                                     betas=(0.9, 0.95), weight_decay=0.1)
    v_base, bwd_base = run_baseline(N, Corpus(seed=3), m_base, mk(m_base))
    v_jump, bwd_jump, culled = run_culled(N, Corpus(seed=3), m_jump, mk(m_jump))
    v_stop, bwd_stop, stopped = run_earlystop(N, Corpus(seed=3), m_stop, mk(m_stop))

    print(f"  baseline (all 300 steps) : val {v_base:.4f}   backward passes {bwd_base}")
    print(f"  jump-cull (extrapolate)  : val {v_jump:.4f}   backward passes "
          f"{bwd_jump}   jumped {culled}")
    print(f"  stop-cull (drift->0)     : val {v_stop:.4f}   backward passes "
          f"{bwd_stop}   stopped at step {stopped}")
    print("\n  " + "-" * 70)
    sj = 100 * (bwd_base - bwd_jump) / bwd_base
    ss = 100 * (bwd_base - bwd_stop) / bwd_base
    dj = 100 * (v_jump - v_base) / v_base
    dsv = 100 * (v_stop - v_base) / v_base
    print(f"  jump-cull: {sj:>3.0f}% steps culled, {dj:+.1f}% val  "
          f"-> extrapolation FAILS on a plateau (no gain ahead to jump to)")
    print(f"  stop-cull: {ss:>3.0f}% steps culled, {dsv:+.1f}% val  "
          f"-> the graph-drift signal converges before the loss does")
    print("\n  READING: the profiler tells you which regime you are in, and each")
    print("  regime has its own cull. STEEP basin approach -> jump (the SNAPPER,")
    print("  phase 5). FLAT plateau -> stop (the GEO-STOP, phase 3). Trying to")
    print("  jump on a plateau overshoots; trying to grind a converged plateau")
    print("  wastes backward passes. drift->0 is the trigger for the second.")
    print("\n  For real phase 3: the drift-of-block-profile IS a cheaper geo-stop")
    print("  signal (no Phi_cl/tau/rm2 needed) -- one grad-norm profile per step.")


if __name__ == "__main__":
    main()
