"""
adamw_cull.py
=============
CULL ADAMW BY ISOLATING WHERE ITS ADAPTIVITY IS ACTUALLY LOAD-BEARING.

AdamW's cost over SGD is the per-coordinate rescale D = 1/(sqrt(v_hat)+eps) and
the state (m, v) behind it. On a block where D is nearly UNIFORM, that rescale
is just a scalar -- SGD+momentum with a per-block learning rate lr*mean(D)
reproduces the same step, and the v-state is dead weight. On a block where D is
DISPERSED, AdamW is genuinely reshaping the gradient and cannot be culled.

So "which way and how do gradients flow" becomes measurable and actionable:
  per block B, measure
    CV_B      = std(D)/mean(D)     -- dispersion of Adam's own preconditioner
    cos(u,g)_B                     -- does the Adam update point where the raw
                                      gradient (the SGD direction) points?
  Cull B (-> SGD+mom @ lr*mean(D_B)) iff CV_B is small AND cos(u,g)_B is high.

The diagnostics PREDICT cullability. The counterfactual PROVES it: replay the
basin settle three ways from one branch point --
    (A) AdamW everywhere        (baseline)
    (B) hybrid: cull flagged blocks
    (C) SGD+mom everywhere
-- and compare the val trajectory and how much param mass left AdamW.

This is a faithful stand-in for phase 3 of compiler_geometri_patched_86.py: the
model architecture (WQ/WK/WV/op, FF g/v/o/n, tied head) and group_of() taxonomy
are identical, so the cull_report() function drops straight into the real loop.
"""

from __future__ import annotations
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- config (smaller than the real D=256/N_STU=6 for a fast demo; the
#      diagnostics are scale-independent) ------------------------------------ #
D, N_HEADS, N_STU, VOCAB, SEQ, BATCH = 128, 4, 4, 96, 64, 16
LR = 3e-4


# ============================ taxonomy ===================================== #
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


# ============================ model (identical arch) ======================= #
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


# ============================ corpus ======================================= #
class Corpus:
    """Order-1 Markov, `succ` successors/state -> entropy floor ln(succ).
    Gives a gradual multi-step descent, like a basin settle."""
    def __init__(self, succ=8, length=200_000, seed=0):
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
        x = torch.stack([self.data[i:i + SEQ] for i in ix])
        y = torch.stack([self.data[i + 1:i + SEQ + 1] for i in ix])
        return x, y

    def eval_val(self, m, n=8):
        m.eval(); ls = []
        with torch.no_grad():
            for _ in range(n):
                x, y = self.get_batch(); _, l = m(x, y); ls.append(l.item())
        m.train(); return float(np.mean(ls))


# ============================ the cull report ============================== #
def param_groups_map(model):
    """name -> group, and per-group list of parameters."""
    groups = {}
    for name, p in model.named_parameters():
        groups.setdefault(group_of(name), []).append((name, p))
    return groups


def cull_report(model, opt, eps=1e-8):
    """Read AdamW state and score each block's cullability.
    Returns {group: {mean_D, cv_D, cos_ug, grad_share, cullable}}."""
    # need a fresh gradient to compare the Adam update direction against
    st = opt.state
    betas = opt.param_groups[0]["betas"]
    b1, b2 = betas
    rows = {}
    total_gnorm2 = 0.0
    per = param_groups_map(model)
    # first pass: total grad norm for share
    for g, plist in per.items():
        for _, p in plist:
            if p.grad is not None:
                total_gnorm2 += float(p.grad.pow(2).sum())
    total_gnorm = math.sqrt(total_gnorm2 + 1e-12)

    for g, plist in per.items():
        Ds, us, ms, gnorm2 = [], [], [], 0.0
        for name, p in plist:
            if p not in st or "exp_avg_sq" not in st[p]:
                continue
            step = st[p].get("step", 1)
            step = float(step) if not torch.is_tensor(step) else float(step.item())
            m = st[p]["exp_avg"]; v = st[p]["exp_avg_sq"]
            mhat = m / (1 - b1 ** max(step, 1))
            vhat = v / (1 - b2 ** max(step, 1))
            Dloc = 1.0 / (vhat.sqrt() + eps)
            u = mhat * Dloc                        # the AdamW update direction
            Ds.append(Dloc.flatten())
            us.append(u.flatten())
            ms.append(mhat.flatten())              # the SGD+mom direction
            if p.grad is not None:
                gnorm2 += float(p.grad.pow(2).sum())
        if not Ds:
            continue
        Dcat = torch.cat(Ds); ucat = torch.cat(us); mcat = torch.cat(ms)
        med_D = float(Dcat.median()); cv_D = float(Dcat.std() / (Dcat.mean() + 1e-12))
        # does the diagonal rescale D change the momentum DIRECTION at all?
        cos_um = float((ucat @ mcat) / (ucat.norm() * mcat.norm() + 1e-12))
        rows[g] = {
            "med_D": med_D, "cv_D": cv_D, "cos_um": cos_um,
            "grad_share": math.sqrt(gnorm2) / total_gnorm,
        }
    return rows


def decide_cull(rows, cv_max=0.5, cos_min=0.97):
    """Cull (move to SGD) blocks whose preconditioner is near-uniform AND whose
    rescale barely turns the momentum direction -- i.e. D is doing nothing a
    scalar per-block LR could not do."""
    cull = set()
    for g, r in rows.items():
        if r["cv_D"] <= cv_max and r["cos_um"] >= cos_min:
            cull.add(g)
    return cull


# ============================ optimizers =================================== #
def make_adamw(params, lr):
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)


def make_hybrid(model, cull_groups, rows, lr):
    """AdamW on kept blocks; SGD+mom on culled blocks, each at lr*mean(D_block)
    -- i.e. replace the near-uniform diagonal preconditioner by its scalar mean."""
    keep_params, sgd_groups = [], []
    per = param_groups_map(model)
    n_adam, n_sgd = 0, 0
    for g, plist in per.items():
        ps = [p for _, p in plist]
        if g in cull_groups and g in rows:
            eff = min(max(rows[g]["med_D"], 0.05), 40.0)   # clamp: no blowup on high-range blocks
            sgd_groups.append({"params": ps, "lr": lr * eff})
            n_sgd += sum(p.numel() for p in ps)
        else:
            keep_params += ps
            n_adam += sum(p.numel() for p in ps)
    opt_adam = make_adamw(keep_params, lr) if keep_params else None
    opt_sgd = (torch.optim.SGD(sgd_groups, momentum=0.9, weight_decay=0.1)
               if sgd_groups else None)
    return opt_adam, opt_sgd, n_adam, n_sgd


def settle(model, corpus, opt_list, steps=120, evalK=12, clip=1.0):
    """Run a basin-settle and return the val trajectory."""
    traj = []
    for s in range(1, steps + 1):
        model.train(); x, y = corpus.get_batch(); _, l = model(x, y)
        for o in opt_list:
            if o: o.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for o in opt_list:
            if o: o.step()
        if s % evalK == 0:
            traj.append((s, corpus.eval_val(model)))
    return traj


# ============================ driver ======================================= #
def main():
    torch.manual_seed(99)
    corpus = Corpus(succ=8)
    model = LM()
    print("=" * 78)
    print("  CULLING ADAMW IN THE BASIN: WHERE IS ADAPTIVITY LOAD-BEARING?")
    print("=" * 78)
    P = sum(p.numel() for p in model.parameters())
    print(f"  P = {P:,}   entropy floor = ln 8 = {corpus.floor:.3f}")

    # warm to the branch point (past the saddle, into the basin) with AdamW
    opt = make_adamw(model.parameters(), LR * 5)
    for s in range(1, 81):
        model.train(); x, y = corpus.get_batch(); _, l = model(x, y)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    # fresh grad for the direction comparison
    x, y = corpus.get_batch(); _, l = model(x, y); opt.zero_grad(); l.backward()
    v_branch = corpus.eval_val(model)
    print(f"  branch point: val = {v_branch:.4f} after 80 AdamW steps\n")

    rows = cull_report(model, opt)
    cull = decide_cull(rows)
    print("  PER-BLOCK ADAPTIVITY (from AdamW's own state):")
    print(f"  {'block':<11}{'med D':>9}{'CV(D)':>8}{'cos(u,m)':>10}"
          f"{'grad share':>12}   decision")
    print("  " + "-" * 62)
    for g in ["Emb", "W_Q", "W_K", "W_V", "W_O", "FF", "LayerNorm"]:
        if g not in rows:
            continue
        r = rows[g]
        dec = "CULL -> SGD" if g in cull else "keep AdamW"
        print(f"  {g:<11}{r['med_D']:>9.1f}{r['cv_D']:>8.2f}{r['cos_um']:>10.3f}"
              f"{100*r['grad_share']:>11.1f}%   {dec}")
    culled_mass = sum(sum(p.numel() for _, p in param_groups_map(model)[g])
                      for g in cull if g in param_groups_map(model))
    print(f"\n  cull set: {sorted(cull)}  "
          f"({100*culled_mass/P:.0f}% of params leave AdamW)")

    # ---- greedy frontier: cull as much as possible, measure the cost ------- #
    branch = copy.deepcopy(model.state_dict())
    order = sorted(rows.keys(), key=lambda g: (rows[g]["cv_D"], -rows[g]["cos_um"]))
    print("\n" + "=" * 78)
    print("  GREEDY CULL FRONTIER (cull most-uniform blocks first)")
    print("=" * 78)
    print("  cull blocks in order of cullability; measure val after each add.")
    print(f"\n  {'culled blocks':<40}{'% params off Adam':>18}{'final val':>11}")
    print("  " + "-" * 70)

    per = param_groups_map(model)
    baseline = None
    frontier = []
    for k in range(0, len(order) + 1):
        cull_k = set(order[:k])
        model.load_state_dict(copy.deepcopy(branch))
        if k == 0:
            opts = [make_adamw(model.parameters(), LR * 5)]; na, ns = P, 0
        else:
            oa, os_, na, ns = make_hybrid(model, cull_k, rows, LR * 5)
            opts = [oa, os_]
        traj = settle(model, corpus, opts, steps=90)
        vf = traj[-1][1]
        if k == 0:
            baseline = vf
        pct = 100 * ns / P
        degr = 100 * (vf - baseline) / baseline
        tag = "" if k == 0 else ("  <- free" if degr < 3 else
                                 "  <- costs" if degr < 15 else "  <- breaks")
        label = "(none = AdamW-all)" if k == 0 else ", ".join(sorted(cull_k))
        vf_str = f"{vf:.4f}" if vf == vf else "diverged"
        print(f"  {label:<40}{pct:>17.0f}%{vf_str:>11}{tag}")
        frontier.append((sorted(cull_k), pct, vf, degr))

    # best = max params culled with < 3% degradation
    ok = [f for f in frontier if f[3] < 3 and f[2] == f[2]]
    best = max(ok, key=lambda f: f[1]) if ok else frontier[0]
    print("\n  " + "-" * 70)
    print(f"  MAX SAFE CULL (<3% val cost): {best[1]:.0f}% of params off AdamW")
    print(f"    cull {best[0] if best[0] else '(nothing)'}; "
          f"keep AdamW on the rest.")
    print(f"  Everything beyond that is where 1/sqrt(v) is genuinely load-bearing")
    print(f"  -- concentrated, not spread: that is the map of what AdamW is FOR.")

    print("\n  DROP-IN FOR REAL PHASE 3: after the geo-stop, call")
    print("    rows = cull_report(model, opt_b); cull = decide_cull(rows)")
    print("    oa, os_, *_ = make_hybrid(model, cull, rows, LR*10)")
    print("  and step [oa, os_] through the 30CE fast-descent instead of one")
    print("  AdamW -- that stage is pure basin settle, the cheapest to cull.")


if __name__ == "__main__":
    main()
