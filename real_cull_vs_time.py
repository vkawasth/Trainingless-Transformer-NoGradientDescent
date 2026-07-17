import math
from collections import deque, defaultdict
import numpy as np
import torch

g_ = {}
src = open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
cut = src.find("# ── PHASE 1")
exec(src[:cut], g_)
model = g_["model"]; get_batch = g_["get_batch"]; eval_val = g_["eval_val"]; LR = g_["LR"]
GROUPS = ["Emb", "W_Q", "W_K", "W_V", "W_O", "FF", "LayerNorm"]

def group_of(name):
    n = name.lower()
    if ".ln." in n or ".n." in n or n.startswith("ln_f"): return "LayerNorm"
    if n.startswith("te") or n.startswith("pe") or n.startswith("head"): return "Emb"
    if ".ff." in n: return "FF"
    if "wk" in n: return "W_K"
    if "wq" in n: return "W_Q"
    if "wv" in n: return "W_V"
    if ".op." in n: return "W_O"
    return "other"

def block_profile(model):
    prof = defaultdict(float)
    for name, p in model.named_parameters():
        if p.grad is not None:
            prof[group_of(name)] += float(p.grad.pow(2).sum())
    v = np.array([math.sqrt(prof.get(k, 0.0)) for k in GROUPS])
    return v / (v.sum() + 1e-12)

def cull_report(model, opt, eps=1e-8):
    st = opt.state; b1, b2 = opt.param_groups[0]["betas"]
    per = defaultdict(list)
    for name, p in model.named_parameters():
        per[group_of(name)].append(p)
    rows = {}
    for g, ps in per.items():
        if g not in GROUPS: continue
        Ds, us, ms = [], [], []
        for p in ps:
            if p not in st or "exp_avg_sq" not in st[p]: continue
            step = st[p].get("step", 1); step = float(step.item()) if torch.is_tensor(step) else float(step)
            m = st[p]["exp_avg"]; v = st[p]["exp_avg_sq"]
            mhat = m / (1 - b1 ** max(step, 1)); vhat = v / (1 - b2 ** max(step, 1))
            Dl = 1.0 / (vhat.sqrt() + eps)
            Ds.append(Dl.flatten()); us.append((mhat * Dl).flatten()); ms.append(mhat.flatten())
        if not Ds: continue
        Dc = torch.cat(Ds); uc = torch.cat(us); mc = torch.cat(ms)
        rows[g] = {"cv": float(Dc.std() / (Dc.mean() + 1e-12)),
                   "cos": float((uc @ mc) / (uc.norm() * mc.norm() + 1e-12)),
                   "n": sum(p.numel() for p in ps)}
    return rows

def cullable_frac(rows, P, cv_max=0.5, cos_min=0.97):
    culled = [g for g, r in rows.items() if r["cv"] <= cv_max and r["cos"] >= cos_min]
    n = sum(rows[g]["n"] for g in culled)
    return 100 * n / P, culled

def main():
    P = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95), weight_decay=0.1)
    prof_hist = deque(maxlen=20)
    checkpoints = [8, 20, 40, 70, 110, 160, 220]
    print("=" * 84)
    print("  UNIFICATION TEST (real model): does cullability track graph drift?")
    print("=" * 84)
    print(f"  P = {P:,}")
    cols = ["W_Q", "W_K", "W_V", "W_O", "FF", "Emb"]
    print(f"\n  {'step':>5}{'val':>8}{'drift':>8}   " +
          "".join(f"{'CV_'+g:>8}" for g in cols) + f"{'cull%':>7}   culled")
    print("  " + "-" * 82)
    step = 0
    for tgt in checkpoints:
        while step < tgt:
            model.train(); x, y = get_batch(); _, l = model(x, y)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            prof_hist.append(block_profile(model)); step += 1
        model.train(); x, y = get_batch(); _, l = model(x, y); opt.zero_grad(); l.backward()
        drift = (float(np.linalg.norm(prof_hist[-1] - prof_hist[0]))
                 if len(prof_hist) == prof_hist.maxlen else float("nan"))
        rows = cull_report(model, opt)
        frac, culled = cullable_frac(rows, P)
        v = eval_val(model, n=8)
        cv = {g: rows.get(g, {"cv": float("nan")})["cv"] for g in GROUPS}
        ds = "  --  " if math.isnan(drift) else f"{drift:>6.3f}"
        print(f"  {step:>5}{v:>8.4f}{ds:>8}   " +
              "".join(f"{cv[g]:>8.2f}" for g in cols) +
              f"{frac:>6.0f}%   {','.join(culled)}")
    print("\n  Rising cull% as drift falls => spatial and temporal culls are one clock.")

if __name__ == "__main__":
    main()
