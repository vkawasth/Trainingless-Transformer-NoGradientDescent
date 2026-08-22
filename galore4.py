"""PER-MATRIX GALORE FOR PHASE 3, WITH MEASURED PARAMETERS.

Everything here is set by measurement rather than default.

  RANK, per role.  E_8 for attention is 0.87-0.91; for FF it is 0.72-0.82 and
  needs r=16-24 for the same energy. A single r would over-serve attention and
  starve FF, which is the largest block.
        r = 8  for ATTN (WQ, WK, WV, op)
        r = 16 for FF
  REFRESH every 3 steps.  Subspace overlap between checkpoints ~60 steps apart
  is 0.215 mean -- FF worst at 0.08-0.30, attention 0.20-0.37. GaLore's stock
  50-100 step refresh assumes persistence this landscape does not have.
  TWO-SIDED PROJECTION.  P^T G Q with P in R^{m x r}, Q in R^{n x r}, so the
  projector costs r(m+n) rather than r*mn. For the FF block that is 3,072 floats
  against 4,718,592 for a flattened frame -- a factor of 1,500, and the reason
  my flat construction could never save memory ( PK > 2P whenever K > 2 ).

WHAT IS COMPARED
  adamw       the reference
  galore      moments on the projected gradient, refreshed every --refresh
  galore-50   the same with GaLore's stock refresh, to isolate the schedule

Matrices only. 1-D parameters (LayerNorm gains, biases) keep plain AdamW --
they are 3,328 of 1,182,080 parameters, so there is nothing to save there and
projecting a vector has no two-sided structure.

DIAGNOSTICS, since they called every previous failure before the loss did:
  capture   fraction of the gradient's energy inside the projector
  flip      sign-flip rate; pure sign hit 0.54 and the hybrid 0.48 while AdamW
            sat at 0.12
"""
import argparse, io, contextlib, subprocess, sys, math, json
import numpy as np, torch


class GaLore:
    def __init__(self, model, lr, betas=(0.9, 0.95), eps=1e-8, wd=0.1,
                 r_attn=8, r_ff=16, refresh=3, scale=True):
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, wd
        self.refresh, self.t, self.scale = refresh, 0, scale
        self.mats, self.vecs = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (self.mats if p.dim() == 2 else self.vecs).append((n, p))
        self.r = {n: (r_ff if ".ff." in n else r_attn) for n, _ in self.mats}
        self.P, self.Qm, self.m, self.v = {}, {}, {}, {}
        self.vm = {n: torch.zeros_like(p) for n, p in self.vecs}
        self.vv = {n: torch.zeros_like(p) for n, p in self.vecs}
        self.stats = {"cap": [], "flip": [], "prev": None}

    def zero_grad(self):
        for _, p in self.mats + self.vecs:
            p.grad = None

    def step(self):
        self.t += 1
        cap_n = cap_d = 0.0
        upd = []
        for n, p in self.mats:
            if p.grad is None:
                continue
            # DECOUPLED decay. Folding wd*W into the projected gradient was a
            # bug: ||0.1 W|| / ||g|| measured 1479x for attention and 13.6x for
            # FF, and W is full-rank, so SVDing (g + wd W) projects the WEIGHT
            # matrix. E_8 fell from 0.776 to 0.349 as a result. GaLore uses
            # decoupled decay for exactly this reason.
            Gr = p.grad
            r = min(self.r[n], min(Gr.shape))
            if self.t % self.refresh == 0 or n not in self.P:
                U, S, Vt = torch.linalg.svd(Gr, full_matrices=False)
                self.P[n], self.Qm[n] = U[:, :r].clone(), Vt[:r].T.clone()
                if n in self.m and self.m[n].shape != (r, r):
                    self.m.pop(n); self.v.pop(n)
            Pm, Qm = self.P[n], self.Qm[n]
            C = Pm.T @ Gr @ Qm                      # r x r
            cap_n += float((C * C).sum()); cap_d += float((Gr * Gr).sum())
            if n not in self.m:
                self.m[n] = torch.zeros_like(C); self.v[n] = torch.zeros_like(C)
            self.m[n] = self.b1 * self.m[n] + (1 - self.b1) * C
            self.v[n] = self.b2 * self.v[n] + (1 - self.b2) * C * C
            mh = self.m[n] / (1 - self.b1 ** self.t)
            vh = self.v[n] / (1 - self.b2 ** self.t)
            core = mh / (vh.sqrt() + self.eps)
            step = -self.lr * (Pm @ core @ Qm.T)
            # SCALE MATCHING. P core Q^T has norm ~ ||core||_F ~ r, while an
            # AdamW step on the same matrix has all entries O(1) and norm
            # ~ sqrt(mn). At r=8, m=n=96 that is a 12x smaller step, so an
            # unscaled comparison measures learning rate rather than direction.
            # GaLore's paper carries an explicit scale for this reason.
            if self.scale:
                tgt = self.lr * math.sqrt(p.numel())
                cur = float(step.norm())
                if cur > 1e-30:
                    step = step * (tgt / cur)
            step = step - self.lr * self.wd * p.data      # decay, outside
            upd.append((p, step))
        for n, p in self.vecs:                      # 1-D: plain AdamW
            if p.grad is None:
                continue
            g = p.grad
            self.vm[n] = self.b1 * self.vm[n] + (1 - self.b1) * g
            self.vv[n] = self.b2 * self.vv[n] + (1 - self.b2) * g * g
            mh = self.vm[n] / (1 - self.b1 ** self.t)
            vh = self.vv[n] / (1 - self.b2 ** self.t)
            upd.append((p, -self.lr * (mh / (vh.sqrt() + self.eps)
                                       + self.wd * p.data)))
        flat = torch.cat([s.reshape(-1) for _, s in upd])
        sg = torch.sign(flat)
        if self.stats["prev"] is not None:
            self.stats["flip"].append(float((sg != self.stats["prev"]).float().mean()))
        self.stats["prev"] = sg
        self.stats["cap"].append(cap_n / max(cap_d, 1e-30))
        with torch.no_grad():
            for p, s in upd:
                p.data.add_(s)

    def memory(self):
        adam = 2 * sum(p.numel() for _, p in self.mats + self.vecs)
        mine = 2 * sum(p.numel() for _, p in self.vecs)
        for n, p in self.mats:
            m_, n_ = p.shape
            r = min(self.r[n], min(m_, n_))
            mine += r * (m_ + n_) + 2 * r * r
        return adam, mine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", type=int, default=3)
    ap.add_argument("--sweep", action="store_true",
                    help="uniform-r control at matched memory + refresh sweep")
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--dim", type=int, default=128)
    a = ap.parse_args()
    subprocess.run([sys.executable, "/mnt/user-data/uploads/build_corpus.py",
                    "--out", "/tmp", "--loops", "300"], check=True,
                   capture_output=True)
    RAW = open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
    src = RAW[:RAW.find("# \u2500\u2500 PHASE 3")]
    src = src.replace("D=256; N_HEADS=4", f"D={a.dim}; N_HEADS=4", 1)
    src = src.replace("for mf_r in range(1, 16):", "for mf_r in range(1, 3):", 1)
    src = src.replace("    if pc == N_STU-1:", "    if False:", 1)
    src = src.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > "
                      "tau_history[-3]:", "    if False:", 1)
    res = {}
    if a.sweep:
        # uniform r chosen so total projector memory matches the role-aware
        # split: role-aware uses r=8 on attn (96x96) and r=16 on ff (192x96),
        # so a uniform r solves sum r(m+n) = the same total.
        arms = [("adamw", None, None, None),
                ("role r8/16 T3", 3, 8, 16),
                ("uniform r11 T3", 3, 11, 11),
                ("role r8/16 T1", 1, 8, 16),
                ("role r8/16 T10", 10, 8, 16)]
    else:
        arms = [("adamw", None, None, None), ("galore", a.refresh, 8, 16),
                ("galore-50", 50, 8, 16)]
    for name, rf, ra, rb in arms:
        G = {}; buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(src, G)
        model, get_batch, LR = G["model"], G["get_batch"], G["LR"]
        EV = [get_batch() for _ in range(10)]

        def L():
            t = 0.0
            with torch.no_grad():
                for x, y in EV: t += float(model(x, y)[1])
            return t / len(EV)
        def Hv(v, ps_):
            acc = torch.zeros(sum(p.numel() for p in ps_))
            for x, y in EV[:4]:
                model.zero_grad(); _, l = model(x, y)
                gr = torch.autograd.grad(l, ps_, create_graph=True)
                gf = torch.cat([t.reshape(-1) for t in gr])
                hv = torch.autograd.grad((gf * v).sum(), ps_, allow_unused=True)
                acc += torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1)
                                  for t, p in zip(hv, ps_)]).detach()
            model.zero_grad(); return acc / 4
        torch.manual_seed(17)
        opt = (torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95),
                                 weight_decay=0.1) if rf is None
               else GaLore(model, LR * 5, refresh=rf, r_attn=ra, r_ff=rb, scale=True))
        ps_ = [p for p in model.parameters() if p.requires_grad]
        Pn = sum(p.numel() for p in ps_)
        curve, unorm, hess = [], [], []
        for st in range(a.steps):
            th = torch.cat([p.data.reshape(-1) for p in ps_]).clone()
            x, y = get_batch(); _, l = model(x, y)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            u = torch.cat([p.data.reshape(-1) for p in ps_]) - th
            unorm.append(float(u.norm()))
            if (st + 1) % 20 == 0:
                curve.append((st + 1, L()))
                # Hessian degeneration: diagonal share and total Frobenius mass
                g_ = torch.Generator().manual_seed(st)
                dsum = torch.zeros(Pn); fro = 0.0
                for _ in range(6):
                    z = (torch.randint(0, 2, (Pn,), generator=g_).float() * 2 - 1)
                    hz = Hv(z, ps_)
                    with torch.no_grad():
                        i = 0
                        for p in ps_:
                            n = p.numel(); p.data.copy_(th[i:i+n].view_as(p) if False
                                                        else p.data); i += n
                    dsum += z * hz; fro += float(hz @ hz)
                d = dsum / 6; fro /= 6
                S = float((d * d).sum())
                Sc = max((S - fro / 6) / (1 - 1 / 6), 0.0)   # bias-corrected
                hess.append((st + 1, fro, Sc / max(fro, 1e-30)))
        res[name] = dict(curve=curve, final=L(),
                         unorm=[float(np.mean(unorm[:20])), float(np.mean(unorm[-20:]))],
                         hess=hess)
        if rf is not None:
            ad, mine = opt.memory()
            res[name].update(mem=(ad, mine),
                             cap=float(np.mean(opt.stats["cap"][-20:])),
                             flip=float(np.mean(opt.stats["flip"][-20:])))
        del G, model
        import gc; gc.collect()
        print(f"  {name} done", flush=True)
    print(f"\n  {'step':>6}" + "".join(f"{n[:11]:>13}" for n, _, _, _ in arms))
    for i in range(len(res["adamw"]["curve"])):
        print(f"  {res['adamw']['curve'][i][0]:>6}"
              + "".join(f"{res[n]['curve'][i][1]:>13.4f}" for n, _, _, _ in arms))
    print(f"\n  {'arm':>15}{'final':>9}{'mem save':>10}{'capture':>9}{'flip':>8}"
          f"{'|u| early':>11}{'|u| late':>10}")
    for n, rf, _, _ in arms:
        d = res[n]
        ms = f"{d['mem'][0]/d['mem'][1]:.1f}x" if rf is not None else "-"
        cp = f"{d['cap']:.3f}" if rf is not None else "-"
        fl = f"{d['flip']:.3f}" if rf is not None else "-"
        print(f"  {n:>15}{d['final']:>9.4f}{ms:>10}{cp:>9}{fl:>8}"
              f"{d['unorm'][0]:>11.4f}{d['unorm'][1]:>10.4f}")
    print(f"\n  HESSIAN DEGENERATION  ||H||_F^2 and bias-corrected diagonal share")
    print(f"  {'step':>6}" + "".join(f"{n[:11]:>26}" for n, _, _, _ in arms))
    for i in range(len(res["adamw"]["hess"])):
        line = f"  {res['adamw']['hess'][i][0]:>6}"
        for n, _, _, _ in arms:
            s_, f_, r_ = res[n]["hess"][i]
            line += f"{f_:>16.2f}{r_:>10.4f}"
        print(line)
    json.dump(res, open("/home/claude/work/res_galore.json", "w"),
              indent=2, default=float)


if __name__ == "__main__":
    main()
