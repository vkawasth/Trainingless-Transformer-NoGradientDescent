"""
geo_phases.py
=============
The remaining geometry-compiler phases as opaque, schedule-callable ops,
extending GeometryCompilerFull. Each method runs its ENTIRE internal algorithm
(loops, geo-stop criteria, polynomial fits) in Python and returns val. The
schedule sequences them and branches on val/tau/phi between phases.

Faithful extractions from compiler_geometri_patched_86_memfixed.py:
  basin_settle()  -> Phase 3 (150-step LR*5 settle + GEO-STOP + Phi-extend)
  tau_retry()     -> Phase 3 tail (fast-descent if geo-stopped else tau-retry)
  snapper_jump()  -> Snapper polynomial jump to floor
  topogate()      -> Phase 4 sign-flip search
  align_lm()      -> Phase 5 alignment + optional LM step
  k0_split()      -> Phase 5 K0 split descent
  joint_ce()      -> Phase 5 joint cosine-annealed descent

These read/write self.model in place and cache the last geo-stop flag so the
schedule's tau-retry branch can mirror the original.
"""

import math, copy
import numpy as np
import torch

from geo_compiler_full import GeometryCompilerFull, N_STU, LR


class GeometryPhases(GeometryCompilerFull):

    # ------- geometric sensor: weighted r_m2^sigma (used inside basin) -------
    def _rm2_sigma(self, rank=6):
        wk = []
        for name, p in self.model.named_parameters():
            n = name.lower()
            if ('key' in n or 'wk' in n or 'w_k' in n) and 'weight' in n and p.ndim >= 2:
                wk.append(p.detach().float().cpu().numpy())
        if len(wk) < 2:
            return 0.0
        wk.sort(key=lambda w: w.shape[0])
        vals = []
        for k in range(len(wk) - 1):
            W0, W1 = wk[k], wk[k + 1]
            try:
                U0, s0, _ = np.linalg.svd(W0, full_matrices=False)
                U1, s1, _ = np.linalg.svd(W1, full_matrices=False)
                r = min(rank, U0.shape[1], U1.shape[1])
                Ur0, Ur1 = U0[:, :r], U1[:, :r]
                sv = np.linalg.svd(Ur0.T @ Ur1, compute_uv=False)
                sv = np.clip(sv, 1e-6, 1 - 1e-6)
                h_strip = sv / (1 - sv ** 2) ** 1.5
                h_loss = s0[:r] / (np.linalg.norm(s0[:r]) + 1e-10)
                h_strip = h_strip / (np.linalg.norm(h_strip) + 1e-10)
                w = 1.0 / (sv ** 2 + 1e-6)
                num = np.dot(h_loss * w, h_strip)
                den = (np.sqrt(np.dot(h_loss ** 2, w)) *
                       np.sqrt(np.dot(h_strip ** 2, w)) + 1e-10)
                vals.append(float(num / den))
            except Exception:
                pass
        return float(np.mean(vals)) if vals else 0.0

    # ----------------------------- Phase 3 ----------------------------------
    def basin_settle(self, max_steps: int = 150) -> float:
        opt = torch.optim.AdamW(self.model.parameters(), lr=LR * 5,
                                betas=(0.9, 0.95), weight_decay=0.1)
        val_hist = [self.eval_val(n=8)]
        geo_stop_count = 0
        self._geo_stopped = False
        step = 0
        for step in range(1, max_steps + 1):
            if step <= 10:
                for pg in opt.param_groups:
                    pg['lr'] = LR * 5 * step / 10
            self.model.train(); x, y = self._get_batch(); _, l = self.model(x, y)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()

            if step % 8 == 0:
                v = self.eval_val(n=8)
                delta = abs(v - val_hist[-1]) / 8
                val_hist.append(v)
                pc = self.phi_clean()
                tau = self.gluing_defect(n=4)
                rm2 = self._rm2_sigma()
                if delta < 0.003:
                    break
                if v < 0.15:
                    break
                if pc >= 4 and 5.0 <= tau <= 7.5 and rm2 >= 0.65:
                    geo_stop_count += 1
                    if geo_stop_count >= 2:
                        self._geo_stopped = True
                        break
                else:
                    geo_stop_count = 0
        self._basin_steps = step
        # Phi extension
        pc_b = self.phi_clean()
        if pc_b < 3:
            for _ in range(16):
                self.model.train(); x, y = self._get_batch(); _, l = self.model(x, y)
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            self._basin_steps += 16
        return self.eval_val(n=8)

    def geo_stopped(self) -> int:
        return 1 if getattr(self, "_geo_stopped", False) else 0

    def tau_retry(self) -> float:
        # mirrors the original: geo-stopped -> fast descent; elif tau>5 -> retry
        pc_b = self.phi_clean()
        tau_b = self.gluing_defect()
        if getattr(self, "_geo_stopped", False):
            n_fast = 30
            opt = torch.optim.AdamW(self.model.parameters(), lr=LR * 10,
                                    betas=(0.9, 0.95), weight_decay=0.1)
            for s in range(n_fast):
                lr_s = LR * 2 + (LR * 10 - LR * 2) * 0.5 * (1 + math.cos(math.pi * s / n_fast))
                for pg in opt.param_groups: pg['lr'] = lr_s
                self.model.train(); x, y = self._get_batch(); _, l = self.model(x, y)
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); opt.step()
        elif tau_b > 5:
            n_retry = 25 if pc_b >= 5 else 75 if pc_b <= 2 else 50
            opt = torch.optim.AdamW(self.model.parameters(), lr=LR * 2,
                                    betas=(0.9, 0.95), weight_decay=0.1)
            for s in range(n_retry):
                lr_s = LR * 2 * 0.5 * (1 + math.cos(math.pi * s / n_retry))
                for pg in opt.param_groups: pg['lr'] = lr_s
                self.model.train(); x, y = self._get_batch(); _, l = self.model(x, y)
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); opt.step()
        return self.eval_val(n=8)

    # -------------------------- Snapper jump --------------------------------
    def snapper_jump(self) -> float:
        # Hessian smallest eigenvector
        n_p = self.num_params()
        torch.manual_seed(43)
        v = torch.randn(n_p); v = v / v.norm()
        for _ in range(10):
            Hv = self._hvp(v, n=4); v = -Hv / max(float((-Hv).norm()), 1e-10)
        direction = v / max(v.norm(), 1e-10)
        # polynomial fit
        w0 = self._flat_params()
        t_vals = np.array([i * 0.1 for i in range(5)])
        loss_vals = []
        for t in t_vals:
            self._set_flat(w0 + t * direction); loss_vals.append(self.eval_val(n=4))
        X = np.vander(t_vals, 5, increasing=True)
        a0, a1, a2, a3, a4 = np.linalg.lstsq(X, loss_vals, rcond=None)[0]
        # minimize
        tg = np.linspace(-0.2, 0.5, 200)
        Lg = a0 + a1 * tg + a2 * tg ** 2 + a3 * tg ** 3 + a4 * tg ** 4
        t_star = tg[int(np.argmin(Lg))]
        for _ in range(5):
            dL = a1 + 2 * a2 * t_star + 3 * a3 * t_star ** 2 + 4 * a4 * t_star ** 3
            d2L = 2 * a2 + 6 * a3 * t_star + 12 * a4 * t_star ** 2
            if abs(d2L) > 1e-10: t_star = t_star - dL / d2L
            t_star = max(-0.2, min(0.5, t_star))
        self._set_flat(w0 + t_star * direction)
        return self.eval_val(n=8)

    # --------------------------- TopoGate -----------------------------------
    def topogate(self) -> float:
        pc_before = self.phi_clean()
        v_before = self.eval_val(n=8)
        best_score = 0.0; best_layers = None
        candidates = [[1, 2], [0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [0, 4], [1, 4]]
        for flip in candidates:
            with torch.no_grad():
                for l in flip:
                    self.model.blocks[l].attn.WV.weight.data.mul_(-1)
                    self.model.blocks[l].attn.op.weight.data.mul_(-1)
            v_try = self.eval_val(n=6); pc_try = self.phi_clean()
            score = (v_before - v_try) + 0.3 * (pc_try - pc_before) / 5.0
            if score > best_score:
                best_score = score; best_layers = flip
            with torch.no_grad():
                for l in flip:
                    self.model.blocks[l].attn.WV.weight.data.mul_(-1)
                    self.model.blocks[l].attn.op.weight.data.mul_(-1)
        if best_layers and best_score > 0:
            with torch.no_grad():
                for l in best_layers:
                    self.model.blocks[l].attn.WV.weight.data.mul_(-1)
                    self.model.blocks[l].attn.op.weight.data.mul_(-1)
        return self.eval_val(n=8)

    # ------------------------- Phase 5 pieces -------------------------------
    def _lm_step(self, mu=0.950, n_grad=25, n_hvp=12, n_cg=6):
        self.model.zero_grad()
        loss = sum(self.model(*self._get_batch())[1] for _ in range(n_grad)) / n_grad
        loss.backward()
        g = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                       for p in self.model.parameters()]).detach(); self.model.zero_grad()
        d = torch.zeros_like(g); r = -g.clone(); p = r.clone(); rr = float((r * r).sum())
        for _ in range(n_cg):
            Hp = self._hvp(p, n=n_hvp) + mu * p
            al = rr / max(float((p * Hp).sum()), 1e-10)
            d += al * p; r -= al * Hp; rr2 = float((r * r).sum())
            p = r + (rr2 / max(rr, 1e-10)) * p; rr = rr2
        w0 = self._flat_params(); v0 = self.eval_val(n=8)
        self._set_flat(w0 + d); v1 = self.eval_val(n=8)
        if v1 < v0: return v1
        self._set_flat(w0); return v0

    def align_lm(self) -> float:
        v_pre = self.eval_val(n=8)
        if v_pre < 0.10:
            return v_pre   # skip LM, already near floor
        # cos alignment vs g_floor
        if self.g_floor is not None:
            self.model.zero_grad()
            ls = [self.model(*self._get_batch())[1] for _ in range(8)]
            torch.stack(ls).mean().backward()
            g = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                           for p in self.model.parameters()]).detach(); self.model.zero_grad()
            _ = float((g * self.g_floor).sum() / (g.norm() * self.g_floor.norm() + 1e-10))
        return self._lm_step()

    def k0_split(self, n_steps: int = 25) -> float:
        base = self.model
        params_base = {n: p.data.clone() for n, p in base.named_parameters()}
        def ptype(name):
            if '.attn.WQ.' in name or '.attn.WK.' in name: return 'Attn'
            if 'te.weight' in name or '.ff.' in name: return 'EmbFF'
            return 'other'
        tau_now = self.gluing_defect(n=6)
        w_ff = 3.5 * (1.5 / max(tau_now, 0.5)) ** 1.5

        m1 = copy.deepcopy(base)
        for name, p in m1.named_parameters():
            if ptype(name) != 'EmbFF': p.requires_grad_(False)
        p1 = [p for p in m1.parameters() if p.requires_grad]
        opt1 = torch.optim.AdamW(p1, lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
        for s in range(1, n_steps + 1):
            for pg in opt1.param_groups: pg['lr'] = LR * 0.5 * (1 + math.cos(math.pi * s / n_steps))
            m1.train(); x, y = self._get_batch(); _, l = m1(x, y)
            opt1.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(p1, 1.0); opt1.step()

        m2 = copy.deepcopy(base)
        for name, p in m2.named_parameters():
            if ptype(name) != 'Attn': p.requires_grad_(False)
        p2 = [p for p in m2.parameters() if p.requires_grad]
        opt2 = torch.optim.AdamW(p2, lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
        for s in range(1, n_steps + 1):
            for pg in opt2.param_groups: pg['lr'] = LR * 0.5 * (1 + math.cos(math.pi * s / n_steps))
            m2.train(); x, y = self._get_batch(); _, l = m2(x, y)
            opt2.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(p2, 1.0); opt2.step()

        with torch.no_grad():
            for name, p in base.named_parameters():
                pt = ptype(name)
                d1 = dict(m1.named_parameters())[name].data - params_base[name]
                d2 = dict(m2.named_parameters())[name].data - params_base[name]
                if pt == 'EmbFF':
                    if 'te.weight' in name: p.data.add_(d1)
                    else: p.data.add_(w_ff * d1)
                elif pt == 'Attn':
                    p.data.add_(d2)
        return self.eval_val(n=15)

    def joint_ce(self, n_steps: int = 25) -> float:
        m = copy.deepcopy(self.model)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
        for s in range(1, n_steps + 1):
            for pg in opt.param_groups: pg['lr'] = LR * 0.5 * (1 + math.cos(math.pi * s / n_steps))
            m.train(); x, y = self._get_batch(); _, l = m(x, y)
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        self.model = m
        return self.eval_val(n=15)


GeometryCompiler = GeometryPhases


if __name__ == "__main__":
    g = GeometryPhases(use_real_corpus=False, build_floor=True, floor_steps=20)
    print("params", g.num_params())
    print("val0", g.eval_val(4))
    print("saddle", g.saddle())
    print("mfpump", g.mfpump(0), "tau", g.gluing_defect(), "phi", g.phi_clean())
    print("basin", g.basin_settle(max_steps=24), "geo_stopped", g.geo_stopped())
    print("tau_retry", g.tau_retry())
    print("snapper", g.snapper_jump())
    print("topogate", g.topogate())
    print("align_lm", g.align_lm())
    print("k0", g.k0_split(n_steps=5))
