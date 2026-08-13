"""GPS FOR PHASE 3: LEAKAGE-TRIGGERED SUBSPACE RECALIBRATION.

Reconnaissance on the real compiler established: per-role k90 <= 4 at P=4.3e6
(k90/n as low as 6.8e-6 for FF), roles expand together (mean pairwise corr of
k90 trajectories +0.624), and the drift ratio does NOT predict k90 within a role
over time (corr -0.61 for W_Q). So the trigger must MEASURE rather than predict.

    gamma_leak(t) = 1 - ||P_Q g(t)||^2 / ||g(t)||^2

aggregated across blocks by ENERGY WEIGHT, not max: LN has 1792 parameters
against FF's 591360, a 330x spread, and a max would let the smallest block
dictate the schedule for the whole model.

Controller: run projected in the current per-block frames; every CHECK steps
evaluate gamma_leak on one full gradient; if it exceeds theta, run WIN full-rank
steps and re-estimate every frame from them.

Arms:
  full          unconstrained (reference)
  gps@theta     theta in {0.1,0.2,0.3,0.4}; the schedule is a CONSEQUENCE of
                theta, not of hindsight about where transitions sit
  uniform       the SAME total full-rank budget each GPS arm spent, placed
                uniformly. Without this, "GPS beats projected-only" would only
                say that full-rank steps help.
  frozen        detect once at step 20, never recalibrate

k = 5 per block, 7 blocks, 35 dimensions = 8e-4 percent of P.
"""
import json, subprocess, numpy as np, torch, io, contextlib

subprocess.run(["python3", "/mnt/user-data/uploads/build_corpus.py", "--out", "/tmp",
                "--loops", "300"], check=True, capture_output=True)
RAW = open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT = RAW.find("# \u2500\u2500 PHASE 3")
BASE = RAW[:CUT].replace("D=256; N_HEADS=4", "D=128; N_HEADS=4", 1)
BASE = BASE.replace("for mf_r in range(1, 16):", "for mf_r in range(1, 3):", 1)
BASE = BASE.replace("    if pc == N_STU-1:", "    if False:", 1)
BASE = BASE.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                    "    if False:", 1)
K = 5
STEPS = 200
CAP = 80          # hard cap on full-rank steps for every non-full arm
CHECK = 5
WIN = 8
DETECT = 20


def role(nm):
    if nm.startswith("te") or nm.startswith("pe"):
        return "EMB"
    if "ln" in nm.lower():
        return "LN"
    if ".ff." in nm:
        return "FF"
    for pat, lab in (("WQ", "W_Q"), ("WK", "W_K"), ("WV", "W_V")):
        if pat in nm:
            return lab
    if "WO" in nm or "op" in nm:
        return "W_O"
    return "other"


ROLES = ["EMB", "LN", "FF", "W_Q", "W_K", "W_V", "W_O"]


def run(mode, theta=None, budget=None, seed=17):
    G = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(BASE, G)
    model = G["model"]; get_batch = G["get_batch"]; LR = G["LR"]; ev = G["eval_val"]
    named = [(n, p) for n, p in model.named_parameters()]
    params = [p for _, p in named]
    P = sum(p.numel() for p in params)
    for p in params:
        p.requires_grad_(True)
    span = {}; i = 0
    for nm, p in named:
        span[nm] = (i, i + p.numel()); i += p.numel()
    idx = {r: [] for r in ROLES}
    for nm, (a, b) in span.items():
        if role(nm) in idx:
            idx[role(nm)].append(torch.arange(a, b))
    idx = {r: torch.cat(v) for r, v in idx.items() if len(v)}
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95),
                            weight_decay=0.1)

    def flat():
        return torch.cat([p.data.flatten() for p in params]).clone()

    def setth(t):
        with torch.no_grad():
            j = 0
            for p in params:
                q = p.numel(); p.data.copy_(t[j:j + q].view_as(p)); j += q

    def gvec():
        return torch.cat([(p.grad.flatten() if p.grad is not None
                           else torch.zeros(p.numel())) for p in params]).clone()

    Q = None
    recent = {r: [] for r in idx}
    full_steps = 0
    recals = []
    curve = []
    leaks = []
    for st in range(STEPS):
        # decide whether this step is full rank
        if mode == "full":
            do_full = True
        elif mode == "uniform":
            ue = max(1, (STEPS - DETECT) // max(1, CAP - DETECT))
            do_full = (st < DETECT) or (full_steps < CAP and st % ue == 0)
        elif mode == "frozen":
            do_full = st < DETECT
        else:  # gps
            do_full = st < DETECT
            if Q is not None and st % CHECK == 0 and full_steps < CAP:
                th = flat()
                x, y = get_batch(); model.zero_grad(); _, l = model(x, y); l.backward()
                g = gvec(); model.zero_grad(); setth(th)
                num = 0.0; den = 0.0
                for r, ii in idx.items():
                    gi = g[ii]; e = float((gi * gi).sum())
                    pj = Q[r] @ (Q[r].T @ gi)
                    num += e * (1 - float((pj * pj).sum()) / max(e, 1e-30))
                    den += e
                gl = num / max(den, 1e-30)
                leaks.append((st, gl))
                if gl > theta:
                    recals.append(st)
                    for _ in range(min(WIN, CAP - full_steps)):
                        th2 = flat()
                        x, y = get_batch(); _, l = model(x, y); opt.zero_grad(); l.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step(); full_steps += 1
                        du = flat() - th2
                        for r, ii in idx.items():
                            recent[r].append(du[ii].clone())
                            if len(recent[r]) > WIN:
                                recent[r].pop(0)
                    Q = {r: torch.linalg.svd(torch.stack(recent[r], 1),
                                             full_matrices=False)[0][:, :K]
                         for r in idx}
                    continue
        th = flat()
        x, y = get_batch(); _, l = model(x, y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        du = flat() - th
        if do_full:
            full_steps += 1
            for r, ii in idx.items():
                recent[r].append(du[ii].clone())
                if len(recent[r]) > max(WIN, DETECT):
                    recent[r].pop(0)
            if mode != "full" and st == DETECT - 1:
                Q = {r: torch.linalg.svd(torch.stack(recent[r], 1),
                                         full_matrices=False)[0][:, :K] for r in idx}
        else:
            nu = torch.zeros_like(du)
            for r, ii in idx.items():
                b = du[ii]
                nu[ii] = Q[r] @ (Q[r].T @ b)
            setth(th + nu)
        if (st + 1) % 50 == 0:
            curve.append((st + 1, float(ev(model, n=6)))); model.train()
    v = float(ev(model, n=8))
    del G, model, params
    import gc; gc.collect()
    return dict(val=v, curve=curve, full=full_steps, recals=recals,
                leaks=leaks, P=P)


print(f"  real compiler D=128, k={K}/block, 7 blocks = {7*K} dims, {STEPS} steps")
print(f"  check every {CHECK}, recal window {WIN}, detect {DETECT}\n")
out = {}
print(f"  {'arm':>12}{'full used':>11}{'val@100':>10}{'val@200':>10}{'recals':>8}")
for lab, mode, th in (("full","full",None),("uniform","uniform",None),
                      ("frozen","frozen",None),
                      ("gps@0.5","gps",0.5),("gps@0.7","gps",0.7),
                      ("gps@0.9","gps",0.9)):
    r = run(mode, theta=th)
    out[lab] = r
    c = dict(r["curve"])
    extra = f"   first {r['recals'][:5]}" if r["recals"] else ""
    print(f"  {lab:>12}{r['full']:>11}{c.get(100,float('nan')):>10.4f}"
          f"{c.get(200,float('nan')):>10.4f}{len(r['recals']):>8}{extra}", flush=True)
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "leaks"} for k, v in out.items()},
          open("/home/claude/work/res_gps43.json", "w"), indent=2)
print(f"\n  gps beats uniform at the SAME cap => placement matters at 4.3M too")
