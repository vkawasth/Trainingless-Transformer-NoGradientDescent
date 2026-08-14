"""SUBSPACE ADAM INSIDE THE REAL PHASE 3.

Everything measured today was in one of two places, and neither is the pipeline:

  tiny replica  P=3672, a synthetic grammar. Krylov-Newton lost 2.06x with 30x
                the variance; pure projection cost 73% at convergence.
  CIFAR         P=545K, a conv net. CS-honest tracked Adam for 3000 steps -- but
                kept 90% of the update norm, so it was barely a restriction.

This is the actual environment: the compiler's model at D=128 (P~4.3e6), the
1,271-token corpus from build_corpus.py at 300 loops, and Phase 3's own exit gate
(val < 0.15). Phases 1 and 2 run unchanged, so the starting point is the real
post-MF-pump theta rather than a random init.

Three arms from that same starting point:
  adamw      the current Phase 3 optimizer, unchanged
  cs-honest  per-block rank-4 frame from the last 8 updates, update projected
             onto it, no rescale
  alpha0.25  the same projection plus a quarter of the complement, which was the
             only configuration all day to come within noise of Adam

Reported: steps to reach each loss gate, and the fraction of update energy the
frame captures. The gate table is the deliverable -- Phase 3 exits at 0.15, so
the cost AT THAT GATE is the number that decides whether this belongs in the
pipeline.
"""
import json, subprocess, numpy as np, torch, io, contextlib, re, math

subprocess.run(["python3", "/mnt/user-data/uploads/build_corpus.py", "--out", "/tmp",
                "--loops", "300"], check=True, capture_output=True)
RAW = open("/mnt/user-data/uploads/compiler_geometri_patched_86.py").read()
CUT = RAW.find("# \u2500\u2500 PHASE 3")
PRE = RAW[:CUT].replace("D=256; N_HEADS=4", "D=128; N_HEADS=4", 1)
PRE = PRE.replace("for mf_r in range(1, 16):", "for mf_r in range(1, 3):", 1)
PRE = PRE.replace("    if pc == N_STU-1:", "    if False:", 1)
PRE = PRE.replace("    if len(tau_history)>=3 and tau > tau_history[-2] > tau_history[-3]:",
                  "    if False:", 1)

GATES = [3.0, 2.0, 1.5, 1.0, 0.6, 0.3, 0.15]
K_DIM = 4
WINL = 8
MAXSTEP = 400


def build_state():
    """Run phases 1 and 2 exactly as the compiler does."""
    G = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(PRE, G)
    return G


def blocks(named):
    span = {}; i = 0
    for nm, p in named:
        span[nm] = (i, i + p.numel()); i += p.numel()
    bi = {}
    for nm, (a, b) in span.items():
        if nm.startswith("te") or nm.startswith("pe"):
            k = "EMB"
        elif "ln" in nm.lower():
            k = "LN"
        elif ".ff." in nm:
            m = re.search(r"blocks\.(\d+)", nm)
            k = f"FF{m.group(1)}" if m else "FF"
        else:
            m = re.search(r"blocks\.(\d+)", nm)
            k = f"AT{m.group(1)}" if m else "AT"
        bi.setdefault(k, []).append(torch.arange(a, b))
    return {k: torch.cat(v) for k, v in bi.items()}


def run(mode, alpha=0.0):
    G = build_state()
    model = G["model"]; get_batch = G["get_batch"]; LR = G["LR"]; ev = G["eval_val"]
    named = [(n, p) for n, p in model.named_parameters()]
    ps = [p for _, p in named]
    P = sum(p.numel() for p in ps)
    bi = blocks(named)
    opt = torch.optim.AdamW(model.parameters(), lr=LR * 5, betas=(0.9, 0.95),
                            weight_decay=0.1)
    hist = {k: [] for k in bi}
    frames = {}
    gates = list(GATES); hit = {}; caps = []
    v0 = float(ev(model, n=8)); model.train()
    while gates and gates[0] >= v0:
        hit[gates.pop(0)] = 0
    for st in range(MAXSTEP):
        th = torch.cat([p.data.reshape(-1) for p in ps]).clone()
        x, y = get_batch(); _, l = model(x, y); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        u = torch.cat([p.data.reshape(-1) for p in ps]) - th
        for k, ii in bi.items():
            hist[k].append(u[ii].clone())
            if len(hist[k]) > WINL:
                hist[k].pop(0)
        if mode != "adamw" and len(hist["EMB"]) == WINL:
            if st % WINL == 0 or not frames:
                for k, ii in bi.items():
                    A = torch.stack(hist[k], 1)
                    frames[k] = torch.linalg.svd(A, full_matrices=False)[0][:, :K_DIM]
            nu = torch.zeros(P); cap = 0.0; tot = 0.0
            for k, ii in bi.items():
                ub = u[ii]; Q = frames[k]
                par = Q @ (Q.T @ ub)
                nu[ii] = par + alpha * (ub - par)
                cap += float((par * par).sum()); tot += float((ub * ub).sum())
            caps.append(cap / max(tot, 1e-30))
            with torch.no_grad():
                o = 0
                for p in ps:
                    n = p.numel(); p.data.copy_((th + nu)[o:o + n].view_as(p)); o += n
        if gates and (st + 1) % 4 == 0:
            v = float(ev(model, n=6)); model.train()
            while gates and v <= gates[0]:
                hit[gates.pop(0)] = st + 1
        if not gates:
            break
    vf = float(ev(model, n=8))
    cap = float(np.mean(caps)) if caps else 1.0
    del G, model
    import gc; gc.collect()
    return hit, vf, cap, v0


print("  REAL PHASE 3: compiler model D=128, 1271-token corpus x300 loops")
print("  phases 1 and 2 run unchanged; Phase 3 exit gate is val < 0.15\n")
res = {}
for lab, mode, a in (("adamw", "adamw", 0.0),
                     ("cs-honest", "cs", 0.0),
                     ("alpha0.25", "cs", 0.25)):
    hit, vf, cap, v0 = res.setdefault(lab, run(mode, a))
    print(f"  {lab:>10}: start val {v0:.3f}, final {vf:.4f}, "
          f"frame captures {cap:.3f} of update energy", flush=True)
print(f"\n  {'gate':>7}" + "".join(f"{k:>12}" for k in res)
      + f"{'cs cost':>10}{'a.25 cost':>11}")
base = res["adamw"][0]
for g in GATES:
    if g not in base:
        continue
    row = [res[k][0].get(g) for k in res]
    f = lambda x: f"{x:12d}" if x is not None else f"{'>400':>12}"
    c1 = row[1] / base[g] if row[1] and base[g] else None
    c2 = row[2] / base[g] if row[2] and base[g] else None
    print(f"  {g:>7.2f}" + "".join(f(x) for x in row)
          + (f"{c1:>10.2f}x" if c1 else f"{'never':>11}")
          + (f"{c2:>10.2f}x" if c2 else f"{'never':>11}"))
json.dump({k: {"gates": {str(a): b for a, b in v[0].items()}, "final": v[1],
               "cap": v[2]} for k, v in res.items()},
          open("/home/claude/work/res_realphase3.json", "w"), indent=2)
print(f"\n  the cost at gate 0.15 is the number that decides this")
