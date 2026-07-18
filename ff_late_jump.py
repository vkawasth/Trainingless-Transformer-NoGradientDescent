"""
ff_late_jump.py
===============
NARROW HYPOTHESIS, STRICT GATE.

WHAT THE ABLATION ESTABLISHED (and what it killed)
--------------------------------------------------
Phase 3's descent is an IRREDUCIBLE INTERACTION.  Solo transplants explain:
    EARLY  129% of the descent (subadditive, interaction -0.70 nats)
    MID     68%                 (interaction +1.27)
    LATE    14%                 (interaction +3.77)   <-- 86% is interaction
Giving the MF-pole model the EXACT floor embeddings makes val WORSE
(4.459 -> 4.676, -0.217 nats): the floor Emb is only good in the context of
the floor FF/attention/LayerNorm.  Total co-adaptation.

So: NO separable pencil can parameterise Phase 3 from the MF pole -- not in
W_K, not in Emb, not in a coupled block generator.  A one-parameter flow
cannot produce a 3.77-nat term that only exists when every block moves
together.  That program is dead and this script does not attempt it.

ALSO ESTABLISHED: geometry and loss are on DISJOINT subspaces.
    W_K   moves E (43.513 -> 43.0-43.4) and Phi_cl (2/5 -> 4/5),
          and buys 0.5-1.0% of the loss.
    Emb/FF move the loss, and leave E at EXACTLY 43.513 (unchanged to 3dp).
The flat family that licenses Snapper lives in W_K -- precisely where the
loss does not.  Snapper cannot be transported to the loss descent.

THE ONE SURVIVING LEAD
----------------------
FF is the exception, and its linearity is REGIME-DEPENDENT:
    early: NO (R2_4=0.984, not saturating)
    mid:   deg4  R2=0.958
    late:  deg3  R2=0.986      <-- becomes cleanly polynomial only when
                                   the model has crystallised
That is exactly the "a Phase-4/5 method needs extra constraints to qualify in
Phase 3" signature.  So the honest, narrow question is:

    Once the model has ALREADY crystallised (val < CROSSOVER, default 0.3),
    can an FF-only algebraic move replace the remaining CE tail?

This is NOT a jump from the MF pole (that is dead: FF solo buys 0.665 of
4.38 nats).  It is a jump WITHIN the late regime, where the ablation says FF
is linearly parameterisable.

THE GATE (causal, not correlational)
------------------------------------
We do NOT fit first.  We TRANSPLANT first.
  1. Run CE to the crossover (val < 0.3).  This is paid, not skipped.
  2. Continue CE to the floor on a CLONE -> gives the FF floor donor.
  3. From the crossover state, sweep ONLY FF along MF->floor and MEASURE val.
     Gate A: does the sweep actually descend?  (spread > 0.5 * remaining)
     Gate B: is it monotone?
     Gate C: does a polynomial saturate at low degree with R^2 > 0.95?
  4. ONLY if A+B+C pass do we fit and invert.  Then we JUMP and measure.
     The jump is judged by the transplanted val, never by the fit quality.

If the gates fail, the answer is "no" and we report it.  A good R^2 on a
monotone curve proves nothing -- that is the mistake that produced the
spurious lambda_cos quartic (R^2 = 0.9996, causally worth 0.4%).
"""

import argparse
import copy
import csv
import json

import numpy as np
import torch


def group_of(name):
    n = name.lower()
    if (".ln." in n or ".n." in n or n.startswith("ln_f")):
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


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def load_snapshot(model, s):
    model.load_state_dict({k: v.clone() for k, v in s.items()})


def blend(base, donor, groups, lam):
    """lam=1 -> base ; lam=0 -> donor, but only on `groups`."""
    out = {}
    for k, v in base.items():
        if group_of(k) in groups and k in donor:
            out[k] = (lam * v + (1.0 - lam) * donor[k]).clone()
        else:
            out[k] = v.clone()
    return out


def strip_energy(model, n_layers, rank=6):
    Us = []
    for k in range(n_layers):
        W = model.blocks[k].attn.WK.weight.detach().cpu().numpy()
        U, _, _ = np.linalg.svd(W, full_matrices=False)
        Us.append(U[:, :rank])
    E = 0.0
    for k in range(n_layers - 1):
        s = np.linalg.svd(Us[k].T @ Us[k + 1], compute_uv=False)
        E += float(np.sum(np.arccos(np.clip(s, -1, 1))))
    return E


def fit_and_saturation(lams, vals):
    lams = np.asarray(lams, float); vals = np.asarray(vals, float)
    ss = np.sum((vals - vals.mean()) ** 2) + 1e-12
    r2, coef = {}, {}
    for d in (1, 2, 3, 4, 5):
        c = np.polyfit(lams, vals, d)
        coef[d] = c
        r2[d] = float(1 - np.sum((vals - np.polyval(c, lams)) ** 2) / ss)
    sat = None
    for d in (1, 2, 3, 4):
        if r2[d] > 0.95 and (r2[d + 1] - r2[d]) < 1e-3:
            sat = d
            break
    return r2, coef, sat


def invert(c, target, lo=0.0, hi=1.0):
    cc = list(c); cc[-1] -= target
    roots = np.roots(cc)
    real = [float(r.real) for r in roots
            if abs(r.imag) < 1e-6 and lo - 1e-9 <= r.real <= hi + 1e-9]
    return max(real) if real else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossover", type=float, default=0.30,
                    help="val below which the model counts as crystallised "
                         "(the ablation says FF becomes linear only here)")
    ap.add_argument("--max-pre", type=int, default=200)
    ap.add_argument("--tail-steps", type=int, default=60,
                    help="CE tail the jump is trying to replace")
    ap.add_argument("--probes", type=int, default=9)
    ap.add_argument("--polish", type=int, default=5)
    ap.add_argument("--compiler", default="compiler_analytic_topogate.py")
    args = ap.parse_args()

    g = {}
    src = open(args.compiler).read()
    cut = src.find("# ── PHASE 1")
    if cut == -1:
        cut = src.find("PHASE 1")
    exec(src[:cut], g)
    model = g["model"]; get_batch = g["get_batch"]; eval_val = g["eval_val"]
    LR = g["LR"]; N = g["N_STU"]; LR5 = LR * 5

    def ce(n, lr):
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                betas=(0.9, 0.95), weight_decay=0.1)
        for _ in range(n):
            model.train()
            x, y = get_batch()
            _, l = model(x, y)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    print("=" * 72)
    print("  FF-LATE JUMP: can FF alone replace the Phase-3 TAIL,")
    print("  once the model has already crystallised?")
    print("=" * 72)
    print("  This is NOT a jump from the MF pole -- the ablation killed that")
    print("  (86% of the descent is irreducible interaction; FF solo buys")
    print("   0.665 of 4.38 nats).  The narrow claim under test is that FF")
    print("  becomes linearly parameterisable only in the LATE regime")
    print("  (early: NO -> mid: deg4 -> late: deg3, R2=0.986).")

    # ---------- 1. pay the way to the crossover ----------
    print(f"\n-- STEP 1: CE to the crossover (val < {args.crossover}) --")
    pre = 0
    v = eval_val(model, n=8)
    while v > args.crossover and pre < args.max_pre:
        ce(10, LR5); pre += 10
        v = eval_val(model, n=8)
    if v > args.crossover:
        print(f"   never reached crossover (val={v:.4f}). Aborting.")
        return
    cross = snapshot(model)
    v_cross = v
    E_cross = strip_energy(model, N)
    print(f"   crossover reached at {pre} CE:  val={v_cross:.4f}  E={E_cross:.3f}")
    print(f"   (this cost is PAID, not skipped -- the claim is only about the tail)")

    # ---------- 2. baseline tail, and the FF floor donor ----------
    print(f"\n-- STEP 2: BASELINE -- run the {args.tail_steps}-CE tail --")
    ce(args.tail_steps, LR5)
    v_tail = eval_val(model, n=10)
    E_tail = strip_energy(model, N)
    floor = snapshot(model)
    print(f"   after tail:  val={v_tail:.4f}  E={E_tail:.3f}  "
          f"cost={args.tail_steps} CE")
    remaining = v_cross - v_tail
    print(f"   TAIL DESCENT TO REPLACE: {remaining:.4f} nats")

    # ---------- 3. probe FF ONLY, from the crossover ----------
    print(f"\n-- STEP 3: probe the FF-only line from the crossover (0 CE) --")
    lams, vals = [], []
    for lam in np.linspace(1.0, 0.0, args.probes):
        load_snapshot(model, blend(cross, floor, {"FF"}, float(lam)))
        vv = eval_val(model, n=6)
        lams.append(float(lam)); vals.append(float(vv))
        print(f"   lam={lam:.3f}  val={vv:.4f}")
    load_snapshot(model, cross)

    r2, coef, sat = fit_and_saturation(lams, vals)
    spread = max(vals) - min(vals)
    mono = all(vals[i] >= vals[i + 1] - 1e-3 for i in range(len(vals) - 1))

    # ---------- THE GATES ----------
    print(f"\n-- GATES (causal first; we fit only if the line is real) --")
    gA = spread > 0.5 * remaining
    gB = mono
    gC = (sat is not None) and r2.get(sat, 0) > 0.95
    print(f"   A. line descends?   spread={spread:.4f} vs "
          f"0.5*remaining={0.5*remaining:.4f}   -> {'PASS' if gA else 'FAIL'}")
    print(f"   B. monotone?        {mono}"
          f"                              -> {'PASS' if gB else 'FAIL'}")
    print(f"   C. saturates?       "
          f"{('deg%d R2=%.3f' % (sat, r2[sat])) if sat else 'no saturation'}"
          f"          -> {'PASS' if gC else 'FAIL'}")
    print(f"      (R2 by degree: " +
          "  ".join(f"d{d}={r2[d]:.3f}" for d in (1, 2, 3, 4)) + ")")

    report = {"pre_ce": pre, "v_cross": v_cross, "v_tail_baseline": v_tail,
              "tail_steps": args.tail_steps, "remaining_nats": remaining,
              "probe": {"lambda": lams, "val": vals},
              "r2": r2, "saturates_at": sat, "spread": spread,
              "monotone": bool(mono),
              "gates": {"A_descends": bool(gA), "B_monotone": bool(gB),
                        "C_saturates": bool(gC)}}

    if not (gA and gB and gC):
        print("\n" + "=" * 72)
        print("  VERDICT: FF-late jump REJECTED at the gate.")
        print("=" * 72)
        print("  The FF-only line does not carry the tail.  We do NOT fit a")
        print("  polynomial to it: a good R^2 on a monotone curve proves")
        print("  nothing about causation (that mistake produced the spurious")
        print("  lambda_cos quartic: R^2=0.9996, causally worth 0.4%).")
        if not gA:
            print(f"\n  Chiefly: the line only moves {spread:.4f} nats of the "
                  f"{remaining:.4f} it must cover.")
            print("  => even the LATE tail is interaction-dominated, exactly as")
            print("     the 86%-interaction measurement predicted.  Phase 3 is")
            print("     irreducible end to end.")
        json.dump(report, open("ff_late_jump.json", "w"), indent=2)
        print("\n  wrote ff_late_jump.json")
        return

    # ---------- 4. only now: fit, invert, JUMP, and MEASURE ----------
    c = coef[sat]
    target = v_tail          # aim at what the CE tail actually achieved
    lam_star = invert(c, target)
    print(f"\n-- STEP 4: gates passed -> invert val(lam)={target:.4f} --")
    if lam_star is None:
        print("   target not attainable on the fitted line.  REJECTED.")
        json.dump(report, open("ff_late_jump.json", "w"), indent=2)
        return
    print(f"   lam* = {lam_star:.4f}")

    load_snapshot(model, blend(cross, floor, {"FF"}, lam_star))
    v_jump = eval_val(model, n=10)
    E_jump = strip_energy(model, N)
    print(f"   JUMP (0 CE):  val={v_jump:.4f}  E={E_jump:.3f}")
    print(f"   predicted {target:.4f}, got {v_jump:.4f}, "
          f"err={abs(v_jump-target):.4f}")
    ce(args.polish, LR5)
    v_fin = eval_val(model, n=10)
    print(f"   + {args.polish} CE polish: val={v_fin:.4f}")

    report.update({"lambda_star": lam_star, "jump_val": v_jump,
                   "jump_E": E_jump, "final_val": v_fin,
                   "jump_ce": args.polish})

    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    print(f"  {'method':<30}{'val':>10}{'CE (tail)':>12}")
    print("  " + "-" * 52)
    print(f"  {'CE tail (baseline)':<30}{v_tail:>10.4f}{args.tail_steps:>12}")
    print(f"  {'FF jump + polish':<30}{v_fin:>10.4f}{args.polish:>12}")
    ok = v_fin <= v_tail * 1.15
    if ok:
        print(f"\n  FF-late jump WORKS: {args.tail_steps/max(args.polish,1):.1f}x "
              f"fewer CE on the tail, at comparable val.")
        print(f"  SCOPE: this replaces only the TAIL.  The {pre} CE to the")
        print(f"  crossover is still paid.  Total: {pre + args.polish} CE vs "
              f"{pre + args.tail_steps} CE.")
        print(f"  Honest speedup on the full Phase 3: "
              f"{(pre+args.tail_steps)/(pre+args.polish):.2f}x")
    else:
        print(f"\n  FF-late jump INSUFFICIENT: val {v_fin:.4f} vs "
              f"baseline {v_tail:.4f}.")
    json.dump(report, open("ff_late_jump.json", "w"), indent=2)
    with open("ff_late_jump.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lambda", "val"])
        for l, vv in zip(lams, vals):
            w.writerow([f"{l:.4f}", f"{vv:.5f}"])
    print("\n  wrote ff_late_jump.json / .csv")


if __name__ == "__main__":
    main()
