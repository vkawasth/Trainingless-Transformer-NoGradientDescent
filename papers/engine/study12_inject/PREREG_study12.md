# Study 12 — injected records that tilt a decision (pre-registration)

Vinay K Awasthi, Johns Hopkins University. Registered 2026-10-28, before any registered design of H1–H4 was simulated. Frozen with
SHA-256 in `FROZEN_PREREG_study12.txt` together with `run_study12.py`, `design12.py`, `../pce/injection.py`,
`../pce/segment_decision.py`, `../pce/lab.py` and `../study11b_settle/design11b.py`.

## What was done before freezing
- Engine checks IN1–IN8 (`../check_injection.py`) on random truths.
- Deterministic predictions (`run_study12.py --predict`): no sampling.
- One smoke run (1000 replicates per cell). Its hidden-channel part used an invented truth (w = 0.5, a1 = (.75, .25, .70),
  a2 = (.25, .70, .20)). **Its tetrad part inadvertently used the m = 4 truth then intended for H4** (w = 0.30,
  a1 = (.70, .30, .80, .55), a2 = (.20, .65, .10, .25)). Because that design has now been seen, it was demoted to smoke use, and H4
  was moved to a fresh truth that has not been simulated (G4 below). The H4 criteria in `evaluate()` were written before the smoke
  run and were not changed. In the smoke run, empirical and predicted rates agreed to within 0.035 in every cell.

## Model
Whole-record injection. In every window, N records are observed; round(εN) of them are injected records drawn from a law Q, the
rest are genuine records from P. Injected and genuine records are independent, and there is no relation between them (a coproduct of
samples). Windows are sampled independently. The regulator runs the pipeline of the lab paper: pair tables give μ̂, v̂ and the
segment R̂; the triple window gives r̂ and the plug-in τ̂; the post-probe set is [r̂ ± zτ̂/√N] ∩ R̂ with z = 1.96, and the set rule
chooses the act. A replicate is *flagged* when its pair tables admit no two-type explanation, or when r̂ falls outside R̂ (the
triple contradicts the pairs).

## Designs
- **H1–H3, the hidden channel.** Design D2 of Study 11b (truth G; acts A: u = 1 − [any harm], and flat C; requirement u0 as there).
  Q = P + s·χ with χ the top Walsh character and s = 0.0580 (the positivity limit in the direction of the A/C switch). Q leaves every
  single and pair table unchanged and moves r̂ by εs/v_T. N = 100000. Theory (Theorem F): the switch budget at infinite N is
  ε* = |r_c − r0|·|v_T|/s = 0.0711; the flag budget is ε_det = |wall − r0|·|v_T|/s = 0.1635; at finite N the share choosing A
  crosses one half at ε½ = (|r_c − r0| − zτ/√N)·|v_T|/s = 0.0583 (the delta-method curve gives 0.0587).
  ε grid for A: 0, .02, .04, .05, .055, .06, .065, .07, .08, .10. ε grid for flags: .12, .14, .15, .16, .17, .18, .20.
- **H4, first-order stealth at m = 4.** Truth G4: w = 0.40, a1 = (.75, .25, .70, .60), a2 = (.15, .70, .20, .20). The regulator
  runs a Wald test of the two tetrads (level 0.05, pair windows of N records each, plug-in delta-method covariance). Two injections
  with the same first-order tilt of r̂ on the triple {0, 1, 2} (Δr = 0.3): the *stealth* law, the maximiser of the tilt over laws in the
  kernel of the tetrad Jacobian (a linear programme), at ε = 0.0296; and the *vertex* law, a point mass at the outcome of largest
  influence, at ε = 0.0279. N = 2000, 10000, 50000.
- 4000 replicates per cell; seed 1212.

## Predictions
- **H1 (tilt).** The share of replicates choosing A is within 0.05 of the delta-method prediction in at least 9 of the 10 cells, and
  within 0.08 in all. The empirical ε at which it crosses one half is within 10% of the predicted ε½.
- **H2 (detection).** The flag rate is within 0.05 of the prediction in at least 6 of 7 cells. The empirical ε at which it crosses
  one half is within 5% of ε_det.
- **H3 (the stealth window).** At ε = 0.07 the decision has been taken over (share choosing A ≤ 0.5) while the flag rate is at most
  0.01.
- **H4 (first-order stealth).** The rejection rate is within 0.04 of the noncentral-χ² prediction in at least 5 of 6 cells. The
  stealth injection is rejected at most 10% of the time at every N. The vertex injection is rejected at least 80% of the time at
  N = 50000.

## Reporting
`run_study12.py` writes `results_study12.json` with an `evaluation` block (H1–H4). All outcomes are reported, failures included.
Deviations are logged in `DEVIATIONS_study12.md` before the affected result is reported, and the frozen script is kept as a text copy.
