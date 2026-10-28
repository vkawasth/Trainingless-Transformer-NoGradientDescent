# Study 11: the latent-actor lab (synthetic; registered 2026-10-27)

## Purpose
This study tests the three theorems of the companion paper, "A lab for latent actors", on synthetic data where the truth is known.
- **Theorem A (hiding segment).** Pair tables leave a segment of explanations. Along it, the probability of an event on k variables is a
  polynomial in r of degree at most k - 2. Worst cases sit at the box walls, or, for k >= 4, at no more than k - 3 interior critical
  points.
- **Theorem B (tempos).** The tempo is zero for the weight under pair-only disclosure; n^-1/2 after one triple with v_S != 0; n^-1/2 for
  coordinate agreement (a regular relation); n^-1/4 for the separation at the merge stratum (Chen 1995).
- **Theorem C (stratified VOI).**
  - The minimax VOI is 0 on the independence stratum and 0 for any probe touching the coincident set Z.
  - Otherwise the probe collapses the segment completely.
  - With n samples, VOI increases with |v_S|.
  - The best probe is argmax |v_S|.
  - Disclosure corollary: unravelling.

## Corrections made before freezing (recorded so the record is complete)
In the design discussion I had predicted response exponent 1/2 for "agreement on one variable" (d_i = 0). Working the proof of
Theorem B showed this is wrong. When at least two other coordinates differ, v_i = m_ij / v_j is linear in the data, so agreement is a
regular relation: exponent 1, rate n^-1/2. The exponent 1/2 and the rate n^-1/4 belong to the merge stratum (d = 0, identical types).
The predictions below use the corrected statement. No data or simulation was examined for this correction; it is algebra.

Two further design fixes were made before any run. L7 was changed from a correlation pooled over truths, which ties would dilute, to
monotonicity within a truth, which is the theorem's claim. I2 was changed from a 2x2 game in which hiding is dominant (so nothing could
cycle) to the standard inspection game.

## What is known before registration
- From the main note (Proposition pairwise, check SC7): the rank of pairs is 2m, and pairs plus one triple gives 2m + 1, for m = 3, 4,
  5. Probabilities along the segment were checked to be affine (m = 3) and quadratic (m = 4) on one random truth, during the design
  discussion.
- The module `pce/lab.py` was smoke-tested on one invented truth, not a registered condition:
  - (w = 0.3, a1 = (.7, .6, .8, .5), a2 = (.2, .3, .1, .4));
  - exact recovery of v, the segment and r;
  - one sampled fit of v at n = 1e5.

## Amendment after freezing, before any run (2026-10-27)
The frozen version required the exact VOI to be positive for G and A. A smoke test on the invented toy truth (not a registered
condition) returned 0 for every triple. The cause is a property of minimax VOI, not of the theorem: perfect information is worth nothing
when one act is optimal along the whole segment. Positivity would therefore test the decision threshold, not Theorem C. L6 now tests the
zero pattern and the common value instead, and the code was changed to match. No registered condition had been computed. Both files
were re-frozen.

## Conditions (m = 4, variables 0-3; registered parameters)
| Name | Description | w | a1 | a2 |
|---|---|---|---|---|
| G | generic, mixed signs | 0.30 | (.70, .30, .80, .55) | (.20, .65, .10, .25) |
| A | agreement on variable 3 | 0.30 | (.70, .30, .80, .55) | (.20, .65, .10, .55) |
| D | dominance (all d > 0) | 0.30 | (.70, .60, .80, .55) | (.20, .30, .10, .25) |
| M | merge (identical types) | 0.30 | (.45, .40, .50, .35) | (.45, .40, .50, .35) |
| X | absent actor (negative control: the same law as M) | 0 | any | (.45, .40, .50, .35) |

**Decision problem (as in Section 17 of the note).**
- Event E = any harm among variables 0, 1, 2.
- Acts: "treat", u = 1 - 1.2 [E]; "nothing", u = 0. Requirement u0 = 0.05.
- The minimax VOI of a probe S, with exact data and the worst case over truths on the segment, is
  VOI(S) = min_x max_a E_x u_a - max_a min_x E_x u_a if v_S != 0, and 0 otherwise.

## 11a Design check (exact)
- **Q1.** Pair rank = 2m, pairs + one triple = 2m + 1, for m = 3, 4, 5 (as SC7, re-run here).
- **Q2.** For G, A and D:
  - P(E) is affine along the segment: linear-fit residual < 1e-10.
  - P(all four harmed) is quadratic and not affine: quadratic residual < 1e-10, linear residual > 1e-6.
- **Q3.** For G, A and D, the extremes of P(E) over the segment lie at its endpoints.

## 11b Relations (exact tables; deterministic perturbations of the pair moments)
**Protocol.** Replace m_ij by m_ij + s g_ij, with g_ij = ±1 fixed (seed 11) and s in 10^{-6}, ..., 10^{-3} (7 values). Refit v. The
exponent is the slope of log(response) against log(s).

**Predictions**
- **R1.** G and D: the response ||v_hat(s) - v|| has exponent in [0.9, 1.1].
- **R2 (corrected).** A: the response |v_hat_3(s)| has exponent in [0.9, 1.1]. Agreement is regular.
- **R3.** M: the response ||v_hat(s)|| has exponent in [0.4, 0.6].
- **R4.** X has exactly the tables of M (difference < 1e-12), so its responses are identical. The absent actor cannot be separated from
  the merge by any perturbation of the data.
- **R5 (null).** Over 200 random generic truths (w ~ U(.15, .45), a ~ U(.1, .9), |d_i| > 0.1), the displacement exponents lie in
  [0.85, 1.15] in at least 99%. Detection rule: exponent < 0.75 means singular (merge). M is detected; G, A and D are not.
- **R6 (support map, A).** The sensitivity of v_hat_3 to pair windows not containing variable 3 is < 1e-6, and to windows containing it
  is > 1e-2. The agreement rests on exactly the windows that contain the variable.

## 11c The probe loop (sampling; each window an independent sample of size n; 300 replications; n in 300, 1000, 3000, 10^4, 3x10^4, 10^5)
- **L1 (positive control).** G with pairs + triple (0,1,2): the slope of RMSE(r_hat) against n lies in [-0.6, -0.4].
- **L2 (tempo zero).** G with pairs only: the mean estimated segment width in r at n = 10^5 is within 10% of the true width, and the true
  width is > 0.5. The weight is not learned, however large n is.
- **L3.** A: the slope of RMSE(v_hat_3) lies in [-0.6, -0.4].
- **L4.** M: the slope of the mean ||v_hat|| lies in [-0.32, -0.18] (the n^-1/4 rate).
- **L5 (probe choice).**
  - For G, the rule argmax_S |v_hat_S| over the 4 triples agrees with the oracle argmax |v_S| in >= 90% of replications at n >= 10^4.
  - For A, the rule picks no triple containing variable 3 in >= 95% at n >= 3000.
- **L6 (exact VOI; amended before any run, see below).**
  - A: VOI = 0 (< 1e-9) for the triples containing 3. The triples avoiding 3 share one common value g, the minimax gap of perfect
    information on the segment, g = min_x max_a - max_a min_x (>= 0, and zero iff one act is optimal along the whole segment).
  - G: all four triples share one value.
  - M: all are 0.
  - In every case VOI >= 0. The value of g is reported.
- **L7 (graded VOI).**
  - With n = 1000 per probe, the realised VOI uses the post-probe interval r_true ± 2 se, intersected with the segment, where
    se = sqrt(E[((x - mu)_S)^2] / n) / |v_S|.
  - Over 300 random truths (as R5), take those where some triple has VOI > 0. Within each truth, the Spearman correlation between VOI and
    |v_S| across its 4 triples, averaged over truths (ties dropped), is >= 0.8. The theorem predicts monotonicity within a truth. The
    correlation pooled over truths is reported, not predicted.
- **L8 (negative VOI at level one).** In the drug example of the note, adding the BC window from a second population with type weights
  (pi, 1 - pi), for pi on a grid of 21 values in [0, 1], lowers the robustness of the best act for at least one pi.

## 11d The disclosure game (m = 3; 2000 random truths; seed 12)
**Truths.** w ~ U(.03, .5), a1 ~ U(.2, .9)^3, a2 ~ U(.02, .35)^3. A truth is harmful iff P(E) > tau, with tau = the median of P(E) over
the draws.

**Sponsor strategies**
- F: full (pairs + triple).
- P: pairs only.
- Q: pooled pairs. The pair tables come from a 50/50 mixture with a benign product population (harms .05), reported as the target, with
  no triple.
- S: selective. The sponsor runs the triple and discloses it iff disclosure weakly raises its approval under the regulator's policy.

**Regulator policies**
- N: naive. Approve iff the segment's midpoint prediction is <= tau, or the exact value when a triple is shown.
- K: skeptical. Approve iff the worst case over the segment is <= tau.
- M: mandated triple. It is always shown.
- R: registration. Every trial run must be disclosed, and pooling is disallowed, so the true-population pairs are used; then K.

**Predictions**
- **D1.** (P, N): false approvals > 0.
- **D2.** (P, K): false approvals = 0 and false rejections > 0.
- **D3.** Policy M, and strategy F under any policy: false approvals = false rejections = 0.
- **D4.** (Q, K): false approvals > 0. Pooling defeats worst-case skepticism.
- **D5.** (Q, R): rates equal to those of (P, K).
- **D6 (unravelling).** (S, K): every safe sponsor discloses, and false approvals = false rejections = 0.
- **D7.** (S, N): false approvals > 0.

## Illustrations (reported, with direction predicted; labelled illustrative in the paper)
- **I1 (RL).**
  - A contextual bandit chooses which triple to probe, over 3000 episodes of random truths (as R5, half with one coincident variable).
  - Features: |v_hat_S| from pairs at n = 3000. Reward: realised VOI as in L7.
  - Policy: epsilon-greedy linear (epsilon = 0.1, decaying).
  - Prediction: the mean reward over the last 500 episodes is >= 95% of the closed-form rule's mean reward and above a random choice.
- **I2 (adversarial dynamics; a stylised inspection game).**
  - The regulator chooses trust or skeptic; skepticism costs k = 0.3 (the delay a mandated triple imposes on every sponsor). The harmful
    sponsor chooses hide or honest.
  - Regulator payoffs: (hide, trust) = -1 (harm reaches patients); (hide, skeptic) = -k; (honest, trust) = 0; (honest, skeptic) = -k.
  - Sponsor payoffs: (hide, trust) = +1; (hide, skeptic) = -0.5 (caught); (honest, ·) = 0.
  - Prediction: simultaneous gradient play on mixed strategies (step 0.05, 2000 steps) does not approach the interior equilibrium (the
    final distance is >= the initial distance). With the regulator committing first (a Stackelberg leader, as pre-registration does), the
    play reaches the commitment outcome.
- **I3 (catch-up).** After a false approval in 11d under (P, N), each post-market round reveals the triple with probability p = 0.2. The
  mean number of rounds to detection is in [4, 6] (a check of the simulation: geometric with mean 1/p = 5).

## Interpretation fixed in advance
- **A pass** means the theorems' quantitative claims hold on the registered conditions.
- **A failure** of R2, L3 or L4 bears on Theorem B.
- **A failure** of L5, L6 or L7 bears on Theorem C.
- **The disclosure predictions** are mostly direct consequences of the definitions. They test the implementation and give magnitudes.
  Their substantive content is D4: skepticism over the segment does not protect against a mis-specified population.
- **I1 and I2** are illustrations for ML readers. They are not evidence for the theorems.
