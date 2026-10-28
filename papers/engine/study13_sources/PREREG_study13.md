# Study 13 — a trusted source that is sometimes adversarial (pre-registration)

Vinay K Awasthi, Johns Hopkins University. Registered 2026-10-28, before any registered design was simulated. Frozen with SHA-256 in
`FROZEN_PREREG_study13.txt` together with `run_study13.py`, `design13.py`, `../pce/sources.py`, `../pce/segment_decision.py`,
`../pce/injection.py` and `../pce/lab.py`.

## What was done before freezing
- Engine checks SR1–SR10 (`../check_sources.py`). They use random truths and an invented truth I, and, on the registered truth W,
  only the Gaussian (delta-method) model, never sampled batches.
- Deterministic predictions (`run_study13.py --predict`): the Gaussian model, no sampling of batches.
- Two smoke runs (`run_study13.py --smoke`, 600 replicates per cell) on an invented truth I (w = 0.45, a1 = (.75, .25, .70),
  a2 = (.25, .70, .20); acts: approve the triple, u = 1 − [any harm], against a flat act; u0 slack). In the first, the flat act sat
  0.3 below the truth on the triple's line and A2 used W's share 0.18: everything settled at the smallest N, so nothing was tested.
  For the second, the flat act was moved to 0.08 and A2's share was made design-relative (1.5 (ε* + ε_det), which is 0.181 on W; W keeps
  its registered 0.18). Second smoke run: H1 and H3 held (99% of cells within 0.05; N½ ratios 1.01, 1.05, 0.91). H2, H4 and H5 failed on
  I, for reasons specific to I: on I, A1b's exposure-½ shift gives a wrong-rate of 0.09 at the smallest N; one A3-vs-A0 gap was 0.053
  at 600 replicates; and on I the grid ends before A1b settles. The criteria were not changed after the smoke runs. The registered
  truth W was not simulated.
- While writing the engine, the order of the two filters was changed (sources-then-batches, then batches-then-sources with a median
  reference) on the strength of Gaussian-model predictions on W, which showed that a leave-one-out batch test with a contaminated
  reference drops genuine batches on the far side (swamping) and so biases the pool toward the adversary. No sampled data on W
  informed this.

## Model
Truth W is the worked example of the lab paper (Section 17): w = 0.6, a1 = (.5, .1, .4), a2 = (.05, .5, .1). The decision: approve the
triple (T: u = 1 − [any harm]) or the pair AB (P: u = 0.872 − [A or B harmed]), requirement u0 = 0.30; the oracle chooses T; the
switch is at triple harm 0.668 (the truth is 0.667).

Three sources, G1, G2 and C, send K_C = 8 batches each (K = 24). A batch is the triple window (the probe) with n records and the three
pair windows with 25 n records each (pair tables come from large registries). All windows are sampled independently. The decision
maker sees each table's label (source, batch) and nothing else about provenance.

Source C is trusted and mostly genuine. In some of its batches a share ε_b of the records of every window is drawn from the hidden
channel Q = P + sχ, s = 0.027 (the positivity limit toward the switch). Q leaves every single and pair table unchanged and moves the
triple moment m_T by ε_b s. **The adversary knows P, n, K and the decision maker's thresholds; it does not know the sampling noise.**

## Rules (all four on the same data)
- *naive*: pool every table.
- *glue*: the gluing filter. (1) Fine cover: each batch's triple moment against the median of the three source means; triple
  sections with |z| > c_b = 3.078 (Bonferroni over 24 batches, family-wise 0.05) are dropped. (2) Coarse cover: each source's pooled
  triple moment (kept batches) against the other two; if the largest |z| exceeds c_s = 2.394 (Bonferroni over 3 sources), that
  source's remaining triple sections are dropped. A source whose batches were all dropped is not tested; at most one source is
  dropped. Pair sections are always kept. The decision is taken on the colimit of the remaining sections.
- *drop_C_triple*: oracle benchmark that knows C and drops its triple sections.
- *write_off_C*: drops every table of C.

Each rule runs the regulator's pipeline (pair tables → μ, v, R; pooled triple → r̂; interval [r̂ ± z τ̂_eff] ∩ R with z = 1.96 and
τ̂_eff² = (var_m / v_T²) / N_triple + r̂² var_logv / N_pair; the set rule). Outcomes: *right* (the set rule settles on T), *wrong*
(P is better than T on the whole interval: a tilt), *open* (the interval straddles the switch: a delay), *flag* (r̂ outside R).

## Adversaries
- A0: none.
- A1: sometimes, just enough: 4 of C's 8 batches, each shifting m_T by 2D, with D chosen so that C's expected source statistic is
  a = c_s − Φ⁻¹(0.8) = 1.552 (exposure risk about 0.2).
- A1b: as A1 with a = c_s (exposure risk about one half).
- A3: concentrated: the source shift of A1 in a single batch (8D).
- A2: tuned to naive pooling: share 0.18 in every C batch (pooled share 0.06, between ε* = 0.037 and ε_det = 0.083; on other truths
  the design uses 1.5 (ε* + ε_det)).

Grid: n = 500, 1000, 2000, 4000, 8000 triple records per batch (N = 24n = 12k … 192k). 2000 replicates per cell; seed 1313.

## Theory (Theorem J, lab paper)
A source of share π whose expected gluing statistic is a moves the pooled m_T by a·√(π/(1−π)) pooled standard errors (B = 1.69 at
a = c_s, 1.10 at a = 1.552, π = 1/3). The shift does not shrink relative to the noise as N grows: the adversary rides inside the
band. Hence (no tilt) the settled-wrong rate under the filter is at most Φ(B − z), and it vanishes as N grows; (delay) the settling
point moves from (zτ/d)² to at most ((zτ + Bσ_m/|v_T|)/d)²; under naive pooling a fixed share between ε* and ε_det settles wrong with
probability tending to one.

## Predictions (Gaussian model, `run_study13.py --predict`, 200000 draws, seed 7)
- **H1 (calibration).** Over the 100 cells (5 adversaries × 5 N × 4 rules) the empirical right-rate is within 0.05 of the prediction in
  at least 90% of cells, and within 0.08 in all.
- **H2 (no tilt under the gluing filter; tilt under pooling).** The glue rule's wrong-rate is at most 0.05 in every cell. Under A2 the
  naive rule's wrong-rate is at least 0.75 at N = 192k. Under A2, at N ≥ 48k, the glue rule's right-rate is within 0.05 of the oracle
  rule drop_C_triple.
- **H3 (delay, the band).** For A0, A1 and A1b, the N at which the glue rule's right-rate crosses one half (log-linear interpolation on
  the grid) is within 15% of the predicted one. Under A1b the oracle drop_C_triple has a right-rate at least the glue rule's at every
  N ≥ 24k (the price of not knowing which source).
- **H4 (localisation; the source is not written off).** Under A3 the adversarial batch is removed in at least 90% of replicates at
  every N, C's triple sections are dropped wholesale in at most 10%, and the glue rule's right-rate is within 0.05 of its A0 value at
  every N. Under A0 some source is dropped in at most 8% of replicates, and the glue rule's right-rate is within 0.04 of naive pooling
  at every N.
- **H5 (worst case).** At N = 192k the glue rule's smallest right-rate over the five adversaries is at least 0.85, and the naive rule's
  is at most 0.05.

## Reporting
`run_study13.py` writes `results_study13.json` with an `evaluation` block (H1–H5) and the per-cell quantiles of r̂ and of the
apparent triple harm (the bands), and of C's source statistic. All outcomes are reported, failures included. Deviations are logged in
`DEVIATIONS_study13.md` before the affected result is reported.
