# O6 baseline: gradient-free EM + split–merge, and where it breaks

**Hardware.** 2 CPU cores, 7 GB RAM, no GPU, numpy/BLAS.
**Method.** `baseline_splitmerge.py`: dense inside–outside EM with a spectral/clustered initialisation, then
split–merge with merges chosen by Bregman information and accepted only if val likelihood improves. The inside pass
is rescaled per node (an earlier unscaled version underflowed at L = 10; that run is excluded).
**Protocols.**
- *full*: warm-up of 60 sweeps, then 3–4 rounds × every level × (25 split + 25 post) sweeps.
- *short*: warm-up of 30 sweeps, then 1 round × (10 + 10) sweeps.

**Grammar.** Binary RHM with m = 3 rules per symbol, reverse-valid, and nleaf = 1.5·V. The data are the ladder
configurations generated with `o6_generate.py` at seed 0.

All gaps are in nats per sequence on the **test** split: ll(true grammar) − ll(model). Recovery is the NMI of the MAP
symbol against the true latent symbol, per level from level 1 (bottom) to the root, reported as model / Bayes
ceiling. The ceiling is the true grammar's own MAP recovery.

## Axis 1: vocabulary V (L = 4, N = 20,000 sequences)

| V | protocol | s/sweep | peak mem | obs/param (top) | test gap | recovery, levels 3 and 4 (model / ceiling) |
|---|---|---|---|---|---|---|
| 8 | full | 0.50 | 0.2 GB | 39 | 0.14 (val) | – |
| 8 | short | 0.48 | 0.2 GB | 39 | 0.575 | 0.57 / 0.88, 0.32 / 0.68 |
| 16 | full | 1.37 | 0.4 GB | 4.9 | **0.020** | 0.81 / 0.98, 0.55 / 0.88 |
| 32 | short | 8.3 | 0.9 GB | 0.61 | 0.150 | 0.87 / 1.00, 0.69 / 0.98 |
| 32, N = 80k | short | 34 | 2.5 GB | 2.4 | 0.237 | 0.87 / 0.99, 0.66 / 0.97 |
| 64 | short | 40 | 3.9 GB | 0.076 | **1.488** | 0.89 / 1.00, 0.72 / 0.99 |
| 128 | one sweep only | 180 | 4.3 GB before any split | 0.0095 | – | – |

- Time per sweep grows as **V^2.17**, fitted over V = 8–128.
- **Memory is the hard wall on this machine at V = 128.** Splitting doubles one level's alphabet, and at V = 128 that does not fit in 7 GB.
- **At V = 32, four times more data did not help.** The gap went from 0.150 to 0.237 and recovery was unchanged. The limit there is optimisation (basins), not data.

## Axis 2: depth L (V = 8)

| L | N | protocol | test gap | recovery at the top levels (model / ceiling) |
|---|---|---|---|---|
| 6 | 20,000 | short | 4.03 | L5 0.18 / 0.87, L6 0.08 / 0.74 |
| 6 | 20,000 | **full** | **0.005** | L5 0.66 / 0.87, L6 0.39 / 0.74 |
| 8 | 1,250 | short | 15.2 | L7 0.06 / 0.88, L8 0.08 / 0.77 |
| 8 | 1,250 | **full** | 2.11 | L7 0.66 / 0.88, L8 0.41 / 0.77 (levels 2–6 within 0.04 of ceiling) |
| 10 | 312 | short | 60.7 | L8–L10 0.04, 0.01, 0.03 / 0.90, 0.83, 0.60 |

- Depth is the harder axis. The full protocol repairs likelihood at L = 6, with a gap of 0.005. It does **not** repair the top two levels, and the shortfall grows with L.

## Findings

1. **Likelihood does not certify structure.** At V = 16 the gap is 0.02 while level 4 recovers only 0.55 of 0.88. At L = 6 the gap is 0.005 while level 6 recovers 0.39 of 0.74. The top of the hierarchy contributes little likelihood, so it has to be scored directly.
2. **The baseline breaks in three places:**
   - optimisation at V ≈ 64 (L = 4), where the gap opens to 1.5 nats;
   - memory at V = 128;
   - top-level structure from L ≈ 6–8, where the top two levels stay well below the ceiling even with the full protocol.
3. **It cannot run O6-S** (V = 64, L = 8) on this hardware. The estimate is about 680 s per sweep and around 1,000 sweeps for the full protocol, i.e. more than a week.
4. **O6-L with dense tables is out of reach for any method.** Dense tables hold 1.8·10¹⁰ parameters. Top-level tables see about 10⁻⁴ observations per parameter at N = 10⁵. One sweep costs about 2·10¹⁴ flops per sequence. A sparse or structured parameterisation is a precondition, whatever the optimiser. The true grammar has 3,072 rules per level.

## Identifiability by level: distance to the merge strata (`merge_strata.py`)
Δ_l(s, s′) = held-out log p(true) − log p(true with symbols s, s′ merged at level l), in nats per sequence, with a
paired se over 4,000 sequences. Pairs involving symbols the grammar never uses are excluded from the medians.

| grammar | level 1 | 2 | 3 | 4 | 5 | 6 (root) |
|---|---|---|---|---|---|---|
| V = 8, L = 6: median Δ (min) | 5.07 (1.37) | 2.28 (0.93) | 0.99 (0.43) | 0.37 (0.07) | 0.13 (0.035) | **0 (exact, all 28 pairs)** |
| V = 16, L = 4: median Δ (min) | 0.69 (0.17) | 0.27 (0.13) | 0.14 (0.0016) | **0 (exact, all 40 pairs)** | | |

Every pair of *used* symbols below the root is resolvable (Δ > 2 se, where se ≤ 0.02). The root is not identifiable,
and Δ falls by a factor of about 2.5–3 per level toward the top. Reading finding 1 in this light:
- the top-level shortfalls in the tables above are at the **root**, which no method can recover;
- at L = 6 the likelihood gap left by EM (0.005 nats) is below every level-5 merge distance (≥ 0.035), so the
  level-5 shortfall is slow EM along weakly curved directions, not a likelihood blind spot;
- at V = 16, L = 4 one level-3 pair costs only 0.0016 nats, which is below the 0.02 gap. The likelihood genuinely
  cannot separate that pair at this sample size.

## Local window EM (bottom-up, random starts) and the hybrid (`local_em.py`)
Idea: learn one level at a time, bottom up, from random Dirichlet starts, looking only at a local neighbourhood.
- **A lone node is not enough.** Its child pair is one categorical draw, and a mixture of single draws is not
  identifiable: merging any two symbols with usage weights is exactly free (`tests/test_root_identifiability.py`).
  The smallest identifying neighbourhood is a **two-level window**: parent → (left, right) → child pairs. The
  sibling is the second view.
- **Annealing collapses.** Tempered E-steps from T0 = 2 drove every symbol to the same table (the symmetric fixed
  point, exactly), and EM cannot leave it. Level-1 NMI was 0.05. Random starts at T = 1 break the symmetry.
- **Random restarts land on merge strata.** At levels 3–5 the fitted symbols duplicate each other (held-out merge
  cost ≈ 0). A targeted local split–merge fixes part of this: merge the redundant pair the diagnostic finds, split
  the most loaded symbol, and keep the change only if held-out window likelihood improves.
- **The hybrid** takes the local solution as the start for the global split–merge, with a *shortened* protocol
  (10 warm-up sweeps, 2 rounds instead of 60 and 4).

NMI on the identifiable levels (1 … L−1), test split; N = 20,000; 2 CPUs.

| V = 8, L = 6 | L1 | L2 | L3 | L4 | L5 | test gap | seconds |
|---|---|---|---|---|---|---|---|
| baseline (clustered init, full split–merge) | 0.909 | 0.970 | 0.957 | 0.794 | 0.655 | 0.005 | 3,007 |
| local window EM only (4 restarts, local SM) | 0.901 | 0.894 | 0.663 | 0.398 | 0.229 | 5.71 | 1,850 |
| local + 60 global EM sweeps | 0.948 | 0.923 | 0.809 | 0.593 | 0.413 | 1.75 | 2,106 |
| **hybrid** (local + short split–merge) | **0.963** | 0.970 | 0.957 | 0.768 | **0.668** | 0.049 | 1,850 + 1,949 |
| Bayes ceiling | 0.993 | 0.970 | 0.956 | 0.866 | 0.872 | 0 | |

| V = 16, L = 4 | L1 | L2 | L3 | test gap | seconds |
|---|---|---|---|---|---|
| baseline | 0.983 | 0.986 | 0.810 | 0.020 | 1,451 |
| local window EM only (3 restarts, local SM) | 0.978 | 0.934 | 0.748 | 0.79 | 2,178 |
| local + 60 global EM sweeps (no split–merge) | **0.987** | 0.981 | 0.815 | 0.28 | 2,466 |
| **hybrid** | 0.977 | 0.982 | **0.832** | 0.021 | 2,178 + 962 |
| Bayes ceiling | 1.000 | 1.000 | 0.983 | 0 | |

Findings:
- **As an initialiser, local EM helps.**
  - After 10 warm-up sweeps the hybrid is at a validation gap of 2.47 nats (V = 8, L = 6). The clustered init
    needs 60 sweeps to reach 5.85.
  - With half the split–merge budget the hybrid matches or beats the baseline's structure on most identifiable
    levels: L1 0.963 vs 0.909 and L5 0.668 vs 0.655 at L = 6; L3 0.832 vs 0.810 at V = 16. The exceptions are
    L4 at L = 6 (0.768 vs 0.794) and L1 at V = 16 (0.977 vs 0.983).
  - At V = 16, local EM plus 60 plain global EM sweeps, with *no* global split–merge, already equals the baseline's
    structure on levels 1–3.
- **On its own, it does not reach the top.** Going only deeper discards the top-down (outside) evidence, and
  errors compound upward. Local-only NMI falls from 0.90 at L1 to 0.23 at L5 at L = 6.
- **Cost.** The window E-step is O(K³) per node in numpy. The local fits cost more than the clustered init, so the
  total time is above the baseline's. Reducing the window cost (sparse parent tables, fewer restarts at lower
  levels) is the obvious next step.
- The merge-cost diagnostic is a useful depth signal: it flags every level where the fit has redundant symbols.

## Reproduce
```
python o6_generate.py --out runs/V16L4 --depth 4 --nsym 16 --nleaf 24 --nrules 3 --reverse-valid --n-train 20000 --n-test 500 --seed 0
python baseline_splitmerge.py --data runs/V16L4 --json results/V16L4_full.json      # also writes .npz
python o6_score.py --data runs/V16L4 --model results/V16L4_full.npz
python merge_strata.py --data runs/V16L4 --n-val 4000 --max-pairs 40 --json results/merge_strata_V16L4.json
python local_em.py --data runs/V16L4 --n-train 20000 --restarts 3 --iters 40 --sm-rounds 3 --polish 60 --json results/local_V16L4_N20000.json
python baseline_splitmerge.py --data runs/V16L4 --n-train 20000 --init-npz results/local_V16L4_N20000.npz --warmup 10 --rounds 2 --json results/hybrid_V16L4.json
python o6_score.py --data runs/V16L4 --model results/hybrid_V16L4.npz
```
