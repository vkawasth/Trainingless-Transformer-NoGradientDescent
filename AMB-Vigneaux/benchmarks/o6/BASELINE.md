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

## Reproduce
```
python o6_generate.py --out runs/V16L4 --depth 4 --nsym 16 --nleaf 24 --nrules 3 --reverse-valid --n-train 20000 --n-test 500 --seed 0
python baseline_splitmerge.py --data runs/V16L4 --json results/V16L4_full.json      # also writes .npz
python o6_score.py --data runs/V16L4 --model results/V16L4_full.npz
```
