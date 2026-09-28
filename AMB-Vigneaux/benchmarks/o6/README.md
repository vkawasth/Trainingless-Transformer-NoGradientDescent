# O6: learning a deep hierarchical grammar at scale

Open problem O6 of *Three layers of realisation* (Awasthi): recover a Random-Hierarchy-Model grammar with
**V = 1024 symbols per level and depth L = 16** from unlabelled leaf sequences. The hierarchy has to be
**learned**. No hand-designed codes are allowed, and no latent labels may be used for training.

## Generator

`o6_generate.py` is a binary Random Hierarchy Model. Each level-l symbol has m ordered productions into pairs of
level-(l-1) symbols. The option `--reverse-valid` makes (y, x) a production of a *different* parent whenever (x, y) is one, so order
carries information. A sequence has 2^L leaves. Train, val and test are disjoint, and the true latent symbol of
every test node is exported for scoring.

```
python o6_generate.py --out DIR --depth L --nsym V --nleaf 1.5V --nrules 3 --reverse-valid \
       --n-train N --n-val NV --n-test 500 --seed 0
```
Files: `train_ids.json` (or `train_ids.npy` with `--npy`, recommended for O6-L), `val_ids.json`, `test_ids.json`, `test_latents.json`, and `rhm_meta.json`, which holds the true rules.
Use `val` for any model selection. `test` is scored once.

## Tiers

| tier | V | L | leaves/seq | N train (seqs) | tokens | note |
|---|---|---|---|---|---|---|
| O6-S | 64 | 8 | 256 | 20,000 | 5.1M | beyond our baseline on 2 CPUs (est. >1 week) |
| O6-M | 256 | 12 | 4,096 | 20,000 | 82M | |
| O6-L (target) | 1024 | 16 | 65,536 | 100,000 | 6.6B | generate locally from the seed |

Why these sizes? A dense parameterisation has V·c² parameters per level: 2.4·10⁹ at level 1 of O6-L and
1.1·10⁹ above that. The top level is observed only N times, so dense tables are unidentifiable at the top for any
feasible N (9·10⁻⁵ observations per parameter at N = 10⁵). The true grammar has only V·m = 3072 rules per level,
about 33 observations per rule at the top when N = 10⁵. A method therefore needs a sparse or structured
parameterisation as well as an optimiser.

## Scoring

`o6_score.py --data DIR --model submission.npz` takes one tensor `P{l}[parent, left, right]` per level, with any
alphabet size. It reports, on the test split:
- `gap` = ll(true grammar) − ll(model), in nats per sequence;
- per level, the NMI and matched accuracy of the model's MAP symbol against the true latent symbol, together with the
  same numbers for the true grammar. The true grammar's own MAP recovery is the Bayes ceiling; it is below 1 at high
  levels.

**Structure recovery is the primary metric; likelihood is secondary.** Our baseline reaches a likelihood gap of
0.02 nats at V = 16 while its level-4 recovery is 0.55 NMI against a ceiling of 0.88. Near-truth likelihood does
not certify that the hierarchy was recovered.

Report as well: wall-clock time, hardware, peak memory, and number of passes over the data.

## Baseline (gradient-free)

`baseline_splitmerge.py` runs dense inside–outside EM with a spectral/clustered initialisation and split–merge,
where merge cost is the Bregman information. It uses no gradients. Results are in `results/` and summarised in
`BASELINE.md`.
