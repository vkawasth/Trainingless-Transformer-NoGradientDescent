# Study 17 results: learned sections in synthetic worlds (pre-registered analysis, run once, 2026-11-06)

Frozen files verified before scoring (`FROZEN_PREREG_study17.txt`: both OK). 154 of 154 models were trained, conditions 0-153, and
`runs17.jsonl` is the concatenation of the two training processes. The output is `study17_final.json`, and the figure is
`fig_study17.pdf`, a plotting script that reads that output.

## Theorem checks: all pass
| check | statement | result |
|---|---|---|
| T1 | Pythagorean identity, relative error < 1e-6, all G1 and C-path models | holds |
| T2 | on the CH facet (ε = 0), TV(P*, P_true) < 1e-6: the fibre is a point | holds |
| T3 | no joint law (G2, CHSH path): mismatch ≥ M* for every model | holds |
| T4 | |q(P*) − q(P_true)| ≤ width of the identified interval | holds |

## Empirical hypotheses
| | hypothesis | result | numbers |
|---|---|---|---|
| H1 | which section learning picks | **maximum-entropy-like** | ι < 0.5 in 94% of the 48 G1 models (mean ι ≈ 0.2) |
| H2 | hallucination from shadows is mostly in-fibre | **not supported** | in-fibre > off-fibre in 54% of G1 models at N = 3000; mean 0.0116 against 0.0116 nats |
| H3 | rigidity lowers total error in practice | **not supported** | mean TV(P_true, Q) at ε = 0: CH facet 0.271, structural zero 0.188 |
| H4 | the implied joint depends on generation order | **supported** | drift > 0.01 in 100% of G1 models (mean 0.074 at N = 300, 0.059 at N = 3000) |
| H5 | learned composition respects independent systems | **not supported** | tensor-square defect below 0.0188 in 0 of 12 cases (median 0.0385; range 0.020-0.072) |
| H6 | shadows from separate cohorts make the model less self-consistent | **supported** | drift G2 0.222 against G1 0.066; 95% CI of the difference [0.115, 0.213] |

## Exploratory (no inference)
**E5, the constructed paths** (mean of 2 seeds; KL in nats, then TV(P*, P_true) and identified width):

| ε | CH in-fibre | CH off-fibre | CH TV(P*, P) | CH width | zero in-fibre | zero off-fibre | zero TV(P*, P) | zero width |
|---|---|---|---|---|---|---|---|---|
| 1/5 | 0.0159 | 0.175 | 0.042 | 0.182 | 0.0187 | 0.038 | 0.082 | 0.672 |
| 1/20 | 0.0053 | 0.287 | 0.014 | 0.045 | 0.0248 | 0.079 | 0.089 | 0.612 |
| 1/100 | 0.0013 | 0.356 | 0.003 | 0.009 | 0.0249 | 0.113 | 0.095 | 0.596 |
| 0 | 0 | 0.359 | 0 | 0 | 0.0242 | 0.127 | 0.093 | 0.592 |

**E3, data with no joint law.**
- CHSH path (λ = 3/5, 4/5, 1): mismatch 0.35, 0.73, 1.55 against the bounds M* = 0.03, 0.29, 1.15; drift 0.34, 0.49, 0.69.
- G2: excess mismatch over M* is 0.069 at N = 300 and 0.053 at N = 3000.
- The effect of N is small throughout.

**E4, learned composites (24 chain models)** have mean TV 0.025 to the Markov composite, 0.047 to the selected composite and 0.044 to
the true composite.

## Reading (not part of the pre-registered inference)
1. **The theory survives contact with learned models.** All four theorem checks hold on 154 trained transformers. In particular, the
   coherent-but-unsupported part of the error falls to zero on the Clauser-Horne facet in step with the shrinking fibre, as Theorem 76
   requires: 0.042 → 0.014 → 0.003 → 0.
2. **Learning picks a maximum-entropy-like section (H1).** Trained only on shadows, the models complete the unobserved structure close
   to the maximum-entropy completion and far from leximin. That gives an empirical answer, for this model class, to which section
   learning picks.
3. **Rigidity does not protect the model (H3, E5).** As the data approach the CH facet, the data allow fewer and fewer joints, but the
   models cannot find the one that is allowed: the off-fibre part, inconsistency with their own training data, grows from 0.17 to
   0.36. At a structural zero the in-fibre part stays at 0.024, with a 4-dimensional fibre, while inconsistency grows less. Near the
   contextuality boundary, then, the model's error moves entirely into self-inconsistency. This is the opposite of what H3 predicted.
4. **Self-inconsistency tracks how real the shadows are (H4, H6).** Even on data from one world, the implied joint depends on
   generation order (drift about 0.06). Shadows from separate cohorts triple it (0.22), and contextual data drive it to 0.69 at the
   PR box. Order drift is a direct, ground-truth-free signal that the shadows do not come from one world.
5. **Maximum-entropy-like is not maximum entropy (H5).** Learned composition breaks the tensor square: median defect 0.039, about half
   the leximin value. The learned composite is closest to the Markov composite (TV 0.025), but not equal to it. Part of the defect may
   be fitting error in the 64-cell product model. Separating the two would need a follow-up with larger models or longer training.

## Limits
These results concern tiny NoPE transformers (2 layers, width 64, 4000 steps) on binary 4-variable worlds and 3-variable chains. They
do not transfer to LLMs without further study. H3 and the C paths use 2 seeds per point.
