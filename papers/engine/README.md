# Posterior Coherence Engine (`pce`)

Separates **uncertainty about the answer** from **failure of the evidence to admit a coherent global model**.
Report: `report/pce_report.pdf` (50 pp).

For a set of local reports the engine returns:

1. **Uncertainty**: the exact three-level split `H(rho) = between + within + noise` (`pce.split`).
2. **Coherence**: `coherent-unique`, `coherent-nonunique` (with fibre width), or `incoherent` (with consistency radius and a finite
   certificate found by Farkas duality) (`pce.coherence`).
3. **Pooling validity**: the Beck–Chevalley defect of a pooling rule and the KL-ball radius with its decision-error bound
   (`pce.defect`, `pce.kl_ball`).

`pce.decide` turns these into an action: answer, abstain, flag-incoherent, or flag-nonunique.

## Quick start

```bash
pip install numpy scipy scikit-learn matplotlib pandas
python check_pce.py                               # 7 checks, all pass
python check_icl.py                               # 8 checks for the in-context-learning algebra, all pass
python check_integrate.py                         # 8 checks for integrating expertise, all pass (about 40 s)
python check_complex.py                           # 8 checks for the context poset and gluability complex (about 30 s)
python check_cohomology_engine.py                 # 8 checks for Cech cohomology on the fibres (about 10 s)
python check_lift.py                              # 8 checks closing the lift (about 50 s)
python check_gs.py                                # 7 checks: Gerstenhaber-Schack cohomology and tolerances (about 10 s)
python check_places_engine.py                     # 7 checks: obstructions by place (supports, tropical, archimedean)
python check_overlap.py                           # 7 checks: the overlap algebra (about 90 s)
python check_overlap2.py                          # 7 checks: the overlap algebra at level two (about 2 min)
python check_cover.py                             # 8 checks: gluing over a cover of contexts (the crossing example)
python check_perturb.py                           # 8 checks: perturbing context and probability, how outcomes move
python check_strata.py                            # 8 checks: walls and chambers (where geometry, probability, context and outcome move together; rebuilt 2026-10-29)
python check_cover_strata.py                      # 8 checks: strata of two-type covers (CS1-CS8)
python check_ehrhart.py                           # 7 checks: the decision and the ring of contingency tables (EH1-EH7; needs Macaulay2)
python check_cbd.py                               # 7 checks: contextuality beyond direct influence; the Study 7 pipeline on simulated models
python check_toric.py                             # 8 checks: several experts as toric families (mixed volumes, mixed subdivisions, pooling, gluing)
python check_updates.py                           # 8 checks: non-commuting updates tracked by conservation of probability mass
python check_persist.py                           # 8 checks: gluability complex filtered by gradation, persistence, vineyards, critical slowing
python check_persist2.py                          # 6 checks: truncated complexes, independent parts and the join rule (16 windows)
python -m pce.cli examples/three_token.json       # incoherent evidence, with its certificate
python demos/synthetic.py                         # synthetic demonstrations (debug and illustration only)
```

```python
from pce import Window, coherence, split, decide
import numpy as np
t = np.array([[0.125, 0.375], [0.375, 0.125]])
r = coherence({"X0": 2, "X1": 2, "X2": 2},
              [Window(p, t) for p in [("X0", "X1"), ("X1", "X2"), ("X0", "X2")]])
print(r.status, r.radius, r.violation)            # incoherent 0.083 0.375

from pce import integrate
cur = [Window(("I", "T"), np.array([[0.45, 0.05], [0.05, 0.45]]))]                  # the doctor
new = [Window(("T", "E"), np.array([[0.45, 0.05], [0.05, 0.45]])),                  # the lawyer
       Window(("I", "E"), np.array([[0.20, 0.30], [0.30, 0.20]]))]                  # the insurer's file
r = integrate({"I": 2, "T": 2, "E": 2}, cur, new, protected={"T"})
print(r.verdict, r.lam, r.repairs)                 # repairable 0.6, four one-split repairs
```

## Contents

| Path | What it is |
|---|---|
| `pce/` | the package: `coherence.py`, `uncertainty.py`, `pooling.py`, `decide.py`, `votes.py`, `selective.py`, `cli.py` |
| `pce/icl.py`, `check_icl.py` | an executable algebra of in-context learning: context states, Bayes update, mixture, abstraction, behavioral equivalence, Hankel rank of a black-box model; ICL1–ICL8 check the minimal-state theorem and the update laws (report §6) |
| `pce/integrate.py`, `check_integrate.py` | integrating new expertise into a context: overlap structure, gluable fraction and residue (what did not agree), maximum-entropy merge and bits added, minimal repairs by splitting shared terms, and irreconcilability relative to protected variables with one witness for every repair; INT1–INT8 (report §7) |
| `pce/complex.py`, `check_complex.py` | the context poset (intersection closure, Hasse diagram), congruence on overlaps at levels one and two, the gluability complex with lam* per cell, fibres and their dimension, minimal obstructions, reduced Betti numbers, and the strict / fillable / hollow trichotomy for triangles of couplings; CX1–CX8 (report §8) |
| `pce/cohomology.py`, `check_cohomology_engine.py` | Cech cochains on the fibres: the level-one complex (higher cohomology vanishes; H^0 = signed global laws), the Efron-Stein signed extension, the Hodge split and sheaf-Laplacian energy of local disagreement, negativity nu*, the separation local / positivity / none, and mu as a cochain map from level two; CH1–CH8 (report §9) |
| `pce/lift.py`, `check_lift.py` | the lift criterion at levels one and two (coupling LP on gluable tuples, with certificate), the audit of the n-cycle theorem, and the coupling simplicial set (faces, degeneracies, horn filling; the parity 3-horn that does not fill); LF1–LF8 (report §10) |
| `pce/gs.py`, `check_gs.py` | Gerstenhaber-Schack cohomology of the diagram of function algebras over the context poset (full double complex and its collapse to the order complex with coefficients), the decomposition by Efron-Stein components, and tolerances (Lipschitz bound for lam*, Weyl rank rule on the secant variety); GS1–GS7 (report §12) |
| `pce/places.py`, `check_places_engine.py` | obstructions by place: support classes and All-versus-Nothing over Z2 (trivial point), the min-plus principal solution and tropical defect with the band delta <= log N (orders of magnitude), and the layer classification; PL1–PL7 (report §13) |
| `pce/overlap.py`, `check_overlap.py` | the overlap algebra: fibres and projective overlaps, maximal filled and minimal unfilled cells, removal sets; binary pair data and the cut polytope (filled triangles force a tetrahedron with uniform marginals, not with arbitrary ones); OV1–OV7 (report §14) |
| `pce/overlap2.py`, `check_overlap2.py` | the overlap algebra at level two: coupling polytope of the experts' hypotheses, forced matching, level-two merge versus the average-then-glue shortcut, tolerant matching, level-two-only obstructions; L2-1–L2-7 (report §15) |
| `pce/cover.py`, `demos/crossing.py`, `check_cover.py` | gluing over a cover of contexts: scopes, context-dependent senses, competence from probes as tolerances, graded gluing theta*, holes, robust (maximin) actions; the street-crossing example; CV1–CV8 (report §17) |
| `pce/perturb.py`, `check_perturb.py`, `report/fig_perturb.py` | how perturbations reach the outcome: tube radius (LP; certifies every merge rule inside it), fibre gap intervals (when the filler decides), bridge windows that pin a decision or expose a hidden error, window radius of the max-entropy merge, context-belief envelopes, tempering and the support skeleton (tropical defect grows like beta delta_inf); PT1–PT8 (report §18) |
| `pce/strata.py`, `check_strata.py`, `report/fig_strata.py` | walls and chambers: chamber complex of the marginal matrix (vertices affine inside chambers), tangent complex H0/H1/chi, filled cells change on facets of marginal polytopes, lam* kinks where the witness changes, robust outcome affine per chamber vs curved max-entropy outcome, context paths, Maslov dequantization (coherent region collapses onto the tropical set); ST1–ST8 (report §19) |
| `pce/cbd.py`, `check_cbd.py`, `study7_pairwise/` | contextuality beyond direct influence (Contextuality-by-Default): direct influence s, degree c (LP, agrees with the Kujala–Dzhafarov cyclic criterion), exact graded test theta_c; Study 7 kit (pairwise judgments in separate prompts with null, direct-influence, order and co-presentation controls), pre-registered and frozen, not yet run; CB1–CB7 (report §20) |
| `pce/toric.py`, `demos/toric_triage.py`, `check_toric.py`, `report/fig_toric.py` | several experts as toric families on one evidence space: joint indecision counted by mixed volumes (tropical Bernstein), joint decisions as cells of the mixed subdivision, product of experts vs linear pool (compromise outcomes), finite-confidence convergence, gluing as a toric fibre product (chain vs 3-cycle); the triage example; TO1–TO8 (report §21) |
| `pce/updates.py`, `check_updates.py` | non-commuting updates as mass flows: likelihood updates commute (log-evidence path-free), Jeffrey = I-projection moving TV(q, p_A), exact holonomy of two updates (factorised order effect), conserved interaction orthogonal to the cover, loops close iff the evidence glues (mass circulating >= (1 - lam*)/(n - 1) otherwise), curvature of linear transports certifies hidden state; UP1–UP8 (report §22) |
| `pce/persist.py`, `check_persist.py`, `report/fig_persist.py` | the gluability complex filtered by phi = 1 - lam* (or theta*): Z/2 persistence, hollow cells as bars (triangle H1, hollow 3-cell H2), bottleneck stability under changes of evidence, vineyards along a path, homology as a coned-off shadow of the minimal obstructions, update circulation per cell and critical slowing on walls; PE1–PE8 (report §23) |
| `pce/persist.py` (filtration_k, parts, join_diagram), `check_persist2.py`, `report/fig_persist2.py` | large complexes: truncated filtration (exact up to dimension kmax - 2), variable-disjoint parts, lam*(A u B) = min, join composition of persistence ([max b, min d) in dim i + j + 1), homology as a global AND; local vs global persistence on 16 windows; relation to Robinson's consistency filtration; PJ1–PJ6 (report §23) |
| `study6_llm/` | pre-registered kit for Study 6: two real open-weight language models as experts on separate case notes, tables read from token probabilities, level-one and level-two merges, frozen plan `PREREG_study6.md` / `FROZEN_PREREG_study6.txt`, `README_study6.md` with run instructions (needs `torch`, `transformers`; runs locally) |
| `study5/` | pre-registered Study 5: two toy Transformer experts trained on separate expertise (`toy_lm.py`, `worlds.py`), four combiners and exact metrics (`systems.py`), `run_study5.py`, `analyze_study5.py`, `fig_study5.py`, frozen plan `PREREG_study5.md` / `FROZEN_PREREG_study5.txt`, pilot and confirmatory outputs. Needs `torch` (CPU). |
| `check_pce.py` | exact checks against the paper (three-token model, certificates, split, defect, KL ball, decisions) |
| `demos/doctor_lawyer.py` | a worked case of mixing expertise: typed propositions, double counting, a common cause through a shared fact, a three-context incoherence with its certificate, and where disagreement sits |
| `demos/synthetic.py` | D1 averaging hides conflicts; D2 non-unique reconstruction; D3 where disagreement sits (with a null world) |
| `study4/` | pre-registered real-data test: `PREREG_engine.md` (frozen, SHA-256 in `FROZEN_PREREG_engine.txt`), `study4_analysis.py`, results |
| `report/` | `pce_report.tex/.pdf` and `fig_engine.py` |

## Study 6 (pre-registered, real language models)

Qwen2.5-1.5B (medical notes) and SmolLM2-1.7B (legal notes) as experts, tables read from token probabilities, 46 toy worlds, run
locally (`study6_llm/`, results in `study6_llm/confirm/`). H4 supported: the noise-calibrated tolerance flagged 4 of 5 incoherent
corrupted case files and 0 of 8 clean ones. H2 not supported on its second criterion: the level-two merge beat averaging first in the
hidden-regime worlds (0.0115 bits, 95% interval [0.0021, 0.0220]) but the regimes were matched exactly in only 6 of 10 worlds. H1 and H3
not supported: gluing beat the linear pool and B alone but tied the product of experts at these models' elicitation noise; the
population shift was detected perfectly (AUROC 1.00). Report §16.

## Study 5 (pre-registered)

Two tiny Transformers, one trained on (diagnosis, symptom, test) records and one on (test, testimony, liability), combined four ways on
48 toy worlds. Gluing through the shared test variable beats product of experts, linear pooling and B alone when Beck–Chevalley holds
(H1 supported) and under a population shift, which the engine detects every time (H3). An audit sample reveals Beck–Chevalley
failures (H2, AUROC 0.88). Corrupted case files were not reliably flagged (H4 not supported, AUROC 0.67): two corruptions were coherent
with the experts, and estimation noise in the models' tables blurs the threshold.

```bash
cd study5 && python run_study5.py --phase confirm && python analyze_study5.py && python fig_study5.py   # about 20 min on 2 CPU threads
```

## Study 4 (pre-registered)

LLM-drift answers of GPT-4 and GPT-3.5 (March and June) on five tasks, 2358 items. Adding where disagreement sits (between versus
within model families) to the averaging baseline improves the prediction of whether the pooled answer is correct by
**0.0175 bits per item, 95% interval [0.0107, 0.0241]**: the pre-registered hypothesis is supported. The gain is concentrated in primes
and happy numbers, where model versions drifted.

```bash
cd study4 && LLMDRIFT_DIR=/path/to/llmdrift python study4_analysis.py
```

The incoherence detector is not tested on real data yet: each item's answers are observed jointly, and jointly observed tables always
admit a global law. Pairwise elicitation (`../halluc_study/kit_pairwise_cycles.py`) collects the data that would test it.
