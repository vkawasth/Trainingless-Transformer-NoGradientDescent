# Reproducing the three-level bias results (BASIL, MBIC)

All commands run from this folder (`examples/bias3`). Python 3.10+ with `numpy scipy pandas scikit-learn openpyxl`.

## 1. Data (not redistributed)
```
mkdir -p data
git clone https://github.com/launchnlp/BASIL data/BASIL                       # BASIL, 2nd release
git clone https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE data/BABE   # contains raw_labels_MBIC.xlsx
git clone https://github.com/fhamborg/NewsWCL50 data/NewsWCL50
```
Other locations: `export BASIL_DIR=/path/to/BASIL`, `export MBIC_XLSX=/path/to/raw_labels_MBIC.xlsx`, `export NEWSWCL50_CSV=/path/to/Annotations.csv`.
Scripts that import `amb_vigneaux` need the repository root on the path: `export PYTHONPATH=../..`.

## 2. Level-3 detection (the loop tests)
| command | what it reproduces | output |
|---|---|---|
| `python bias3.py` | BASIL stance: house leans, the 36 loop tests (camp / theme / window), toric exact vs Markov-basis MCMC, gluing map | `bias3_out.txt`, `bias3_results.json` |
| `python bias3_spans.py` | BASIL span favourability: loops Φ, exact no-3-way p, event bootstrap; by window and theme (Fox–HPO Φ = +1.34, p = 0.0017) | `bias3_spans_out.txt`, `bias3_spans_results.json` |
| `python bias3_spans.py --exclude-trump` | robustness without Trump as a target (Fox–HPO Φ = +1.09, p = 0.015) | `bias3_spans_results_notrump.json` |
| `python global_gluing.py` | planted global failures on disjoint BASIL regions; ring of K outlets | `global_gluing_results.json` |
| `python mbic.py` | MBIC: single-source theorem check, outlet × topic global failure (p = 0.0005), hostile-media loops | `mbic_results.json` |
| `python arity_graded.py` | arity-graded holonomy: full grade closes exactly; grade-2 plant recovered (BASIL thinned); MBIC grades | `arity_graded_results.json` |
| `python phase_change.py` (+ `--exclude-trump`) | change-point scans of lean and loop holonomy over 2010–19 | `phase_change_results*.json` |
| `python newswcl50.py` | NewsWCL50: Trump favourability slope, outlet × target loops, global G² | `newswcl50_results.json` |
| `python gluing_lattice_run.py` | gluing lattices: maximal glueable / minimal obstructed regions (NewsWCL50 targets and outlets, MBIC topics); ≈ 7 min | `gluing_lattice_results.json` |
| `python naturality.py` | 2016 phase change split into natural part (Δ lean) and non-natural part (ΔΦ), stance and spans, with/without Trump | `naturality_results.json` |

## 3. Prediction
| command | what it reproduces |
|---|---|
| `python predict.py` | nested models M1–M4: BASIL stance and spans (rolling origin), MBIC (10-fold) |
| `python predict_sens.py` | regularisation and split sensitivity of the loop gain |
| `python predict_tree.py` | tree-pooled loop (outlet type × topic) |
| `python trump_predict.py` | HPO favourability toward Trump (calibration), Fox criticising Trump (cross-source) |

Hand labels used by all BASIL scripts: `event_labels.py` (theme, focal party per event) and `target_party.py`
(party per span target). Edit them to change the topic cells. Permutation and bootstrap p-values use fixed seeds;
expect the same numbers up to Monte-Carlo error in the third decimal.

Runtime on 2 CPUs: `bias3.py` ≈ 30 s, `bias3_spans.py` ≈ 15 s, `global_gluing.py` ≈ 6 s, `mbic.py` ≈ 50 s, predictions ≈ 1–2 min.

## 4. Dynamics (synthetic three-layer generator; no data needed)
| command (from `examples/dynamics/`) | what it reproduces |
|---|---|
| `python three_layer_dynamics.py 400` | detection matrix: planted changes A–E × detectors d1–d5, 200 runs, 400 events/period; projected-outcome log loss (≈ 3 min) |
| `python three_layer_dynamics.py 150` | same at 150 events/period (power drop for d3, d5) |
| `python spectral_synthetic.py` | Slepian step, multitaper bands, harmonic line F on planted level / persistence / cycle / trend changes (≈ 2 min) |

## 5. AllSides big events (Baly et al. data; see `examples/allsides/README.md`)
| command (from `examples/allsides/`) | what it reproduces |
|---|---|
| `python events.py` | six pre-specified events: outcome, lean, naturality, Slepian step, multitaper, line F |
| `python scan.py` | unrestricted change-point scan of the weekly left−right gap |

## 6. Robustness certificate and outcome toggling
| command | what it reproduces |
|---|---|
| `cd examples/certificate && python run_all.py` | spectral certificate (Fiedler vs bootstrap ‖ΔL‖), obstruction SNR, refinement survival, leave-one-source-out on BASIL, MBIC, NewsWCL50, AllSides (≈ 20 s; needs BASIL_DIR, MBIC_XLSX, NEWSWCL50_CSV, ALLSIDES_DIR) |
| `cd examples/allsides && python toggle_covid.py` | centre alignment before/after the COVID emergency, placebo pairs, toggle radius |

## 7. Probing inside arity (hull trees, p-adic order profile)
| command | what it reproduces |
|---|---|
| `cd examples/arity_probe && python run.py` | coalition tests inside filled cells (AllSides stories, MBIC annotator groups, NewsWCL50), period shifts, p-adic order profiles (≈ 1 min; needs ALLSIDES_DIR, MBIC_XLSX, NEWSWCL50_CSV) |
| `cd examples/arity_probe && python replicate.py` | variogram replication: MBIC 3/5 groups, outlet types, article years; BABE annotator Mantel test (needs MBIC_XLSX, BABE_DIR) |

## 8. Moduli: toric strata, tree space, grammar strata
| command | what it reproduces |
|---|---|
| `cd examples/strata && python amb_toggle.py` | exact AMB toggle radii (MILP) on MBIC and AllSides, before/after COVID (≈ 1 min) |
| `cd examples/strata && python treespace_run.py` | BHV spider tests: sticky Fréchet means, energy tests (≈ 30 s) |
| `cd benchmarks/o6 && python o6_generate.py --out runs/V8L6 --depth 6 --nsym 8 --nleaf 12 --nrules 3 --reverse-valid --n-train 2000 --n-val 4000 --seed 0 && python merge_strata.py --data runs/V8L6 --n-val 4000` | merge-strata distances by level; root non-identifiability (≈ 40 s) |
| `cd examples/allsides && python stream_outcome.py` | streaming gluability → outcome map, flip costs, rate/mix decomposition, AMB toggles; writes `stream_outcome.pdf` (≈ 10 s) |
| `cd examples/dynamics && python gluing_lattice_dynamics.py` | PR-box path: CF, γ, S0, S2, IPF, statistical verdict at 4 sample sizes; figure (≈ 10 s) |
| `cd examples/arity_probe && python padic_arity.py` | p-adic arity grading: nested gluing radii by unit arity (synthetic) |

## 9. Outcome series (approval, 2016 margin) and the all-layers chart
Data: `git clone --depth 1 https://github.com/kiranrangaraj/Trump-Tweets-and-Approval-Rating-ETL` (APPROVAL_CSV = Resources/approval_polllist.csv) and
`git clone --depth 1 --filter=blob:none --sparse https://github.com/TheEconomist/us-potus-model` + `git sparse-checkout set data` (POLLS2016_CSV = data/all_polls.csv).
| command (from `examples/outcomes/`) | what it reproduces |
|---|---|
| `python outcome_series.py` | weekly outcome series, lead–lag against coverage, event windows (≈ 10 s) |
| `python layers_chart.py` | all layers on one axis: outcome, probabilities, natural/non-natural perturbation, CF / signalling / γ on the L–C–R triangle, gluing defect, gluability; figure (≈ 10 s) |

## 10. Paths over the gluing lattice
| command (from `examples/lattice_path/`) | what it reproduces |
|---|---|
| `python mbic_paths.py` (env `MIN_SIZE=5`) | envelopes, raise/lower pruning paths, region-selection rules vs held-out truth on MBIC (≈ 3 min) |
| `python synthetic_paths.py` | when consistency-driven gluing helps: contaminated topics, 640 cases (≈ 3 min) |
| `python landscape.py` | lattice landscape: first-order R², influence vs loop residual, aligned-move surface, feedback hill-climbing (≈ 5 min) |
| `python surface_chart.py` (env `OUTLET`, `TOPIC`; default breitbart × gender) | the loop-content surface figure: influences vs residuals, surface, raise/lower paths (≈ 1 min) |

## 11. Presheaf of Markov kernels (viability)

    python examples/kernel_presheaf/run.py        # synthetic; (4) reads examples/outcomes/layers_chart_results.json if present
    python -m pytest -q tests/test_kernel_presheaf.py

## 12. Hull covers on the lattice (direction test)

    MBIC_XLSX=... NEWSWCL50_CSV=... ALLSIDES_DIR=... BASIL_DIR=... python examples/lattice_path/hull_experiment.py
    python -m pytest -q tests/test_hull_direction.py

## 13. Bounds over all gluings

    MBIC_XLSX=... ALLSIDES_DIR=... python examples/bounds/run.py     # ~9 min (bootstrap); PLOT_ONLY=1 replots from results.json
    python -m pytest -q tests/test_bounds.py

## 14. Grey (what-if) contexts and the population test

    MBIC_XLSX=... ALLSIDES_DIR=... python examples/grey/run.py      # ~5 min; PLOT_ONLY=1 replots
    python -m pytest -q tests/test_grey.py

## 15. Several conditionals (multi what-if)

    MBIC_XLSX=... ALLSIDES_DIR=... python examples/multi_conditional/run.py     # ~15 s; PLOT_ONLY=1 replots
    python -m pytest -q tests/test_multi_conditional.py

## 16. Second-order holonomy (tetrahedral covers) and simplicial-complex instrumentation

    MBIC_XLSX=... python examples/higher/run.py      # ~3 min; PLOT_ONLY=1 replots
    python -m pytest -q tests/test_higher.py
    # instrumentation: amb_vigneaux.higher.complex_instrument(model) -> JSON (vertices/edges/faces with positions and weights);
    #                  amb_vigneaux.higher.plot_complex(inst, ax) draws it

## 17. Toric components [F] [T] [I] [S] on sphere covers

    MBIC_XLSX=... ALLSIDES_DIR=... python examples/toric/run.py     # ~10 s; PLOT_ONLY=1 replots
    python -m pytest -q tests/test_toric.py

## 18. Curvature [C] on the 2016 polls (sponsor affiliation)

    POLLS2016_CSV=... python examples/curvature/run.py     # ~2 min (cluster bootstraps); PLOT_ONLY=1 replots
    python -m pytest -q tests/test_curvature.py

## 19. Deformations [D]: T^1, T^2 and the flop at the boundary of the toric model

    MBIC_XLSX=... ALLSIDES_DIR=... python examples/deform/run.py     # ~10 s; PLOT_ONLY=1 replots
    python -m pytest -q tests/test_deform.py

## 20. Singular statistics [L]: learning coefficients, WBIC, Bayes vs plug-in

    POLLS2016_CSV=... MBIC_XLSX=... ALLSIDES_DIR=... python examples/singular/run.py     # ~2 min; PLOT_ONLY=1 replots
    python -m pytest -q tests/test_singular.py

## 21. Split–merge and the learning coefficient

    python examples/splitmerge/run.py      # ~15 min (MCMC); PLOT_ONLY=1 replots

## 22. Crowdsourcing pilot: tetrahedral covers, co-errors, planted checks

    git clone https://github.com/zhydhkcws/crowd_truth_infer /tmp/ds/crowd_truth_infer   # d_jn-product
    CROWD_DIR=/tmp/ds/crowd_truth_infer/datasets/d_jn-product python examples/crowd/explore.py     # (A)-(E); PLOT_ONLY=1 replots
    CROWD_DIR=... python examples/crowd/synthetic.py      # planted positives and nulls on the real design
    python -m pytest -q tests/test_crowd.py tests/test_crowd_synthetic.py

## 23. Second-order information geometry [G2] on JetClass-II (paper §sec:infogeo2)

Data (not redistributed; MIT licence): a 300k-jet subsample of `jet-universe/jetclass2` on Hugging Face, jet-level
columns only, 10k jets from each of 30 files spread over the dataset (QCD, Res2P, Res34P all represented):

    pip install huggingface_hub pyarrow pandas
    python - <<'PY'
    import pyarrow.parquet as pq, pyarrow as pa, pandas as pd
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(); files = sorted(fs.glob("datasets/jet-universe/jetclass2/**/*.parquet"))
    pick = files[:: max(1, len(files) // 30)][:30]
    cols = [f.name for f in pq.ParquetFile(fs.open(pick[0])).schema_arrow if not pa.types.is_nested(f.type)]
    parts = [pq.read_table(fs.open(f), columns=cols).to_pandas().sample(n=10_000, random_state=0) for f in pick]
    pd.concat(parts, ignore_index=True).to_parquet("jetclass2_300k.parquet")
    PY
    export JETCLASS2_PARQUET=$PWD/jetclass2_300k.parquet

Runs (CPU, seconds; no gradient steps):

    python examples/jets/census.py     # flatness / holonomy / node census                      -> census_out.txt, census.json
    python examples/jets/planted.py    # planted checks of every [G2] quantity (~5 s)             -> planted_out.txt, planted.json, planted_chart.pdf
    python examples/jets/run.py        # sphere path, curvatures, wobble, holonomy, nodes, order-4 (~20 s) -> run_out.txt, results.json, jets_chart.pdf
    PLOT_ONLY=1 python examples/jets/run.py
    python -m pytest -q tests/test_infogeo2.py

## 24. Holonomy as a size of the class, the flat ridge, flip prediction (paper §sec:bridge2)

    python examples/holonomy_norm/run.py      # gauge / zero set / Z_n / tilts / JetClass-II plaquettes (part e needs JETCLASS2_PARQUET)
    python examples/ridge/run.py              # dim ker R vs Fisher rank vs WBIC lambda on five covers (~20 s)
    ALLSIDES_DIR=... POLLS2016_CSV=... python examples/flips/run.py   # planted, AllSides, 2016 polls (several minutes); PLOT_ONLY=1 replots
    python -m pytest -q tests/test_holonomy_norm.py tests/test_ridge.py

## 25. Dynamic programming on the gluing lattice; the gradient-descent check (paper §sec:latticedp, Remark rem:nogd)

    python examples/lattice_dp/run.py          # planted 50x(50..400) benchmark, beta sweep (~5 min); SIZES=50,100 for a quick run; PLOT_ONLY=1 replots
    python examples/jets/gd_check.py           # needs JETCLASS2_PARQUET
    python -m pytest -q tests/test_lattice_dp.py

## 26. Which context to buy next (paper §sec:acquire)

    REPS=12 python examples/acquire/run.py               # six scenarios x four policies (~15 min); PLOT_ONLY=1 replots
    CROSSOVER=1 PLOT_ONLY=1 python examples/acquire/run.py   # cost x distance grid (~9 min), appended to results.json
    python -m pytest -q tests/test_acquire.py

## 27. A reward loop with all three layers (paper §sec:rewardloop)

    python examples/reward_loop/run.py      # six regimes x seven policies (~4 min); PLOT_ONLY=1 replots
    python -m pytest -q tests/test_reward_loop.py

## 28. Grading regions by Blackwell order (paper §sec:blackwell)

    python examples/blackwell/run.py        # n = 8, 10, 12 frontiers; parity paths; obstruction intervals (~15 s); PLOT_ONLY=1 replots
    python -m pytest -q tests/test_blackwell.py

## 29. Exact frontier growth; non-binary targets (paper §sec:blackwell-growth)

    python examples/blackwell/growth.py      # exactness vs enumeration (k = 2, 3), scaling to 35 measurements (~3 min), counterexample

## 30. CausalIDView partial-identification benchmark (paper §sec:causalidview)

    git clone https://github.com/MLAI-Yonsei/CausalIDView /tmp/ds/CausalIDView
    pip install xgboost scikit-learn
    CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView WORLDS=40 BOOT=30 python examples/causalidview/run.py   # ~30 min on 2 CPUs (bootstrap uses the structured backbone); PLOT_ONLY=1 replots
    # ablations (each 5-10 min; numbers in examples/causalidview/ablations/README.md)
    CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView python examples/causalidview/ablations/m_kernel_select.py
    CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView python examples/causalidview/ablations/iv_free_instrument.py
    # support-guided priors (each 3-10 min; plot.py redraws the chart from the recorded means)
    for f in oracle_support oracle_subspace subspace_prior crossfit_prior eb_prior eb_hyper law_prepass law_oracle beacon beacon_oracle; do CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView python examples/causalidview/support/$f.py; done
    python examples/causalidview/support/plot.py
    # shape of the conditional (PAVA / low-parameter links ~5 min each; icl_partial_order ~30 min)
    for f in isotonic_link link_family oracle_link_honest icl_partial_order; do CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView python examples/causalidview/shape/$f.py; done
    # posterior under a prior, per task, Metropolis (set OMP_NUM_THREADS=1; ~4 min per world per 4 chains; resumable)
    CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView SEEDS="0 1 2 3 5 6 7 8" NIT=1500 BURN=600 CHAINS=2 python examples/causalidview/bayes/ceiling.py
    CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView SEEDS="$(seq -s ' ' 0 39)" NIT=3000 BURN=1500 CHAINS=4 TAG=full python examples/causalidview/bayes/generic.py
    CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView SEEDS="$(seq -s ' ' 0 39)" NIT=3000 BURN=1500 CHAINS=4 CHSEED=1 TAG=more python examples/causalidview/bayes/generic.py
    CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView SEEDS="$(seq -s ' ' 0 39)" NIT=3000 BURN=1500 CHAINS=8 CHSEED=2 TAG=x16 python examples/causalidview/bayes/generic.py
    CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView python examples/causalidview/bayes/pool.py      # pools every saved chain, chart
    CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView python examples/causalidview/bayes/chain_structure.py   # 1/k tail test, mode count


## 31. Structured posteriors (paper §sec:sp)

    python -m pytest -q tests/test_sp.py                       # exact monad laws, expected failures, conditioning
    python examples/sp/multimodal.py                           # bimodal stress test (exact)
    OMP_NUM_THREADS=1 CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView TAG=mlp python examples/causalidview/bayes/stacking_mlp.py               # ~1 h, resumable
    OMP_NUM_THREADS=1 CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView SEEDS="0 1 2 3 5 6 7 8" TAG=control python examples/causalidview/bayes/stacking_mlp.py

    OMP_NUM_THREADS=1 CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView SEEDS="0 1 2 3 4" python examples/causalidview/bayes/persist.py   # ~75 min, resumable
    python examples/causalidview/bayes/persist_plot.py
    python -m pytest -q tests/test_shadow.py           # shadow algebra and context comonad, exact

    python examples/tower/check_tower_independent.py    # tower paper Thms 6.1 (E = D∘D) and 9.8, exact
    OMP_NUM_THREADS=1 CAUSALIDVIEW_DIR=/tmp/ds/CausalIDView python examples/tower/problem6.py   # ~5 min

    python -m pytest -q tests/test_attribution.py      # context attribution: EM projection, bands, unknown, switching, absorbed reference
    OMP_NUM_THREADS=1 BASIL_DIR=... BABE_DIR=... examples/context_attribution/runall.sh   # ~15 min on 2 cores; writes docs_*_n*.json, SUMMARY.md, attribution.png
    python docs/tower/build_v8_attribution.py <three_level_tower_v8.tex> docs/tower/three_level_tower_v8_attribution.tex   # inserts tower §5.4 with the results
    python examples/tower/claims_theorems.py           # claim-learning theorems: Lemma 1, Thm 2/Cor 3, Thm 4, Props 5a/5b, ties (~1 min)
    python -m pytest -q tests/test_claims_theorems.py
