# Handoff — Posterior Coherence Engine (PCE) programme

Status as of 23 October 2026. Attach this file, `tower_geometry_github.zip` and `engine/report/pce_report.pdf` to the new chat.

Author block, used on every paper: **Vinay K Awasthi, Johns Hopkins University, vawasth1@jh.edu.** The main paper keeps
"Department of Applied Mathematics and Statistics" above the university.

---

## 1. Standing rules (verbatim, still in force)

- "MIMIC-IV may be run only locally by the user under the PhysioNet DUA."
- "Never upload credentialed or patient-level data; only aggregated results with small cells suppressed may leave."
- "Pre-register plans before examining data, and freeze them with SHA-256."
- "Keep patient-level caches outside the monitor folder."
- "Never package raw C19PRC data (monitor/ext/c19/raw)."
- "SciFact data stays outside the code archive."
- "Do not ask for, store or embed the user's API keys."
- "Just give me pdf, no need to create .zip unless you add any script."

**Process rules learned the hard way (October 2026), now mandatory:**

1. **Gate before any real-model run.** A short instrument check must show that the model's answer changes when the question
   changes, in the right direction, by more than its prompt-variant noise. A plumbing probe (does it load, how fast) is not enough:
   Study 7 lost a full run because only plumbing was checked.
2. **Power analysis by simulation before freezing.** Use a simulated model matched to the gate's measured behaviour, and set the
   detection thresholds from it.
3. **Dry-run every analysis on simulated models before freezing.** The Study 7b dry run caught an item-ID collision and a
   rank-correlation tie bug.
4. **Freeze with SHA-256 before data.** Log every later change in `DEVIATIONS.md` before the data it affects. Never adjust a threshold
   after seeing a result.
5. **Declare prior exposure.** If a real dataset (GSS, C19PRC) was analysed in earlier phases, the pre-registration must say so.
6. **Verify citations** (WebSearch/WebFetch) before adding them. One Robinson reference was wrong from memory.
7. **Report failures as failures**, with a diagnosis.
8. **Watch out when waiting on a background job with `pgrep -f`.** The waiting command's own command line contains the pattern, so
   it waits on itself forever. Don't edit `run_checks.sh` while a suite is running.

---

## 2. Where everything is

**Sandbox (this cloud session):**
- Working tree: `/home/claude/paper/unified/`. The engine is in `engine/`; the report is `engine/report/pce_report.tex/.pdf`.
- Package tree: `/home/claude/github_papers/tower_geometry/`. It is zipped to `/home/claude/tower_geometry_github.zip`, and the
  SHA-256 manifest is `FROZEN.txt` at its root.
- Kits: `/home/claude/study7_pairwise_kit.zip` and `/home/claude/study7b_records_kit.zip`.
- Real data on the machine (no outcome statistics computed in this phase): the GSS extract `/home/claude/gss/gss_extract.csv`, with
  split-ballot wording versions of 11 spending items and the 2006/08/10 three-wave panels as `.rda`; C19PRC waves in
  `/home/claude/monitor/ext/c19/` (raw, never package); LLMDrift CSVs in `/home/claude/monitor/data/llmdrift/`; HaluEval in
  `/home/claude/halueval_data/`.

**The user's Mac:**
- Project: `/Users/vaw1/Downloads/OGB/connectome/phaseTransition_phaseTransition_complex/FukayaAUComplex/longChainHllucinationp-adic/j-holomorphic-Fukaya/Papers_Synched/tower_geometry/`
  (Study 7 ran from `engine/engine_ollama/study7_pairwise`, Study 7b from `engine/engine/study7b_records`).
- Python: miniconda, Python 3.12, transformers and torch on Apple MPS.
- **Disk space is tight**, and Hugging Face downloads stall: one took a day.
- Hugging Face cache: `Qwen/Qwen2.5-1.5B` (base; snapshot `8faed761…`) and `Qwen/Qwen2.5-1.5B-Instruct` (snapshot `989aa798…`).
  Use `export HF_HUB_OFFLINE=1`.
- Ollama app: `/Applications/Ollama.app/Contents/Resources/ollama`, server on `localhost:11434`, started by the app (a second
  `ollama serve` fails with "address already in use"). Models: `llama3.1:latest` (8B, 4-bit, ID `46e0c10c039e`), `llama3.2:1b`,
  two gpt2 variants. Its `/api/generate` returns `logprobs` with `top_logprobs`.

---

## 3. What exists

**The report** (`engine/report/pce_report.pdf`, 50 pages, 24 sections):
1. What the engine reports
2. Using it
3. Synthetic demonstrations
4. A pre-registered real-data test (Study 4)
5. A worked case: a doctor and a lawyer
6. Toward an algebra of in-context learning
7. Integrating expertise
8. The context poset and the gluability complex
9. Cohomology on the fibres
10. Closing the lift
11. Study 5 (toy language models)
12. Gerstenhaber–Schack cohomology
13. Separating the layers (places)
14. Overlap algebra
15. Overlap algebra at level two
16. Study 6 (two open-weight language models)
17. Gluing over a cover of contexts
18. Perturbing context and probability
19. Walls and chambers
20. Contextuality beyond direct influence; Studies 7 and 7b
21. Several experts as toric families
22. Non-commuting updates and conservation of mass
23. The filtered gluability complex
24. What is not yet tested

Figures sit inside their own sections (`\usepackage[section]{placeins}`, `[!htbp]`).

**Checks: 733, all passing** (`./run_checks.sh` at the package root, about three hours on two cores; see the run logs): 94 for the paper, 6 for the grid, 9 for the companion note,
545 for the engine. The engine modules in `engine/pce/`, each with its check file:

| Module | Checks | What it covers |
|---|---|---|
| `coherence` | 7 | |
| `icl` | 8 | |
| `integrate` | 8 | |
| `complex` | 8 | |
| `cohomology` | 8 | |
| `lift` | 8 | |
| `gs` | 7 | |
| `places` | 7 | |
| `overlap` | 7 | |
| `overlap2` | 7 | |
| `cover` | 8 | |
| `perturb` | 8 | |
| `strata` | 8 | |
| `cbd` | 7 | |
| `toric` | 8 | |
| `updates` | 8 | |
| `persist` | 8 + 6 | filtration, truncation, join rule (`check_persist2.py`) |
| `level2` | 9 | graded explanations, latent-class fibres, strata, κ, Puiseux, monodromy, λ*₂ (note: `level2/partial_gluing.pdf`) |
| `check_scheme.py` | 6 | exact (sympy): moment normal form, Stanley–Reisner fibre, closed-form inverse, hyperdeterminant, flattening strata |
| `ladder` | 6 | ladder of base semirings ℂ ⊃ ℝ ⊃ ℝ≥0: linear gap, negativity vs 1 − λ*, level-two rungs (`check_ladder.py`) |
| `noncomm` | 8 | commutation cover, order bound and QQ, Kirkwood–Dirac section, Bargmann phase, non-abelian holonomy, Zeno dragging (`check_noncomm.py`) |
| `masked` | 6 + 1 | given measurements partly masked: linear identifiability, dimension of the linear ambiguity, cone culling, lattice sweeps (`check_masked.py`, about 13 min; `check_masked_gauge.py`) |
| `qglue` | 7 | quantum graded gluing as the positivity domain of a hereditary quadratic (Islam–Volčič–Woerdeman, arXiv:2610.14053); dephasing bound; Werner formula proved by symmetry (QG7); SDP dual certificates (`check_qglue.py`, needs cvxpy) |
| `npa` | 6 | categorical home: NPA (Tsirelson; isotropic closed forms), hierarchy order, data processing, q^n = 1 obstruction, Peres–Mermin KS vs Gleason (`check_npa.py`, needs cvxpy) |
| `fingerprint` | 7 | origin fingerprint: nested model-class fractions (direct influence, CbD c, λ*, NPA quantum, level-two rung) and the origin label (`check_fingerprint.py`) |
| `cbdfrac` | 7 | fractions allowing direct influence: consistification, classical 1 − CNTF (= 1 − c on cyclic systems), dimension-free quantum fraction over the right-angled Coxeter group (n-cycle bounds, Bell reduction, a quantum model with direct influence) (`check_cbdfrac.py`, needs cvxpy and Clarabel) |
| `ball` | 7 | perturbing connexions, data and fibres: connexion-budget profile (concave, price ≤ 1, λ(r) ≤ min(1, λ(0) + r); = min(1, 1 − c + r) on cyclic systems), grading by connexion on a non-cyclic system, exact data ball by orientations, level-two fibre tracking, wall distance and ball rung shares (`check_ball.py`; figure `level2/fig_ball.py`) |
| `coxeter` | 6 | observable algebra W_Γ of a compatibility graph: Rep_n tangent spaces, second-order obstructions (exact sympy certificates), Künneth cup-product mechanism, all graphs on 4–5 vertices: Rep_3 singular iff non-chordal; NPA quantum gap iff non-chordal (`check_coxeter.py`, about 10 min, needs networkx) |
| `birth` | 11 | CHSH (C4): contextual models branch off the classical locus only at birth points (classical points containing a degenerate 2⊗2 block), all of them singular; most singular points give no birth; CHSH = 2√(1+sin²t) along the birth curve; KCBS (C5) via the complete n-cycle witness: born at a smooth class of Rep_3, while the singular class gives none; what degenerates: a block of size ≥ 3 is necessary, not sufficient; Rep_4(C5) birth and singular loci overlap only partly (`check_birth.py`, reads `level2/c5_scan_n3.json`, `c5_scan_n4.json` from `scan_c5.py`) |
| `invariant` | 3 | the second-order birth invariant Λ₂ (`pce/birth2.py`): the first-order term vanishes at every classical point; Λ₂ > 0 exactly at birth for CHSH (1/2) and KCBS Rep_3 (3/4); in KCBS Rep_4, Λ₂ > 0 ⇒ birth (25 classes), and the other 4 birth classes are higher order (`check_invariant.py`, reads `level2/c5_invariant_n4.json`, `c4_invariant_n4.json` from `scan_invariant.py`) |

**The other papers:** the main paper `tower_geometry.pdf` (55 pages) and the companion note `companion/pooling_agents.pdf` (5 pages).

---

## 4. Empirical record (all pre-registered and frozen)

- **Study 4** (LLMDrift, jointly observed). The engine's features predict pooled-answer correctness modestly. Coherence status is
  uninformative there, because jointly observed tables always glue.
- **Study 5** (toy language models). H1–H3 supported, H4 not.
- **Study 6** (Qwen2.5-1.5B and SmolLM2-1.7B as experts, in-context case notes).
  - H1–H3 not supported: the tower merge is about the same as product-of-experts, with an exact gap of 0.001–0.004 bits against an
    elicitation noise of 0.02–0.05 bits.
  - H4 supported.
- **Study 7** (stated percentages, a single-token answer over joint options). **Failed as an instrument.**
  - Qwen2.5-1.5B-Instruct: H1 and H5 pass; H2 fails (0 of 8 detected; measured c = 0 everywhere). An earlier printed Spearman of
    −0.15 came from a tie bug and is corrected in `DEVIATIONS.md`.
  - The model returns the same table, about (0.13, 0.32, 0.34, 0.21), for every item. Direct influence 0.024 against presentation
    effects of about 0.1.
  - Llama 3.1 8B failed `gate7.py`, both plain and with its chat template.
- **Study 7b** (in-context records: 40 trials per pair experiment; `study7b_records/`). **The first real-model result.**
  - Qwen2.5-1.5B base passed `gate7b.py`: sensitivity 0.388, noise 0.110, reading 0.095.
  - 102 items: 40 coherent, 20 null, 22 designed-contextual, 20 signalling. 16 variants per pair, tolerance of two standard errors.
  - **H1 passes:** 0 of 20 null items contextual.
  - **H2 passes:** 4 of 8 strong items detected, exactly at the threshold; Spearman 0.88. The model attenuates contextuality: designed
    0.55 comes out as 0.08, 0.7 as about 0.3, 0.85 as about 0.6, 1.0 as about 0.92.
  - **H3 passes:** all 20 signalling items are PCE-incoherent, and none is contextual by CbD.
  - **H4 fails,** as the power analysis predicted (Spearman 0.21, CI [−0.12, 0.49]).
  - The model's own estimates were noncontextual on all 60 coherent and null items.
  - Llama 3.1 8B failed `gate7b.py` on noise (0.171 against a limit of 0.12), so Study 7b is a single-model result.
- **Study 8** (Snow Queen, Cervantes & Dzhafarov 2018; published aggregated tables; `engine/study8_snowqueen/`). The conclusion was
  known in advance from the title; the registration records this.
  - All six predictions held. On correct responses (D2), direct influence is 0.77 and c = 0.23; λ* = λ_Q = 0, which is structural.
  - On all responses (D3), direct influence is 0.53, c = 0.138 with 95% CI [0.069, 0.194]: contextual beyond direct influence.
  - D3 fractions: λ* = 0.327, λ_Q = 0.460, gap CI [0.081, 0.148].
  - c equals half the authors' Kujala–Dzhafarov statistic (0.452, 0.279), which the transcription reproduces exactly.
  - One deviation is logged: D2 is bootstrapped with the correct-response counts.
- **The two towers as one grid** (Section 16; `check_grid.py`, TG1–TG6).
  - Rows: the semiring S ∈ {ℂ, ℝ, ℝ≥0, 𝕋}. Columns: tower depth D_S^n.
  - The main paper's Beck table is identical in all four rows, and change of base preserves coherence and both laws.
  - Convex order, the martingale reading and defect sizes exist only in the ℝ≥0 row; explicit signed counterexamples are given.
  - Table 8 of the note places every result in the grid.
- **Observable algebra** (Section 15).
  - Contextuality is possible ⇔ the graph is non-chordal ⇔ W_Γ is not virtually free ⇔ ℚ[W_Γ] is not formally smooth.
  - Computed: Rep_3 has a singular point iff the graph is non-chordal, for all 45 graphs on 4–5 vertices. Exact certificates exist for C4,
    C5 and C6, and a retraction argument extends them to every graph with a chordless cycle of length 4–6.
  - So 3×3 matrices suffice, and no non-matrix coefficients are needed. Cycles longer than 6 remain a conjecture.
  - Birth of contextuality, for CHSH: contextual models appear next to a classical point only at birth points.
    - All 4 birth points of Rep_4 are singular. Another 240 singular points give no birth.
    - Near birth points 37% of nearby models are contextual. Near the others, none of 1500 are.
    - The violation grows like t² (second order).
  - KCBS (C5): contextuality of a model is defined by the complete n-cycle inequalities, which needs no block classification.
    - In Rep_3, exactly one symmetry class (160 points) gives birth, at second order (witness ≈ 0.7 ε²), and it is a smooth point.
    - The singular class (160 points) gives no birth.
    - So "born only at singular points" holds for CHSH only. The singular locus marks where contextuality is possible, not where it is
      born.
  - What degenerates (BR9–BR11; scan_c5.py for n = 3 and 4).
    - Birth needs a non-commuting block of size ≥ 3 degenerating onto the classical point. Near the birth class of Rep_3 every model is
      irreducible and contextual; near the singular class every model is 2 + 1 and none is.
    - Not sufficient: in Rep_3, 6 classes have irreducible neighbours and 1 gives birth; in Rep_4, 99 have a block ≥ 3 and 29 give birth.
      Irreducible models are mostly noncontextual globally.
    - In Rep_4(C5), birth (6160 points) and singularity (4360) share only 840 points.
    - The second-order invariant Λ₂ (Proposition 30; INV1–INV3; `pce/birth2.py`).
      - It is the top eigenvalue of the effective operator PB₂P + PB₁(n−2−B₀)⁺B₁P on extendable tangent directions.
      - The first-order term vanishes at all 342 classes checked.
      - Λ₂ > 0 exactly at the birth classes for CHSH (1/2) and for KCBS Rep_3 (3/4, matching the observed 0.747).
      - KCBS Rep_4: 25 second-order birth classes, no false positives among the 213 classes without birth. The remaining 4 birth classes are
        higher order (fitted exponents 2.8–4.1).
      - Open: a higher-order invariant, and a proof that the first-order term vanishes off the diagonal.
- **Perturbations** (Section 14, exploratory, not pre-registered).
  - On Snow Queen, all responses: the connexion budget needed is exactly c = 0.138. Any one connexion can carry it, so the grading is flat
    on a cyclic system.
  - A coordinated move of every window by TV 0.017 removes the contextuality. The median random sampling move per window is 0.020, and
    the 95% move is 0.044. So the result is contextual under resampling, but not robust to the worst case at the sampling scale.
  - On a non-cyclic 2×3 system, only the connexion on both contextual cycles buys support.
- **Study 8b** (same data, secondary; consistified fractions, bipartite commutation).
  - 1 − λ* splits into three shares: direct influence (DI), quantum-explainable contextuality (QC), and contextuality beyond quantum (BQ).
  - All responses: DI 0.535, which equals Σs (the bound λ* ≥ λ_C,cbd − Σs is attained). QC 0.134 [0.068, 0.147]. BQ 0.005 at level 1+E
    and 0.010 at level 2, CI [0, 0.096]. BQ ≈ 0 in 42% of replicates, so the registered label is **undetermined**.
  - Correct responses only: QC = 0 and BQ = c = 0.226. These data are selected on correctness.
  - With minimal commutation, nothing exceeds quantum on either dataset. The beyond-quantum verdict rests on the bipartite assumption.
  - R5 failed: level 1+E is not tight. Level 3 runs out of memory.
  - Deviations logged: the relaxation is cached, the solver is Clarabel, solver accuracy is about 2e-4 because the program is not
    strictly feasible (facial reduction was tried and did not help), and the sensitivity analyses are post hoc.

---

## 5. Key mathematics (latest sections)

- **§17, cover.** Experts as sections, with scope, senses, competence from probes, and graded gluing θ\*. No overlap means no
  validation. Robust maximin actions are global sections.
- **§18, perturbation.**
  - The tube radius ρ\* (an LP) certifies a decision for every merge rule.
  - ρ\* = 0 means the filler decides, not the windows.
  - Bridge windows pin a decision, or expose a hidden error.
  - Tempering δ(β) = βδ∞ + O(log n).
- **§19, walls and chambers.**
  - Vertices of the fibre are affine inside chambers.
  - Tangent complex H⁰/H¹, with χ = |S| − rank.
  - The robust outcome is affine per chamber; the max-entropy outcome is curved.
  - Maslov dequantization. The L∞ structure is abelian at level one, so higher brackets first appear at level two.
- **§20, contextuality by default.**
  - The LP agrees with the Kujala–Dzhafarov cyclic criterion.
  - The graded test θ_c is exact: 2^k LPs, one per orientation of the maximal couplings.
  - Incoherence certifies error: 1 − λ\* ≤ Σ TV(window, truth).
- **§21, toric experts.**
  - Joint indecision equals a mixed volume (tropical Bernstein).
  - Joint decisions are the cells of the mixed subdivision.
  - The linear pool never creates a winner; product-of-experts can (compromise outcomes).
  - Gluing on a chain is a toric fibre product; on a 3-cycle there is only a degree-4 move.
- **§22, updates.**
  - Jeffrey's rule = I-projection; it moves exactly TV(q, p_A).
  - The holonomy of two updates factorises as b_j / a_i.
  - The component of log p orthogonal to the cover is conserved.
  - Loops of updates close iff the evidence glues. Otherwise the circulating mass is at least (1 − λ\*)/(n − 1) per round.
  - The curvature K₁₃ − K₂₃K₁₂ of a linear transport certifies a hidden state.
- **§23, filtered complex.**
  - φ = 1 − λ\*; hollow cells are bars; diagrams are stable under changes of evidence.
  - Homology is a shadow; critical slowing appears on walls (observed, not proved).
  - **Join rule:** λ\*(A ∪ B) = min, so the persistence of independent parts composes into one class one dimension higher, dying at
    the weakest part. Homology is a global AND, so read large families locally.

**Literature positioning** (cited in the report):
- λ\* is the contextual fraction (Abramsky, Barbosa and Mansfield 2017).
- The consistency radius and filtration are Robinson's: *Assignments to sheaves of pseudometric spaces*, Compositionality 2 (2020);
  sensor integration, arXiv:1603.01446.
- Contextuality-by-default is Dzhafarov and Kujala's.
- Contextuality in LLMs was measured by Lo, Sadrzadeh and Mansfield, arXiv:2412.16806 (BERT, 71% CbD-contextual).
- Overall value (honest): **moderate.** The best framing is a methods paper: a calibrated sheaf-theoretic instrument for incoherence
  in model outputs, with Study 7b as evidence. Possibly novel, and needing a literature check before being claimed:
  - the circulating-mass bound;
  - the factorised order effect;
  - decision radii and bridges;
  - critical slowing on walls;
  - the join rule.

---

## 6. Open items (choose next)

1. **Replicate Study 7b on another base model** that passes `gate7b.py` (Qwen-family base models seem best suited). Same frozen kit;
   add a deviation entry first.
2. **GSS wording experiment** (real survey contexts): contextuality-by-default and local persistence on split-ballot wording.
   Pre-register first, and declare prior exposure.
3. **Panel curvature test:** K₁₃ − K₂₃K₁₂ on the GSS 2006/08/10 panels or the C19PRC waves. Pre-register first.
4. **Order-effect test:** the factorised holonomy predicts equal odds ratios in both question orders. Needs question-order data,
   such as the Pew data used by Wang and Busemeyer, or an LLM order study.
5. **A bridge study** (§18): do engine-chosen bridge questions improve decisions more than random extra questions?
6. **Literature checks** for the possibly novel results listed above.
7. **Theory:**
   - level-two second-order obstructions (ℓ₂ ≠ 0);
   - a proof of the critical-slowing rates;
   - local persistence formalised as a Robinson-style localisation.

## Decisions on a time-varying sheaf (Section 17)

- `engine/pce/timesheaf.py`: covers as decision variables, graded explanation sets per cover, robust value (Huber contamination with
  epsilon >= 1 - lam*), info-gap robustness, robust satisficing over (cover, act), hypothesis support, refinement/coarsening and drift.
- `engine/timesheaf_example.py`: the worked example (three drugs, five times: tree, conflicting trial, awareness growth, drift, pendant).
- `engine/check_timesheaf.py`: TS1-TS10 (cover monotonicity, acyclic gluing, awareness growth, new/pendant windows, drift Lipschitz,
  robust values and frontier, selection hazard, the worked example, tree bound, vacuity); writes `level2/timesheaf_results.json`.
- `level2/fig_timesheaf.py`: `fig_timesheaf_a` (complexes over time above, decision planes below) and `fig_timesheaf_b` (lam* of
  covers, frontiers, robustness and expiry).
- Open: expiry when a new distinction enters the utility is not predicted by any radius; robustness uses contamination only (connexion
  budget and data ball not yet added); level-two fibres are not used in the decision example.

## Study 8c: the Snow Queen as a decision on a time-varying sheaf

- Files are in `engine/study8_snowqueen/`: `PREREG_study8c.md` and `analyze_study8c.py`, frozen in `FROZEN_PREREG_study8c.txt` together
  with `pce/timesheaf.py`, `integrate.py` and `coherence.py`. Outputs are `results_study8c.json` and `run_study8c.txt`, and
  `level2/fig_snowqueen_decide.py` draws the figure. The code was dry-run on synthetic tables only before freezing.
- Results:
  - S1-S6 and S8 hold.
  - S7 fails as registered: "Gerda beautiful and kind" is also good enough on 12+21, where the marginals alone almost force it
    (robustness 0.011). Its substantive half holds: every cover on which a claim is good enough omits a question the other claim rests
    on.
  - Rule R prints "Gerda beautiful and kind" from t2 onward (robustness 0.273), on cover 11+12+21, in 100% of replicates for D3 and D2.
  - The title claim is good enough only on 21+22 and 11+21+22 (0.150).
  - At t3, the printed note depends on arrival order (Gerda 12 of 24, title claim 6, SQ evil 6), and half of the t3 notes are retracted
    at t4.
  - At t5 (correct responses only), the title claim and the Gerda claim tie exactly at 0.375. The after-the-fact choice between them is
    settled by rounding, so its bootstrap split is not evidence.
- No procedural deviations.
- Added afterwards (`posthoc_study8c.py` -> `posthoc_study8c.json`; not pre-registered): three ways of deciding over repairs, with
  repairs taken as the maximal covers with lam* >= 0.80.
  - The three semantics agree up to t3.
  - At t4: credulous gives the title claim or Gerda, skeptical gives none, rule R gives Gerda.
  - At t5: credulous gives Gerda or "SQ evil"; the title claim drops out because 21+22 glues only 0.789.
  - The note now has Proposition 33 (vacuity: lam* < (u0 - u_min)/(u_max - u_min) means no act is good enough; check TS10) and a
    paragraph "Choosing what to set aside" on credulous, skeptical and declared rules and on explaining versus ignoring.

## Reading guide and colour key (note front matter)

- "How to read this note" comes before Section 1. It gives three sides (geometry of latent actors, contextuality, decisions), a
  table (Table 1: if you want / read / because / you can skip) and a colour key.
- Each section heading carries a letter and a colour bar, and theorem heads take their section's colour.
  - F foundation: 1-2.
  - G geometry: 3-7.
  - C contextuality: 8, 9, 11-13, 15.
  - D decisions: 14, 17.
  - P parallel: 10.
  - M maps: 16, 18.
- The face assignment came from a cross-reference scan plus a term scan. Sections 11-12 are not optional for contextuality, because
  they supply lambda_Q, which Section 13 uses throughout.
- Text colours have at least 4.5:1 contrast on white; the letters carry the meaning in greyscale.
- The macros are \setface, \facemark and \facebox. Theorem styles: facedplain and faceddef.
- Cross-references are coloured by their target's face.
  - At each \\label, the anchor -> face pair is written to the aux file as \\facean.
  - \\find@pdflink is patched at begindocument/end so that \\@linkcolor follows the target.
  - Bibliography entries take the face of the section that first cites them.
    - \\setface writes \\facemarkaux to the aux file.
    - Reading the aux back, \\citation records the first face as anchor cite.<key>.
    - \\bibitem colours the label and adds the face letter; citations take the same colour.
    - Verified against the source: 93/93 entries match, none uncited (C 54, G 15, D 10, M 7, F 4, P 3).
  - The patch avoids # inside the hook: it hooks before \\expandafter\\Hy@colorlink using \\Hy@testname.
- Reproducibility moved to Appendix A.
- Open: the abstract still stops at Section 12.

## Study 9: cup products decide second-order obstructions on chordless cycles

- Files:
  - `engine/pce/cup.py`: the block-by-block cup classifier and the closed form for cycles;
  - `engine/check_cup.py`: CU1-CU6, about 4 minutes;
  - `engine/study9_cup/`: PREREG, run script and results, frozen together with `cup.py` and `coxeter.py`, plus `posthoc_study9.py`.
- Theorem (note, Theorem "cup criterion"): on the n-cycle, n >= 4,
  - dim H^1(psi) = arcs - 1;
  - an obstruction lives only in the all-minus character eps, as the circulation (1/2) sum q(B_r)(p(A_{r+1}) - p(A_r)), a perfect
    pairing of rank c - 1;
  - a direct sum of characters is obstructed iff two of its characters multiply to eps and a third differs from one of them in at
    least 2 arcs.
- Corollary: Rep_3(W_G) is singular iff G is not chordal, now proved for all graphs (previously certified for cycles of length 4-6).
- Results:
  - P1, P2, P4 and P5 hold.
  - Full agreement with the engine on all points of Rep_3(C4-C6) and Rep_4(C4, C5), on samples of C7 and C8, and on all 45 small graphs,
    with no adjudication needed.
  - Counts in Rep_3(C_n), n = 4-8: 16, 160, 1024, 5376, 25344. Rep_4(C4) has 244 and Rep_4(C5) has 4360; these match the earlier
    singular sets.
  - P3 failed as registered: the code measured the value span (at most 1), not the matrix rank; post hoc, the matrix rank is arcs - 1.
  - P7 failed for the block criterion: 4.3x against a target of 10x. The closed form is 515x.
- Open: the general-graph closed form (circulation in H^1 of the flag complex of the induced subgraph); whether part (a) and the cup
  formula are in Davis's book.

## Coherence pass (2026-10-26, after Study 9)

- **Abstract** rewritten. It covers all three sides: geometry, contextuality (including the Snow Queen split, the four-way equivalence,
  the cup criterion and birth with Lambda_2) and decisions. It closes with the studies and the reading apparatus. The old per-file
  check list is gone.
- **Section 1** now has a roadmap of Sections 3-17 by side.
- **Section 13** now points forward to Sections 14 and 17 (Study 8c).
- **Section 15**:
  - It now answers its own question (is non-smoothness visible in Rep_n?) with Corollary 31.
  - Proposition 29 is restated as a retraction statement plus the certified cases.
  - The "Reading" paragraph and the limits are corrected: "cycles longer than 6 are a conjecture" was removed, and there are now 5 limits.
- **Section 16** had a wrong claim, "contextuality is born where that embedding sits singularly"; KCBS refutes it, and it is corrected.
  The grid table rows are updated.
- **Section 18**:
  - The dictionary is in section order, with rows for singularity (Theorem 30), birth and choosing what to set aside.
  - The programme paragraph covers all three sides.
  - Related work has a contextuality paragraph that says what is new.
  - The limits gain contextuality and decision limits. The Stanley-Reisner limit is corrected: Theorem 13 holds for every m, and
    m = 3, 4 are only cross-checks.
  - Next steps: items 7 and 10 are updated, and items 11-14 are added.
- **Open:** the title still names only the geometry side.

## Related programmes and Study 10 (2026-10-26)

- Cited:
  - Chang, CSD Geometry II (hal-05676869) and Algebra I (hal-05711275);
  - Ikeda, arXiv:2511.04326v2.
- Where they are cited:
  - Section 8: Ikeda's acyclic linear Cech complex is the quantum counterpart of the ladder.
  - Section 11: Ikeda's yes/no marginal problem; our quantum graded gluing grades it.
  - Section 15: the paragraph "Other obstruction theories (Study 10)", with Table 10 (spectra).
  - Section 18: the related-work paragraph and a dictionary row.
  - The abstract names the programmes without citing them.
- Points raised about CSD, stated as "as we read them":
  - the Pauli pair is chordal and glues, yet is assigned a non-zero obstruction;
  - HH^2 = 0 for finite-dimensional semisimple algebras;
  - HH^2(Q[W_G]) is infinite-dimensional for non-chordal graphs;
  - the F_2 caveat.
- Study 10 (`engine/study10_spectrum/`, frozen together with `pce/racg_cohom.py` and `pce/cup.py`):
  - Method: H^n(W; psi) = Hred^{n-1}(Flag(G[N(psi)])), via the Davis complex in characteristic 0.
  - Predictions R1-R5 all hold.
  - Totals by degree: CHSH 1,2,1,0; KCBS 1,10,1,0; Mermin-Peres 1,120,247,0 (H^2(eps) = 4); Pauli pair 1,1,0,0; triangle 1,0,0,0.
  - Nothing appears in degree >= 3, so the hypothetical MP polynomial 2t^2 + t^3 (Algebra I, Example 6) is not realised by this candidate.
  - The cup values on the rook's graph land in H^2 (458 non-zero values among 24608 triples).
- CU7 was added to check_cup.py.
- Open:
  - HH of Q[W_G] needs all centralisers and was not computed;
  - CSD's Obs• may be a different object.

## BACKLOG (agreed 2026-10-27; do in this order)

### Main note first
1. **Section 15, André–Quillen vs Hochschild remark.**
   - Latent fibre (commutative): cotangent complex / André–Quillen.
   - Observable algebra (non-commutative): Hochschild / Cuntz–Quillen.
2. **Section 15, twirl conjecture.**
   - Conjecture: contextuality is born where the twirl's covariance group (the image of W_G under the model) stops being abelian.
   - Verify Aniello's definition and citation first.
   - Pre-register a test on the Rep4(C4) birth data; KCBS stays an open test.
   - Status: testable conjecture.
3. **Sections 7 and 18, toric fibre products.**
   - Level one: decomposable = iterated codimension-zero TFPs (Sullivant; Engström–Kahle–Sullivant; Rauh–Sullivant). Verify the references.
   - Level two: open.
4. **Section 17, decision pipeline paragraph and diagram.**
   - Cup product (design) → latent actors (explanation) → cover (choice), each marked implemented or open.
   - Design rule: chordal + no direct influence ⇒ the all-evidence cover is never vacuous because of contextuality.
5. **Coherence pass after items 1–4:** abstract, dictionary, related work, next steps. Title decision pending with the user.

### Companion paper: the latent-actor lab (an OODA loop; three readerships: ML/RL/GAN, decisions/VOI, persuasion/disclosure)
6. **Verify the literature.** Knight; Wald; Gilboa–Schmeidler; Huber; Ben-Haim; Schwartz–Ben-Haim–Dacso; Long–Sim–Zhou; Howard; Raiffa;
   Clemen; Blackwell; Al-Najjar–Weinstein; Siniscalchi; Kamenica–Gentzkow; Milgrom; Grossman; Henry–Ottaviani;
   Di Tillio–Ottaviani–Sørensen; Chen 1995; Watanabe; Lindley; Chaloner–Verdinelli; vsOED; POSI (ask the user for the source);
   Goodfellow; Mescheder et al.; f-GAN; Kaelbling–Littman–Cassandra; Boyd; Arenas–Bertossi–Chomicki; FDA registration and
   results-reporting rules as model constraints.
7. **Theorems with proofs.**
   - **A, the hiding segment.**
     - Pairwise fibre = a segment in r = (w2 - w1)/sqrt(w1 w2).
     - Event probabilities are polynomials in r of degree <= m - 2 (verified numerically: affine for m = 3, quadratic for m = 4).
     - Worst cases lie at the box walls.
   - **B, three tempos.**
     - 0 under pairwise-only disclosure.
     - n^-1/2 after one triple window with v_S != 0.
     - At a coincidence, n^-1/4 for the relation (Chen) but n^-1/2 for predictions.
   - **C, stratified VOI and the mandated window.**
     - VOI is 0 on the independence stratum and 0 for probes touching the coincident set Z.
     - A probe avoiding Z collapses the segment completely.
     - VOI is continuous in |v_S| sqrt(n).
     - The best probe is argmax |v_S|.
     - Corollary: Milgrom unravelling.
     - Remark: negative VOI at level one, through the lambda* floor.
   - **Rejected proposals, with reasons, so they are not reintroduced:**
     - "Worst case on the non-flat locus" (it is at the box walls).
     - "The minimax probe separates along two variables" (it is a triple with v_S != 0).
     - "VOI ~ sqrt(s) at a coincidence" (VOI there is either 0 or the full collapse).
     - "One-sided VOI on the non-flat locus" (it is 0).
     - "The m = 4 pairwise fibre is finite" (rank 2m, so it is a segment).
8. **Study 11 pre-registration, frozen before running.**
   - 11a: design check.
   - 11b: relations, with (b) absent actor as a negative control (unidentifiable), exponents with a calibrated null, and the
     relation-support map.
   - 11c: probe loop, with the positive control (pairs + one triple).
   - 11d: disclosure game, sponsor strategies × regulator policies.
9. **Run, figures and write-up.**
   - Figures for the three readerships, as listed in the task.
   - Open problem: a toric-fibre-product rule for latent models (which covers identify the actors).

### Carry-over, not yet scheduled
- Study 8d (explain vs ignore).
- Multi-radius robustness.
- Level-two decisions in Section 17.
- General-graph cup closed form.
- Third- and fourth-order birth invariants, and a proof of Lambda_2.
- HH of Q[W_G] via centralisers.
- CSD Obs• comparison.
- **A full-suite rerun.** Not done since several check additions: TS9, TS10, CU1-CU7.

## Backlog items 1-5 done (2026-10-27)

1. **Section 15, André-Quillen remark.** Latent fibre: cotangent complex / André-Quillen. Observable algebra: Hochschild /
   Cuntz-Quillen. Cites Quillen 1970 and André 1974.
2. **Section 15, twirl paragraph.**
   - Cites Aniello, J. Phys. A 52 (2019) 305302, for what its abstract states (the exact twirl formula was not retrieved).
   - The conjecture is decided by the existing results; no new study was needed.
   - Covariance group non-abelian: necessary for contextuality.
   - Sufficient for CHSH, which is the existing classification.
   - Not sufficient for KCBS (BR10).
3. **Section 7, Proposition "pairwise windows".**
   - Pairwise windows see mu and sqrt(w1 w2) d, not w.
   - m_S = v_S s_|S|(r), with s_(k+1) = r s_k + s_(k-1).
   - The fibre is a segment, event probabilities are polynomials of degree <= m - 2, and one triple with v_S != 0 pins r.
   - Check: SC7 in check_scheme.py.
4. **Toric fibre product paragraph** (Sullivant 2007, Thms 3.5 and 3.7; Engström-Kahle-Sullivant 2014; Rauh-Sullivant 2016).
   - Correction from the original backlog: level two is not wholly open. Sullivant's Thm 3.7 glues hidden-variable models along
     *observed* separators.
   - Our latent model is the excluded case (hidden separator), and that case is open.
5. **Section 17, "From design to decision".**
   - Figure: fig_pipeline.py (two rows; the feedback arrow points to Evidence).
   - Proposition "design rule":
     - chordal + all maximal cliques measured + agreeing overlaps ⇒ lambda* = 1;
     - sub-covers need not glue (acyclicity is not hereditary). Counterexample with lambda* = 0. Check: TS11.
6. **Coherence edits:** abstract (pairwise sentence, design sentence), dictionary rows, related work (toric fibre products, Aniello),
   next steps 15-16, appendix (SC1-SC7, TS1-TS11, fig_pipeline).
7. **Still pending:** the title decision (user).

## Companion paper written: the latent-actor lab (2026-10-27)

- **Files:** `lab/latent_actor_lab.tex/.pdf` (14 pp), `lab/fig_lab.py` (8 figures), `engine/pce/lab.py`, `engine/study11_lab/`.
- **Study 11** (pre-registered, frozen; amendments logged before the run; deviation DV1 logged): 21 of 24 registered predictions held.
  - Q2 failed (stratum effect: A's four-variable event is affine because v_0123 = 0).
  - L7 failed narrowly (0.795 vs 0.8); post hoc the signal is |v_S|/sigma_S (0.870).
  - L8 failed; explained by Proposition "when evidence can hurt" (robustness falls only by a jump to vacuity), confirmed post hoc by
    exact bisection.
  - Illustrations: I1 and I3 held; I2 failed as registered (gradient play approaches, 0.83 -> 0.33, then cycles at 0.26-0.34).
- **Correction to Theorem C(3)** (found while drawing fig_voi): Wald's gap g is 0 iff U has a saddle point on acts x segment, not
  "iff one act is optimal throughout". For a three-variable event U is affine in r, so with treat/nothing both worst cases sit at the
  same wall and g = 0 at every cost. Savage's minimax regret is positive whenever the best act changes along the segment; fig_voi(a)
  now shows the regret VOI at a post-hoc cost 1.235 (PH4 in `posthoc_study11.py/.json`: 0.0103 per triple for G, only triple 012
  for A, 0 for M; Wald g = 0 throughout).
- **Rejected/corrected, do not reintroduce:** "g = 0 iff one act is optimal on the whole segment"; "I2: gradient play spirals away".
- **POSI resolved (2026-10-27):** Shi, Liang and Shroff, arXiv:2306.08762v4 (current title "Near-optimal partially observable RL with
  partial online state information"). Verified: hardness Omega(A^(H-1)/eps^2); tractable subclasses have regret O~(sqrt K), not
  O~(K) as first quoted to us. Cited in the lab paper's ML section as an analogue (the lab never queries the type).
- **Still open:** the main note's Section 17 / Study 8c robustness values are grid-based (exact
  bisection gives e.g. 0.0937 for 0.09); the toric-fibre-product rule for a hidden separator.

## Theorem D and Study 11b: the decision on the segment (2026-10-27)

- **Why:** Theorems A-C did not close the decision on the segment. The user's draft "Theorem E" (existence, uniqueness, stability,
  expiry) was adopted as a frame but corrected: Prop. 36's vacuity threshold is idle on the pair cover (lambda* = 1); an act is not a
  point of the segment; r -> a*(r) is the oracle map, not the satisficing act; "expiry" in Section 17 is utility-driven and not
  predictable, so the evidence-driven notion is called *settling*.
- **Theorem D** (lab paper Section 6), two types, pair cover:
  1. Existence: U_a is a polynomial of degree <= k-2; for k = 3 it is affine with slope beta_a = v_T * sum_x u(x) chi(x) (top Walsh
     coefficient); safe on the segment iff safe at both walls.
  2. Uniqueness/flat acts: set rule unique up to equal worst values; info-gap horizon = margin/|beta| or +inf (saturation ties);
     info-gap prefers small top Walsh coefficient; info-gap and set rule can differ.
  3. Invariance/stability: the set rule depends on pair tables only (constant on the fibre); a pre-probe info-gap nominal is a
     convention (midpoint in r vs in t differ); changes only at wall-value = u0 or worst-value crossings.
  4. Settling (k = 3): constraints g_j(c) - gamma_j h >= 0; radius d = min g_j/gamma_j; P(settled) -> P(g sqrt n + xi >= gamma z tau);
     median n* = (z tau / d)^2 when one constraint binds, independent of Sigma; plateau Phi(-gamma z tau / s) on the stratum d = 0.
- **Code:** `engine/pce/segment_decision.py`; `engine/check_decision_segment.py` DS1-DS9 (9/9; added to run_checks.sh; total 363).
- **Study 11b** (`engine/study11b_settle/`, frozen before simulating; smoke runs on an invented design led to the one-binding-constraint
  design rule, recorded in the registration): S1 20/20 cells within 0.05 (max 0.014); S2 median/n* = 0.94, 1.08, 1.00, 0.93;
  S3 plateau 0.27-0.30 vs predicted 0.28; S4 98-100%. All four hold. No deviations.
- **Lab paper:** 18 pp; abstract says four theorems; reading guide, VOI, persuasion (mandate needs a size n*), ML (sample complexity
  of the decision), open problems, limits and appendix updated; new fig_decide.
- **Open:** combined level-one contamination + segment; settling for k >= 4 (quadratic, interior critical point); more types.

## Part II of the lab paper: injected records (2026-10-28)

- **User's request:** study actors who feed data to tilt decisions (slopes, polytopes, geometry, topological separation), with
  comonads and schemes; injected and genuine points have relations within each set but none between. User chose: Part II of the lab
  paper; whole-record injection only (no per-window injection, no editing of genuine records).
- **Framing:** P_eps = (1 - eps) P + eps Q (coproduct of samples -> mixture; Huber contamination with intent). Environment comonad
  E x (-), E = {genuine, injected}; counit = what the regulator sees; coalgebras = labellings. Stated as framing; the theorems come from
  the polytope, Walsh characters and the tetrad scheme. Of the user's list of classical theorems, used: Maschke/characters, first-order
  and degenerate perturbation theory, dual numbers (scheme tangent), parametric LP. Not used (said so): Peter-Weyl, Weyl character
  formula, GRR, projective bundle formula, Kontsevich formality.
- **Theorem E** (slopes): reachable set = polytope; slopes linear in Q; first-order budget = margin / max single-record influence,
  attained by a point mass; exact = first-order x (1 + O(margin)); a product injection is a third latent type, rank-one preserved iff
  mean shift parallel to v.
- **Theorem F** (hidden channel, m = 3): pair-invisible injections are Q = P + s chi_T, s in a positivity segment; rhat moves by
  exactly eps s / v_T (sum_x chi prod(x_i - mu_i) = 1); eps* = |r_c - r0||v_T|/|s| < eps_det = |wall - r0||v_T|/|s| (stealth
  window always exists); finite-n half-way point (|r_c - r0| - kappa z tau / sqrt n)|v_T|/|s|; general m: shift 2^(m-3) eps s / v_S,
  so budget proportional to |v_S| (best probe = hardest to poison). Channel capacity bounded by the rarest outcomes.
- **Theorem G** (m = 4): tetrads; N(Q) = 0 injections are dual-number points of Spec R[M]/(T1, T2) (tetrads move at eps^2);
  stealth-tilt frontier concave piecewise linear (parametric LP); Jacobian rank 2 generic and single coincidence, 1 for any two
  coincidences, 0 at merge.
- **Code:** `engine/pce/injection.py`; `engine/check_injection.py` IN1-IN8 (8/8; total checks 371).
- **Study 12** (`engine/study12_inject/`, frozen 2026-10-28): H1 10/10 cells, eps_half 0.0581 vs 0.0587; H2 crossing 0.1636 vs
  eps_det 0.1635; H3 at eps = 0.07: 69% taken over, 0 flagged; H4 stealth rejection 0.045-0.048 at level 0.05, vertex 0.84/1.00/1.00.
  All four hold. **Disclosed irregularity:** the smoke run's tetrad part used the m = 4 truth then intended for H4; it was demoted to
  smoke use and H4 moved to a fresh truth G4 before freezing (criteria in code unchanged). Recorded in the registration and the paper.
- **Paper:** title now "A Lab for Latent Actors: Hiding Segments, Probes, Settled Decisions and Injected Records" (25 pp); abstract,
  reading guide, related work (partial identification: Manski 2000, Stoye 2009, Giacomini-Kitagawa 2021, Montiel Olea-Qiu-Stoye;
  Duarte et al. 2024; SLT: Watanabe 2009, Drton-Plummer 2017; robust stats/poisoning: Huber 1964, Hampel 1974, Biggio et al. 2012,
  Steinhardt-Koh-Liang 2017, Diakonikolas et al. 2019; tetrads: Drton-Sturmfels-Sullivant 2007), open problems, limits, appendix.
- **Open:** per-window injection (breaks gluing, lambda* < 1); editing (TV ball); adversary uncertain about P; cohomological
  description of the second-order obstruction; window design minimising channel capacity.

## Theorem H, G(4) and the two-boundaries proposition (2026-10-28, after user review)

- **User asked for:** duality theorem (adversary's and regulator's segments share 1/v_T); partial-identification paragraph; explicit
  André-Quillen statement; frontier as adversarial analogue of Theorem D(3)'s margin (two boundaries of one geometry). Also sent a
  comonad account of detection ("detection = counit failing to commute; stealthy iff coalgebra morphism; best detector = triple;
  overlaps cheap"). **Declined, with reasons given to the user:** the counit is natural and commutes with every map; stealth is a
  property of where the injected law lands, not of coalgebra morphisms; the triple is the attack surface (m = 3 flags only at ~16%,
  after the ~6% takeover); overlaps detect nothing under whole-record injection (Theorem E(1)). Detection lives in model equations
  involving the moved moment (a second triple at m = 4).
- **Theorem H (one geometry):** (1) m = 3: the pair fibre is the line P + s chi; the regulator's segment is its model part, strictly
  inside its positivity part; triple-moment coordinates v_T R inside m_T + [s_lo, s_hi]; 1/v_T converts both. (2) m = 4: pair fibre
  dim 5; first-order directions preserving every model equation (triple-consistency of r_S, quadruple m_0123 = v_0123 (r^2 + 1);
  tetrads constant on the fibre) = the tangent of the hiding segment; a single chi_S channel is seen by a second triple (0.629 ->
  0.533 on {0,1,2}, others unchanged at eps = 0.05).
- **Proposition (two boundaries):** decision boundary r_c (regulator's margin, post-probe D(3)) vs detection boundary (walls; at m = 4
  frontier + model equations); stealthy takeover iff decision boundary nearer; always when r_c in R.
- **Theorem G(4) (obstruction, André-Quillen):** ring R[M]/(T1, T2) commutative; obstruction in Ext^1(L_{A/R}, kappa(P)) = coker J on
  the quadratic term (regular sequence). **Correction of my earlier text:** the eps^2 tetrad movement of a straight stealthy injection
  is curvature, not obstruction; bending (Q depending on eps) removes it at smooth points (slope 3.0 at generic and single
  coincidence); obstruction non-zero on the two-coincidence stratum (slope 2.0; coker = T1 - T2, quadratic term dM03 dM12 -
  dM02 dM13). Hochschild is for the noncommutative observable algebras (main note Section 15); HKR ties them only at smooth points.
- **Partial identification:** Horowitz & Manski, Econometrica 63 (1995) 281-302 added (contaminated data; regulator's side of the
  reachable polytope).
- **Code:** `engine/pce/duality.py` (new; `pce/injection.py` untouched because it is frozen with Study 12); checks IN9-IN11 added to
  `check_injection.py` (11/11; total 374). New figure fig_oneline. Paper 27 pp.

## Worked example, end to end (2026-10-28)

- **User asked:** one fully worked example in the lab paper (no new paper), following a 10-step guide (tables, polytope, tower, sheaf,
  contextuality, VOI, poisoning, detection, Stanley-Reisner, breaking). Built on the main note's drug example (population 1:
  w = (0.6, 0.4), a1 = (0.5, 0.1, 0.4), a2 = (0.05, 0.5, 0.1)).
- **Lab paper Section 17** (Steps 0-12, Table "example", fig_worked): cells 0.333 ... 0.013 (exact to 3 dp); channel capacity
  s in [-P(111), P(110)] = [-0.013, 0.027]; level one [0.654, 0.694]; mu, m_ij, m_ABC = 0.002592, v = sqrt(0.24)(0.45, -0.4, 0.3),
  v_T = -0.006349, r0 = -0.408; segment r in [-0.763, 0.573] (walls a2A = 0, a1B = 0); level two [0.66077, 0.66925]; w1w2 = Delta_m/Det
  = 0.24 (top rung); decision T (1 - [any]) vs pair AB (0.872 - [A or B]), u0 = 0.30: set rule picks the pair before the probe
  (0.332 > 0.33075), oracle the triple (0.333), switch at harm 0.668; Wald gap 0, regret VOI 0.00125 (Savage picks the triple);
  tau = 12.71, d = 0.1575, n* ~ 25,000; vertex attack record 111 at 0.19%; hidden channel eps* = 0.001/0.027 = 3.7%, eps_det =
  0.00225/0.027 = 8.3%, eps_half(n = 1e5) = 1.85%; rung change needs record 011 at 5.4%; SR Groebner basis w1(1 - w1) di dj (SymPy).
- **Guide items handled differently (said so to the user):** the 7%/16% figures belonged to Study 11b's D2 truth, not this one (here
  3.7%/8.3%); detection is not "counit failure"; the per-window H^1 story is replaced by what the main note actually computes (overlap
  disagreement, linear gap 0.28, lambda* = 0.86, bottom rung); "the complex grows a facet under injection" is not right: the complex
  belongs to the model; what breaks is the post-probe fibre (empty past eps_det); a product injection would need a three-type model.
- **Correction to the main note:** Example "latent structure sharpens prediction" reported the level-two interval [0.6613, 0.6687] from
  a 49-point weight grid that misses the walls; exact is [0.6608, 0.6693] ("almost five times narrower"). Fixed in the text; LV8 in
  `engine/check_level2.py` now computes the exact interval (grid kept as a cross-check, which lies inside); level2_results.json and
  fig_level2 regenerated; main note cites the lab paper (bibitem Lab).
- **Code:** `engine/worked_example.py` (WE0-WE13, 14/14; in run_checks.sh; total 388), writes `lab/worked_example.json`.

## Proof details and corollaries (2026-10-28, user's five points)

1. Theorem H(1) proof: explicit containment (Pi ends where a cell reaches 0; X ends at a box wall where the other type still gives every
   cell positive mass; worked example: smallest wall cells 0.0152 and 0.0068).
2. Corollary "second-triple defence" (m = 4): chi_S moves r_S alone by 2 eps s / v_S; any second triple with v_S' != 0 sees it; a
   model-invisible injection must move along the hiding segment. Numbers in the paper (0.629 -> 0.533; others 0.629); check IN12.
3. Theorem G(4) stated exactly: ob(A) = [H(A,A)/2] in coker J; vanishes for every A when J is surjective. **A non-zero cokernel is
   necessary, not sufficient** (user had "non-zero exactly when the cokernel is non-trivial"; corrected): on the two-coincidence stratum
   ob(A) = q(A) = dM03 dM12 - dM02 dM13, zero exactly on a quadric cone; at the merge ob(A) = H(A,A)/2.
4. Theorem G(2) closed form: Phi(delta) = min_lambda [max_x (IF(x) - lambda.n_x) + delta |lambda|_1] (LP dual); optimum mixes <= 3
   records; reaches vertex tilt at ||n_x*||_inf. H4 truth: 10.144 + 0.417 d to 0.219, slope 0.201 to 2.744, then 10.743; supports
   3/2/1 (1100, 1101, 0111 -> 1100, 1101 -> 1101). Check IN13.
5. Corollary "capacity": cap_+ = min_{chi_S<0} P(x), cap_- = min_{chi_S>0} P(x); eps*(S) = |r_c - r0||v_S| / (2^(m-3) cap); prefer
   triples with large |v_S| / cap. On truth G the order by |v_S|/sigma_S equals the order by poisoning resistance (shifts 4.61, 5.94,
   7.75, 10.85). Check IN14.
- **Toric proposal (coupling polytopes, 2x3 family) not built:** the alphaxiv PDF was unreadable (metadata only); several claims in the
  attachment do not hold in our setting (see the reply to the user). Asked for the PDF.

## Step 13 of the worked example: the toric reading (2026-10-28)

- User supplied the coupling-polytope companion (Awasthi, "Coupling polytopes and toric geometry: the 2x3 family, the square case,
  higher levels and stability", Oct 2026 revised) and asked for as much algebraic geometry as fits.
- Added to lab paper Section 17 (fig_toric; table rows; bib: Coupling, Delzant 1988, Pachter-Sturmfels 2005, GKZ 1994, Cho-Oh 2006):
  the line = boundary fibre of shape 2x2x2 [Coupling Prop 10.5(i)], Delzant segment, CP^1, lattice Z chi, length 0.040; facet values at
  the truth 0.027, 0.013 = channel capacities, [omega] = 0.027[D+] + 0.013[D-]; moment map = third cumulant (m_ABC = 0.002592 + s);
  adversary's Q = north pole (p110 = 0) [Prop 6.4]; Birch quartic restricts to the cubic -s^3 - 27/1250 s^2 - 37971/5e6 s -
  143613/6.25e9 with one real root s = -0.003048 (max entropy, harm 0.6640, inside the hiding segment, below 0.668; immune to the
  channel but forgoes the probe), complex roots -0.00928 +- 0.0863i, fourth point at infinity; Cayley hyperdeterminant on the line =
  s^2 + 81/15625 s + 6561/39062500 = m_ABC^2 + 4 Delta_m, roots -0.002592 +- 0.012698 i (never real: real rank 2 throughout); near-wall
  (cell 011 at 0.004 at the north pole); monotone midpoint s = 0.007 (Cho-Oh: the only Floer-nontrivial fibre; no statistical meaning).
- Not adopted (reasons given to the user): Bridgeland central charges / Kaehler class as stealth detectors (fixed under stealth), parity
  horns (per-window), the 2x3 chamber table and del Pezzo surfaces (different shape); "level two" differs between the two papers.
- Checks WE14-WE18 (exact rationals via SymPy); worked_example.py now 19/19; total 396.
- Suggested next: the four-variable version (5-dim boundary fibre; simple/smooth or orbifold? [Coupling Prop 10.5(ii)]).

## Four-variable worked example, staged (2026-10-28; lab paper Section 18)

- Truth: w = (0.6, 0.4), a1 = (0.5, 0.1, 0.4, 0.2), a2 = (0.05, 0.5, 0.1, 0.7); cells exact 4-decimal rationals (0.1809 ... 0.0031).
  Types labelled so v_A > 0 (fit_v alone would make v_D positive and flip r's sign).
- Stage 1 faces: r_S = -0.408 on all four; capacities ABC 0.027/0.013, ABD 0.027/0.013, ACD 0.0486/0.0254, BCD 0.0252/0.0188; odds
  ratios 0.569, 1.449, 0.522, 1.132 (sign = side of the face Birch point); all face hyperdeterminants positive.
- Stage 2: global pairwise max-ent (IPF) induces three-way odds ratios 0.977, 0.985, 0.960, 0.965 on the faces (not their own Birch
  points); four-way odds ratio of the truth 0.507.
- Stage 3: pair fibre dim 5; chi-lattice index 16 in the saturated lattice; 20 vertices, 9 facets (cells 0011 0110 0111 1001 1011 1100
  1101 1110 1111), simple and Delzant -> smooth toric 5-fold, Euler char 20, f = (20,50,54,31,9,1), h = Betti = (1,4,5,5,4,1);
  Kaehler data = the nine facet cells' probabilities; lattice volume 1.27e-8. Two-face projections (subtorus moment maps): five
  rectangles, ACD x BCD a pentagon (corner cut 0.002).
- Stage 4: model = conic arc (3 linear + quadric m_ABCD = v_ABCD (r^2+1), v_ABCD = 0.001555); D's walls slack (t in [0.408, 1.633]):
  R4 = R3, ABC interval unchanged.
- Stage 5: chi_ABC on 16 cells capacity 0.0099 (face units 0.0198): flip 5.05%, wall 11.4%; second-triple test (n = 1e5, sd of
  rhat_ABC - rhat_ABD 0.047) fires at 3.0%; model-invisible (pairs + model kept) exact budget 1 - min_x P_rc(x)/P(x) = 18.3% (wall
  42.2%; gives 3.7% at m = 3).
- Stage 6 chambers: a2D 0.5, 0.7 smooth; 0.6 wall (not simple, 17 vertices); (a1D, a2D) = (0.3, 0.7) orbifold (index 2); 50 random laws
  always simple, never smooth.
- Code: `engine/worked_example4.py` (W4-1..W4-14 incl. 6b, 6c; 16/16; in run_checks.sh; total 412); figure fig_worked4.

---

## Sources (Theorem J) and Study 13 (2026-10-28)
- Theory: lab paper Section "Part II, continued. Sources": three sources, batches, one trusted source C adversarial in some batches
  through the hidden channel. Pooling = monad multiplication mu: D(DX) -> DX; gluing = membership in the image of eta; the labels are
  the provenance comonad's coefficients. Filter: batches against the median of source means (no swamping; a leave-one-out reference
  swamps), then sources (Bonferroni c_b = 3.078, c_s = 2.394). Theorem J: band B = a sqrt(pi/(1-pi)) pooled se at every N; no tilt
  (wrong-settle <= Phi(B - z)); delay N_1/2 <= ((z tau + B sigma_m/|v_T|)/d)^2; localisation; two sources cannot localise.
- Code: `engine/pce/sources.py` (frozen with Study 13), `engine/check_sources.py` SR1-SR10 (10/10).
- Study 13 (`engine/study13_sources/`, frozen 2026-10-28, FROZEN_PREREG_study13.txt): truth W (Section 17), K_C = 8, rho = 25,
  N = 12k..192k, 2000 reps. Smoke runs only on invented truth I (two runs, disclosed); filter order changed before freezing on
  Gaussian-model grounds (disclosed). Results: H1-H5 all held. H1 100/100 cells within 0.05 (max 0.031); H2 glue wrong <= 0.0195,
  naive wrong under A2 0.877 at 192k, glue vs oracle under A2 <= 0.006; H3 N_1/2 24.7k / 52.8k / 55.7k vs 24.6k / 53.0k / 56.6k;
  H4 A3 batch removed 0.96-0.97, C dropped 0.011-0.017; H5 worst right-rate at 192k glue 0.918, naive 0.
- Figure fig_sources (bands, settling curves, gluing-statistic paths, tilt vs delay).

## Lifting (Theorem K) and propositions (2026-10-28)
- Lab paper Section "Lifting: the decision maker's patchwork": Theorem K (decision arrangement: U_T - U_P = 0.128 - P(001), slabs at
  m = 4; act lift 85/15, 0.00107; blow-up of the merge, rho^2 at n^-1/2; adversarial lift block radius 1/128 (pair line 1/14,
  forward 0.037); orbifold atlas). Remarks: stability of the lift (exact; analogy to Fournier-Facio-Willett arXiv:2603.18456),
  smoothness deceives (analogy to Li arXiv:2607.19654). Propositions: dilation (width = Delta s / P(a,b); Delta s = 0.00848),
  value of a probe in its size (EVSI flat start, inflection ~1700, EVPI 9.24e-5, optimum ~18k at c = 1e-9; Wald value negative at
  every n), backtracking (worst constant-dose shift 1.05 se forward-only vs 0.57 backtracking), price of not gluing
  (VOI >= -osc(u) d_H(C, rect C); example -0.1 vs bound -0.3). Conjecture (sharpness; gluable fraction of a tree) not tested;
  testing would need its own pre-registration.
- Code: `engine/pce/lifting.py`, `engine/check_lifting.py` LF1-LF10 (10/10, ~10 min). Figure fig_lifting.

## One picture (Theorem L, 2026-10-28)
- Lab paper Section "One picture: chambers, slices and gauges": (1) settling iff M in one chamber; (2) Savage value R(M) - max R(S_y),
  Wald value positive iff a wall inside M and strictly opposite slopes; (3) robustness = margin / capacity (forward: support function
  of the gauge; observer: water-filling, delta/(c - min l) when the lowest cell suffices; sampling (z sigma_l/delta)^2; labelled
  B sigma_l/sqrt N); (4) a gluing failure along xi is decision-relevant iff l(xi) != 0; (5) stealth window iff wall inside M.
  Scope in the statement: two types, pair cover; (1), (2), (5) at m = 3, (3), (4) any m. Worked example: margin 0.001 over 0.873,
  0.128, 0.027, 0.014, sigma_l. m = 4 support function = m = 3 (3.70%) < single chi_ABC channel (5.05%).
- Code: `engine/pce/capstone.py`, `engine/check_capstone.py` CL1-CL9 (9/9). Figure fig_capstone. Corrected an external note's
  claim that VOI is "the fraction of slices that separate chambers" (false: point slices separate yet Wald value is 0 at a saddle).

## The descent tower (Proposition "spectrum and descent", Theorem M, 2026-10-29)
- Lab paper Section "The descent tower". Level 0: algebraic descent holds up to the kernel spanned by uncovered Walsh characters (dims 1, 5,
  1); linear Cech H^q = 0 for q >= 1; real descent fails only through positivity (A=B, B=C, A!=C: lambda* = 0, nu* = 1/2). Corrected
  an external note: "global compatible sections = joint laws" holds only algebraically mod the kernel.
- Level 1 (M = delta+ v, unsigned coboundary): kernel Z/2 = label switch (R-kernel 0 non-bipartite, scale on bipartite); obstruction
  H^1_+(G; R^x) = H^1(G; Z/2) + coker B: triangle Z/2 = sign(Delta_m); K4 (Z/2)^3 + R^2, coker = the tetrads; toric ideal of B; walls
  are semialgebraic. Level 2: delta r on triples (a chi_S channel moves the obstruction up exactly one level). Labels: degree-0 obstruction
  delta^0 m_T on a simplex nerve, energy = K (between + within). The image diagram forgets the sign obstruction; the realization diagram
  (stack quotient by Z/2) keeps it.
- Code: `engine/pce/descent.py`, `engine/check_descent.py` DC1-DC14 (14/14); Remark "three cases" (incongruence, contextuality, gluing; liar-cycle inequality P(A=B)+P(B=C)+P(A!=C) <= 2, value 3).

## Filtration and the CSD framework (2026-10-29)
- Corollary "the descent filtration" D0 > D1 > ... > D5 (compatible, glue, sign 0, modulus 0, walls, triple coherence), each strict
  (DC3, DC5, DC10, DC16); order of D2-D4 a convention. General count for any connected pair pattern: E - rank B model equations
  (E - m non-bipartite, E - m + 1 bipartite), b1 sign conditions, 2^-b1 reachable patterns (DC15, 57 random graphs).
- Section "Relation to the CSD framework" (Chang, hal-05711275; the two attached PDFs were identical): Spec(A) not constructed (no
  operad in the classical setting). Proposition "the lab inside CSD": (1) no invariant factoring through linear Cech cohomology or the
  support presheaf characterises gluing of probability laws (liar cycle; noisy liar full support, contextual for 0 < q < 2/3, glues
  from 2/3); (2) where cohomology detects (all-versus-nothing, latent sign) the degree is 1, H^1(Gamma; Z/2); Cech H^2 = 0 on the
  lab covers (nerve Betti m=3: b1 = 1; m=4: b1 = 3); (3) the lab's degree-2 obstruction is the Andre-Quillen deformation class
  (Theorem G(4)), matching CSD's tangent-complex reading; (4) the Z/2 realization stack is the first groupoid case, S_k open.
- Code: `engine/check_csd.py` CX1-CX5 (5/5); `engine/check_descent.py` now DC1-DC16.

## Worked descent complex (Section 17 Step 14, Section 18 Stage 7, 2026-10-29)
- Step 14: nerve = hollow triangle; Cech dims 12, 6, 1, ranks 5, 1, H = 7, 0, 0; marginal map rank 7, kernel chi_ABC (the line s).
  Stages: D1 lambda* = 1 (inequality value 1.40; liar 3); D2 product 4.03e-5, sigma (+,-,+), label switch, witness "exactly one harm"
  (M = -1/9, product -1/729); D3 coker 0, |v| = sqrt(0.24)(0.45, 0.40, 0.30); D4 t in [0.689, 1.327], witness with crossing walls;
  D5 r0 = -0.408 in R, rhat = -0.834 at eps 0.10. Labels: one A1b replicate, energy 24 (1.48e-5 + 1.12e-4), z_C 1.53.
- Stage 7 (m = 4): log tetrads 0 on truth, (-1.65, -0.49) after product injection at 0.1; r_S = 0.408 (label switch of -0.408),
  chi_ABC at 0.05 moves r_ABC to 0.486 only.
- Code: `engine/worked_descent.py` WD1-WD11 (11/11, writes lab/worked_descent.json). Figures fig_descent, fig_filtration.

## Two comonads and the comonadic form of the tower (2026-10-29)
- Descent section: paragraph "Two comonads" (provenance W = L x (-): co-Kleisli = label-aware rules; cover comonad G = f^* f_*:
  coalgebras = descent data; they meet in the labelled cover; pooling = Giry monad multiplication). Proposition "the comonadic form of
  the descent tower" (user's draft, corrected: object-level comparison at one presheaf; surjectivity/acyclicity proved, not assumed;
  the latent class is a class; positivity automatic for the labelled whole-record case; m = 3 batch laws equal). "How each classical
  theorem applies": Eilenberg-Moore, Benabou-Roubaud, Beck (object-level counterparts only), Godement, Dold-Kan normalisation, Giraud.
- Check DC17: unnormalised ordered Cech complex (dims 12, 24, 54) has cohomology 7, 0, 0 like the alternating one.

## Relation to Ikeda's entanglement obstruction (arXiv:2511.04326v2, 2026-10-29)
- Paragraph + Proposition "the lab inside the quantum marginal problem" (end of the CSD section): Ikeda's (A) gluing failure / (B)
  non-uniqueness of j_U = the lab's (ii)(b) / (ii)(a) of K. (1) dephasing commutes with partial trace: classical and quantum gluing
  agree on diagonal families (liar cycle fails in both); (2) quantum kernel on the pair cover = 27 weight-3 Pauli strings vs classical
  chi_ABC = -ZZZ; (3) witnesses are supporting hyperplanes in both. Theorem 3.1 (BSS/Dupont, D_A W_sep = 0) as comparison only;
  normalisation (2 pi i)^-k Tr F^k = (-1)^k k! ch_k. The lab's curvature-type invariant is the Kahler class of the toric CP^1: lattice
  length 0.040 = total capacity, unchanged by the hidden channel.
- Code: `engine/check_quantum.py` QU1-QU4 (4/4).

## Markov categories (2026-10-29)
- Section "Markov categories: factorisation, sufficiency, de Finetti and Blackwell", Proposition "the lab in a Markov category" with four
  hand-worked examples: string diagram (kernels; P(000) = 0.162 + 0.171); Fisher information about s: pairs 0, triple sum 1/P = 177.41
  (82% from 011, 110, 111), efficient sd 0.0751 vs sigma_m 0.0797; four batches (one at share 0.2): pi = 3/4 delta_P + 1/4 delta_P0.2,
  mu(pi) = P_0.05 (tilted, unflagged, rhat -0.621), exchangeability defect 0.0028, Var_pi(m_T) = 5.47e-6; Blackwell on three states:
  Bayes 0.334327 = 0.334327 < 0.334744 (noisy probe on a plateau), maxmin 0.332 throughout, fixed rule 0.331875.
- Correction: the probe-size proposition's "negative Wald value" is a property of the fixed set rule; ex-ante maxmin-optimal rules are
  Blackwell-monotone (value 0 here). Abstract amended. Tangent categories: one correspondence (Theorem G(4) as a jet-lifting obstruction).
  Roumen not tested; no claim. Citations verified online: Fritz 2020, Fritz-Gonda-Perrone 2021, Chen et al. 2025, FGPR arXiv:2010.07416,
  Cockett-Cruttwell 2014, Roumen EPTCS 236, Manski 2025.
- Code: `engine/pce/markov.py`, `engine/check_markov.py` MK1-MK6 (6/6, writes lab/markov.json). Figure fig_markov.

## Roumen, causal semantics, Theorem N (2026-10-29)
- Roumen (EPTCS 236 (2017) 174-201, read in full from the user's PDF): cyclic cohomology detects only signed extension (Thm 9.1 and
  start of Sec. 10) = our no-go (i); order cohomology (precones, ordered coefficients, monoids, non-exact) is complete (Thm 10.5) and by
  Prop 10.4 equals "sigma is the restriction of a positive measure on P(X)" = the marginal LP. Checks RO1-RO3: signed test passes on
  the whole noisy-liar family; literal H^1_<= test (6049 disjoint-pair constraints) fails exactly for q < 2/3; equals the LP on 200
  random families. Obstruction in relative degree 2 -> CSD-section degree statements qualified ("Cech on Gamma"; degrees depend on
  the complex).
- Causal: whole-record injection = soft intervention on the provenance node E in E -> X (CA1); per-window injection is not one model
  and can break gluing (lambda* 0.78, CA2). Yin-Zhang and Di Lavore-Roman-Sobocinski cited as language only.
- Theorem N (assembly, Section "Assembly: descent, identification and decision"): (i) existence (torsor cut by positivity; Roumen
  precone; labelled delta^0), (ii) latent membership (Z/2 class + walls; D5), (iii) decision sensitivity (Theorems D, L, J); the three
  fail independently; scope two types, pair cover, m = 3.
- Code: `engine/check_roumen.py` RO1-RO3, CA1-CA2 (5/5).


## Tangent structure (companion b, 2026-10-29)
- Paper: `tangent/tangent_structure.tex` -> 12 pp. Theorem 4 (cone structure on lexicographic sets/polyhedra with affine maps; universality iff relint, fibre C_x x C_{x,u}); Prop 6 (stochastic matrices; non-affine counterexample at a vertex); Prop 7 (interior: Theorem E(2) = T F, stealth vertical, pole of T rhat at coincidences, wall defect |l(xi)|); Theorem 8 (tetrad scheme: universal lift; completion class [H(u,v)], = |q|/sqrt2 on the two-coincidence stratum; 2-jets = c-fixed diagonal points, diagonal = 2 ob(A)); Corollary 9 (dichotomy).
- Code: `engine/pce/tangent.py` (W_k arithmetic, substitutions, lexicographic lemma, universality, tetrad jets); `engine/check_tangent.py` TG1-TG13, TG6b (14/14); writes `lab/tangent.json`.
- Lab paper: the paragraph "Tangent structure, built" replaces "at one point"; scope-table row added; bibitem TangentCompanion.
- Open: non-affine maps at corners (restriction categories or a positivity condition on D^2 f); next is the two-type arbitrary-cover theory with k=3, m=5.

## Arbitrary covers (companion c), Part I (2026-10-29)
- Paper: `covers/arbitrary_covers.tex` -> 6 pp. Lemma 1 (rank), Theorem 2 (factorisation: zero strata, sign class, modulus class; product form of the solution set), Theorem 5 (hiding polytopes; rectangle for bipartite; dim b(G)+1; 2^{c*-1} sheets; K3 = lab segment); examples C4 (all numbers), two edges at w = 0.9 (sheet killed), triangle at a coincidence.
- Code: `engine/pce/covers.py`; `engine/check_covers.py` CV1-CV9 (9/9, ~2 min); writes `lab/covers.json`.
- Next: Phase III (extended incidence matrix with windows; relative signs by parity), Phase IV-V, then k=3, m=5 (pentad, O(2), S_3 sheets).
- Phase III Stage 1 (2026-10-29): Theorem 9 (toric ideal of the extended incidence matrix; pair part = even closed walks; test ladder; Hochster homology; syzygies = dependencies among tests, df = codim); bowtie Hilbert-Burch worked example; Betti table for 8 covers. Code `engine/pce/syzygy.py` (Macaulay2 runner, Hochster's formula), `engine/check_syzygy.py` SZ1-SZ8 (8/8, ~30 s; needs Macaulay2). Paper now 10 pp. Next: Stage 2 (windows of size >= 4, non-toric), Stage 3 (real signs, parity), Stage 4 (strata; cut-vertex df conjecture).
- Phase III Stage 2 + three types (2026-10-29): Lemma 10 (s_k = moments of the standardised two-point latent); Theorem 11 (graded pieces; Fibonacci gluing = Hankel; ideals of the full 4-window, tet+tri, two tets, full 5-window; the lab's 6 residuals = codim); Prop 12 (collinear k types: moment space, sizes 3..2k-1, two types = lower boundary, one-sided test, M3 range = two-type segment); Prop 13 (general 3 types m=5: pentad, ladder 14,15,16,16,17, 4-window defective 13/14, two overlapping 4-windows identify = S_3 descent, coincidence rank 14). Code `engine/pce/threetype.py`, `engine/check_higher.py` HW1-HW8 (8/8; Macaulay2 + Singular). Paper 14 pp. Open: Betti of the two big covers; Stages 3-4; general three types on covers.
- Tangent on decision boundaries (2026-10-29): Theorem 10 (second-order cone morphisms: bilinear nonnegativity of vanishing cells' Hessians on K_y = C_x cap ker Df_y); Corollary 11 (subcategory: stochastic maps, products, conditioning as a restriction-category map); Prop 12 (pooling paradox iff boundary curvature; affine decisions never flip; lab: II eigen -4.72..2.35, flips for shares 0.14-0.85); Prop 13 (2-jet budgets; reversals for records 010, 101; hidden channel exact). Code `engine/pce/tangent_decide.py`, `engine/check_tangent_decide.py` TD1-TD7 (7/7). Lab paper paragraph updated (71 pp).
- Layers (2026-10-29): Definition 14 (layered maps: strata with restriction idempotents, resolutions, multi-maps); Theorem 15 (open faces universal; corner fibre = union over entered layers, C_{x,u} = C_{G(x,u)}; layered bundle closed under +); Prop 16 (coincidence lifted to parameters; merge blown up, rhat a coordinate on the exceptional line; triangle multi-map polygon closing at lam* = 0.3115; branch dies at w* = 0.6231; pipeline domain not convex); Example 17 with Figure 4 (six panels). Code `engine/pce/layers.py`, `engine/check_layers.py` LY1-LY8 (8/8).
- Phase III Stage 3 (2026-10-29): Theorem 14 (signs: G_2 = ker P^T sheets; relative signs over F_2, the odd-meeting rule only a special case; visible vs invisible sign classes, invisible = 2-torsion of coker A (even Smith invariants), b_1 = codim I_B + #non-bipartite components; syzygies at the torus identity = dependencies among sign tests; sign ladder (1,1)/(0,0)/(1,0)/(0,1), (1,0) iff an edge closes an odd cycle). Example 15 (two triangles sharing only r) + Figure 4. Code `engine/pce/signs.py`, `engine/check_signs.py` SG1-SG8 (8/8). Next: Stage 4 (strata).
- Phase III Stage 4 (2026-10-29): Theorem 16 (strata: latent rank = rank A on faces avoiding Z + singly-met coincidences; tests: A-degree >= 2 generators vanish to 2nd order; pair graphs lose max(0, lobes-1) df (>= proved, = verified); cut-variable conjecture refined; r = 0 un-aligns labels; merge rank 0). Example 17 (bowtie strata) + Figure; Section 10 summary table. Code `engine/pce/cover_strata.py`, `engine/check_cover_strata.py` CS1-CS8 (8/8; originally saved as strata.py / check_strata.py, which overwrote the walls-and-chambers files — fixed 2026-10-29, see below). Phase III complete.
- Tangent paper, reviewer points (2026-10-29): Def 18 + Prop 19 (restriction category of the lab's partial maps with joins; T a restriction functor; R1-R4 checked on 600 laws); Prop 20 (tangent structure on a hiding sheet = its tangent cone; worst case certified by the cone; facet shifts); Theorem 21 (pooling along a chord: q, h of degree <= 6; <= 4 domain intervals, <= 6 act changes; Sturm certificate; holes force flips; leaving the domain = the odd-cycle sign class). Global picture: no flips inside a domain interval in 450 pairs; failure is by leaving the domain. Code `engine/pce/restriction.py`, `engine/check_restriction.py` (11/11). Covers paper: remark linking sheets/pooling to the tangent paper.
- Gaps 1-2 (2026-10-29): covers paper Lemma "algebra, reals, positivity" (I_A = ideal of the image; fibre = torus coset of dim m+1-rank A; real points = |G_2| translates; positive part = sheet polytope; nesting algebraic > real > positive with witnesses); tangent paper Prop "restriction to the hiding polytope" (cone structure restricted; facets = positivity walls; universality fails exactly on walls). `engine/check_bridge.py` BR1-BR5 (5/5). Gap 3 (Ehrhart ring / Fisher exact test / settling as wall-crossing) awaits the coupling-algebra paper.

- Gap 3 (2026-10-29): the decision functional and the ring of contingency tables (companion "Coupling algebras", Sec. 7). Lab Section 8
  (`lab/sec_ehrhart.tex`): Prop. 2 (hiding set inside the ring: fibre = lattice segment, count k l + 1; R strict sub-segment with
  irrational ends, no quasi-polynomial; slope = Walsh coefficient), Prop. 3 (worst case = tropical limit of the weighted count, not a
  section; over the full fibre an integer programme solved by the single quartic Groebner binomial, linear on chambers; n* not a
  wall-crossing but only ~10 lattice steps from the wall; exact conditional probe efficient, tau_c <= tau_m). Code `engine/pce/ehrhart.py`,
  `engine/check_ehrhart.py` EH1-EH7 (7/7). Study 11c (pre-registered, frozen SHA-256): E1, E3, E4 hold; E2 fails (D5: exact 1954 vs
  plug-in 1860; conservativeness of the exact interval cancels the 8% efficiency gain). Figure `lab/fig_ehrhart.py`.
- Repair (2026-10-29): `engine/pce/strata.py` (walls and chambers) restored from an Oct-18 copy; `engine/check_strata.py` rebuilt from the
  report's Theorem (a)-(g) because the original file had no surviving copy (8/8); `check_persist.py` and `report/fig_strata.py` import it
  again. Covers Stage 4 now `cover_strata.py` / `check_cover_strata.py`, labels CS1-CS8.
- Stage 1 consolidation (2026-10-30): full check suite re-run after the strata repair (log `CHECK_RUN_20261030.log`, with the
  FROZEN.txt hash of the snapshot, Python/numpy/scipy/sympy, Macaulay2 1.22 and Singular 4.3.2, per-script seconds; `run_checks.sh` now
  prints timings). Shared appendices `shared/symbols.tex` and `shared/status.tex` in all three papers. Lab: reading guide and short
  scope table now one float. Tangent and lab: the exact conditional probe as a composite of partial maps in the restriction category.
  Covers: the O(2)/S_3-torsor plan reworded as a conjecture. Papers stay companions and cross-reference each other.
- Stratified descent (2026-10-30): covers Section "Stratified descent and the four obstruction levels" (`covers/sec_levels.tex`).
  Def. (multiplicative incidence complex delta: (R^x)^{V*} -> (R^x)^{E*}, (delta v)_ij = v_i v_j; a two-term complex, not a Cech
  resolution); Prop. (H^1_inc = H^1(G*; Z/2) + coker B, both in degree one; factorisations a torsor under H^0_inc = (R^x)^b x (Z/2)^q,
  free isolated vertices kept separate); level 0 transitions (triangle to edge); Prop. positivity (Farkas certificate; each sheet is a
  UTVPI system, so after doubling nodes a negative cycle exists iff the sheet is empty, proved; two components suffice by Helly in
  dimension one). Law descent (the lab's acyclic signed Cech complex) kept separate from factorisation descent, in both papers.
  Code `engine/pce/levels.py`, `engine/check_levels.py` LV1-LV8 (8/8). Figure `covers/fig_levels.py`.
- Lobe rule closed (2026-10-30): covers Lemma "embedding dimension at a face" (tangent space of a not-necessarily-normal toric variety
  at a point of a face orbit: rank of the face lattice plus the indecomposables of S'/U). Theorem "strata" (3) now exact,
  max(0, l_i - 1) on pair graphs; new (3') loss = rank A_{K-i} + mu_i - rank A on any cover with pairs and triples. Code
  `cover_strata.excess_at`, `gens_only`; check CS9 (17 covers, 85 coincidences). Exploratory run before the write-up: 314 coincidences,
  0 mismatches. Remaining Phase III loose end: the two largest Betti tables.
- Study 14 (2026-10-30): magnesium in acute MI. Registration frozen 20:00:58Z before extraction; counts from Egger, Davey Smith &
  Altman 2001 Table 18.2 (metadat dat.egger2001), trials 1-7 reproduce Teo 1991's totals exactly. Observed non-rejection band
  B = 1.96 sqrt(pi/(1-pi)) (the c_s = 1.96, beta = 1/2 case of Theorem J(2)). M1, M2, M4, M5, M6 held; M3 failed (M - B = 2.08).
  Before ISIS-4: statistically settled at h = 0.5, 1pp but not decision-settled; h = 0 verdict fragile (3/7 leave-one-out undo it).
  Post-run exploratory: Egger regression p = 0.17 before, 0.018 after; LIMIT-2 alone M = 2.04. Lab Section 22, now 82 pp.
- Global S_k descent (2026-10-30): covers Theorem "S_k descent through a shared overlap" (stack the window tables along the private
  parts into one three-way array; Kruskal: kappa(A*) + kappa(B) + kappa(C) >= 2k + 2 gives uniqueness up to S_k; Jennrich constructs it).
  Corollary for k = 3, m = 5 with explicit conditions (G1) private points not collinear, (G2) a pivot with distinct values and the
  other two overlap variables off the Segre lines. Strata: whole-overlap agreement not identifiable (explicit 3-parameter family;
  equal weights NOT needed, correcting the earlier draft); ranks 15 and 16 on two computed strata; open: cyclic sharing (rank 17, no
  pivot meets Kruskal). Code `engine/pce/three.py`, check `engine/check_three.py` TH1-TH7. Programme for chains/cycles of windows.
- Label holonomy (2026-10-30): covers Definition (nerve of identified nodes, profiles, transitions) and Theorem "label holonomy":
  (1) for a true k-type law every transition is sigma_V^-1 sigma_U, so the holonomy of every closed walk is the identity, on any cover;
  (2) conversely, trivial holonomy on a connected nerve (nodes sharing a variable adjacent) glues to a unique k-type law; (3) a twisted
  three-node example: three-type tables, equal on overlaps, gluing to an ordinary law (LP), holonomy a 3-cycle, so no three-type law.
  Descent class in non-abelian Cech cohomology with S_k coefficients. Checks TH8-TH10. Open: coincident profiles (double cosets),
  which covers have enough identified nodes, the O(2) pair layer, and whether the two-type holonomy equals the sign class.

- Frames, the O(2) layer (2026-10-30): covers Lemma "frames" (a - mu = tau u^T, whitened atoms; pairs = off-diagonal rank-2 Gram; at
  fixed frame the hidden set is the whitened weighted triangles, a circle of hidden laws by rotation) and Theorem "frame holonomy":
  (1) O(2) cocycle on rigid patches, trivial for true data, gluing, twists by any angle or a reflection invisible on overlaps;
  (2) at identified nodes the frame transition carries atoms by the S_3 transition (twisted example: rotation by 120 degrees);
  (3) rank one: holonomy = sign of c around a lift, so the two-type label holonomy is the sign class (under a generation hypothesis).
  Rigidity of off-diagonal rank-2 Gram data: K5, K5-e, K6 rigid; K4, W5, K3,3, K3,4, Petersen not; K5 globally rigid (computed).
  The bipartite scale is the rank-one flex, so no CO(2) enlargement. Code `engine/pce/frames.py`, check `engine/check_frames.py` FR1-FR9.
- Categorical reading (2026-10-30): covers Definition "realisations" (factorisations through a k-point object in FinStoch; groupoid;
  restriction; twisted realisations) and Proposition "the prestack of realisations": fibres are S_k-torsors; descent data exist but glue
  iff holonomy trivial, so realisations form a prestack, not a stack (cycles unfilled by triple overlaps); twisted realisations classified
  by H^1(nerve; S_k); the law layer separate (latent holonomy is not contextuality); the pair layer is the Gaussian shadow, O(2) =
  automorphisms of N(0, I_2), the factor-analysis rotation; complete graphs n >= 5 identified up to O(2) by Anderson-Rubin. Three-layer
  table (law / realisation / decision). Lab Markov section points to it. Text only; no new scripts.

- Phases IV-V (2026-10-30): covers Section "Phases IV-V" (`covers/sec_ident.tex`). Definition: (F) factorisation, (P) parameters (local,
  global up to labels), (L) law. Theorem "identifiability ladder": (F) torsor H^0_inc, finite iff no bipartite component; local (P) iff
  rank m + 1; global (P): if G_2 = {1, swap} no positivity condition, else relative explanations decided by positivity walls; (L) = (P)
  with three loading variables (Kruskal 2+2+2), strictly weaker below; the ladder is strict. Example: two triangles, triple on one only:
  wall s* = 2 w2 a23 / (1 - 2 w1). Wall table (support, rank, positivity, three types). Code `engine/pce/ident.py`, check
  `engine/check_ident.py` ID1-ID6. Lab: paragraph "Covers as test spaces" (Foulis-Randall; Staton-Uijlen; Gudder-Greechie).

- Large Betti tables (2026-10-31): Macaulay2's kernel and full resolutions of the two largest covers exceed 8 GB (killed three times).
  New route: the ideals are homogeneous for the v-exponent multigrading, so each graded piece is the null space of the r-polynomial
  coefficient matrix (exact lift, two primes). Generators: two tetrahedra 71 (v-degrees 4-7: 8, 34, 25, 4), five-window 159 (4-8: 10, 49,
  60, 30, 10). Resolving from these over Z/32003 reproduces the published tables and gives beta_2 = 481 for two tetrahedra; beta_>=3
  (two tetrahedra) and beta_>=2 (five-window) remain open for lack of memory. Term counts (binomial/trinomial) depend on the generating
  set, so the paper now states degree counts. Code `engine/betti_large/`, check `engine/check_betti_large.py` BG1-BG3.

- What does not matter (2026-10-31): lab Section "What does not matter: the decision-relevant quotient" (`lab/sec_relevance.tex`).
  Infeasible (positivity) vs irrelevant (decision). Prop: gauge invariance for observable acts (latent-indexed acts are not, and are
  undefined under label holonomy); only rho(F(O)) matters; relevant rank = rank(rho on ker K) (m = 4: kernel 5, two acts see 1, three 2,
  pair-event acts 0); decision-redundant windows by LP, decision-sufficient subcover. Prop (Vigneaux): V = R - X.R is a 1-cocycle; k-type
  families adapted but curved (Vigneaux uses subcomplexes); V[X] = 0 iff a common optimal act. Ambiguity: ex-ante maxmin = prior-by-prior
  updating on rectangular sets (Epstein-Schneider, RL7); the hiding segment is not rectangular (chi couples conditionals): updating loses
  value in 110/300, negative value of X in 24 (RL8). Code `engine/pce/relevance.py`, check `engine/check_relevance.py` RL1-RL8.

- Tangent scope audit and higher order (2026-10-31): tangent paper Section "Scope audit" (every claim with status and evidence; the
  categorical claims not made). Second-order theorem re-checked line by line. Proposition "order k": necessity of multilinear
  non-negativity on generator tuples of K_y (with vanishing lower sub-tuples) proved for every k; X1(X1 - X2)^2 passes order two but fails
  order three, so the second-order subcategory is not closed under T^3; sufficiency at order three computed (70 random cubic, 5 designed,
  140 quadratic maps). Conjecture "order k". Corollary 11's product-map justification corrected. Andre-Quillen naming confirmed (tetrads a
  regular sequence, Ext^1(L, kappa) = coker J). Code `engine/pce/order3.py`, check `engine/check_order3.py` O31-O33.
- Decision walls (2026-10-31): lab Proposition "wall-crossing for the value of a probe" (V[X] = min<l,P> - sum_x min<l 1_x, P>, kinks on
  decision and conditional walls, jumps = loss differences; RL9) and paragraph "Decision walls: a programme" (three arrangements; open:
  joint chamber complex, decision-identifiable region, robust wall-crossing). Lab's "we do not know one" updated to point at S_k descent.

- Inversion (2026-10-31): lab Subsection "Inversion" in Section 29. Three inversions distinguished: Bayesian inversion (added),
  covariant involutions (already present: label swap t -> 1/t, G_2, the flip c), additive inverses (not imposed; cones are groups only at
  relative-interior points). Prop "inversion": coupling swap, involutive a.s., reverses composition, partial (domain ends at a positivity
  wall), C_X built from it, latent posterior equivariant; defect sandwich 0 <= D_X(F) <= ante(F) - ante(Rect_X F) (proved; bound attained
  in 54/97). Prop "inversion, feasibility and decision walls": affine inside the data, linear-fractional in general, preserves
  one-sidedness, no pooling flips, pole 1/q(y). Citations: Cho-Jacobs 2019; Dahlqvist-Danos-Garnier-Kammar CONCUR 2016 (not Cho-Jacobs,
  as the reviewer's note had it). Code `engine/pce/inversion.py`, check `engine/check_inversion.py` IV1-IV10.

- Fourth companion, "Probability, Context and Differentiation: The Layers" (`context/layers.tex`), Step 1 (2026-10-31): finite
  distribution monad D; context comonad W = L x (-); distributive law lambda: W D => D W, (l, p) -> delta_l (x) p, with the four axioms
  written out by type and proved; the two-sided Kleisli category = context-indexed Markov kernels (one context held through a composite).
  Marginalisation total and affine; conditioning partial and nonlinear (domain p(A) > 0, total probability, iteration, commutes with
  marginalisation for cylinder events only, derivatives with |DC_A| ~ 1/p(A)). Contextual Bayes decomposition; forgetting context
  commutes with window marginals and is not injective (labelled windows see source contrasts). Code `engine/pce/context.py`, check
  `engine/check_context.py` DL1-DL11 (exact rational arithmetic). Steps 2-5 not begun (await review). Shared appendices now list four papers (Y).
- Layers Step 1 revised (2026-10-31): W renamed the provenance comonad (L = sources), separated from measurement contexts
  (Abramsky-Brandenburger); Section "Related work and what is new" (Markov categories, Fritz-Perrone bimonoidal structure, Jacobs
  effectuses, Cho-Jacobs inversion, Simpson probability sheaves, Uustalu-Vene environment comonads, Power-Watanabe, Brookes-Van Stone),
  with a candid statement that lambda and its two-sided Kleisli category are standard and the contribution is the exact
  non-commutation results. New Proposition "conditioning at the wall" (no continuous extension; FinStoch conditionals arbitrary there).
  Worked examples (two sources: 1/4, 3/4 -> 8/23, 15/23; 9/23 vs 63/160; square fails: 9/19 vs 23/40) and four figures
  (`context/fig_layers.py`). Checks DL12, DL13. 14 pp.


- Layers Step 2 (2026-10-31): the tangent layer. `engine/pce/tangent_layers.py`, `engine/check_tangent_layers.py` (TL1-TL9, 9/9, ~25 s).
  T commutes with W; the source face (every feasible direction = face direction + injection under other labels; injection and drift
  agree after forgetting); marginalisation and conditioning onto feasible cones with explicit lifts and kernels; Theorem: T^k H is the
  cone generated by elementary tensors, hence closed under products, so products of laws and pooling are morphisms at every order
  (proves the tangent companion's checked TG5 claim for products); differentiated contextual Bayes decomposition; worked example
  f(1/4) = 9/23, f' = -16/529, f'' = 384/12167, dpi^A = 640/529. Figures fig_face, fig_sens. Step 3 not begun (awaits review).
  Proposed next (user, 2026-10-31): stratified conditioning-provenance operator, support theorem, regimes of jumps.
- Layers Step 2, Section "Strata and jumps" (2026-10-31, user request): `engine/pce/jumps.py`, `engine/check_jumps.py` (SJ1-SJ6, 2 s).
  Theorem (provenance support: surviving sources = {pi_l p_l(A) > 0}, odds, strata S_J); Proposition (four questions: weights preserved iff
  equal p_l(A); old-weight pooling commutes iff sum (pi^A - pi) C_A p_l = 0; conditioning maps forgetting-fibres onto fibres, dims
  (n-1)|supp P| = on-A + off-A; provenance unrecoverable from C_A P; identifiability can be destroyed, never created); Theorem (regimes:
  rate <= 2/P(A); extinction continuous; wall limit = conditional of the leading jet; analytic passage removable; jumps need kink,
  non-analytic path or jump of the law); Example gradual-then-sudden (width 4 eta, slope -1/(4 eta)); fig_jumps. Open: hybrid dynamics with
  transition operators; beyond finite sets.
- Layers Step 3 (2026-10-31, user request "order of a probability transition"): `engine/pce/orders.py`, `engine/check_orders.py` (OR1-OR7).
  Theorem (orders tropicalise: v(f+g) = min, v(fg) = sum on non-negative Hardy germs); Corollary (order calculus table for labelling,
  pooling, marginalisation, conditioning, Bayes weights); Theorem (hierarchy: common order, unequal order with extinction order m_l - m,
  one-sided mismatch, flat; limit exists in a Hardy field; C-infinity flat path can jump); Proposition (k-jet / T^k sees the transition iff
  k >= m; Weil coefficients |S|! c_|S|); Example (order-m near miss, width 2(2 eta)^(1/m)). Figures fig_orders, architecture updated.
  Graded classes of maps moved to Step 4. Refs Hardy 1910, Rosenlicht 1983 (Hardy fields).
- Layers Step 4a-4b (2026-10-31): `engine/pce/fans.py`, `engine/check_fans.py` (MP1-MP7, ~2 min). Proposition (arc limits = normalised
  initial forms); Theorem (order fan = normal fan of the Newton polyhedron, limit set cone by cone, componentwise minimum sufficient,
  iff for monomial cells, degenerate directions; curve selection lemma, Milnor 1968); Theorem (iterated limits = lexicographic vertices;
  corners can agree while the middle differs, x^3+y^3 vs xy); Proposition (multilinear Weil points in T^d; the lexicographic lemma computes
  one corner, the flip the other). Examples: race, dominance, curved path, degenerate direction, two sources in the tower. fig_fan.
  Next: 4c decisions on the fan; 4d graded classes and contextual second-order theorem (cross term = e1e2 coefficient).
- Layers Step 4c (2026-10-31, user: "ambient category to record parameters that change context"): `engine/pce/para.py`, `engine/check_para.py`
  (PR1-PR6). Definition (Para_L: parameterised source-indexed kernels, 2-cells = reparametrisations; Fong-Spivak-Tuyeras 2019, Cruttwell et
  al. 2022); Proposition (2-category; layer operations commute with reparametrisation; conditioning domain pulls back); germ-field reading
  (real Puiseux series: valuation = trop, standard part = arc limit, nested fields = corners); Theorem (decision fan: eventual acts, ties broken
  at higher order, walls in c on edge cones, transport along monomial reparametrisations w = R w'); Examples (audit decision in the race: tie at
  c = 1 broken at order 3; two study designs t = s u and (uv, u) fill one cone each; t = s(1+u) audits source 1). fig_para. Next: 4d.
- Layers Step 4d (2026-10-31): `engine/pce/statfan.py`, `engine/check_statfan.py` (ST1-ST7, ~1 min, Monte Carlo with fixed seeds).
  Definition (decorated order fan); Proposition (decoration = sheaf on faces; edge limit maps join vertex limits; same N different limits);
  Proposition (exact cross term, whole curvature of pooling; conditioned mixed derivative); Theorem (J as inequality of orders on the
  statistical fan s = N^-1/2: v(a) = w_D + w_pi/2 - 1, v(I) = w_pi + w_D - 1, band B at (0,1) only, vanishing share, no tilt);
  Proposition (Gaussian cone w_A < 2, Poisson ray w_A = 2, empirical wall w_A > 2; J needs w_pi < 2). fig_stat. Open: Poisson-ray band.
  Next: 4e graded classes.
- Layers Step 4d revisions + Poisson band + Step 4e (2026-10-31): terminology "probability-decorated order fan", O = facewise presheaf of
  initial forms (remark: Alexandrov/cellular sheaf justification, Curry 2014, not used); statistical assumptions (S1)-(S5) stated; Theorem
  (exact band on the Poisson ray, conditional binomial test: k(T+1) <= k(T)+1, monotone exposure, Gaussian limit = J's B, blind spot
  t_b = ceil(log alpha / log pi) - 1, wall side Delta_max -> lambda*, adversary owns the event); Step 4e Theorem (graded classes M_k:
  descending chain of subcategories, strict at 2 and 3; subtraction-free maps in M_infinity; every layer operation incl. conditioning in
  M_infinity -- closes "conditioning beyond order two"). Checks `engine/check_graded.py` PB1-PB5, GC1-GC6. fig_graded.
  Next (user: "eventually"): recursive Newton-Puiseux theorem for degenerate initial forms, with termination bound.
- Layers Step 4f (2026-11-01): `engine/pce/degen.py`, `engine/check_degen.py` (DG1-DG6, exact; sympy for edge-polynomial roots).
  Theorem (recursive resolution, two parameters): refine iff the edge polynomial of P(A) = sum of cells vanishes (non-negative cells cannot
  cancel; partially vanishing cells drop out); substitution x = t^q, y = t^p (c* + y1); Newton-Puiseux for P(A) (Walker 1950, Casas-Alvero 2000):
  multiplicities non-increasing, infinite chains = exact common real roots (inside the wall), finite tree, limit set node by node.
  Examples: (y - x - x^2)^2 vs x^5 (two refinements, share b^2/(1+b^2) on y = x + x^2 + b x^(5/2)); (y^2 - x^2 - x^3)^2 vs x^K (chain of length
  (K-2)/2 - 1 following x sqrt(1+x); root coefficient = zero of the limit map). Also corrected Prop 35 / Thm 36(5) wording (refine only when
  ALL leading cells vanish). fig_degen. Next: Step 5 (functions, duality, log/exp).
- Layers Step 5 (2026-11-01): `engine/pce/effects.py` (NOT pce/duality.py, which is the older lab module and was briefly overwritten
  then restored from the mirror; check_tangent and check_injection re-verified 14/14), `engine/check_effects.py` (FD1-FD9).
  Variance table; Proposition (conditioning/Bayes weights/Bayesian inverse as updates by effects; C_r C_q = C_qr; Para naturality; L2
  presheaf = effect side of marginalisation); Theorem (Maslov bound; Bayes exact in logs; conditioning = tropical division; order fan =
  normal fan / tropical hypersurface of P(A) in log coordinates; m/e split; subtraction = where logs fail); Proposition (cotangent cone,
  contravariance, robust ties, decision walls = positive roots of signed N_ab); Proposition (exclusion principles: annihilator identity
  rank(rho|ker K) = dim span rho - dim(span rho cap im K^T), conditioning, wall, feasibility, statistics). Refs Jacobs 2015, Maclagan-Sturmfels 2015.
- Layers: fractures and blind spots (2026-11-01): `engine/pce/fracture.py`, `engine/check_fracture.py` (FX1-FX6). Definition (tower, fracture,
  order, blind spot); Proposition (catalogue of six fracture kinds; blind spots = exclusion principles; forgetting is a blind spot);
  Theorem (poison order vs tie order: flip iff v(E) = kp + r < m with opposite sign, or equal orders; hidden poisons k >= 2 need a contrast);
  Example (covariance tie, first-derivative-free poison, flip iff 2p < m or 2p = m and e^2/16 > c); cross-layer table. Also: Prop 8(5)
  typeset with a(h) first (reviewer's extraction lost it). Proposed next: LLM hallucination study (see conversation).
- Layers §24 revised (2026-11-01): outcome-relative fractures (coherence, definedness, act, VoI sign, identifiability), order, fragility,
  reach, observational fractures; Theorem (fragility exponent delta^(1/k); ties with gap in C_x^* need k >= 2; linear ones unbreakable);
  Proposition (reach on the race ray; observational fracture of the audit act); Theorem (decision bands near walls: Gaussian, N s^(m+2q),
  Poisson ray exact, decoration dependence). Checks `engine/check_fracband.py` FB1-FB5.
- Study 15 PRE-REGISTERED (2026-11-01), NOT RUN: `llm_fracture_study/PREREG_llm_fracture.md`, `kit_icl_poison.py` (items / run / score /
  smoke), frozen in FROZEN_PREREG_study15.txt (SHA-256). In-context poisoning, synthetic triggers, Qwen2.5-1.5B-Instruct primary,
  SmolLM2-1.7B-Instruct secondary; H1 count vs share (Delta_common < 0, Delta_rare - Delta_common > 0), H2 owned rare events. The author runs
  `python kit_icl_poison.py items --out prompts.jsonl`, then `run`, then `score` locally.
- Layers: exclusion filtration, fibre homology and a computational blueprint (2026-11-01): `engine/pce/blueprint.py` (new module),
  `engine/check_blueprint.py` (BP1-BP8, 8/8), figures fig_blueprint_a/b in `context/fig_layers.py`. Definition (exclusion filtration);
  Proposition (graded pieces, relevant ranks drop one level at a time, convex exclusions contractible); Proposition (fibre homology:
  gauge-invariant decisions ignore H0, latent-indexed acts undetermined across components, identifiability fractures = Betti jumps
  forced by Jacobian rank drop, contextuality needs context-dependent provenance). Blueprint: two-type source vs stale counter-monotone
  source with matching margins; wall theta*(kappa) from four Boole inequalities (= LP on 1353 points), theta*(1) = 43/54, absent for
  kappa < ~0.55; filtration ranks 4,7,14,16 resolving E[X0], E[X0X1], E[l X0X1], parity; fibre b0 2 -> 1 at kappa = 0 (Jacobian rank
  7 -> 3, sigma_min ~ kappa^3); flag event P(A) = s^3 + 7/4 s^2 t + ts + t^3 (fan rays slopes 1/2, 2; s^2 t owns no cone),
  N_ab = s^3 - 7/4 s^2 t - t^3 (one ray, wall c = 1/2), refinement 4 cones, tie order m > 0 in the middle cones; relabelling the
  uninformative cell flips the act there only.
- Study 15 Deviation 1 (2026-11-01): kit passes attention_mask = ones (warning from transformers when pad = eos); outputs unchanged;
  v1.1 hash 3098ad6f... in FROZEN_PREREG_study15.txt.
- Layers readability (2026-11-01): one-page "In plain language" before Section 1 (object, what goes wrong, the tool, the payoff) and a
  "Reading paths" box (everyone, statistician, ML reader, category theorist, geometer) with \cref links that track renumbering.
  Checks table converted to longtable (it overran a page).
- Layers: "Provenance of results" section (§3, 2026-11-01): colour-coded longtable (Used grey / Adapted blue / Derived amber / New green),
  labels audited against proofs; only three results labelled New (Thm tensor, Thm graded (3)-(4), Thm Poisson band). Dependency diagram
  (TikZ, Fig 2). New definitions and "not used" listed. Eleven new bib entries (Litvinov, Sturmfels, Rockafellar, Billingsley, van der
  Vaart, Le Cam, Przyborowski-Wilenski, Lehmann-Romano, Rudin, Boole, Pitowsky). Reading paths point to it.
- Layers bands (2026-11-02): §27.6 "How much data the decision needs" (fig_blueprint_c, BP9). Exact Skellam P(wrong act); N* ~ tau^-e with
  e = 2 v(N) - v_dec per cone (outer e = v(P); middle e = v(P) + m; wall ray at root 3 + 2r). Thm 72(3) CORRECTED: added spread order rho
  (sigma^2 ~ s^rho); learned iff N s^(m+2q-rho) -> infinity; old statement assumed sigma_infty > 0, which fails in the middle cones.
  Plain-language "sharper reading" wording replaced. Reviewer T1-T6 assessed in conversation: T1 (fan poison-vs-tie) and T3 (fracture
  loci as walls of one stratification) next; T5 (d > 2 toric) separate; T2 folds into T1; T4 declined; T6 subsumed by the band subsection.
- Layers Step 4f, several parameters (2026-11-02): Theorem "several parameters" (thm:degend): (1) per-arc limits in any d, determined by
  the arc truncated at the order of P(A) along it; (2) the limit set is compact semialgebraic with finitely many components (Tarski-
  Seidenberg + curve selection; Bochnak-Coste-Roy); (3) Newton non-degenerate P(A) (no positive zeros of any in_F P): limit set = union of
  normalised initial forms over bounded faces, toric charts (Khovanskii, Varchenko); (4) degenerate d > 2: explicit tree OPEN (Hironaka
  cited, not used). Example (x-y)^2 z vs x^2 y^2. Module engine/pce/degen_d.py, checks engine/check_degen_d.py DD1-DD5, fig_degen_d.
  Reviewer's T5 sketch assumed positive coefficients (which makes degeneracy impossible); corrected to signed non-negative cells.
- Layers cone-wise pass (2026-11-02): Lemma "the spread order" (rho from fan data: 0 iff leading cells carry two gap values; otherwise
  min over cells with a different gap; on walls rho = v_dec - v(P)) -- the reviewer's "rho = 0 iff a least-order cell has non-zero gap" is
  FALSE (outer cones: single leading cell, sigma_infty = 0). Corollary "the sample law is a fan quantity": learned iff N t^(2v(N)-v_dec)
  -> infinity, in every cone and on every wall. Thm 74(2) gets the sigma_infty = 0 sentence. Theorem "poison order against tie order on
  the fan" (T1 + T2 folded in: fragility exponent p*(w) = trop G - trop H, piecewise linear on the common refinement; in the blueprint
  p* = tie-order map m(w)). Small edits: Remark (Study 15) "registered prediction, not evidence"; catalogue sentence tying gluing failures
  to absent global sections and identifiability to b0 jumps. Module engine/pce/conewise.py, checks engine/check_conewise.py CW1-CW6.
  Next: T3 in stratification form.
- Layers Step 4f, principalisation (2026-11-02): Theorem 63(5) (Derived): principalising the CELL IDEAL (Hironaka; Bierstone-Milman;
  Kollar) gives f_c o pi = y^alpha u_c, u_c without common zero; limit set = continuous image of E+ (positive real part of the
  exceptional fibre = closure of the lifted open orthant intersected with pi^-1(0)) under u / sum u; both inclusions proved (approachability
  from the closure definition; the reverse from properness). (4) sharpened: Newton-guided refinement with explicit charts and a bound in
  Newton data is open. Remark: Thm 59's step is a weighted blow-up plus translation. Examples: isolated direction (y-x)^2 + (z-x)^2 vs x^3
  (one weighted blow-up (2,1,1)); Example 63 in charts (blow-up of the z-axis, then four unimodular toric charts x both signs of u,
  respecting the wall 2w_u + w_z = 2w_x). Checks DD6-DD7 (sympy pullbacks, residual generators, 311 lifted arcs vs exact limits).
  Next: T3 (stratification form).
- Layers T3, fracture strata (2026-11-02): new section "Fracture strata: one stratification for all walls". Proposition (fracture strata:
  definedness / act / value of information / coherence / identifiability are invariants constant on one finite semialgebraic
  stratification; Tarski-Seidenberg, Hardt triviality; loci closed, dim < d, unions of strata; reach constant on strata; eventual values
  along arcs read from the fans; kinds (iii)-(v) are not walls in parameter space). Corollary (the catalogue as one stratification --
  coherence = absence of a global section with a separating certificate, NOT nonzero H^0; VoI never negative, so its fracture is 0 ->
  positive). Corollary (global fragility function eps*: semialgebraic, 1-Lipschitz, zero exactly on the locus, rational exponent along
  arcs; the exponent, not the value, is what the fans control). Example (blueprint: 8 strata; corner has reach 3; eps*_act = |2t-s|/sqrt5,
  exponent min(1,w)). Remark: Thm 63(3) is the toric case of (5) (cell ideal principal near E+). Module engine/pce/fracstrata.py, checks
  engine/check_fracstrata.py FS1-FS5, fig_strata. Session note: container was replaced; work resumed from tower_geometry_github19.zip.
- Layers G1 + E1 (2026-11-02). Step 4g "geometric realization, the toric picture" (Theorem "the toric realization": non-negative
  coefficients -> monomial law = point of the toric model; Birch: moment map x grad log P onto relint conv(S); non-negative toric variety
  homeomorphic to Q; arc limits = face laws on lower faces; limit set = R mu^-1(lower boundary); order fan = positive part of the normal fan;
  signed coefficients = where it stops). Remark (outlook, symplectic: conditioned states <-> moment fibres, NOT sources <-> Lagrangians;
  Birch vs Floer critical points as an open question). Module engine/pce/toricreal.py + engine/check_toricreal.py (TR1-TR5); NOTE the
  check file appeared on disk before this turn's code; pce/toricreal.py was (re)written to its API, TR3 strengthened to certify LP verdicts.
  Section "An observed environment" (Theorem "contextual Bayes with an environment": forgetting square; observed env not reweighted,
  forgotten env reweighted; Simpson reversal of a source comparison as a wall, impossible with env-independent source mix; pooled order
  v_l = min_e (r_e + m_le) (tropical sum); transport: a rare environment's explanation is lost). Module engine/pce/environment.py +
  engine/check_environment.py (EN1-EN5). Figures fig_toric. Next options: E2 (structured environments: real comonad + iterated laws) only
  if needed; G2 (symplectic note) needs a computed example first.
- Layers E2, structured environments (2026-11-02): section "Structured environments". Theorem (hierarchies: coarsening = morphism of
  comonads and distributive laws, Yang-Baxter for exchanges (Cheng 2011); nested Bayes; nested orders = associativity of the tropical sum;
  pooling never creates an explanation (survivors at a node lie among a child's); reversals not monotone in the hierarchy, explicit
  four-site example 0.49 vs 0.51 in a region, 97/180 vs 69/220 at the top). Proposition (store comonad X^E x E with the independent
  coupling: unit, multiplication, counit hold; comultiplication FAILS, copies agree w.p. 1 vs sum P(g)^|E|; existence of another law
  not settled; modelling consequence: dependence between sites goes into a law on configurations, the comonad stays a product).
  Module engine/pce/envhier.py, checks engine/check_envhier.py EH1-EH6.
- Layers clarity pass (2026-11-02): abstract cut to one page (the old step-by-step list moved to the introduction as "Results by step",
  extended with Step 4g, environments, strata, blueprint; band statement updated to N s^(m+2q-rho)); subtitle no longer "Steps 1-5";
  roadmap caption and plain-language payoffs mention environments; reading paths: statistician -> environment section. Theorem
  (environment)(2): observed vs forgotten regimes stated separately; (3): reversal wall as signed polynomial R (check EN6). Hierarchy
  theorem: "explanation" = least-order source; one-level case = environment theorem; typed comonad/distributive-law squares displayed.
  Store proposition: typed comultiplication square, "rules out the independent coupling only", sum_g P(g)^2 for two sites. Toric
  section: signed-coefficient expansion of (x-y)^2 z with weights (2,-4,2,1). Terminology: source = L, environment = E, window = cover.
  Not done (optional): crossing-planes degenerate example; Example 9 recurrence; second kink family.
- Layers: example "why the construction in more than two parameters is open" (2026-11-02), after the toric-case remark in Step 4f.
  Crossing planes ((x-y)(x-z))^2 vs x^5: multiplicity 2 on planes, 4 at the crossing; one critical exponent on a plane, a line
  e1 + e2 = 20 of them at the crossing; one blow-up gives x^4 (u^2 v^2, x), normal crossings. Tangent curves of contact order k need
  exactly k - 1 point blow-ups (k = 1..4: 0, 1, 2, 3). Absolute-value coefficients keep the Newton polyhedron and remove all degeneracy:
  centre and step count are coefficient data. Checks DD8-DD10. Abstract count now 139 checks.
- G2 companion note (2026-11-02): kahler/conditioned_states_kahler.tex (6 pp). Tier A (classical): Fisher = Kahler metric in log coords,
  symplectic potential = relative entropy, DH = Lebesgue on Q, positive real locus = Lagrangian section. Lemma (balance = Birch at 0).
  Theorem (Birch bridge, Derived): one positive critical point of W_T on a compact Delzant polytope, mean-zero law; its valuation has
  balanced leading terms; converse fails (F_1 alpha > 1/3: a segment of candidates, higher order selects); non-positive critical points
  balance only with signed weights. Floer reading cited (Cho-Oh, Fano; FOOO general): the positive critical point's fibre has HF != 0.
  Examples CP^2 and F_1 (F_1 IS Fano; F_n, n >= 2, is not). Module engine/pce/kahlerbridge.py, checks engine/check_kahlerbridge.py KB1-KB6.
  NOTE: engine/pce/torickahler.py appeared on disk at 18:39 from another writer (not this session); left untouched and not imported.
  Layers Remark (outlook) now points to the note.
- Companion note round 2 (2026-11-03): kahler/conditioned_states_kahler.tex (10 pp). Reviewer fixes: entropy = the unique minimiser of
  D(q||c) with mean p (exponential-family law); compact extension only on the open orbit in general, Fubini-Study restriction when S = lattice
  points of a Delzant lattice polytope (ample => very ample); strict positivity of limit weights on K(u*) and affine spanning; Thm 3(4) = signed or
  complex linear dependence, not a law; corollary: one fibre, not the only one; F_1 segment endpoints excluded (leading sets {v1,v2,v4},
  {v2,v3,v4}); checks corroborate, not prove; alpha values listed; abstract leads with candidates versus selection.
  Provenance: Thm 3(1) = Galkin's lemma (arXiv:1404.7388), Used; u* maximises Trop W = Judd-Rietsch Thm 1.2 (arXiv:1911.04463), Used;
  GGI (Duke 2016), Iritani (Adv. Math. 2009) cited for the positive thimble. New section 5: Prop 7 (orthant = ascending thimble J_+, Hessian
  diag(H,-H), probability law, Laplace), Thm 8 (tropical limit, hbar = T^h, u = log y/log T: uniform on Q_h for h < h*, = DH at h = 0;
  concentration on argmax min l for h >= h*; argmax = {0 in conv v_K(u)} contains candidates and u*), Example 9 (F_1 levels: uniform on the
  segment at h*, uniform on [h, (1+alpha)/2 - h] up to h2 = (1+alpha)/4, then at u*; sketch). Outlook adds signed-branch Stokes data (open).
  New functions in kahlerbridge.py: maxmin, tropical_density, laplace_moments, tropical_grid, argmax_set, polygon_moments. Checks KB7, KB8;
  KB5 gains the endpoints. Figure gains panel (c), the u1-marginals of the thimble law. Layers paper unchanged.
- D1 (2026-11-03): companion note section 6 "Soft evidence moves lambda" (note now 13 pp). Convention: T = e^-N, evidence e^(-n rho_k) on term k
  gives lam_eff = lam - (n/N) rho (sign fixed per review). Prop 11: h*(lam) = -max_{q in B} <q, lam>, B = laws on facets with mean exponent 0
  (LP duality); candidates = relative interior of argmax, endpoints are not; dim argmax = n - dim sigma_0; the positive limit law lies in the
  optimal face. Cor 12: outcome walls = normal fan of B; translations and constants never cross; only m - n - 1 circuit directions matter; one
  circuit: s* = -c/<a, rho>, least evidence |c| / (||a||_1 / 2). CP^2: no circuit, no wall. F_1: circuit (2,-1,2,-3), wall alpha/L = 1/3 inside
  the Kahler cone; u* continuous, leading set / candidate dimension / selection mechanism jump; wall law from x^4 = x + 1 (T-independent,
  interior of B); discriminant -27T^2 - 256T^(3-3alpha) never vanishes (tropical-discriminant wall); from alpha = 1/5, rho = e1 reaches the
  wall at s* = 1/5 (n* = N/5), least evidence 1/10 at rho = (1,0,1,0). Decision, contextuality and VOI walls deliberately NOT computed (need
  their own definitions). New: engine/pce/datawalls.py, engine/check_datawalls.py (DW1-DW7), kahler/fig_datawalls.py. Layers unchanged.
- Part II + ledger (2026-11-03): the companion note is now one note in two parts (17-18 pp): Part I geometry, Part II "Back to contingency
  tables". User preferred a single note (the reviewer suggested a separate one); Part II is self-contained so it can be split later.
  Prop 15: two configurations of one log-linear model: cells ((r-1)(c-1) basic moves; Tier A) vs facets (exactly one circuit
  (c 1_r, -r 1_c)/gcd; W). The reviewer's (r-1)(c-1) count is the cell side; both are right. Prop 16: a product adds exactly one circuit (relative
  scale). Thm 17 (independence): h* = min(L1/r, L2/c), wall c L1 = r L2, rc critical points all over uniform margins for every L1, L2 (selected
  point never moves), so by Cho-Oh the wall changes no Floer data (contrast F_1). Prop 18: margins b pick the Birch fibre; evidence moves lam;
  both in moment coordinates; Fisher-Rao distance sqrt(n1 d_row^2 + n2 d_col^2); centrality = smallest margin / (n / max(r,c)). Ex 19: 2x2 on the
  wall for every n (joint sampling); 2x3 needs n/2 more column observations. Ex 20: a structural zero in 2x2 leaves a triangle that is not a
  face; no circuit (CP^2). Facet <-> MLE non-existence is claimed only for independence (EFRS 2006, Fienberg-Rinaldo 2012 cited for general).
  LEDGER.yaml (9 rows: soft-translation, soft-constant, soft-circuit, margins, refinement, structural-zero, independent-product, forgetting,
  coupling) with assumptions, parameter_map, levels, walls, wall_certificate, checks; statuses Proved/Used/Cited/Exact/Numerical/Defined/Open/N/A;
  category level never filled. Appendix A is generated from it. New: engine/pce/contingency_toric.py, engine/check_contingency_toric.py
  (CT1-CT7), engine/check_ledger.py (LG1-LG3), kahler/render_ledger.py, kahler/ledger_table.tex. Layers unchanged.
- Section 11 (2026-11-03), "One coupling polytope through every layer" (note now 25 pp), motivated by the architecture note and the
  coupling-polytopes paper. Example f = (2/5, 3/5), g = (3/10, 43/100, 27/100) (architecture note section 6). Base/fibre duality: the margins are a
  point of the base and the Kahler class of the fibre (corrects the framing of Prop 18). Prop 22: Tier A on the fibre = Fisher's noncentral
  hypergeometric family (weights 1/prod t!, u = log odds ratios); u = 0 gives exactly N f(x)g; leximin needs odds ratios (0.82, 0.46).
  Thm 23 (New): h* = min(min f_i/m, min g_j/n) for nondegenerate n x m margins (proof: row averages + h J plus product of residuals; binding
  cells are facets since n(m-1) < m is impossible); leading cells = binding line; candidate dimension (n-2)(m-1) / (n-1)(m-2); (4) positive
  valuation = leximin coupling checked numerically on 20 random 2x3 margins (not proved). Ex 24: five critical points by exact resultant: 3 over
  leximin (one positive), real (-,-) over (13/100,13/100), real (+,-) over (1/5,1/10): matches Bridges Thm 8.2. Thimble: uniform on couplings
  -> uniform on Q_h -> leximin. Walls: Kahler-chamber walls f(I) = g(J) vs leximin walls (ratios tie): different. Ex 25: one count from column 3
  to 2 crosses a leximin wall (1/600 in ratio), column 3 binds, candidates a segment, distinct valuations 3 -> 2; blow-up wall f1 = g2 three counts
  away. Four new figures (fig_cpl_glance/tierA/thimble/walls) and a plain-language box. Ledger row margins-on-the-fibre. New:
  engine/pce/couplingfibre.py, engine/check_couplingfibre.py (CF1-CF8), kahler/fig_coupling.py. Layers unchanged.
- Section 11.6 (2026-11-03): Def 26 (lexicographic max-min), Thm 27 (selection, Adapted: FOOO procedure / JR recursion / Cominetti-San
  Martin; proof here): for v positively spanning and coefficients in [c, C], the minimiser u_L of sum a_k e^(-L l_k) satisfies |u_L - u_lex| = O(1/L).
  Proof: level one pinned to O(1/L) via F_L(u_L) <= mC e^(-L h1) and the balanced law on S1; projecting grad F_L = 0 onto D1 removes S1 exactly,
  so p_L minimises the reduced potential on E1 with coefficients a_k e^(-L<v_k, w_L>) in a compact set; induction on dimension. Cor 29: Theorem
  23(4) is now a theorem: the positive critical point of Cpl(f, g) sits over the facet-leximin coupling; facet-leximin = all-cell leximin in
  292 random margins (computed, not proved). Ex 30 (repurposed coupling-polytopes Ex 3.4(b), f = (2/5,3/5), g = (1/2,3/10,1/5)): column 3 binds;
  level 1 h = 1/10 (cells 13, 23), candidates the segment x + y = 3/10, 1/10 < x < 1/5; level 2 h = 3/20 (cells 11, 12, 22), u* = (3/20, 3/20);
  finite-T correction x_L = 3/20 + ln2/(2L) (cells 11 and 22 coincide on the segment, against 12); five critical points: positive and (-,-) at
  the leximin, three drifting along the segment toward about (0.133, 0.167) (not analysed; Outlook). One-count example is column-binding too.
  Figure fig_leximin (4 panels). New: engine/check_leximin.py (LX1-LX9), kahler/fig_leximin.py; datawalls.py gains lex_maxmin, softmin_point.
- Part III / section 12 (2026-11-03), "Objects, morphisms and a categorical comparison" (note now 32 pp; abstract rewritten to fit one page;
  checks table now a longtable). Objects = critical points of W read (Cho-Oh) as torus fibres with rank-one local systems, leading holonomy
  y_c / T^(u_c): F_1 alpha = 1/5: barycentre {pos (1,1), (z,z), (z^2,z^2)} + (1/5,3/5) {real (-1,1)}; alpha = 3/5: all four over (2/5,1/5) with
  (1,1), (-1,1), (+-i,-1); 2x3: 3 + 1 + 1 -> 4 + 1 after one count. Prop 31 (Cited: Cho 2005 Thm 5.6 Clifford ring with Hessian; Smith 2018
  eigenvalue summands, generation conditional for non-monotone): Hom = 0 between distinct objects; End = Cl(Q); positive object
  Q = W Cov_w(v) (Fisher information). Prop 32 (Derived on cited m_2): Cl(Q) separable -> HH^{>0} = 0 -> intrinsically formal (Kadeishvili,
  Seidel-Thomas). Thm 33 (New, proved): Q/W of the positive object reduces mod T to Cov_q*(v), rank dim sigma_0 = n - dim candidates; F_1
  2,2,1 across the wall (small eigenvalue ~ T^((3 alpha - 1)/4) = T^(h2 - h1)); 2x3 one count 2 -> 1 (1/200); column example 1 (1/20).
  Thm 34 (Derived): across the F_1 wall the semisimple category over Lambda is unchanged up to A-infinity equivalence (discriminant != 0);
  no continuation map in the strict sense (the Kahler class moves). Prop 35 (computed + sketch): Stokes walls for theta = 0.3, 1, pi/2 lie
  within 4/log(1/T) of alpha = 1/3, rate 1/log(1/T); none outside; the positive thimble (real hbar) never mutates. Summary table "what sees the
  wall". Outlook adds "the integral category". New: engine/pce/floermorph.py, engine/check_floermorph.py (MO1-MO9), kahler/fig_morph.py.
- Section 12.7 (2026-11-03), column-binding example classified (Stages I-III; Stage IV wall-crossing deferred). tau = T^(1/10). Prop 36
  (exact, sympy): y1 dW1 - y2 dW2 = y1 - y2 + tau^3/y2 -> y1 = s - tau^3/s; quintic q = s^5 + tau^2 s^4 - 2 tau^3 s^3 - 2 tau^5 s^2 + tau^8; Newton
  polygon (0,8)-(3,3)-(5,0): three roots s ~ 2^(-1/3) omega tau^(5/3) (fibre (2/15,1/6)), two s ~ +-sqrt2 tau^(3/2) (leximin (3/20,3/20));
  disc_s q = -tau^31 (256 tau^2 - 837 tau + 3456), Hessian resultant tau^54 (same quadratic, negative discriminant): distinct and nondegenerate
  for all T > 0; W = 2s + tau^4/(s^2 - tau^3) + (s^2 - tau^3)/tau^2; critical-value discriminant -tau^225 (tau-1)^2 (quadratic)^3: distinct on
  (0,1). The "drift" was -log|coef|/log(1/T) with coef moduli 2^(+-1/3); the ln2/(2L) of Ex 30 is log(sqrt2/2). Floer convention: FOOO
  Lambda_0 bounding cochains (non-unit leading moduli allowed); Cho's Clifford formula used in that convention (flagged as the one stretch).
  Prop 31(2) now grounded on distinct m_0 (Auroux 2007, FOOO), Smith only for generation. Thm 38 (reduction of the Clifford form): with H1
  (sum_K rho != 0) Q/W -> M_c = sum_K rho v v^T / sum rho; rank <= span, = span under H2 (one-sign real); tested on 23 objects (H2 for 13),
  rank = span in all 23. Column example: M_c = J (rank 1) for all five; P, N: small eigenvalue +-(sqrt2/2) tau^(1/2) (P pos. def., N indefinite);
  R, C: first-order correction vanishes in the small direction; small eigenvalue -(3/2) 2^(-4/3) omega^-2 tau^(2/3). Prop 39: Hom table
  (0 off-diagonal: disjoint fibres / distinct m_0; End = Cl(Q_c) = M_2). New: engine/pce/colbind.py, engine/check_colbind.py (CB1-CB8),
  kahler/fig_colbind.py (fig_cb_newton, fig_cb_objects, fig_cb_hom). Note 36 pp.
- Part IV, section 13 (2026-11-03), 2 x 4 couplings: higher selection levels. Coordinates u = (a11, a12, a13); W over facets is
  sum_j ([1j] y_j + [2j] T^(g_j)/y_j) + [14] T^(f1)/P + [24] T^(g4-f1) P, P = y1 y2 y3. Prop 40: ampleness by curves (Nakai-Moishezon via Kleiman 1966;
  toric Kleiman CLS Thm 6.3.13; wall relation v_k + v_l + a v_i + b v_j = 0, -K.C = 2 + a + b; Kahler degree = lattice length, checked on all 2208
  edges). Thm 41 (exact computation): 148 Kahler chambers (sign vectors of f1 - g(J)), all smooth; CP^3 x2, Bl_pt CP^3 x8, (CP^1)^3 x4 (Fano),
  weak Fano with 6/7/8 facets x24/64/46 (1/3/6 curves of -K.C = 0; (-K)^3 = 48/40/32); vol conv(v) = #vertices = chi in all. Completeness of
  the chamber list rests on LP infeasibility; each chamber has an exact rational interior point. Prop 42: box iff some g_j > max(f1, f2);
  CP^3 chambers select at once (3); Bl_pt CP^3 chambers (3) or (1,2); so coupled three-step selection only in weak Fano chambers. Examples:
  A f = (2/5,3/5), g = (1/20,13/20,1/10,1/5) (cube; 8 points over (1/40,9/40,1/20); one critical-value coincidence at x^3 + x - 1 = 0,
  x = T^(1/40)); B f = (3/20,17/20), g = (1/20,1/5,1/4,1/2) (Bl_pt CP^3; c^6 + tau^2 c^4 - tau^5, tau = T^(1/20); 6 points over (1/40,1/24,1/24),
  holonomies (zeta^-3, zeta, zeta); value discriminant 2^36 tau^19 (4 tau - 27)^6 (4 tau + 1)^2); C f = (3/10,7/10), g = (1/10,3/20,1/4,1/2)
  (7 facets, weak Fano, levels 1/20, 3/40, 7/80, 13/80, drops (1,1,1,0); degree-10 resultant; 8 points over the leximin fibre, 2 over
  (1/40,1/40,9/40) with rho = (i,i,i,-i): H2 fails, rank 3 = span). Remark 46: semi-Fano corrections (Chan-Lau-Leung-Tseng PAMQ 2020 Prop 2.15)
  rescale coefficients by (1 + delta_k); levels and selection unchanged (if delta_k converge); Floer reading of C withheld. Def 47: scale
  valuations = invariant-factor valuations (determinantal divisors), = eigenvalue valuations for the positive object. Conjecture 48: scale
  valuations 0 (x r_1) and h_j - h_1 (x r_j). Prop 49: holds at leading order in A (0,1/40,3/40), B (0,1/60,1/60), C (0,1/40,3/80).
  New: engine/pce/cpl24.py, engine/check_cpl24.py (TF1-TF9), kahler/fig_cpl24.py (fig_tf_polytopes, fig_tf_spectrum, fig_tf_objects).
  LEDGER row deeper-selection. Note 44 pp. Next: Stage IV (the pentagon sheet and a path), then 3 x 3.
- Review fixes (2026-11-03, round 2): Prop 31(2) and Prop 39(2) no longer use distinct holonomies to separate objects (m_0 only); Prop 40 text: -K.C = 0
  curves are the source of classes that *may* contribute; Theorem 41 states that leximin walls subdivide the 148 chambers; Remark 46(ii) states the
  hypotheses (real, eventually positive, uniformly bounded 1 + delta_k; formal case = valuation statement) and lists four statuses per zero curve;
  Prop 44(5) cites Prop 31(2)-(3) and Prop 32. Certificate kahler/chambers24.json (from cpl24.certificate()) with the independent verifier
  engine/check_chambers24_cert.py (CC1-CC4; stdlib + Fractions; reads only the JSON).
- Section 12.8 (2026-11-03), Stage IV. Path f = (2/5,3/5), g = (7/10 - gamma, 3/10, gamma), gamma in [3/20, 11/40]; pentagon chamber throughout
  (no blow-up wall for gamma in (1/10, 3/10)); only cell 23 moves: A = T^(2/5), B = T^(3/10), C = T^(gamma - 2/5); critical points
  y1 = (s^2 - B)/s, y2 = s, (s^2 - B)^2 (1 + Cs) = As. Prop (path): leximin walls at gamma = 1/5 (level 2; 11, 12, 22 tie: THE COLUMN-BINDING
  EXAMPLE LIES ON IT) and 4/15 (level 1); Newton polygon fibres: 4 x (1/4 - g/2, 3/20) + R (g - 1/10, 2/5 - g) below; 2 + 3 at 1/5; 4 x
  ((4/5 - g)/4)(1,1) (i^k) + R (1/10, 1/5) between; 3 x (2/15,2/15) (omega^k) + N (2/5 - g)(1,1) + R above 4/15. Prop (crossing): 5 real -> 3 real
  forces a collision at gamma_c(T) -> 1/5 (one discriminant zero at L = 50..400: 0.19235, 0.19918, 0.199985, 0.19999999); none at 4/15; values
  distinct away from gamma_c; positive object's scale 3/20 - g/2, then 1/5 - 3g/4, closing at 4/15 (rank 1 -> 2). Stokes alignments at phases
  0.3, 1.0, pi/2 within 1.5/L of a wall (L = 25, 50, 100). Not claimed: an A-infinity functor along the path. New: engine/pce/pentpath.py,
  engine/check_pentpath.py (PP1-PP9), kahler/fig_pentpath.py (fig_s4_sheet, fig_s4_objects, fig_s4_track). LEDGER row path-across-walls.
  Next: 3 x 3 (rank question of Thm 38 with the five-column record; four-dimensional margin walls).
- Section 12.9 (2026-11-04), composing couplings. Def: objects (X, p) full support, morphisms Pi(p, q). Markov gluing g1 D^-1 g2 (Prop: associative,
  unital, maximum entropy = Markov by the chain rule; convex fibres so K_n fillers exist and force nothing). Selection sigma(g1..gk) = end marginal
  of the facet-leximin point of the multi-coupling fibre (T -> 0 limit of the positive critical point, Thm 27). Faces of K_n = dissections of the
  (n+1)-gon; each cell one selection in its own table. Prop: selection unital, transposition-compatible, NOT associative (26/30). Thm (exact K_4):
  binary laws first entries 2/5, 1/3, 3/7, 1/4, 5/11; couplings a00 = 13/60, 7/30, 1/80, 3/80; 11 distinct composites (vertices 173/960, 61/384,
  61/320, 38/165, 93/385; edges 997/5760, 197/1152, 14941/73920, 1249/5280, 15403/73920; top 28573/147840); Markov 2371469/12672000 on every face.
  2 -> 3 state example: 11 distinct points in a 2-d polytope, the pentagon image folds, top inside hull, Markov outside. Finite T: associator
  -41/960 at L = 25..200, so the filtered-m_3 conjecture "valuation = h_j - h_1" is refuted (stated in Remark). Open: chain-level complexes, the
  associativity locus, K_5. New: engine/pce/selcomp.py, engine/check_selcomp.py (AS1-AS7), kahler/fig_selcomp.py (fig_as_pentagon, fig_as_polytope).
  LEDGER row composition. Note 51 pp.
- Section 12.10 (2026-11-04). Prop (no-go): any uncurved A-infinity category with H(hom) = H(Pi(p,q)) = Q[0], composition of point classes and the diagonal as
  unit, Z or Z/2 graded, is quasi-equivalent to one object with End = Q, m_n = 0 (n >= 3): the associator carries no invariant there. Thm (binary
  selection = Frechet-midpoint gluing): sigma_00 = sum_b (max(0, r_b + s_b - q_b) + min(r_b, s_b))/2 (X0, X2 binary, X1 any; proof: product of slice
  segments, blockwise leximin, midpoint); Markov = sum_b r_b s_b / q_b. Cor: uniform margins, rho = 4 t - 1: Markov rho1 rho2, selection
  sign(rho1 rho2) min(|rho1|, |rho2|) (in v = -log|rho|: Markov adds, selection takes the max); associative. Prop: associator explicit PL; among 9^4
  margin tuples (denominator 10) only uniform is associative on the test grid; the locus has positive measure, not a union of dependence-sign classes.
  Ex: margins 2/5, 1/3, 3/7, 1/4, t1 = 13/60, t2 = 7/30: A(t3) has kinks 1/30, 13/120, 2/15, 3/20, 143/840, 41/210, 7/30; A = 0 on [13/120, 2/15] and at
  5/28. Continuity: affine on fixed (facet set, level sets) (proved, SB3); no jump across facet-set changes (numerical, SB4); proof open. Module
  change: selcomp.facet_cells now uses face dimension (tied cells sharing a facet all count); AS1-AS7 rerun and pass. New: engine/check_selbin.py
  (SB1-SB7), kahler/fig_selbin.py (fig_sb). Provenance table is now a longtable. Note 55 pp.
- Review round (2026-11-04): Prop 50(2) now rigorous: full-dimensional cells where the binary associator vanishes, certified by strict branches at a
  rational point and zero affine coefficients (margins 2/5, 1/3, 3/7, 1/4 at (13/60, 7/30, 29/240); margins 3/20, 1/4, 3/4, 17/20 at (1/40, 1/48, 51/80));
  grid fractions kept as a census (Prop 50(3)). Example 51's interval is a slice of that cell; the independence remark is now only an observation.
  New selcomp.certify_leximin: exact certificates (feasibility, forced zeros with dual vectors, fibre dimension, facet witnesses, non-facet lower
  bounds or forced-zero certificates, balanced laws q > 0 per level in the row space, strictly increasing levels); all 57 selections behind Thm 45,
  the ternary example and K_3 pass (SX2). Exact orientation tests for the ternary configuration (SX3): top inside, Markov outside, the edge paths
  through [0..3] and [1..3] cross the one through [2..4]. Thm 48 proof expanded (product of slices, blockwise recursion); global continuity stated.
  Prop 47 explains why the associator is a boundary and says nothing about other enhancements; abstract made conditional. New §12.11 (three
  categorical directions, with missing ingredients) and Appendix B (exact inputs). New: engine/check_selcert.py (SX1-SX3). Note 57 pp.
- Section 12.11 (2026-11-04), a Floer reading of selection (objects only). Composition = reduction in stages + a choice of point in the fibre polytope:
  Markov = Birch (max entropy), selection = facet-leximin = valuation of the positive critical point (Thm 27) = FOOO's non-displaceable fibre (FOOO
  2010 Thm 1.5, procedure of their §9). Prop (binary): X_F = prod_b CP^1 (areas U_b - L_b, Fano); (1) Cho-Oh: HF(L_c) != 0 for a unitary local system iff
  c = product of midpoints (2^k objects y_b = +-T^(w_b/2)); (2) Alston-Amorim (arXiv:1003.3651, Thms 1.2, 4.3; F_2-bar local systems): HF(R, (L_c, rho))
  != 0 iff c = product of midpoints, rho = 1, rank 2^k. So binary selection = marginal of the unique Floer-nontrivial fibre; Markov = Birch point
  (differs in 37/40 chains). K_4 fibres (exact traversal, selcomp.delzant): 14 distinct 3-way fibres CP^1 x CP^1 (Fano); 5 distinct 4-way fibres, none
  Fano (-K.C = -4), 3 Delzant, 2 orbifold; 5-way top not Fano (-K.C = -16), unimodular at 400 vertices visited (full traversal running separately).
  No functor, morphisms or products claimed. New: engine/check_selfloer.py (SF1-SF5). Note 58 pp.

## 2026-11-05: sections 12.13-12.14, Study 15 Deviation 2
- Note 12.13, a small dagger category for finite probability: Def cpl, Prop cplcat (Markov gluing: dagger symmetric monoidal, natural for end
  coarse-graining), Example umbrella (forecast -> umbrella -> wet; the K_4 couplings gamma_1, gamma_2), Prop selkeeps (closed, unital, reversal,
  relabelling, copied factors gamma x D_w; proofs), Prop selbreaks (exact counterexamples with all inputs: tensor 29/1400 vs 31/1200; end
  coarse-graining 17/120 vs 11/90; middle coarse-graining breaks Markov 79/450 vs 73/420 and selection 1283/8064 vs 305/2016), entropy argument
  for the Markov tensor square, comparison table, Remark selcat, fig_selcat.
- Note 12.14, how little it takes: Prop identified (binary ends: the data determine [sum L_b, sum U_b]; selection is its centre, least
  worst-case error; middle refinement only shrinks it), Prop poison (Markov slope (t2 - p1 p2)/(p1(1 - p1)); selection slopes in {0, +-1/2,
  +-1}, flat region), Example howlittle (tau = 1/2: Markov flips at 1/38 = 2.6% of records, selection at 9/70 = 12.9%, the data force a warning
  at 1/6; an unrecorded middle distinction flips Markov in 56 of 195 splits, never selection; finer end and an independent factor move
  selection only), "Both sides" paragraph, fig_selrob. Cites Manski 2003, Cho-Jacobs 2019.
- Remark selfloer / SF5 corrected: the complete traversal of the 5-way fibre (34166 vertices, 4620 s) is simple but NOT unimodular
  (orbifold), min -K.C = -32; the earlier "unimodular where visited" was an artefact of the cap. check_selfloer.py --full reruns it.
- New: engine/pce/selnat.py, engine/check_selnat.py (NC1-NC9), kahler/fig_selnat.py. Ledger row probability-category. Note 63 pp.
- Study 15 Deviation 2 (llm_fracture_study/DEVIATIONS_study15.md): run_icl_mps.py (mps, bfloat16, eager attention, prompt cache shared
  by the 10 samples, flush per answer, --resume); kit unchanged. An interim score of a partial file was viewed on 2026-11-05 (disclosed in the
  deviation file; not interpreted); the final score goes to study15_final.json.

## 2026-11-05 (later): sections 12.15-12.18
- 12.15 Selection as a section: projection pi: J -> D, L = locally consistent, contextual = L \ D; Def section (equivariant, interior, system
  independent, natural); Thm projection (D = conv of deterministic images; fibre over relint G has dim |Omega(G)| - 1 - dim G; Lipschitz
  fibres, Walkup-Wets); Thm sections (strictly concave -> continuous; max entropy interior, equivariant, system independent; leximin
  equivariant, not system independent; stabiliser with one fixed point forces every equivariant section); Thm sec2cat (inner-face
  consistency => associative category; max entropy = Cpl; leximin not); Open problem prob:assocsec; Remark Shore-Johnson scoped.
- 12.16 The 4-cycle: Prop cycleD (24 facets = 16 local + 8 CH; D = L cut by CH; symmetric path contextual past S = 2), Prop cyclefibres
  (2721 faces; interior 7, positivity facets 4, CH facets simplices -> point fibres on all 1752 faces in them; 400 more point fibres).
- 12.17 Boundary: Prop chshpath (stabiliser of order 16, fixed fibre a point, q interval [lam/2, (1 - lam)/2]); path table (sections
  disagree until the fibre closes; only CH facets close it); TV <= 8 x width.
- 12.18 Simpson: Prop simpson (selection makes a reversal: slices 1/276, 1/99, marginal -31/4048; data allow [-95/1518, 287/6072];
  Markov gluing reverses under middle merging 85/4032 -> -43/6272). Wrap-up "What the sections support". fig_cycle.
- New: engine/pce/margsec.py, engine/check_margsec.py (MS1-MS9), kahler/fig_margsec.py. Ledger row marginal-sections (category level
  Defined: Thm sec2cat is an elementary statement recovering the standard Cpl). Note 71 pp.
- Ledger rule revised (author's decision, 2026-11-05): the category level records mathematical status, not novelty. It may be Proved
  only with the qualifier "(standard" in its text, and is never Exact or Numerical (check_ledger.py LG1). Row marginal-sections:
  category Proved (standard categorical construction): Thm sec2cat; max entropy gives Markov gluing; leximin fails inner-face
  consistency (witness: the associator and Thm k4); uniqueness open (Problem assocsec, kept separate from the theorem).

## 2026-11-05 (later still): rigidity of the contextuality boundary; section 12.19
- Cor rigid (4-cycle): every point of the boundary of D off the positivity facets lies on a CH facet, so it has one compatible joint
  law; positivity facets are facets of L too (crossing them is invalid data, not contextuality).
- Thm ncycle (12.19): for every n >= 3 the 2^(n-1) noncontextuality facets (Araujo et al. 2013, cited for n >= 4) contain exactly 2n
  deterministic points and are simplices (proof by the odd-parity count), so the contextuality boundary is rigid on every cycle; local
  facets leave fibre 3 * 2^(n-2) - 2n. MS10: all facets exact for n = 3, 4, 5; listed facets and vertex counts for n = 6, 7, 8.
- Simpson mechanisms named: selection sensitivity (rule-made) vs information-loss sensitivity (coarsening).
- Literature: we did not find the identification reading (unique joint law along the contextuality boundary) stated; the note says so.
  Pironio's "All CHSH polytopes" does not count vertices on CHSH facets, so it is not cited for that.
- Abstract updated (n-cycle rigidity, Simpson mechanisms, open associative-section problem). Note 72 pp.

## 2026-11-05 (night): three layers, what the cocycle gives
- Remark layers (12.15): Abramsky-Brandenburger = existence; ours = geometry of extensions and the choice; Vigneaux / Baudot-Bennequin =
  identities of particular choices. Fibre vertices = deterministic global sections; Thm projection(2) read in that language. Bridge
  constructed only as far as Prop cocycle.
- Prop cocycle: cocycles are c H; c > 0 gives system independence (identity = cocycle, inequality I >= 0 = concavity); c < 0 gives no
  section (uniform binary chain: 4 minimising vertices, exact). Power family phi_q (Cressie-Read): tensor-square defect 0 only at q = 1,
  0.0008 at 0.99 ... 0.075 at -8, facet-leximin 0.0752 (numerical, 40 digits; MS11). Consistent with Shore-Johnson; not a proof of
  Problem assocsec. New: margsec.power_section, chain_power_section, markov_point; kahler/fig_sections.py. Note 74 pp.

## 2026-11-05 (night): Study 16 drafted (NOT frozen, NOT run)
- llm_fracture_study/PREREG_study16.md (DRAFT) and kit_cycle_context.py: 40 items x 6 strengths lam x 6 questions = 1440 prompts, K = 20.
  Four fictitious gauges; relation documents for the odd cycle (same, same, same, different) with support fraction (1 + lam)/2, so an
  ideal follower has S = 4 lam. Questions: the four cycle pairs and the two unasked pairs. H1 drift (excess over a split-half baseline),
  H2 contextuality-by-default at lam = 1 and not at lam = 0 (criterion of Kujala-Dzhafarov-Larsson 2015), H3 coherence: the elicited
  unasked-pair value lies in the identified interval from the 95% boxes of the cycle tables. Smoke: follower (H1 no, H2 yes, H3 6/6),
  drifter (H1 yes). Kit refuses to overwrite a non-empty answer file. Freeze (SHA-256 to FROZEN_PREREG_study16.txt) only on the author's go.

## 2026-11-06: Study 15 analysed (primary model)
- Complete answers (1800/1800) scored once with the frozen kit v1.1 -> llm_fracture_study/study15_final.json; write-up RESULTS_study15.md.
  H1 not supported (Delta_common < 0, but rare minus common CI [-0.039, 0.021] includes 0); H2 not testable (no eligible cells); ASR at most
  0.135 (rare, N = 20, k = 8), near 0 in the common regime; invalid 3.6% mean, no condition above 20%. Secondary model not run.

## 2026-11-06: Study 17 drafted (NOT frozen, NOT run)
- synthetic_sections/PREREG_study17.md (DRAFT): tiny NoPE transformers trained only on context marginals of exact synthetic worlds
  (W1 4-cycle: interior, approach to a CH facet, approach to a structural zero, contextual PR path; W2 two chains and their product).
  Exact implied joints by forced-order scoring; Csiszar decomposition into in-fibre and off-fibre terms; theorem checks T1-T4;
  hypotheses H1-H5 (which section learning picks; in-fibre dominance; rigidity in practice; order drift; tensor-square defect of the
  learned composite). 75 models, about 1.5 h on CPU here (CPU torch 2.15 installed in the workspace). Kit kit_learned_sections.py to
  be written and smoke-tested before freezing.
- Study 17 revised (2026-11-06) for realism: latent-class worlds G1 (8), cohort worlds G2 (8; realistic shadows that may drift or be
  contextual), finite samples N = 300 / 3000 for G1 and G2, constructed paths C only for theorem checks and H3, chains with a hidden
  confounder W2; 154 models; H6 added. Kit synthetic_sections/kit_learned_sections.py written (plan / train / score / smoke); the smoke
  test found that projection onto a boundary fibre needs the forced-zero cells removed first (fixed). Not frozen, not run.
- 2026-11-06 19:10: Study 15 Deviation 3 (SmolLM2 replication not run: swap, about 290 s per prompt). Study 17 FROZEN
  (synthetic_sections/FROZEN_PREREG_study17.txt) and training started here in two processes (even / odd condition indices,
  runs17_even.jsonl and runs17_odd.jsonl; score on their concatenation).

## 2026-11-06 (late): Study 17 results
- 154/154 models trained; frozen files verified; analysis run once -> synthetic_sections/study17_final.json, RESULTS_study17.md,
  fig_study17.pdf (fig_study17.py plots only). T1-T4 all hold. H1 maximum-entropy-like (94%); H2 not supported (in-fibre = off-fibre);
  H3 not supported (CH total error 0.271 > zero 0.188: in-fibre -> 0 but off-fibre inconsistency grows near the CH facet); H4 supported
  (drift 0.06-0.07); H5 not supported (tensor defect median 0.039, 0/12 below 0.0188); H6 supported (cohort drift 0.22 vs 0.066,
  CI [0.115, 0.213]).

## 2026-11-06 (night): fine-sheaf reading; Study 18 drafted
- Note, Remark layers: gluing = nonempty fibre (no contextuality), locality = point fibre (identification); signed extensions always
  exist (Abramsky-Brandenburger 2011; Al-Safi-Short PRL 111, 170403) and form an affine space A_d of constant dimension; J_d = A_d cut by
  the simplex; rigidity = at the contextuality boundary the affine space meets the simplex in one point; contextual fraction
  (Abramsky-Barbosa-Mansfield PRL 119, 050504). Note 75 pp.
- Study 18 DRAFT (synthetic_sections/PREREG_study18.md): post-hoc confounds of Study 17 disclosed (H5 below seed noise floor; H3
  sparsity 8 vs 4 zeros); sparsity-matched faces F0/F1/F2 (8 vertices, fibre 0/1/2); consistency training (order-invariance KL);
  weight decay 0 contrast; contextual fraction; in-distribution fit vs failed combination; noise floors; H7-H11. Kit to be written.
- 2026-11-06 21:00: Study 18 FROZEN (synthetic_sections/FROZEN_PREREG_study18.txt) after fixing, before the freeze, the step count
  (8000; 33 ms per step), seeds (3, the pre-specified cut), epsilon = 0 only, and the form of the consistency penalty (squared log-ratio
  of two orders). Training started in two processes, runs18_even.jsonl / runs18_odd.jsonl, weights in weights18/. Each 2-hour segment
  is restarted with --resume.
- Note: Remark beckchevalley (after Problem assocsec). Inner-face consistency read as a Beck-Chevalley-type condition (Markov property;
  gluing = composition, Fritz 2020 sections 11-13). Associativity failures = failures of that condition for a non-Markov gluing. Hyperconnected
  morphisms do not explain them (maps of finite spaces are localic); topos theory enters only through symmetry (Gamma-sets; hyperconnected
  quotients, Hora arXiv:2302.06851). Note 76 pp.


## 2026-11-07: Study 18 results
- 168/168 models trained (three 2-hour segments per half, resumed; no rescoring during training); frozen files verified with
  sha256sum -c; analysis run once -> synthetic_sections/study18_final.json, RESULTS_study18.md, fig_study18.pdf (fig_study18.py plots only).
- T1-T4 hold; fibre dims F0/F1/F2 = 0/1/2; F0 TV(P*, P_true) <= 1.3e-14.
- H7 not supported (consistent means F0 0.196 < F1 0.227 < F2 0.257, the predicted order, CIs include 0). H8 supported (consistency lowers
  F0 off-fibre by 0.218 nats, CI [0.119, 0.310]). H9 not supported (weight decay 0 no effect). H10 not supported, alternative reported
  (defect exceeds the seed floor by about 0.010 TV, CI [0.0015, 0.018]). H11 supported (24/24 maximum-entropy-like, mean iota 0.107).
- Exploratory: consistency drives order drift 0.31 -> 0.01 and failed combination to about 0.03 but worsens fit of the direct pair
  predictions (summed KL to 0.07-0.17); in-fibre error rises on F1/F2. Post hoc: plain ordering is reversed (F0 - F2 = 0.108, CI
  [0.032, 0.189]). E6: on cohort worlds the models' direct pair tables have about half the data's contextual fraction (0.21 vs 0.43).
- Disclosure: E7's two parts are in different units (summed KL vs summed TV) in the frozen kit; reported side by side.
- weights18/ (46 MB) stays in the workspace; not packaged.

## 2026-11-07: section 12.20, how little poisoning in many dimensions
- New module engine/pce/poisonflip.py, checks engine/check_poisonflip.py (PF1-PF9, 9/9 PASS, about 2 min), figure kahler/fig_poisonflip.py.
- Note section 12.20 (Def flipcost, Props 79-82 twothresh / sparse / placement / removal, Example 83, Fig 22): two thresholds
  c_some <= c_s <= c_all (LP; dual-vertex union); cheapest undetectable flip = conformal sum of at most min(a, dim) circuits (Rockafellar 1969;
  Muller-Regensburger 2016); n-cycle circuits 188 (n=4), 481 (n=5), 5n of support 4; one 4-cell circuit optimal in 30/30, 30/30, 28/30, 28/30,
  24/29 for n = 4..8 at gap 0.01; placement Pi(S) monotone in the hidden support (2721 faces, 0 violations; vertex 13, CH 252, positivity 287,
  interior 376); dim G + fibre = |S| - 1 splits hidden freedom into data room and rule room; at matched sparsity rigid faces have MORE
  placements; with a budget what matters is whether the decision is identified (width 0: >= g and <= 4 single circuits within 1.5 g;
  else 0 cost along all 376). Removal presheaf on information states (batches, contexts); context forgetting widens the interval; one-batch
  flip lasts while k/(N_t+k) >= q*; time example: 2298 poisoned units vs 12000 flip every rule 1 step, maximum entropy 6 steps; only the
  poisoned batch's removal reverts; first-order estimate within 6%.
- Retracted before writing: an exploratory claim that rigid faces have the fewest budgeted placements (a tie at exactly 2g; below 2g the
  CH and fibre-1 faces both have 4).
- Ledger row 'poisoning' (category level Defined). Log TOTAL 114 scripts, 912 PASS. Note 81 pp.
- Not done: the inversion additions (awaiting the author); Study 16 freeze; a Study 19 arm measuring these quantities on learned models.
