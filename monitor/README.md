# monitor — a structural-change monitor for couplings under changing information

Code, data and checks for *Empirical Site Separation: Bending, Tearing, Contextuality and Holonomy in a National Survey,
Retrained Language Models and Clinical Data* (`validation.tex` / `validation.pdf`).

## Run
```
python3 check_monitor.py          # ~3 min: unit tests, fast reruns, 63 claim checks (all numbers in the text)
```
Slow analyses (outputs are saved and verified by the check; rerun only to reproduce from scratch):
```
python3 gss_analysis.py           # ~10 min: GSS 1985-2024 wording systems, regimes, panels (churn, gluing gap, Markov null)
python3 cbd_boot.py               # ~5 min: Dirichlet posterior screen for Contextuality-by-Default (canonical and ordinal cuts)
python3 panel_models.py           # ~10 min: latent-class models of the panels (stable / drifting / latent Markov)
python3 site_separation.py        # ~15 min: Layers 1-3 on 55 GSS item pairs x 24 years (CbD canonical vs naive, holonomy on K_{2,11}, MI)
python3 boundary_tests.py          # ~5 min: boundary-null tests of the canonical-CbD flags
python3 ctx_allpairs.py            # ~15 min: boundary null combined over years for all 55 pairs, Bonferroni
python3 panel_paths.py             # ~10 min: kernel composition and path dependence (vote, party ID)
python3 toric_public.py            # ~10 min: toric surfaces, exact torus branes, leximin, isotopy, 3x3 vertex counts (needs ../bridges/fukaya)
python3 make_tables.py             # LaTeX tables from the JSON results
python3 make_figures.py            # figs/*.pdf from the JSON results and the data
```
Requires numpy, scipy, pandas, pyreadr, and `../bridges/cbd/cbd.py` (from the bridges folder of the repository).

## Files
| file | what it does |
|---|---|
| `monitor.py` | bands, walls, tear/break margins, regimes (bend / tear? / tear / break), stress, CbD rank 2 and canonical (all dichotomizations; `ordinal=True` for order-respecting cuts), churn, gluing gap; unit tests |
| `llm_analysis.py` | GPT-3.5 / GPT-4, March vs June 2023, eight tasks (LLMDrift) |
| `usmle_prompts.py` | USMLE intact vs damaged prompts as a second change of information |
| `gss_analysis.py`, `cbd_boot.py`, `panel_models.py` | General Social Survey |
| `clinical_analysis.py` | NHANES 2009-10 vs 2011-12 |
| `site_separation.py`, `boundary_tests.py`, `ctx_allpairs.py`, `panel_paths.py` | site separation on the GSS (Layers 1-3), panels' paths |
| `toric_public.py` | toric/Floer objects (Bridges §5, §8) for margins from NHANES and the GSS |
| `make_figures.py` | the eight figures |
| `check_monitor.py` | reproducibility and claim checks |

## Data (`data/`)
* `llmdrift/` — answers from Chen, Zaharia & Zou, https://github.com/lchen001/LLMDrift (`generation/*_EVAL.csv`).
* `gss/gss_extract.csv` — columns of the GSS 1972-2024 cumulative file (from the `gssr` R package, `gss_all`), made by `extract.R`;
  `gss_panel{06,08,10}_long.rda` — the GSS panels from `gssr`.
* `nhanes/NHANESraw.rda` — from the `NHANES` R package.
* Clinical records (MIMIC-IV) are **not** included: run the protocol of Bands §11 locally under the PhysioNet data use agreement.

## Notes found while building this
* The USMLE file asks every question twice; the first copy has a passage (or option A) deleted. Items are keyed by prompt.
* OpinionQA has no correct answer (`ref_answer` is always "(A)"); it is coded answered vs not.
