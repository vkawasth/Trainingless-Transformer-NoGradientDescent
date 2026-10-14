# Shadows, Fibres and Secants of the Posterior Tower

Paper, check scripts, figure generators, an interactive explorer and the hallucination studies for
*Shadows, Fibres and Secants of the Posterior Tower* (V. K. Awasthi, October 2026).

Repository: <https://github.com/vkawasth/Trainingless-Transformer-NoGradientDescent/tree/main/papers>

## Contents

| Path | What it is |
|---|---|
| `tower_geometry.tex`, `tower_geometry.pdf` | The paper (50 pp). Builds with `pdflatex` (run twice). Figures are read from `atlas/figs/`. |
| `check_tower.py` | Tower definitions, imported verbatim by `check_shadows.py`. |
| `check_shadows.py` | 30 checks: algebra of conditional expectations, shadows, martingale, categorical structure (§§3–8). |
| `check_tropical.py` | 8 checks: positive tropicalisation, Berkovich factorisation, secants (§§9–10). |
| `check_cohomology.py` | 10 checks: the L² presheaf, Beck–Chevalley, gluing, tropical fibre nerves (§§12–13). |
| `check_hallucination.py` | 9 checks: the three-token model (§16). |
| `check_barr.py` | 4 checks: D, η and μ are monotone for the Barr lifting (Lemma 6.11), with negative controls and a counterexample to the converse. |
| `check_places.py` | 9 checks: the context presheaf over the Berkovich spectrum M(ℤ) (§17), and the explorer's numbers. |
| `check_bc_test.py` | 7 Monte Carlo checks of the finite-sample Beck–Chevalley test used in §18 (size, power, centre enlargement, Study 2 replica). |
| `check_*_output.txt` | The output of each script as run for the paper. |
| `atlas/` | `atlas.py` draws Figs 1–8, `fig_places.py` draws the M(ℤ) figure, `fibre_scan.py` computes the Betti map; `fibre_topology.py` is shared. |
| `explorer/context_sheaf.html` | Interactive explorer of the three-token example. Open it in any browser; no install, no network needed except for web fonts. |
| `halluc_study/` | Pre-registration (`PREREG_hallucination.md`) and Amendment 1, both with frozen SHA-256 hashes; Study 1 script and results; Study 2 sampler, kit, analysis, samples and results; Study 3 kit. |

## Quick start

```bash
pip install -r requirements.txt
./run_checks.sh            # runs all seven check scripts (about 2 minutes)
python atlas/atlas.py      # regenerate Figs 1-8 into atlas/figs/
python atlas/fig_places.py # regenerate the M(Z) figure
pdflatex tower_geometry.tex && pdflatex tower_geometry.tex
```

Every check prints `PASS`/`FAIL` per item and a final count. All 77 checks pass.

## The hallucination studies (`halluc_study/`)

The plan was pre-registered before any statistic was computed (`PREREG_hallucination.md`, hash in `FROZEN_PREREG.txt`).
Amendment 1 (`AMENDMENT1.md`, hash in `FROZEN_AMENDMENT1.txt`) was frozen after items 0–199 of Study 2 were inspected and
before any statistic on items 200–1999 was computed.

**Study 1 (LLM-drift).** Download the `<TASK>_EVAL.csv` files of <https://github.com/lchen001/LLMDrift> into a folder and run

```bash
cd halluc_study
LLMDRIFT_DIR=/path/to/llmdrift python study1_llmdrift.py
```

**Study 2 (HaluEval QA).** `qa_data.json` is in <https://github.com/RUCAIBox/HaluEval> (`data/qa_data.json`, MIT licence).
The Llama 3.1 8B samples used in the paper are in `study2/samples_2000.jsonl`. To reproduce the analysis:

```bash
cd halluc_study
HALUEVAL_QA=/path/to/qa_data.json python study2_analysis.py
```

To draw new samples from your own model (Ollama, Anthropic or OpenAI-compatible APIs):

```bash
python kit_halueval.py prepare --data /path/to/qa_data.json --out prompts.jsonl
python sample_halueval.py --backend ollama --model llama3.1 --limit 2000 --workers 1
python kit_halueval.py score --data /path/to/qa_data.json --samples samples.jsonl --out study2_results.json
```

API keys are read from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`); none is stored in this folder.

**Study 3 (pairwise elicitation).** `kit_pairwise_cycles.py` scores pairwise answers for the 3-cycle inequalities;
`python kit_pairwise_cycles.py smoke` runs it on synthetic data.

## Data and licences

- HaluEval (Li et al., EMNLP 2023): MIT licence. Not redistributed here; the Study 2 samples are model outputs on its prompts.
- LLM-drift answers (Chen, Zaharia and Zou, 2024): download from the original repository.
- No credentialed or patient-level data are used or included.
