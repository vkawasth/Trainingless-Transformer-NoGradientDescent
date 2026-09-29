# Reproducing the three-level bias results (BASIL, MBIC)

All commands run from this folder (`examples/bias3`). Python 3.10+ with `numpy scipy pandas scikit-learn openpyxl`.

## 1. Data (not redistributed)
```
mkdir -p data
git clone https://github.com/launchnlp/BASIL data/BASIL                       # BASIL, 2nd release
git clone https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE data/BABE   # contains raw_labels_MBIC.xlsx
```
Other locations: `export BASIL_DIR=/path/to/BASIL` and `export MBIC_XLSX=/path/to/raw_labels_MBIC.xlsx`.

## 2. Level-3 detection (the loop tests)
| command | what it reproduces | output |
|---|---|---|
| `python bias3.py` | BASIL stance: house leans, the 36 loop tests (camp / theme / window), toric exact vs Markov-basis MCMC, gluing map | `bias3_out.txt`, `bias3_results.json` |
| `python bias3_spans.py` | BASIL span favourability: loops Φ, exact no-3-way p, event bootstrap; by window and theme (Fox–HPO Φ = +1.34, p = 0.0017) | `bias3_spans_out.txt`, `bias3_spans_results.json` |
| `python bias3_spans.py --exclude-trump` | robustness without Trump as a target (Fox–HPO Φ = +1.09, p = 0.015) | `bias3_spans_results_notrump.json` |
| `python global_gluing.py` | planted global failures on disjoint BASIL regions; ring of K outlets | `global_gluing_results.json` |
| `python mbic.py` | MBIC: single-source theorem check, outlet × topic global failure (p = 0.0005), hostile-media loops | `mbic_results.json` |

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
